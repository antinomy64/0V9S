#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal person-only part presence analysis.

Correct project assumption for this Talk2DINO/OVPS experiment:
  - Stage1 projector is T -> V.
  - Visual cropaug_patch_tokens are already in DINO/V space.
  - Therefore visual patch tokens are NOT passed through any projector.
  - Only part text features are passed through Stage1 text projector, optionally followed by W.

Goal:
  For all object-level samples of one class (default: person), infer which parts
  appear in each image/object crop using global patch statistics, then compare
  predicted part set with GT visible part set.

Outputs only:
  1) <out_dir>/<target_class>_presence_eval.csv
  2) <out_dir>/<target_class>_presence_summary.json

GT rule:
  GT masks are used ONLY to form gt_parts and evaluation metrics.
  GT masks are NOT used to build clusters, thresholds, scores, or predictions.
"""

import argparse
import csv
import json
import os
import sys
import random
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F
import yaml
import importlib
from tqdm import tqdm


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_key(obj: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in obj:
        raise KeyError(f"Missing key '{key}' in {ctx}. Please pass the correct field argument.")
    return obj[key]


def load_stage1_model(model_config: str, stage1_ckpt: str, device: torch.device):
    with open(model_config, "r") as f:
        cfg = yaml.safe_load(f)

    model_class_name = cfg["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(cfg["model"])

    ckpt = torch.load(stage1_ckpt, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    ret = model.load_state_dict(state, strict=False)
    print("[Stage1] Missing keys:", getattr(ret, "missing_keys", []))
    print("[Stage1] Unexpected keys:", getattr(ret, "unexpected_keys", []))

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_required_text_projector(model, fn_name: str):
    if not hasattr(model, fn_name):
        raise AttributeError(f"Stage1 model has no function '{fn_name}'. Please pass the exact text projector name.")
    fn = getattr(model, fn_name)
    if not callable(fn):
        raise TypeError(f"Attribute '{fn_name}' exists but is not callable.")
    return fn


def load_w(w_ckpt: str, w_key: str, device: torch.device) -> Optional[torch.Tensor]:
    if not w_ckpt:
        return None
    ckpt = torch.load(w_ckpt, map_location="cpu")
    if not isinstance(ckpt, dict) or w_key not in ckpt:
        raise KeyError(f"W checkpoint must be a dict containing key '{w_key}'.")
    W = ckpt[w_key].float().to(device)
    print(f"[W] Loaded {w_key} from {w_ckpt}, shape={tuple(W.shape)}")
    return W


@torch.no_grad()
def normalize_patches(patches: torch.Tensor) -> torch.Tensor:
    # Visual tokens are already DINO/V-space tokens. Do NOT pass them through Stage1.
    return F.normalize(patches.float().cpu(), dim=-1)


@torch.no_grad()
def project_parts_to_v_space(
    part_feats: torch.Tensor,
    text_projector,
    device: torch.device,
    W: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> torch.Tensor:
    outs = []
    part_feats = part_feats.float()
    for st in range(0, part_feats.shape[0], chunk_size):
        x = part_feats[st:st + chunk_size].to(device)
        z = text_projector(x)
        z = F.normalize(z.float(), dim=-1)
        if W is not None:
            z = F.normalize(z @ W.t(), dim=-1)
        outs.append(z.cpu())
    return torch.cat(outs, dim=0)


def filter_samples(
    data: Dict[str, Any],
    target_class: str,
    class_field: str,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    images = {img["id"]: img for img in data["images"]}
    pairs = []
    for ann in data["annotations"]:
        cls = str(require_key(ann, class_field, "annotation")).lower()
        if cls == target_class.lower():
            pairs.append((ann, images[ann["image_id"]]))
    return pairs


def to_bool_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.bool().cpu()
    return torch.as_tensor(x).bool().cpu()


def canonical_valid_parts(
    ann: Dict[str, Any],
    part_name_field: str,
) -> List[str]:
    # In the current test15 data chain, the object-level part list is stored directly
    # in ann[part_class_name]. There is no part_valid_mask field.
    return [str(x) for x in require_key(ann, part_name_field, "annotation")]


def gt_visible_parts(
    ann: Dict[str, Any],
    part_name_field: str,
    gt_mask_field: str,
    min_gt_patches: int,
) -> List[str]:
    names = [str(x) for x in require_key(ann, part_name_field, "annotation")]

    gt = require_key(ann, gt_mask_field, "annotation")
    if not isinstance(gt, torch.Tensor):
        gt = torch.as_tensor(gt)
    gt = gt.cpu()

    if gt.shape[0] != len(names):
        raise ValueError(f"GT mask first dim {gt.shape[0]} != number of part names {len(names)}")

    gt_flat = gt.reshape(gt.shape[0], -1)
    visible = gt_flat.float().sum(dim=1) >= float(min_gt_patches)

    return [names[i] for i in range(len(names)) if bool(visible[i])]


def sample_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    # n <= 0 means use all rows.
    if n <= 0 or x.shape[0] <= n:
        return x
    idx = torch.randperm(x.shape[0])[:n]
    return x[idx]


def assign_to_centers(x: torch.Tensor, centers: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    labels = []
    for st in range(0, x.shape[0], chunk_size):
        xb = x[st:st + chunk_size]
        sim = xb @ centers.t()
        labels.append(sim.argmax(dim=1).cpu())
    return torch.cat(labels, dim=0)


def torch_kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    if x.shape[0] < k:
        raise ValueError(f"Not enough patches for kmeans: num_patches={x.shape[0]} < k={k}")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    init_idx = torch.randperm(x.shape[0], generator=g)[:k]
    centers = F.normalize(x[init_idx].clone().float(), dim=-1)

    for _ in tqdm(range(iters), desc="kmeans"):
        labels = assign_to_centers(x, centers)
        new_centers = []
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centers.append(x[mask].mean(dim=0))
            else:
                ridx = torch.randint(0, x.shape[0], (1,), generator=g).item()
                new_centers.append(x[ridx])
        centers = F.normalize(torch.stack(new_centers, dim=0).float(), dim=-1)

    return centers


def minmax_norm(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    v = v.float()
    lo, hi = v.min(), v.max()
    if float(hi - lo) < eps:
        return torch.zeros_like(v)
    return (v - lo) / (hi - lo + eps)


def relative_score_max_other(sim: torch.Tensor) -> torch.Tensor:
    # sim: [P, N]. For one part, use absolute similarity because no "other" exists.
    P, N = sim.shape
    if P == 1:
        return sim
    vals, idx = sim.topk(k=2, dim=0)
    max1, max2 = vals[0], vals[1]
    arg1 = idx[0]
    rows = torch.arange(P).view(P, 1)
    max_other = torch.where(arg1.view(1, N).cpu() == rows, max2.view(1, N).cpu(), max1.view(1, N).cpu())
    return sim.cpu() - max_other


def nnls_projected_gradient(h: torch.Tensor, B: torch.Tensor, steps: int, lr: float) -> torch.Tensor:
    # h: [G], B: [P, G], solve min ||x @ B - h||^2, x >= 0
    P = B.shape[0]
    x = torch.zeros(P, dtype=torch.float32)
    for _ in range(steps):
        recon = x @ B
        grad = 2.0 * ((recon - h) @ B.t())
        x = torch.clamp(x - lr * grad, min=0.0)
    return x


def score_one_sample(
    ann: Dict[str, Any],
    part_z: torch.Tensor,
    centers: torch.Tensor,
    part_cluster_B: torch.Tensor,
    args,
) -> torch.Tensor:
    patches = require_key(ann, args.patch_field, "annotation")
    if not isinstance(patches, torch.Tensor):
        patches = torch.as_tensor(patches)

    patch_z = normalize_patches(patches)

    sim = part_z @ patch_z.t()  # [P, N]
    rel = relative_score_max_other(sim)

    topm = min(args.top_m, rel.shape[1])
    direct = rel.topk(k=topm, dim=1).values.mean(dim=1)

    labels = assign_to_centers(patch_z, centers)
    hist = torch.bincount(labels, minlength=args.num_clusters).float()
    hist = hist / hist.sum().clamp_min(1.0)

    x = nnls_projected_gradient(hist, part_cluster_B, args.nnls_steps, args.nnls_lr)

    direct_n = minmax_norm(direct)
    nnls_n = minmax_norm(x)

    final = args.direct_weight * direct_n + args.nnls_weight * nnls_n
    return final.cpu()


def predict_parts_from_scores(
    scores: torch.Tensor,
    part_names: List[str],
    args,
    global_thresholds: Optional[torch.Tensor] = None,
) -> List[str]:
    P = len(part_names)

    if args.threshold_method == "image_zscore":
        thr = scores.mean() + args.image_z * scores.std(unbiased=False)
        mask = scores >= thr
    elif args.threshold_method == "global_percentile":
        if global_thresholds is None:
            raise ValueError("global_thresholds required for global_percentile")
        mask = scores >= global_thresholds
    elif args.threshold_method == "topk":
        k = min(args.topk, P)
        mask = torch.zeros(P, dtype=torch.bool)
        mask[torch.topk(scores, k=k).indices] = True
    else:
        raise ValueError(f"Unknown threshold_method: {args.threshold_method}")

    if int(mask.sum()) < args.min_pred_parts:
        need = min(args.min_pred_parts, P)
        mask = torch.zeros(P, dtype=torch.bool)
        mask[torch.topk(scores, k=need).indices] = True

    if args.max_pred_parts > 0 and int(mask.sum()) > args.max_pred_parts:
        selected = torch.nonzero(mask).reshape(-1)
        keep_local = torch.topk(scores[selected], k=args.max_pred_parts).indices
        keep = selected[keep_local]
        mask = torch.zeros(P, dtype=torch.bool)
        mask[keep] = True

    return [part_names[i] for i in range(P) if bool(mask[i])]


def set_metrics(pred: List[str], gt: List[str]) -> Dict[str, float]:
    p, g = set(pred), set(gt)
    inter = len(p & g)
    union = len(p | g)
    iou = 1.0 if union == 0 else inter / union
    precision = 1.0 if len(p) == 0 and len(g) == 0 else (inter / len(p) if len(p) > 0 else 0.0)
    recall = 1.0 if len(p) == 0 and len(g) == 0 else (inter / len(g) if len(g) > 0 else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact = 1.0 if p == g else 0.0
    return {"set_iou": iou, "exact_match": exact, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target_class", default="person")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--text_projector_fn", required=True)
    parser.add_argument("--w_ckpt", default="")
    parser.add_argument("--w_key", default="W")
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--class_field", default="class_name")
    parser.add_argument("--patch_field", default="cropaug_patch_tokens")
    parser.add_argument("--part_text_field", default="part_ann_feats")
    parser.add_argument("--part_name_field", default="part_class_name")
    parser.add_argument("--gt_mask_field", default="part_gt_mask_patch")
    parser.add_argument("--min_gt_patches", type=int, default=1)

    parser.add_argument("--num_clusters", type=int, default=64)
    parser.add_argument("--kmeans_iters", type=int, default=25)
    parser.add_argument("--patches_per_image_for_kmeans", type=int, default=0, help="0 means use all patches from every image")
    parser.add_argument("--max_kmeans_patches", type=int, default=0, help="0 means no global cap")
    parser.add_argument("--cluster_temp", type=float, default=0.07)

    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--direct_weight", type=float, default=0.6)
    parser.add_argument("--nnls_weight", type=float, default=0.4)
    parser.add_argument("--nnls_steps", type=int, default=100)
    parser.add_argument("--nnls_lr", type=float, default=0.2)

    parser.add_argument("--threshold_method", choices=["image_zscore", "global_percentile", "topk"], default="image_zscore")
    parser.add_argument("--image_z", type=float, default=0.25)
    parser.add_argument("--global_percentile", type=float, default=70.0)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--min_pred_parts", type=int, default=1)
    parser.add_argument("--max_pred_parts", type=int, default=0, help="0 means no max")
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Load] {args.dataset}")
    data = torch.load(args.dataset, map_location="cpu")
    pairs = filter_samples(data, args.target_class, args.class_field)
    if len(pairs) == 0:
        raise RuntimeError(f"No samples found for class {args.target_class}")
    print(f"[Data] target_class={args.target_class}, num_samples={len(pairs)}")

    # Only evaluate/predict images that have at least one part-level GT.
    # Samples with object GT but no visible part GT are excluded from this analysis.
    filtered_pairs = []
    filtered_gt_cache = {}
    skipped_no_part_gt = 0
    for ann, img in pairs:
        gt_parts = gt_visible_parts(
            ann,
            args.part_name_field,
            args.gt_mask_field,
            args.min_gt_patches,
        )
        if len(gt_parts) == 0:
            skipped_no_part_gt += 1
            continue
        filtered_pairs.append((ann, img))
        filtered_gt_cache[ann.get("id", len(filtered_gt_cache))] = gt_parts

    pairs = filtered_pairs
    if len(pairs) == 0:
        raise RuntimeError(
            f"No {args.target_class} samples with non-empty part GT were found. "
            f"Check --gt_mask_field={args.gt_mask_field} and --min_gt_patches={args.min_gt_patches}."
        )

    print(f"[Data] samples_with_part_gt={len(pairs)}")
    print(f"[Data] skipped_no_part_gt={skipped_no_part_gt}")

    model = load_stage1_model(args.model_config, args.stage1_ckpt, device)
    text_projector = get_required_text_projector(model, args.text_projector_fn)
    W = load_w(args.w_ckpt, args.w_key, device)

    part_names = canonical_valid_parts(pairs[0][0], args.part_name_field)
    if len(part_names) == 0:
        raise RuntimeError("No valid parts found in the first target sample.")
    print(f"[Parts] num_valid_parts={len(part_names)}")
    print("[Parts]", ", ".join(part_names))

    # Build class-level part text directions by averaging projected same-name part directions.
    part_sum: Dict[str, torch.Tensor] = {}
    part_count: Dict[str, int] = {}

    for ann, _img in tqdm(pairs, desc="project part text"):
        names_all = [str(x) for x in require_key(ann, args.part_name_field, "annotation")]
        feats = require_key(ann, args.part_text_field, "annotation")
        if not isinstance(feats, torch.Tensor):
            feats = torch.as_tensor(feats)

        if feats.shape[0] != len(names_all):
            raise ValueError(
                f"part_ann_feats first dim {feats.shape[0]} != part_class_name length {len(names_all)}"
            )

        z = project_parts_to_v_space(feats, text_projector, device, W, chunk_size=4096)
        for i, nm in enumerate(names_all):
            if nm in part_names:
                part_sum[nm] = part_sum.get(nm, torch.zeros_like(z[i])) + z[i]
                part_count[nm] = part_count.get(nm, 0) + 1

    part_z = []
    for nm in part_names:
        if nm not in part_sum:
            raise RuntimeError(f"Part '{nm}' was in canonical part list but was never accumulated.")
        part_z.append(part_sum[nm] / max(part_count[nm], 1))
    part_z = F.normalize(torch.stack(part_z, dim=0).float(), dim=-1)

    # Collect raw DINO/V-space patch tokens for kmeans. No visual projection.
    patch_bank = []
    for ann, _img in tqdm(pairs, desc="collect patches"):
        patches = require_key(ann, args.patch_field, "annotation")
        if not isinstance(patches, torch.Tensor):
            patches = torch.as_tensor(patches)
        pz = normalize_patches(patches)
        pz = sample_rows(pz, args.patches_per_image_for_kmeans)
        patch_bank.append(pz)

    patch_bank = torch.cat(patch_bank, dim=0)
    if args.max_kmeans_patches > 0 and patch_bank.shape[0] > args.max_kmeans_patches:
        patch_bank = sample_rows(patch_bank, args.max_kmeans_patches)
    patch_bank = F.normalize(patch_bank.float(), dim=-1)

    print(f"[KMeans] patch_bank={tuple(patch_bank.shape)}, num_clusters={args.num_clusters}")
    centers = torch_kmeans(patch_bank, args.num_clusters, args.kmeans_iters, args.seed)

    # Part-to-cluster soft distribution.
    part_cluster_logits = part_z @ centers.t()
    part_cluster_B = F.softmax(part_cluster_logits / args.cluster_temp, dim=1).cpu()

    rows_cache = []
    all_scores = []

    for ann, img in tqdm(pairs, desc="score images"):
        scores = score_one_sample(
            ann=ann,
            part_z=part_z,
            centers=centers,
            part_cluster_B=part_cluster_B,
            args=args,
        )
        gt_parts = filtered_gt_cache[ann.get("id", "")]
        rows_cache.append((ann, img, scores, gt_parts))
        all_scores.append(scores)

    all_scores = torch.stack(all_scores, dim=0)

    global_thresholds = None
    if args.threshold_method == "global_percentile":
        q = args.global_percentile / 100.0
        global_thresholds = torch.quantile(all_scores, q=q, dim=0)

    csv_path = os.path.join(args.out_dir, f"{args.target_class}_presence_eval.csv")
    metrics_accum = []
    num_pred_list, num_gt_list = [], []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ann_id", "image_id", "file_name",
                "pred_parts", "gt_parts", "num_pred", "num_gt",
                "set_iou", "exact_match", "precision", "recall", "f1",
            ],
        )
        writer.writeheader()

        for ann, img, scores, gt_parts in rows_cache:
            pred_parts = predict_parts_from_scores(scores, part_names, args, global_thresholds)
            m = set_metrics(pred_parts, gt_parts)

            metrics_accum.append(m)
            num_pred_list.append(len(pred_parts))
            num_gt_list.append(len(gt_parts))

            writer.writerow({
                "ann_id": ann.get("id", ""),
                "image_id": ann.get("image_id", ""),
                "file_name": img.get("file_name", ""),
                "pred_parts": ";".join(pred_parts),
                "gt_parts": ";".join(gt_parts),
                "num_pred": len(pred_parts),
                "num_gt": len(gt_parts),
                "set_iou": f"{m['set_iou']:.6f}",
                "exact_match": f"{m['exact_match']:.6f}",
                "precision": f"{m['precision']:.6f}",
                "recall": f"{m['recall']:.6f}",
                "f1": f"{m['f1']:.6f}",
            })

    def mean_metric(name: str) -> float:
        return float(sum(m[name] for m in metrics_accum) / max(len(metrics_accum), 1))

    summary = {
        "target_class": args.target_class,
        "num_images": len(metrics_accum),
        "num_target_samples_before_filter": len(metrics_accum) + skipped_no_part_gt,
        "skipped_no_part_gt": skipped_no_part_gt,
        "num_valid_parts": len(part_names),
        "valid_parts": part_names,
        "primary_metric": "mean_set_iou",
        "mean_set_iou": mean_metric("set_iou"),
        "exact_match_rate": mean_metric("exact_match"),
        "mean_precision": mean_metric("precision"),
        "mean_recall": mean_metric("recall"),
        "mean_f1": mean_metric("f1"),
        "avg_num_pred": float(sum(num_pred_list) / max(len(num_pred_list), 1)),
        "avg_num_gt": float(sum(num_gt_list) / max(len(num_gt_list), 1)),
        "threshold_method": args.threshold_method,
        "image_z": args.image_z,
        "global_percentile": args.global_percentile,
        "topk": args.topk,
        "min_pred_parts": args.min_pred_parts,
        "max_pred_parts": args.max_pred_parts,
        "num_clusters": args.num_clusters,
        "top_m": args.top_m,
        "direct_weight": args.direct_weight,
        "nnls_weight": args.nnls_weight,
        "note": "Only samples with non-empty part_gt_mask_patch are predicted/evaluated. Samples with object GT but no part GT are skipped. Visual cropaug_patch_tokens are not projected; only normalized. The part list is read from part_class_name; part_valid_mask is not used in this test15 data chain. GT masks are used only for filtering/evaluation, not for prediction scores.",
    }

    summary_path = os.path.join(args.out_dir, f"{args.target_class}_presence_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== Presence Evaluation Summary ==========")
    print(f"class              : {args.target_class}")
    print(f"num_images         : {summary['num_images']}")
    print(f"skipped no part GT : {summary['skipped_no_part_gt']}")
    print(f"primary mean IoU   : {summary['mean_set_iou']:.6f}")
    print(f"exact match rate   : {summary['exact_match_rate']:.6f}")
    print(f"mean precision     : {summary['mean_precision']:.6f}")
    print(f"mean recall        : {summary['mean_recall']:.6f}")
    print(f"mean F1            : {summary['mean_f1']:.6f}")
    print(f"avg #pred / #gt    : {summary['avg_num_pred']:.3f} / {summary['avg_num_gt']:.3f}")
    print(f"CSV                : {csv_path}")
    print(f"Summary            : {summary_path}")


if __name__ == "__main__":
    main()
