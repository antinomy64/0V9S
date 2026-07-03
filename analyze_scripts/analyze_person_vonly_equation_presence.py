#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V-only equation-style person part presence analysis.

This script tests the user's original idea:

  only cropaug_patch_tokens
      -> global visual prototypes
      -> image-level prototype histogram
      -> solve an equation-like nonnegative mixture
      -> infer which parts are present

No text side:
  - no part_ann_feats
  - no ann_feats
  - no Stage1 ckpt
  - no projector
  - no W

Required fields in the input pth:
  images[*]["id"], images[*]["file_name"]
  annotations[*]["image_id"]
  annotations[*]["class_name"]
  annotations[*]["cropaug_patch_tokens"]
  annotations[*]["part_class_name"]
  annotations[*]["part_gt_mask_patch"]

Important:
  This is an analysis/audit script.
  GT patch masks are NOT used to learn visual prototypes.
  GT patch masks ARE used after clustering to build a part-to-prototype
  distribution matrix B and to evaluate pred_parts against gt_parts.
  Therefore this answers:
      "Do V-side patch-token prototypes contain enough information to reconstruct
       part presence through an equation-like mixture?"
  It is not a deployable non-oracle predictor yet.

Outputs:
  <out_dir>/<target_class>_vonly_equation_presence_eval.csv
  <out_dir>/<target_class>_vonly_equation_presence_summary.json
"""

import argparse
import csv
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_key(obj: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in obj:
        available = ", ".join(sorted([str(k) for k in obj.keys()]))
        raise KeyError(f"Missing key '{key}' in {ctx}. Available keys: {available}")
    return obj[key]


def to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def normalize_patches(patches: torch.Tensor) -> torch.Tensor:
    patches = to_tensor(patches).float().cpu()
    if patches.dim() > 2:
        patches = patches.reshape(-1, patches.shape[-1])
    return F.normalize(patches, dim=-1)


def filter_target_samples(
    data: Dict[str, Any],
    target_class: str,
    class_field: str,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    images_by_id = {img["id"]: img for img in data["images"]}
    pairs = []
    for ann in data["annotations"]:
        cls = str(require_key(ann, class_field, "annotation")).lower()
        if cls == target_class.lower():
            image_id = require_key(ann, "image_id", "annotation")
            if image_id not in images_by_id:
                raise KeyError(f"image_id={image_id} not found in images")
            pairs.append((ann, images_by_id[image_id]))
    return pairs


def part_names_from_ann(ann: Dict[str, Any], part_name_field: str) -> List[str]:
    return [str(x) for x in require_key(ann, part_name_field, "annotation")]


def gt_visible_parts(
    ann: Dict[str, Any],
    part_names: List[str],
    gt_mask_field: str,
    min_gt_patches: int,
) -> List[str]:
    gt = to_tensor(require_key(ann, gt_mask_field, "annotation")).cpu()
    if gt.shape[0] != len(part_names):
        raise ValueError(f"GT mask first dim={gt.shape[0]} != len(part_names)={len(part_names)}")

    gt_flat = gt.reshape(gt.shape[0], -1)
    visible = gt_flat.float().sum(dim=1) >= float(min_gt_patches)
    return [part_names[i] for i in range(len(part_names)) if bool(visible[i])]


def check_same_part_list(
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    part_name_field: str,
) -> List[str]:
    canonical = part_names_from_ann(pairs[0][0], part_name_field)
    for ann, _ in pairs:
        names = part_names_from_ann(ann, part_name_field)
        if names != canonical:
            raise ValueError(
                f"part_class_name list differs for ann id={ann.get('id', 'unknown')}. "
                f"This script expects one fixed part order for target_class."
            )
    return canonical


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
            if bool(mask.any()):
                new_centers.append(x[mask].mean(dim=0))
            else:
                ridx = torch.randint(0, x.shape[0], (1,), generator=g).item()
                new_centers.append(x[ridx])
        centers = F.normalize(torch.stack(new_centers, dim=0).float(), dim=-1)

    return centers


def build_patch_bank(
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    patch_field: str,
    patches_per_image_for_kmeans: int,
    max_kmeans_patches: int,
) -> torch.Tensor:
    bank = []
    for ann, _ in tqdm(pairs, desc="collect V patches"):
        patches = normalize_patches(require_key(ann, patch_field, "annotation"))
        patches = sample_rows(patches, patches_per_image_for_kmeans)
        bank.append(patches)

    bank = torch.cat(bank, dim=0)
    if max_kmeans_patches > 0 and bank.shape[0] > max_kmeans_patches:
        bank = sample_rows(bank, max_kmeans_patches)
    return F.normalize(bank.float(), dim=-1)


def build_image_hist_and_gt(
    ann: Dict[str, Any],
    centers: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patches = normalize_patches(require_key(ann, args.patch_field, "annotation"))
    labels = assign_to_centers(patches, centers, chunk_size=args.assign_chunk_size)

    hist = torch.bincount(labels, minlength=args.num_prototypes).float()
    hist = hist / hist.sum().clamp_min(1.0)

    gt = to_tensor(require_key(ann, args.gt_mask_field, "annotation")).cpu().bool()
    gt_flat = gt.reshape(gt.shape[0], -1)

    if gt_flat.shape[1] != labels.numel():
        raise ValueError(
            f"GT flattened patches={gt_flat.shape[1]} != patch tokens={labels.numel()} "
            f"for ann id={ann.get('id', 'unknown')}"
        )

    return hist, labels, gt_flat


def build_part_proto_basis(
    rows_cache: List[Tuple[Dict[str, Any], Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, List[str]]],
    num_parts: int,
    num_prototypes: int,
    smoothing: float,
) -> torch.Tensor:
    """
    B[p, c] = P(prototype c | part p) estimated from GT-labeled patches.
    """
    counts = torch.full((num_parts, num_prototypes), float(smoothing), dtype=torch.float64)

    for _ann, _img, _hist, labels, gt_flat, _gt_parts in rows_cache:
        for p in range(num_parts):
            mask = gt_flat[p]
            if bool(mask.any()):
                counts[p] += torch.bincount(labels[mask], minlength=num_prototypes).double()

    B = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    return B.float()


def nnls_pg(
    h: torch.Tensor,
    B: torch.Tensor,
    steps: int,
    lr: float,
    l1: float,
) -> torch.Tensor:
    """
    Solve approximately:
        min_x || x @ B - h ||_2^2 + l1 * sum(x)
        s.t. x >= 0

    h: [C]
    B: [P, C]
    x: [P]
    """
    P = B.shape[0]
    x = torch.zeros(P, dtype=torch.float32)

    for _ in range(steps):
        recon = x @ B
        grad = 2.0 * ((recon - h) @ B.t())
        if l1 > 0:
            grad = grad + float(l1)
        x = torch.clamp(x - lr * grad, min=0.0)

    return x


def normalize_score(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.sum() <= eps:
        return x
    return x / x.sum().clamp_min(eps)


def predict_parts_from_coeff(
    x: torch.Tensor,
    part_names: List[str],
    args,
    global_thresholds: torch.Tensor = None,
) -> List[str]:
    P = len(part_names)

    if args.score_normalize == "sum":
        scores = normalize_score(x)
    elif args.score_normalize == "none":
        scores = x
    else:
        raise ValueError(f"Unknown score_normalize: {args.score_normalize}")

    if args.threshold_method == "topk":
        k = min(args.topk, P)
        mask = torch.zeros(P, dtype=torch.bool)
        mask[torch.topk(scores, k=k).indices] = True
    elif args.threshold_method == "image_zscore":
        thr = scores.mean() + args.image_z * scores.std(unbiased=False)
        mask = scores >= thr
    elif args.threshold_method == "global_percentile":
        if global_thresholds is None:
            raise ValueError("global_thresholds is required for global_percentile")
        mask = scores >= global_thresholds
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


def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target_class", default="person")
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--class_field", default="class_name")
    parser.add_argument("--patch_field", default="cropaug_patch_tokens")
    parser.add_argument("--part_name_field", default="part_class_name")
    parser.add_argument("--gt_mask_field", default="part_gt_mask_patch")
    parser.add_argument("--min_gt_patches", type=int, default=1)

    parser.add_argument("--num_prototypes", type=int, default=64)
    parser.add_argument("--kmeans_iters", type=int, default=25)
    parser.add_argument("--patches_per_image_for_kmeans", type=int, default=0, help="0 means use all patches from every image.")
    parser.add_argument("--max_kmeans_patches", type=int, default=0, help="0 means no global cap.")
    parser.add_argument("--assign_chunk_size", type=int, default=8192)

    parser.add_argument("--basis_smoothing", type=float, default=1e-3)

    parser.add_argument("--nnls_steps", type=int, default=300)
    parser.add_argument("--nnls_lr", type=float, default=0.5)
    parser.add_argument("--nnls_l1", type=float, default=0.0)
    parser.add_argument("--score_normalize", choices=["sum", "none"], default="sum")

    parser.add_argument("--threshold_method", choices=["topk", "image_zscore", "global_percentile"], default="image_zscore")
    parser.add_argument("--image_z", type=float, default=-0.5)
    parser.add_argument("--topk", type=int, default=9)
    parser.add_argument("--global_percentile", type=float, default=50.0)
    parser.add_argument("--min_pred_parts", type=int, default=1)
    parser.add_argument("--max_pred_parts", type=int, default=0, help="0 means no max.")

    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    print(f"[Load] {args.dataset}")
    data = torch.load(args.dataset, map_location="cpu")

    pairs_all = filter_target_samples(data, args.target_class, args.class_field)
    if len(pairs_all) == 0:
        raise RuntimeError(f"No samples found for class={args.target_class}")
    print(f"[Data] target_class={args.target_class}, num_samples_before_part_gt_filter={len(pairs_all)}")

    # This script intentionally evaluates only images with non-empty part GT.
    part_names = check_same_part_list(pairs_all, args.part_name_field)
    num_parts = len(part_names)
    print(f"[Parts] num_parts={num_parts}")
    print("[Parts]", ", ".join(part_names))

    filtered_pairs = []
    skipped_no_part_gt = 0
    for ann, img in pairs_all:
        gt_parts = gt_visible_parts(ann, part_names, args.gt_mask_field, args.min_gt_patches)
        if len(gt_parts) == 0:
            skipped_no_part_gt += 1
            continue
        filtered_pairs.append((ann, img))

    if len(filtered_pairs) == 0:
        raise RuntimeError("No samples with non-empty part GT.")

    print(f"[Data] samples_with_part_gt={len(filtered_pairs)}")
    print(f"[Data] skipped_no_part_gt={skipped_no_part_gt}")

    patch_bank = build_patch_bank(
        pairs=filtered_pairs,
        patch_field=args.patch_field,
        patches_per_image_for_kmeans=args.patches_per_image_for_kmeans,
        max_kmeans_patches=args.max_kmeans_patches,
    )

    print(f"[KMeans] patch_bank={tuple(patch_bank.shape)}, num_prototypes={args.num_prototypes}")
    centers = torch_kmeans(patch_bank, args.num_prototypes, args.kmeans_iters, args.seed)

    rows_cache = []
    for ann, img in tqdm(filtered_pairs, desc="build image histograms"):
        hist, labels, gt_flat = build_image_hist_and_gt(ann, centers, args)
        gt_parts = gt_visible_parts(ann, part_names, args.gt_mask_field, args.min_gt_patches)
        rows_cache.append((ann, img, hist, labels, gt_flat, gt_parts))

    B = build_part_proto_basis(
        rows_cache=rows_cache,
        num_parts=num_parts,
        num_prototypes=args.num_prototypes,
        smoothing=args.basis_smoothing,
    )

    # Solve coefficients for all images first.
    coeff_cache = []
    for ann, img, hist, labels, gt_flat, gt_parts in tqdm(rows_cache, desc="solve NNLS equations"):
        x = nnls_pg(
            h=hist,
            B=B,
            steps=args.nnls_steps,
            lr=args.nnls_lr,
            l1=args.nnls_l1,
        )
        if args.score_normalize == "sum":
            score = normalize_score(x)
        else:
            score = x
        coeff_cache.append((ann, img, x, score, gt_parts))

    global_thresholds = None
    if args.threshold_method == "global_percentile":
        all_scores = torch.stack([row[3] for row in coeff_cache], dim=0)
        global_thresholds = torch.quantile(all_scores, q=args.global_percentile / 100.0, dim=0)

    csv_path = os.path.join(args.out_dir, f"{args.target_class}_vonly_equation_presence_eval.csv")
    metrics_accum = []
    num_pred_list, num_gt_list = [], []

    # All-parts baseline under the same per-image mean metric.
    all_parts = list(part_names)
    baseline_metrics = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ann_id", "image_id", "file_name",
                "pred_parts", "gt_parts", "num_pred", "num_gt",
                "set_iou", "exact_match", "precision", "recall", "f1",
                "coefficients",
            ],
        )
        writer.writeheader()

        for ann, img, x, _score, gt_parts in coeff_cache:
            pred_parts = predict_parts_from_coeff(x, part_names, args, global_thresholds)
            m = set_metrics(pred_parts, gt_parts)
            b = set_metrics(all_parts, gt_parts)

            metrics_accum.append(m)
            baseline_metrics.append(b)
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
                "coefficients": ";".join(f"{float(v):.6f}" for v in x.tolist()),
            })

    def mean_metric(ms: List[Dict[str, float]], name: str) -> float:
        return mean([m[name] for m in ms])

    summary = {
        "target_class": args.target_class,
        "num_images": len(metrics_accum),
        "num_target_samples_before_filter": len(pairs_all),
        "skipped_no_part_gt": skipped_no_part_gt,
        "num_parts": num_parts,
        "part_names": part_names,
        "num_prototypes": args.num_prototypes,
        "primary_metric": "mean_set_iou",
        "mean_set_iou": mean_metric(metrics_accum, "set_iou"),
        "exact_match_rate": mean_metric(metrics_accum, "exact_match"),
        "mean_precision": mean_metric(metrics_accum, "precision"),
        "mean_recall": mean_metric(metrics_accum, "recall"),
        "mean_f1": mean_metric(metrics_accum, "f1"),
        "avg_num_pred": mean([float(x) for x in num_pred_list]),
        "avg_num_gt": mean([float(x) for x in num_gt_list]),

        "all_parts_baseline": {
            "mean_set_iou": mean_metric(baseline_metrics, "set_iou"),
            "exact_match_rate": mean_metric(baseline_metrics, "exact_match"),
            "mean_precision": mean_metric(baseline_metrics, "precision"),
            "mean_recall": mean_metric(baseline_metrics, "recall"),
            "mean_f1": mean_metric(baseline_metrics, "f1"),
            "avg_num_pred": float(num_parts),
        },

        "threshold_method": args.threshold_method,
        "image_z": args.image_z,
        "topk": args.topk,
        "global_percentile": args.global_percentile,
        "min_pred_parts": args.min_pred_parts,
        "max_pred_parts": args.max_pred_parts,
        "score_normalize": args.score_normalize,

        "nnls_steps": args.nnls_steps,
        "nnls_lr": args.nnls_lr,
        "nnls_l1": args.nnls_l1,
        "basis_smoothing": args.basis_smoothing,
        "kmeans_iters": args.kmeans_iters,
        "patches_per_image_for_kmeans": args.patches_per_image_for_kmeans,
        "max_kmeans_patches": args.max_kmeans_patches,

        "note": (
            "V-only equation-style analysis. Visual prototypes are learned from cropaug_patch_tokens only. "
            "GT masks are used after clustering to build the part-to-prototype basis B and to evaluate. "
            "No text features, no projector, no W."
        ),
    }

    summary_path = os.path.join(args.out_dir, f"{args.target_class}_vonly_equation_presence_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== V-only Equation Presence Summary ==========")
    print(f"class              : {args.target_class}")
    print(f"num_images         : {summary['num_images']}")
    print(f"skipped no part GT : {summary['skipped_no_part_gt']}")
    print(f"num parts/protos   : {summary['num_parts']} / {summary['num_prototypes']}")
    print(f"primary mean IoU   : {summary['mean_set_iou']:.6f}")
    print(f"exact match rate   : {summary['exact_match_rate']:.6f}")
    print(f"mean precision     : {summary['mean_precision']:.6f}")
    print(f"mean recall        : {summary['mean_recall']:.6f}")
    print(f"mean F1            : {summary['mean_f1']:.6f}")
    print(f"avg #pred / #gt    : {summary['avg_num_pred']:.3f} / {summary['avg_num_gt']:.3f}")
    print("---------- all-parts baseline ----------")
    print(f"baseline IoU       : {summary['all_parts_baseline']['mean_set_iou']:.6f}")
    print(f"baseline precision : {summary['all_parts_baseline']['mean_precision']:.6f}")
    print(f"baseline recall    : {summary['all_parts_baseline']['mean_recall']:.6f}")
    print(f"baseline F1        : {summary['all_parts_baseline']['mean_f1']:.6f}")
    print(f"CSV                : {csv_path}")
    print(f"Summary            : {summary_path}")


if __name__ == "__main__":
    main()
