#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset_joint import DinoClipJointDataset  # noqa: E402
from src.dataset_global import CategoryPatchPoolDataset, global_pool_collate_fn  # noqa: E402
from src.loss_global import PartLoss  # noqa: E402

try:
    from scripts.anlysis import get_part_names  # type: ignore
except Exception:
    try:
        from anlysis import get_part_names  # type: ignore
    except Exception:
        def get_part_names(num_parts: int) -> List[str]:
            return [f"part_{i}" for i in range(num_parts)]

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def load_model(model_config: str, init_weights: str, device: torch.device):
    with open(model_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cls = cfg["model"].get("model_class", "ProjectionLayer")
    Model = getattr(importlib.import_module("src.model"), model_cls)
    model = Model.from_config(cfg["model"]).to(device)

    print(f"[model] load {init_weights}")
    ckpt = torch.load(init_weights, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict):
        ckpt = {str(k).replace("module.", "", 1): v for k, v in ckpt.items()}
    msg = model.load_state_dict(ckpt, strict=False)
    print("[model] missing keys:", getattr(msg, "missing_keys", []))
    print("[model] unexpected keys:", getattr(msg, "unexpected_keys", []))
    model.eval()
    return model, cfg


def build_joint_dataset(args, cfg) -> DinoClipJointDataset:
    min_obj_area_ratio = float(cfg.get("dataset", {}).get("min_obj_area_ratio", args.min_obj_area_ratio))
    return DinoClipJointDataset(
        args.dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=".tar" in args.dataset,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=min_obj_area_ratio,
    )


def category_name_map(dataset: DinoClipJointDataset) -> Dict[int, str]:
    out: Dict[int, str] = {}
    samples = dataset.data.values() if isinstance(dataset.data, dict) else dataset.data
    for s in samples:
        cat = int(s["category_id"])
        if cat not in out:
            out[cat] = str(s.get("class_name", f"class_{cat}"))
    return out


def rank_vector(x: torch.Tensor) -> torch.Tensor:
    # Average-tie-free ranking by argsort; enough for diagnostic Spearman.
    order = torch.argsort(x)
    r = torch.empty_like(order, dtype=torch.float32)
    r[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return r


def corrcoef_1d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    den = x.norm() * y.norm()
    if float(den.item()) <= eps:
        return float("nan")
    return float((x * y).sum().div(den + eps).item())


def spearman_offdiag(a: torch.Tensor, b: torch.Tensor) -> float:
    k = int(a.shape[0])
    if k < 3:
        return float("nan")
    tri = torch.triu_indices(k, k, offset=1)
    av = a[tri[0], tri[1]].detach().cpu()
    bv = b[tri[0], tri[1]].detach().cpu()
    return corrcoef_1d(rank_vector(av), rank_vector(bv))


def hungarian_max(sim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return row indices and col indices maximizing sim."""
    if linear_sum_assignment is None:
        # Greedy fallback, only for environments without scipy.
        rows = []
        cols = []
        used_r = set()
        used_c = set()
        flat = torch.argsort(sim.reshape(-1), descending=True)
        n, m = sim.shape
        for fid in flat.tolist():
            r = fid // m
            c = fid % m
            if r in used_r or c in used_c:
                continue
            rows.append(r)
            cols.append(c)
            used_r.add(r)
            used_c.add(c)
            if len(rows) == min(n, m):
                break
        return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)

    row_np, col_np = linear_sum_assignment((-sim.detach().cpu()).numpy())
    return torch.tensor(row_np, dtype=torch.long), torch.tensor(col_np, dtype=torch.long)


@torch.no_grad()
def build_global_pool_pseudo_and_gt(args, model, cfg, joint_dataset):
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    dino_dim = int(cfg["model"].get("dino_embed_dim", 768))
    num_parts = int(args.num_parts)
    train_cfg = cfg.get("train", {})
    patch_temperature = float(args.patch_temperature if args.patch_temperature is not None else train_cfg.get("patch_temperature", 0.07))
    em_iters = int(args.em_iters if args.em_iters is not None else train_cfg.get("em_iters", 1))

    pool_dataset = CategoryPatchPoolDataset(
        joint_dataset,
        sample_patches_per_step=args.sample_patches_per_step,
        steps_per_epoch=None,
        store_dtype=torch.float16,
        seed=args.seed,
        fixed_subsample=bool(args.fixed_subsample),
    )

    helper = PartLoss(
        model,
        lambda_inst=0.0,
        lambda_overlap=0.0,
        lambda_spear=0.0,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
    ).to(device)
    helper.present_only_anchor = bool(args.present_only_anchor)
    helper.eval()

    pseudo_mean = torch.zeros((num_parts, dino_dim), dtype=torch.float32)
    gt_mean = torch.zeros((num_parts, dino_dim), dtype=torch.float32)
    pseudo_valid = torch.zeros((num_parts,), dtype=torch.bool)
    gt_valid = torch.zeros((num_parts,), dtype=torch.bool)
    pseudo_count = torch.zeros((num_parts,), dtype=torch.long)
    gt_count = torch.zeros((num_parts,), dtype=torch.long)

    part_to_cat: Dict[int, int] = {}
    cat_to_part_ids: Dict[int, List[int]] = {}

    anchor_total_valid = 0.0
    anchor_total_hits = 0.0

    for idx in tqdm(range(len(pool_dataset.categories)), desc="global pool pseudo/GT"):
        item = pool_dataset[idx]
        batch = to_device(global_pool_collate_fn([item]), device)

        patch_tokens = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask_patch = batch["obj_mask_patch"].bool()
        part_gt_mask_patch = batch["part_gt_mask_patch"].bool()
        part_valid_mask = batch["part_valid_mask"].bool()
        part_ids = batch["part_category_id"].long()
        part_text_feat = batch["part_text_feat"].float()
        cat = int(batch["category_id"][0].item())

        if bool(args.present_only_anchor):
            part_present_mask = (part_gt_mask_patch & obj_mask_patch[:, None, :]).sum(dim=-1) > 0
            part_anchor_mask = part_valid_mask & part_present_mask
        else:
            part_anchor_mask = part_valid_mask

        part_proj = model.project_clip_txt(part_text_feat)
        part_proj = safe_normalize(part_proj, dim=-1)
        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / patch_temperature
        abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        ret = helper._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_anchor_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=em_iters,
        )
        z_proto = safe_normalize(ret[0].float(), dim=-1)  # [1,K,D]
        metrics = ret[1]
        anchor_total_valid += float(metrics["anchor_total_valid_parts"].detach().cpu().item())
        anchor_total_hits += float(metrics["anchor_total_hits"].detach().cpu().item())

        # Global GT prototypes from the same full category pool.
        gt_mask = part_gt_mask_patch & obj_mask_patch[:, None, :] & part_valid_mask[:, :, None]
        gt_pix = gt_mask.sum(dim=-1).float()  # [1,K]
        gt_proto = torch.einsum("bkn,bnd->bkd", gt_mask.float(), patch_tokens)
        gt_proto = gt_proto / gt_pix.clamp_min(1.0)[:, :, None]
        gt_proto = safe_normalize(gt_proto.float(), dim=-1)

        K = int(part_ids.shape[1])
        cat_to_part_ids[cat] = []
        for k in range(K):
            pid = int(part_ids[0, k].item())
            if pid < 0 or pid >= num_parts:
                continue
            cat_to_part_ids[cat].append(pid)
            part_to_cat[pid] = cat
            if bool(part_anchor_mask[0, k].item()):
                pseudo_mean[pid] = z_proto[0, k].detach().cpu()
                pseudo_valid[pid] = True
                pseudo_count[pid] = 1
            if bool(part_valid_mask[0, k].item()) and float(gt_pix[0, k].item()) > 0:
                gt_mean[pid] = gt_proto[0, k].detach().cpu()
                gt_valid[pid] = True
                gt_count[pid] = int(gt_pix[0, k].item())

    print(
        "[anchor]",
        f"valid={anchor_total_valid:.0f}",
        f"hits={anchor_total_hits:.0f}",
        f"hit_rate={(0.0 if anchor_total_valid <= 0 else anchor_total_hits / anchor_total_valid):.4f}",
    )
    print("[valid]", f"pseudo={int(pseudo_valid.sum())}/{num_parts}", f"gt={int(gt_valid.sum())}/{num_parts}")

    return {
        "pseudo_mean": pseudo_mean,
        "gt_mean": gt_mean,
        "pseudo_valid": pseudo_valid,
        "gt_valid": gt_valid,
        "pseudo_count": pseudo_count,
        "gt_count": gt_count,
        "cat_to_part_ids": cat_to_part_ids,
        "part_to_cat": part_to_cat,
        "pool_categories": list(pool_dataset.categories),
        "patch_temperature": patch_temperature,
        "em_iters": em_iters,
        "sample_patches_per_step": args.sample_patches_per_step,
    }


def analyze_and_write(args, proto: Dict[str, Any], cat_names: Dict[int, str], out_dir: Path):
    part_names = get_part_names(int(args.num_parts))
    pseudo = safe_normalize(proto["pseudo_mean"].float(), dim=-1)
    gt = safe_normalize(proto["gt_mean"].float(), dim=-1)
    pseudo_valid = proto["pseudo_valid"].bool()
    gt_valid = proto["gt_valid"].bool()
    cat_to_part_ids: Dict[int, List[int]] = proto["cat_to_part_ids"]

    per_obj_rows: List[Dict[str, Any]] = []
    per_part_rows: List[Dict[str, Any]] = []

    for cat in sorted(cat_to_part_ids.keys()):
        pids = [int(p) for p in cat_to_part_ids[cat] if int(p) < int(args.num_parts)]
        pids = [p for p in pids if bool(pseudo_valid[p]) and bool(gt_valid[p])]
        K = len(pids)
        if K == 0:
            continue
        pv = pseudo[pids]
        gv = gt[pids]
        sim = pv @ gv.T  # source pseudo rows vs GT cols, same object

        top_vals, top_cols = sim.max(dim=1)
        direct_top1 = [(pids[int(top_cols[i].item())] == pids[i]) for i in range(K)]
        direct_top1_acc = float(sum(direct_top1) / K)
        topk = min(5, K)
        topk_cols = torch.topk(sim, k=topk, dim=1).indices
        direct_top5 = []
        for i in range(K):
            direct_top5.append(pids[i] in [pids[int(c.item())] for c in topk_cols[i]])
        direct_top5_acc = float(sum(direct_top5) / K)
        self_cos = sim.diag()
        if K > 1:
            wrong = sim.clone()
            wrong[torch.arange(K), torch.arange(K)] = -1e9
            self_margin = self_cos - wrong.max(dim=1).values
        else:
            self_margin = torch.full_like(self_cos, float("nan"))

        row_ind, col_ind = hungarian_max(sim)
        match_src_to_col = {int(r.item()): int(c.item()) for r, c in zip(row_ind, col_ind)}
        hung_cos = torch.stack([sim[r, c] for r, c in zip(row_ind, col_ind)]) if len(row_ind) else torch.tensor([])
        hung_identity = [pids[match_src_to_col[i]] == pids[i] for i in range(K) if i in match_src_to_col]

        pseudo_graph = pv @ pv.T
        gt_graph_identity = gv @ gv.T
        rho_identity = spearman_offdiag(pseudo_graph, gt_graph_identity)
        if len(match_src_to_col) == K:
            matched_cols = [match_src_to_col[i] for i in range(K)]
            gt_graph_hung = gv[matched_cols] @ gv[matched_cols].T
            rho_hung = spearman_offdiag(pseudo_graph, gt_graph_hung)
        else:
            rho_hung = float("nan")

        per_obj_rows.append({
            "category_id": cat,
            "class_name": cat_names.get(cat, f"class_{cat}"),
            "num_parts": K,
            "direct_top1_acc": direct_top1_acc,
            "direct_top5_acc": direct_top5_acc,
            "direct_self_cos_mean": float(torch.nanmean(self_cos).item()),
            "direct_self_margin_mean": float(torch.nanmean(self_margin).item()) if K > 1 else "",
            "hungarian_mean_cos": float(hung_cos.mean().item()) if hung_cos.numel() else "",
            "hungarian_acc_identity_label": float(sum(hung_identity) / len(hung_identity)) if hung_identity else "",
            "rho_pseudo_vs_gt_identity_order": rho_identity,
            "rho_pseudo_vs_gt_hungarian_order": rho_hung,
        })

        for i, src_pid in enumerate(pids):
            top_col = int(top_cols[i].item())
            hung_col = match_src_to_col.get(i, -1)
            if K > 1:
                rank = int((sim[i] > sim[i, i]).sum().item()) + 1
                margin = float(self_margin[i].item())
            else:
                rank = 1
                margin = float("nan")
            per_part_rows.append({
                "category_id": cat,
                "class_name": cat_names.get(cat, f"class_{cat}"),
                "source_part_id": src_pid,
                "source_part": part_names[src_pid] if src_pid < len(part_names) else f"part_{src_pid}",
                "top1_gt_part_id": pids[top_col],
                "top1_gt_part": part_names[pids[top_col]] if pids[top_col] < len(part_names) else f"part_{pids[top_col]}",
                "top1_cos": float(top_vals[i].item()),
                "hungarian_gt_part_id": pids[hung_col] if hung_col >= 0 else -1,
                "hungarian_gt_part": (part_names[pids[hung_col]] if hung_col >= 0 and pids[hung_col] < len(part_names) else ""),
                "hungarian_cos": float(sim[i, hung_col].item()) if hung_col >= 0 else "",
                "self_rank_identity": rank,
                "self_cos_identity": float(sim[i, i].item()),
                "self_margin_identity": margin,
            })

    def finite_mean(key: str):
        vals = []
        for r in per_obj_rows:
            v = r.get(key, "")
            try:
                vf = float(v)
                if torch.isfinite(torch.tensor(vf)):
                    vals.append(vf)
            except Exception:
                pass
        return float(sum(vals) / len(vals)) if vals else float("nan")

    summary = {
        "num_parts": int(args.num_parts),
        "pseudo_valid_parts": int(pseudo_valid.sum().item()),
        "gt_valid_parts": int(gt_valid.sum().item()),
        "num_objects_with_valid_parts": len(per_obj_rows),
        "mean_direct_top1_acc": finite_mean("direct_top1_acc"),
        "mean_direct_top5_acc": finite_mean("direct_top5_acc"),
        "mean_direct_self_cos": finite_mean("direct_self_cos_mean"),
        "mean_direct_self_margin": finite_mean("direct_self_margin_mean"),
        "mean_hungarian_mean_cos": finite_mean("hungarian_mean_cos"),
        "mean_hungarian_acc_identity_label": finite_mean("hungarian_acc_identity_label"),
        "mean_rho_identity_order": finite_mean("rho_pseudo_vs_gt_identity_order"),
        "mean_rho_hungarian_order": finite_mean("rho_pseudo_vs_gt_hungarian_order"),
    }

    proto_save = dict(proto)
    proto_save["part_names"] = part_names
    proto_save["category_names"] = cat_names
    proto_save["summary"] = summary
    torch.save(proto_save, out_dir / "global_pool_pseudo_gt_prototypes.pth")

    if per_obj_rows:
        with (out_dir / "per_object_matching_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_obj_rows[0].keys()))
            w.writeheader(); w.writerows(per_obj_rows)
    if per_part_rows:
        with (out_dir / "per_part_pseudo_to_gt_matching.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_part_rows[0].keys()))
            w.writeheader(); w.writerows(per_part_rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[summary]")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n[saved] {out_dir}")


def main():
    p = argparse.ArgumentParser("Audit global-pool pseudo prototypes vs GT prototypes with per-object Hungarian matching.")
    p.add_argument("--model_config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--init_weights", required=True)
    p.add_argument("--obj_feature_name", default="avg_self_attn_out")
    p.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    p.add_argument("--obj_text_name", default="ann_feats")
    p.add_argument("--part_text_name", default="part_ann_feats")
    p.add_argument("--resize_dim", type=int, default=448)
    p.add_argument("--crop_dim", type=int, default=448)
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--with_background", action="store_true", default=False)
    p.add_argument("--path_prefix", default=None)
    p.add_argument("--min_obj_area_ratio", type=float, default=0.0)
    p.add_argument("--num_parts", type=int, default=116)
    p.add_argument("--device", default="cuda")
    p.add_argument("--patch_temperature", type=float, default=None)
    p.add_argument("--em_iters", type=int, default=None)
    p.add_argument("--sample_patches_per_step", type=int, default=None, help="None means full category pool; use 65536 for sampled audit.")
    p.add_argument("--fixed_subsample", action="store_true", default=False)
    p.add_argument("--present_only_anchor", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.model_config, args.init_weights, device)
    joint_dataset = build_joint_dataset(args, cfg)
    cat_names = category_name_map(joint_dataset)
    proto = build_global_pool_pseudo_and_gt(args, model, cfg, joint_dataset)
    analyze_and_write(args, proto, cat_names, out_dir)


if __name__ == "__main__":
    main()
