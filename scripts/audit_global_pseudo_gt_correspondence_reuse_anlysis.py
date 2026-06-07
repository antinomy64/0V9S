#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Put this file under scripts/ in the Talk2DINO repo.
# It reuses scripts/anlysis.py directly instead of reimplementing dataset/model/loss collection.
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

try:
    # When this file is placed in scripts/, anlysis.py is a sibling.
    from anlysis import FeatureAnalyser, mean_features_by_part, get_part_names
except Exception:
    # When imported/executed from repo root as a package-like path.
    from scripts.anlysis import FeatureAnalyser, mean_features_by_part, get_part_names

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def rankdata_1d(x: torch.Tensor) -> torch.Tensor:
    """Average-rank implementation for Spearman; no scipy.stats dependency."""
    x = x.detach().float().cpu()
    n = int(x.numel())
    if n == 0:
        return x
    order = torch.argsort(x)
    ranks = torch.empty(n, dtype=torch.float32)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j].item() == sorted_x[i].item():
            j += 1
        # ranks are 1-based; tied values receive average rank.
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = float(avg_rank)
        i = j
    return ranks


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.float().cpu()
    y = y.float().cpu()
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() + eps) * torch.sqrt((y * y).sum() + eps)
    return float(((x * y).sum() / denom).item())


def spearman_vec(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    return pearson_corr(rankdata_1d(x), rankdata_1d(y))


def spearman_offdiag(a: torch.Tensor, b: torch.Tensor) -> float:
    n = int(a.shape[0])
    if n < 3:
        return float("nan")
    tri = torch.triu_indices(n, n, offset=1)
    return spearman_vec(a[tri[0], tri[1]], b[tri[0], tri[1]])


def split_object_name(part_name: str) -> str:
    # VOC116 names usually look like "dog's nose".
    if "'s " in part_name:
        return part_name.split("'s ", 1)[0]
    return part_name.split("/", 1)[0].split(".", 1)[0]


def build_object_to_part_ids(part_names: List[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for pid, name in enumerate(part_names):
        obj = split_object_name(str(name))
        out.setdefault(obj, []).append(pid)
    return out


def hungarian_maximize(sim: torch.Tensor) -> Tuple[List[int], List[int]]:
    """Return row indices and col indices maximizing sim."""
    n, m = sim.shape
    if n == 0 or m == 0:
        return [], []
    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment((-sim).detach().cpu().numpy())
        return [int(x) for x in row_ind], [int(x) for x in col_ind]

    # Small fallback: greedy unique matching. Not optimal but avoids hard dependency.
    used_r, used_c = set(), set()
    pairs = []
    flat = []
    for i in range(n):
        for j in range(m):
            flat.append((float(sim[i, j].item()), i, j))
    for _, i, j in sorted(flat, reverse=True):
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        pairs.append((i, j))
        if len(used_r) == min(n, m):
            break
    pairs.sort()
    return [i for i, _ in pairs], [j for _, j in pairs]


def self_ranks_and_margins(sim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For square sim where identity is intended correspondence."""
    k = int(sim.shape[0])
    ranks = []
    self_cos = []
    margins = []
    for i in range(k):
        vals = sim[i]
        target = vals[i]
        # rank is 1 + number strictly greater than self.
        rank = int((vals > target).sum().item()) + 1
        if k > 1:
            wrong = torch.cat([vals[:i], vals[i + 1:]])
            margin = target - wrong.max()
        else:
            margin = torch.tensor(float("nan"))
        ranks.append(rank)
        self_cos.append(float(target.item()))
        margins.append(float(margin.item()))
    return torch.tensor(ranks), torch.tensor(self_cos), torch.tensor(margins)


def mean_float(xs: List[float]) -> float:
    xs = [float(x) for x in xs if not math.isnan(float(x))]
    if len(xs) == 0:
        return float("nan")
    return float(sum(xs) / len(xs))


def audit(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[reuse] Using FeatureAnalyser.collect_vision_feature() from scripts/anlysis.py")
    analyser = FeatureAnalyser(
        model_config=args.model_config,
        dataset=args.dataset,
        init_weights=args.init_weights,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_parts=args.num_parts,
        device=args.device,
        show_progress=not args.no_progress,
    )

    # Reuse existing analyser: fake = image-wise pseudo prototypes by source part slot;
    # gt = GT-mask visual prototypes by true global part id.
    fake_by_part, gt_by_part = analyser.collect_vision_feature()

    dino_dim = int(analyser.cfg["model"].get("dino_embed_dim", 768))
    pseudo_mean, pseudo_valid, pseudo_count = mean_features_by_part(fake_by_part, dim=dino_dim)
    gt_mean, gt_valid, gt_count = mean_features_by_part(gt_by_part, dim=dino_dim)

    pseudo_mean = safe_normalize(pseudo_mean.float(), dim=-1)
    gt_mean = safe_normalize(gt_mean.float(), dim=-1)

    part_names = get_part_names(args.num_parts)
    obj_to_pids = build_object_to_part_ids(part_names)

    torch.save(
        {
            "pseudo_mean": pseudo_mean,
            "pseudo_valid": pseudo_valid,
            "pseudo_count": pseudo_count,
            "gt_mean": gt_mean,
            "gt_valid": gt_valid,
            "gt_count": gt_count,
            "part_names": part_names,
            "meta": {
                "model_config": args.model_config,
                "dataset": args.dataset,
                "init_weights": args.init_weights,
                "num_parts": args.num_parts,
                "source": "FeatureAnalyser.collect_vision_feature",
            },
        },
        out_dir / "global_pseudo_gt_prototypes_reuse_anlysis.pth",
    )

    per_obj_rows = []
    per_part_rows = []

    for obj_name, pids_all in obj_to_pids.items():
        pids = [pid for pid in pids_all if bool(pseudo_valid[pid]) and bool(gt_valid[pid])]
        if len(pids) == 0:
            per_obj_rows.append({
                "object": obj_name,
                "num_parts": 0,
                "direct_top1_acc": "nan",
                "direct_mean_self_cos": "nan",
                "direct_mean_self_margin": "nan",
                "hungarian_mean_cos": "nan",
                "hungarian_acc_identity_label": "nan",
                "rho_pseudo_vs_gt_identity_order": "nan",
                "rho_pseudo_vs_gt_hungarian_order": "nan",
            })
            continue

        P = pseudo_mean[pids]  # [K,D], source slots
        G = gt_mean[pids]      # [K,D], true GT labels in identity order
        sim = P @ G.T          # [K,K]
        K = len(pids)

        ranks, self_cos, self_margin = self_ranks_and_margins(sim)
        direct_top1_acc = float((ranks == 1).float().mean().item())
        direct_top5_acc = float((ranks <= min(5, K)).float().mean().item())

        row_ind, col_ind = hungarian_maximize(sim)
        row_to_col = {r: c for r, c in zip(row_ind, col_ind)}
        hungarian_cos = [float(sim[r, c].item()) for r, c in zip(row_ind, col_ind)]
        hungarian_mean_cos = mean_float(hungarian_cos)
        hungarian_acc_identity = float(sum(1 for r, c in zip(row_ind, col_ind) if r == c) / max(len(row_ind), 1))

        pseudo_graph = P @ P.T
        gt_graph_identity = G @ G.T
        rho_identity = spearman_offdiag(pseudo_graph, gt_graph_identity)

        if len(row_ind) >= 2:
            # Reorder GT columns/rows according to Hungarian assignment for source pseudo row order.
            matched_cols = [row_to_col[i] for i in range(K) if i in row_to_col]
            matched_rows = [i for i in range(K) if i in row_to_col]
            P_h = P[matched_rows]
            G_h = G[matched_cols]
            rho_hungarian = spearman_offdiag(P_h @ P_h.T, G_h @ G_h.T)
        else:
            rho_hungarian = float("nan")

        per_obj_rows.append({
            "object": obj_name,
            "num_parts": K,
            "direct_top1_acc": direct_top1_acc,
            "direct_top5_acc": direct_top5_acc,
            "direct_mean_self_cos": float(self_cos.mean().item()),
            "direct_mean_self_margin": float(self_margin[~torch.isnan(self_margin)].mean().item()) if (~torch.isnan(self_margin)).any() else "nan",
            "median_self_rank": float(ranks.float().median().item()),
            "hungarian_mean_cos": hungarian_mean_cos,
            "hungarian_acc_identity_label": hungarian_acc_identity,
            "rho_pseudo_vs_gt_identity_order": rho_identity,
            "rho_pseudo_vs_gt_hungarian_order": rho_hungarian,
        })

        for local_i, pid in enumerate(pids):
            vals = sim[local_i]
            top1_local = int(torch.argmax(vals).item())
            hung_local = int(row_to_col.get(local_i, -1))
            per_part_rows.append({
                "object": obj_name,
                "pid": pid,
                "source_part": part_names[pid],
                "pseudo_count": int(pseudo_count[pid].item()),
                "gt_count": int(gt_count[pid].item()),
                "self_rank_identity": int(ranks[local_i].item()),
                "self_cos_identity": float(self_cos[local_i].item()),
                "self_margin_identity": float(self_margin[local_i].item()) if not math.isnan(float(self_margin[local_i].item())) else "nan",
                "top1_gt_pid": pids[top1_local],
                "top1_gt_part": part_names[pids[top1_local]],
                "top1_cos": float(vals[top1_local].item()),
                "hungarian_gt_pid": pids[hung_local] if hung_local >= 0 else -1,
                "hungarian_gt_part": part_names[pids[hung_local]] if hung_local >= 0 else "",
                "hungarian_cos": float(vals[hung_local].item()) if hung_local >= 0 else "nan",
                "hungarian_is_identity": bool(hung_local == local_i),
            })

    per_obj_csv = out_dir / "per_object_matching_summary.csv"
    with per_obj_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_obj_rows[0].keys()) if per_obj_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_obj_rows)

    per_part_csv = out_dir / "per_part_pseudo_to_gt_matching.csv"
    with per_part_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_part_rows[0].keys()) if per_part_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_part_rows)

    valid_obj = [r for r in per_obj_rows if int(r.get("num_parts", 0)) > 0]
    summary = {
        "num_parts": args.num_parts,
        "pseudo_valid_parts": int(pseudo_valid.sum().item()),
        "gt_valid_parts": int(gt_valid.sum().item()),
        "num_objects_with_valid_parts": len(valid_obj),
        "mean_direct_top1_acc": mean_float([float(r["direct_top1_acc"]) for r in valid_obj]),
        "mean_direct_top5_acc": mean_float([float(r["direct_top5_acc"]) for r in valid_obj if "direct_top5_acc" in r]),
        "mean_direct_self_cos": mean_float([float(r["direct_mean_self_cos"]) for r in valid_obj]),
        "mean_direct_self_margin": mean_float([float(r["direct_mean_self_margin"]) for r in valid_obj]),
        "mean_hungarian_mean_cos": mean_float([float(r["hungarian_mean_cos"]) for r in valid_obj]),
        "mean_hungarian_acc_identity_label": mean_float([float(r["hungarian_acc_identity_label"]) for r in valid_obj]),
        "mean_rho_identity_order": mean_float([float(r["rho_pseudo_vs_gt_identity_order"]) for r in valid_obj]),
        "mean_rho_hungarian_order": mean_float([float(r["rho_pseudo_vs_gt_hungarian_order"]) for r in valid_obj]),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[summary]")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n[saved] {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Reuse scripts/anlysis.py FeatureAnalyser to audit global pseudo-vs-GT prototype correspondence."
    )
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
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_parts", type=int, default=116)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    audit(parse_args())
