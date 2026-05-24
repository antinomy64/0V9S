#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Spearman between raw text features T and GT-mask visual prototypes V.

Object level:
  Spearman( upper_tri(cos(raw object T)), upper_tri(cos(GT-mask object V)) )

Part level:
  For each object category:
  Spearman( upper_tri(cos(raw part T)), upper_tri(cos(GT-mask part V)) )

Obj-part relation:
  For each object category:
  Spearman( cos(raw object T, raw part T_i), cos(GT-mask object V, GT-mask part V_i) )
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def pairwise_cosine_sim(x: torch.Tensor) -> torch.Tensor:
    x = safe_normalize(x.float(), dim=-1)
    return x @ x.T


def upper_tri_no_diag(x: torch.Tensor) -> np.ndarray:
    k = x.shape[0]
    idx = torch.triu_indices(k, k, offset=1, device=x.device)
    return x[idx[0], idx[1]].detach().cpu().numpy().astype(np.float64)


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    if x.size < 2:
        return float("nan")
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def mse_np(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(x) - np.asarray(y)) ** 2))


def mean_feat(xs: List[torch.Tensor]) -> torch.Tensor:
    return safe_normalize(torch.stack([x.float().cpu() for x in xs], dim=0).mean(dim=0), dim=-1)


def pick_key(batch: Dict[str, Any], candidates: List[str]) -> str:
    for k in candidates:
        if k in batch:
            return k
    raise KeyError(f"None of keys found: {candidates}. Available keys={sorted(batch.keys())}")


def build_dataset(args, cfg):
    min_obj_area_ratio = float(cfg.get("dataset", {}).get("min_obj_area_ratio", 0.0))
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


def clean_name(x: Any) -> str:
    return str(x).replace("\n", " ").strip()


def collect_names_from_raw_pth(dataset_path: str) -> Tuple[Dict[int, str], Dict[Tuple[int, int], str]]:
    cat_counter = defaultdict(Counter)
    part_counter = defaultdict(Counter)
    obj = torch.load(dataset_path, map_location="cpu")
    anns = obj.get("annotations", []) if isinstance(obj, dict) else []

    for ann in anns:
        if "category_id" not in ann:
            continue
        cat = int(ann["category_id"])
        cat_counter[cat][clean_name(ann.get("class_name", str(cat)))] += 1
        part_ids = ann.get("part_category_id", []) or []
        part_names = ann.get("part_class_name", []) or []
        for i, pid in enumerate(part_ids):
            if i < len(part_names):
                part_counter[(cat, int(pid))][clean_name(part_names[i])] += 1

    cat_names = {cat: ctr.most_common(1)[0][0] for cat, ctr in cat_counter.items()}
    part_names = {key: ctr.most_common(1)[0][0] for key, ctr in part_counter.items()}
    return cat_names, part_names


@torch.no_grad()
def collect_raw_text_and_gtmask_visual_prototypes(args, cfg, device):
    dataset = build_dataset(args, cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=joint_collate_fn,
        pin_memory=True,
    )

    obj_text_bank = defaultdict(list)
    obj_visual_bank = defaultdict(list)
    part_text_bank = defaultdict(list)
    part_visual_bank = defaultdict(list)

    for batch in tqdm(loader, total=len(loader), desc="collect raw T and GTmask V"):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        cat_key = pick_key(batch, ["category_id", "obj_category_id"])
        obj_text_key = pick_key(batch, ["obj_text_feat", "ann_feats", "obj_ann_feats"])
        part_text_key = pick_key(batch, ["part_text_feat", "part_ann_feats"])
        part_id_key = pick_key(batch, ["part_category_id"])
        patch_key = pick_key(batch, ["patch_tokens", args.part_feature_name])
        obj_mask_key = pick_key(batch, ["obj_mask_patch"])
        part_valid_key = pick_key(batch, ["part_valid_mask"])
        part_gt_key = pick_key(batch, ["part_gt_mask_patch"])

        cat_ids = batch[cat_key].long()
        obj_text = batch[obj_text_key].float()
        part_text = batch[part_text_key].float()
        part_ids = batch[part_id_key].long()

        patch_tokens = safe_normalize(batch[patch_key].float(), dim=-1)
        obj_mask = batch[obj_mask_key].bool()
        part_valid = batch[part_valid_key].bool()
        part_gt = batch[part_gt_key].bool()

        B = int(cat_ids.shape[0])
        for b in range(B):
            cat = int(cat_ids[b].item())
            obj_text_bank[cat].append(obj_text[b].detach().cpu())

            obj_idx = torch.nonzero(obj_mask[b], as_tuple=False).squeeze(1)
            if obj_idx.numel() > 0:
                obj_visual_bank[cat].append(patch_tokens[b, obj_idx].mean(dim=0).detach().cpu())

            K = int(part_ids.shape[1])
            for k in range(K):
                if not bool(part_valid[b, k]):
                    continue
                pid = int(part_ids[b, k].item())
                if pid < 0 or pid >= int(args.num_parts):
                    continue

                part_text_bank[(cat, pid)].append(part_text[b, k].detach().cpu())

                mask_idx = torch.nonzero(part_gt[b, k], as_tuple=False).squeeze(1)
                if mask_idx.numel() > 0:
                    part_visual_bank[(cat, pid)].append(patch_tokens[b, mask_idx].mean(dim=0).detach().cpu())

    return {
        "obj_text": {cat: mean_feat(xs) for cat, xs in obj_text_bank.items() if len(xs) > 0},
        "obj_visual": {cat: mean_feat(xs) for cat, xs in obj_visual_bank.items() if len(xs) > 0},
        "part_text": {key: mean_feat(xs) for key, xs in part_text_bank.items() if len(xs) > 0},
        "part_visual": {key: mean_feat(xs) for key, xs in part_visual_bank.items() if len(xs) > 0},
    }


def compute_graph_metrics(text_feats: torch.Tensor, visual_feats: torch.Tensor) -> Dict[str, float]:
    C_t = pairwise_cosine_sim(text_feats)
    C_v = pairwise_cosine_sim(visual_feats)
    vt = upper_tri_no_diag(C_t)
    vv = upper_tri_no_diag(C_v)
    return {
        "spearman": spearman_rho(vt, vv),
        "pearson": pearson_corr(vt, vv),
        "mse": mse_np(vt, vv),
        "num_pairs": int(len(vt)),
    }


def compute_obj_part_relation_metrics(
    obj_text: torch.Tensor,
    obj_visual: torch.Tensor,
    part_text: torch.Tensor,
    part_visual: torch.Tensor,
) -> Dict[str, float]:
    obj_text = safe_normalize(obj_text.float(), dim=-1)
    obj_visual = safe_normalize(obj_visual.float(), dim=-1)
    part_text = safe_normalize(part_text.float(), dim=-1)
    part_visual = safe_normalize(part_visual.float(), dim=-1)
    raw_scores = (part_text @ obj_text).detach().cpu().numpy().astype(np.float64)
    visual_scores = (part_visual @ obj_visual).detach().cpu().numpy().astype(np.float64)
    return {
        "spearman": spearman_rho(raw_scores, visual_scores),
        "pearson": pearson_corr(raw_scores, visual_scores),
        "mse": mse_np(raw_scores, visual_scores),
        "num_pairs": int(len(raw_scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--dataset", required=True)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_parts", type=int, default=116)
    parser.add_argument("--save_dir", default="audits/rawT_vs_gtmaskV_spearman")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.model_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("train", {})
    if args.batch_size is None:
        args.batch_size = int(train_cfg.get("batch_size", 128))

    set_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("[device]", device)

    cat_names, part_names = collect_names_from_raw_pth(args.dataset)
    protos = collect_raw_text_and_gtmask_visual_prototypes(args, cfg, device)

    obj_text = protos["obj_text"]
    obj_visual = protos["obj_visual"]
    part_text = protos["part_text"]
    part_visual = protos["part_visual"]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    common_obj_cats = sorted(set(obj_text.keys()) & set(obj_visual.keys()))
    obj_T = torch.stack([obj_text[c] for c in common_obj_cats], dim=0)
    obj_V = torch.stack([obj_visual[c] for c in common_obj_cats], dim=0)
    obj_graph = compute_graph_metrics(obj_T, obj_V)

    rows = []
    weighted_part_spear = 0.0
    weighted_part_pearson = 0.0
    weighted_part_mse = 0.0
    weighted_part_pairs = 0

    weighted_objrel_spear = 0.0
    weighted_objrel_pearson = 0.0
    weighted_objrel_mse = 0.0
    weighted_objrel_pairs = 0

    cats_for_parts = sorted({cat for cat, _ in part_text.keys()} & {cat for cat, _ in part_visual.keys()})
    for cat in cats_for_parts:
        pids = sorted(
            {pid for c, pid in part_text.keys() if c == cat}
            & {pid for c, pid in part_visual.keys() if c == cat}
        )
        if len(pids) < 2:
            continue

        T = torch.stack([part_text[(cat, pid)] for pid in pids], dim=0)
        V = torch.stack([part_visual[(cat, pid)] for pid in pids], dim=0)

        if len(pids) >= 3:
            part_graph = compute_graph_metrics(T, V)
        else:
            part_graph = {"spearman": float("nan"), "pearson": float("nan"), "mse": float("nan"), "num_pairs": 1}

        if cat in obj_text and cat in obj_visual:
            objrel = compute_obj_part_relation_metrics(obj_text[cat], obj_visual[cat], T, V)
        else:
            objrel = {"spearman": float("nan"), "pearson": float("nan"), "mse": float("nan"), "num_pairs": len(pids)}

        rows.append({
            "category_id": cat,
            "class_name": cat_names.get(cat, str(cat)),
            "num_parts": len(pids),
            "part_ids": " ".join(str(x) for x in pids),
            "part_names": " | ".join(part_names.get((cat, pid), f"part_{pid}") for pid in pids),
            "part_graph_spearman": part_graph["spearman"],
            "part_graph_pearson": part_graph["pearson"],
            "part_graph_mse": part_graph["mse"],
            "part_graph_num_pairs": part_graph["num_pairs"],
            "obj_part_relation_spearman": objrel["spearman"],
            "obj_part_relation_pearson": objrel["pearson"],
            "obj_part_relation_mse": objrel["mse"],
            "obj_part_relation_num_pairs": objrel["num_pairs"],
        })

        if not np.isnan(part_graph["spearman"]):
            n = int(part_graph["num_pairs"])
            weighted_part_spear += part_graph["spearman"] * n
            weighted_part_pearson += part_graph["pearson"] * n
            weighted_part_mse += part_graph["mse"] * n
            weighted_part_pairs += n

        if not np.isnan(objrel["spearman"]):
            n = int(objrel["num_pairs"])
            weighted_objrel_spear += objrel["spearman"] * n
            weighted_objrel_pearson += objrel["pearson"] * n
            weighted_objrel_mse += objrel["mse"] * n
            weighted_objrel_pairs += n

    part_summary = {
        "weighted_part_graph_spearman": weighted_part_spear / max(weighted_part_pairs, 1),
        "weighted_part_graph_pearson": weighted_part_pearson / max(weighted_part_pairs, 1),
        "weighted_part_graph_mse": weighted_part_mse / max(weighted_part_pairs, 1),
        "weighted_part_graph_pairs": weighted_part_pairs,
        "weighted_obj_part_relation_spearman": weighted_objrel_spear / max(weighted_objrel_pairs, 1),
        "weighted_obj_part_relation_pearson": weighted_objrel_pearson / max(weighted_objrel_pairs, 1),
        "weighted_obj_part_relation_mse": weighted_objrel_mse / max(weighted_objrel_pairs, 1),
        "weighted_obj_part_relation_pairs": weighted_objrel_pairs,
    }

    csv_path = save_dir / "rawT_vs_gtmaskV_part_level_by_category.csv"
    fieldnames = [
        "category_id",
        "class_name",
        "num_parts",
        "part_ids",
        "part_names",
        "part_graph_spearman",
        "part_graph_pearson",
        "part_graph_mse",
        "part_graph_num_pairs",
        "obj_part_relation_spearman",
        "obj_part_relation_pearson",
        "obj_part_relation_mse",
        "obj_part_relation_num_pairs",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": args.dataset,
        "model_config": args.model_config,
        "object_level": {
            "num_object_classes": len(common_obj_cats),
            "category_ids": common_obj_cats,
            "spearman": obj_graph["spearman"],
            "pearson": obj_graph["pearson"],
            "mse": obj_graph["mse"],
            "num_pairs": obj_graph["num_pairs"],
        },
        "part_level": part_summary,
        "num_part_categories": len(rows),
        "csv": str(csv_path),
    }
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 120)
    print("[OBJECT LEVEL] raw object T graph vs GT-mask object V graph")
    print(f"classes={len(common_obj_cats)} pairs={obj_graph['num_pairs']}")
    print(f"spearman={obj_graph['spearman']:.8f} pearson={obj_graph['pearson']:.8f} mse={obj_graph['mse']:.8e}")
    print("-" * 120)
    print("[PART LEVEL] raw part T graph vs GT-mask part V graph")
    print(f"categories={len(rows)} pairs={part_summary['weighted_part_graph_pairs']}")
    print(
        f"weighted spearman={part_summary['weighted_part_graph_spearman']:.8f} "
        f"pearson={part_summary['weighted_part_graph_pearson']:.8f} "
        f"mse={part_summary['weighted_part_graph_mse']:.8e}"
    )
    print("-" * 120)
    print("[OBJ-PART RELATION] raw obj-to-part relation vs GT-mask obj-to-part relation")
    print(f"pairs={part_summary['weighted_obj_part_relation_pairs']}")
    print(
        f"weighted spearman={part_summary['weighted_obj_part_relation_spearman']:.8f} "
        f"pearson={part_summary['weighted_obj_part_relation_pearson']:.8f} "
        f"mse={part_summary['weighted_obj_part_relation_mse']:.8e}"
    )
    print("-" * 120)
    print(f"[saved csv] {csv_path}")
    print(f"[saved summary] {summary_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
