#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check whether a randomly initialized near-orthogonal Linear(512 -> 768)
preserves the internal structure of REAL text part features T.

Important:
  - T is NOT random.
  - T is read from the meta pth: data["annotations"][*]["part_ann_feats"].
  - The category-level T block is built in the same spirit as src/dataset_joint.py:
      category -> part_id -> average feature
    then one block per object category.

Projector:
  z = W t
  W: [768, 512], columns are orthonormal, so W^T W ~= I_512.

Expected:
  Pairwise cosine similarity before/after projection should be almost identical.
  Spearman ~= 1, Pearson ~= 1, MSE ~= 0.
"""

import argparse
import csv
import math
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn


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


def upper_tri_no_diag(x: torch.Tensor) -> torch.Tensor:
    k = x.shape[0]
    idx = torch.triu_indices(k, k, offset=1, device=x.device)
    return x[idx[0], idx[1]]


def rankdata_numpy(a: np.ndarray) -> np.ndarray:
    """scipy-free rankdata with average ranks for ties."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)

    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def spearman_numpy(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra = rankdata_numpy(a)
    rb = rankdata_numpy(b)

    ra = ra - ra.mean()
    rb = rb - rb.mean()

    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def pearson_numpy(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a = a - a.mean()
    b = b - b.mean()

    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


class OrthogonalLinearProjector(nn.Module):
    """
    Linear in_dim -> out_dim with W^T W ~= I_in_dim.
    For 512->768, W is [768,512], so columns can be orthonormal.
    """
    def __init__(self, in_dim: int = 512, out_dim: int = 768):
        super().__init__()
        if out_dim < in_dim:
            raise ValueError(f"Need out_dim >= in_dim for W^T W=I, got {out_dim} < {in_dim}")
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        out_dim, in_dim = self.proj.weight.shape
        q, _ = torch.linalg.qr(torch.randn(out_dim, in_dim), mode="reduced")
        self.proj.weight.copy_(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


def orthogonality_report(projector: OrthogonalLinearProjector) -> Dict[str, float]:
    W = projector.proj.weight.detach().float()       # [768,512]
    I = torch.eye(W.shape[1], device=W.device)
    gram = W.T @ W
    diff = gram - I
    return {
        "WtW_mse": float(diff.pow(2).mean().item()),
        "WtW_max_abs": float(diff.abs().max().item()),
    }


def clean_name(x: Any) -> str:
    return str(x).replace("\n", " ").strip()


def to_tensor(x, dtype=None) -> torch.Tensor:
    if torch.is_tensor(x):
        out = x
    else:
        out = torch.tensor(x)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def build_category_part_blocks_from_raw_pth(
    pth_path: str,
    part_text_name: str = "part_ann_feats",
    part_id_name: str = "part_category_id",
    category_key: str = "category_id",
) -> List[Dict[str, Any]]:
    """
    Match the important part of src/dataset_joint.py:
      data = torch.load(pth)
      annotations = data["annotations"]
      category -> part_id -> list[text feature]
      average features per part_id
      one text block per category
    """
    obj = torch.load(pth_path, map_location="cpu")

    if not isinstance(obj, dict) or "annotations" not in obj:
        raise RuntimeError(
            "Expected pth structure with top-level key 'annotations'. "
            f"Got type={type(obj)}, keys={list(obj.keys())[:20] if isinstance(obj, dict) else None}"
        )

    annotations = obj["annotations"]
    print(f"[raw pth] annotations={len(annotations)}")
    if len(annotations) == 0:
        raise RuntimeError("No annotations found in pth.")

    print("[sample annotation keys]", sorted(list(annotations[0].keys())))

    # category -> part_id -> list[feat]
    feat_bank = defaultdict(lambda: defaultdict(list))
    name_bank = defaultdict(dict)
    class_name_bank = {}

    missing = 0
    bad_shape = 0

    for ann in annotations:
        if category_key not in ann or part_id_name not in ann or part_text_name not in ann:
            missing += 1
            continue

        cat = int(ann[category_key])
        class_name_bank.setdefault(cat, ann.get("class_name", ""))

        part_ids = ann.get(part_id_name, [])
        part_feats = ann.get(part_text_name, None)
        part_names = ann.get("part_class_name", [])

        if part_feats is None or len(part_ids) == 0:
            missing += 1
            continue

        part_feats = to_tensor(part_feats, dtype=torch.float32)
        part_ids = to_tensor(part_ids, dtype=torch.long).view(-1)

        if part_feats.ndim != 2 or part_feats.shape[0] < part_ids.numel():
            bad_shape += 1
            continue

        for j, pid in enumerate(part_ids.tolist()):
            pid = int(pid)
            feat_bank[cat][pid].append(part_feats[j].float())
            if isinstance(part_names, (list, tuple)) and j < len(part_names):
                name_bank[cat][pid] = clean_name(part_names[j])

    print(f"[skip] missing={missing}, bad_shape={bad_shape}")
    print(f"[categories with part text] {len(feat_bank)}")

    blocks = []
    for cat in sorted(feat_bank.keys()):
        pids = sorted(feat_bank[cat].keys())
        if len(pids) < 3:
            continue

        feats = []
        names = []
        for pid in pids:
            feats.append(torch.stack(feat_bank[cat][pid], dim=0).mean(dim=0))
            names.append(name_bank.get(cat, {}).get(pid, f"part_{pid}"))

        T = torch.stack(feats, dim=0)
        blocks.append({
            "category_id": int(cat),
            "class_name": clean_name(class_name_bank.get(cat, "")),
            "part_ids": torch.tensor(pids, dtype=torch.long),
            "part_names": names,
            "T": T,
        })

    return blocks


@torch.no_grad()
def evaluate_block(block: Dict[str, Any], projector: OrthogonalLinearProjector, device: torch.device) -> Dict[str, Any]:
    T = block["T"].to(device).float()
    Z = projector(T)

    C_t = pairwise_cosine_sim(T)
    C_z = pairwise_cosine_sim(Z)

    vt = upper_tri_no_diag(C_t).detach().cpu().numpy()
    vz = upper_tri_no_diag(C_z).detach().cpu().numpy()

    return {
        "category_id": int(block["category_id"]),
        "class_name": block.get("class_name", ""),
        "num_parts": int(T.shape[0]),
        "num_pairs": int(len(vt)),
        "spearman": spearman_numpy(vt, vz),
        "pearson": pearson_numpy(vt, vz),
        "mse": float(torch.mean((C_t - C_z).pow(2)).item()),
        "max_abs_diff": float(torch.max(torch.abs(C_t - C_z)).item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--part_id_name", default="part_category_id")
    parser.add_argument("--category_key", default="category_id")
    parser.add_argument("--in_dim", type=int, default=512)
    parser.add_argument("--out_dim", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_blocks", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--out_csv", default="audits/random_orthogonal_projector_t_spearman.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("[device]", device)

    blocks = build_category_part_blocks_from_raw_pth(
        pth_path=args.dataset,
        part_text_name=args.part_text_name,
        part_id_name=args.part_id_name,
        category_key=args.category_key,
    )
    if args.max_blocks and args.max_blocks > 0:
        blocks = blocks[:args.max_blocks]

    print(f"[blocks] {len(blocks)}")
    if len(blocks) == 0:
        raise RuntimeError("No valid category-level T blocks loaded. Check pth keys printed above.")

    for b in blocks[:5]:
        print(
            f"  cat={b['category_id']} class={b['class_name']} "
            f"K={b['T'].shape[0]} T_shape={tuple(b['T'].shape)}"
        )

    projector = OrthogonalLinearProjector(args.in_dim, args.out_dim).to(device)
    projector.eval()

    ortho = orthogonality_report(projector)
    print(f"[projector] Linear({args.in_dim}->{args.out_dim}), bias=False")
    print(f"[orthogonality] W^T W mse={ortho['WtW_mse']:.6e}, max_abs={ortho['WtW_max_abs']:.6e}")

    rows = []
    total_pairs = 0
    weighted_spear = 0.0
    weighted_pearson = 0.0
    weighted_mse = 0.0

    for block in blocks:
        if block["T"].shape[-1] != args.in_dim:
            raise ValueError(
                f"T dim mismatch: expected {args.in_dim}, got {block['T'].shape[-1]}. "
                f"Check --part_text_name or --in_dim."
            )

        row = evaluate_block(block, projector, device)
        rows.append(row)

        n = row["num_pairs"]
        if n > 0 and not math.isnan(row["spearman"]):
            total_pairs += n
            weighted_spear += row["spearman"] * n
            weighted_pearson += row["pearson"] * n
            weighted_mse += row["mse"] * n

    avg_spear = weighted_spear / max(total_pairs, 1)
    avg_pearson = weighted_pearson / max(total_pairs, 1)
    avg_mse = weighted_mse / max(total_pairs, 1)

    print("=" * 100)
    print(f"[overall weighted by pairs] spearman={avg_spear:.8f} pearson={avg_pearson:.8f} mse={avg_mse:.8e}")
    print("=" * 100)

    for row in rows[:20]:
        print(
            f"cat={row['category_id']:>3} class={row['class_name']:<15} "
            f"K={row['num_parts']:<3} spear={row['spearman']:.8f} "
            f"pearson={row['pearson']:.8f} mse={row['mse']:.3e} maxdiff={row['max_abs_diff']:.3e}"
        )
    if len(rows) > 20:
        print(f"... ({len(rows) - 20} more blocks saved to CSV)")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "category_id",
        "class_name",
        "num_parts",
        "num_pairs",
        "spearman",
        "pearson",
        "mse",
        "max_abs_diff",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
