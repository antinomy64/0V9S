#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from anlysis import FeatureAnalyser


def collect_vision_by_part(args, model_config: str, init_weights: str):
    analyser = FeatureAnalyser(
        model_config=model_config,
        dataset=args.dataset,
        init_weights=init_weights,
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
        show_progress=args.show_progress,
    )

    if hasattr(analyser, "collect_vision_feature"):
        fake_by_part, gt_by_part = analyser.collect_vision_feature()
    else:
        fake_by_part, gt_by_part = analyser.collect()

    return analyser.part_names, fake_by_part, gt_by_part


@torch.no_grad()
def compactness_metrics(x: torch.Tensor) -> Dict[str, float]:
    if x is None or x.numel() == 0:
        return {
            "count": 0,
            "cos_to_centroid_mean": float("nan"),
            "angular_var": float("nan"),
            "pairwise_cos_mean": float("nan"),
            "trace_var": float("nan"),
            "mean_dim_var": float("nan"),
        }

    x = F.normalize(x.float(), dim=-1)
    n, d = x.shape

    centroid = x.mean(dim=0, keepdim=True)
    centroid_norm = F.normalize(centroid, dim=-1)

    cos_to_centroid = (x * centroid_norm).sum(dim=-1)
    cos_to_centroid_mean = float(cos_to_centroid.mean().item())
    angular_var = float((1.0 - cos_to_centroid).mean().item())

    centered = x - centroid
    trace_var = float(centered.square().sum(dim=-1).mean().item())
    mean_dim_var = trace_var / float(d)

    if n > 1:
        s = x.sum(dim=0)
        pairwise_cos_mean = float(((s.dot(s) - n) / (n * (n - 1))).item())
    else:
        pairwise_cos_mean = float("nan")

    return {
        "count": int(n),
        "cos_to_centroid_mean": cos_to_centroid_mean,
        "angular_var": angular_var,
        "pairwise_cos_mean": pairwise_cos_mean,
        "trace_var": trace_var,
        "mean_dim_var": mean_dim_var,
    }


def add_prefix(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def is_number(x) -> bool:
    try:
        return x == x
    except Exception:
        return False


def fmt_float(x, ndigits: int = 4) -> str:
    if not is_number(x):
        return "nan"
    return f"{float(x):.{ndigits}f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-part intra-feature compactness/variance for fake and GT prototypes."
    )
    parser.add_argument("--dataset", required=True)

    parser.add_argument("--before_model_config", required=True)
    parser.add_argument("--before_init_weights", required=True)
    parser.add_argument("--after_model_config", required=True)
    parser.add_argument("--after_init_weights", required=True)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show_progress", action="store_true")

    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--print_gt", action="store_true")
    args = parser.parse_args()

    print("[collect before features]")
    before_part_names, before_fake, before_gt = collect_vision_by_part(
        args, args.before_model_config, args.before_init_weights
    )

    print("[collect after features]")
    after_part_names, after_fake, after_gt = collect_vision_by_part(
        args, args.after_model_config, args.after_init_weights
    )

    assert before_part_names == after_part_names, "before/after part_names mismatch"
    part_names = before_part_names

    rows = []
    for pid, part_name in enumerate(part_names):
        bf = compactness_metrics(before_fake[pid])
        af = compactness_metrics(after_fake[pid])
        gt = compactness_metrics(after_gt[pid])

        row = {"part_id": pid, "part_name": part_name}
        row.update(add_prefix("before_fake", bf))
        row.update(add_prefix("after_fake", af))
        row.update(add_prefix("gt", gt))

        if is_number(row["before_fake_angular_var"]) and is_number(row["after_fake_angular_var"]):
            row["delta_fake_angular_var"] = row["after_fake_angular_var"] - row["before_fake_angular_var"]
        else:
            row["delta_fake_angular_var"] = float("nan")

        if is_number(row["before_fake_cos_to_centroid_mean"]) and is_number(row["after_fake_cos_to_centroid_mean"]):
            row["delta_fake_cos_to_centroid"] = row["after_fake_cos_to_centroid_mean"] - row["before_fake_cos_to_centroid_mean"]
        else:
            row["delta_fake_cos_to_centroid"] = float("nan")

        rows.append(row)

    if args.print_gt:
        print("| part_id | part_name | before_n | before_cos_mean | before_ang_var | after_n | after_cos_mean | after_ang_var | delta_ang_var | gt_n | gt_cos_mean | gt_ang_var |")
        print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            print(
                f"| {r['part_id']} | {r['part_name']} | "
                f"{int(r['before_fake_count'])} | {fmt_float(r['before_fake_cos_to_centroid_mean'])} | {fmt_float(r['before_fake_angular_var'])} | "
                f"{int(r['after_fake_count'])} | {fmt_float(r['after_fake_cos_to_centroid_mean'])} | {fmt_float(r['after_fake_angular_var'])} | "
                f"{fmt_float(r['delta_fake_angular_var'])} | "
                f"{int(r['gt_count'])} | {fmt_float(r['gt_cos_to_centroid_mean'])} | {fmt_float(r['gt_angular_var'])} |"
            )
    else:
        print("| part_id | part_name | before_n | before_cos_mean | before_ang_var | after_n | after_cos_mean | after_ang_var | delta_ang_var |")
        print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            print(
                f"| {r['part_id']} | {r['part_name']} | "
                f"{int(r['before_fake_count'])} | {fmt_float(r['before_fake_cos_to_centroid_mean'])} | {fmt_float(r['before_fake_angular_var'])} | "
                f"{int(r['after_fake_count'])} | {fmt_float(r['after_fake_cos_to_centroid_mean'])} | {fmt_float(r['after_fake_angular_var'])} | "
                f"{fmt_float(r['delta_fake_angular_var'])} |"
            )

    print("")
    print("[metric meaning]")
    print("cos_mean higher = more compact/stable")
    print("ang_var = 1 - cos_mean, lower = more compact/stable")
    print("delta_ang_var < 0 means after fake features become more compact")

    if args.out_csv is not None:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
