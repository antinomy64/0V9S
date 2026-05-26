#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from anlysis import DatasetAnalyser


def main():
    parser = argparse.ArgumentParser(
        description="Print part occurrence frequency per object crop for each part."
    )

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--min_obj_area_ratio", type=float, default=0.0)

    parser.add_argument("--max_obj_slots", type=int, default=256)
    parser.add_argument("--out_csv", default=None)

    args = parser.parse_args()

    analyser = DatasetAnalyser(
        dataset=args.dataset,
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
        min_obj_area_ratio=args.min_obj_area_ratio,
    )

    part_freq = analyser.compute_part_occurrence_freq_per_obj_crop(
        max_obj_slots=args.max_obj_slots
    )

    part_names = analyser.part_names
    assert len(part_names) == args.num_parts
    assert part_freq.numel() == args.num_parts

    rows = []
    for pid, part_name in enumerate(part_names):
        value = float(part_freq[pid].item())
        rows.append({
            "part_id": pid,
            "part_name": part_name,
            "part_occurrence_freq_per_obj_crop": value,
            "part_occurrence_percent_per_obj_crop": value * 100.0,
        })

    print("| part_id | part_name | part_occurrence_freq_per_obj_crop | part_occurrence_percent_per_obj_crop |")
    print("|---:|---|---:|---:|")
    for r in rows:
        print(
            f"| {r['part_id']} | {r['part_name']} | "
            f"{r['part_occurrence_freq_per_obj_crop']:.6f} | "
            f"{r['part_occurrence_percent_per_obj_crop']:.2f} |"
        )

    if args.out_csv is not None:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "part_id",
                    "part_name",
                    "part_occurrence_freq_per_obj_crop",
                    "part_occurrence_percent_per_obj_crop",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
