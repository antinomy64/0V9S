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
        description=(
            "Print and save fine-grained part average pixel count and "
            "occurrence frequency per object crop."
        )
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
    parser.add_argument("--num_parts", type=int, default=116)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--min_obj_area_ratio", type=float, default=0.0)

    parser.add_argument("--max_obj_slots", type=int, default=256)
    parser.add_argument("--out_csv", default="audits/fine_part_area_frequency.csv")
    parser.add_argument("--out_md", default="audits/fine_part_area_frequency.md")

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

    avg_pixels = analyser.compute_part_avg_pixel_count_per_obj_crop(
        max_obj_slots=args.max_obj_slots
    )
    part_freq = analyser.compute_part_occurrence_freq_per_obj_crop(
        max_obj_slots=args.max_obj_slots
    )

    part_names = analyser.part_names
    assert len(part_names) == args.num_parts
    assert avg_pixels.numel() == args.num_parts
    assert part_freq.numel() == args.num_parts

    rows = []
    for pid, part_name in enumerate(part_names):
        freq = float(part_freq[pid].item())
        rows.append(
            {
                "part_id": int(pid),
                "part_name": str(part_name),
                "avg_pixel_count_per_obj_crop": float(avg_pixels[pid].item()),
                "part_occurrence_freq_per_obj_crop": freq,
                "part_occurrence_percent_per_obj_crop": freq * 100.0,
            }
        )

    header = (
        "| part_id | part_name | avg_pixel_count_per_obj_crop | "
        "part_occurrence_freq_per_obj_crop | part_occurrence_percent_per_obj_crop |"
    )
    sep = "|---:|---|---:|---:|---:|"

    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['part_id']} | {r['part_name']} | "
            f"{r['avg_pixel_count_per_obj_crop']:.6f} | "
            f"{r['part_occurrence_freq_per_obj_crop']:.6f} | "
            f"{r['part_occurrence_percent_per_obj_crop']:.2f} |"
        )

    text = "\n".join(lines)
    print(text)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "part_id",
                    "part_name",
                    "avg_pixel_count_per_obj_crop",
                    "part_occurrence_freq_per_obj_crop",
                    "part_occurrence_percent_per_obj_crop",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved csv] {out_csv}")

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(text + "\n", encoding="utf-8")
        print(f"[saved md] {out_md}")


if __name__ == "__main__":
    main()
