#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project existing fine PascalPart116 feature .pth into coarse PascalPart taxonomy.

This script does NOT recompute DINO features. It keeps:
  - images[*].avg_self_attn_out
  - annotations[*].cropaug_patch_tokens
  - annotations[*].cropaug_box_xyxy
  - annotations[*].ann_feats

It rewrites only coarse part metadata:
  - annotations[*].part_category_id
  - annotations[*].part_class_name
  - annotations[*].part_caption
  - removes annotations[*].part_ann_feats by default

Then run text_features_extraction_mean_caption.py again to regenerate coarse part_ann_feats.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.voc116_part_coarse import (
    COARSE_PART_CLASSES,
    COARSE_PART_GROUPS,
    COARSE_PART_NAME_TO_ID,
    coarse_parts_for_object,
    build_prompts,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument(
        "--keep_part_ann_feats",
        action="store_true",
        default=False,
        help="Not recommended. By default part_ann_feats is removed and should be regenerated for coarse part captions.",
    )
    args = parser.parse_args()

    data = torch.load(args.in_path, map_location="cpu")
    assert isinstance(data, dict) and "images" in data and "annotations" in data

    converted = 0
    empty = 0
    for ann in data["annotations"]:
        obj_name = ann.get("class_name", "")
        coarse_names = coarse_parts_for_object(obj_name)
        if len(coarse_names) == 0:
            empty += 1
        coarse_ids = [COARSE_PART_NAME_TO_ID[name] for name in coarse_names]

        ann["part_category_id"] = coarse_ids
        ann["part_class_name"] = coarse_names
        ann["part_caption"] = [build_prompts(name) for name in coarse_names]

        if not args.keep_part_ann_feats:
            ann.pop("part_ann_feats", None)

        converted += 1

    data["part_taxonomy"] = "coarse_pascalpart116_v1"
    data["coarse_part_classes"] = list(COARSE_PART_CLASSES)
    data["coarse_part_groups"] = {k: list(v) for k, v in COARSE_PART_GROUPS.items()}
    data["num_parts"] = len(COARSE_PART_CLASSES)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)

    print(f"[saved] {out_path}")
    print(f"[converted annotations] {converted}")
    print(f"[annotations with no coarse parts] {empty}")
    print(f"[coarse parts] {len(COARSE_PART_CLASSES)}")
    print("[next] regenerate coarse part_ann_feats:")
    print(f"CUDA_VISIBLE_DEVICES=0 python text_features_extraction_mean_caption.py --ann_path {out_path} --out_path {out_path} --use_caption_ensemble")


if __name__ == "__main__":
    main()
