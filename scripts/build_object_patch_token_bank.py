#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis import ObjectPatchTokenBankBuilder


def parse_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_pth", required=True)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--store_dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--min_obj_area_ratio", type=float, default=0.0)
    parser.add_argument("--keep_all_crop_patches", action="store_true", default=False)

    args = parser.parse_args()

    builder = ObjectPatchTokenBankBuilder(
        dataset_path=args.dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        store_dtype=parse_dtype(args.store_dtype),
        min_obj_area_ratio=args.min_obj_area_ratio,
        use_obj_mask=not bool(args.keep_all_crop_patches),
    )

    builder.build_and_save(args.out_pth)


if __name__ == "__main__":
    main()
