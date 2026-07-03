#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect one sample from Talk2DINO/VOC116 pth.

Usage:
  python analyze_scripts/inspect_voc116_pth_one.py \
    --pth feature/voc116_obj_part_test15/train_voc116_obj_with_text.pth \
    --target_class person \
    --ann_index 0

This script only reads and prints fields. It does not modify the pth.
"""

import argparse
import json
from typing import Any

import torch


KEY_PATTERNS = [
    "mask", "gt", "patch", "part", "obj", "crop",
    "box", "class", "category", "feat", "ann", "caption",
]


def short_value(x: Any, max_list: int = 10):
    if torch.is_tensor(x):
        info = {
            "type": "Tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }
        if x.numel() > 0 and x.numel() <= 20:
            info["values"] = x.detach().cpu().tolist()
        elif x.numel() > 0:
            xf = x.detach().float().cpu().reshape(-1)
            info["min"] = float(xf.min().item())
            info["max"] = float(xf.max().item())
            info["mean"] = float(xf.mean().item())
        return info

    if isinstance(x, (list, tuple)):
        out = {
            "type": type(x).__name__,
            "len": len(x),
        }
        if len(x) <= max_list:
            out["values"] = [short_value(v, max_list=max_list) for v in x]
        else:
            out["head"] = [short_value(v, max_list=max_list) for v in x[:max_list]]
        return out

    if isinstance(x, dict):
        return {
            "type": "dict",
            "keys": list(x.keys()),
        }

    return x


def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_keys_and_values(name: str, obj: dict, only_interesting: bool = False):
    print_section(name)
    keys = list(obj.keys())
    print(f"num_keys = {len(keys)}")
    print("keys =", keys)

    for k in keys:
        if only_interesting:
            kl = str(k).lower()
            if not any(p in kl for p in KEY_PATTERNS):
                continue
        print(f"\n[{k}]")
        print(json.dumps(short_value(obj[k]), indent=2, ensure_ascii=False))


def find_first_by_class(annotations, target_class: str):
    for i, ann in enumerate(annotations):
        if str(ann.get("class_name", "")).lower() == target_class.lower():
            return i, ann
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True)
    parser.add_argument("--target_class", default="person")
    parser.add_argument("--ann_index", type=int, default=0)
    parser.add_argument("--print_all_values", action="store_true",
                        help="Print all fields. Default prints all keys but detailed values only for interesting fields.")
    args = parser.parse_args()

    print(f"[Load] {args.pth}")
    data = torch.load(args.pth, map_location="cpu")

    print_section("TOP LEVEL")
    print(type(data))
    if isinstance(data, dict):
        print("top_level_keys =", list(data.keys()))
        for k, v in data.items():
            if isinstance(v, list):
                print(f"{k}: list len={len(v)}")
            elif isinstance(v, dict):
                print(f"{k}: dict keys={list(v.keys())[:20]}")
            else:
                print(f"{k}: {short_value(v)}")
    else:
        print(short_value(data))

    if not isinstance(data, dict):
        return

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    print(f"\nnum_images={len(images)}")
    print(f"num_annotations={len(annotations)}")

    if len(images) > 0:
        print_keys_and_values("FIRST IMAGE", images[0], only_interesting=not args.print_all_values)

    if len(annotations) == 0:
        return

    idx = max(0, min(args.ann_index, len(annotations) - 1))
    print_keys_and_values(
        f"ANNOTATION BY INDEX ann_index={idx}",
        annotations[idx],
        only_interesting=not args.print_all_values,
    )

    cls_idx, cls_ann = find_first_by_class(annotations, args.target_class)
    if cls_ann is not None:
        print_keys_and_values(
            f"FIRST ANNOTATION WITH class_name={args.target_class}, index={cls_idx}",
            cls_ann,
            only_interesting=not args.print_all_values,
        )
    else:
        print_section(f"FIRST ANNOTATION WITH class_name={args.target_class}")
        print("not found")

    # Field coverage summary.
    print_section("FIELD COVERAGE SUMMARY")
    all_keys = {}
    interesting_keys = {}
    class_counts = {}
    for ann in annotations:
        cls = str(ann.get("class_name", ""))
        class_counts[cls] = class_counts.get(cls, 0) + 1
        for k in ann.keys():
            all_keys[k] = all_keys.get(k, 0) + 1
            kl = str(k).lower()
            if any(p in kl for p in KEY_PATTERNS):
                interesting_keys[k] = interesting_keys.get(k, 0) + 1

    print("all annotation keys with counts:")
    for k, c in sorted(all_keys.items(), key=lambda kv: str(kv[0])):
        print(f"  {k}: {c}/{len(annotations)}")

    print("\ninteresting annotation keys with counts:")
    for k, c in sorted(interesting_keys.items(), key=lambda kv: str(kv[0])):
        print(f"  {k}: {c}/{len(annotations)}")

    print("\nclass counts head:")
    for cls, c in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:30]:
        print(f"  {cls}: {c}")


if __name__ == "__main__":
    main()
