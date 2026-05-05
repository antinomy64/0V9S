import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from pathlib import Path
from PIL import Image


def resolve_path(path_str: str, path_prefix: str | None = None) -> str:
    if os.path.exists(path_str):
        return path_str
    if path_prefix is not None:
        candidate = os.path.join(path_prefix, path_str)
        if os.path.exists(candidate):
            return candidate
    return path_str


def read_mask(seg_path: str, path_prefix: str | None = None) -> np.ndarray:
    seg_path = resolve_path(seg_path, path_prefix)
    mask = np.array(Image.open(seg_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Check object area ratio in original image using object segmentation mask."
    )
    parser.add_argument("--ann_path", type=str, required=True, help="Path to .pth dataset")
    parser.add_argument("--out_csv", type=str, required=True, help="Per-annotation csv output")
    parser.add_argument("--out_json", type=str, default=None, help="Summary json output")
    parser.add_argument("--path_prefix", type=str, default=None, help="Optional prefix for relative mask paths")
    parser.add_argument(
        "--with_background",
        action="store_true",
        default=False,
        help="If object mask uses 0 as background and class ids are shifted by +1 in mask, enable this.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold for summary, e.g. 0.01 means 1%% of full image area.",
    )
    parser.add_argument(
        "--sort_ascending",
        action="store_true",
        default=False,
        help="Sort csv by ratio ascending instead of descending.",
    )
    args = parser.parse_args()

    data = torch.load(args.ann_path, map_location="cpu")
    images = {imm["id"]: imm for imm in data["images"]}

    rows = []
    class_ratios = defaultdict(list)
    missing_masks = 0

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in images:
            continue
        imm = images[image_id]
        seg_path = imm.get("seg_file_name", None)
        if seg_path is None:
            missing_masks += 1
            continue

        seg_path_resolved = resolve_path(seg_path, args.path_prefix)
        if not os.path.exists(seg_path_resolved):
            missing_masks += 1
            continue

        mask = read_mask(seg_path, args.path_prefix)
        h, w = mask.shape[:2]
        total_pixels = int(h * w)

        category_id = int(ann["category_id"])
        mask_value = category_id + 1 if args.with_background else category_id
        obj_pixels = int((mask == mask_value).sum())
        area_ratio = float(obj_pixels / max(total_pixels, 1))

        row = {
            "annotation_id": int(ann["id"]),
            "image_id": int(image_id),
            "class_name": ann.get("class_name", ""),
            "category_id": category_id,
            "seg_file_name": seg_path,
            "height": int(h),
            "width": int(w),
            "total_pixels": total_pixels,
            "obj_pixels": obj_pixels,
            "obj_area_ratio": area_ratio,
        }
        rows.append(row)
        class_ratios[row["class_name"]].append(area_ratio)

    rows = sorted(rows, key=lambda x: x["obj_area_ratio"], reverse=not args.sort_ascending)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else [
                "annotation_id", "image_id", "class_name", "category_id", "seg_file_name",
                "height", "width", "total_pixels", "obj_pixels", "obj_area_ratio"
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "ann_path": args.ann_path,
        "num_annotations": len(rows),
        "missing_masks": int(missing_masks),
        "with_background": bool(args.with_background),
    }

    if rows:
        all_ratios = np.array([r["obj_area_ratio"] for r in rows], dtype=np.float64)
        summary["global"] = {
            "mean": float(all_ratios.mean()),
            "median": float(np.median(all_ratios)),
            "min": float(all_ratios.min()),
            "max": float(all_ratios.max()),
            "q01": float(np.quantile(all_ratios, 0.01)),
            "q05": float(np.quantile(all_ratios, 0.05)),
            "q10": float(np.quantile(all_ratios, 0.10)),
            "q25": float(np.quantile(all_ratios, 0.25)),
            "q75": float(np.quantile(all_ratios, 0.75)),
            "q90": float(np.quantile(all_ratios, 0.90)),
            "q95": float(np.quantile(all_ratios, 0.95)),
            "q99": float(np.quantile(all_ratios, 0.99)),
        }
        if args.threshold is not None:
            summary["global"]["threshold"] = float(args.threshold)
            summary["global"]["num_below_threshold"] = int((all_ratios < args.threshold).sum())
            summary["global"]["ratio_below_threshold"] = float((all_ratios < args.threshold).mean())

    per_class = {}
    for cls_name, ratios in sorted(class_ratios.items(), key=lambda kv: kv[0]):
        arr = np.array(ratios, dtype=np.float64)
        cls_summary = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "q10": float(np.quantile(arr, 0.10)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
            "q90": float(np.quantile(arr, 0.90)),
        }
        if args.threshold is not None:
            cls_summary["num_below_threshold"] = int((arr < args.threshold).sum())
            cls_summary["ratio_below_threshold"] = float((arr < args.threshold).mean())
        per_class[cls_name] = cls_summary
    summary["per_class"] = per_class

    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved per-annotation csv to {out_csv}")
    if args.out_json is not None:
        print(f"Saved summary json to {args.out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
