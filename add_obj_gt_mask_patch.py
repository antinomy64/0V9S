#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add object-level patch masks to test15 pth.

Input pth must contain:
  images[*]["id"], images[*]["file_name"] or ["seg_file_name"]
  annotations[*]["image_id"], ["category_id"], ["cropaug_box_xyxy"], ["cropaug_patch_tokens"]

Output adds:
  ann["obj_gt_mask_patch"] = BoolTensor [Gh, Gw]

Default assumes object PNG ids are 0-based category_id.
If your object PNG uses 0 as background and object ids are category_id + 1, pass:
  --with_background_obj
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def load_mask_png(path: str) -> torch.Tensor:
    arr = torch.as_tensor(__import__("numpy").array(Image.open(path)), dtype=torch.long)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def resolve_mask_path(img: Dict[str, Any], obj_mask_dir: str) -> str:
    # Prefer seg_file_name if present; otherwise use image file stem.
    if "seg_file_name" in img and img["seg_file_name"]:
        name = os.path.basename(str(img["seg_file_name"]))
    else:
        name = Path(str(img["file_name"])).with_suffix(".png").name
    return os.path.join(obj_mask_dir, name)


def crop_resize_pool(mask_hw: torch.Tensor, box_xyxy: torch.Tensor, crop_dim: int, patch_size: int) -> torch.Tensor:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy.tolist()]
    H, W = mask_hw.shape
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop box after clamp: {(x1, y1, x2, y2)} for mask shape {(H, W)}")

    crop = mask_hw[y1:y2, x1:x2].float()[None, None]  # [1,1,h,w]
    crop = F.interpolate(crop, size=(crop_dim, crop_dim), mode="nearest")[0, 0].bool()

    if crop_dim % patch_size != 0:
        raise ValueError(f"crop_dim={crop_dim} must be divisible by patch_size={patch_size}")
    gh = crop_dim // patch_size
    gw = crop_dim // patch_size

    pooled = crop.float().reshape(gh, patch_size, gw, patch_size).amax(dim=(1, 3)).bool()
    return pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pth", required=True)
    parser.add_argument("--output_pth", required=True)
    parser.add_argument("--obj_mask_dir", required=True)
    parser.add_argument("--obj_mask_field", default="obj_gt_mask_patch")
    parser.add_argument("--crop_box_field", default="cropaug_box_xyxy")
    parser.add_argument("--patch_field", default="cropaug_patch_tokens")
    parser.add_argument("--category_field", default="category_id")
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background_obj", action="store_true",
                        help="Use mask id category_id + 1 instead of category_id.")
    args = parser.parse_args()

    data = torch.load(args.input_pth, map_location="cpu")
    if "images" not in data or "annotations" not in data:
        raise KeyError("input pth must contain top-level keys images and annotations")

    images_by_id = {img["id"]: img for img in data["images"]}

    mask_cache: Dict[str, torch.Tensor] = {}
    added = 0
    skipped = 0

    for ann in tqdm(data["annotations"], desc="add obj_gt_mask_patch"):
        try:
            image_id = ann["image_id"]
            img = images_by_id[image_id]

            mask_path = resolve_mask_path(img, args.obj_mask_dir)
            if mask_path not in mask_cache:
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(mask_path)
                mask_cache[mask_path] = load_mask_png(mask_path)

            obj_id = int(ann[args.category_field])
            if args.with_background_obj:
                obj_id += 1

            obj_mask_full = mask_cache[mask_path].eq(obj_id)
            box = torch.as_tensor(ann[args.crop_box_field], dtype=torch.long)
            obj_patch = crop_resize_pool(obj_mask_full, box, args.crop_dim, args.patch_size)

            patches = ann[args.patch_field]
            if not torch.is_tensor(patches):
                patches = torch.as_tensor(patches)
            n_patch = patches.reshape(-1, patches.shape[-1]).shape[0]
            if obj_patch.numel() != n_patch:
                raise ValueError(
                    f"ann id={ann.get('id','unknown')}: obj_patch.numel={obj_patch.numel()} "
                    f"!= patch tokens N={n_patch}"
                )

            ann[args.obj_mask_field] = obj_patch.cpu()
            added += 1
        except Exception as e:
            skipped += 1
            raise RuntimeError(f"Failed at ann id={ann.get('id','unknown')}: {e}") from e

    os.makedirs(os.path.dirname(args.output_pth), exist_ok=True)
    torch.save(data, args.output_pth)
    print(f"Saved: {args.output_pth}")
    print(f"added={added}, skipped={skipped}, obj_mask_field={args.obj_mask_field}")


if __name__ == "__main__":
    main()
