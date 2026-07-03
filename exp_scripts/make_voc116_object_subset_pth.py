#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create object-name filtered VOC116 joint feature .pth files.

Use case:
  Make train/val pth that contain only annotations whose class_name is person,
  so DinoClipJointDataset/DataLoader will really iterate only person samples.

This script does not change feature tensors or annotation fields. It only filters:
  - data["annotations"]
  - data["images"] to images referenced by remaining annotations
All other top-level keys are kept unchanged.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Set

import torch


def parse_names(s: str) -> Set[str]:
    names: List[str] = []
    for x in str(s).replace(",", " ").split():
        x = x.strip().lower()
        if x:
            names.append(x)
    if not names:
        raise ValueError("--object_names is empty")
    return set(names)


def ann_class_name(ann: Dict) -> str:
    # In VOC116 test15 annotations this should be class_name, e.g. "person".
    # Do not guess category_id mapping here; fail loudly if class_name is absent.
    if "class_name" not in ann:
        raise KeyError(
            "Annotation has no 'class_name'. This subset script intentionally does not guess category_id mapping."
        )
    return str(ann["class_name"]).strip().lower()


def make_subset(in_pth: str, out_pth: str, object_names: Set[str]) -> None:
    data = torch.load(in_pth, map_location="cpu")
    if not isinstance(data, dict) or "annotations" not in data or "images" not in data:
        raise ValueError("Expected pth dict with top-level keys 'images' and 'annotations'.")

    annotations = data["annotations"]
    images = data["images"]

    kept_annotations = [ann for ann in annotations if ann_class_name(ann) in object_names]
    kept_image_ids = {int(ann["image_id"]) for ann in kept_annotations}
    kept_images = [img for img in images if int(img["id"]) in kept_image_ids]

    out = dict(data)
    out["annotations"] = kept_annotations
    out["images"] = kept_images

    os.makedirs(os.path.dirname(out_pth) or ".", exist_ok=True)
    torch.save(out, out_pth)

    print(f"[subset] in:  {in_pth}")
    print(f"[subset] out: {out_pth}")
    print(f"[subset] object_names: {sorted(object_names)}")
    print(f"[subset] annotations: {len(annotations)} -> {len(kept_annotations)}")
    print(f"[subset] images:      {len(images)} -> {len(kept_images)}")
    if len(kept_annotations) == 0:
        raise RuntimeError("No annotations kept. Check --object_names spelling.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_pth", required=True)
    p.add_argument("--out_pth", required=True)
    p.add_argument("--object_names", required=True, help="comma/space separated names, e.g. person or cat,dog")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    make_subset(args.in_pth, args.out_pth, parse_names(args.object_names))


if __name__ == "__main__":
    main()
