#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Llama3 object/part text-feature fields for VOC116 pth files.

Minimal-purpose script for test15:
  1) Load a global Llama3 part bank with 116 part features.
  2) Build object-level Llama3 feature by mean pooling all part features that
     belong to the same object class.
  3) Inject two new fields into train/val pth annotations:
       - llama_part_ann_feats: [K, 4096], local visible/available part features
       - llama_ann_feats:      [4096], object class feature = mean(class part features)

This script does not train anything and does not modify image/vision features.
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

SCRIPT_VERSION = "v2_partobjonly_skip_no_part_class_20260701"


def to_int_list(x) -> List[int]:
    if torch.is_tensor(x):
        return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    raise TypeError(f"Expected list/tuple/tensor part ids, got {type(x).__name__}")


def load_llama_part_bank(bank_path: str, bank_key: str, expected_num_parts: int) -> torch.Tensor:
    bank = torch.load(bank_path, map_location="cpu")
    if bank_key not in bank:
        raise KeyError(f"{bank_path} does not contain key '{bank_key}'. Available keys: {list(bank.keys())}")
    feats = bank[bank_key]
    if not torch.is_tensor(feats):
        feats = torch.as_tensor(feats)
    feats = feats.detach().cpu().float()
    if feats.ndim != 2:
        raise ValueError(f"bank['{bank_key}'] must be [num_parts, dim], got {tuple(feats.shape)}")
    if feats.shape[0] != expected_num_parts:
        raise ValueError(f"Expected {expected_num_parts} part features, got {feats.shape[0]}")
    return feats


def load_pth(path: str) -> Dict:
    data = torch.load(path, map_location="cpu")
    if "annotations" not in data:
        raise KeyError(f"{path} has no top-level key 'annotations'.")
    if "images" not in data:
        raise KeyError(f"{path} has no top-level key 'images'.")
    return data


def build_class_part_ids(
    datasets: Sequence[Tuple[str, Dict]],
    class_name_key: str,
    part_id_key: str,
    part_name_key: str,
) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    class_to_part_ids = defaultdict(set)
    part_id_to_name: Dict[int, str] = {}

    for pth_name, data in datasets:
        for ann_idx, ann in enumerate(data["annotations"]):
            if class_name_key not in ann:
                raise KeyError(f"{pth_name} annotation {ann_idx} missing '{class_name_key}'.")
            if part_id_key not in ann:
                raise KeyError(f"{pth_name} annotation {ann_idx} missing '{part_id_key}'.")

            class_name = str(ann[class_name_key])
            part_ids = to_int_list(ann[part_id_key])
            for pid in part_ids:
                class_to_part_ids[class_name].add(pid)

            if part_name_key in ann:
                part_names = ann[part_name_key]
                if len(part_names) != len(part_ids):
                    raise ValueError(
                        f"{pth_name} annotation {ann_idx}: len({part_name_key})={len(part_names)} "
                        f"!= len({part_id_key})={len(part_ids)}"
                    )
                for pid, pname in zip(part_ids, part_names):
                    pname = str(pname)
                    if pid in part_id_to_name and part_id_to_name[pid] != pname:
                        raise ValueError(
                            f"Part id {pid} has inconsistent names: "
                            f"'{part_id_to_name[pid]}' vs '{pname}'"
                        )
                    part_id_to_name[pid] = pname

    out = {cls: sorted(list(ids)) for cls, ids in class_to_part_ids.items()}
    return out, part_id_to_name


def build_class_obj_feats(
    class_to_part_ids: Dict[str, List[int]],
    part_bank: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    class_obj_feats = {}
    for cls, ids in sorted(class_to_part_ids.items()):
        if len(ids) == 0:
            raise ValueError(f"Class '{cls}' has no part ids; cannot build object feature.")
        idx = torch.as_tensor(ids, dtype=torch.long, device=part_bank.device)
        class_obj_feats[cls] = part_bank.index_select(0, idx).mean(dim=0).cpu()
    return class_obj_feats


def inject_one_pth(
    data: Dict,
    pth_name: str,
    part_bank: torch.Tensor,
    class_obj_feats: Dict[str, torch.Tensor],
    class_name_key: str,
    part_id_key: str,
    output_obj_key: str,
    output_part_key: str,
    dtype: str,
) -> Dict[str, int]:
    out_dtype = torch.float16 if dtype == "float16" else torch.float32
    stats = {
        "input_annotations": len(data["annotations"]),
        "kept_annotations": 0,
        "skipped_no_part_class": 0,
        "empty_part_annotations": 0,
        "total_local_parts": 0,
        "max_local_parts": 0,
    }
    new_annotations = []

    for ann_idx, ann in enumerate(tqdm(data["annotations"], desc=f"inject {Path(pth_name).name}")):
        if class_name_key not in ann:
            raise KeyError(f"{pth_name} annotation {ann_idx} missing '{class_name_key}'.")
        if part_id_key not in ann:
            raise KeyError(f"{pth_name} annotation {ann_idx} missing '{part_id_key}'.")

        cls = str(ann[class_name_key])
        if cls not in class_obj_feats:
            # Scheme A: keep only object categories that have Pascal-Part taxonomy.
            # Classes such as boat/chair/diningtable/sofa have no part ids in VOC116,
            # so their Llama object feature cannot be defined as mean(part features).
            stats["skipped_no_part_class"] += 1
            continue

        part_ids = to_int_list(ann[part_id_key])
        stats["kept_annotations"] += 1
        stats["total_local_parts"] += len(part_ids)
        stats["max_local_parts"] = max(stats["max_local_parts"], len(part_ids))
        if len(part_ids) == 0:
            stats["empty_part_annotations"] += 1
            part_feats = torch.zeros((0, part_bank.shape[-1]), dtype=out_dtype)
        else:
            min_id, max_id = min(part_ids), max(part_ids)
            if min_id < 0 or max_id >= part_bank.shape[0]:
                raise IndexError(
                    f"{pth_name} annotation {ann_idx}: part id outside bank range: "
                    f"min={min_id}, max={max_id}, bank_size={part_bank.shape[0]}"
                )
            idx = torch.as_tensor(part_ids, dtype=torch.long, device=part_bank.device)
            part_feats = part_bank.index_select(0, idx).to(dtype=out_dtype).cpu()

        ann[output_part_key] = part_feats
        ann[output_obj_key] = class_obj_feats[cls].to(dtype=out_dtype).cpu()
        new_annotations.append(ann)

    data["annotations"] = new_annotations
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank_path", type=str, required=True)
    parser.add_argument("--bank_key", type=str, default="mean_feats")
    parser.add_argument("--train_pth", type=str, required=True)
    parser.add_argument("--val_pth", type=str, required=True)
    parser.add_argument("--out_train_pth", type=str, required=True)
    parser.add_argument("--out_val_pth", type=str, required=True)
    parser.add_argument("--class_name_key", type=str, default="class_name")
    parser.add_argument("--part_id_key", type=str, default="part_category_id")
    parser.add_argument("--part_name_key", type=str, default="part_class_name")
    parser.add_argument("--output_obj_key", type=str, default="llama_ann_feats")
    parser.add_argument("--output_part_key", type=str, default="llama_part_ann_feats")
    parser.add_argument("--expected_num_parts", type=int, default=116)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--summary_path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[VERSION] {SCRIPT_VERSION}")

    part_bank = load_llama_part_bank(args.bank_path, args.bank_key, args.expected_num_parts)
    print(f"[bank] path={args.bank_path}")
    print(f"[bank] key={args.bank_key}, shape={tuple(part_bank.shape)}, dtype={part_bank.dtype}")

    train_data = load_pth(args.train_pth)
    val_data = load_pth(args.val_pth)
    datasets = [(args.train_pth, train_data), (args.val_pth, val_data)]

    class_to_part_ids, part_id_to_name = build_class_part_ids(
        datasets=datasets,
        class_name_key=args.class_name_key,
        part_id_key=args.part_id_key,
        part_name_key=args.part_name_key,
    )
    class_obj_feats = build_class_obj_feats(class_to_part_ids, part_bank)

    covered_part_ids = sorted({pid for ids in class_to_part_ids.values() for pid in ids})
    print(f"[taxonomy] classes={len(class_to_part_ids)}")
    print(f"[taxonomy] covered_part_ids={len(covered_part_ids)} / {args.expected_num_parts}")
    if len(covered_part_ids) != args.expected_num_parts:
        missing = sorted(set(range(args.expected_num_parts)) - set(covered_part_ids))
        raise ValueError(f"Covered part ids != {args.expected_num_parts}. Missing ids: {missing}")
    for cls, ids in sorted(class_to_part_ids.items()):
        print(f"  {cls:<15} K={len(ids):2d} ids={ids}")

    train_stats = inject_one_pth(
        data=train_data,
        pth_name=args.train_pth,
        part_bank=part_bank,
        class_obj_feats=class_obj_feats,
        class_name_key=args.class_name_key,
        part_id_key=args.part_id_key,
        output_obj_key=args.output_obj_key,
        output_part_key=args.output_part_key,
        dtype=args.dtype,
    )
    val_stats = inject_one_pth(
        data=val_data,
        pth_name=args.val_pth,
        part_bank=part_bank,
        class_obj_feats=class_obj_feats,
        class_name_key=args.class_name_key,
        part_id_key=args.part_id_key,
        output_obj_key=args.output_obj_key,
        output_part_key=args.output_part_key,
        dtype=args.dtype,
    )

    os.makedirs(os.path.dirname(args.out_train_pth), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_val_pth), exist_ok=True)
    torch.save(train_data, args.out_train_pth)
    torch.save(val_data, args.out_val_pth)

    summary = {
        "version": SCRIPT_VERSION,
        "bank_path": args.bank_path,
        "bank_key": args.bank_key,
        "bank_shape": list(part_bank.shape),
        "output_obj_key": args.output_obj_key,
        "output_part_key": args.output_part_key,
        "dtype": args.dtype,
        "num_classes": len(class_to_part_ids),
        "covered_part_ids": len(covered_part_ids),
        "class_to_part_ids": class_to_part_ids,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "out_train_pth": args.out_train_pth,
        "out_val_pth": args.out_val_pth,
    }

    if args.summary_path:
        import json
        os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[OK] wrote summary: {args.summary_path}")

    print(f"[OK] wrote train: {args.out_train_pth}")
    print(f"[OK] wrote val:   {args.out_val_pth}")
    print(f"[OK] added annotation fields: {args.output_obj_key}, {args.output_part_key}")
    print(f"[stats] train={train_stats}")
    print(f"[stats] val={val_stats}")


if __name__ == "__main__":
    main()
