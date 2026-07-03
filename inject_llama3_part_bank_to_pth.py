#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inject an already-extracted Llama3 VOC116 part feature bank into Talk2DINO pth files.

This script does NOT import or run Llama. It only loads a saved bank such as:
  feature_new_exp/voc116/part/llama3_part_stage2_v1.pt

Expected bank fields:
  bank['mean_feats'] : Tensor [116, 4096]  (default feature used for injection)
  bank['feats']      : Tensor [116, N, 4096] optional, for inspection
  bank['names']      : list[str] optional
  bank['prompts']    : list[str] optional
  bank['meta']       : dict optional

For each annotation in each input pth:
  ann[output_part_key] = bank[feature_key][ann[part_id_key]]

Object feature ann['ann_feats'] is kept unchanged.
Original CLIP part feature ann['part_ann_feats'] is kept unchanged unless --replace_part_ann_feats is set.
"""

import argparse
import os
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def to_int_list(x: Any) -> List[int]:
    if torch.is_tensor(x):
        return [int(v) for v in x.detach().cpu().view(-1).tolist()]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def parse_io_pairs(input_pths: List[str], output_pths: List[str]) -> List[Tuple[str, str]]:
    if len(input_pths) != len(output_pths):
        raise ValueError(
            f"--input_pth and --output_pth must have the same number of values, "
            f"got {len(input_pths)} and {len(output_pths)}."
        )
    return list(zip(input_pths, output_pths))


def load_feature_bank(bank_path: str, feature_key: str, expected_num_parts: int) -> torch.Tensor:
    if not os.path.isfile(bank_path):
        raise FileNotFoundError(f"Bank file not found: {bank_path}")

    bank = torch.load(bank_path, map_location="cpu")
    if not isinstance(bank, dict):
        raise TypeError(f"Expected bank to be a dict, got {type(bank)} from {bank_path}")

    print(f"[BANK] loaded: {bank_path}")
    print(f"       keys: {list(bank.keys())}")

    if "feats" in bank and torch.is_tensor(bank["feats"]):
        print(f"       feats:      {tuple(bank['feats'].shape)} {bank['feats'].dtype}")
    if "mean_feats" in bank and torch.is_tensor(bank["mean_feats"]):
        print(f"       mean_feats: {tuple(bank['mean_feats'].shape)} {bank['mean_feats'].dtype}")
    if "prompts" in bank:
        print(f"       prompts:    {len(bank['prompts'])}")
    if "names" in bank:
        print(f"       names:      {len(bank['names'])}, first 5: {bank['names'][:5]}")
    if "meta" in bank and isinstance(bank["meta"], dict):
        meta = bank["meta"]
        for k in ["num_epochs", "num_prompts", "temperature", "top_p", "response_pool", "normalize"]:
            if k in meta:
                print(f"       meta.{k}: {meta[k]}")

    if feature_key not in bank:
        raise KeyError(f"Feature key '{feature_key}' not found in bank. Available keys: {list(bank.keys())}")
    feats = bank[feature_key]
    if not torch.is_tensor(feats):
        raise TypeError(f"bank['{feature_key}'] must be a torch.Tensor, got {type(feats)}")
    if feats.ndim != 2:
        raise ValueError(f"bank['{feature_key}'] must be [num_parts, dim], got shape {tuple(feats.shape)}")
    if expected_num_parts > 0 and feats.shape[0] != expected_num_parts:
        raise ValueError(
            f"Expected {expected_num_parts} part rows, but bank['{feature_key}'] has {feats.shape[0]} rows."
        )

    feats = feats.detach().cpu().float().contiguous()
    if torch.isnan(feats).any().item():
        raise ValueError(f"bank['{feature_key}'] contains NaN values.")

    print(f"[BANK] using feature_key='{feature_key}', shape={tuple(feats.shape)}, dtype=float32 on CPU")
    return feats


def inject_one_pth(
    input_pth: str,
    output_pth: str,
    feats: torch.Tensor,
    part_id_key: str,
    output_part_key: str,
    replace_part_ann_feats: bool,
    pth_dtype: str,
) -> None:
    if not os.path.isfile(input_pth):
        raise FileNotFoundError(f"Input pth not found: {input_pth}")

    data = torch.load(input_pth, map_location="cpu")
    if "annotations" not in data:
        raise KeyError(f"{input_pth} has no top-level key 'annotations'.")

    out_dtype = torch.float16 if pth_dtype == "float16" else torch.float32
    feats = feats.detach().cpu().float().contiguous()

    n_empty = 0
    n_total_parts = 0
    max_k = 0

    for ann_idx, ann in enumerate(tqdm(data["annotations"], desc=f"inject {Path(input_pth).name}")):
        if part_id_key not in ann:
            raise KeyError(
                f"Annotation {ann_idx} does not contain key '{part_id_key}'. "
                f"Available keys: {list(ann.keys())}"
            )

        part_ids = to_int_list(ann[part_id_key])
        if len(part_ids) == 0:
            part_tensor = torch.zeros((0, feats.shape[-1]), dtype=out_dtype)
            n_empty += 1
        else:
            min_id, max_id = min(part_ids), max(part_ids)
            if min_id < 0 or max_id >= feats.shape[0]:
                raise IndexError(
                    f"Annotation {ann_idx} has part id outside feature-bank range: "
                    f"min={min_id}, max={max_id}, bank_size={feats.shape[0]}"
                )
            part_idx = torch.as_tensor(part_ids, dtype=torch.long, device=feats.device)
            part_tensor = feats.index_select(0, part_idx).to(dtype=out_dtype).cpu()
            n_total_parts += len(part_ids)
            max_k = max(max_k, len(part_ids))

        ann[output_part_key] = part_tensor
        if replace_part_ann_feats:
            ann["part_ann_feats"] = part_tensor

    out_dir = os.path.dirname(output_pth)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data, output_pth)

    print(f"[OK] wrote {output_pth}")
    print(f"     input:          {input_pth}")
    print(f"     annotations:    {len(data['annotations'])}")
    print(f"     empty parts:    {n_empty}")
    print(f"     total part ids: {n_total_parts}")
    print(f"     max K/ann:      {max_k}")
    print(f"     added field:    {output_part_key}, dim={feats.shape[-1]}, dtype={out_dtype}")
    if replace_part_ann_feats:
        print("     also replaced original field: part_ann_feats")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load an existing Llama3 VOC116 part bank and inject it into Talk2DINO pth files."
    )

    parser.add_argument("--bank_path", type=str, required=True, help="Saved Llama3 bank .pt path.")
    parser.add_argument("--input_pth", type=str, nargs="+", required=True, help="Input Talk2DINO pth file(s).")
    parser.add_argument("--output_pth", type=str, nargs="+", required=True, help="Output pth file(s), same count/order as input_pth.")

    parser.add_argument("--feature_key", type=str, default="mean_feats", help="Bank tensor key to inject, default: mean_feats.")
    parser.add_argument("--part_id_key", type=str, default="part_category_id")
    parser.add_argument("--output_part_key", type=str, default="llama_part_ann_feats")
    parser.add_argument("--expected_num_parts", type=int, default=116)
    parser.add_argument("--pth_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--replace_part_ann_feats", action="store_true", help="Also overwrite ann['part_ann_feats'].")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("\n[CONFIG] Inject existing Llama3 part bank into Talk2DINO pth")
    print(f"  bank_path:       {args.bank_path}")
    print(f"  input_pth:       {args.input_pth}")
    print(f"  output_pth:      {args.output_pth}")
    print(f"  feature_key:     {args.feature_key}")
    print(f"  part_id_key:     {args.part_id_key}")
    print(f"  output_part_key: {args.output_part_key}")
    print(f"  expected_parts:  {args.expected_num_parts}")
    print(f"  pth_dtype:       {args.pth_dtype}")
    print(f"  replace_part_ann_feats: {args.replace_part_ann_feats}\n")

    feats = load_feature_bank(
        bank_path=args.bank_path,
        feature_key=args.feature_key,
        expected_num_parts=args.expected_num_parts,
    )

    for input_pth, output_pth in parse_io_pairs(args.input_pth, args.output_pth):
        inject_one_pth(
            input_pth=input_pth,
            output_pth=output_pth,
            feats=feats,
            part_id_key=args.part_id_key,
            output_part_key=args.output_part_key,
            replace_part_ann_feats=args.replace_part_ann_feats,
            pth_dtype=args.pth_dtype,
        )


if __name__ == "__main__":
    main()
