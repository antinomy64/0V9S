#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any

import torch


def convert_to_half(obj: Any):
    """
    Recursively convert all floating-point torch tensors to float16.
    Keep bool/int/str/list metadata unchanged except recursive traversal.
    """
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            return obj.half()
        return obj
    if isinstance(obj, dict):
        return {k: convert_to_half(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_half(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_half(v) for v in obj)
    return obj


def count_tensor_dtypes(obj: Any, counter=None):
    if counter is None:
        counter = {}
    if torch.is_tensor(obj):
        key = str(obj.dtype)
        counter[key] = counter.get(key, 0) + 1
        return counter
    if isinstance(obj, dict):
        for v in obj.values():
            count_tensor_dtypes(v, counter)
        return counter
    if isinstance(obj, list) or isinstance(obj, tuple):
        for v in obj:
            count_tensor_dtypes(v, counter)
        return counter
    return counter


def sizeof_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def process_file(in_path: Path, out_path: Path):
    print(f"Loading: {in_path}")
    data = torch.load(in_path, map_location="cpu")

    before_counts = count_tensor_dtypes(data)
    data_half = convert_to_half(data)
    after_counts = count_tensor_dtypes(data_half)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_half, out_path)

    print(f"Saved:   {out_path}")
    print(f"Before dtypes: {before_counts}")
    print(f"After  dtypes: {after_counts}")
    print(f"Output size: {sizeof_mb(out_path):.2f} MB")
    print("-" * 80)


def default_output_path(in_path: Path, suffix: str) -> Path:
    if suffix:
        return in_path.with_name(in_path.stem + suffix + in_path.suffix)
    return in_path.with_name(in_path.stem + "_fp16" + in_path.suffix)


def main():
    parser = argparse.ArgumentParser(
        description="Convert all floating-point tensors inside one or more .pth files to float16."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input .pth files, e.g. train_voc116_obj_with_text.pth val_voc116_obj_with_text.pth",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Optional output directory. If omitted, save next to each input file.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_fp16",
        help="Suffix appended before .pth when --out_dir is not used. Default: _fp16",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input files in place. Use carefully.",
    )
    args = parser.parse_args()

    for in_file in args.inputs:
        in_path = Path(in_file)
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        if args.inplace:
            out_path = in_path
        elif args.out_dir:
            out_path = Path(args.out_dir) / in_path.name
        else:
            out_path = default_output_path(in_path, args.suffix)

        process_file(in_path, out_path)


if __name__ == "__main__":
    main()
