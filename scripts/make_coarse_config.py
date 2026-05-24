#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a coarse-part Stage2 config by copying an existing YAML and setting train.num_parts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from src.voc116_part_coarse import COARSE_PART_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_config", default="configs/vitb_mlp_infonce_exp2.yaml")
    parser.add_argument("--out_config", default="configs/vitb_mlp_infonce_coarse.yaml")
    args = parser.parse_args()

    with open(args.in_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("train", {})
    cfg["train"]["num_parts"] = len(COARSE_PART_CLASSES)

    out = Path(args.out_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[saved] {out}")
    print(f"train.num_parts = {len(COARSE_PART_CLASSES)}")


if __name__ == "__main__":
    main()
