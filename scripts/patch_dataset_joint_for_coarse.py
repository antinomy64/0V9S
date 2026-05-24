#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Patch src/dataset_joint.py to support coarse PascalPart116 masks.

Run from repository root:
  python scripts/patch_dataset_joint_for_coarse.py
'''

from __future__ import annotations

from pathlib import Path
import re
import shutil


TARGET = Path("src/dataset_joint.py")


def must_sub(pattern: str, repl: str, text: str, desc: str, count: int = 1) -> str:
    new_text, n = re.subn(pattern, repl, text, count=count, flags=re.MULTILINE)
    if n != count:
        raise RuntimeError(f"Patch failed at: {desc}. Expected {count} replacement(s), got {n}.")
    return new_text


def main():
    if not TARGET.exists():
        raise FileNotFoundError(f"Cannot find {TARGET}. Run this script from repo root.")

    text = TARGET.read_text(encoding="utf-8")

    if "COARSE_ID_TO_FINE_IDS" in text and "def _part_binary_mask" in text:
        print("[skip] src/dataset_joint.py already appears patched.")
        return

    backup = TARGET.with_suffix(".py.bak_before_coarse")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"[backup] {backup}")

    if "COARSE_ID_TO_FINE_IDS" not in text:
        text = must_sub(
            r"^(from torchvision\.transforms import InterpolationMode\s*)$",
            "\\1\n\ntry:\n    from src.voc116_part_coarse import COARSE_ID_TO_FINE_IDS\nexcept Exception:\n    COARSE_ID_TO_FINE_IDS = {}\n",
            text,
            "insert coarse mapping import",
        )

    if "self.part_taxonomy" not in text:
        text = must_sub(
            r"^(\s*self\.class_part_bank\s*=\s*\{\}\s*)$",
            "\\1\n        self.part_taxonomy = \"fine\"\n",
            text,
            "insert self.part_taxonomy default",
        )

    if "def _part_binary_mask" not in text:
        pattern = (
            r"(\n\s*def _obj_to_part_seg_path\(self, obj_seg_path: str\) -> str:\s*\n"
            r"\s*return obj_seg_path\.replace\(\"annotations_detectron2_obj\", \"annotations_detectron2_part\"\)\s*\n)"
        )
        repl = (
            "\\1\n"
            "    def _part_binary_mask(self, mask: np.ndarray, part_id: int) -> np.ndarray:\n"
            "        \"\"\"Return uint8 binary mask for either fine or coarse part id.\"\"\"\n"
            "        if getattr(self, \"part_taxonomy\", \"fine\") == \"coarse_pascalpart116_v1\":\n"
            "            fine_ids = COARSE_ID_TO_FINE_IDS.get(int(part_id), [int(part_id)])\n"
            "            binary = np.isin(mask, fine_ids).astype(np.uint8) * 255\n"
            "        else:\n"
            "            binary = (mask == int(part_id)).astype(np.uint8) * 255\n"
            "        return binary\n\n"
        )
        text = must_sub(pattern, repl, text, "add _part_binary_mask helper")

    text = must_sub(
        r"binary\s*=\s*\(mask\s*==\s*int\(part_id\)\)\.astype\(np\.uint8\)\s*\*\s*255",
        r"binary = self._part_binary_mask(mask, int(part_id))",
        text,
        "replace cropaug fine part binary",
        count=1,
    )

    text = must_sub(
        r"binary\s*=\s*\(full_mask\s*==\s*int\(pid\)\)\.astype\(np\.uint8\)\s*\*\s*255",
        r"binary = self._part_binary_mask(full_mask, int(pid))",
        text,
        "replace full fine part binary",
        count=1,
    )

    if "data.get(\"part_taxonomy\", \"fine\")" not in text:
        pattern = (
            r"(\n\s*def _load_pth_dataset\(self, features_file: str\) -> None:\s*\n"
            r"\s*print\(\"Loading joint dataset\.\.\.\"\)\s*\n"
            r"\s*data = torch\.load\(features_file, map_location=\"cpu\"\)\s*\n)"
        )
        repl = (
            "\\1\n"
            "        self.part_taxonomy = str(data.get(\"part_taxonomy\", \"fine\"))\n"
            "        print(f\"Part taxonomy: {self.part_taxonomy}\")\n"
        )
        text = must_sub(pattern, repl, text, "read pth part_taxonomy")

    TARGET.write_text(text, encoding="utf-8")
    print(f"[patched] {TARGET}")
    print("[check] python -m py_compile src/dataset_joint.py")


if __name__ == "__main__":
    main()
