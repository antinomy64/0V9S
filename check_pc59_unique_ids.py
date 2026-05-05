#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_label_names(path: str):
    if not path:
        return None
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 支持两种格式：
            # 1) "1: accordion"
            # 2) "accordion"
            if ":" in line:
                left, right = line.split(":", 1)
                left = left.strip()
                right = right.strip()
                if left.isdigit():
                    mapping[int(left)] = right
            else:
                # 若不给编号，则按 0-based 依次编号
                mapping[len(mapping)] = line
    return mapping


def iter_mask_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def read_mask(path: Path):
    # 强制转 numpy，保留单通道 mask 的原始整数值
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        # 若是 RGB 调色板展开后的三通道，尝试取第一通道并提醒
        # 但语义分割 mask 通常应为单通道 / palette 模式
        if arr.shape[2] == 3:
            arr = arr[:, :, 0]
        elif arr.shape[2] == 4:
            arr = arr[:, :, 0]
    return arr.astype(np.int64, copy=False)


def main():
    parser = argparse.ArgumentParser(
        description="扫描 Pascal Context / PC59 annotation 目录，统计所有出现过的 unique label ids。"
    )
    parser.add_argument(
        "--ann-dir",
        type=str,
        default="data/pascal_context/VOCdevkit/VOC2010/annotations_detectron2/pc59_train",
        help="annotation 目录，例如 data/pascal_context/VOCdevkit/VOC2010/annotations_detectron2/pc59_train",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="可选，标签名字文件，例如 labels.txt；支持 '1: class_name' 格式。",
    )
    parser.add_argument(
        "--save-txt",
        type=str,
        default="",
        help="可选，将结果额外保存到 txt 文件。",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="显示每个 id 首次出现在哪些文件中（每个 id 最多展示 5 个文件）。",
    )
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    if not ann_dir.exists():
        raise FileNotFoundError(f"annotation 目录不存在: {ann_dir}")

    label_names = load_label_names(args.labels) if args.labels else None

    files = list(iter_mask_files(ann_dir))
    if not files:
        raise RuntimeError(f"在目录下没有找到可读取的 mask 文件: {ann_dir}")

    pixel_counter = Counter()
    file_counter = Counter()
    sample_files = defaultdict(list)

    for fp in files:
        mask = read_mask(fp)
        uniq, counts = np.unique(mask, return_counts=True)
        for uid, cnt in zip(uniq.tolist(), counts.tolist()):
            pixel_counter[uid] += int(cnt)
            file_counter[uid] += 1
            if len(sample_files[uid]) < 5:
                sample_files[uid].append(str(fp.relative_to(ann_dir)))

    unique_ids = sorted(pixel_counter.keys())

    lines = []
    lines.append(f"annotation dir: {ann_dir}")
    lines.append(f"num mask files: {len(files)}")
    lines.append(f"num unique ids: {len(unique_ids)}")
    lines.append("unique ids (sorted):")
    lines.append(" ".join(map(str, unique_ids)))
    lines.append("")
    lines.append("details:")
    for uid in unique_ids:
        name = ""
        if label_names is not None and uid in label_names:
            name = f" ({label_names[uid]})"
        lines.append(
            f"id={uid}{name}\tfiles={file_counter[uid]}\tpixels={pixel_counter[uid]}"
        )
        if args.show_files:
            for sf in sample_files[uid]:
                lines.append(f"  - {sf}")

    output = "\n".join(lines)
    print(output)

    if args.save_txt:
        out_path = Path(args.save_txt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
