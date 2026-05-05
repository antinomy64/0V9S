#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil

MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def collect_stems(mask_dir: Path):
    stems = []
    for p in sorted(mask_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in MASK_EXTS:
            stems.append(p.stem)
    return stems


def find_image(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def ensure_output(src: Path, dst: Path, copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def process_split(ann_root: Path, images_dir: Path, out_root: Path, split: str, copy: bool = False):
    mask_dir = ann_root / split
    if not mask_dir.exists():
        raise FileNotFoundError(f"split annotation dir not found: {mask_dir}")

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = collect_stems(mask_dir)
    matched = 0
    missing = []

    for stem in stems:
        img = find_image(images_dir, stem)
        if img is None:
            missing.append(stem)
            continue
        dst = out_dir / img.name
        ensure_output(img, dst, copy=copy)
        matched += 1

    return {
        "split": split,
        "num_masks": len(stems),
        "matched": matched,
        "missing": missing,
        "out_dir": out_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="根据 annotations_detectron2/pc59_train,pc59_val 的 mask 文件名前缀，从 JPEGImages 中查找同名图片，并在 pc59_image 下创建 split 目录。默认创建软链接。"
    )
    parser.add_argument(
        "--ann-root",
        type=str,
        default="data/pascal_context/VOCdevkit/VOC2010/annotations_detectron2",
        help="annotation 根目录，下面应包含 pc59_train 和 pc59_val",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/pascal_context/VOCdevkit/VOC2010/JPEGImages",
        help="原始图片目录",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/pascal_context/VOCdevkit/VOC2010/pc59_image",
        help="输出根目录，会创建 pc59_train 和 pc59_val 子目录",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["pc59_train", "pc59_val"],
        help="要处理的 split 列表",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制图片而不是创建软链接",
    )
    parser.add_argument(
        "--save-missing",
        type=str,
        default="",
        help="可选，把缺失图片的 stem 保存到文本文件",
    )
    args = parser.parse_args()

    ann_root = Path(args.ann_root)
    images_dir = Path(args.images_dir)
    out_root = Path(args.out_root)

    if not ann_root.exists():
        raise FileNotFoundError(f"annotation root not found: {ann_root}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")

    all_missing_lines = []
    total_masks = 0
    total_matched = 0

    for split in args.splits:
        res = process_split(ann_root, images_dir, out_root, split, copy=args.copy)
        total_masks += res["num_masks"]
        total_matched += res["matched"]
        print(f"[{res['split']}] masks={res['num_masks']} matched={res['matched']} missing={len(res['missing'])}")
        print(f"output: {res['out_dir']}")
        if res["missing"]:
            print("missing stems (first 20):", ", ".join(res["missing"][:20]))
            for stem in res["missing"]:
                all_missing_lines.append(f"{split}\t{stem}")

    print(f"[total] masks={total_masks} matched={total_matched} missing={total_masks - total_matched}")

    if args.save_missing:
        save_path = Path(args.save_missing)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text("\n".join(all_missing_lines), encoding="utf-8")
        print(f"[saved missing list] {save_path}")


if __name__ == "__main__":
    main()
