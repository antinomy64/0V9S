import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

VOC116_OBJ_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

PC59_CLASSES = [
    "aeroplane", "bag", "bed", "bedclothes", "bench",
    "bicycle", "bird", "boat", "book", "bottle",
    "building", "bus", "cabinet", "car", "cat",
    "ceiling", "chair", "cloth", "computer", "cow",
    "cup", "curtain", "dog", "door", "fence",
    "floor", "flower", "food", "grass", "ground",
    "horse", "keyboard", "light", "motorbike", "mountain",
    "mouse", "person", "plate", "platform", "pottedplant",
    "road", "rock", "sheep", "shelves", "sidewalk",
    "sign", "sky", "snow", "sofa", "table",
    "track", "train", "tree", "truck", "tvmonitor",
    "wall", "water", "window", "wood",
]

CLASSES = VOC116_OBJ_CLASSES


def find_image_for_mask(mask_path: Path, images_dir: Path) -> Optional[Path]:
    stem = mask_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None



def extract_present_class_ids(mask_path: Path, num_classes: int, one_based_masks: bool) -> List[int]:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    uniq = sorted(int(x) for x in np.unique(mask))

    present: List[int] = []
    for uid in uniq:
        if uid == 255:
            continue
        if one_based_masks:
            # background is 0, class ids are 1..num_classes
            if 1 <= uid <= num_classes:
                present.append(uid - 1)
        else:
            # class ids are 0..num_classes-1; values outside that range ignored
            if 0 <= uid < num_classes:
                present.append(uid)
    return present


def build_split(
    split: str,
    ann_dir: Path,
    img_dir: Path,
    with_background: bool,
) -> Dict:
    masks_dir = ann_dir / split
    images_dir = img_dir / split
    if not masks_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {masks_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    mask_files = sorted(masks_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No PNG masks found in: {masks_dir}")

    images: List[Dict] = []
    annotations: List[Dict] = []
    skipped_no_img: List[str] = []
    skipped_empty: List[str] = []

    for img_id, mask_path in enumerate(tqdm(mask_files, desc=f"Building {split}")):
        image_path = find_image_for_mask(mask_path, images_dir)
        if image_path is None:
            skipped_no_img.append(mask_path.name)
            continue

        present_ids = extract_present_class_ids(mask_path, len(CLASSES), with_background)
        if not present_ids:
            skipped_empty.append(mask_path.name)
            continue

        images.append(
            {
                "id": len(images),
                "file_name": os.path.relpath(image_path.resolve()),
                "seg_file_name": os.path.relpath(mask_path.resolve()),
                "split": split,
            }
        )
        current_image_id = images[-1]["id"]

        for class_idx in present_ids:
            class_name = CLASSES[class_idx]
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": current_image_id,
                    "category_id": class_idx,
                    "class_name": class_name,
                    "caption": f"a photo of a {class_name}.",
                }
            )

    result = {"images": images, "annotations": annotations}

    print(f"[{split}] images kept: {len(images)}")
    print(f"[{split}] annotations: {len(annotations)}")
    print(f"[{split}] skipped (no matching image): {len(skipped_no_img)}")
    print(f"[{split}] skipped (empty/no valid classes): {len(skipped_empty)}")
    if images:
        print(f"[{split}] example image: {images[0]}")
    if annotations:
        print(f"[{split}] example annotation: {annotations[0]}")
    if skipped_no_img:
        print(f"[{split}] first missing image mask: {skipped_no_img[0]}")
    if skipped_empty:
        print(f"[{split}] first empty mask: {skipped_empty[0]}")

    return result



def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset's meta pth file.")
    parser.add_argument("--ann_dir", type=str, required=True, help="Dir of annotations_detectron2_obj, containing train/ and val/.")
    parser.add_argument("--img_dir", type=str, required=True, help="Dir of images, containing train/ and val/.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for train/val .pth files.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process.")
    parser.add_argument("--with_background", action="store_true", help="Set this if mask labels are 1..20 with 0 as background.")
    parser.add_argument(
        "--output_name",
        type=str,
        default="{split}_meta.pth",
        help="Output file name.",
    )
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        result = build_split(
            split=split,
            ann_dir=ann_dir,
            img_dir=img_dir,
            with_background=args.with_background,
        )
        out_path = out_dir / args.output_name.format(split=split)
        torch.save(result, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
