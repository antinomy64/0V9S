#!/usr/bin/env python3
from pathlib import Path
from PIL import Image

TOP_DIR = Path("anchor_point_pesudo_obj")
BOTTOM_DIR = Path("anchor_point_pesudo_obj_part")
OUT_DIR = Path("anchor_point_pesudo_joint")

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_images(folder: Path):
    return {
        p.name: p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }


def resize_to_width(img: Image.Image, target_w: int) -> Image.Image:
    if img.width == target_w:
        return img
    new_h = round(img.height * target_w / img.width)
    return img.resize((target_w, new_h))


def main():
    if not TOP_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {TOP_DIR}")
    if not BOTTOM_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {BOTTOM_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    top_files = load_images(TOP_DIR)
    bottom_files = load_images(BOTTOM_DIR)
    common_names = sorted(set(top_files) & set(bottom_files))

    print(f"top images: {len(top_files)}")
    print(f"bottom images: {len(bottom_files)}")
    print(f"common names: {len(common_names)}")

    for name in common_names:
        top_img = Image.open(top_files[name]).convert("RGB")
        bottom_img = Image.open(bottom_files[name]).convert("RGB")

        target_w = max(top_img.width, bottom_img.width)
        top_img = resize_to_width(top_img, target_w)
        bottom_img = resize_to_width(bottom_img, target_w)

        canvas = Image.new("RGB", (target_w, top_img.height + bottom_img.height), (255, 255, 255))
        canvas.paste(top_img, (0, 0))
        canvas.paste(bottom_img, (0, top_img.height))

        out_path = OUT_DIR / name
        canvas.save(out_path)

    print(f"Saved merged images to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
