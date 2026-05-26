import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


try:
    from src.voc116_part_coarse import FINE_PART_CLASSES
except Exception:
    FINE_PART_CLASSES = []


def resolve_path(p: str, root: str = ".") -> Path:
    path = Path(p)
    if path.exists():
        return path
    path2 = Path(root) / p
    if path2.exists():
        return path2
    return path


def obj_to_part_seg_path(obj_seg_path: str) -> str:
    return obj_seg_path.replace("annotations_detectron2_obj", "annotations_detectron2_part")


def read_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def part_name(pid: int) -> str:
    if 0 <= pid < len(FINE_PART_CLASSES):
        return FINE_PART_CLASSES[pid]
    return f"part_{pid}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="pth file containing images list")
    parser.add_argument("--root", default=".", help="repo root for resolving relative paths")
    parser.add_argument("--num_parts", type=int, default=116)
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--print_all", action="store_true")
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()

    data = torch.load(args.dataset, map_location="cpu")
    images = data["images"]

    sum_ratio = np.zeros(args.num_parts, dtype=np.float64)
    present_count = np.zeros(args.num_parts, dtype=np.int64)
    present_sum_ratio = np.zeros(args.num_parts, dtype=np.float64)
    total_pixels_per_part = np.zeros(args.num_parts, dtype=np.int64)

    valid_images = 0
    skipped = 0

    for img in tqdm(images, desc="scan fine-116 part masks"):
        obj_seg = img.get("seg_file_name", None)
        if obj_seg is None:
            skipped += 1
            continue

        part_seg = obj_to_part_seg_path(str(obj_seg))
        part_path = resolve_path(part_seg, args.root)

        if not part_path.exists():
            skipped += 1
            continue

        mask = read_mask(part_path)
        h, w = mask.shape[:2]
        image_pixels = int(h * w)
        if image_pixels <= 0:
            skipped += 1
            continue

        # Count fine part ids 0..115. Ignore background / invalid ids.
        valid = (mask >= 0) & (mask < args.num_parts)
        counts = np.bincount(mask[valid].reshape(-1), minlength=args.num_parts)[:args.num_parts]

        ratios = counts.astype(np.float64) / float(image_pixels)

        sum_ratio += ratios
        total_pixels_per_part += counts

        appeared = counts > 0
        present_count += appeared.astype(np.int64)
        present_sum_ratio += ratios * appeared.astype(np.float64)

        valid_images += 1

    mean_ratio_all_images = sum_ratio / max(valid_images, 1)

    mean_ratio_present_images = np.zeros(args.num_parts, dtype=np.float64)
    nonzero = present_count > 0
    mean_ratio_present_images[nonzero] = present_sum_ratio[nonzero] / present_count[nonzero]

    rows = []
    for pid in range(args.num_parts):
        rows.append({
            "part_id": pid,
            "part_name": part_name(pid),
            "mean_ratio_all_images": float(mean_ratio_all_images[pid]),
            "mean_ratio_present_images": float(mean_ratio_present_images[pid]),
            "present_images": int(present_count[pid]),
            "total_images": int(valid_images),
            "presence_rate": float(present_count[pid] / max(valid_images, 1)),
            "total_part_pixels": int(total_pixels_per_part[pid]),
        })

    rows = sorted(rows, key=lambda x: x["mean_ratio_all_images"])

    print("=" * 120)
    print("[dataset]", args.dataset)
    print("total images:", len(images))
    print("valid images:", valid_images)
    print("skipped images:", skipped)
    print("sort key: mean_ratio_all_images = average(part_pixels / image_pixels) over all images")
    print("=" * 120)

    show_rows = rows if args.print_all else rows[:args.topk]
    for r in show_rows:
        print(
            f"id={r['part_id']:>3} | "
            f"mean_all={r['mean_ratio_all_images']:.10f} | "
            f"mean_present={r['mean_ratio_present_images']:.8f} | "
            f"present={r['present_images']:>5}/{r['total_images']:<5} | "
            f"presence={r['presence_rate']:.6f} | "
            f"pixels={r['total_part_pixels']:<10} | "
            f"{r['part_name']}"
        )

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "part_id",
                    "part_name",
                    "mean_ratio_all_images",
                    "mean_ratio_present_images",
                    "present_images",
                    "total_images",
                    "presence_rate",
                    "total_part_pixels",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print("=" * 120)
        print("[saved]", out)


if __name__ == "__main__":
    main()
