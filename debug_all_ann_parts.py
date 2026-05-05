from pathlib import Path
import argparse
import json
from collections import Counter

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_path", type=str, required=True)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    data = torch.load(args.pth_path, map_location="cpu")

    anns = data["annotations"]
    images = {img["id"]: img for img in data.get("images", [])}

    stats = Counter()
    raw_part_label_counter = Counter()
    raw_part_id_counter = Counter()

    per_ann_examples = []

    for ann in anns:
        stats["annotations_total"] += 1

        part_names = ann.get("part_class_name", []) or []
        part_ids = ann.get("part_category_id", []) or []
        if torch.is_tensor(part_ids):
            part_ids = part_ids.tolist()

        part_captions = ann.get("part_caption", []) or []
        crop_box = ann.get("cropaug_box_xyxy", None)

        if len(part_names) == 0:
            stats["annotations_without_part_names"] += 1
        else:
            stats["annotations_with_part_names"] += 1

        if len(part_ids) == 0:
            stats["annotations_without_part_ids"] += 1
        else:
            stats["annotations_with_part_ids"] += 1

        if crop_box is None:
            stats["annotations_without_crop_box"] += 1
        else:
            stats["annotations_with_crop_box"] += 1

        if len(part_names) != len(part_ids):
            stats["ann_part_name_id_len_mismatch"] += 1

        if len(part_names) != len(part_captions):
            stats["ann_part_name_caption_len_mismatch"] += 1

        stats["raw_num_part_instances"] += len(part_names)

        for n in part_names:
            raw_part_label_counter[str(n)] += 1
        for pid in part_ids:
            raw_part_id_counter[int(pid)] += 1

        if len(per_ann_examples) < 20:
            per_ann_examples.append({
                "annotation_id": ann.get("id"),
                "image_id": ann.get("image_id"),
                "class_name": ann.get("class_name", ""),
                "num_part_names": len(part_names),
                "num_part_ids": len(part_ids),
                "has_crop_box": crop_box is not None,
                "part_names": list(part_names[:10]),
                "part_ids": list(part_ids[:10]),
                "seg_file_name": images.get(ann.get("image_id"), {}).get("seg_file_name", None),
            })

    summary = {
        "num_images": len(images),
        "num_annotations": len(anns),
        "num_unique_raw_part_labels": len(raw_part_label_counter),
        "num_unique_raw_part_ids": len(raw_part_id_counter),
        "top50_raw_part_labels": dict(raw_part_label_counter.most_common(50)),
        "top50_raw_part_ids": dict(raw_part_id_counter.most_common(50)),
        "stats": dict(stats),
        "examples": per_ann_examples,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
