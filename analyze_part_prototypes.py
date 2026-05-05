import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


def resolve_path(path_str: str, path_prefix: str | None = None) -> str:
    if os.path.exists(path_str):
        return path_str
    if path_prefix is not None:
        candidate = os.path.join(path_prefix, path_str)
        if os.path.exists(candidate):
            return candidate
    return path_str


def obj_to_part_seg_path(obj_seg_path: str) -> str:
    return obj_seg_path.replace("annotations_detectron2_obj", "annotations_detectron2_part")


def read_mask(seg_path: str, path_prefix: str | None = None) -> np.ndarray:
    seg_path = resolve_path(seg_path, path_prefix)
    mask = np.array(Image.open(seg_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def build_cropaug_part_mask_patch(mask: np.ndarray, part_id: int, crop_box_xyxy, grid_size: int) -> torch.Tensor:
    if torch.is_tensor(crop_box_xyxy):
        crop_box_xyxy = crop_box_xyxy.tolist()
    x1, y1, x2, y2 = [int(v) for v in crop_box_xyxy]
    binary = (mask == int(part_id)).astype(np.uint8) * 255
    binary = binary[y1:y2, x1:x2]
    pil_mask = Image.fromarray(binary)
    pil_mask = TF.resize(
        pil_mask,
        [grid_size, grid_size],
        interpolation=InterpolationMode.NEAREST,
    )
    return torch.from_numpy((np.array(pil_mask) > 0).astype(np.bool_)).view(-1)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def describe_global_distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    robust_sigma = 1.4826 * mad
    return {
        "n_pairs": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": med,
        "mad": mad,
        "robust_sigma": float(robust_sigma),
        "q50": med,
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q97_5": float(np.quantile(values, 0.975)),
        "q99": float(np.quantile(values, 0.99)),
    }


def judge_by_mean_std(z_mean_vs_global: float | None, std_ratio_vs_global: float | None) -> str:
    """
    Judgment rule using ONLY cosine mean and std:
      z_o = (m_o - mu_global) / sigma_global
      r_o = s_o / sigma_global
    """
    if z_mean_vs_global is None or std_ratio_vs_global is None:
        return "undefined"

    # Good separability: lower-than-global mean similarity and not more uneven than global
    if z_mean_vs_global < 0.0 and std_ratio_vs_global <= 1.0:
        return "good_separability"

    # Local difficulty: average not worse than global, but dispersion is larger
    if z_mean_vs_global <= 0.0 and std_ratio_vs_global > 1.0:
        return "local_difficulty"

    # Overall poor: average similarity is higher than global, distribution not more uneven
    if z_mean_vs_global > 0.0 and std_ratio_vs_global <= 1.0:
        return "overall_poor"

    # Both mean is high and dispersion is high
    return "overall_poor_and_uneven"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_path", type=str, required=True)
    parser.add_argument("--path_prefix", type=str, default=None)
    parser.add_argument("--part_feature_name", type=str, default="cropaug_patch_tokens")
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None, help="Object-level summary CSV")
    parser.add_argument("--topk_pairs", type=int, default=50)
    args = parser.parse_args()

    data = torch.load(args.pth_path, map_location="cpu")
    images = {imm["id"]: imm for imm in data["images"]}
    grid_size = args.crop_dim // args.patch_size
    num_patches_expected = grid_size * grid_size

    # ---------------------------------------------------------------------
    # Build one global prototype per part label by aggregating all masked patch
    # tokens from all images/instances that belong to that part.
    # ---------------------------------------------------------------------
    part_sum = {}
    part_patch_count = Counter()
    part_instance_count = Counter()
    part_to_obj = {}
    build_stats = Counter()

    for ann in data["annotations"]:
        build_stats["annotations_total"] += 1

        class_name = ann.get("class_name", "")
        part_names = ann.get("part_class_name", []) or []
        part_ids = ann.get("part_category_id", []) or []
        if torch.is_tensor(part_ids):
            part_ids = part_ids.tolist()
        crop_box = ann.get("cropaug_box_xyxy", None)

        if len(part_names) == 0 or len(part_ids) == 0:
            build_stats["annotations_without_parts"] += 1
            continue
        if crop_box is None:
            build_stats["annotations_missing_crop_box"] += 1
            continue

        if args.part_feature_name in ann:
            patch_tokens = ann[args.part_feature_name]
        else:
            imm = images[ann["image_id"]]
            patch_tokens = imm.get(args.part_feature_name, None)

        if patch_tokens is None or not torch.is_tensor(patch_tokens):
            build_stats["annotations_missing_patch_tokens"] += 1
            continue

        patch_tokens = patch_tokens.float()
        if patch_tokens.shape[0] != num_patches_expected:
            build_stats["patch_token_shape_unexpected"] += 1

        obj_seg_path = images[ann["image_id"]]["seg_file_name"]
        part_seg_path = obj_to_part_seg_path(obj_seg_path)
        part_seg_path = resolve_path(part_seg_path, args.path_prefix)
        if not os.path.exists(part_seg_path):
            build_stats["missing_part_seg_path"] += 1
            continue
        mask = read_mask(part_seg_path, None)

        valid_in_ann = 0
        for part_name, part_id in zip(part_names, part_ids):
            part_mask_patch = build_cropaug_part_mask_patch(mask, int(part_id), crop_box, grid_size)
            if part_mask_patch.numel() != patch_tokens.shape[0]:
                build_stats["mask_token_shape_mismatch"] += 1
                continue
            if part_mask_patch.sum().item() == 0:
                build_stats["empty_part_masks"] += 1
                continue

            selected = patch_tokens[part_mask_patch]
            if selected.numel() == 0:
                build_stats["empty_selected_patch_tokens"] += 1
                continue

            lb = str(part_name)
            if lb not in part_sum:
                part_sum[lb] = selected.sum(dim=0)
            else:
                part_sum[lb] += selected.sum(dim=0)
            part_patch_count[lb] += int(selected.shape[0])
            part_instance_count[lb] += 1
            part_to_obj[lb] = class_name
            valid_in_ann += 1
            build_stats["valid_part_instances"] += 1

        if valid_in_ann == 0:
            build_stats["annotations_no_valid_parts"] += 1

    if len(part_sum) == 0:
        raise RuntimeError("No valid part prototypes were built. Check part masks / crop boxes / path_prefix.")

    proto_by_part = {}
    for part_label, s in part_sum.items():
        proto = s / max(part_patch_count[part_label], 1)
        proto_by_part[part_label] = l2_normalize(proto.unsqueeze(0)).squeeze(0)

    # ---------------------------------------------------------------------
    # Compute within-object part-pair cosine similarities.
    # ---------------------------------------------------------------------
    obj_to_parts = defaultdict(list)
    for part_label, obj_name in part_to_obj.items():
        obj_to_parts[obj_name].append(part_label)

    pair_rows = []
    per_object_matrix = {}

    for obj_name in sorted(obj_to_parts.keys()):
        parts = sorted(obj_to_parts[obj_name])
        if len(parts) < 2:
            per_object_matrix[obj_name] = {"parts": parts, "cosine_matrix": None}
            continue

        feats = torch.stack([proto_by_part[p] for p in parts], dim=0)
        sims = feats @ feats.T
        per_object_matrix[obj_name] = {
            "parts": parts,
            "cosine_matrix": sims.cpu().numpy().tolist(),
        }

        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                pair_rows.append({
                    "object_class": obj_name,
                    "part_a": parts[i],
                    "part_b": parts[j],
                    "cosine": float(sims[i, j]),
                    "part_a_instances": int(part_instance_count[parts[i]]),
                    "part_b_instances": int(part_instance_count[parts[j]]),
                    "part_a_patches": int(part_patch_count[parts[i]]),
                    "part_b_patches": int(part_patch_count[parts[j]]),
                })

    if len(pair_rows) == 0:
        raise RuntimeError("No within-object part pairs were built.")

    all_vals = np.array([r["cosine"] for r in pair_rows], dtype=np.float64)
    global_dist = describe_global_distribution(all_vals)
    mu_global = global_dist["mean"]
    sigma_global = global_dist["std"]

    # ---------------------------------------------------------------------
    # Object-level summary using ONLY mean and std relative to the global
    # within-object pairwise cosine distribution.
    # ---------------------------------------------------------------------
    summary_rows = []
    for obj_name in sorted(obj_to_parts.keys()):
        parts = sorted(obj_to_parts[obj_name])

        if len(parts) < 2:
            summary_rows.append({
                "object_class": obj_name,
                "num_parts": len(parts),
                "pair_count": 0,
                "pair_cos_mean": None,
                "pair_cos_std": None,
                "pair_cos_median": None,
                "pair_cos_max": None,
                "pair_cos_min": None,
                "global_pair_cos_mean": float(mu_global),
                "global_pair_cos_std": float(sigma_global),
                "z_mean_vs_global": None,
                "std_ratio_vs_global": None,
                "judge_mean_std_only": "undefined",
                "hardest_pair_a": None,
                "hardest_pair_b": None,
                "hardest_pair_cos": None,
                "easiest_pair_a": None,
                "easiest_pair_b": None,
                "easiest_pair_cos": None,
            })
            continue

        rows = [r for r in pair_rows if r["object_class"] == obj_name]
        vals = np.array([r["cosine"] for r in rows], dtype=np.float64)

        pair_cos_mean = float(vals.mean())
        pair_cos_std = float(vals.std())
        pair_cos_median = float(np.median(vals))
        pair_cos_max = float(vals.max())
        pair_cos_min = float(vals.min())

        # z_o = (m_o - mu_global) / sigma_global
        z_mean_vs_global = float((pair_cos_mean - mu_global) / max(sigma_global, 1e-12))
        # r_o = s_o / sigma_global
        std_ratio_vs_global = float(pair_cos_std / max(sigma_global, 1e-12))

        hardest = max(rows, key=lambda x: x["cosine"])
        easiest = min(rows, key=lambda x: x["cosine"])

        summary_rows.append({
            "object_class": obj_name,
            "num_parts": len(parts),
            "pair_count": int(vals.size),
            "pair_cos_mean": pair_cos_mean,
            "pair_cos_std": pair_cos_std,
            "pair_cos_median": pair_cos_median,
            "pair_cos_max": pair_cos_max,
            "pair_cos_min": pair_cos_min,
            "global_pair_cos_mean": float(mu_global),
            "global_pair_cos_std": float(sigma_global),
            "z_mean_vs_global": z_mean_vs_global,
            "std_ratio_vs_global": std_ratio_vs_global,
            "judge_mean_std_only": judge_by_mean_std(z_mean_vs_global, std_ratio_vs_global),
            "hardest_pair_a": hardest["part_a"],
            "hardest_pair_b": hardest["part_b"],
            "hardest_pair_cos": float(hardest["cosine"]),
            "easiest_pair_a": easiest["part_a"],
            "easiest_pair_b": easiest["part_b"],
            "easiest_pair_cos": float(easiest["cosine"]),
        })

    # sort by mean z-score descending, then std ratio descending
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda x: (
            x["z_mean_vs_global"] if x["z_mean_vs_global"] is not None else -999,
            x["std_ratio_vs_global"] if x["std_ratio_vs_global"] is not None else -999,
        ),
        reverse=True,
    )

    pair_rows_sorted = sorted(pair_rows, key=lambda x: x["cosine"], reverse=True)

    output = {
        "build_stats": dict(build_stats),
        "global_pair_distribution": global_dist,
        "criterion_explanation": {
            "z_mean_vs_global": "z_o = (pair_cos_mean - global_pair_cos_mean) / global_pair_cos_std",
            "std_ratio_vs_global": "r_o = pair_cos_std / global_pair_cos_std",
            "judge_mean_std_only": {
                "good_separability": "z_o < 0 and r_o <= 1",
                "local_difficulty": "z_o <= 0 and r_o > 1",
                "overall_poor": "z_o > 0 and r_o <= 1",
                "overall_poor_and_uneven": "z_o > 0 and r_o > 1",
            },
        },
        "per_object_summary": summary_rows_sorted,
        "top_confusing_pairs_global": pair_rows_sorted[: args.topk_pairs],
        "per_object_matrix": per_object_matrix,
    }

    if args.out_json is not None:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    if args.out_csv is not None:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        base, _ = os.path.splitext(args.out_csv)
        summary_csv = args.out_csv
        pairs_csv = base + "_pairs.csv"

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows_sorted[0].keys()) if summary_rows_sorted else ["object_class"])
            writer.writeheader()
            for row in summary_rows_sorted:
                writer.writerow(row)

        with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows_sorted[0].keys()) if pair_rows_sorted else ["object_class", "part_a", "part_b", "cosine"])
            writer.writeheader()
            for row in pair_rows_sorted:
                writer.writerow(row)

        print(f"Saved object summary csv to {summary_csv}")
        print(f"Saved pairwise csv to {pairs_csv}")

    preview = {
        "build_stats": dict(build_stats),
        "global_pair_distribution": global_dist,
        "objects_sorted_by_z_mean": [
            {
                "object_class": row["object_class"],
                "num_parts": row["num_parts"],
                "pair_cos_mean": row["pair_cos_mean"],
                "pair_cos_std": row["pair_cos_std"],
                "z_mean_vs_global": row["z_mean_vs_global"],
                "std_ratio_vs_global": row["std_ratio_vs_global"],
                "judge_mean_std_only": row["judge_mean_std_only"],
                "hardest_pair_a": row["hardest_pair_a"],
                "hardest_pair_b": row["hardest_pair_b"],
                "hardest_pair_cos": row["hardest_pair_cos"],
            }
            for row in summary_rows_sorted[:20]
        ],
        "top_confusing_pairs_global": pair_rows_sorted[: min(args.topk_pairs, 20)],
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
