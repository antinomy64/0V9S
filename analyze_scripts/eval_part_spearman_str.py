#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate per-object part-structure alignment for Talk2DINO test15.

This version is adapted to pth files whose annotations contain:
  - part_ann_feats
  - llama_part_ann_feats
  - cropaug_patch_tokens
  - cropaug_box_xyxy
but do NOT contain precomputed patch-level GT part masks.

GT visual prototypes are rebuilt from the original PascalPart116 part segmentation PNG:
  images[i]['seg_file_name'] points to annotations_detectron2_obj/...
  part mask path is derived by replacing annotations_detectron2_obj with annotations_detectron2_part.

For each annotation:
  part segmentation mask -> crop by cropaug_box_xyxy -> nearest resize to patch grid
  -> select cropaug_patch_tokens belonging to each GT part id -> accumulate mean prototype.

Outputs:
  analyze/part_spearman_str/eval_part_spearman_str.txt
  analyze/part_spearman_str/eval_part_spearman_str.csv
  analyze/part_spearman_str/part_support.csv
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from scipy.stats import spearmanr
    from scipy.optimize import linear_sum_assignment
except Exception as exc:
    raise ImportError("This script requires scipy.") from exc


DEFAULT_INPUT_PTH = "feature/voc116_obj_part_test15/train_voc116_obj_with_llama3_part.pth"
DEFAULT_OUT_DIR = "analyze/part_spearman_str"
SCRIPT_VERSION = "v2_from_pth_segmask_20260701"


def to_int_list(x: Any) -> List[int]:
    if torch.is_tensor(x):
        return [int(v) for v in x.detach().cpu().view(-1).tolist()]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def to_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(x)]


def ensure_2d_tensor(x: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach().cpu()

    # Accept:
    #   [D]          -> [1, D]
    #   [K, D]       -> [K, D]
    #   [K, P, D]    -> [K, D] by averaging P prompt features
    # This is needed when LLaMA3 features are injected with inject_mode=all.
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() == 3:
        x = x.float().mean(dim=1)

    if x.dim() != 2:
        raise ValueError(f"{name} must be 2D after loading, got shape={tuple(x.shape)}")
    return x.float()


def ensure_patch_tokens(x: Any, key: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach().cpu()
    if x.dim() == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.dim() != 2:
        raise ValueError(f"{key} must be [num_patches, dim], got shape={tuple(x.shape)}")
    return x.float()


def parse_object_name(part_full_name: str) -> str:
    if "'s" not in part_full_name:
        raise ValueError(f"Part name does not contain \"'s\": {part_full_name}")
    return part_full_name.split("'s", 1)[0].strip()


def infer_part_names(data: Dict[str, Any], part_id_key: str, part_name_key: str, expected_num_parts: int) -> List[str]:
    id_to_name: Dict[int, str] = {}
    for ann_idx, ann in enumerate(data["annotations"]):
        if part_id_key not in ann:
            raise KeyError(f"Annotation {ann_idx} missing required key: {part_id_key}")
        if part_name_key not in ann:
            raise KeyError(f"Annotation {ann_idx} missing required key: {part_name_key}")
        part_ids = to_int_list(ann[part_id_key])
        part_names = to_str_list(ann[part_name_key])
        if len(part_ids) != len(part_names):
            raise ValueError(
                f"Annotation {ann_idx}: len({part_id_key})={len(part_ids)} but "
                f"len({part_name_key})={len(part_names)}"
            )
        for pid, pname in zip(part_ids, part_names):
            if pid in id_to_name and id_to_name[pid] != pname:
                raise ValueError(
                    f"Conflicting part name for id {pid}: {id_to_name[pid]} vs {pname} at annotation {ann_idx}"
                )
            id_to_name[pid] = pname

    if len(id_to_name) == 0:
        raise RuntimeError("No part names found in annotations.")

    max_id = max(id_to_name.keys())
    names: List[str] = []
    missing: List[int] = []
    for pid in range(max_id + 1):
        if pid not in id_to_name:
            missing.append(pid)
            names.append(f"__missing_part_{pid}__")
        else:
            names.append(id_to_name[pid])

    if missing:
        raise RuntimeError(f"Missing part names for ids: {missing[:30]}{'...' if len(missing) > 30 else ''}")
    if expected_num_parts > 0 and len(names) != expected_num_parts:
        raise RuntimeError(f"Expected {expected_num_parts} parts, inferred {len(names)} parts.")
    return names


def derive_part_seg_path(image_info: Dict[str, Any], obj_seg_key: str, part_seg_key: str) -> str:
    if part_seg_key in image_info:
        return str(image_info[part_seg_key])
    if obj_seg_key not in image_info:
        raise KeyError(
            f"image info has no '{part_seg_key}' and no '{obj_seg_key}', cannot derive part mask path. "
            f"Available keys: {list(image_info.keys())}"
        )
    obj_path = str(image_info[obj_seg_key])
    if "annotations_detectron2_obj" not in obj_path:
        raise ValueError(
            f"Cannot derive part mask path from {obj_seg_key}={obj_path}. "
            "Expected substring 'annotations_detectron2_obj'."
        )
    return obj_path.replace("annotations_detectron2_obj", "annotations_detectron2_part")


def load_label_mask(path: str, cache: Dict[str, np.ndarray]) -> np.ndarray:
    if path in cache:
        return cache[path]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Part segmentation mask not found: {path}")
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = arr.astype(np.int64)
    cache[path] = arr
    return arr


def crop_resize_part_mask(mask: np.ndarray, box_xyxy: torch.Tensor, patch_grid: int) -> np.ndarray:
    if not torch.is_tensor(box_xyxy):
        box_xyxy = torch.as_tensor(box_xyxy)
    box = [int(v) for v in box_xyxy.detach().cpu().view(-1).tolist()]
    if len(box) != 4:
        raise ValueError(f"cropaug_box_xyxy must have 4 values, got {box}")

    h, w = mask.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop from box={box} for mask shape={mask.shape}")

    crop_img = Image.fromarray(crop.astype(np.int32), mode="I")
    resized = crop_img.resize((patch_grid, patch_grid), resample=Image.NEAREST)
    return np.array(resized).astype(np.int64)


def infer_dims(data: Dict[str, Any], args: argparse.Namespace) -> Tuple[int, int, int, int]:
    for ann_idx, ann in enumerate(data["annotations"]):
        part_ids = to_int_list(ann[args.part_id_key])
        if len(part_ids) == 0:
            continue
        for k in [args.clip_part_key, args.llama_part_key, args.patch_token_key, args.crop_box_key]:
            if k not in ann:
                raise KeyError(f"Annotation {ann_idx} missing required key: {k}")
        clip_dim = ensure_2d_tensor(ann[args.clip_part_key], args.clip_part_key).shape[-1]
        llama_dim = ensure_2d_tensor(ann[args.llama_part_key], args.llama_part_key).shape[-1]
        patch_tokens = ensure_patch_tokens(ann[args.patch_token_key], args.patch_token_key)
        num_patches, vision_dim = patch_tokens.shape
        patch_grid = int(round(num_patches ** 0.5))
        if patch_grid * patch_grid != num_patches:
            raise ValueError(f"num_patches={num_patches} is not a square number; cannot infer patch grid.")
        return clip_dim, llama_dim, vision_dim, patch_grid
    raise RuntimeError("Could not infer dims because all annotations have empty part ids.")


def build_prototypes_from_pth(args: argparse.Namespace) -> Dict[str, Any]:
    data = torch.load(args.input_pth, map_location="cpu")
    if "images" not in data or "annotations" not in data:
        raise KeyError(f"{args.input_pth} must have top-level keys 'images' and 'annotations'.")

    part_names = infer_part_names(data, args.part_id_key, args.part_name_key, args.expected_num_parts)
    num_parts = len(part_names)
    clip_dim, llama_dim, vision_dim, patch_grid = infer_dims(data, args)

    image_by_id = {int(img["id"]): img for img in data["images"]}
    mask_cache: Dict[str, np.ndarray] = {}

    clip_sum = torch.zeros(num_parts, clip_dim, dtype=torch.float64)
    llama_sum = torch.zeros(num_parts, llama_dim, dtype=torch.float64)
    text_count = torch.zeros(num_parts, dtype=torch.float64)

    vision_sum = torch.zeros(num_parts, vision_dim, dtype=torch.float64)
    vision_patch_count = torch.zeros(num_parts, dtype=torch.float64)
    vision_instance_count = torch.zeros(num_parts, dtype=torch.float64)

    empty_part_ann = 0
    ann_zero_patch_part = 0
    total_local_parts = 0

    for ann_idx, ann in enumerate(tqdm(data["annotations"], desc=f"build GT prototypes from {Path(args.input_pth).name}")):
        part_ids = to_int_list(ann[args.part_id_key])
        if len(part_ids) == 0:
            empty_part_ann += 1
            continue
        total_local_parts += len(part_ids)

        image_id = int(ann[args.image_id_key])
        if image_id not in image_by_id:
            raise KeyError(f"Annotation {ann_idx} image_id={image_id} not found in images.")
        image_info = image_by_id[image_id]
        part_seg_path = derive_part_seg_path(image_info, args.obj_seg_key, args.part_seg_key)

        clip_feats = ensure_2d_tensor(ann[args.clip_part_key], args.clip_part_key)
        llama_feats = ensure_2d_tensor(ann[args.llama_part_key], args.llama_part_key)
        patch_tokens = ensure_patch_tokens(ann[args.patch_token_key], args.patch_token_key)

        if clip_feats.shape[0] != len(part_ids):
            raise ValueError(
                f"Annotation {ann_idx}: {args.clip_part_key}.shape[0]={clip_feats.shape[0]} "
                f"but len(part_ids)={len(part_ids)}"
            )
        if llama_feats.shape[0] != len(part_ids):
            raise ValueError(
                f"Annotation {ann_idx}: {args.llama_part_key}.shape[0]={llama_feats.shape[0]} "
                f"but len(part_ids)={len(part_ids)}"
            )
        if patch_tokens.shape[0] != patch_grid * patch_grid:
            raise ValueError(
                f"Annotation {ann_idx}: patch token count changed: {patch_tokens.shape[0]} vs {patch_grid * patch_grid}"
            )

        clip_feats = F.normalize(torch.nan_to_num(clip_feats, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)
        llama_feats = F.normalize(torch.nan_to_num(llama_feats, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)
        patch_tokens = F.normalize(torch.nan_to_num(patch_tokens, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)

        full_part_mask = load_label_mask(part_seg_path, mask_cache)
        patch_label_grid = crop_resize_part_mask(full_part_mask, ann[args.crop_box_key], patch_grid)
        patch_labels = torch.from_numpy(patch_label_grid.reshape(-1)).long()

        for local_idx, pid in enumerate(part_ids):
            if pid < 0 or pid >= num_parts:
                raise IndexError(f"Annotation {ann_idx}: part id {pid} outside [0, {num_parts})")

            clip_sum[pid] += clip_feats[local_idx].double()
            llama_sum[pid] += llama_feats[local_idx].double()
            text_count[pid] += 1.0

            m = patch_labels == int(pid)
            n = int(m.sum().item())
            if n == 0:
                ann_zero_patch_part += 1
                continue
            vision_sum[pid] += patch_tokens[m].sum(dim=0).double()
            vision_patch_count[pid] += float(n)
            vision_instance_count[pid] += 1.0

    clip_proto = torch.zeros(num_parts, clip_dim, dtype=torch.float32)
    llama_proto = torch.zeros(num_parts, llama_dim, dtype=torch.float32)
    vision_proto = torch.zeros(num_parts, vision_dim, dtype=torch.float32)

    text_valid = text_count > 0
    vision_valid = vision_patch_count > 0

    clip_proto[text_valid] = (clip_sum[text_valid] / text_count[text_valid].unsqueeze(1)).float()
    llama_proto[text_valid] = (llama_sum[text_valid] / text_count[text_valid].unsqueeze(1)).float()
    vision_proto[vision_valid] = (vision_sum[vision_valid] / vision_patch_count[vision_valid].unsqueeze(1)).float()

    clip_proto = F.normalize(torch.nan_to_num(clip_proto, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)
    llama_proto = F.normalize(torch.nan_to_num(llama_proto, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)
    vision_proto = F.normalize(torch.nan_to_num(vision_proto, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)

    return {
        "part_names": part_names,
        "clip_proto": clip_proto,
        "llama_proto": llama_proto,
        "vision_proto": vision_proto,
        "text_count": text_count,
        "vision_patch_count": vision_patch_count,
        "vision_instance_count": vision_instance_count,
        "stats": {
            "num_images": len(data["images"]),
            "num_annotations": len(data["annotations"]),
            "empty_part_annotations": empty_part_ann,
            "total_local_parts": total_local_parts,
            "ann_zero_patch_part": ann_zero_patch_part,
            "num_mask_files_loaded": len(mask_cache),
            "patch_grid": patch_grid,
            "clip_dim": clip_dim,
            "llama_dim": llama_dim,
            "vision_dim": vision_dim,
        },
    }


def structure_retrieval_core(feat_1: torch.Tensor, feat_2: torch.Tensor, use_hm: bool = False) -> float:
    feat_1 = F.normalize(feat_1.float().cpu(), dim=-1, eps=1e-6)
    feat_2 = F.normalize(feat_2.float().cpu(), dim=-1, eps=1e-6)
    sim_1 = feat_1 @ feat_1.t()
    sim_2 = feat_2 @ feat_2.t()
    n = feat_1.shape[0]
    if n < 2:
        return float("nan")

    eye = torch.eye(n, dtype=torch.bool)
    sim_1 = sim_1[~eye].view(n, -1)
    sim_2 = sim_2[~eye].view(n, -1)

    sim_1 = sim_1 - sim_1.mean(dim=-1, keepdim=True)
    sim_2 = sim_2 - sim_2.mean(dim=-1, keepdim=True)
    sim_1 = sim_1 / sim_1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sim_2 = sim_2 / sim_2.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    sim_1_2 = sim_1 @ sim_2.t()
    if not use_hm:
        idx = sim_1_2.argmax(dim=0).cpu().numpy()
        return float((idx == np.arange(n)).sum() / n)

    row_ind, col_ind = linear_sum_assignment((1.0 - sim_1_2).cpu().numpy())
    assigned = np.empty(n, dtype=np.int64)
    assigned[row_ind] = col_ind
    return float((assigned == np.arange(n)).sum() / n)


def spearman_structure(feat_t: torch.Tensor, feat_v: torch.Tensor) -> float:
    feat_t = F.normalize(feat_t.float().cpu(), dim=-1, eps=1e-6)
    feat_v = F.normalize(feat_v.float().cpu(), dim=-1, eps=1e-6)
    n = feat_t.shape[0]
    if n < 2:
        return float("nan")
    sim_t = (feat_t @ feat_t.t()).numpy()
    sim_v = (feat_v @ feat_v.t()).numpy()
    iu = np.triu_indices(n, k=1)
    rho = spearmanr(sim_t[iu], sim_v[iu]).correlation
    return float(rho) if rho is not None else float("nan")


def object_groups(part_names: List[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(part_names):
        obj = parse_object_name(name)
        groups.setdefault(obj, []).append(idx)
    return groups


def evaluate_one_text(
    text_name: str,
    text_proto: torch.Tensor,
    vision_proto: torch.Tensor,
    part_names: List[str],
    text_count: torch.Tensor,
    vision_patch_count: torch.Tensor,
    min_parts: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    groups = object_groups(part_names)
    rows: List[Dict[str, Any]] = []

    for obj, indices in groups.items():
        valid_indices = [
            i for i in indices
            if text_count[i].item() > 0 and vision_patch_count[i].item() > 0
        ]
        if len(valid_indices) < min_parts:
            rows.append({
                "text_feature": text_name,
                "object": obj,
                "num_parts": len(indices),
                "valid_parts": len(valid_indices),
                "spearman": float("nan"),
                "str_t2v": float("nan"),
                "str_v2t": float("nan"),
                "str_hm_t2v": float("nan"),
                "str_hm_v2t": float("nan"),
            })
            continue

        idx = torch.tensor(valid_indices, dtype=torch.long)
        t = text_proto.index_select(0, idx)
        v = vision_proto.index_select(0, idx)

        rows.append({
            "text_feature": text_name,
            "object": obj,
            "num_parts": len(indices),
            "valid_parts": len(valid_indices),
            "spearman": spearman_structure(t, v),
            "str_t2v": structure_retrieval_core(t, v, use_hm=False),
            "str_v2t": structure_retrieval_core(v, t, use_hm=False),
            "str_hm_t2v": structure_retrieval_core(t, v, use_hm=True),
            "str_hm_v2t": structure_retrieval_core(v, t, use_hm=True),
        })

    valid_rows = [r for r in rows if not np.isnan(r["spearman"])]
    summary = {
        "num_valid_objects": float(len(valid_rows)),
        "mean_spearman": float(np.mean([r["spearman"] for r in valid_rows])) if valid_rows else float("nan"),
        "mean_str_t2v": float(np.mean([r["str_t2v"] for r in valid_rows])) if valid_rows else float("nan"),
        "mean_str_v2t": float(np.mean([r["str_v2t"] for r in valid_rows])) if valid_rows else float("nan"),
        "mean_str_hm_t2v": float(np.mean([r["str_hm_t2v"] for r in valid_rows])) if valid_rows else float("nan"),
        "mean_str_hm_v2t": float(np.mean([r["str_hm_v2t"] for r in valid_rows])) if valid_rows else float("nan"),
    }
    return rows, summary


def fmt_float(x: float) -> str:
    if x is None or np.isnan(x):
        return "nan"
    return f"{x:.6f}"


def write_outputs(result: Dict[str, Any], all_rows: List[Dict[str, Any]], summaries: Dict[str, Dict[str, float]], args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    txt_path = os.path.join(args.out_dir, "eval_part_spearman_str.txt")
    csv_path = os.path.join(args.out_dir, "eval_part_spearman_str.csv")
    support_path = os.path.join(args.out_dir, "part_support.csv")

    lines: List[str] = []
    lines.append("==================== Eval Part Spearman / STR ====================\n")
    lines.append(f"input_pth: {args.input_pth}\n")
    lines.append(f"out_dir: {args.out_dir}\n")
    lines.append(f"clip_part_key: {args.clip_part_key}\n")
    lines.append(f"llama_part_key: {args.llama_part_key}\n")
    lines.append(f"patch_token_key: {args.patch_token_key}\n")
    lines.append(f"crop_box_key: {args.crop_box_key}\n")
    lines.append(f"obj_seg_key: {args.obj_seg_key}\n")
    lines.append(f"derived part mask: replace annotations_detectron2_obj -> annotations_detectron2_part\n")
    lines.append(f"min_parts: {args.min_parts}\n")
    lines.append("\n[build stats]\n")
    for k, v in result["stats"].items():
        lines.append(f"{k}: {v}\n")

    for text_name, summary in summaries.items():
        lines.append(f"\n==================== {text_name} vs GT patch prototype ====================\n")
        lines.append(
            "Object               | #Parts | Valid | Spearman | STR T->V | STR V->T | HM T->V | HM V->T\n"
        )
        lines.append("-" * 100 + "\n")
        for r in [x for x in all_rows if x["text_feature"] == text_name]:
            lines.append(
                f"{r['object']:<20} | {int(r['num_parts']):<6} | {int(r['valid_parts']):<5} | "
                f"{fmt_float(r['spearman']):>8} | {fmt_float(r['str_t2v']):>8} | "
                f"{fmt_float(r['str_v2t']):>8} | {fmt_float(r['str_hm_t2v']):>7} | {fmt_float(r['str_hm_v2t']):>7}\n"
            )
        lines.append("-" * 100 + "\n")
        lines.append(
            f"Average over valid objects ({int(summary['num_valid_objects'])}): "
            f"spearman={fmt_float(summary['mean_spearman'])}, "
            f"str_t2v={fmt_float(summary['mean_str_t2v'])}, "
            f"str_v2t={fmt_float(summary['mean_str_v2t'])}, "
            f"str_hm_t2v={fmt_float(summary['mean_str_hm_t2v'])}, "
            f"str_hm_v2t={fmt_float(summary['mean_str_hm_v2t'])}\n"
        )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    fieldnames = [
        "text_feature", "object", "num_parts", "valid_parts", "spearman",
        "str_t2v", "str_v2t", "str_hm_t2v", "str_hm_v2t",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    part_names = result["part_names"]
    with open(support_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["part_id", "part_name", "object", "text_count", "vision_instance_count", "vision_patch_count"])
        for i, name in enumerate(part_names):
            writer.writerow([
                i,
                name,
                parse_object_name(name),
                int(result["text_count"][i].item()),
                int(result["vision_instance_count"][i].item()),
                int(result["vision_patch_count"][i].item()),
            ])

    print("\n".join(lines[-18:]))
    print(f"[OK] wrote txt:     {txt_path}")
    print(f"[OK] wrote csv:     {csv_path}")
    print(f"[OK] wrote support: {support_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Evaluate CLIP/Llama3 part T features vs GT patch-token visual prototypes.")
    p.add_argument("--input_pth", type=str, default=DEFAULT_INPUT_PTH)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    p.add_argument("--clip_part_key", type=str, default="part_ann_feats")
    p.add_argument("--llama_part_key", type=str, default="llama_part_ann_feats")
    p.add_argument("--patch_token_key", type=str, default="cropaug_patch_tokens")
    p.add_argument("--crop_box_key", type=str, default="cropaug_box_xyxy")

    p.add_argument("--image_id_key", type=str, default="image_id")
    p.add_argument("--obj_seg_key", type=str, default="seg_file_name")
    p.add_argument("--part_seg_key", type=str, default="part_seg_file_name")
    p.add_argument("--part_id_key", type=str, default="part_category_id")
    p.add_argument("--part_name_key", type=str, default="part_class_name")

    p.add_argument("--expected_num_parts", type=int, default=116)
    p.add_argument("--min_parts", type=int, default=3)
    return p


def main() -> None:
    args = build_parser().parse_args()
    print(f"[VERSION] {SCRIPT_VERSION}")
    print("[CONFIG]")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    result = build_prototypes_from_pth(args)

    all_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, float]] = {}
    for text_name, proto in [
        ("clip_part_ann_feats", result["clip_proto"]),
        ("llama_part_ann_feats", result["llama_proto"]),
    ]:
        rows, summary = evaluate_one_text(
            text_name=text_name,
            text_proto=proto,
            vision_proto=result["vision_proto"],
            part_names=result["part_names"],
            text_count=result["text_count"],
            vision_patch_count=result["vision_patch_count"],
            min_parts=args.min_parts,
        )
        all_rows.extend(rows)
        summaries[text_name] = summary

    write_outputs(result, all_rows, summaries, args)


if __name__ == "__main__":
    main()
