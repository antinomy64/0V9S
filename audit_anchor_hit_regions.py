#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import yaml
except Exception:
    yaml = None

try:
    from PIL import Image
except Exception:
    Image = None


def load_pth(path: str) -> Any:
    return torch.load(path, map_location="cpu")


def get_annotations(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and "annotations" in obj:
        return obj["annotations"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported annotation container format.")


def get_images(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and "images" in obj:
        return obj["images"]
    return []


def find_first_key(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def resolve_path(path_str: str, path_prefix: Optional[str]) -> str:
    if os.path.exists(path_str):
        return path_str
    if path_prefix is not None:
        candidate = os.path.join(path_prefix, path_str)
        if os.path.exists(candidate):
            return candidate
    return path_str


def resolve_seg_path(feat_ann, meta_ann, feat_images_by_id, meta_images_by_id, path_prefix):
    for source in [feat_ann, meta_ann]:
        k = find_first_key(source, ["seg_path", "seg_file_name", "segmentation_path"])
        if k is not None:
            return resolve_path(str(source[k]), path_prefix)

    image_id = feat_ann.get("image_id", meta_ann.get("image_id", None))
    if image_id is not None:
        for images_by_id in [feat_images_by_id, meta_images_by_id]:
            if image_id in images_by_id:
                img = images_by_id[image_id]
                k = find_first_key(img, ["seg_file_name", "seg_path", "segmentation_path"])
                if k is not None:
                    return resolve_path(str(img[k]), path_prefix)
    return None


def resolve_part_seg_path(obj_seg_path: Optional[str], feat_ann, meta_ann, path_prefix: Optional[str]) -> Optional[str]:
    for source in [feat_ann, meta_ann]:
        k = find_first_key(source, ["part_seg_path", "part_seg_file_name", "part_segmentation_path"])
        if k is not None:
            return resolve_path(str(source[k]), path_prefix)

    if obj_seg_path is None:
        return None

    candidate = obj_seg_path.replace("annotations_detectron2_obj", "annotations_detectron2_part")
    return resolve_path(candidate, path_prefix)


def resolve_crop_box(feature_ann: Dict[str, Any], meta_ann: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    candidates = [
        ("cropaug_box_xyxy", feature_ann),
        ("crop_box_xyxy", feature_ann),
        ("obj_box_xyxy", feature_ann),
        ("bbox_xyxy", feature_ann),
        ("cropaug_box_xyxy", meta_ann),
        ("crop_box_xyxy", meta_ann),
        ("obj_box_xyxy", meta_ann),
        ("bbox_xyxy", meta_ann),
        ("bbox", meta_ann),
    ]
    for key, source in candidates:
        if key in source:
            v = source[key]
            if torch.is_tensor(v):
                v = v.tolist()
            if isinstance(v, (list, tuple)) and len(v) >= 4:
                x1, y1, x2, y2 = map(int, v[:4])
                return x1, y1, x2, y2
    return None


def resolve_patch_tokens(feature_ann: Dict[str, Any]) -> Tuple[torch.Tensor, str]:
    for key in ["cropaug_patch_tokens", "patch_tokens"]:
        if key in feature_ann:
            x = feature_ann[key]
            if not torch.is_tensor(x):
                raise ValueError(f"{key} exists but is not a tensor.")
            return x.float(), key
    raise KeyError("Could not find cropaug_patch_tokens or patch_tokens.")


def resolve_part_text_feat(feature_ann: Dict[str, Any]) -> torch.Tensor:
    for key in ["part_ann_feats", "part_text_feat"]:
        if key in feature_ann:
            x = feature_ann[key]
            if not torch.is_tensor(x):
                raise ValueError(f"{key} exists but is not a tensor.")
            x = x.float()
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return x
    raise KeyError("Could not find part_ann_feats or part_text_feat in feature annotation.")


def resolve_part_ids(feature_ann: Dict[str, Any], meta_ann: Dict[str, Any], num_parts: int) -> List[int]:
    for source in [feature_ann, meta_ann]:
        key = find_first_key(source, ["part_category_id", "part_category_ids", "part_ids"])
        if key is not None:
            ids = source[key]
            if torch.is_tensor(ids):
                ids = ids.tolist()
            if isinstance(ids, tuple):
                ids = list(ids)
            if isinstance(ids, list):
                return [int(x) for x in ids[:num_parts]]
    return list(range(num_parts))


def load_palette_module(py_path: str):
    spec = importlib.util.spec_from_file_location("palette_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def resolve_classes(palette_py: Optional[str]):
    if palette_py is None:
        return []
    mod = load_palette_module(palette_py)
    preferred_names = ["PascalPart116_PART", "PascalPart116Part", "PascalPart116"]
    for name in preferred_names:
        if hasattr(mod, name):
            cls = getattr(mod, name)
            if isinstance(cls, type):
                classes = getattr(cls, "CLASSES", None)
                if classes is not None:
                    return list(classes)

    candidates = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if not isinstance(obj, type):
            continue
        classes = getattr(obj, "CLASSES", None)
        if classes is None:
            continue
        try:
            candidates.append((len(classes), obj))
        except TypeError:
            continue
    if not candidates:
        return []
    _, dataset_cls = max(candidates, key=lambda x: x[0])
    return list(dataset_cls.CLASSES)


def resolve_part_names(feature_ann: Dict[str, Any], meta_ann: Dict[str, Any], classes: List[str], part_ids: List[int], num_parts: int) -> List[str]:
    name_keys = [
        "part_names", "part_name_list", "part_category_names", "part_categories",
        "part_labels", "part_texts", "part_captions", "part_class_name"
    ]
    for source in [feature_ann, meta_ann]:
        key = find_first_key(source, name_keys)
        if key is not None:
            names = source[key]
            if torch.is_tensor(names):
                names = names.tolist()
            if isinstance(names, tuple):
                names = list(names)
            if isinstance(names, list):
                names = [str(x) for x in names]
                if len(names) >= num_parts:
                    return names[:num_parts]

    names = []
    for pid in part_ids[:num_parts]:
        if 0 <= pid < len(classes):
            names.append(classes[pid])
        elif 1 <= pid <= len(classes):
            names.append(classes[pid - 1])
        else:
            names.append(f"part_id_{pid}")
    return names


def unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise ValueError("Unsupported checkpoint format.")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def parse_act(name: str):
    if name is None or str(name).lower() == "none":
        return None
    name = str(name).lower()
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    raise ValueError(f"Unknown act: {name}")


def infer_model_type(state_dict: Dict[str, torch.Tensor]) -> str:
    if any(k.startswith("visual_linear") or k.startswith("visual_hidden_layers") for k in state_dict.keys()):
        return "doublemlp"
    return "projection"


def infer_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]):
    if "linear_layer.weight" not in state_dict:
        raise KeyError("Checkpoint missing linear_layer.weight; cannot infer dims.")
    dino_embed_dim, clip_embed_dim = state_dict["linear_layer.weight"].shape
    hidden_layers = sorted(
        set(
            int(k.split(".")[1])
            for k in state_dict.keys()
            if k.startswith("hidden_layers.") and k.endswith(".weight")
        )
    )
    hidden_layer = len(hidden_layers) if len(hidden_layers) > 0 else False
    return dino_embed_dim, clip_embed_dim, hidden_layer


def build_model(repo_root: str, ckpt_path: str, config_path: Optional[str], act: Optional[str]):
    sys.path.insert(0, repo_root)
    from src.model import ProjectionLayer, DoubleMLP

    ckpt = load_pth(ckpt_path)
    state_dict = strip_module_prefix(unwrap_state_dict(ckpt))
    model_type = infer_model_type(state_dict)

    if config_path is not None:
        if yaml is None:
            raise RuntimeError("pyyaml is not installed but config_path was provided.")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)["model"]
        dino_embed_dim = cfg.get("dino_embed_dim", 768)
        clip_embed_dim = cfg.get("clip_embed_dim", 512)
        hidden_layer = cfg.get("hidden_layer", False)
        act_name = cfg.get("act", "tanh")
    else:
        dino_embed_dim, clip_embed_dim, hidden_layer = infer_dims_from_state_dict(state_dict)
        act_name = act or "tanh"

    act_module = parse_act(act_name)
    if model_type == "doublemlp":
        model = DoubleMLP(
            act=act_module,
            hidden_layer=hidden_layer,
            cosine=True,
            dino_embed_dim=dino_embed_dim,
            clip_embed_dim=clip_embed_dim,
        )
    else:
        model = ProjectionLayer(
            act=act_module,
            hidden_layer=hidden_layer,
            cosine=True,
            dino_embed_dim=dino_embed_dim,
            clip_embed_dim=clip_embed_dim,
        )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def read_mask(path: str) -> np.ndarray:
    if Image is None:
        raise RuntimeError("Pillow is not installed.")
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def build_obj_mask_patch(seg_path: str, category_id: int, crop_box_xyxy, patch_tokens_key: str, grid_size: int, with_background: bool):
    mask = read_mask(seg_path)
    mask_value = category_id + 1 if with_background else category_id
    binary = (mask == mask_value).astype(np.uint8) * 255
    pil_mask = Image.fromarray(binary)

    if patch_tokens_key == "cropaug_patch_tokens" and crop_box_xyxy is not None:
        x1, y1, x2, y2 = crop_box_xyxy
        pil_mask = pil_mask.crop((int(x1), int(y1), int(x2), int(y2)))
        pil_mask = pil_mask.resize((grid_size, grid_size), resample=Image.NEAREST)
    else:
        pil_mask = pil_mask.resize((448, 448), resample=Image.NEAREST)
        pil_mask = pil_mask.crop((0, 0, 448, 448))
        pil_mask = pil_mask.resize((grid_size, grid_size), resample=Image.NEAREST)

    return torch.from_numpy((np.array(pil_mask) > 0).astype(np.bool_)).view(-1)


def build_part_label_patch(part_seg_path: str, crop_box_xyxy, patch_tokens_key: str, grid_size: int):
    mask = read_mask(part_seg_path)
    pil_mask = Image.fromarray(mask.astype(np.int32), mode="I")

    if patch_tokens_key == "cropaug_patch_tokens" and crop_box_xyxy is not None:
        x1, y1, x2, y2 = crop_box_xyxy
        pil_mask = pil_mask.crop((int(x1), int(y1), int(x2), int(y2)))
        pil_mask = pil_mask.resize((grid_size, grid_size), resample=Image.NEAREST)
    else:
        pil_mask = pil_mask.resize((448, 448), resample=Image.NEAREST)
        pil_mask = pil_mask.crop((0, 0, 448, 448))
        pil_mask = pil_mask.resize((grid_size, grid_size), resample=Image.NEAREST)

    return torch.from_numpy(np.array(pil_mask).astype(np.int64)).view(-1)


def compute_relative_scores(local_scores: torch.Tensor) -> torch.Tensor:
    Kb, Mb = local_scores.shape
    if Kb <= 1:
        return local_scores
    top2_vals, top2_idx = torch.topk(local_scores, k=min(2, Kb), dim=0)
    best_vals = top2_vals[0]
    best_idx = top2_idx[0]
    second_vals = top2_vals[1]
    row_ids = torch.arange(Kb, device=local_scores.device)[:, None]
    is_top1 = row_ids == best_idx[None, :]
    best_other = torch.where(is_top1, second_vals[None, :], best_vals[None, :])
    return local_scores - best_other


@torch.no_grad()
def debug_anchor_mapping(model, patch_tokens, part_text_feat, obj_mask_patch, patch_temperature):
    patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
    part_proj = model.project_clip_txt(part_text_feat.float())
    part_proj = F.normalize(part_proj, dim=-1)

    logits = torch.einsum("kd,nd->kn", part_proj, patch_tokens) / patch_temperature
    logits = logits.masked_fill(~obj_mask_patch[None, :], -1e4)

    valid_patch_tokens = patch_tokens[obj_mask_patch]
    local_scores = logits[:, obj_mask_patch]

    Kb, Mb = local_scores.shape
    if Kb == 0 or Mb == 0:
        return None

    rel_scores = compute_relative_scores(local_scores)
    flat_scores = rel_scores.reshape(-1)
    sorted_idx = torch.argsort(flat_scores, descending=True)

    anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=local_scores.device)
    patch_taken = torch.zeros((Mb,), dtype=torch.bool, device=local_scores.device)

    assigned_parts = 0
    for flat_id in sorted_idx:
        p_local = torch.div(flat_id, Mb, rounding_mode="floor")
        n_local = flat_id % Mb
        if anchor_idx_local[p_local] != -1:
            continue
        if patch_taken[n_local]:
            continue
        anchor_idx_local[p_local] = n_local
        patch_taken[n_local] = True
        assigned_parts += 1
        if assigned_parts == Kb:
            break

    unassigned = torch.nonzero(anchor_idx_local < 0, as_tuple=False).squeeze(1)
    if unassigned.numel() > 0:
        local_best = rel_scores.argmax(dim=1)
        anchor_idx_local[unassigned] = local_best[unassigned]

    anchor_feats = valid_patch_tokens[anchor_idx_local]
    region_scores = valid_patch_tokens @ anchor_feats.T
    region_assign_local = region_scores.argmax(dim=1)
    region_assign_local[anchor_idx_local] = torch.arange(Kb, device=region_assign_local.device)

    valid_patch_idx_global = torch.nonzero(obj_mask_patch, as_tuple=False).squeeze(1)
    anchor_idx_global = valid_patch_idx_global[anchor_idx_local]

    return {
        "anchor_idx_local": anchor_idx_local.cpu(),
        "anchor_idx_global": anchor_idx_global.cpu(),
        "region_assign_local": region_assign_local.cpu(),
        "valid_patch_idx_global": valid_patch_idx_global.cpu(),
        "rel_scores": rel_scores.cpu(),
        "raw_scores": local_scores.cpu(),
    }


def label_to_name(label: int, id_to_name: Dict[int, str], classes: List[str]) -> str:
    label = int(label)
    if label in id_to_name:
        return id_to_name[label]
    if label == 0:
        return "background_or_unlabeled"
    if 0 <= label < len(classes):
        return classes[label]
    if 1 <= label <= len(classes):
        return classes[label - 1]
    return f"label_{label}"


def safe_counter_major(labels: List[int]) -> Tuple[int, int, float]:
    if len(labels) == 0:
        return -1, 0, 0.0
    c = Counter([int(x) for x in labels])
    lab, cnt = c.most_common(1)[0]
    return lab, cnt, cnt / max(len(labels), 1)


def process_one(ann_idx, feat_ann, meta_ann, feat_images_by_id, meta_images_by_id, model, classes, args):
    patch_tokens, patch_tokens_key = resolve_patch_tokens(feat_ann)
    part_text_feat = resolve_part_text_feat(feat_ann)
    num_parts = int(part_text_feat.shape[0])
    if num_parts == 0:
        return [], None

    part_ids = resolve_part_ids(feat_ann, meta_ann, num_parts)
    part_names = resolve_part_names(feat_ann, meta_ann, classes, part_ids, num_parts)
    id_to_name = {int(pid): str(name) for pid, name in zip(part_ids, part_names)}

    crop_box = resolve_crop_box(feat_ann, meta_ann)
    obj_seg_path = resolve_seg_path(feat_ann, meta_ann, feat_images_by_id, meta_images_by_id, args.path_prefix)
    if obj_seg_path is None:
        raise ValueError(f"Cannot resolve object seg path for ann_idx={ann_idx}")
    part_seg_path = resolve_part_seg_path(obj_seg_path, feat_ann, meta_ann, args.path_prefix)
    if part_seg_path is None or not os.path.exists(part_seg_path):
        raise ValueError(f"Cannot resolve part seg path for ann_idx={ann_idx}; got {part_seg_path}")

    category_id = int(feat_ann.get("category_id", meta_ann.get("category_id")))
    N = int(patch_tokens.shape[0])
    grid_size = int(round(math.sqrt(N)))
    if grid_size * grid_size != N:
        raise ValueError(f"Patch token count is not square: N={N}")

    obj_mask_patch = build_obj_mask_patch(
        seg_path=obj_seg_path,
        category_id=category_id,
        crop_box_xyxy=crop_box,
        patch_tokens_key=patch_tokens_key,
        grid_size=grid_size,
        with_background=args.with_background,
    )
    part_label_patch = build_part_label_patch(
        part_seg_path=part_seg_path,
        crop_box_xyxy=crop_box,
        patch_tokens_key=patch_tokens_key,
        grid_size=grid_size,
    )

    debug = debug_anchor_mapping(
        model=model,
        patch_tokens=patch_tokens,
        part_text_feat=part_text_feat,
        obj_mask_patch=obj_mask_patch,
        patch_temperature=args.patch_temperature,
    )
    if debug is None:
        return [], None

    rows = []
    valid_patch_idx_global = debug["valid_patch_idx_global"]
    region_assign_local = debug["region_assign_local"]
    anchor_idx_global = debug["anchor_idx_global"]

    for k in range(num_parts):
        pred_part_id = int(part_ids[k]) if k < len(part_ids) else k
        pred_part_name = str(part_names[k])
        anchor_patch_id = int(anchor_idx_global[k].item())
        anchor_row = int(anchor_patch_id // grid_size)
        anchor_col = int(anchor_patch_id % grid_size)

        anchor_gt_id = int(part_label_patch[anchor_patch_id].item())
        anchor_gt_name = label_to_name(anchor_gt_id, id_to_name, classes)

        region_local_idx = torch.nonzero(region_assign_local == k, as_tuple=False).squeeze(1)
        region_global_idx = valid_patch_idx_global[region_local_idx]
        region_labels = part_label_patch[region_global_idx].tolist()
        region_major_id, region_major_count, region_major_ratio = safe_counter_major(region_labels)
        region_major_name = label_to_name(region_major_id, id_to_name, classes)

        pred_region_mask = torch.zeros((N,), dtype=torch.bool)
        pred_region_mask[region_global_idx] = True

        best_iou = -1.0
        best_iou_id = -1
        for pid in part_ids:
            gt_mask = part_label_patch == int(pid)
            union = (pred_region_mask | gt_mask).sum().item()
            if union <= 0:
                continue
            inter = (pred_region_mask & gt_mask).sum().item()
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_iou_id = int(pid)

        best_iou_name = label_to_name(best_iou_id, id_to_name, classes) if best_iou_id >= 0 else "none"

        rows.append({
            "ann_idx": int(ann_idx),
            "annotation_id": feat_ann.get("id", ""),
            "image_id": feat_ann.get("image_id", meta_ann.get("image_id", "")),
            "class_name": feat_ann.get("class_name", meta_ann.get("class_name", "")),
            "category_id": category_id,
            "pred_part_slot": int(k),
            "pred_part_id": pred_part_id,
            "pred_part_name": pred_part_name,
            "anchor_patch_id": anchor_patch_id,
            "anchor_row": anchor_row,
            "anchor_col": anchor_col,
            "anchor_gt_part_id": anchor_gt_id,
            "anchor_gt_part_name": anchor_gt_name,
            "anchor_text_match": int(anchor_gt_id == pred_part_id),
            "region_size": int(region_global_idx.numel()),
            "region_major_gt_part_id": int(region_major_id),
            "region_major_gt_part_name": region_major_name,
            "region_major_ratio": float(region_major_ratio),
            "region_major_text_match": int(region_major_id == pred_part_id),
            "best_iou_gt_part_id": int(best_iou_id),
            "best_iou_gt_part_name": best_iou_name,
            "best_iou": float(max(best_iou, 0.0)),
        })

    return rows, {
        "ann_idx": int(ann_idx),
        "num_parts": num_parts,
        "num_obj_patches": int(obj_mask_patch.sum().item()),
    }


def write_csv(path: Path, rows: List[Dict]):
    if len(rows) == 0:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_long_counter(path: Path, counter: Counter, fields: Tuple[str, str, str]):
    rows = []
    a, b, c = fields
    for (x, y), count in counter.most_common():
        rows.append({a: x, b: y, c: int(count)})
    write_csv(path, rows)


def main():
    parser = argparse.ArgumentParser(description="Audit which GT part regions each text anchor / pseudo region hits.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--feature_pth", required=True)
    parser.add_argument("--meta_pth", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--act", default="tanh")
    parser.add_argument("--palette_py", default=None)
    parser.add_argument("--patch_temperature", type=float, default=0.07)
    parser.add_argument("--out_dir", default="anchor_hit_region_audit")
    parser.add_argument("--ann_idx", type=int, default=None, help="Optional single ann idx. Omit to process all.")
    parser.add_argument("--max_anns", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--path_prefix", type=str, default=None)
    parser.add_argument("--with_background", action="store_true", default=False)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_obj = load_pth(args.feature_pth)
    meta_obj = load_pth(args.meta_pth)
    feat_anns = get_annotations(feature_obj)
    meta_anns = get_annotations(meta_obj)
    if len(meta_anns) < len(feat_anns):
        raise ValueError(f"meta annotations fewer than feature annotations: {len(meta_anns)} < {len(feat_anns)}")

    feat_images_by_id = {img["id"]: img for img in get_images(feature_obj) if "id" in img}
    meta_images_by_id = {img["id"]: img for img in get_images(meta_obj) if "id" in img}

    classes = resolve_classes(args.palette_py)
    model = build_model(args.repo_root, args.ckpt, args.config, args.act)

    if args.ann_idx is None:
        indices = list(range(len(feat_anns)))
    else:
        indices = [args.ann_idx]
    if args.max_anns is not None:
        indices = indices[:args.max_anns]

    all_rows = []
    ann_summaries = []
    errors = []

    for i, ann_idx in enumerate(indices):
        try:
            rows, summary = process_one(
                ann_idx=ann_idx,
                feat_ann=feat_anns[ann_idx],
                meta_ann=meta_anns[ann_idx],
                feat_images_by_id=feat_images_by_id,
                meta_images_by_id=meta_images_by_id,
                model=model,
                classes=classes,
                args=args,
            )
            all_rows.extend(rows)
            if summary is not None:
                ann_summaries.append(summary)
            if (i + 1) % 100 == 0:
                print(f"processed {i + 1}/{len(indices)} anns, rows={len(all_rows)}")
        except Exception as e:
            errors.append({"ann_idx": int(ann_idx), "error": repr(e)})
            print(f"[WARN] ann_idx={ann_idx} failed: {e}")

    write_csv(out_dir / "anchor_hit_per_part.csv", all_rows)

    anchor_conf = Counter()
    region_major_conf = Counter()
    best_iou_conf = Counter()
    class_anchor_conf = Counter()

    for r in all_rows:
        pred = r["pred_part_name"]
        cls = r["class_name"]
        anchor_gt = r["anchor_gt_part_name"]
        region_major = r["region_major_gt_part_name"]
        best_iou = r["best_iou_gt_part_name"]
        anchor_conf[(pred, anchor_gt)] += 1
        region_major_conf[(pred, region_major)] += 1
        best_iou_conf[(pred, best_iou)] += 1
        class_anchor_conf[(cls + " :: " + pred, anchor_gt)] += 1

    write_long_counter(
        out_dir / "confusion_anchor_text_to_anchor_gt.csv",
        anchor_conf,
        ("pred_text_part", "anchor_gt_part", "count"),
    )
    write_long_counter(
        out_dir / "confusion_region_text_to_region_major_gt.csv",
        region_major_conf,
        ("pred_text_part", "region_major_gt_part", "count"),
    )
    write_long_counter(
        out_dir / "confusion_region_text_to_best_iou_gt.csv",
        best_iou_conf,
        ("pred_text_part", "best_iou_gt_part", "count"),
    )
    write_long_counter(
        out_dir / "confusion_class_text_to_anchor_gt.csv",
        class_anchor_conf,
        ("class_and_pred_text_part", "anchor_gt_part", "count"),
    )

    total = len(all_rows)
    anchor_match = sum(int(r["anchor_text_match"]) for r in all_rows)
    region_major_match = sum(int(r["region_major_text_match"]) for r in all_rows)
    avg_best_iou = sum(float(r["best_iou"]) for r in all_rows) / max(total, 1)

    summary = {
        "num_anchor_rows": total,
        "num_ann_processed": len(ann_summaries),
        "num_errors": len(errors),
        "anchor_text_match_rate": anchor_match / max(total, 1),
        "region_major_text_match_rate": region_major_match / max(total, 1),
        "avg_best_iou": avg_best_iou,
        "errors": errors[:50],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "ann_summary.json", "w", encoding="utf-8") as f:
        json.dump(ann_summaries, f, ensure_ascii=False, indent=2)
    with open(out_dir / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved audit files to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
