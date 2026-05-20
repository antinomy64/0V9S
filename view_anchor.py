#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import sys
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
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


def load_pth(path: str) -> Any:
    return torch.load(path, map_location="cpu")


def unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise ValueError("Unsupported checkpoint format.")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out


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


def resolve_image_path(meta_ann: Dict[str, Any], meta_images_by_id: Dict[Any, Dict[str, Any]], path_prefix: Optional[str]) -> Optional[str]:
    direct_keys = ["crop_image_path", "crop_path", "image_path", "file_name", "filepath", "img_path"]
    k = find_first_key(meta_ann, direct_keys)
    if k is not None:
        return resolve_path(str(meta_ann[k]), path_prefix)

    image_id_key = find_first_key(meta_ann, ["image_id", "img_id", "id_image"])
    if image_id_key is None:
        return None
    image_id = meta_ann[image_id_key]
    if image_id not in meta_images_by_id:
        return None
    img = meta_images_by_id[image_id]
    path_key = find_first_key(img, ["file_name", "image_path", "filepath", "img_path"])
    if path_key is None:
        return None
    return resolve_path(str(img[path_key]), path_prefix)


def resolve_seg_path(feat_ann: Dict[str, Any], meta_ann: Dict[str, Any], feat_images_by_id: Dict[Any, Dict[str, Any]], meta_images_by_id: Dict[Any, Dict[str, Any]], path_prefix: Optional[str]) -> Optional[str]:
    for source in [feat_ann, meta_ann]:
        k = find_first_key(source, ["seg_path", "seg_file_name", "segmentation_path"])
        if k is not None:
            return resolve_path(str(source[k]), path_prefix)

    image_id = feat_ann.get("image_id", meta_ann.get("image_id", None))
    if image_id is not None:
        if image_id in feat_images_by_id:
            img = feat_images_by_id[image_id]
            k = find_first_key(img, ["seg_file_name", "seg_path", "segmentation_path"])
            if k is not None:
                return resolve_path(str(img[k]), path_prefix)
        if image_id in meta_images_by_id:
            img = meta_images_by_id[image_id]
            k = find_first_key(img, ["seg_file_name", "seg_path", "segmentation_path"])
            if k is not None:
                return resolve_path(str(img[k]), path_prefix)
    return None


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


def load_palette_module(py_path: str):
    spec = importlib.util.spec_from_file_location("palette_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def resolve_classes_palette(palette_py: str):
    mod = load_palette_module(palette_py)

    preferred_names = ["PascalPart116_PART", "PascalPart116Part", "PascalPart116"]
    for name in preferred_names:
        if hasattr(mod, name):
            cls = getattr(mod, name)
            if isinstance(cls, type):
                classes = getattr(cls, "CLASSES", None)
                palette = getattr(cls, "PALETTE", None)
                if classes is not None and palette is not None:
                    return list(classes), [tuple(map(int, c)) for c in palette]

    candidates = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if not isinstance(obj, type):
            continue
        classes = getattr(obj, "CLASSES", None)
        palette = getattr(obj, "PALETTE", None)
        if classes is None or palette is None:
            continue
        try:
            candidates.append((len(classes), obj))
        except TypeError:
            continue

    if not candidates:
        raise ValueError(f"Could not find a valid dataset class with non-empty CLASSES and PALETTE in {palette_py}")

    _, dataset_cls = max(candidates, key=lambda x: x[0])
    return list(dataset_cls.CLASSES), [tuple(map(int, c)) for c in dataset_cls.PALETTE]


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


def build_obj_mask_patch(seg_path: str, category_id: int, crop_box_xyxy: Optional[Tuple[int, int, int, int]], patch_tokens_key: str, grid_size: int, with_background: bool):
    if Image is None:
        raise RuntimeError("Pillow is not installed.")
    mask = np.array(Image.open(seg_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_value = category_id + 1 if with_background else category_id
    binary = (mask == mask_value).astype(np.uint8) * 255

    pil_mask = Image.fromarray(binary)
    if patch_tokens_key == "cropaug_patch_tokens" and crop_box_xyxy is not None:
        x1, y1, x2, y2 = crop_box_xyxy
        pil_mask = pil_mask.crop((x1, y1, x2, y2))
        pil_mask = pil_mask.resize((grid_size, grid_size), resample=Image.NEAREST)
    else:
        pil_mask = pil_mask.resize((448, 448), resample=Image.NEAREST)
        pil_mask = pil_mask.crop((0, 0, 448, 448))
        pil_mask = pil_mask.resize((grid_size, grid_size), resample=Image.NEAREST)

    return torch.from_numpy((np.array(pil_mask) > 0).astype(np.bool_)).view(-1)


def color_for_part(part_name: str, classes: List[str], palette: List[Tuple[int, int, int]], idx_fallback: int):
    if part_name in classes:
        return palette[classes.index(part_name)]
    return palette[idx_fallback % len(palette)]


def _text_box(draw, xy, text, font, fill_text=(255, 255, 255), fill_box=(0, 0, 0)):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad_x, pad_y = 3, 2
    rect = (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y)
    draw.rectangle(rect, fill=fill_box)
    draw.text((x, y), text, fill=fill_text, font=font)


def save_mask_with_numbers(
    img_path: str,
    crop_box: Optional[Tuple[int, int, int, int]],
    debug: Dict[str, Any],
    part_names: List[str],
    num_patches: int,
    classes: List[str],
    palette: List[Tuple[int, int, int]],
    out_dir: Path,
    ann_idx: int,
    alpha: int = 110,
):
    if Image is None:
        raise RuntimeError("Pillow is not installed.")

    img = Image.open(img_path).convert("RGB")
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        crop = img.crop((x1, y1, x2, y2))
    else:
        crop = img.copy()

    out_dir.mkdir(parents=True, exist_ok=True)

    grid_size = int(round(math.sqrt(num_patches)))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a square grid.")

    patch_w = crop.width / float(grid_size)
    patch_h = crop.height / float(grid_size)

    overlay_rgba = crop.convert("RGBA")
    mask_layer = Image.new("RGBA", overlay_rgba.size, (0, 0, 0, 0))

    anchor_colors = [color_for_part(name, classes, palette, i) for i, name in enumerate(part_names)]

    region_assign_local = debug["region_assign_local"].tolist()
    valid_patch_idx_global = debug["valid_patch_idx_global"].tolist()

    draw = ImageDraw.Draw(mask_layer)
    for local_i, patch_id in enumerate(valid_patch_idx_global):
        part_local = int(region_assign_local[local_i])
        color = anchor_colors[part_local]
        row = patch_id // grid_size
        col = patch_id % grid_size

        x1 = int(round(col * patch_w))
        y1 = int(round(row * patch_h))
        x2 = int(round((col + 1) * patch_w))
        y2 = int(round((row + 1) * patch_h))
        draw.rectangle((x1, y1, x2, y2), fill=(color[0], color[1], color[2], alpha))

    composed = Image.alpha_composite(overlay_rgba, mask_layer).convert("RGB")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    composed_draw = ImageDraw.Draw(composed)
    for i, patch_id in enumerate(debug["anchor_idx_global"].tolist()):
        row = patch_id // grid_size
        col = patch_id % grid_size
        cx = int(round(col * patch_w + patch_w / 2.0))
        cy = int(round(row * patch_h + patch_h / 2.0))
        _text_box(composed_draw, (cx - 5, cy - 7), str(i), font)

    if debug.get("obj_anchor_patch_global", None) is not None:
        patch_id = int(debug["obj_anchor_patch_global"])
        row = patch_id // grid_size
        col = patch_id % grid_size
        cx = int(round(col * patch_w + patch_w / 2.0))
        cy = int(round(row * patch_h + patch_h / 2.0))
        _text_box(composed_draw, (cx - 5, cy + 8), "O", font, fill_text=(255, 255, 255), fill_box=(220, 20, 60))

    legend_items = []
    for i, name in enumerate(part_names):
        patch_id = int(debug["anchor_idx_global"][i].item())
        row = int(debug["rows"][i])
        col = int(debug["cols"][i])
        legend_items.append({
            "idx": i,
            "name": name,
            "patch_id": patch_id,
            "row": row,
            "col": col,
            "color": anchor_colors[i],
        })

    legend_w = max(520, crop.width)
    legend_h = max(composed.height, 60 + 28 * len(legend_items) + 20)
    canvas = Image.new("RGB", (composed.width + legend_w, legend_h), (255, 255, 255))
    canvas.paste(composed, (0, 0))
    legend_draw = ImageDraw.Draw(canvas)

    legend_draw.text((composed.width + 12, 10), f"dataset_idx={ann_idx}", fill=(0, 0, 0), font=font)
    if debug.get("obj_anchor_patch_global", None) is not None:
        txt = f"OBJ anchor: p={int(debug['obj_anchor_patch_global'])} ({int(debug['obj_anchor_row'])},{int(debug['obj_anchor_col'])})"
        legend_draw.rectangle((composed.width + 12, 34, composed.width + 30, 52), fill=(220, 20, 60), outline=(0, 0, 0))
        legend_draw.text((composed.width + 38, 36), txt, fill=(0, 0, 0), font=font)

    y = 70
    for item in legend_items:
        c = item["color"]
        legend_draw.rectangle((composed.width + 12, y, composed.width + 30, y + 18), fill=(0, 0, 0))
        legend_draw.text((composed.width + 17, y + 2), str(item["idx"]), fill=(255, 255, 255), font=font)
        legend_draw.rectangle((composed.width + 38, y, composed.width + 56, y + 18), fill=c, outline=(0, 0, 0))
        txt = f"{item['name']}  p={item['patch_id']} ({item['row']},{item['col']})"
        legend_draw.text((composed.width + 64, y + 2), txt, fill=(0, 0, 0), font=font)
        y += 26

    out_path = out_dir / f"dataset_{ann_idx:06d}_mask_numbered_obj_anchor.png"
    canvas.save(out_path)
    return out_path


def build_debug_loss(repo_root: str, model, patch_temperature: float):
    sys.path.insert(0, repo_root)
    from src.loss_joint import JointObjPartLoss

    class JointObjPartLossDebug(JointObjPartLoss):
        @torch.no_grad()
        def debug_anchor_mapping(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
            part_text_feat = batch["part_text_feat"]
            obj_text_feat = batch["obj_text_feat"]
            patch_tokens = batch["patch_tokens"]
            obj_mask_patch = batch["obj_mask_patch"]
            part_valid_mask = batch["part_valid_mask"]
            part_gt_mask_patch = batch.get("part_gt_mask_patch", None)

            part_proj = self.sim_model.project_clip_txt(part_text_feat.float())
            obj_proj = self.sim_model.project_clip_txt(obj_text_feat.float())
            part_proj = self._safe_normalize(part_proj, dim=-1)
            obj_proj = self._safe_normalize(obj_proj, dim=-1)
            patch_tokens = self._safe_normalize(patch_tokens.float(), dim=-1)

            abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / self.patch_temperature
            abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

            obj_logits = torch.einsum("bd,bnd->bn", obj_proj, patch_tokens) / self.patch_temperature
            obj_logits = obj_logits.masked_fill(~obj_mask_patch, -1e4)

            B, K, N = abs_logits.shape
            results = []
            for b in range(B):
                valid_patch_mask = obj_mask_patch[b]
                valid_part_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)

                obj_valid_patch_idx_global = torch.nonzero(valid_patch_mask, as_tuple=False).squeeze(1)
                if obj_valid_patch_idx_global.numel() > 0:
                    obj_local_scores = obj_logits[b][valid_patch_mask]
                    obj_anchor_local = int(torch.argmax(obj_local_scores).item())
                    obj_anchor_patch_global = int(obj_valid_patch_idx_global[obj_anchor_local].item())
                    grid_size_obj = int(round(math.sqrt(int(N))))
                    if grid_size_obj * grid_size_obj != N:
                        obj_anchor_row, obj_anchor_col = -1, -1
                    else:
                        obj_anchor_row = int(obj_anchor_patch_global // grid_size_obj)
                        obj_anchor_col = int(obj_anchor_patch_global % grid_size_obj)
                else:
                    obj_anchor_patch_global = None
                    obj_anchor_row = -1
                    obj_anchor_col = -1

                if valid_part_idx.numel() == 0 or valid_patch_mask.sum() == 0:
                    results.append({
                        "sample_idx": b,
                        "valid_part_idx": valid_part_idx.cpu(),
                        "anchor_idx_local": torch.empty((0,), dtype=torch.long),
                        "anchor_idx_global": torch.empty((0,), dtype=torch.long),
                        "region_assign_local": torch.empty((0,), dtype=torch.long),
                        "valid_patch_idx_global": torch.empty((0,), dtype=torch.long),
                        "patch_count_per_part": [],
                        "anchor_hit_vec": [],
                        "anchor_hit_rate": 0.0,
                        "obj_anchor_patch_global": obj_anchor_patch_global,
                        "obj_anchor_row": obj_anchor_row,
                        "obj_anchor_col": obj_anchor_col,
                        "grid_size": None,
                        "rows": [],
                        "cols": [],
                    })
                    continue

                valid_patch_tokens = patch_tokens[b][valid_patch_mask]
                local_scores = abs_logits[b][valid_part_idx][:, valid_patch_mask]

                Kb, Mb = local_scores.shape
                if Mb == 0:
                    results.append({
                        "sample_idx": b,
                        "valid_part_idx": valid_part_idx.cpu(),
                        "anchor_idx_local": torch.empty((0,), dtype=torch.long),
                        "anchor_idx_global": torch.empty((0,), dtype=torch.long),
                        "region_assign_local": torch.empty((0,), dtype=torch.long),
                        "valid_patch_idx_global": torch.empty((0,), dtype=torch.long),
                        "patch_count_per_part": [],
                        "anchor_hit_vec": [],
                        "anchor_hit_rate": 0.0,
                        "obj_anchor_patch_global": obj_anchor_patch_global,
                        "obj_anchor_row": obj_anchor_row,
                        "obj_anchor_col": obj_anchor_col,
                        "grid_size": None,
                        "rows": [],
                        "cols": [],
                    })
                    continue

                rel_scores = self._compute_relative_scores(local_scores)
                flat_scores = rel_scores.reshape(-1)
                sorted_idx = torch.argsort(flat_scores, descending=True)

                anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=local_scores.device)
                patch_taken = torch.zeros((Mb,), dtype=torch.bool, device=local_scores.device)

                assigned_parts = 0
                for flat_id in sorted_idx:
                    p_local = torch.div(flat_id, Mb, rounding_mode='floor')
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

                valid_patch_idx_global = torch.nonzero(valid_patch_mask, as_tuple=False).squeeze(1)
                anchor_idx_global = valid_patch_idx_global[anchor_idx_local]

                hit_vec = []
                anchor_hit_rate = 0.0
                if part_gt_mask_patch is not None:
                    gt_masks = part_gt_mask_patch[b, valid_part_idx]
                    hit = gt_masks[torch.arange(Kb, device=gt_masks.device), anchor_idx_global]
                    hit_vec = hit.detach().cpu().int().tolist()
                    anchor_hit_rate = float(hit.float().mean().item())

                C = valid_patch_tokens[anchor_idx_local]
                assign = torch.zeros((Mb,), dtype=torch.long, device=local_scores.device)

                for _ in range(max(int(self.em_iters), 1)):
                    assign_scores = valid_patch_tokens @ C.T
                    assign = assign_scores.argmax(dim=1)
                    assign[anchor_idx_local] = torch.arange(Kb, device=assign.device)

                    onehot = F.one_hot(assign, num_classes=Kb).float()
                    count = onehot.sum(dim=0).clamp_min(1.0)
                    proto_sum = onehot.T @ valid_patch_tokens
                    C = proto_sum / count[:, None]
                    C = self._safe_normalize(C, dim=-1)

                patch_assignment_global = [-1] * int(N)
                for local_n, global_n in enumerate(valid_patch_idx_global.detach().cpu().tolist()):
                    patch_assignment_global[global_n] = int(assign[local_n].detach().cpu().item())

                onehot = F.one_hot(assign, num_classes=Kb).float()
                patch_count_per_part = onehot.sum(dim=0).detach().cpu().int().tolist()

                grid_size = int(round(math.sqrt(int(N))))
                if grid_size * grid_size != N:
                    grid_size = None

                rows, cols = [], []
                for patch_id in anchor_idx_global.tolist():
                    if grid_size is not None:
                        rows.append(int(patch_id // grid_size))
                        cols.append(int(patch_id % grid_size))
                    else:
                        rows.append(-1)
                        cols.append(-1)

                results.append({
                    "sample_idx": b,
                    "valid_part_idx": valid_part_idx.cpu(),
                    "anchor_idx_local": anchor_idx_local.cpu(),
                    "anchor_idx_global": anchor_idx_global.cpu(),
                    "region_assign_local": assign.cpu(),
                    "valid_patch_idx_global": valid_patch_idx_global.cpu(),
                    "patch_assignment_global": patch_assignment_global,
                    "patch_count_per_part": patch_count_per_part,
                    "anchor_hit_vec": hit_vec,
                    "anchor_hit_rate": anchor_hit_rate,
                    "obj_anchor_patch_global": obj_anchor_patch_global,
                    "obj_anchor_row": obj_anchor_row,
                    "obj_anchor_col": obj_anchor_col,
                    "grid_size": grid_size,
                    "rows": rows,
                    "cols": cols,
                })
            return results

    return JointObjPartLossDebug(model, patch_temperature=patch_temperature)


def process_one(
    dataset_idx: int,
    sample: Dict[str, Any],
    feat_ann: Dict[str, Any],
    meta_ann: Dict[str, Any],
    feat_images_by_id: Dict[Any, Dict[str, Any]],
    meta_images_by_id: Dict[Any, Dict[str, Any]],
    criterion,
    classes: List[str],
    palette: List[Tuple[int, int, int]],
    out_dir: Path,
    alpha: int,
    path_prefix: Optional[str],
    with_background: bool,
    collate_fn,
):
    patch_tokens_key = "cropaug_patch_tokens" if "cropaug_box_xyxy" in sample.get("metadata", {}) else "patch_tokens"
    batch = collate_fn([sample])
    debug = criterion.debug_anchor_mapping(batch)[0]

    part_ids = sample["part_category_id"].detach().cpu().tolist()
    part_names = [classes[int(pid)] if 0 <= int(pid) < len(classes) else f"part_id_{int(pid)}" for pid in part_ids]

    img_path = resolve_image_path(meta_ann, meta_images_by_id, path_prefix)
    crop_box = resolve_crop_box(feat_ann, meta_ann)

    seg_path = resolve_seg_path(feat_ann, meta_ann, feat_images_by_id, meta_images_by_id, path_prefix)
    if seg_path is None:
        raise ValueError(f"Cannot resolve seg path for dataset_idx={dataset_idx}")

    category_id = int(sample["category_id"].item()) if torch.is_tensor(sample["category_id"]) else int(sample["category_id"])
    N = int(sample["patch_tokens"].shape[0])

    grid_size = int(round(math.sqrt(N)))
    _ = build_obj_mask_patch(
        seg_path=seg_path,
        category_id=category_id,
        crop_box_xyxy=crop_box,
        patch_tokens_key=patch_tokens_key,
        grid_size=grid_size,
        with_background=with_background,
    )

    mask_path = None
    if img_path is not None and Path(img_path).exists():
        mask_path = save_mask_with_numbers(
            img_path=img_path,
            crop_box=crop_box,
            debug=debug,
            part_names=part_names,
            num_patches=N,
            classes=classes,
            palette=palette,
            out_dir=out_dir,
            ann_idx=dataset_idx,
            alpha=alpha,
        )

    anchor_colors = [color_for_part(name, classes, palette, i) for i, name in enumerate(part_names)]
    anchors = []
    for i, part_slot in enumerate(debug["valid_part_idx"].tolist()):
        anchors.append({
            "display_idx": int(i),
            "part_slot": int(part_slot),
            "part_name": str(part_names[i]),
            "patch_id": int(debug["anchor_idx_global"][i].item()),
            "row": int(debug["rows"][i]),
            "col": int(debug["cols"][i]),
            "hit": int(debug["anchor_hit_vec"][i]) if i < len(debug["anchor_hit_vec"]) else None,
            "color_rgb": list(map(int, anchor_colors[i])),
        })

    meta = sample.get("metadata", {})
    return {
        "dataset_idx": int(dataset_idx),
        "annotation_id": meta.get("annotation_id", feat_ann.get("id", None)),
        "image_id": meta.get("image_id", feat_ann.get("image_id", meta_ann.get("image_id", None))),
        "class_name": meta.get("class_name", feat_ann.get("class_name", "")),
        "mask_path": str(mask_path) if mask_path is not None else None,
        "anchor_hit_rate": float(debug["anchor_hit_rate"]),
        "obj_anchor_patch_global": debug["obj_anchor_patch_global"],
        "obj_anchor_row": debug["obj_anchor_row"],
        "obj_anchor_col": debug["obj_anchor_col"],
        "patch_count_per_part": debug["patch_count_per_part"],
        "anchors": anchors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize current pseudo labels and mark the highest-response object-text patch."
    )
    parser.add_argument("--repo_root", default=".", help="Project root that contains src/")
    parser.add_argument("--feature_pth", required=True, help="Path to train_voc116_obj_with_text.pth (or val_*.pth)")
    parser.add_argument("--meta_pth", required=True, help="Path to meta pth for image paths")
    parser.add_argument("--ckpt", required=True, help="Path to projector/model checkpoint")
    parser.add_argument("--config", default=None, help="Optional yaml to build model exactly")
    parser.add_argument("--act", default="tanh", help="Fallback act if --config is not provided")
    parser.add_argument("--patch_temperature", type=float, default=0.07)
    parser.add_argument("--palette_py", required=True, help="Python file that contains dataset CLASSES and PALETTE")
    parser.add_argument("--alpha", type=int, default=110, help="Mask alpha in [0,255]")
    parser.add_argument("--out_dir", default="anchor_mask_numbered_out_current_loss_obj_anchor")
    parser.add_argument("--sample_idx", type=int, default=0, help="Dataset sample start idx")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--path_prefix", type=str, default=None)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--min_obj_area_ratio", type=float, default=0.0)
    args = parser.parse_args()

    sys.path.insert(0, args.repo_root)
    from src.dataset_joint import DinoClipJointDataset, joint_collate_fn

    feature_obj = load_pth(args.feature_pth)
    meta_obj = load_pth(args.meta_pth)
    feat_anns = get_annotations(feature_obj)
    meta_anns = get_annotations(meta_obj)

    feat_ann_by_id = {ann.get("id"): ann for ann in feat_anns if "id" in ann}
    meta_ann_by_id = {ann.get("id"): ann for ann in meta_anns if "id" in ann}

    feat_images_by_id = {img["id"]: img for img in get_images(feature_obj) if "id" in img}
    meta_images_by_id = {img["id"]: img for img in get_images(meta_obj) if "id" in img}

    classes, palette = resolve_classes_palette(args.palette_py)
    model = build_model(args.repo_root, args.ckpt, args.config, args.act)
    criterion = build_debug_loss(args.repo_root, model, args.patch_temperature)

    is_wds = ".tar" in args.feature_pth
    dataset = DinoClipJointDataset(
        args.feature_pth,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=is_wds,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=args.min_obj_area_ratio,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = max(int(args.sample_idx), 0)
    end = min(start + max(int(args.num_samples), 1), len(dataset))

    results = []
    for dataset_idx in range(start, end):
        sample = dataset[dataset_idx]
        ann_id = sample.get("metadata", {}).get("annotation_id", None)

        feat_ann = feat_ann_by_id.get(ann_id, feat_anns[dataset_idx] if dataset_idx < len(feat_anns) else {})
        meta_ann = meta_ann_by_id.get(ann_id, meta_anns[dataset_idx] if dataset_idx < len(meta_anns) else {})

        res = process_one(
            dataset_idx=dataset_idx,
            sample=sample,
            feat_ann=feat_ann,
            meta_ann=meta_ann,
            feat_images_by_id=feat_images_by_id,
            meta_images_by_id=meta_images_by_id,
            criterion=criterion,
            classes=classes,
            palette=palette,
            out_dir=out_dir,
            alpha=args.alpha,
            path_prefix=args.path_prefix,
            with_background=args.with_background,
            collate_fn=joint_collate_fn,
        )
        results.append(res)
        print(f"[dataset_idx={dataset_idx}] saved mask -> {res['mask_path']}")

    summary_json = out_dir / "anchor_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary_jsonl = out_dir / "anchor_summary.jsonl"
    with open(summary_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved summary: {summary_json}")
    print(f"Saved summary: {summary_jsonl}")
    print(f"Processed {len(results)} dataset samples.")


if __name__ == "__main__":
    main()
