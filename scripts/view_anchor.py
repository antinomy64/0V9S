#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn
from src.loss_joint import JointObjPartLoss


def load_projector_from_config(model_config: str, init_weights: str, device: torch.device):
    with open(model_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"].get("model_class", "ProjectionLayer")
    Model = getattr(importlib.import_module("src.model"), model_name)
    model = Model.from_config(cfg["model"]).to(device)

    print(f"[load projector] {init_weights}")
    ckpt = torch.load(init_weights, map_location="cpu")
    msg = model.load_state_dict(ckpt, strict=False)
    print("  missing keys   :", getattr(msg, "missing_keys", []))
    print("  unexpected keys:", getattr(msg, "unexpected_keys", []))

    model.eval()
    return model, cfg


def build_joint_dataset(args, cfg):
    min_obj_area_ratio = float(cfg.get("dataset", {}).get("min_obj_area_ratio", 0.0))
    return DinoClipJointDataset(
        args.dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=".tar" in args.dataset,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=min_obj_area_ratio,
    )


def safe_name(x: Any) -> str:
    s = str(x)
    for ch in ["/", "\\", " ", ":", ";", ",", "[", "]", "(", ")", "'", '"']:
        s = s.replace(ch, "_")
    return s[:120]


def get_meta(batch: Dict[str, Any], b: int) -> Dict[str, Any]:
    meta = batch.get("metadata", None)
    if isinstance(meta, (list, tuple)) and b < len(meta) and isinstance(meta[b], dict):
        return meta[b]
    return {}


def get_part_name(meta: Dict[str, Any], local_idx: int, pid: int) -> str:
    names = meta.get("part_class_name", [])
    if isinstance(names, (list, tuple)) and local_idx < len(names):
        return str(names[local_idx])
    return f"part_{pid}"


def resolve_path(dataset: Optional[DinoClipJointDataset], p: str) -> Optional[Path]:
    if p is None:
        return None

    # Try dataset resolver first: this matches DinoClipJointDataset.
    if dataset is not None and hasattr(dataset, "_resolve_path"):
        try:
            q = Path(dataset._resolve_path(str(p)))
            if q.exists():
                return q
        except Exception:
            pass

    candidates = [Path(str(p)), REPO_ROOT / str(p)]
    for q in candidates:
        if q.exists():
            return q
    return None


def image_candidates_from_meta(dataset: Optional[DinoClipJointDataset], meta: Dict[str, Any]) -> List[Path]:
    candidates: List[Path] = []

    # Prefer original RGB image keys if metadata has them.
    for key in [
        "image_path",
        "img_path",
        "file_path",
        "file_name",
        "filename",
        "image_file",
        "img_file",
    ]:
        v = meta.get(key, None)
        if v is None:
            continue
        q = resolve_path(dataset, str(v))
        if q is not None:
            candidates.append(q)

    # Fall back by deriving RGB image path from seg_path.
    seg_path = meta.get("seg_path", None)
    if seg_path is not None:
        q = resolve_path(dataset, str(seg_path))
        if q is not None:
            s = str(q)
            repls = [
                ("annotations_detectron2_obj", "images"),
                ("annotations_detectron2_part", "images"),
                ("annotations_detectron2", "images"),
                ("annotations", "images"),
                ("segmentation", "images"),
                ("SegmentationClass", "JPEGImages"),
            ]
            for old, new in repls:
                if old in s:
                    base = Path(s.replace(old, new))
                    for suf in [".jpg", ".jpeg", ".png"]:
                        cand = base.with_suffix(suf)
                        if cand.exists():
                            candidates.append(cand)

            # As last fallback, use seg image itself.
            candidates.append(q)

    # Remove duplicates while preserving order.
    uniq: List[Path] = []
    seen = set()
    for q in candidates:
        qs = str(q)
        if qs not in seen:
            uniq.append(q)
            seen.add(qs)
    return uniq


def open_stage2_crop_image(
    dataset: Optional[DinoClipJointDataset],
    meta: Dict[str, Any],
    *,
    crop_to_obj: bool = True,
    resize_to_crop_dim: bool = True,
    crop_dim: int = 448,
) -> Optional[Image.Image]:
    """
    Open the original RGB image, crop the object by cropaug_box_xyxy, then optionally
    resize to crop_dim x crop_dim.

    This is aligned with Stage2 cropaug_patch_tokens:
      original image -> object crop by cropaug_box_xyxy -> resize/crop_dim -> DINO patch tokens.

    The old visualizer opened seg_path first, which can make the output look like a
    full annotation map rather than the exact object crop.
    """
    img = None
    for q in image_candidates_from_meta(dataset, meta):
        try:
            img = Image.open(q).convert("RGB")
            break
        except Exception:
            continue

    if img is None:
        return None

    if crop_to_obj and meta.get("cropaug_box_xyxy", None) is not None:
        box = meta["cropaug_box_xyxy"]
        if torch.is_tensor(box):
            box = box.detach().cpu().tolist()
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
            x1 = max(0, min(x1, img.width - 1))
            y1 = max(0, min(y1, img.height - 1))
            x2 = max(x1 + 1, min(x2, img.width))
            y2 = max(y1 + 1, min(y2, img.height))
            img = img.crop((x1, y1, x2, y2))

    if resize_to_crop_dim:
        img = img.resize((int(crop_dim), int(crop_dim)), resample=Image.Resampling.BILINEAR)

    return img


def infer_grid_size_from_tokens(num_patches: int, fallback_crop_dim: int, patch_size: int) -> int:
    g = int(round(math.sqrt(int(num_patches))))
    if g * g == int(num_patches):
        return g
    return int(fallback_crop_dim // patch_size)


def compute_part_anchor_mask(
    *,
    part_valid_mask: torch.Tensor,       # [B,K]
    part_gt_mask_patch: torch.Tensor,    # [B,K,N]
    obj_mask_patch: torch.Tensor,        # [B,N]
    present_only_anchor: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    part_valid_mask = part_valid_mask.bool()
    part_gt_mask_patch = part_gt_mask_patch.bool()
    obj_mask_patch = obj_mask_patch.bool()

    part_present_mask = ((part_gt_mask_patch & obj_mask_patch[:, None, :]).sum(dim=-1) > 0)
    if present_only_anchor:
        part_anchor_mask = part_valid_mask & part_present_mask
    else:
        part_anchor_mask = part_valid_mask
    return part_anchor_mask, part_present_mask


@torch.no_grad()
def get_stage2_anchor_outputs(
    model,
    anchor_helper: JointObjPartLoss,
    batch: Dict[str, Any],
    patch_temperature: float,
    em_iters: int,
    present_only_anchor: bool = False,
):
    """
    Reuse the exact Stage2 anchor selector from JointObjPartLoss._anchor_proto_em_pool.

    Important:
      - If present_only_anchor=False, this visualizes original all-candidate anchor.
      - If present_only_anchor=True, this visualizes the oracle training logic:
        only parts with non-empty GT mask inside the current object crop are allowed
        to mine anchors and build pseudo labels.

    Anchor search is on batch["patch_tokens"].
    With --part_feature_name cropaug_patch_tokens, these are object-crop tokens,
    not full-image tokens.
    """
    part_text_feat = batch["part_text_feat"].float()
    patch_tokens = batch["patch_tokens"].float()
    obj_mask_patch = batch["obj_mask_patch"].bool()
    part_valid_mask = batch["part_valid_mask"].bool()
    part_gt_mask_patch = batch["part_gt_mask_patch"].bool()

    part_anchor_mask, part_present_mask = compute_part_anchor_mask(
        part_valid_mask=part_valid_mask,
        part_gt_mask_patch=part_gt_mask_patch,
        obj_mask_patch=obj_mask_patch,
        present_only_anchor=bool(present_only_anchor),
    )

    part_proj = model.project_clip_txt(part_text_feat)
    part_proj = anchor_helper._safe_normalize(part_proj, dim=-1)
    patch_tokens_norm = anchor_helper._safe_normalize(patch_tokens, dim=-1)

    abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens_norm) / float(patch_temperature)
    abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

    ret = anchor_helper._anchor_proto_em_pool(
        patch_tokens=patch_tokens_norm,
        abs_logits=abs_logits,
        obj_mask_patch=obj_mask_patch,
        part_valid_mask=part_anchor_mask,
        part_gt_mask_patch=part_gt_mask_patch,
        num_iters=em_iters,
        return_anchor_tokens=True,
    )
    if not isinstance(ret, (tuple, list)) or len(ret) < 5:
        raise RuntimeError(
            "JointObjPartLoss._anchor_proto_em_pool(..., return_anchor_tokens=True) "
            "must return anchor_tokens and anchor_valid as the last two values."
        )

    return {
        "patch_tokens_norm": patch_tokens_norm,
        "obj_mask_patch": obj_mask_patch,
        "part_valid_mask": part_valid_mask,
        "part_present_mask": part_present_mask,
        "part_anchor_mask": part_anchor_mask,
        "anchor_tokens": ret[-2],
        "anchor_valid": ret[-1],
    }


@torch.no_grad()
def recover_anchor_patch_indices(
    patch_tokens_norm: torch.Tensor,
    obj_mask_patch: torch.Tensor,
    anchor_tokens: torch.Tensor,
    anchor_valid: torch.Tensor,
) -> torch.Tensor:
    """
    _anchor_proto_em_pool returns anchor tokens but not patch indices.
    For visualization only, recover patch index by nearest token match inside object mask.
    """
    B, K, _ = anchor_tokens.shape
    device = anchor_tokens.device
    out = torch.full((B, K), -1, dtype=torch.long, device=device)

    for b in range(B):
        valid_patch_idx = torch.nonzero(obj_mask_patch[b], as_tuple=False).squeeze(1)
        if valid_patch_idx.numel() == 0:
            continue

        valid_patch_tokens = patch_tokens_norm[b, valid_patch_idx]
        for k in range(K):
            if not bool(anchor_valid[b, k]):
                continue
            sim = valid_patch_tokens @ anchor_tokens[b, k]
            out[b, k] = valid_patch_idx[int(sim.argmax().item())]

    return out


@torch.no_grad()
def build_pseudo_pid_map(
    patch_tokens_norm_b: torch.Tensor,
    obj_mask_patch_b: torch.Tensor,
    part_anchor_mask_b: torch.Tensor,
    part_ids_b: torch.Tensor,
    anchor_tokens_b: torch.Tensor,
    anchor_idx_global_b: torch.Tensor,
    em_iters: int,
) -> Optional[torch.Tensor]:
    """
    Build pseudo part map only for the anchor-candidate parts.

    For present-only visualization this means absent parts are not assigned / drawn.
    """
    valid_part_idx = torch.nonzero(part_anchor_mask_b, as_tuple=False).squeeze(1)
    valid_part_idx = valid_part_idx[anchor_idx_global_b[valid_part_idx] >= 0]
    valid_patch_idx = torch.nonzero(obj_mask_patch_b, as_tuple=False).squeeze(1)

    if valid_part_idx.numel() == 0 or valid_patch_idx.numel() == 0:
        return None

    patch = patch_tokens_norm_b[valid_patch_idx]
    C = anchor_tokens_b[valid_part_idx].clone()
    C = C / C.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    global_to_local = {int(g.item()): i for i, g in enumerate(valid_patch_idx)}
    anchor_idx_local = []
    for pidx in valid_part_idx.tolist():
        g = int(anchor_idx_global_b[pidx].item())
        anchor_idx_local.append(global_to_local.get(g, -1))
    anchor_idx_local = torch.tensor(anchor_idx_local, dtype=torch.long, device=patch.device)

    K = int(valid_part_idx.numel())
    assign = torch.zeros((patch.shape[0],), dtype=torch.long, device=patch.device)

    for _ in range(max(int(em_iters), 1)):
        score = patch @ C.T
        assign = score.argmax(dim=1)

        valid_anchor = anchor_idx_local >= 0
        if valid_anchor.any():
            assign[anchor_idx_local[valid_anchor]] = torch.arange(K, device=patch.device)[valid_anchor]

        onehot = torch.nn.functional.one_hot(assign, num_classes=K).float()
        count = onehot.sum(dim=0).clamp_min(1.0)
        C = onehot.T @ patch
        C = C / count[:, None]
        C = C / C.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    pseudo_pid = torch.full((obj_mask_patch_b.numel(),), -1, dtype=torch.long, device=patch.device)
    assigned_part_local_idx = valid_part_idx[assign]
    pseudo_pid[valid_patch_idx] = part_ids_b[assigned_part_local_idx].long()
    return pseudo_pid


def color_for_pid(pid: int) -> np.ndarray:
    rng = np.random.default_rng(seed=int(pid) * 1009 + 17)
    return rng.integers(40, 235, size=3).astype(np.float32)


def upsample_label_grid(label_1d: torch.Tensor, grid_size: int, width: int, height: int) -> np.ndarray:
    grid = label_1d.detach().cpu().numpy().reshape(grid_size, grid_size).astype(np.int32)
    img = Image.fromarray(grid, mode="I")
    img = img.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.array(img).astype(np.int32)


def overlay_pseudo(
    base_img: Image.Image,
    pseudo_pid: torch.Tensor,
    grid_size: int,
    alpha: float,
    anchor_idx: Optional[torch.Tensor] = None,
    draw_anchor: bool = True,
) -> Image.Image:
    base = base_img.convert("RGB")
    W, H = base.size
    labels = upsample_label_grid(pseudo_pid, grid_size, W, H)

    arr = np.array(base).astype(np.float32)
    valid = labels >= 0
    if valid.any():
        color = np.zeros_like(arr)
        for pid in np.unique(labels[valid]):
            color[labels == int(pid)] = color_for_pid(int(pid))
        arr[valid] = (1.0 - alpha) * arr[valid] + alpha * color[valid]

    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if draw_anchor and anchor_idx is not None:
        draw = ImageDraw.Draw(out)
        for idx in anchor_idx.detach().cpu().tolist():
            idx = int(idx)
            if idx < 0:
                continue
            y = idx // grid_size
            x = idx % grid_size
            cx = (x + 0.5) / grid_size * W
            cy = (y + 0.5) / grid_size * H
            r = max(2, int(0.006 * max(W, H)))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 0, 0), width=max(1, r // 2))

    return out


def add_legend(
    img: Image.Image,
    meta: Dict[str, Any],
    part_ids: torch.Tensor,
    part_anchor_mask: torch.Tensor,
    part_present_mask: torch.Tensor,
    anchor_idx: torch.Tensor,
    grid_size: int,
    mode_name: str,
) -> Image.Image:
    rows = []
    for k in range(int(part_ids.numel())):
        if not bool(part_anchor_mask[k]):
            continue
        pid = int(part_ids[k].item())
        pname = get_part_name(meta, k, pid)
        aidx = int(anchor_idx[k].item())
        pos = f"{aidx} ({aidx // grid_size},{aidx % grid_size})" if aidx >= 0 else "invalid"
        present = "P" if bool(part_present_mask[k]) else "A"
        rows.append((k, pid, pname, pos, present, color_for_pid(pid).astype(np.uint8).tolist()))

    if len(rows) == 0:
        return img

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    legend_w = 540
    line_h = 22
    H = max(img.height, 55 + line_h * len(rows))
    canvas = Image.new("RGB", (img.width + legend_w, H), (255, 255, 255))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    x0 = img.width + 12
    draw.text((x0, 10), f"mode={mode_name}", fill=(0, 0, 0), font=font)
    draw.text((x0, 28), f"class={meta.get('class_name', '')} ann={meta.get('annotation_id', '')}", fill=(0, 0, 0), font=font)

    y = 55
    for local_idx, pid, pname, pos, present, c in rows:
        draw.rectangle((x0, y + 3, x0 + 16, y + 19), fill=tuple(c), outline=(0, 0, 0))
        draw.text((x0 + 24, y), f"{local_idx}: [{present}] {pname} | anchor {pos}", fill=(0, 0, 0), font=font)
        y += line_h

    return canvas


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        "Visualize Stage2 anchor/pseudo-label on object crops. "
        "This uses cropaug_patch_tokens by default and draws anchors on the cropped object image."
    )

    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--init_weights", default=None)

    # Backward-compatible aliases from old view_anchor_.py.
    parser.add_argument("--feature_pth", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", default=None)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patch_temperature", type=float, default=None)
    parser.add_argument("--em_iters", type=int, default=None)

    parser.add_argument("--out_dir", "--save_dir", dest="out_dir", default="visualizations/stage2_crop_anchor")
    parser.add_argument("--max_images", type=int, default=0, help="0 means all.")
    parser.add_argument("--ann_idx", type=int, default=None, help="Visualize one dataset index only.")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--no_anchor", action="store_true", help="Do not draw black anchor circles.")
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--present_only_anchor",
        action="store_true",
        default=False,
        help="Visualize oracle present-only anchor: only GT-present parts in the current object crop get anchors.",
    )
    parser.add_argument(
        "--no_crop_to_obj",
        action="store_true",
        default=False,
        help="Debug only. Do not crop original RGB image by cropaug_box_xyxy.",
    )
    parser.add_argument(
        "--no_resize_crop",
        action="store_true",
        default=False,
        help="Debug only. Do not resize cropped object image to crop_dim x crop_dim.",
    )

    # Kept only so old commands do not crash; unused now.
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--meta_pth", default=None)
    parser.add_argument("--palette_py", default=None)
    parser.add_argument("--act", default=None)

    args = parser.parse_args()

    args.dataset = args.dataset or args.feature_pth
    args.model_config = args.model_config or args.config
    args.init_weights = args.init_weights or args.ckpt

    if args.dataset is None:
        raise ValueError("Pass --dataset or old alias --feature_pth")
    if args.model_config is None:
        raise ValueError("Pass --model_config or old alias --config")
    if args.init_weights is None:
        raise ValueError("Pass --init_weights or old alias --ckpt")

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    print("[device]", device)
    print("[view mode]", "present-only crop anchor" if args.present_only_anchor else "all-candidate crop anchor")
    print("[part feature]", args.part_feature_name)
    if args.part_feature_name != "cropaug_patch_tokens":
        print("[warning] part_feature_name is not cropaug_patch_tokens; anchor view may not correspond to object-crop training.")

    model, cfg = load_projector_from_config(args.model_config, args.init_weights, device)
    train_cfg = cfg.get("train", {})
    patch_temperature = float(args.patch_temperature if args.patch_temperature is not None else train_cfg.get("patch_temperature", 0.07))
    em_iters = int(args.em_iters if args.em_iters is not None else train_cfg.get("em_iters", 1))
    obj_ltype = train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce"))

    dataset = build_joint_dataset(args, cfg)

    if args.ann_idx is not None:
        subset = torch.utils.data.Subset(dataset, [int(args.ann_idx)])
        loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=joint_collate_fn, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=joint_collate_fn, pin_memory=True)

    anchor_helper = JointObjPartLoss(
        sim_model=model,
        obj_ltype=obj_ltype,
        lambda_obj=0.0,
        lambda_inst=0.0,
        lambda_overlap=0.0,
        lambda_spear=0.0,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
    ).to(device)
    anchor_helper.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_no_image = 0
    skipped_no_pseudo = 0

    mode_name = "present-only" if args.present_only_anchor else "all-candidate"
    pbar = tqdm(loader, total=len(loader), desc="view-stage2-crop-anchor")
    for batch in pbar:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        stage2 = get_stage2_anchor_outputs(
            model=model,
            anchor_helper=anchor_helper,
            batch=batch,
            patch_temperature=patch_temperature,
            em_iters=em_iters,
            present_only_anchor=bool(args.present_only_anchor),
        )
        anchor_idx = recover_anchor_patch_indices(
            patch_tokens_norm=stage2["patch_tokens_norm"],
            obj_mask_patch=stage2["obj_mask_patch"],
            anchor_tokens=stage2["anchor_tokens"],
            anchor_valid=stage2["anchor_valid"],
        )

        B = int(batch["patch_tokens"].shape[0])
        grid_size = infer_grid_size_from_tokens(
            int(batch["patch_tokens"].shape[1]),
            fallback_crop_dim=int(args.crop_dim),
            patch_size=int(args.patch_size),
        )

        for b in range(B):
            if args.max_images > 0 and saved >= args.max_images:
                break

            meta = get_meta(batch, b)
            base_img = open_stage2_crop_image(
                dataset=dataset,
                meta=meta,
                crop_to_obj=not bool(args.no_crop_to_obj),
                resize_to_crop_dim=not bool(args.no_resize_crop),
                crop_dim=int(args.crop_dim),
            )
            if base_img is None:
                skipped_no_image += 1
                continue

            part_ids = batch["part_category_id"][b].long()
            pseudo_pid = build_pseudo_pid_map(
                patch_tokens_norm_b=stage2["patch_tokens_norm"][b],
                obj_mask_patch_b=stage2["obj_mask_patch"][b],
                part_anchor_mask_b=stage2["part_anchor_mask"][b],
                part_ids_b=part_ids,
                anchor_tokens_b=stage2["anchor_tokens"][b],
                anchor_idx_global_b=anchor_idx[b],
                em_iters=em_iters,
            )
            if pseudo_pid is None:
                skipped_no_pseudo += 1
                continue

            overlay = overlay_pseudo(
                base_img=base_img,
                pseudo_pid=pseudo_pid,
                grid_size=grid_size,
                alpha=float(args.alpha),
                anchor_idx=anchor_idx[b],
                draw_anchor=not bool(args.no_anchor),
            )
            overlay = add_legend(
                img=overlay,
                meta=meta,
                part_ids=part_ids,
                part_anchor_mask=stage2["part_anchor_mask"][b],
                part_present_mask=stage2["part_present_mask"][b],
                anchor_idx=anchor_idx[b],
                grid_size=grid_size,
                mode_name=mode_name,
            )

            cls = meta.get("class_name", "unknown")
            image_id = meta.get("image_id", saved)
            ann_id = meta.get("annotation_id", saved)
            out_name = f"{saved:06d}_{mode_name}_{safe_name(cls)}_img{safe_name(image_id)}_ann{safe_name(ann_id)}.png"
            overlay.save(out_dir / out_name)

            saved += 1
            pbar.set_description(f"view-stage2-crop-anchor saved={saved}")

        if args.max_images > 0 and saved >= args.max_images:
            break

    print(f"[done] saved={saved} out_dir={out_dir}")
    print(f"[skipped] no_image={skipped_no_image}, no_pseudo={skipped_no_pseudo}")


if __name__ == "__main__":
    main()
