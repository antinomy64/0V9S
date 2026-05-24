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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn
from src.loss_joint import JointObjPartLoss


# -----------------------------------------------------------------------------
# Load model / dataset
# -----------------------------------------------------------------------------

def load_projector(model_config: str, init_weights: str, device: torch.device):
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


def build_dataset(args, cfg):
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


# -----------------------------------------------------------------------------
# Metadata helpers
# -----------------------------------------------------------------------------

def _as_python(x):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().tolist()
    return x


def get_meta(batch: Dict[str, Any], b: int) -> Dict[str, Any]:
    meta = batch.get("metadata", None)
    if isinstance(meta, (list, tuple)) and b < len(meta) and isinstance(meta[b], dict):
        return meta[b]
    return {}


def get_from_batch_or_meta(batch: Dict[str, Any], meta: Dict[str, Any], b: int, keys: List[str]):
    for key in keys:
        if key in batch:
            v = batch[key]
            if torch.is_tensor(v):
                if v.ndim >= 1 and v.shape[0] > b:
                    return _as_python(v[b])
            elif isinstance(v, (list, tuple)) and len(v) > b:
                return _as_python(v[b])
            else:
                return _as_python(v)
        if key in meta:
            return _as_python(meta[key])
    return None


def resolve_image_path(meta: Dict[str, Any], image_root: str = "") -> Optional[Path]:
    keys = [
        "image_path", "img_path", "image_file", "file_name", "filename",
        "coco_file_name", "path", "img", "image",
    ]

    candidates: List[Path] = []
    for key in keys:
        if key not in meta:
            continue
        value = _as_python(meta[key])
        if not isinstance(value, str):
            continue
        p = Path(value)
        candidates.append(p)
        if image_root:
            candidates.append(Path(image_root) / p)
            candidates.append(Path(image_root) / p.name)

    # COCO-style fallback from image_id.
    if image_root and "image_id" in meta:
        image_id = _as_python(meta["image_id"])
        try:
            image_id_int = int(image_id)
            candidates.append(Path(image_root) / f"{image_id_int:012d}.jpg")
            candidates.append(Path(image_root) / f"{image_id_int}.jpg")
            candidates.append(Path(image_root) / f"{image_id_int:012d}.png")
            candidates.append(Path(image_root) / f"{image_id_int}.png")
        except Exception:
            pass

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def resolve_crop_box_xyxy(batch: Dict[str, Any], meta: Dict[str, Any], b: int, image: Image.Image, assume_full_image_if_missing: bool):
    keys = [
        "cropaug_box_xyxy", "crop_box_xyxy", "box_xyxy", "bbox_xyxy",
        "obj_box_xyxy", "object_box_xyxy", "square_box_xyxy",
    ]
    value = get_from_batch_or_meta(batch, meta, b, keys)

    if value is None:
        if assume_full_image_if_missing:
            w, h = image.size
            return [0.0, 0.0, float(w), float(h)]
        return None

    if isinstance(value, dict):
        for subkey in ("xyxy", "box_xyxy", "cropaug_box_xyxy"):
            if subkey in value:
                value = value[subkey]
                break

    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]

    return None


def safe_filename(x: Any) -> str:
    s = str(x)
    for ch in ['/', '\\', ' ', ':', ';', ',', '[', ']', '(', ')']:
        s = s.replace(ch, "_")
    return s[:120]


# -----------------------------------------------------------------------------
# Stage2 anchor + pseudo label
# -----------------------------------------------------------------------------

@torch.no_grad()
def call_stage2_anchor_pool(
    model,
    anchor_helper: JointObjPartLoss,
    batch: Dict[str, Any],
    patch_temperature: float,
    em_iters: int,
):
    """
    Anchor finding is directly delegated to the original Stage2 routine:
      JointObjPartLoss._anchor_proto_em_pool(..., return_anchor_tokens=True)

    The helper returns selected anchor tokens. This script only reconstructs
    per-patch pseudo labels for visualization from those returned anchors.
    """
    part_text_feat = batch["part_text_feat"].float()
    patch_tokens = batch["patch_tokens"].float()
    obj_mask_patch = batch["obj_mask_patch"].bool()
    part_valid_mask = batch["part_valid_mask"].bool()
    part_gt_mask_patch = batch["part_gt_mask_patch"].bool()

    part_proj = model.project_clip_txt(part_text_feat)
    part_proj = anchor_helper._safe_normalize(part_proj, dim=-1)
    patch_tokens_norm = anchor_helper._safe_normalize(patch_tokens, dim=-1)

    abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens_norm) / float(patch_temperature)
    abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

    ret = anchor_helper._anchor_proto_em_pool(
        patch_tokens=patch_tokens_norm,
        abs_logits=abs_logits,
        obj_mask_patch=obj_mask_patch,
        part_valid_mask=part_valid_mask,
        part_gt_mask_patch=part_gt_mask_patch,
        num_iters=em_iters,
        return_anchor_tokens=True,
    )

    if not isinstance(ret, (tuple, list)) or len(ret) < 5:
        raise RuntimeError(
            "Expected _anchor_proto_em_pool(..., return_anchor_tokens=True) "
            "to return (..., anchor_tokens, anchor_valid)."
        )

    return {
        "patch_tokens_norm": patch_tokens_norm,
        "obj_mask_patch": obj_mask_patch,
        "part_valid_mask": part_valid_mask,
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
    Convert returned anchor tokens back to patch indices by nearest matching
    inside object mask.
    """
    B, K, _ = anchor_tokens.shape
    device = anchor_tokens.device
    anchor_idx_global = torch.full((B, K), -1, dtype=torch.long, device=device)

    for b in range(B):
        valid_patch_idx = torch.nonzero(obj_mask_patch[b], as_tuple=False).squeeze(1)
        if valid_patch_idx.numel() == 0:
            continue

        valid_patch_tokens = patch_tokens_norm[b, valid_patch_idx]
        for k in range(K):
            if not bool(anchor_valid[b, k]):
                continue
            sim = valid_patch_tokens @ anchor_tokens[b, k]
            best_local = int(sim.argmax().item())
            anchor_idx_global[b, k] = valid_patch_idx[best_local]

    return anchor_idx_global


@torch.no_grad()
def build_pseudo_pid_map_for_sample(
    patch_tokens_norm_b: torch.Tensor,
    obj_mask_patch_b: torch.Tensor,
    part_valid_mask_b: torch.Tensor,
    part_ids_b: torch.Tensor,
    anchor_tokens_b: torch.Tensor,
    anchor_idx_global_b: torch.Tensor,
    em_iters: int,
):
    """
    Reconstruct Stage2 pseudo label over crop patches.

    Output:
      pseudo_pid: [N], value is global part id, -1 outside object/unassigned.
    """
    device = patch_tokens_norm_b.device

    valid_part_idx = torch.nonzero(part_valid_mask_b, as_tuple=False).squeeze(1)
    valid_part_idx = valid_part_idx[anchor_idx_global_b[valid_part_idx] >= 0]

    valid_patch_idx = torch.nonzero(obj_mask_patch_b, as_tuple=False).squeeze(1)
    if valid_part_idx.numel() == 0 or valid_patch_idx.numel() == 0:
        return None

    valid_patch_tokens = patch_tokens_norm_b[valid_patch_idx]
    C = anchor_tokens_b[valid_part_idx].clone()
    C = C / C.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    global_to_local_patch = {int(g.item()): i for i, g in enumerate(valid_patch_idx)}
    anchor_idx_local = []
    for pidx in valid_part_idx.tolist():
        g = int(anchor_idx_global_b[pidx].item())
        anchor_idx_local.append(global_to_local_patch.get(g, -1))
    anchor_idx_local = torch.tensor(anchor_idx_local, dtype=torch.long, device=device)

    K = int(valid_part_idx.numel())
    M = int(valid_patch_idx.numel())
    assign = torch.zeros(M, dtype=torch.long, device=device)

    for _ in range(max(int(em_iters), 1)):
        scores = valid_patch_tokens @ C.T
        assign = scores.argmax(dim=1)

        valid_anchor = anchor_idx_local >= 0
        if valid_anchor.any():
            assign[anchor_idx_local[valid_anchor]] = torch.arange(K, device=device)[valid_anchor]

        onehot = torch.nn.functional.one_hot(assign, num_classes=K).float()
        count = onehot.sum(dim=0).clamp_min(1.0)
        C = onehot.T @ valid_patch_tokens
        C = C / count[:, None]
        C = C / C.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    N = int(obj_mask_patch_b.numel())
    pseudo_pid = torch.full((N,), -1, dtype=torch.long, device=device)
    assigned_local_part_idx = valid_part_idx[assign]
    assigned_pids = part_ids_b[assigned_local_part_idx].long()
    pseudo_pid[valid_patch_idx] = assigned_pids

    return pseudo_pid


# -----------------------------------------------------------------------------
# Overlay
# -----------------------------------------------------------------------------

def color_for_pid(pid: int) -> Tuple[int, int, int]:
    """
    Deterministic vivid color per global part id.
    """
    rng = np.random.default_rng(seed=int(pid) * 1009 + 17)
    color = rng.integers(low=40, high=235, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def upsample_label_grid(label_1d: torch.Tensor, grid_size: int, out_w: int, out_h: int) -> np.ndarray:
    grid = label_1d.detach().cpu().numpy().reshape(grid_size, grid_size).astype(np.int32)
    img = Image.fromarray(grid, mode="I")
    img = img.resize((out_w, out_h), resample=Image.Resampling.NEAREST)
    return np.array(img).astype(np.int32)


def overlay_labels_on_original(
    image: Image.Image,
    pseudo_pid_1d: torch.Tensor,
    grid_size: int,
    crop_box_xyxy: List[float],
    alpha: float,
    draw_anchor: bool,
    anchor_idx_global: Optional[torch.Tensor] = None,
):
    base = image.convert("RGB")
    base_np = np.array(base).astype(np.float32)

    W, H = base.size
    x1, y1, x2, y2 = crop_box_xyxy
    crop_x1 = int(round(x1))
    crop_y1 = int(round(y1))
    crop_x2 = int(round(x2))
    crop_y2 = int(round(y2))

    crop_w = max(1, crop_x2 - crop_x1)
    crop_h = max(1, crop_y2 - crop_y1)

    label_crop = upsample_label_grid(pseudo_pid_1d, grid_size, crop_w, crop_h)

    # Clip crop to image bounds.
    ix1 = max(0, crop_x1)
    iy1 = max(0, crop_y1)
    ix2 = min(W, crop_x2)
    iy2 = min(H, crop_y2)

    if ix2 <= ix1 or iy2 <= iy1:
        return base

    lx1 = ix1 - crop_x1
    ly1 = iy1 - crop_y1
    lx2 = lx1 + (ix2 - ix1)
    ly2 = ly1 + (iy2 - iy1)

    sub_label = label_crop[ly1:ly2, lx1:lx2]
    overlay = base_np[iy1:iy2, ix1:ix2].copy()

    mask = sub_label >= 0
    if mask.any():
        color_arr = np.zeros_like(overlay)
        for pid in np.unique(sub_label[mask]):
            color_arr[sub_label == int(pid)] = color_for_pid(int(pid))
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_arr[mask]
        base_np[iy1:iy2, ix1:ix2] = overlay

    out = Image.fromarray(np.clip(base_np, 0, 255).astype(np.uint8))

    if draw_anchor and anchor_idx_global is not None:
        draw = ImageDraw.Draw(out)
        for idx in anchor_idx_global.detach().cpu().tolist():
            idx = int(idx)
            if idx < 0:
                continue
            py = idx // grid_size
            px = idx % grid_size
            cx = crop_x1 + (px + 0.5) / grid_size * crop_w
            cy = crop_y1 + (py + 0.5) / grid_size * crop_h
            r = max(2, int(0.006 * max(W, H)))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 0, 0), width=max(1, r // 2))

    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init_weights", required=True)

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

    parser.add_argument("--image_root", default="", help="Root dir for original images if metadata stores relative file names.")
    parser.add_argument("--assume_full_image_if_no_crop_box", action="store_true", help="Fallback: overlay crop grid on full image if crop box is absent.")
    parser.add_argument("--class_keywords", nargs="*", default=[], help="Optional class filter. Empty means all.")
    parser.add_argument("--max_images", type=int, default=0, help="0 means all annotations.")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--draw_anchor", action="store_true", default=True)
    parser.add_argument("--no_draw_anchor", action="store_false", dest="draw_anchor")

    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("[device]", device)

    model, cfg = load_projector(args.model_config, args.init_weights, device)

    train_cfg = cfg.get("train", {})
    patch_temperature = float(args.patch_temperature if args.patch_temperature is not None else train_cfg.get("patch_temperature", 0.07))
    em_iters = int(args.em_iters if args.em_iters is not None else train_cfg.get("em_iters", 1))
    obj_ltype = train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce"))

    dataset = build_dataset(args, cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
        pin_memory=True,
    )

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

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    grid_size = int(args.crop_dim // args.patch_size)
    saved = 0
    skipped_no_image = 0
    skipped_no_crop = 0

    for batch in tqdm(loader, total=len(loader), desc="visualize-stage2-pseudo-on-original"):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        stage2 = call_stage2_anchor_pool(
            model=model,
            anchor_helper=anchor_helper,
            batch=batch,
            patch_temperature=patch_temperature,
            em_iters=em_iters,
        )
        anchor_idx_global = recover_anchor_patch_indices(
            patch_tokens_norm=stage2["patch_tokens_norm"],
            obj_mask_patch=stage2["obj_mask_patch"],
            anchor_tokens=stage2["anchor_tokens"],
            anchor_valid=stage2["anchor_valid"],
        )

        B = int(batch["patch_tokens"].shape[0])
        for b in range(B):
            if args.max_images > 0 and saved >= args.max_images:
                break

            meta = get_meta(batch, b)
            class_name = str(meta.get("class_name", ""))
            if args.class_keywords and not contains_any(class_name, args.class_keywords):
                continue

            img_path = resolve_image_path(meta, args.image_root)
            if img_path is None:
                skipped_no_image += 1
                continue

            image = Image.open(img_path).convert("RGB")
            crop_box = resolve_crop_box_xyxy(
                batch=batch,
                meta=meta,
                b=b,
                image=image,
                assume_full_image_if_missing=args.assume_full_image_if_no_crop_box,
            )
            if crop_box is None:
                skipped_no_crop += 1
                continue

            part_ids_b = batch["part_category_id"][b].long()
            pseudo_pid = build_pseudo_pid_map_for_sample(
                patch_tokens_norm_b=stage2["patch_tokens_norm"][b],
                obj_mask_patch_b=stage2["obj_mask_patch"][b],
                part_valid_mask_b=stage2["part_valid_mask"][b],
                part_ids_b=part_ids_b,
                anchor_tokens_b=stage2["anchor_tokens"][b],
                anchor_idx_global_b=anchor_idx_global[b],
                em_iters=em_iters,
            )
            if pseudo_pid is None:
                continue

            out_img = overlay_labels_on_original(
                image=image,
                pseudo_pid_1d=pseudo_pid,
                grid_size=grid_size,
                crop_box_xyxy=crop_box,
                alpha=float(args.alpha),
                draw_anchor=bool(args.draw_anchor),
                anchor_idx_global=anchor_idx_global[b],
            )

            image_id = meta.get("image_id", saved)
            ann_id = meta.get("annotation_id", meta.get("ann_id", saved))
            out_name = f"{saved:06d}_{safe_filename(class_name)}_img{safe_filename(image_id)}_ann{safe_filename(ann_id)}.png"
            out_img.save(save_dir / out_name)
            saved += 1

        if args.max_images > 0 and saved >= args.max_images:
            break

    print(f"[done] saved={saved} to {save_dir}")
    print(f"[skipped] no_image={skipped_no_image}, no_crop_box={skipped_no_crop}")
    if skipped_no_image > 0:
        print("[hint] If images are relative paths or absent in metadata, pass --image_root /path/to/images")
    if skipped_no_crop > 0:
        print("[hint] If every annotation is already full-image aligned, pass --assume_full_image_if_no_crop_box")


if __name__ == "__main__":
    main()
