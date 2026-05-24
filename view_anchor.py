#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

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


def open_stage2_annotation_image(dataset: DinoClipJointDataset, meta: Dict[str, Any], crop_for_cropaug: bool = True) -> Optional[Image.Image]:
    """
    Use the same path convention as DinoClipJointDataset:
      metadata["seg_path"] -> dataset._resolve_path(seg_path)
    """
    seg_path = meta.get("seg_path", None)
    if seg_path is None:
        return None

    path = Path(dataset._resolve_path(str(seg_path)))
    if not path.exists():
        return None

    img = Image.open(path).convert("RGB")

    if crop_for_cropaug and meta.get("cropaug_box_xyxy", None) is not None:
        box = meta["cropaug_box_xyxy"]
        if torch.is_tensor(box):
            box = box.detach().cpu().tolist()
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
            img = img.crop((x1, y1, x2, y2))

    return img


@torch.no_grad()
def get_stage2_anchor_outputs(
    model,
    anchor_helper: JointObjPartLoss,
    batch: Dict[str, Any],
    patch_temperature: float,
    em_iters: int,
):
    """
    Stage2 anchor selection is reused from JointObjPartLoss._anchor_proto_em_pool.
    This script does not reimplement the relative-score greedy anchor logic.
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
            "JointObjPartLoss._anchor_proto_em_pool(..., return_anchor_tokens=True) "
            "must return anchor_tokens and anchor_valid as the last two values."
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
    part_valid_mask_b: torch.Tensor,
    part_ids_b: torch.Tensor,
    anchor_tokens_b: torch.Tensor,
    anchor_idx_global_b: torch.Tensor,
    em_iters: int,
) -> Optional[torch.Tensor]:
    valid_part_idx = torch.nonzero(part_valid_mask_b, as_tuple=False).squeeze(1)
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
    part_valid: torch.Tensor,
    anchor_idx: torch.Tensor,
    grid_size: int,
) -> Image.Image:
    rows = []
    for k in range(int(part_ids.numel())):
        if not bool(part_valid[k]):
            continue
        pid = int(part_ids[k].item())
        pname = get_part_name(meta, k, pid)
        aidx = int(anchor_idx[k].item())
        pos = f"{aidx} ({aidx // grid_size},{aidx % grid_size})" if aidx >= 0 else "invalid"
        rows.append((k, pid, pname, pos, color_for_pid(pid).astype(np.uint8).tolist()))

    if len(rows) == 0:
        return img

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    legend_w = 460
    line_h = 22
    H = max(img.height, 35 + line_h * len(rows))
    canvas = Image.new("RGB", (img.width + legend_w, H), (255, 255, 255))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    x0 = img.width + 12
    draw.text((x0, 10), f"class={meta.get('class_name', '')} ann={meta.get('annotation_id', '')}", fill=(0, 0, 0), font=font)

    y = 35
    for local_idx, pid, pname, pos, c in rows:
        draw.rectangle((x0, y + 3, x0 + 16, y + 19), fill=tuple(c), outline=(0, 0, 0))
        draw.text((x0 + 24, y), f"{local_idx}: {pname} | anchor {pos}", fill=(0, 0, 0), font=font)
        y += line_h

    return canvas


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Visualize Stage2 pseudo-label regions. Reuses repo dataset/loss/model code.")

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

    parser.add_argument("--out_dir", "--save_dir", dest="out_dir", default="visualizations/stage2_pseudo")
    parser.add_argument("--max_images", type=int, default=0, help="0 means all.")
    parser.add_argument("--ann_idx", type=int, default=None, help="Visualize one dataset index only.")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--no_anchor", action="store_true", help="Do not draw black anchor circles.")
    parser.add_argument("--device", default="cuda")

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

    grid_size = int(args.crop_dim // args.patch_size)
    saved = 0
    skipped_no_image = 0
    skipped_no_pseudo = 0

    pbar = tqdm(loader, total=len(loader), desc="view-stage2-pseudo")
    for batch in pbar:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        stage2 = get_stage2_anchor_outputs(
            model=model,
            anchor_helper=anchor_helper,
            batch=batch,
            patch_temperature=patch_temperature,
            em_iters=em_iters,
        )
        anchor_idx = recover_anchor_patch_indices(
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
            base_img = open_stage2_annotation_image(
                dataset=dataset,
                meta=meta,
                crop_for_cropaug=(args.part_feature_name == "cropaug_patch_tokens"),
            )
            if base_img is None:
                skipped_no_image += 1
                continue

            part_ids = batch["part_category_id"][b].long()
            pseudo_pid = build_pseudo_pid_map(
                patch_tokens_norm_b=stage2["patch_tokens_norm"][b],
                obj_mask_patch_b=stage2["obj_mask_patch"][b],
                part_valid_mask_b=stage2["part_valid_mask"][b],
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
                part_valid=stage2["part_valid_mask"][b],
                anchor_idx=anchor_idx[b],
                grid_size=grid_size,
            )

            cls = meta.get("class_name", "unknown")
            image_id = meta.get("image_id", saved)
            ann_id = meta.get("annotation_id", saved)
            out_name = f"{saved:06d}_{safe_name(cls)}_img{safe_name(image_id)}_ann{safe_name(ann_id)}.png"
            overlay.save(out_dir / out_name)

            saved += 1
            pbar.set_description(f"view-stage2-pseudo saved={saved}")

        if args.max_images > 0 and saved >= args.max_images:
            break

    print(f"[done] saved={saved} out_dir={out_dir}")
    print(f"[skipped] no_image={skipped_no_image}, no_pseudo={skipped_no_pseudo}")


if __name__ == "__main__":
    main()
