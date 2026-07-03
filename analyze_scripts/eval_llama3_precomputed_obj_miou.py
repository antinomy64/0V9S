#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-image pixel-level object mIoU evaluation for precomputed text features.

Version: v4_full_image_pixel_miou_20260701

This is the full-image replacement for the earlier crop/object-foreground script.
It does NOT use prompt templates or CLIP text encoder during evaluation.

Pipeline:
  1) Read class text features from a VOC116 feature pth, e.g. ann_feats or llama_ann_feats.
  2) Load a Stage1 projector config/checkpoint and project class text to DINO/V space.
  3) For each full image:
       image file -> DINOv2 dense patch tokens on the whole resized image
       32x32 text-patch logits -> bilinear upsample to original GT mask size
       classify every foreground object pixel among the text prototypes
  4) Accumulate a class confusion matrix over pixels and report mIoU/mAcc/aAcc.

Primary validation target:
  With CLIP ann_feats + original object projector, this full-image pixel script should be
  close to the original official object-level mIoU (~85). If it is not close, do not
  use the Llama result as final evidence; adjust image transform / background / eval set.
"""

import argparse
import csv
import importlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_VERSION = "v4_full_image_pixel_miou_20260701"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-image object mIoU using precomputed class text features + Stage1 projector."
    )
    parser.add_argument("--val_pth", required=True, type=str)
    parser.add_argument("--model_config", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)

    parser.add_argument("--text_key", default="ann_feats", type=str,
                        help="Annotation field used as precomputed object text feature, e.g. ann_feats or llama_ann_feats.")
    parser.add_argument("--image_file_name_key", default="file_name", type=str)
    parser.add_argument("--seg_file_name_key", default="seg_file_name", type=str)
    parser.add_argument("--class_name_key", default="class_name", type=str)
    parser.add_argument("--category_id_key", default="category_id", type=str)
    parser.add_argument("--image_id_key", default="image_id", type=str)

    parser.add_argument("--image_size", default=448, type=int,
                        help="Resize full image and full GT mask to image_size before patch evaluation.")
    parser.add_argument("--patch_size", default=14, type=int,
                        help="DINO patch size. 448/14=32 patch grid for ViT-B/14.")
    parser.add_argument("--dino_model", default="dinov2_vitb14_reg", type=str,
                        help="torch.hub DINOv2 model name, e.g. dinov2_vitb14_reg.")
    parser.add_argument("--dino_repo", default="facebookresearch/dinov2", type=str,
                        help="torch.hub repo or local repo path if --dino_source local.")
    parser.add_argument("--dino_source", default="github", choices=["github", "local"],
                        help="torch.hub source. Use local if you have a local dinov2 repo path.")

    parser.add_argument("--path_prefix", default="", type=str,
                        help="Optional prefix for relative image/mask paths.")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--max_images", default=0, type=int)
    parser.add_argument("--max_classes", default=0, type=int)
    parser.add_argument("--ignore_missing_file", action="store_true")
    parser.add_argument("--ignore_index", default=255, type=int,
                        help="GT value ignored if present. Pixels whose value is not an evaluated class id are ignored anyway.")
    parser.add_argument("--eval_mode", default="pixel", choices=["pixel", "patch"],
                        help="pixel: upsample patch logits to GT mask size and compute pixel mIoU; patch: old v3 downsample-GT patch mIoU.")
    parser.add_argument("--upsample_mode", default="bilinear", choices=["bilinear", "nearest"],
                        help="How to upsample CxHxW logits in pixel mode. Official-style dense eval usually uses bilinear logits.")
    parser.add_argument("--align_corners", action="store_true",
                        help="Use align_corners=True for bilinear interpolation. Default False.")
    return parser.parse_args()


def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def resolve_path(p: str, path_prefix: str = "") -> str:
    if os.path.exists(p):
        return p
    if path_prefix:
        q = os.path.join(path_prefix, p)
        if os.path.exists(q):
            return q
    return p


def load_projector(config_path: str, weights_path: str, device: torch.device, strict_load: bool) -> torch.nn.Module:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "model" not in cfg:
        raise KeyError(f"{config_path} has no top-level key 'model'.")

    model_class_name = cfg["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(cfg["model"])

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    ret = model.load_state_dict(state_dict, strict=strict_load)
    print(f"[projector] config:  {config_path}")
    print(f"[projector] weights: {weights_path}")
    print("[projector] missing keys:", getattr(ret, "missing_keys", []))
    print("[projector] unexpected keys:", getattr(ret, "unexpected_keys", []))

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def project_text(model: torch.nn.Module, text_feats: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not hasattr(model, "project_clip_txt"):
        available = [x for x in dir(model) if not x.startswith("_")]
        raise AttributeError(
            "Projector model has no method 'project_clip_txt'. "
            f"Available public attrs include: {available[:80]}"
        )
    z = model.project_clip_txt(text_feats.to(device=device, dtype=torch.float32))
    return normalize(z.float(), dim=-1)


def build_class_text_bank(
    annotations: List[dict],
    text_key: str,
    class_name_key: str,
    category_id_key: str,
) -> Tuple[List[str], torch.Tensor, Dict[str, int], Dict[int, int], Dict[str, int], Dict[str, int]]:
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = defaultdict(int)
    class_to_category_id: Dict[str, int] = {}

    for ann_idx, ann in enumerate(annotations):
        if class_name_key not in ann:
            raise KeyError(f"Annotation {ann_idx} missing class_name_key '{class_name_key}'.")
        if category_id_key not in ann:
            raise KeyError(f"Annotation {ann_idx} missing category_id_key '{category_id_key}'.")
        if text_key not in ann:
            raise KeyError(
                f"Annotation {ann_idx} class={ann.get(class_name_key)} missing text_key '{text_key}'. "
                f"Available keys: {list(ann.keys())}"
            )

        cls = str(ann[class_name_key])
        cat_id = int(ann[category_id_key])
        if cls in class_to_category_id and class_to_category_id[cls] != cat_id:
            raise ValueError(f"Class {cls} has inconsistent category_id: {class_to_category_id[cls]} vs {cat_id}")
        class_to_category_id[cls] = cat_id

        feat = ann[text_key]
        if not torch.is_tensor(feat):
            feat = torch.as_tensor(feat)
        feat = feat.float()
        if feat.ndim != 1:
            raise ValueError(
                f"Annotation {ann_idx} class={cls}: {text_key} must be [D], got {tuple(feat.shape)}"
            )
        if cls not in sums:
            sums[cls] = feat.clone()
        else:
            if sums[cls].shape != feat.shape:
                raise ValueError(f"Class {cls}: inconsistent text dim {tuple(sums[cls].shape)} vs {tuple(feat.shape)}")
            sums[cls] += feat
        counts[cls] += 1

    class_names = sorted(sums.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    category_id_to_idx = {class_to_category_id[c]: class_to_idx[c] for c in class_names}
    feats = torch.stack([sums[c] / counts[c] for c in class_names], dim=0)
    return class_names, feats, class_to_idx, category_id_to_idx, dict(counts), class_to_category_id


def load_dinov2(dino_repo: str, dino_model: str, dino_source: str, device: torch.device) -> torch.nn.Module:
    print(f"[dino] loading {dino_model} from {dino_repo} source={dino_source}")
    try:
        model = torch.hub.load(dino_repo, dino_model, source=dino_source)
    except TypeError:
        # Older torch.hub may not accept source for github default.
        model = torch.hub.load(dino_repo, dino_model)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def preprocess_image(image_path: str, image_size: int) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)


def load_gt_mask(seg_path: str) -> np.ndarray:
    """Load full-resolution integer GT mask."""
    return np.asarray(Image.open(seg_path))


def resize_mask_to_patch_grid(seg_path: str, patch_grid: int) -> np.ndarray:
    """Old v3 behavior: downsample GT mask to patch grid for patch-level debug."""
    mask = Image.open(seg_path)
    mask = mask.resize((patch_grid, patch_grid), resample=Image.NEAREST)
    return np.asarray(mask)


def upsample_logits_to_mask_size(logits_grid: torch.Tensor, mask_shape: Tuple[int, int], mode: str, align_corners: bool) -> torch.Tensor:
    """
    logits_grid: [grid*grid, C] or [grid, grid, C], on device.
    Return pixel logits [H*W, C].
    """
    if logits_grid.ndim == 2:
        N, C = logits_grid.shape
        grid = int(round(math.sqrt(N)))
        if grid * grid != N:
            raise ValueError(f"Cannot reshape logits with N={N} into square grid.")
        x = logits_grid.reshape(grid, grid, C).permute(2, 0, 1).unsqueeze(0)
    elif logits_grid.ndim == 3:
        _, _, C = logits_grid.shape
        x = logits_grid.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Expected logits [N,C] or [H,W,C], got {tuple(logits_grid.shape)}")

    H, W = int(mask_shape[0]), int(mask_shape[1])
    if mode == "nearest":
        y = F.interpolate(x, size=(H, W), mode="nearest")
    else:
        y = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=align_corners)
    return y.squeeze(0).permute(1, 2, 0).reshape(-1, C)


@torch.no_grad()
def extract_patch_tokens(dino: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    image_tensor = image_tensor.to(device=device, dtype=torch.float32)
    out = dino.forward_features(image_tensor)
    if isinstance(out, dict):
        if "x_norm_patchtokens" in out:
            tokens = out["x_norm_patchtokens"]
        elif "x_prenorm" in out:
            tokens = out["x_prenorm"][:, 1:]
        else:
            raise KeyError(f"DINO forward_features dict has no patch token key. Keys: {list(out.keys())}")
    else:
        # Some models return tokens directly.
        tokens = out
        if tokens.ndim == 3 and tokens.shape[1] > 1:
            tokens = tokens[:, 1:]
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(f"Expected patch tokens [1,N,D], got {tuple(tokens.shape)}")
    return normalize(tokens[0].float(), dim=-1)


def update_confusion(confusion: torch.Tensor, gt_idx: torch.Tensor, pred_idx: torch.Tensor) -> None:
    C = confusion.shape[0]
    flat = gt_idx.to(torch.long) * C + pred_idx.to(torch.long)
    binc = torch.bincount(flat, minlength=C * C).reshape(C, C)
    confusion += binc.cpu()


def compute_metrics(confusion: torch.Tensor, class_names: List[str], ann_support: Counter, patch_support: Counter, text_counts: Dict[str, int]) -> Tuple[Dict, List[Dict]]:
    tp = confusion.diag().float()
    gt_sum = confusion.sum(dim=1).float()
    pred_sum = confusion.sum(dim=0).float()
    union = gt_sum + pred_sum - tp
    valid = gt_sum > 0
    iou = torch.full((len(class_names),), float("nan"))
    iou[valid] = tp[valid] / union[valid].clamp_min(1.0)
    acc = torch.full((len(class_names),), float("nan"))
    acc[valid] = tp[valid] / gt_sum[valid].clamp_min(1.0)

    miou = float(iou[valid].mean().item() * 100.0) if valid.any() else float("nan")
    macc = float(acc[valid].mean().item() * 100.0) if valid.any() else float("nan")
    all_acc = float(tp.sum().item() / gt_sum.sum().clamp_min(1.0).item() * 100.0)

    rows = []
    for i, cls in enumerate(class_names):
        rows.append({
            "class_index": i,
            "class_name": cls,
            "text_count": int(text_counts.get(cls, 0)),
            "ann_support": int(ann_support.get(cls, 0)),
            "patch_support": int(patch_support.get(cls, 0)),
            "gt_pixels": int(gt_sum[i].item()),
            "pred_pixels": int(pred_sum[i].item()),
            "tp": int(tp[i].item()),
            "iou": None if torch.isnan(iou[i]) else float(iou[i].item()),
            "iou_percent": None if torch.isnan(iou[i]) else float(iou[i].item() * 100.0),
            "acc": None if torch.isnan(acc[i]) else float(acc[i].item()),
            "acc_percent": None if torch.isnan(acc[i]) else float(acc[i].item() * 100.0),
        })

    return {"mIoU_percent": miou, "mAcc_percent": macc, "allAcc_percent": all_acc}, rows


def save_outputs(args: argparse.Namespace, summary: Dict, rows: List[Dict], confusion: torch.Tensor) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "eval_precomputed_text_full_image_miou_summary.json")
    csv_path = os.path.join(args.out_dir, "eval_precomputed_text_full_image_miou_per_class.csv")
    txt_path = os.path.join(args.out_dir, "eval_precomputed_text_full_image_miou.txt")
    conf_path = os.path.join(args.out_dir, "eval_precomputed_text_full_image_miou_confusion.pt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    torch.save({"confusion": confusion, "class_names": summary["class_names"], "args": vars(args)}, conf_path)

    lines = []
    lines.append(f"[VERSION] {SCRIPT_VERSION}")
    lines.append("definition: Full-image pixel mIoU using precomputed text features; logits are upsampled to GT mask size; no template; no CLIP text encoder.")
    lines.append(f"eval_mode: {args.eval_mode}")
    lines.append(f"upsample_mode: {args.upsample_mode}")
    lines.append(f"align_corners: {bool(args.align_corners)}")
    lines.append(f"val_pth: {args.val_pth}")
    lines.append(f"model_config: {args.model_config}")
    lines.append(f"weights: {args.weights}")
    lines.append(f"text_key: {args.text_key}")
    lines.append(f"dino_model: {args.dino_model}")
    lines.append(f"image_size: {args.image_size}")
    lines.append(f"patch_size: {args.patch_size}")
    lines.append(f"num_classes: {summary['num_classes']}")
    lines.append(f"used_images: {summary['used_images']}")
    lines.append(f"total_eval_pixels: {summary['total_eval_patches']}")
    lines.append("")
    lines.append(f"mIoU:  {summary['mIoU_percent']:.4f}")
    lines.append(f"mAcc:  {summary['mAcc_percent']:.4f}")
    lines.append(f"aAcc:  {summary['allAcc_percent']:.4f}")
    lines.append("")
    lines.append(f"{'class':<16} {'IoU':>10} {'Acc':>10} {'ann':>8} {'pixel':>10} {'tp':>10} {'pred':>10}")
    lines.append("-" * 86)
    for r in rows:
        iou_s = "nan" if r["iou_percent"] is None else f"{r['iou_percent']:.4f}"
        acc_s = "nan" if r["acc_percent"] is None else f"{r['acc_percent']:.4f}"
        lines.append(
            f"{r['class_name']:<16} {iou_s:>10} {acc_s:>10} "
            f"{r['ann_support']:>8} {r['patch_support']:>10} {r['tp']:>10} {r['pred_pixels']:>10}"
        )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"[save] {json_path}")
    print(f"[save] {csv_path}")
    print(f"[save] {txt_path}")
    print(f"[save] {conf_path}")


def evaluate(args: argparse.Namespace) -> Dict:
    print(f"[VERSION] {SCRIPT_VERSION}")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.image_size % args.patch_size != 0:
        raise ValueError(f"image_size must be divisible by patch_size. Got {args.image_size}/{args.patch_size}")
    patch_grid = args.image_size // args.patch_size

    data = torch.load(args.val_pth, map_location="cpu")
    if "images" not in data or "annotations" not in data:
        raise KeyError(f"{args.val_pth} must contain top-level keys 'images' and 'annotations'.")
    images = data["images"]
    annotations = data["annotations"]

    print(f"[data] val_pth: {args.val_pth}")
    print(f"[data] images={len(images)}, annotations={len(annotations)}")

    class_names, class_text_feats, class_to_idx, cat_id_to_idx, text_counts, class_to_cat = build_class_text_bank(
        annotations=annotations,
        text_key=args.text_key,
        class_name_key=args.class_name_key,
        category_id_key=args.category_id_key,
    )
    if args.max_classes > 0:
        class_names = class_names[:args.max_classes]
        keep = set(class_names)
        keep_idx = [i for i, c in enumerate(sorted(text_counts.keys())) if c in keep]
        class_text_feats = class_text_feats[keep_idx]
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        cat_id_to_idx = {class_to_cat[c]: class_to_idx[c] for c in class_names}

    C = len(class_names)
    print(f"[bank] classes={C}, text_dim={class_text_feats.shape[-1]}")
    print(f"[bank] class_names={class_names}")
    print(f"[bank] category ids={[class_to_cat[c] for c in class_names]}")

    projector = load_projector(args.model_config, args.weights, device, args.strict_load)
    text_z = project_text(projector, class_text_feats, device)  # [C,D]
    print(f"[bank] projected_text={tuple(text_z.shape)}")

    dino = load_dinov2(args.dino_repo, args.dino_model, args.dino_source, device)

    # Annotation support is reported from pth; eval itself is image-level.
    ann_support = Counter(str(a[args.class_name_key]) for a in annotations if str(a[args.class_name_key]) in class_to_idx)
    patch_support = Counter()
    confusion = torch.zeros((C, C), dtype=torch.long)

    used_images = 0
    skipped_missing_file = 0
    skipped_no_eval_patch = 0
    total_eval_patches = 0

    pbar = tqdm(images[: args.max_images if args.max_images > 0 else len(images)], desc="eval full-image precomputed text mIoU")
    for img_info in pbar:
        if args.image_file_name_key not in img_info:
            raise KeyError(f"Image {img_info.get('id')} missing '{args.image_file_name_key}'.")
        if args.seg_file_name_key not in img_info:
            raise KeyError(f"Image {img_info.get('id')} missing '{args.seg_file_name_key}'.")

        img_path = resolve_path(str(img_info[args.image_file_name_key]), args.path_prefix)
        seg_path = resolve_path(str(img_info[args.seg_file_name_key]), args.path_prefix)
        if not os.path.exists(img_path) or not os.path.exists(seg_path):
            if args.ignore_missing_file:
                skipped_missing_file += 1
                continue
            missing = img_path if not os.path.exists(img_path) else seg_path
            raise FileNotFoundError(f"Missing image/mask file: {missing}")

        img_tensor = preprocess_image(img_path, args.image_size)
        patch_z = extract_patch_tokens(dino, img_tensor, device)  # [N,D]
        if patch_z.shape[0] != patch_grid * patch_grid:
            raise ValueError(
                f"DINO produced {patch_z.shape[0]} patches, expected {patch_grid * patch_grid}. "
                f"image_size={args.image_size}, patch_size={args.patch_size}"
            )
        sims = patch_z @ text_z.T  # [grid*grid, C]

        if args.eval_mode == "patch":
            pred = sims.argmax(dim=1).detach().cpu().long()
            gt_mask = resize_mask_to_patch_grid(seg_path, patch_grid).reshape(-1)
            gt_idx_np = np.full((gt_mask.shape[0],), -1, dtype=np.int64)
            for cat_id, idx in cat_id_to_idx.items():
                gt_idx_np[gt_mask == int(cat_id)] = int(idx)
            valid_np = gt_idx_np >= 0
            if valid_np.sum() == 0:
                skipped_no_eval_patch += 1
                continue
            gt_idx = torch.from_numpy(gt_idx_np[valid_np]).long()
            pred_idx = pred[torch.from_numpy(valid_np).bool()]
        else:
            # Official-style dense eval: upsample low-res class logits to full GT-mask resolution,
            # then argmax and accumulate pixel-level confusion. This avoids the v3 mismatch of
            # downsampling thin/small GT regions to 32x32 before evaluation.
            gt_mask_full = load_gt_mask(seg_path)
            logits_pix = upsample_logits_to_mask_size(
                sims,
                mask_shape=gt_mask_full.shape[:2],
                mode=args.upsample_mode,
                align_corners=args.align_corners,
            )
            pred_full = logits_pix.argmax(dim=1).detach().cpu().long()
            gt_flat = gt_mask_full.reshape(-1)
            gt_idx_np = np.full((gt_flat.shape[0],), -1, dtype=np.int64)
            for cat_id, idx in cat_id_to_idx.items():
                gt_idx_np[gt_flat == int(cat_id)] = int(idx)
            if args.ignore_index >= 0:
                gt_idx_np[gt_flat == int(args.ignore_index)] = -1
            valid_np = gt_idx_np >= 0
            if valid_np.sum() == 0:
                skipped_no_eval_patch += 1
                continue
            gt_idx = torch.from_numpy(gt_idx_np[valid_np]).long()
            pred_idx = pred_full[torch.from_numpy(valid_np).bool()]

        update_confusion(confusion, gt_idx, pred_idx)

        for cls in class_names:
            idx = class_to_idx[cls]
            n = int((gt_idx == idx).sum().item())
            if n > 0:
                patch_support[cls] += n
        used_images += 1
        total_eval_patches += int(valid_np.sum())
        pbar.set_postfix(used=used_images, skip=skipped_no_eval_patch, pixels=total_eval_patches)

    metrics, rows = compute_metrics(confusion, class_names, ann_support, patch_support, text_counts)
    summary = {
        "script_version": SCRIPT_VERSION,
        "definition": "Full-image pixel mIoU using precomputed text features; logits are upsampled to GT mask size; no template; no CLIP text encoder.",
        "eval_mode": args.eval_mode,
        "upsample_mode": args.upsample_mode,
        "align_corners": bool(args.align_corners),
        "val_pth": args.val_pth,
        "model_config": args.model_config,
        "weights": args.weights,
        "text_key": args.text_key,
        "dino_model": args.dino_model,
        "dino_repo": args.dino_repo,
        "dino_source": args.dino_source,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "patch_grid": patch_grid,
        "num_classes": C,
        "class_names": class_names,
        "class_to_category_id": {c: int(class_to_cat[c]) for c in class_names},
        "used_images": used_images,
        "skipped_missing_file": skipped_missing_file,
        "skipped_no_eval_patch": skipped_no_eval_patch,
        "total_eval_patches": total_eval_patches,
        **metrics,
        "per_class": rows,
    }
    save_outputs(args, summary, rows, confusion)
    return summary


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
