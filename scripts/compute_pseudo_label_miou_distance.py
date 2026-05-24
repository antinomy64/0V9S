#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn
from src.loss_joint import JointObjPartLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = safe_normalize(a.float(), dim=-1)
    b = safe_normalize(b.float(), dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


def pick_key(batch: Dict[str, Any], candidates: List[str]) -> str:
    for k in candidates:
        if k in batch:
            return k
    raise KeyError(f"None of keys found: {candidates}. Available keys={sorted(batch.keys())}")


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


def build_projector(args, cfg, device):
    model_class_name = cfg.get("model", {}).get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(cfg["model"])

    ckpt = torch.load(args.init_weights, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()
    if not hasattr(model, "project_clip_txt"):
        raise AttributeError(
            f"{model_class_name} has no project_clip_txt(textual_embedding). "
            "This script expects ProjectionLayer/DoubleMLP-style text projector."
        )
    return model


def build_stage2_loss_helper(model, cfg, device):
    train_cfg = cfg.get("train", {})
    helper = JointObjPartLoss(
        sim_model=model,
        obj_ltype=train_cfg.get("ltype", "infonce"),
        obj_margin=float(train_cfg.get("obj_margin", 0.2)),
        obj_max_violation=bool(train_cfg.get("obj_max_violation", True)),
        lambda_obj=float(train_cfg.get("lambda_obj", 1.0)),
        lambda_inst=float(train_cfg.get("lambda_inst", 0.2)),
        lambda_overlap=float(train_cfg.get("lambda_overlap", 0.05)),
        lambda_spear=float(train_cfg.get("lambda_spear", 0.0)),
        topk_ratio=float(train_cfg.get("topk_ratio", 0.1)),
        patch_temperature=float(train_cfg.get("patch_temperature", 0.07)),
        eps=float(train_cfg.get("eps", 1e-6)),
        em_iters=int(train_cfg.get("em_iters", 3)),
    )
    helper.to(device)
    helper.eval()
    if not hasattr(helper, "_anchor_proto_em_pool"):
        raise AttributeError("JointObjPartLoss has no _anchor_proto_em_pool. Please check src/loss_joint.py.")
    return helper


def make_pred_masks_from_proto(
    patch_tokens_b: torch.Tensor,
    obj_mask_b: torch.Tensor,
    proto_b: torch.Tensor,
    valid_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assign each object patch to nearest Stage2 part prototype.

    Args:
      patch_tokens_b: [N, D]
      obj_mask_b:    [N] bool
      proto_b:       [K, D]
      valid_idx:     [P] valid part slot indices

    Returns:
      pred_masks: [P2, N] bool, non-empty predicted masks only
      pred_slots: [P2] slot indices corresponding to pred_masks rows
    """
    N = patch_tokens_b.shape[0]
    P = int(valid_idx.numel())
    pred_masks = torch.zeros((P, N), dtype=torch.bool, device=patch_tokens_b.device)

    if P == 0 or not obj_mask_b.any():
        return pred_masks[:0], valid_idx[:0]

    valid_patch_idx = torch.nonzero(obj_mask_b, as_tuple=False).squeeze(1)
    valid_patch_tokens = patch_tokens_b[valid_patch_idx]  # [M, D]

    C = safe_normalize(proto_b[valid_idx].float(), dim=-1)  # [P, D]
    scores = valid_patch_tokens @ C.T  # [M, P]
    assign = scores.argmax(dim=1)  # [M]

    pred_masks[assign, valid_patch_idx] = True
    nonempty = pred_masks.sum(dim=1) > 0
    return pred_masks[nonempty], valid_idx[nonempty]


def mask_inter_union(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a = a.bool()
    b = b.bool()
    inter = float((a & b).sum().item())
    union = float((a | b).sum().item())
    return inter, union


def best_inter_union(gt_mask: torch.Tensor, pred_masks: torch.Tensor) -> Tuple[float, float]:
    """
    Return inter/union of the predicted pseudo mask with best IoU to gt_mask.
    If no predicted mask exists, return empty-pred inter/union.
    """
    if pred_masks.numel() == 0 or pred_masks.shape[0] == 0:
        return 0.0, float(gt_mask.bool().sum().item())

    gt = gt_mask.bool()
    pred = pred_masks.bool()
    inter = (pred & gt[None, :]).sum(dim=1).float()
    union = (pred | gt[None, :]).sum(dim=1).float()
    iou = inter / union.clamp_min(1.0)
    best = int(iou.argmax().item())
    return float(inter[best].item()), float(union[best].item())


def mean_iou_from_accumulators(inter_by_pid: Dict[int, float], union_by_pid: Dict[int, float]) -> float:
    vals = []
    for pid in sorted(union_by_pid.keys()):
        union = union_by_pid[pid]
        if union > 0:
            vals.append(inter_by_pid.get(pid, 0.0) / union)
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def mean_distance_from_lists(dist_by_pid: Dict[int, List[float]]) -> float:
    vals = []
    for pid in sorted(dist_by_pid.keys()):
        ds = dist_by_pid[pid]
        if len(ds) > 0:
            vals.append(float(np.mean(ds)))
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init_weights", required=True, help="Stage2 projector checkpoint")

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show_progress", action="store_true", default=False)

    parser.add_argument(
        "--eval_mode",
        choices=["both", "class_aware", "class_agnostic"],
        default="both",
        help=(
            "Which metrics to print. Default prints four values. "
            "class_aware prints class-aware mIoU and class-aware distance only. "
            "class_agnostic prints class-agnostic mIoU and class-agnostic distance only."
        ),
    )
    args = parser.parse_args()

    with open(args.model_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("train", {})
    if args.batch_size is None:
        args.batch_size = int(train_cfg.get("batch_size", 128))

    set_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    dataset = build_dataset(args, cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=joint_collate_fn,
        pin_memory=True,
    )

    model = build_projector(args, cfg, device)
    stage2_helper = build_stage2_loss_helper(model, cfg, device)

    # mIoU accumulators per global part_category_id.
    aware_inter_by_pid = defaultdict(float)
    aware_union_by_pid = defaultdict(float)
    agnostic_inter_by_pid = defaultdict(float)
    agnostic_union_by_pid = defaultdict(float)

    # Distance accumulators per global part_category_id.
    aware_dist_by_pid = defaultdict(list)
    agnostic_dist_by_pid = defaultdict(list)

    for batch in tqdm(
        loader,
        total=len(loader),
        desc="compute Stage2 pseudo-label mIoU/distance",
        disable=not args.show_progress,
    ):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        part_text_key = pick_key(batch, ["part_text_feat", "part_ann_feats"])
        patch_key = pick_key(batch, ["patch_tokens", args.part_feature_name])
        obj_mask_key = pick_key(batch, ["obj_mask_patch"])
        part_valid_key = pick_key(batch, ["part_valid_mask"])
        part_gt_key = pick_key(batch, ["part_gt_mask_patch"])
        part_id_key = pick_key(batch, ["part_category_id"])

        part_text = batch[part_text_key].float()
        part_ids = batch[part_id_key].long()
        patch_tokens = safe_normalize(batch[patch_key].float(), dim=-1)
        obj_mask = batch[obj_mask_key].bool()
        part_valid = batch[part_valid_key].bool()
        part_gt = batch[part_gt_key].bool()

        if part_text.shape[1] == 0 or not part_valid.any():
            continue

        part_proj = model.project_clip_txt(part_text)
        part_proj = safe_normalize(part_proj.float(), dim=-1)

        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / stage2_helper.patch_temperature
        abs_logits = abs_logits.masked_fill(~obj_mask[:, None, :], -1e4)

        anchor_result = stage2_helper._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask,
            part_valid_mask=part_valid,
            part_gt_mask_patch=part_gt,
            num_iters=stage2_helper.em_iters,
            return_anchor_tokens=True,
        )
        if len(anchor_result) != 5:
            raise RuntimeError(
                "Expected _anchor_proto_em_pool(..., return_anchor_tokens=True) to return "
                "(z, proto_part, anchor_metrics, anchor_tokens, anchor_valid)."
            )
        _, proto_part, _, anchor_tokens, anchor_valid = anchor_result

        B = int(part_text.shape[0])
        for b in range(B):
            if not obj_mask[b].any():
                continue

            # Pseudo masks are generated only from valid Stage2 anchors.
            pseudo_idx = torch.nonzero(part_valid[b] & anchor_valid[b], as_tuple=False).squeeze(1)
            pred_masks, pred_slots = make_pred_masks_from_proto(
                patch_tokens_b=patch_tokens[b],
                obj_mask_b=obj_mask[b],
                proto_b=proto_part[b],
                valid_idx=pseudo_idx,
            )
            slot_to_pred_row = {int(slot.item()): r for r, slot in enumerate(pred_slots)}

            valid_gt_idx = torch.nonzero(part_valid[b], as_tuple=False).squeeze(1)
            if valid_gt_idx.numel() == 0:
                continue

            valid_anchor_idx = torch.nonzero(part_valid[b] & anchor_valid[b], as_tuple=False).squeeze(1)
            valid_anchor_tokens = anchor_tokens[b, valid_anchor_idx] if valid_anchor_idx.numel() > 0 else None

            for k_t in valid_gt_idx:
                k = int(k_t.item())
                pid = int(part_ids[b, k].item())
                if pid < 0:
                    continue

                gt_mask = part_gt[b, k].bool()
                if gt_mask.sum().item() == 0:
                    continue

                # GT prototype from GT mask.
                gt_proto = safe_normalize(patch_tokens[b, gt_mask].mean(dim=0), dim=-1)

                # class-aware mIoU: same slot / same part only.
                if k in slot_to_pred_row:
                    pred_same = pred_masks[slot_to_pred_row[k]]
                    inter, union = mask_inter_union(gt_mask, pred_same)
                else:
                    inter, union = 0.0, float(gt_mask.sum().item())
                aware_inter_by_pid[pid] += inter
                aware_union_by_pid[pid] += union

                # class-agnostic mIoU: best pseudo part mask, regardless of class.
                inter_best, union_best = best_inter_union(gt_mask, pred_masks)
                agnostic_inter_by_pid[pid] += inter_best
                agnostic_union_by_pid[pid] += union_best

                # class-aware distance: same anchor / same part only.
                if bool(anchor_valid[b, k]):
                    d = float(cosine_distance(anchor_tokens[b, k], gt_proto).item())
                    aware_dist_by_pid[pid].append(d)

                # class-agnostic distance: closest valid anchor in this object, regardless of class.
                if valid_anchor_tokens is not None and valid_anchor_tokens.numel() > 0:
                    gt_expand = gt_proto[None, :].expand(valid_anchor_tokens.shape[0], -1)
                    dists = cosine_distance(valid_anchor_tokens, gt_expand)
                    d_min = float(dists.min().item())
                    agnostic_dist_by_pid[pid].append(d_min)

    aware_miou = mean_iou_from_accumulators(aware_inter_by_pid, aware_union_by_pid)
    agnostic_miou = mean_iou_from_accumulators(agnostic_inter_by_pid, agnostic_union_by_pid)
    aware_dist = mean_distance_from_lists(aware_dist_by_pid)
    agnostic_dist = mean_distance_from_lists(agnostic_dist_by_pid)

    if args.eval_mode in ["both", "class_aware"]:
        print(f"pseudo label mIoU(class aware): {aware_miou:.8f}")
    if args.eval_mode in ["both", "class_agnostic"]:
        print(f"pseudo label mIoU(class agnostic): {agnostic_miou:.8f}")

    if args.eval_mode in ["both", "class_aware"]:
        print(f"distance of anchor to GT prototype: {aware_dist:.8f}")
    if args.eval_mode in ["both", "class_agnostic"]:
        print(f"distance of anchor to GT prototype(class agnostic): {agnostic_dist:.8f}")


if __name__ == "__main__":
    main()
