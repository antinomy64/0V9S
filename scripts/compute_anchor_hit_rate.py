#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Stage2-exact anchor hit rate on patch grid.

This script uses the exact anchor-hit statistics returned by:
  JointObjPartLoss._anchor_proto_em_pool(...)

It does NOT recover anchor patch index from anchor_tokens.
Instead, it directly accumulates:
  anchor_metrics["anchor_total_hits"]
  anchor_metrics["anchor_total_valid_parts"]

Therefore the reported anchor hit rate matches the Stage2 training definition:
  anchor hit rate = total_anchor_hits / total_valid_parts

In Stage2, a hit means:
  the selected anchor patch for a valid part falls inside that part's GT mask.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


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

    total_hits = 0.0
    total_valid_parts = 0.0

    for batch in tqdm(
        loader,
        total=len(loader),
        desc="compute Stage2-exact anchor hit rate",
        disable=not args.show_progress,
    ):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        part_text_key = pick_key(batch, ["part_text_feat", "part_ann_feats"])
        patch_key = pick_key(batch, ["patch_tokens", args.part_feature_name])
        obj_mask_key = pick_key(batch, ["obj_mask_patch"])
        part_valid_key = pick_key(batch, ["part_valid_mask"])
        part_gt_key = pick_key(batch, ["part_gt_mask_patch"])

        part_text = batch[part_text_key].float()
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

        _, _, anchor_metrics = stage2_helper._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask,
            part_valid_mask=part_valid,
            part_gt_mask_patch=part_gt,
            num_iters=stage2_helper.em_iters,
            return_anchor_tokens=False,
        )

        total_hits += float(anchor_metrics["anchor_total_hits"].detach().cpu().item())
        total_valid_parts += float(anchor_metrics["anchor_total_valid_parts"].detach().cpu().item())

    anchor_hit_rate = total_hits / max(total_valid_parts, 1.0)

    print(f"anchor hit rate: {anchor_hit_rate:.8f}")
    print(f"anchor hits: {int(total_hits)} / {int(total_valid_parts)}")


if __name__ == "__main__":
    main()
