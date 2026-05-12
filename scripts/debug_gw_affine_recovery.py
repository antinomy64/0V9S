#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug whether the current Stage3 GW loss can recover visual prototypes from
synthetic fake text features constructed by rotating visual prototypes.

Positive-control idea:
    real visual_proto V
      -> fake_text T = V @ R       # R is near-identity orthogonal rotation
      -> Stage3GWLoss(fake_text, V)
      -> train a small debug projector P
      -> check whether P(T) retrieves V again.

This script does NOT modify your main training code.
Recommended location in repo:
    scripts/debug_gw_affine_recovery.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


# Make script runnable from repo root or from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint_with_part_anchoraudit import DinoClipJointDataset, joint_collate_fn
from src.loss_stage3_gw import (
    Stage3GWLoss,
    build_class_part_blocks_from_dataset,
    build_stage2_visual_prototypes,
    pairwise_cosine_similarity,
    safe_normalize,
)


def set_seed(seed: int) -> None:
    print(f"[seed] {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DebugLinearProjector(nn.Module):
    """
    Minimal projector for synthetic fake text features.

    fake_text and visual_proto are in the same dimensional space, so this is
    intentionally just D -> D. It exposes project_clip_txt(), so it can be used
    directly by the existing Stage3GWLoss.
    """

    def __init__(self, dim: int, init: str = "identity"):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

        if init == "identity":
            nn.init.eye_(self.proj.weight)
        elif init == "random":
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init: {init}")

    def project_clip_txt(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


@torch.no_grad()
def random_near_identity_orthogonal(
    dim: int,
    device: torch.device,
    strength: float = 0.2,
) -> torch.Tensor:
    """
    Create an orthogonal matrix near identity.

    strength=0.0 -> almost identity
    strength=1.0 -> stronger random rotation
    """
    eye = torch.eye(dim, device=device)
    noise = torch.randn(dim, dim, device=device) / math.sqrt(dim)
    A = eye + float(strength) * noise

    Q, R = torch.linalg.qr(A)

    # Fix QR sign ambiguity so Q remains close to A / identity.
    signs = torch.sign(torch.diag(R))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    Q = Q * signs[None, :]

    return Q


@torch.no_grad()
def make_fake_text_blocks_from_visual(
    class_blocks: List[Dict],
    visual_proto: torch.Tensor,
    transform: str = "orthogonal",
    rotation_strength: float = 0.2,
    affine_scale: float = 0.15,
    affine_bias: float = 0.0,
    permute_within_block: bool = False,
) -> Tuple[List[Dict], Dict[int, torch.Tensor], Dict]:
    """
    Replace each block's part_text with fake text generated from visual_proto.

    For each class block:
        fake_part_text = fake_global[part_ids]

    If permute_within_block=True:
        fake_part_text rows are shuffled. Then target[row_i] tells which visual
        column this fake text row should retrieve.
    """
    device = visual_proto.device
    dim = visual_proto.shape[-1]

    R = random_near_identity_orthogonal(
        dim=dim,
        device=device,
        strength=rotation_strength,
    )

    if transform == "orthogonal":
        fake_global = visual_proto @ R

    elif transform == "affine":
        # This is a stress test. It may break cosine structure.
        # Use orthogonal first for the positive-control experiment.
        scale = 1.0 + affine_scale * torch.randn(dim, device=device)
        bias = affine_bias * torch.randn(dim, device=device)
        fake_global = (visual_proto @ R) * scale[None, :] + bias[None, :]

    else:
        raise ValueError(f"Unknown transform: {transform}")

    fake_global = safe_normalize(fake_global.float(), dim=-1)

    debug_blocks: List[Dict] = []
    true_match: Dict[int, torch.Tensor] = {}

    for block in class_blocks:
        part_ids = block["part_ids"].to(device).long()

        if part_ids.numel() < 2:
            continue

        fake_part_text = fake_global[part_ids]  # [K, D]
        k = fake_part_text.shape[0]

        if permute_within_block:
            perm = torch.randperm(k, device=device)
            fake_part_text = fake_part_text[perm]
            target = perm
        else:
            target = torch.arange(k, device=device)

        new_block = dict(block)
        new_block["part_text"] = fake_part_text.detach()
        debug_blocks.append(new_block)

        true_match[int(block["category_id"])] = target.detach()

    info = {
        "transform": transform,
        "rotation_strength": rotation_strength,
        "affine_scale": affine_scale,
        "affine_bias": affine_bias,
        "permute_within_block": permute_within_block,
        "num_debug_blocks": len(debug_blocks),
        "dim": dim,
    }
    return debug_blocks, true_match, info


@torch.no_grad()
def evaluate_recovery(
    criterion: Stage3GWLoss,
    projector: nn.Module,
    true_match: Dict[int, torch.Tensor],
) -> Dict[str, float]:
    """
    Evaluate whether GW transport and the learned projector recover the known
    synthetic correspondence.
    """
    device = criterion.visual_proto.device

    total_parts = 0
    transport_hits = 0
    retrieval_hits = 0

    transport_mass_sum = 0.0
    target_sim_sum = 0.0
    target_margin_sum = 0.0

    pre_visual_struct_mse_vals = []
    post_visual_struct_mse_vals = []
    pre_post_struct_mse_vals = []

    for block in criterion.gw_blocks:
        cat_id = int(block["category_id"])
        if cat_id not in true_match:
            continue

        part_text = block["part_text"]          # fake T, [K, D]
        visual = safe_normalize(block["visual"], dim=-1)  # real V, [K, D]
        transport = block["T"]                  # GW plan, [K, K]
        target = true_match[cat_id].to(device)  # correct visual col for each fake row

        k = part_text.shape[0]
        if k < 2:
            continue

        row = torch.arange(k, device=device)

        # 1. Does GW transport itself find the right visual index?
        transport_pred = transport.argmax(dim=1)
        transport_hits += int((transport_pred == target).sum().item())

        transport_mass = transport[row, target].sum() / transport.sum().clamp_min(1e-8)
        transport_mass_sum += float(transport_mass.item()) * k

        # 2. Does projected fake text retrieve the right visual prototype?
        projected = projector.project_clip_txt(part_text)
        projected = safe_normalize(projected, dim=-1)

        sim = projected @ visual.T
        retrieval_pred = sim.argmax(dim=1)
        retrieval_hits += int((retrieval_pred == target).sum().item())

        target_sim = sim[row, target]
        sim_without_target = sim.clone()
        sim_without_target[row, target] = -1e4
        wrong_best = sim_without_target.max(dim=1).values
        margin = target_sim - wrong_best

        target_sim_sum += float(target_sim.mean().item()) * k
        target_margin_sum += float(margin.mean().item()) * k

        # 3. Structure diagnostics.
        pre = safe_normalize(part_text, dim=-1)
        post = projected
        visual_aligned = visual[target]

        sim_pre = pairwise_cosine_similarity(pre)
        sim_post = pairwise_cosine_similarity(post)
        sim_visual = pairwise_cosine_similarity(visual_aligned)

        idx = torch.triu_indices(k, k, offset=1, device=device)

        pre_visual_struct_mse_vals.append(
            F.mse_loss(sim_pre[idx[0], idx[1]], sim_visual[idx[0], idx[1]])
        )
        post_visual_struct_mse_vals.append(
            F.mse_loss(sim_post[idx[0], idx[1]], sim_visual[idx[0], idx[1]])
        )
        pre_post_struct_mse_vals.append(
            F.mse_loss(sim_post[idx[0], idx[1]], sim_pre[idx[0], idx[1]])
        )

        total_parts += k

    if total_parts == 0:
        return {
            "num_parts": 0,
            "transport_acc": float("nan"),
            "retrieval_acc": float("nan"),
            "transport_mass": float("nan"),
            "target_sim": float("nan"),
            "target_margin": float("nan"),
            "pre_visual_struct_mse": float("nan"),
            "post_visual_struct_mse": float("nan"),
            "pre_post_struct_mse": float("nan"),
        }

    return {
        "num_parts": int(total_parts),
        "transport_acc": transport_hits / total_parts,
        "retrieval_acc": retrieval_hits / total_parts,
        "transport_mass": transport_mass_sum / total_parts,
        "target_sim": target_sim_sum / total_parts,
        "target_margin": target_margin_sum / total_parts,
        "pre_visual_struct_mse": float(torch.stack(pre_visual_struct_mse_vals).mean().item()),
        "post_visual_struct_mse": float(torch.stack(post_visual_struct_mse_vals).mean().item()),
        "pre_post_struct_mse": float(torch.stack(pre_post_struct_mse_vals).mean().item()),
    }


def load_projection_model(config_path: str, init_weights: str | None, device: torch.device) -> nn.Module:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"]).to(device)

    if init_weights:
        print(f"[model] Loading init weights from {init_weights}")
        ckpt = torch.load(init_weights, map_location="cpu")
        ret = model.load_state_dict(ckpt, strict=False)
        print("[model] missing keys:", getattr(ret, "missing_keys", []))
        print("[model] unexpected keys:", getattr(ret, "unexpected_keys", []))

    model.eval()
    return model


def build_dataset(args, config, is_train: bool) -> DinoClipJointDataset:
    dataset_cfg = config.get("dataset", {})
    min_obj_area_ratio = float(dataset_cfg.get("min_obj_area_ratio", 0.0)) if is_train else 0.0

    path = args.train_dataset if is_train else args.val_dataset
    is_wds = ".tar" in path

    return DinoClipJointDataset(
        path,
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
        min_obj_area_ratio=min_obj_area_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    # Same dataset/model inputs as train_stage3_gw.py, but no mIoU args needed.
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, default=None)
    parser.add_argument("--init_weights", type=str, default="")

    parser.add_argument("--obj_feature_name", type=str, default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", type=str, default="patch_tokens")
    parser.add_argument("--obj_text_name", type=str, default="ann_feats")
    parser.add_argument("--part_text_name", type=str, default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", type=str, default=None)

    # Debug knobs.
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--visual_source", type=str, default=None, choices=["zpart", "anchor"])
    parser.add_argument("--num_parts", type=int, default=None)

    parser.add_argument("--transform", type=str, default="orthogonal", choices=["orthogonal", "affine"])
    parser.add_argument("--rotation_strength", type=float, default=0.2)
    parser.add_argument("--affine_scale", type=float, default=0.15)
    parser.add_argument("--affine_bias", type=float, default=0.0)
    parser.add_argument("--permute_within_block", action="store_true", default=False)

    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--projector_init", type=str, default="identity", choices=["identity", "random"])

    parser.add_argument("--lambda_gw", type=float, default=None)
    parser.add_argument("--lambda_struct", type=float, default=None)
    parser.add_argument("--gw_epsilon", type=float, default=None)
    parser.add_argument("--gw_max_iter", type=int, default=None)
    parser.add_argument("--sinkhorn_iter", type=int, default=None)
    parser.add_argument("--min_proto_count", type=int, default=None)

    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--save_json", type=str, default="debug_gw_affine_recovery.json")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config["train"]

    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 128))
    num_parts = args.num_parts if args.num_parts is not None else int(train_cfg.get("num_parts", 116))
    visual_source = args.visual_source if args.visual_source is not None else str(train_cfg.get("stage2_visual_source", "zpart")).lower()

    lambda_gw = args.lambda_gw if args.lambda_gw is not None else float(train_cfg.get("lambda_gw", 1.0))
    lambda_struct = args.lambda_struct if args.lambda_struct is not None else float(train_cfg.get("lambda_struct", 0.0))
    gw_epsilon = args.gw_epsilon if args.gw_epsilon is not None else float(train_cfg.get("gw_epsilon", 0.05))
    gw_max_iter = args.gw_max_iter if args.gw_max_iter is not None else int(train_cfg.get("gw_max_iter", 20))
    sinkhorn_iter = args.sinkhorn_iter if args.sinkhorn_iter is not None else int(train_cfg.get("sinkhorn_iter", 50))
    min_proto_count = args.min_proto_count if args.min_proto_count is not None else int(train_cfg.get("min_proto_count", 1))

    patch_temperature = float(train_cfg.get("patch_temperature", 0.07))
    em_iters = int(train_cfg.get("em_iters", 1))

    print(
        "[config] "
        f"batch_size={batch_size}, num_parts={num_parts}, visual_source={visual_source}, "
        f"lambda_gw={lambda_gw}, lambda_struct={lambda_struct}, "
        f"gw_epsilon={gw_epsilon}, gw_max_iter={gw_max_iter}, sinkhorn_iter={sinkhorn_iter}, "
        f"min_proto_count={min_proto_count}, patch_temperature={patch_temperature}, em_iters={em_iters}"
    )

    # 1. Load original projector only for Stage2 visual prototype extraction.
    base_model = load_projection_model(args.model_config, args.init_weights, device)

    # 2. Build train dataset/dataloader exactly like current Stage3 flow.
    train_dataset = build_dataset(args, config, is_train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
    )

    # 3. Current Stage2 flow: build global visual prototypes from train set.
    print(f"[Stage2] Building visual prototypes: source={visual_source}")
    proto_pack = build_stage2_visual_prototypes(
        model=base_model,
        dataloader=train_loader,
        num_parts=num_parts,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
        visual_source=visual_source,
    )
    visual_proto = proto_pack["visual_proto"].detach().to(device)
    proto_count = proto_pack["proto_count"].detach().to(device)

    print(
        "[Stage2] visual_proto:",
        tuple(visual_proto.shape),
        "valid_proto_count:",
        int((proto_count >= min_proto_count).sum().item()),
        "/",
        int(proto_count.numel()),
    )

    # 4. Current flow: build per-object class part blocks.
    class_blocks = build_class_part_blocks_from_dataset(train_dataset, device=device)
    print(f"[blocks] class_blocks={len(class_blocks)}")

    # 5. Synthetic fake text = transformed visual prototypes.
    fake_blocks, true_match, fake_info = make_fake_text_blocks_from_visual(
        class_blocks=class_blocks,
        visual_proto=visual_proto,
        transform=args.transform,
        rotation_strength=args.rotation_strength,
        affine_scale=args.affine_scale,
        affine_bias=args.affine_bias,
        permute_within_block=args.permute_within_block,
    )
    print("[fake]", fake_info)

    # 6. Debug projector receives fake text D -> visual D.
    dim = visual_proto.shape[-1]
    debug_projector = DebugLinearProjector(dim=dim, init=args.projector_init).to(device)

    # 7. Current GW loss object, but with fake part_text blocks.
    criterion = Stage3GWLoss(
        sim_model=debug_projector,
        visual_proto=visual_proto,
        class_blocks=fake_blocks,
        obj_ltype=train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce")),
        obj_margin=float(train_cfg.get("margin", 0.2)),
        obj_max_violation=bool(train_cfg.get("max_violation", True)),
        lambda_obj=0.0,
        lambda_gw=lambda_gw,
        lambda_struct=lambda_struct,
        gw_epsilon=gw_epsilon,
        gw_max_iter=gw_max_iter,
        sinkhorn_iter=sinkhorn_iter,
        min_proto_count=min_proto_count,
        proto_count=proto_count,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
    ).to(device)

    optimizer = torch.optim.AdamW(
        debug_projector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = {
        "args": vars(args),
        "fake_info": fake_info,
        "metrics": [],
    }

    print("\n[before training]")
    metrics = evaluate_recovery(criterion, debug_projector, true_match)
    print(json.dumps(metrics, indent=2))
    history["metrics"].append({"step": -1, **metrics})

    pbar = tqdm(range(args.steps), desc="debug-gw-affine")
    for step in pbar:
        debug_projector.train()

        losses = criterion(
            batch=None,
            do_anchor_audit=False,
            do_structure_audit=False,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        if step % args.print_every == 0 or step == args.steps - 1:
            debug_projector.eval()
            metrics = evaluate_recovery(criterion, debug_projector, true_match)

            row = {
                "step": int(step),
                "total": float(losses["total"].detach().item()),
                "gw": float(losses["gw"].detach().item()),
                "struct": float(losses["struct"].detach().item()),
                **metrics,
            }
            history["metrics"].append(row)

            pbar.set_description(
                f"total={row['total']:.4f} "
                f"gw={row['gw']:.4f} "
                f"struct={row['struct']:.4f} "
                f"Tacc={row['transport_acc']:.3f} "
                f"Racc={row['retrieval_acc']:.3f} "
                f"sim={row['target_sim']:.3f} "
                f"margin={row['target_margin']:.3f}"
            )

            print("\n[debug]", json.dumps(row, indent=2))

    print("\n[final]")
    final_metrics = evaluate_recovery(criterion, debug_projector, true_match)
    print(json.dumps(final_metrics, indent=2))

    if args.save_json:
        save_path = Path(args.save_json)
        if not save_path.is_absolute():
            save_path = REPO_ROOT / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"[save] {save_path}")


if __name__ == "__main__":
    main()
