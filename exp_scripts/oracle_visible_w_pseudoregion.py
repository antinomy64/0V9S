#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean test15 oracle-visible orthogonal W training, pseudo-region supervision version.

Purpose
-------
This is a minimal version of the previous best-W pipeline:
  Stage1 text projector is frozen.
  V-side cropaug_patch_tokens stay in DINO/V space and are only normalized.
  Only a global orthogonal W is trained.
  GT part masks are used only to choose oracle-visible positive parts and to audit anchor hit.
  Anchor selection and loss do not use GT patch locations.

Training definition
-------------------
  z0 = normalize(stage1.project_clip_txt(part_text))
  z  = normalize(z0 @ W)
  sim = z @ patch_tokens[obj_mask].T
  rel[k,m] = sim[k,m] - max_{q != k} sim[q,m]
  seed = greedy distinct assignment on rel
  support_k = high-confidence patches where part k is top-1 and rel[k,m] >= threshold
  proto_k = normalize(mean(patch_tokens[support_k]))
  loss = mean(1 - cos(z[k], proto_k))

This script intentionally removes the redundant modes from earlier scripts:
  - no hardmin secondary eval
  - no mixed/relative/CE losses
  - no Hungarian assignment
  - no confidence weighting
  - support-region selection uses no GT; GT is audit only
  - no class-bank rebuilding
  - no non-oracle mode
  - no W.T training definition
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def load_stage1_model(config_path: str, weights_path: str, device: torch.device):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    model_cls = getattr(importlib.import_module("src.model"), model_class_name)
    model = model_cls.from_config(config["model"])

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    ret = model.load_state_dict(state_dict, strict=False)
    print(f"[model] loaded: {weights_path}")
    print("[model] missing keys:", getattr(ret, "missing_keys", []))
    print("[model] unexpected keys:", getattr(ret, "unexpected_keys", []))

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def build_dataset(path: str, args, class_part_bank=None, min_obj_area_ratio: float = 0.0):
    return DinoClipJointDataset(
        path,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=False,
        is_wds=".tar" in path,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=min_obj_area_ratio,
        class_part_bank=class_part_bank,
    )


def build_loaders(args, config):
    dataset_cfg = config.get("dataset", {})
    train_min_area = float(dataset_cfg.get("min_obj_area_ratio", 0.0))

    train_set = build_dataset(args.train_dataset, args, class_part_bank=None, min_obj_area_ratio=train_min_area)
    val_set = build_dataset(args.val_dataset, args, class_part_bank=train_set.class_part_bank, min_obj_area_ratio=0.0)

    if train_set.part_taxonomy != val_set.part_taxonomy:
        raise ValueError(f"taxonomy mismatch: train={train_set.part_taxonomy}, val={val_set.part_taxonomy}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader


def move_batch(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def project_batch(model, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    part_z0 = model.project_clip_txt(batch["part_text_feat"].float())
    part_z0 = normalize(part_z0.float(), dim=-1)
    patch_z = normalize(batch["patch_tokens"].float(), dim=-1)
    return part_z0, patch_z


def infer_dim(model, loader, device: torch.device) -> int:
    batch = move_batch(next(iter(loader)), device)
    with torch.no_grad():
        part_z0, _ = project_batch(model, batch)
    return int(part_z0.shape[-1])


@torch.no_grad()
def orthogonalize_(W: torch.Tensor) -> None:
    u, _, vh = torch.linalg.svd(W.data, full_matrices=False)
    W.data.copy_(u @ vh)


def relative_scores(sim: torch.Tensor) -> torch.Tensor:
    """sim: [K, M]. rel[k,m] = sim[k,m] - best other-part score at patch m."""
    K, _ = sim.shape
    if K <= 1:
        return sim
    top2_vals, top2_idx = torch.topk(sim, k=2, dim=0)
    best_vals, second_vals = top2_vals[0], top2_vals[1]
    best_idx = top2_idx[0]
    row = torch.arange(K, device=sim.device)[:, None]
    best_other = torch.where(row == best_idx[None, :], second_vals[None, :], best_vals[None, :])
    return sim - best_other


@torch.no_grad()
def greedy_distinct_anchors(score: torch.Tensor) -> torch.Tensor:
    """score: [K, M]. Return one patch index per part, distinct when possible."""
    K, M = score.shape
    flat = score.reshape(-1)
    order = torch.argsort(flat, descending=True)
    anchor = torch.full((K,), -1, dtype=torch.long, device=score.device)
    patch_taken = torch.zeros((M,), dtype=torch.bool, device=score.device)
    assigned = 0
    for flat_id in order:
        k = torch.div(flat_id, M, rounding_mode="floor")
        m = flat_id % M
        if anchor[k] >= 0 or patch_taken[m]:
            continue
        anchor[k] = m
        patch_taken[m] = True
        assigned += 1
        if assigned == K:
            break
    missing = torch.nonzero(anchor < 0, as_tuple=False).flatten()
    if missing.numel() > 0:
        best = score.argmax(dim=1)
        anchor[missing] = best[missing]
    return anchor


def get_oracle_visible_sample(
    part_z0_b: torch.Tensor,
    patch_z_b: torch.Tensor,
    part_valid_b: torch.Tensor,
    part_gt_b: torch.Tensor,
    obj_mask_b: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return z_visible, object_patches, visible_gt_on_object, object_patch_global_idx."""
    obj_idx = torch.nonzero(obj_mask_b.bool(), as_tuple=False).flatten()
    if obj_idx.numel() == 0:
        return None

    obj_mask = obj_mask_b.bool().reshape(-1)
    gt = part_gt_b.bool()
    if gt.ndim != 2:
        gt = gt.reshape(gt.shape[0], -1)
    visible = (gt & obj_mask[None, :]).sum(dim=1) > 0
    visible = visible & part_valid_b.bool().reshape(-1)
    vis_idx = torch.nonzero(visible, as_tuple=False).flatten()
    if vis_idx.numel() == 0:
        return None

    z_vis = part_z0_b[vis_idx]
    p_obj = patch_z_b[obj_idx]
    gt_vis_obj = gt[vis_idx][:, obj_idx].bool()
    return z_vis, p_obj, gt_vis_obj, obj_idx


def build_pseudo_regions(
    sim: torch.Tensor,
    rel: torch.Tensor,
    seed_anchor: torch.Tensor,
    args,
) -> List[torch.Tensor]:
    """Build high-confidence pseudo-region support for each visible part.

    This function uses only model scores, not GT masks.
    For part k, a patch enters support if:
      1) k is the top-1 predicted part for that patch;
      2) rel[k, patch] >= support_rel_min.
    Then keep at most support_max_size patches ranked by rel[k].
    If support is empty, fallback to the single greedy seed anchor.
    """
    K, _ = sim.shape
    winner = sim.argmax(dim=0)
    supports: List[torch.Tensor] = []
    for k in range(K):
        mask = winner == k
        if args.support_rel_min is not None:
            mask = mask & (rel[k] >= float(args.support_rel_min))
        cand = torch.nonzero(mask, as_tuple=False).flatten()
        if cand.numel() > 0 and args.support_max_size > 0 and cand.numel() > args.support_max_size:
            _, order = torch.topk(rel[k, cand], k=int(args.support_max_size), largest=True)
            cand = cand[order]
        if cand.numel() < int(args.support_min_size):
            cand = seed_anchor[k:k + 1]
        supports.append(cand)
    return supports


def sample_region_loss(
    z_vis: torch.Tensor,
    p_obj: torch.Tensor,
    W: torch.Tensor,
    args,
    gt_vis_obj: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    z_w = normalize(z_vis @ W, dim=-1)
    sim = z_w @ p_obj.T
    rel = relative_scores(sim)
    with torch.no_grad():
        seed_anchor = greedy_distinct_anchors(rel)
        supports = build_pseudo_regions(sim, rel, seed_anchor, args)

    region_losses: List[torch.Tensor] = []
    anchor_losses: List[torch.Tensor] = []

    support_parts = 0
    support_slots = 0
    support_hit_slots = 0
    support_any_hits = 0
    fallback_parts = 0

    row = torch.arange(sim.shape[0], device=sim.device)
    selected_sim = sim[row, seed_anchor]

    for k, idx in enumerate(supports):
        if idx.numel() == 1 and int(idx[0].detach().cpu().item()) == int(seed_anchor[k].detach().cpu().item()):
            # This is either a true one-patch support or fallback. Count fallback only when
            # the high-confidence rule would have produced too few patches.
            winner = sim.argmax(dim=0)
            raw_mask = (winner == k) & (rel[k] >= float(args.support_rel_min))
            if int(raw_mask.sum().detach().cpu().item()) < int(args.support_min_size):
                fallback_parts += 1

        proto = normalize(p_obj[idx].mean(dim=0), dim=-1)
        region_losses.append(1.0 - (z_w[k] * proto).sum())
        anchor_losses.append(1.0 - selected_sim[k])

        support_parts += 1
        support_slots += int(idx.numel())
        if gt_vis_obj is not None:
            hit_vec = gt_vis_obj[k, idx].bool()
            support_hit_slots += int(hit_vec.float().sum().detach().cpu().item())
            support_any_hits += int(bool(hit_vec.any().detach().cpu().item()))

    if not region_losses:
        return W.new_tensor(0.0), {
            "support_parts": 0.0,
            "support_slots": 0.0,
            "support_slot_purity": 0.0,
            "support_any_hit_rate": 0.0,
            "fallback_parts": 0.0,
        }

    region_loss = torch.stack(region_losses).mean()
    anchor_loss = torch.stack(anchor_losses).mean()
    loss = region_loss + float(args.anchor_loss_weight) * anchor_loss

    return loss, {
        "support_parts": float(support_parts),
        "support_slots": float(support_slots),
        "support_slot_purity": float(support_hit_slots) / max(float(support_slots), 1.0),
        "support_any_hit_rate": float(support_any_hits) / max(float(support_parts), 1.0),
        "fallback_parts": float(fallback_parts),
    }


def batch_loss(model, batch: Dict, W: torch.Tensor, args) -> Tuple[torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        part_z0, patch_z = project_batch(model, batch)

    losses: List[torch.Tensor] = []
    used_samples = 0
    used_visible_parts = 0
    support_slots = 0.0
    support_parts = 0.0
    support_hit_slots = 0.0
    support_any_hits = 0.0
    fallback_parts = 0.0

    for b in range(part_z0.shape[0]):
        sample = get_oracle_visible_sample(
            part_z0[b],
            patch_z[b],
            batch["part_valid_mask"][b],
            batch["part_gt_mask_patch"][b],
            batch["obj_mask_patch"][b],
        )
        if sample is None:
            continue
        z_vis, p_obj, gt_vis_obj, _ = sample
        loss_b, stat_b = sample_region_loss(z_vis, p_obj, W, args, gt_vis_obj=gt_vis_obj)
        losses.append(loss_b)
        used_samples += 1
        used_visible_parts += int(z_vis.shape[0])
        support_slots += stat_b["support_slots"]
        support_parts += stat_b["support_parts"]
        support_hit_slots += stat_b["support_slot_purity"] * stat_b["support_slots"]
        support_any_hits += stat_b["support_any_hit_rate"] * stat_b["support_parts"]
        fallback_parts += stat_b["fallback_parts"]

    if not losses:
        return W.new_tensor(0.0), {
            "used_samples": 0.0,
            "visible_parts": 0.0,
            "support_slots": 0.0,
            "support_parts": 0.0,
            "support_slot_purity": 0.0,
            "support_any_hit_rate": 0.0,
            "mean_support_size": 0.0,
            "fallback_rate": 0.0,
        }
    return torch.stack(losses).mean(), {
        "used_samples": float(used_samples),
        "visible_parts": float(used_visible_parts),
        "support_slots": float(support_slots),
        "support_parts": float(support_parts),
        "support_slot_purity": float(support_hit_slots) / max(float(support_slots), 1.0),
        "support_any_hit_rate": float(support_any_hits) / max(float(support_parts), 1.0),
        "mean_support_size": float(support_slots) / max(float(support_parts), 1.0),
        "fallback_rate": float(fallback_parts) / max(float(support_parts), 1.0),
    }


def train_W(model, train_loader, device: torch.device, dim: int, args) -> Tuple[torch.Tensor, List[Dict]]:
    W = torch.nn.Parameter(torch.eye(dim, device=device))
    opt = torch.optim.Adam([W], lr=args.lr, weight_decay=0.0)
    history: List[Dict] = []

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_samples = 0.0
        total_parts = 0.0
        total_support_slots = 0.0
        total_support_parts = 0.0
        total_support_hit_slots = 0.0
        total_support_any_hits = 0.0
        total_fallback_parts = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"train W epoch {epoch}")
        for step, batch in enumerate(pbar):
            if args.max_train_batches > 0 and step >= args.max_train_batches:
                break
            batch = move_batch(batch, device)
            loss, info = batch_loss(model, batch, W, args)
            if info["used_samples"] <= 0:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            orthogonalize_(W)

            total_loss += float(loss.detach().cpu().item())
            total_samples += info["used_samples"]
            total_parts += info["visible_parts"]
            total_support_slots += info["support_slots"]
            total_support_parts += info["support_parts"]
            total_support_hit_slots += info["support_slot_purity"] * info["support_slots"]
            total_support_any_hits += info["support_any_hit_rate"] * info["support_parts"]
            total_fallback_parts += info["fallback_rate"] * info["support_parts"]
            steps += 1
            pbar.set_postfix(
                loss=total_loss / max(steps, 1),
                samples=int(total_samples),
                parts=int(total_parts),
                sup=f"{total_support_slots / max(total_support_parts, 1.0):.1f}",
                pur=f"{total_support_hit_slots / max(total_support_slots, 1.0):.3f}",
            )

        record = {
            "epoch": epoch,
            "loss": total_loss / max(steps, 1),
            "used_samples": total_samples,
            "visible_parts": total_parts,
            "mean_support_size": total_support_slots / max(total_support_parts, 1.0),
            "support_slot_purity": total_support_hit_slots / max(total_support_slots, 1.0),
            "support_any_hit_rate": total_support_any_hits / max(total_support_parts, 1.0),
            "fallback_rate": total_fallback_parts / max(total_support_parts, 1.0),
            "steps": steps,
        }
        print("[train]", json.dumps(record, ensure_ascii=False))
        history.append(record)
    return W.detach().clone(), history


class AnchorStats:
    def __init__(self) -> None:
        self.total = 0
        self.before_hit = 0
        self.after_hit = 0
        self.before_cost = 0.0
        self.after_cost = 0.0
        self.before_cos_gt = 0.0
        self.after_cos_gt = 0.0

    def update(self, bh: torch.Tensor, ah: torch.Tensor, bc: torch.Tensor, ac: torch.Tensor, bg: torch.Tensor, ag: torch.Tensor) -> None:
        n = int(bh.numel())
        self.total += n
        self.before_hit += int(bh.float().sum().item())
        self.after_hit += int(ah.float().sum().item())
        self.before_cost += float(bc.sum().item())
        self.after_cost += float(ac.sum().item())
        self.before_cos_gt += float(bg.sum().item())
        self.after_cos_gt += float(ag.sum().item())

    def to_dict(self) -> Dict[str, float]:
        d = max(self.total, 1)
        return {
            "total_visible_parts": self.total,
            "before_anchor_hit_rate": self.before_hit / d,
            "after_anchor_hit_rate": self.after_hit / d,
            "delta_anchor_hit_rate": (self.after_hit - self.before_hit) / d,
            "before_mean_anchor_cost": self.before_cost / d,
            "after_mean_anchor_cost": self.after_cost / d,
            "delta_mean_anchor_cost": (self.after_cost - self.before_cost) / d,
            "before_mean_cos_gt_proto": self.before_cos_gt / d,
            "after_mean_cos_gt_proto": self.after_cos_gt / d,
            "delta_mean_cos_gt_proto": (self.after_cos_gt - self.before_cos_gt) / d,
        }


class RegionStats:
    def __init__(self) -> None:
        self.total_parts = 0
        self.total_slots = 0
        self.hit_slots = 0
        self.any_hits = 0
        self.fallback_parts = 0

    def update(self, supports: List[torch.Tensor], gt_vis_obj: torch.Tensor, sim: torch.Tensor, rel: torch.Tensor, seed_anchor: torch.Tensor, args) -> None:
        winner = sim.argmax(dim=0)
        for k, idx in enumerate(supports):
            self.total_parts += 1
            self.total_slots += int(idx.numel())
            hit_vec = gt_vis_obj[k, idx].bool()
            self.hit_slots += int(hit_vec.float().sum().item())
            self.any_hits += int(bool(hit_vec.any().item()))
            raw_mask = (winner == k) & (rel[k] >= float(args.support_rel_min))
            if int(raw_mask.sum().item()) < int(args.support_min_size):
                self.fallback_parts += 1

    def to_dict(self) -> Dict[str, float]:
        return {
            "region_total_parts": self.total_parts,
            "region_total_slots": self.total_slots,
            "region_mean_support_size": self.total_slots / max(self.total_parts, 1),
            "region_slot_purity": self.hit_slots / max(self.total_slots, 1),
            "region_any_hit_rate": self.any_hits / max(self.total_parts, 1),
            "region_fallback_rate": self.fallback_parts / max(self.total_parts, 1),
        }


@torch.no_grad()
def evaluate_anchor(model, loader, W: torch.Tensor, device: torch.device, args) -> Dict[str, float]:
    stats = AnchorStats()
    region_before = RegionStats()
    region_after = RegionStats()
    W = W.to(device)
    pbar = tqdm(loader, desc="eval relative anchor")
    for step, batch in enumerate(pbar):
        if args.max_eval_batches > 0 and step >= args.max_eval_batches:
            break
        batch = move_batch(batch, device)
        part_z0, patch_z = project_batch(model, batch)
        for b in range(part_z0.shape[0]):
            sample = get_oracle_visible_sample(
                part_z0[b],
                patch_z[b],
                batch["part_valid_mask"][b],
                batch["part_gt_mask_patch"][b],
                batch["obj_mask_patch"][b],
            )
            if sample is None:
                continue
            z_vis, p_obj, gt_vis_obj, _ = sample

            z_after = normalize(z_vis @ W, dim=-1)
            sim_before = z_vis @ p_obj.T
            sim_after = z_after @ p_obj.T

            rel_before = relative_scores(sim_before)
            rel_after = relative_scores(sim_after)
            anchor_before = greedy_distinct_anchors(rel_before)
            anchor_after = greedy_distinct_anchors(rel_after)
            supports_before = build_pseudo_regions(sim_before, rel_before, anchor_before, args)
            supports_after = build_pseudo_regions(sim_after, rel_after, anchor_after, args)
            region_before.update(supports_before, gt_vis_obj, sim_before, rel_before, anchor_before, args)
            region_after.update(supports_after, gt_vis_obj, sim_after, rel_after, anchor_after, args)
            row = torch.arange(z_vis.shape[0], device=device)

            before_hit = gt_vis_obj[row, anchor_before]
            after_hit = gt_vis_obj[row, anchor_after]
            before_cost = 1.0 - sim_before[row, anchor_before]
            after_cost = 1.0 - sim_after[row, anchor_after]

            gt_proto = []
            for k in range(gt_vis_obj.shape[0]):
                m = gt_vis_obj[k]
                if m.any():
                    gt_proto.append(normalize(p_obj[m].mean(dim=0), dim=-1))
                else:
                    gt_proto.append(torch.zeros_like(z_vis[k]))
            gt_proto = torch.stack(gt_proto, dim=0)
            before_cos_gt = (z_vis * gt_proto).sum(dim=-1)
            after_cos_gt = (z_after * gt_proto).sum(dim=-1)

            stats.update(before_hit, after_hit, before_cost, after_cost, before_cos_gt, after_cos_gt)
        if stats.total > 0:
            d = stats.to_dict()
            pbar.set_postfix(before=f"{d['before_anchor_hit_rate']:.4f}", after=f"{d['after_anchor_hit_rate']:.4f}", parts=stats.total)
    out = stats.to_dict()
    out.update({f"before_{k}": v for k, v in region_before.to_dict().items()})
    out.update({f"after_{k}": v for k, v in region_after.to_dict().items()})
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--val_dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--path_prefix", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--support_rel_min", type=float, default=0.0, help="patch enters pseudo-region only if part is top-1 and rel >= this value")
    parser.add_argument("--support_min_size", type=int, default=1, help="fallback to greedy seed if support has fewer patches")
    parser.add_argument("--support_max_size", type=int, default=8, help="keep at most this many support patches per part, ranked by relative score")
    parser.add_argument("--anchor_loss_weight", type=float, default=0.0, help="optional auxiliary single-anchor selected-cost weight")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_stage1_model(args.model_config, args.weights, device)
    train_loader, val_loader = build_loaders(args, config)
    dim = infer_dim(model, train_loader, device)
    print(f"[setup] device={device}, dim={dim}")
    print("[setup] train definition: part_z = normalize(project_clip_txt(part_text) @ W)")
    print("[setup] loss: pseudo-region prototype cost")
    print(f"[setup] support: top1 part + rel >= {args.support_rel_min}, min={args.support_min_size}, max={args.support_max_size}, anchor_loss_weight={args.anchor_loss_weight}")

    identity_W = torch.eye(dim, device=device)
    print("[eval] identity W")
    before = evaluate_anchor(model, val_loader, identity_W, device, args)
    print(json.dumps({"identity_eval_relative": before}, indent=2))

    W, history = train_W(model, train_loader, device, dim, args)

    print("[eval] fitted W")
    after = evaluate_anchor(model, val_loader, W, device, args)
    print(json.dumps({"after_eval_relative": after}, indent=2))

    w_path = os.path.join(args.out_dir, "oracle_pseudoregion_W.pt")
    summary_path = os.path.join(args.out_dir, "summary.json")
    torch.save({"W": W.cpu(), "args": vars(args), "fit_history": history}, w_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "identity_eval_relative": before,
            "after_eval_relative": after,
            "fit_history": history,
            "notes": {
                "oracle_visible": "GT masks are used only to choose visible parts and audit anchor hits.",
                "W_definition": "part_z = normalize(project_clip_txt(part_text) @ W)",
                "objective": "relative greedy seed + high-confidence pseudo-region prototype cost",
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"[save] W: {w_path}")
    print(f"[save] summary: {summary_path}")
    print("Key numbers [pseudo-region W, evaluated by relative single-anchor audit]:")
    print(f"  before = {before['before_anchor_hit_rate']:.6f}")
    print(f"  after  = {after['after_anchor_hit_rate']:.6f}")
    print(f"  delta  = {after['after_anchor_hit_rate'] - before['before_anchor_hit_rate']:.6f}")


if __name__ == "__main__":
    main()
