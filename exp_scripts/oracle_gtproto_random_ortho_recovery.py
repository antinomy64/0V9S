#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GT-prototype random-orthogonal recovery experiment for test15 W pipeline.

Goal
----
Test whether the current oracle-visible selected-anchor W pipeline can recover from
an artificial orthogonal perturbation when the input features are ideal visual
GT prototypes.

Synthetic input definition
--------------------------
For each visible part in each object crop:
  gt_proto = normalize(mean DINO patch token inside GT part mask)
  z0       = normalize(gt_proto @ R)        # R is a fixed random orthogonal matrix
  z        = normalize(z0 @ W)              # train W only

If the pipeline can invert the perturbation, W should behave like R.T and
z should align back to gt_proto.

Important
---------
This script is an anchor/recovery audit, not a normal baked-projector mIoU eval.
The synthetic z0 is image-dependent GT prototype, so it cannot be baked into a
static text projector for the standard OVPS evaluator.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
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


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def get_patch_z(batch: Dict) -> torch.Tensor:
    return normalize(batch["patch_tokens"].float(), dim=-1)


def infer_dim(loader, device: torch.device) -> int:
    batch = move_batch(next(iter(loader)), device)
    patch_z = get_patch_z(batch)
    return int(patch_z.shape[-1])


@torch.no_grad()
def random_orthogonal(dim: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    A = torch.randn(dim, dim, generator=g, device=device)
    # QR gives an orthogonal matrix. Flip signs deterministically so the result is stable.
    Q, R = torch.linalg.qr(A, mode="reduced")
    signs = torch.sign(torch.diag(R))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    Q = Q * signs[None, :]
    return Q.contiguous()


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


def get_gtproto_visible_sample(
    patch_z_b: torch.Tensor,
    part_valid_b: torch.Tensor,
    part_gt_b: torch.Tensor,
    obj_mask_b: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return gt_proto_visible, object_patches, visible_gt_on_object."""
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

    p_obj = patch_z_b[obj_idx]
    gt_vis_obj = gt[vis_idx][:, obj_idx].bool()

    gt_proto = []
    for k in range(gt_vis_obj.shape[0]):
        m = gt_vis_obj[k]
        if not m.any():
            # Should not happen because of visible filtering.
            continue
        gt_proto.append(normalize(p_obj[m].mean(dim=0), dim=-1))
    if len(gt_proto) == 0:
        return None
    gt_proto = torch.stack(gt_proto, dim=0)
    return gt_proto, p_obj, gt_vis_obj


def sample_loss(gt_proto: torch.Tensor, p_obj: torch.Tensor, R: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    z0 = normalize(gt_proto @ R, dim=-1)
    z = normalize(z0 @ W, dim=-1)
    sim = z @ p_obj.T
    rel = relative_scores(sim)
    with torch.no_grad():
        anchor = greedy_distinct_anchors(rel)
    row = torch.arange(sim.shape[0], device=sim.device)
    selected_sim = sim[row, anchor]
    return (1.0 - selected_sim).mean()


def batch_loss(batch: Dict, R: torch.Tensor, W: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        patch_z = get_patch_z(batch)

    losses: List[torch.Tensor] = []
    used_samples = 0
    used_visible_parts = 0

    for b in range(patch_z.shape[0]):
        sample = get_gtproto_visible_sample(
            patch_z[b],
            batch["part_valid_mask"][b],
            batch["part_gt_mask_patch"][b],
            batch["obj_mask_patch"][b],
        )
        if sample is None:
            continue
        gt_proto, p_obj, _ = sample
        losses.append(sample_loss(gt_proto, p_obj, R, W))
        used_samples += 1
        used_visible_parts += int(gt_proto.shape[0])

    if not losses:
        return W.new_tensor(0.0), {"used_samples": 0.0, "visible_parts": 0.0}
    return torch.stack(losses).mean(), {"used_samples": float(used_samples), "visible_parts": float(used_visible_parts)}


def train_W(train_loader, device: torch.device, dim: int, R: torch.Tensor, args) -> Tuple[torch.Tensor, List[Dict]]:
    W = torch.nn.Parameter(torch.eye(dim, device=device))
    opt = torch.optim.Adam([W], lr=args.lr, weight_decay=0.0)
    history: List[Dict] = []

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_samples = 0.0
        total_parts = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"train recovery W epoch {epoch}")
        for step, batch in enumerate(pbar):
            if args.max_train_batches > 0 and step >= args.max_train_batches:
                break
            batch = move_batch(batch, device)
            loss, info = batch_loss(batch, R, W)
            if info["used_samples"] <= 0:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            orthogonalize_(W)

            total_loss += float(loss.detach().cpu().item())
            total_samples += info["used_samples"]
            total_parts += info["visible_parts"]
            steps += 1
            pbar.set_postfix(loss=total_loss / max(steps, 1), samples=int(total_samples), parts=int(total_parts))

        record = {
            "epoch": epoch,
            "loss": total_loss / max(steps, 1),
            "used_samples": total_samples,
            "visible_parts": total_parts,
            "steps": steps,
        }
        print("[train]", json.dumps(record, ensure_ascii=False))
        history.append(record)
    return W.detach().clone(), history


class AnchorStats:
    def __init__(self) -> None:
        self.total = 0
        self.hit = 0
        self.cost = 0.0
        self.cos_gt = 0.0

    def update(self, hit: torch.Tensor, cost: torch.Tensor, cos_gt: torch.Tensor) -> None:
        n = int(hit.numel())
        self.total += n
        self.hit += int(hit.float().sum().item())
        self.cost += float(cost.sum().item())
        self.cos_gt += float(cos_gt.sum().item())

    def to_dict(self) -> Dict[str, float]:
        d = max(self.total, 1)
        return {
            "total_visible_parts": self.total,
            "anchor_hit_rate": self.hit / d,
            "mean_anchor_cost": self.cost / d,
            "mean_cos_gt_proto": self.cos_gt / d,
        }


@torch.no_grad()
def evaluate_one(loader, R: torch.Tensor, W: torch.Tensor, device: torch.device, args, desc: str) -> Dict[str, float]:
    stats = AnchorStats()
    W = W.to(device)
    R = R.to(device)
    pbar = tqdm(loader, desc=desc)
    for step, batch in enumerate(pbar):
        if args.max_eval_batches > 0 and step >= args.max_eval_batches:
            break
        batch = move_batch(batch, device)
        patch_z = get_patch_z(batch)
        for b in range(patch_z.shape[0]):
            sample = get_gtproto_visible_sample(
                patch_z[b],
                batch["part_valid_mask"][b],
                batch["part_gt_mask_patch"][b],
                batch["obj_mask_patch"][b],
            )
            if sample is None:
                continue
            gt_proto, p_obj, gt_vis_obj = sample
            z0 = normalize(gt_proto @ R, dim=-1)
            z = normalize(z0 @ W, dim=-1)
            sim = z @ p_obj.T
            anchor = greedy_distinct_anchors(relative_scores(sim))
            row = torch.arange(gt_proto.shape[0], device=device)
            hit = gt_vis_obj[row, anchor]
            cost = 1.0 - sim[row, anchor]
            cos_gt = (z * gt_proto).sum(dim=-1)
            stats.update(hit, cost, cos_gt)
        if stats.total > 0:
            d = stats.to_dict()
            pbar.set_postfix(hit=f"{d['anchor_hit_rate']:.4f}", cost=f"{d['mean_anchor_cost']:.4f}", cos=f"{d['mean_cos_gt_proto']:.4f}", parts=stats.total)
    return stats.to_dict()


@torch.no_grad()
def recovery_metrics(W: torch.Tensor, R: torch.Tensor) -> Dict[str, float]:
    # Row-vector convention: z0 = gt @ R, target inverse is W* = R.T.
    target = R.T
    Wc = W.to(target.device)
    dim = Wc.shape[0]
    frob = torch.linalg.norm(Wc - target, ord="fro").item()
    rel_frob = frob / (dim ** 0.5)
    matrix_cos = (Wc * target).sum().item() / max(float(dim), 1.0)
    # Check composition R @ W should be close to identity.
    I = torch.eye(dim, device=target.device)
    compose_error = torch.linalg.norm(R @ Wc - I, ord="fro").item() / (dim ** 0.5)
    return {
        "target_is_R_T": True,
        "frob_W_minus_R_T_over_sqrtD": rel_frob,
        "matrix_cos_W_with_R_T": matrix_cos,
        "frob_R_W_minus_I_over_sqrtD": compose_error,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--val_dataset", required=True)
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
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--random_ortho_seed", type=int, default=777)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.model_config)
    train_loader, val_loader = build_loaders(args, config)
    dim = infer_dim(train_loader, device)
    R = random_orthogonal(dim, device=device, seed=args.random_ortho_seed)
    identity_W = torch.eye(dim, device=device)
    oracle_inverse_W = R.T.contiguous()

    print(f"[setup] device={device}, dim={dim}")
    print("[setup] synthetic input: z0 = normalize(gt_proto @ R)")
    print("[setup] train definition: z = normalize(z0 @ W); target inverse is W = R.T")
    print("[setup] objective: relative greedy distinct anchor + selected cost only")

    print("[eval] identity W on random-rotated GT prototypes")
    before = evaluate_one(val_loader, R, identity_W, device, args, desc="eval identity W")
    print(json.dumps({"identity_eval_rotated_gtproto": before}, indent=2))

    print("[eval] oracle inverse W = R.T")
    oracle = evaluate_one(val_loader, R, oracle_inverse_W, device, args, desc="eval oracle inverse R.T")
    print(json.dumps({"oracle_inverse_eval": oracle}, indent=2))

    W, history = train_W(train_loader, device, dim, R, args)

    print("[eval] fitted W")
    after = evaluate_one(val_loader, R, W, device, args, desc="eval fitted W")
    print(json.dumps({"after_eval_recovery": after}, indent=2))

    rec = recovery_metrics(W, R)
    print(json.dumps({"matrix_recovery_metrics": rec}, indent=2))

    w_path = os.path.join(args.out_dir, "gtproto_random_ortho_recovery_W.pt")
    summary_path = os.path.join(args.out_dir, "summary.json")
    torch.save({"W": W.cpu(), "R": R.cpu(), "args": vars(args), "fit_history": history}, w_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "identity_eval_rotated_gtproto": before,
                "oracle_inverse_eval": oracle,
                "after_eval_recovery": after,
                "matrix_recovery_metrics": rec,
                "fit_history": history,
                "notes": {
                    "synthetic_input": "z0 = normalize(gt_proto @ R), where gt_proto is GT mask mean DINO patch prototype.",
                    "W_definition": "z = normalize(z0 @ W), so the target inverse is R.T.",
                    "objective": "relative greedy distinct anchor + selected cost only",
                    "not_bakeable": "This input is image-dependent GT prototype, not a static text projector.",
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[save] W/R: {w_path}")
    print(f"[save] summary: {summary_path}")
    print("Key numbers [GT prototype random-orthogonal recovery]:")
    print(f"  identity_hit = {before['anchor_hit_rate']:.6f}")
    print(f"  oracle_inv_hit = {oracle['anchor_hit_rate']:.6f}")
    print(f"  fitted_hit  = {after['anchor_hit_rate']:.6f}")
    print(f"  fitted_cos_gt = {after['mean_cos_gt_proto']:.6f}")
    print(f"  matrix_cos(W, R.T) = {rec['matrix_cos_W_with_R_T']:.6f}")
    print(f"  ||R@W-I||_F/sqrt(D) = {rec['frob_R_W_minus_I_over_sqrtD']:.6f}")


if __name__ == "__main__":
    main()
