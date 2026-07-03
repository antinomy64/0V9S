#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean test15 oracle-visible orthogonal W training, multi-anchor version.

This is a minimal extension of oracle_visible_w_clean_single_anchor.py.
Only one thing changes:
  single anchor per visible part  ->  K anchors per visible part

Kept fixed from the verified best-W pipeline:
  - Stage1 text projector is frozen.
  - V-side cropaug_patch_tokens stay in DINO/V space and are only normalized.
  - Only a global orthogonal W is trained.
  - GT part masks are used only to choose oracle-visible positive parts and to audit anchor hits.
  - Anchor selection and loss do not use GT patch locations.
  - W definition is z_after = normalize(z_before @ W), NOT z_before @ W.T.
  - Relative score is used for anchor assignment.
  - Greedy distinct patch assignment is used.
  - Loss is selected-cost only: mean(1 - selected_sim).

Multi-anchor definition
-----------------------
For each sample:
  z0 = normalize(stage1.project_clip_txt(part_text))          # [K, D]
  z  = normalize(z0 @ W)                                      # [K, D]
  p  = normalize(cropaug_patch_tokens[obj_mask])              # [M, D]
  sim = z @ p.T                                               # [K, M]
  rel[k,m] = sim[k,m] - max_{q != k} sim[q,m]

Then greedy_distinct_multi_anchors(rel, A) selects up to A distinct object patches
for every visible part, while trying not to reuse the same patch across parts.
The training loss is part-balanced:
  loss = mean_k mean_{a in anchors(k)} (1 - sim[k, a])

Recommended first sweeps:
  --anchors_per_part 2
  --anchors_per_part 3
  --anchors_per_part 5

Interpretation:
  If multi-anchor improves mIoU over single-anchor, the bottleneck was likely
  insufficient region/support coverage. If anchor-set hit improves but mIoU does
  not, the pipeline can find more correct seed patches but dense argmax remains
  the bottleneck.
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
def greedy_distinct_multi_anchors(score: torch.Tensor, anchors_per_part: int) -> torch.Tensor:
    """Select up to A distinct anchors per part using global greedy assignment.

    Args:
        score: [K, M], usually relative score. Higher is better.
        anchors_per_part: desired anchors per visible part.

    Returns:
        anchors: [K, A_eff] local object-patch indices. A_eff = min(anchors_per_part, M).

    Rule:
      - Traverse all (part, patch) pairs by descending score.
      - Assign a pair if that part still needs anchors and that patch has not
        been globally used.
      - If some part still lacks anchors after global distinct assignment,
        fill from that part's own best remaining patches. This fallback is only
        for degenerate cases where object patches are too few or scores are tied.
    """
    if anchors_per_part <= 0:
        raise ValueError("anchors_per_part must be positive")
    K, M = score.shape
    A = min(int(anchors_per_part), int(M))
    anchors = torch.full((K, A), -1, dtype=torch.long, device=score.device)
    counts = torch.zeros((K,), dtype=torch.long, device=score.device)
    patch_taken = torch.zeros((M,), dtype=torch.bool, device=score.device)

    flat = score.reshape(-1)
    order = torch.argsort(flat, descending=True)
    target_total = K * A
    assigned = 0
    for flat_id in order:
        k = torch.div(flat_id, M, rounding_mode="floor")
        m = flat_id % M
        if counts[k] >= A:
            continue
        if patch_taken[m]:
            continue
        anchors[k, counts[k]] = m
        counts[k] += 1
        patch_taken[m] = True
        assigned += 1
        if assigned == target_total:
            break

    # Robust fallback. Normally not used because M_obj is much larger than K*A.
    for k in range(K):
        if counts[k] >= A:
            continue
        order_k = torch.argsort(score[k], descending=True)
        already = set(int(x.item()) for x in anchors[k, : counts[k]].detach().cpu() if int(x.item()) >= 0)
        for m in order_k:
            mi = int(m.item())
            if mi in already:
                continue
            anchors[k, counts[k]] = m
            counts[k] += 1
            already.add(mi)
            if counts[k] >= A:
                break
        if counts[k] < A:
            # This should only happen if M == 0, already excluded by caller.
            raise RuntimeError(f"Could not fill anchors for part {k}: count={int(counts[k].item())}, A={A}, M={M}")
    return anchors


@torch.no_grad()
def greedy_distinct_single_anchor(score: torch.Tensor) -> torch.Tensor:
    return greedy_distinct_multi_anchors(score, anchors_per_part=1).squeeze(1)


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


def sample_loss(z_vis: torch.Tensor, p_obj: torch.Tensor, W: torch.Tensor, anchors_per_part: int) -> torch.Tensor:
    z_w = normalize(z_vis @ W, dim=-1)
    sim = z_w @ p_obj.T
    rel = relative_scores(sim)
    with torch.no_grad():
        anchors = greedy_distinct_multi_anchors(rel, anchors_per_part=anchors_per_part)  # [K, A]
    K, A = anchors.shape
    row = torch.arange(K, device=sim.device)[:, None].expand(K, A)
    selected_sim = sim[row, anchors]  # [K, A]
    # Part-balanced selected-cost: each part contributes equally.
    per_part_loss = (1.0 - selected_sim).mean(dim=1)
    return per_part_loss.mean()


def batch_loss(model, batch: Dict, W: torch.Tensor, anchors_per_part: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        part_z0, patch_z = project_batch(model, batch)

    losses: List[torch.Tensor] = []
    used_samples = 0
    used_visible_parts = 0
    used_anchors = 0

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
        z_vis, p_obj, _, _ = sample
        losses.append(sample_loss(z_vis, p_obj, W, anchors_per_part=anchors_per_part))
        used_samples += 1
        used_visible_parts += int(z_vis.shape[0])
        used_anchors += int(z_vis.shape[0] * min(anchors_per_part, p_obj.shape[0]))

    if not losses:
        return W.new_tensor(0.0), {"used_samples": 0.0, "visible_parts": 0.0, "used_anchors": 0.0}
    return torch.stack(losses).mean(), {
        "used_samples": float(used_samples),
        "visible_parts": float(used_visible_parts),
        "used_anchors": float(used_anchors),
    }


def train_W(model, train_loader, device: torch.device, dim: int, args) -> Tuple[torch.Tensor, List[Dict]]:
    W = torch.nn.Parameter(torch.eye(dim, device=device))
    opt = torch.optim.Adam([W], lr=args.lr, weight_decay=0.0)
    history: List[Dict] = []

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_samples = 0.0
        total_parts = 0.0
        total_anchors = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"train multi-anchor W epoch {epoch}")
        for step, batch in enumerate(pbar):
            if args.max_train_batches > 0 and step >= args.max_train_batches:
                break
            batch = move_batch(batch, device)
            loss, info = batch_loss(model, batch, W, anchors_per_part=args.anchors_per_part)
            if info["used_samples"] <= 0:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            orthogonalize_(W)

            total_loss += float(loss.detach().cpu().item())
            total_samples += info["used_samples"]
            total_parts += info["visible_parts"]
            total_anchors += info["used_anchors"]
            steps += 1
            pbar.set_postfix(
                loss=total_loss / max(steps, 1),
                samples=int(total_samples),
                parts=int(total_parts),
                anchors=int(total_anchors),
            )

        record = {
            "epoch": epoch,
            "loss": total_loss / max(steps, 1),
            "used_samples": total_samples,
            "visible_parts": total_parts,
            "used_anchors": total_anchors,
            "anchors_per_part": args.anchors_per_part,
            "steps": steps,
        }
        print("[train]", json.dumps(record, ensure_ascii=False))
        history.append(record)
    return W.detach().clone(), history


class MultiAnchorStats:
    def __init__(self) -> None:
        self.total_parts = 0
        self.before_part_any_hit = 0
        self.after_part_any_hit = 0
        self.total_anchor_slots = 0
        self.before_anchor_hits = 0
        self.after_anchor_hits = 0
        self.before_cost = 0.0
        self.after_cost = 0.0
        self.before_cos_gt = 0.0
        self.after_cos_gt = 0.0

    def update(
        self,
        before_anchor_hit: torch.Tensor,  # [K,A]
        after_anchor_hit: torch.Tensor,   # [K,A]
        before_cost: torch.Tensor,        # [K,A]
        after_cost: torch.Tensor,         # [K,A]
        before_cos_gt: torch.Tensor,      # [K]
        after_cos_gt: torch.Tensor,       # [K]
    ) -> None:
        K, A = before_anchor_hit.shape
        self.total_parts += int(K)
        self.before_part_any_hit += int(before_anchor_hit.any(dim=1).float().sum().item())
        self.after_part_any_hit += int(after_anchor_hit.any(dim=1).float().sum().item())
        self.total_anchor_slots += int(K * A)
        self.before_anchor_hits += int(before_anchor_hit.float().sum().item())
        self.after_anchor_hits += int(after_anchor_hit.float().sum().item())
        self.before_cost += float(before_cost.sum().item())
        self.after_cost += float(after_cost.sum().item())
        self.before_cos_gt += float(before_cos_gt.sum().item())
        self.after_cos_gt += float(after_cos_gt.sum().item())

    def to_dict(self) -> Dict[str, float]:
        part_d = max(self.total_parts, 1)
        anchor_d = max(self.total_anchor_slots, 1)
        return {
            "total_visible_parts": self.total_parts,
            "total_anchor_slots": self.total_anchor_slots,
            "before_part_any_hit_rate": self.before_part_any_hit / part_d,
            "after_part_any_hit_rate": self.after_part_any_hit / part_d,
            "delta_part_any_hit_rate": (self.after_part_any_hit - self.before_part_any_hit) / part_d,
            "before_anchor_slot_hit_rate": self.before_anchor_hits / anchor_d,
            "after_anchor_slot_hit_rate": self.after_anchor_hits / anchor_d,
            "delta_anchor_slot_hit_rate": (self.after_anchor_hits - self.before_anchor_hits) / anchor_d,
            "before_mean_anchor_cost": self.before_cost / anchor_d,
            "after_mean_anchor_cost": self.after_cost / anchor_d,
            "delta_mean_anchor_cost": (self.after_cost - self.before_cost) / anchor_d,
            "before_mean_cos_gt_proto": self.before_cos_gt / part_d,
            "after_mean_cos_gt_proto": self.after_cos_gt / part_d,
            "delta_mean_cos_gt_proto": (self.after_cos_gt - self.before_cos_gt) / part_d,
        }


@torch.no_grad()
def evaluate_multi_anchor(model, loader, W: torch.Tensor, device: torch.device, args, anchors_per_part: int) -> Dict[str, float]:
    stats = MultiAnchorStats()
    W = W.to(device)
    pbar = tqdm(loader, desc=f"eval relative multi-anchor A={anchors_per_part}")
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

            anchors_before = greedy_distinct_multi_anchors(rel_before, anchors_per_part=anchors_per_part)
            anchors_after = greedy_distinct_multi_anchors(rel_after, anchors_per_part=anchors_per_part)
            K, A = anchors_before.shape
            row = torch.arange(K, device=device)[:, None].expand(K, A)

            before_hit = gt_vis_obj[row, anchors_before]
            after_hit = gt_vis_obj[row, anchors_after]
            before_cost = 1.0 - sim_before[row, anchors_before]
            after_cost = 1.0 - sim_after[row, anchors_after]

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
        if stats.total_parts > 0:
            d = stats.to_dict()
            pbar.set_postfix(
                before_any=f"{d['before_part_any_hit_rate']:.4f}",
                after_any=f"{d['after_part_any_hit_rate']:.4f}",
                slot=f"{d['after_anchor_slot_hit_rate']:.4f}",
                parts=stats.total_parts,
            )
    return stats.to_dict()


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
    parser.add_argument("--anchors_per_part", type=int, default=3, help="number of selected anchors per visible part")
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
    print("[setup] loss: multi-anchor selected_cost = mean_k mean_a (1 - sim[k, anchor_{k,a}])")
    print(f"[setup] anchors_per_part={args.anchors_per_part}")

    identity_W = torch.eye(dim, device=device)
    print("[eval] identity W")
    before = evaluate_multi_anchor(model, val_loader, identity_W, device, args, anchors_per_part=args.anchors_per_part)
    print(json.dumps({"identity_eval_relative_multi_anchor": before}, indent=2))

    W, history = train_W(model, train_loader, device, dim, args)

    print("[eval] fitted W")
    after = evaluate_multi_anchor(model, val_loader, W, device, args, anchors_per_part=args.anchors_per_part)
    print(json.dumps({"after_eval_relative_multi_anchor": after}, indent=2))

    w_path = os.path.join(args.out_dir, "oracle_relative_multi_anchor_W.pt")
    summary_path = os.path.join(args.out_dir, "summary.json")
    torch.save({"W": W.cpu(), "args": vars(args), "fit_history": history}, w_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "identity_eval_relative_multi_anchor": before,
            "after_eval_relative_multi_anchor": after,
            "fit_history": history,
            "notes": {
                "oracle_visible": "GT masks are used only to choose visible parts and audit anchor hits.",
                "W_definition": "part_z = normalize(project_clip_txt(part_text) @ W)",
                "objective": "relative greedy distinct multi-anchor + selected cost only",
                "anchor_metrics": {
                    "part_any_hit_rate": "whether at least one selected anchor for a part hits that part GT mask",
                    "anchor_slot_hit_rate": "fraction of selected anchor slots that hit GT mask",
                },
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"[save] W: {w_path}")
    print(f"[save] summary: {summary_path}")
    print("Key numbers [relative multi-anchor]:")
    print(f"  before_any = {before['before_part_any_hit_rate']:.6f}")
    print(f"  after_any  = {after['after_part_any_hit_rate']:.6f}")
    print(f"  delta_any  = {after['after_part_any_hit_rate'] - before['before_part_any_hit_rate']:.6f}")
    print(f"  before_slot = {before['before_anchor_slot_hit_rate']:.6f}")
    print(f"  after_slot  = {after['after_anchor_slot_hit_rate']:.6f}")
    print(f"  delta_slot  = {after['after_anchor_slot_hit_rate'] - before['before_anchor_slot_hit_rate']:.6f}")


if __name__ == "__main__":
    main()
