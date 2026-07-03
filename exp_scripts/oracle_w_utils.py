#!/usr/bin/env python
"""Shared utilities for test15 oracle-visible orthogonal-W audits."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import csv
import importlib
import random
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def load_model(config_path: str, weights_path: str, device: torch.device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"])
    ckpt = torch.load(weights_path, map_location="cpu")
    ret = model.load_state_dict(ckpt, strict=False)
    print(f"Loaded projector weights from {weights_path}")
    if ret is not None:
        print("Missing keys:", getattr(ret, "missing_keys", []))
        print("Unexpected keys:", getattr(ret, "unexpected_keys", []))
    model.to(device)
    model.eval()
    return model


def build_datasets(args):
    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)
    dataset_cfg = config.get("dataset", {})
    min_obj_area_ratio = float(dataset_cfg.get("min_obj_area_ratio", 0.0))

    train_dataset = DinoClipJointDataset(
        args.train_dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=".tar" in args.train_dataset,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=min_obj_area_ratio,
    )
    val_dataset = DinoClipJointDataset(
        args.val_dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=".tar" in args.val_dataset,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=0.0,
        class_part_bank=train_dataset.class_part_bank,
    )
    if train_dataset.part_taxonomy != val_dataset.part_taxonomy:
        raise ValueError(
            f"Train/val taxonomy mismatch: train={train_dataset.part_taxonomy}, "
            f"val={val_dataset.part_taxonomy}"
        )
    return train_dataset, val_dataset


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=joint_collate_fn,
    )


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


@torch.no_grad()
def project_parts(model, part_text_feat: torch.Tensor, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    part_proj = model.project_clip_txt(part_text_feat.float())
    part_proj = safe_normalize(part_proj, dim=-1)
    patch_tokens = safe_normalize(patch_tokens.float(), dim=-1)
    return part_proj, patch_tokens


def infer_projected_dim(model, loader, device: torch.device) -> int:
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        part_proj = model.project_clip_txt(batch["part_text_feat"].float())
    return int(part_proj.shape[-1])


def orthogonalize_(W: torch.Tensor) -> None:
    with torch.no_grad():
        U, _, Vh = torch.linalg.svd(W.data, full_matrices=False)
        W.data.copy_(U @ Vh)


def compute_relative_scores(local_scores: torch.Tensor) -> torch.Tensor:
    """Relative score used by joint anchor selection.

    local_scores: [K, M], K visible parts, M object-mask patches.
    rel[k, m] = score[k, m] - best score of any other part at patch m.
    """
    K, M = local_scores.shape
    if K <= 1:
        return local_scores
    top2_vals, top2_idx = torch.topk(local_scores, k=min(2, K), dim=0)
    best_vals = top2_vals[0]
    best_idx = top2_idx[0]
    second_vals = top2_vals[1]
    row_ids = torch.arange(K, device=local_scores.device)[:, None]
    is_top1 = row_ids == best_idx[None, :]
    best_other = torch.where(is_top1, second_vals[None, :], best_vals[None, :])
    return local_scores - best_other


@torch.no_grad()
def greedy_distinct_anchors(score_for_assignment: torch.Tensor) -> torch.Tensor:
    """Greedy one-anchor-per-part with distinct patches when possible.

    Mirrors the current joint pipeline's patch_taken greedy assignment.
    score_for_assignment: [K, M], higher is better.
    Returns anchor indices in local object-patch coordinates: [K].
    """
    K, M = score_for_assignment.shape
    flat_scores = score_for_assignment.reshape(-1)
    sorted_idx = torch.argsort(flat_scores, descending=True)
    anchor_idx = torch.full((K,), -1, dtype=torch.long, device=score_for_assignment.device)
    patch_taken = torch.zeros((M,), dtype=torch.bool, device=score_for_assignment.device)
    assigned = 0
    for flat_id in sorted_idx:
        k = torch.div(flat_id, M, rounding_mode="floor")
        m = flat_id % M
        if anchor_idx[k] != -1:
            continue
        if patch_taken[m]:
            continue
        anchor_idx[k] = m
        patch_taken[m] = True
        assigned += 1
        if assigned == K:
            break
    unassigned = torch.nonzero(anchor_idx < 0, as_tuple=False).flatten()
    if unassigned.numel() > 0:
        local_best = score_for_assignment.argmax(dim=1)
        anchor_idx[unassigned] = local_best[unassigned]
    return anchor_idx


def visible_sample_tensors(
    part_proj_b: torch.Tensor,
    patch_tokens_b: torch.Tensor,
    part_valid_b: torch.Tensor,
    part_gt_mask_b: torch.Tensor,
    obj_mask_b: torch.Tensor,
    part_category_b: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    obj_patch_global_idx = torch.nonzero(obj_mask_b.bool(), as_tuple=False).flatten()
    if obj_patch_global_idx.numel() == 0:
        return None
    visible_mask = (part_gt_mask_b.bool() & obj_mask_b.bool()[None, :]).sum(dim=-1) > 0
    visible_mask = visible_mask & part_valid_b.bool()
    vis_idx = torch.nonzero(visible_mask, as_tuple=False).flatten()
    if vis_idx.numel() == 0:
        return None
    z_vis = part_proj_b[vis_idx]
    p_obj = patch_tokens_b[obj_patch_global_idx]
    gt_vis_obj = part_gt_mask_b[vis_idx][:, obj_patch_global_idx].bool()
    part_ids = part_category_b[vis_idx].long()
    return z_vis, p_obj, gt_vis_obj, part_ids, obj_patch_global_idx


def select_anchors(sim: torch.Tensor, anchor_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return anchor indices and score matrix used for assignment.

    sim: [K, M], cosine similarity.
    anchor_mode:
      hardmin  -> independent argmax per part.
      relative -> joint-style relative score + greedy distinct patch.
    """
    if anchor_mode == "hardmin":
        return sim.argmax(dim=1), sim
    if anchor_mode == "relative":
        rel = compute_relative_scores(sim)
        return greedy_distinct_anchors(rel), rel
    raise ValueError(f"Unknown anchor_mode={anchor_mode}")


class CounterStats:
    def __init__(self):
        self.total = 0
        self.before_hits = 0
        self.after_hits = 0
        self.before_cost_sum = 0.0
        self.after_cost_sum = 0.0
        self.before_cos_gt_sum = 0.0
        self.after_cos_gt_sum = 0.0

    def update(self, before_hit: bool, after_hit: bool, before_cost: float, after_cost: float,
               before_cos_gt: Optional[float] = None, after_cos_gt: Optional[float] = None):
        self.total += 1
        self.before_hits += int(bool(before_hit))
        self.after_hits += int(bool(after_hit))
        self.before_cost_sum += float(before_cost)
        self.after_cost_sum += float(after_cost)
        if before_cos_gt is not None:
            self.before_cos_gt_sum += float(before_cos_gt)
        if after_cos_gt is not None:
            self.after_cos_gt_sum += float(after_cos_gt)

    def to_dict(self) -> Dict[str, float]:
        denom = max(self.total, 1)
        return {
            "total_visible_parts": self.total,
            "before_anchor_hit_rate": self.before_hits / denom,
            "after_anchor_hit_rate": self.after_hits / denom,
            "delta_anchor_hit_rate": (self.after_hits - self.before_hits) / denom,
            "before_mean_anchor_cost": self.before_cost_sum / denom,
            "after_mean_anchor_cost": self.after_cost_sum / denom,
            "delta_mean_anchor_cost": (self.after_cost_sum - self.before_cost_sum) / denom,
            "before_mean_cos_gt_proto": self.before_cos_gt_sum / denom,
            "after_mean_cos_gt_proto": self.after_cos_gt_sum / denom,
            "delta_mean_cos_gt_proto": (self.after_cos_gt_sum - self.before_cos_gt_sum) / denom,
        }


@torch.no_grad()
def evaluate_W_anchor(
    model,
    data_loader,
    W: Union[torch.Tensor, Dict[int, torch.Tensor]],
    device: torch.device,
    args,
    anchor_mode: str = "relative",
):
    overall = CounterStats()
    per_object: Dict[int, CounterStats] = defaultdict(CounterStats)
    per_part: Dict[int, CounterStats] = defaultdict(CounterStats)

    def get_W(obj_id: int) -> torch.Tensor:
        if isinstance(W, dict):
            return W.get(int(obj_id), W.get("default"))
        return W

    pbar = tqdm(data_loader, desc=f"eval W [{anchor_mode}]")
    for step, batch in enumerate(pbar):
        if getattr(args, "max_eval_batches", 0) > 0 and step >= args.max_eval_batches:
            break
        batch = move_batch_to_device(batch, device)
        part_proj, patch_tokens = project_parts(model, batch["part_text_feat"], batch["patch_tokens"])
        for b in range(part_proj.shape[0]):
            out = visible_sample_tensors(
                part_proj[b], patch_tokens[b], batch["part_valid_mask"][b],
                batch["part_gt_mask_patch"][b], batch["obj_mask_patch"][b], batch["part_category_id"][b]
            )
            if out is None:
                continue
            z_vis, p_obj, gt_vis_obj, part_ids, _ = out
            obj_id = int(batch["category_id"][b].detach().cpu().item())
            W_b = get_W(obj_id).to(device)
            z_w = safe_normalize(z_vis @ W_b, dim=-1)
            sim_before = z_vis @ p_obj.T
            sim_after = z_w @ p_obj.T
            anchor_before, _ = select_anchors(sim_before, anchor_mode=anchor_mode)
            anchor_after, _ = select_anchors(sim_after, anchor_mode=anchor_mode)

            before_hit_vec = gt_vis_obj[torch.arange(gt_vis_obj.shape[0], device=device), anchor_before]
            after_hit_vec = gt_vis_obj[torch.arange(gt_vis_obj.shape[0], device=device), anchor_after]
            before_cost = 1.0 - sim_before[torch.arange(sim_before.shape[0], device=device), anchor_before]
            after_cost = 1.0 - sim_after[torch.arange(sim_after.shape[0], device=device), anchor_after]

            gt_proto = []
            for j in range(gt_vis_obj.shape[0]):
                m = gt_vis_obj[j]
                if m.any():
                    gt_proto.append(safe_normalize(p_obj[m].mean(dim=0), dim=-1))
                else:
                    gt_proto.append(torch.zeros_like(z_vis[j]))
            gt_proto = torch.stack(gt_proto, dim=0)
            before_cos_gt = (z_vis * gt_proto).sum(dim=-1)
            after_cos_gt = (z_w * gt_proto).sum(dim=-1)

            for j in range(z_vis.shape[0]):
                pid = int(part_ids[j].detach().cpu().item())
                vals = (
                    bool(before_hit_vec[j].detach().cpu().item()),
                    bool(after_hit_vec[j].detach().cpu().item()),
                    float(before_cost[j].detach().cpu().item()),
                    float(after_cost[j].detach().cpu().item()),
                    float(before_cos_gt[j].detach().cpu().item()),
                    float(after_cos_gt[j].detach().cpu().item()),
                )
                overall.update(*vals)
                per_object[obj_id].update(*vals)
                per_part[pid].update(*vals)
        if overall.total > 0:
            d = overall.to_dict()
            pbar.set_postfix(before=f"{d['before_anchor_hit_rate']:.4f}", after=f"{d['after_anchor_hit_rate']:.4f}", parts=overall.total)
    return overall.to_dict(), {k: v.to_dict() for k, v in per_object.items()}, {k: v.to_dict() for k, v in per_part.items()}


def save_csv(path: str, key_name: str, stats: Dict[int, Dict]) -> None:
    rows = []
    for k, d in sorted(stats.items(), key=lambda x: x[0]):
        row = {key_name: k}
        row.update(d)
        rows.append(row)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_common_args(parser):
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--obj_feature_name", type=str, default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", type=str, default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", type=str, default="ann_feats")
    parser.add_argument("--part_text_name", type=str, default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_train_batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser
