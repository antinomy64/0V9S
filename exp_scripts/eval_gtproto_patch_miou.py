#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate GT-prototype upper bounds on VOC116 object-crop patch tokens.

Two modes:
  1) per_image_gtproto:
     For each validation object crop, compute each visible part prototype from that same
     crop's GT part mask, then classify object-mask patches by nearest prototype.
     This is a test-time oracle upper bound for DINO patch-token separability.

  2) train_global_gtproto:
     Build one global prototype per part_category_id from the training split GT masks,
     then classify validation object-mask patches by nearest corresponding visible-part
     prototypes. This is still oracle-visible at evaluation time, but prototypes are
     not computed from the validation GT masks.

Important:
  This script reports PATCH-LEVEL object-crop mIoU over patch tokens. It is a diagnostic
  upper bound and is not identical to the official full-resolution evaluator in
  src/open_vocabulary_segmentation/main.py.

It reuses the repository's DinoClipJointDataset and joint_collate_fn, and does not use
text features or the Stage1 projector.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def build_loaders(args):
    config = load_config(args.model_config)
    dataset_cfg = config.get("dataset", {})
    train_min_area = float(dataset_cfg.get("min_obj_area_ratio", 0.0))

    train_set = None
    train_loader = None
    if args.train_dataset:
        train_set = build_dataset(args.train_dataset, args, class_part_bank=None, min_obj_area_ratio=train_min_area)
        class_part_bank = train_set.class_part_bank
    else:
        class_part_bank = None

    val_set = build_dataset(args.val_dataset, args, class_part_bank=class_part_bank, min_obj_area_ratio=0.0)

    if train_set is not None:
        if train_set.part_taxonomy != val_set.part_taxonomy:
            raise ValueError(f"taxonomy mismatch: train={train_set.part_taxonomy}, val={val_set.part_taxonomy}")
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
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
    return train_loader, val_loader, train_set, val_set


def move_batch(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def visible_sample_tensors(batch: Dict, b: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return p_obj [M,D], gt_vis_obj [K,M], part_ids [K], obj_idx [M]."""
    patch_z = normalize(batch["patch_tokens"][b].float(), dim=-1)
    obj_mask = batch["obj_mask_patch"][b].bool().reshape(-1)
    obj_idx = torch.nonzero(obj_mask, as_tuple=False).flatten()
    if obj_idx.numel() == 0:
        return None

    gt = batch["part_gt_mask_patch"][b].bool()
    if gt.ndim != 2:
        gt = gt.reshape(gt.shape[0], -1)
    valid = batch["part_valid_mask"][b].bool().reshape(-1)
    visible = ((gt & obj_mask[None, :]).sum(dim=1) > 0) & valid
    vis_idx = torch.nonzero(visible, as_tuple=False).flatten()
    if vis_idx.numel() == 0:
        return None

    p_obj = patch_z[obj_idx]
    gt_vis_obj = gt[vis_idx][:, obj_idx].bool()
    part_ids = batch["part_category_id"][b][vis_idx].long().reshape(-1)
    return p_obj, gt_vis_obj, part_ids, obj_idx


@torch.no_grad()
def build_train_global_prototypes(train_loader, device: torch.device, args):
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = defaultdict(int)
    total_samples = 0
    total_parts = 0
    dim = None

    pbar = tqdm(train_loader, desc="build train global GT prototypes")
    for step, batch in enumerate(pbar):
        if args.max_train_batches > 0 and step >= args.max_train_batches:
            break
        batch = move_batch(batch, device)
        for b in range(batch["patch_tokens"].shape[0]):
            sample = visible_sample_tensors(batch, b)
            if sample is None:
                continue
            p_obj, gt_vis_obj, part_ids, _ = sample
            dim = int(p_obj.shape[-1])
            total_samples += 1
            for k in range(gt_vis_obj.shape[0]):
                mask = gt_vis_obj[k]
                if not bool(mask.any().item()):
                    continue
                pid = int(part_ids[k].detach().cpu().item())
                # Aggregate normalized patch tokens directly; normalize only after all accumulation.
                vec_sum = p_obj[mask].sum(dim=0)
                n = int(mask.float().sum().detach().cpu().item())
                if pid not in sums:
                    sums[pid] = torch.zeros((dim,), device=device, dtype=torch.float32)
                sums[pid] += vec_sum.float()
                counts[pid] += n
                total_parts += 1
        pbar.set_postfix(samples=total_samples, proto=len(sums))

    protos = {}
    for pid, vec in sums.items():
        if counts[pid] > 0:
            protos[pid] = normalize(vec / float(counts[pid]), dim=-1)
    meta = {
        "num_part_prototypes": len(protos),
        "total_train_samples_used": total_samples,
        "total_train_visible_part_instances": total_parts,
        "prototype_pixel_counts": {str(k): int(v) for k, v in counts.items()},
    }
    return protos, meta


class IoUAccumulator:
    def __init__(self):
        self.inter = defaultdict(int)
        self.union = defaultdict(int)
        self.gt_count = defaultdict(int)
        self.pred_count = defaultdict(int)
        self.total_valid = 0
        self.total_correct = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred_cpu = pred.detach().cpu().long().reshape(-1)
        target_cpu = target.detach().cpu().long().reshape(-1)
        valid = target_cpu >= 0
        pred_cpu = pred_cpu[valid]
        target_cpu = target_cpu[valid]
        self.total_valid += int(target_cpu.numel())
        self.total_correct += int((pred_cpu == target_cpu).sum().item())
        labels = torch.unique(torch.cat([pred_cpu, target_cpu], dim=0)).tolist() if target_cpu.numel() else []
        for c in labels:
            c = int(c)
            p = pred_cpu == c
            t = target_cpu == c
            self.inter[c] += int((p & t).sum().item())
            self.union[c] += int((p | t).sum().item())
            self.gt_count[c] += int(t.sum().item())
            self.pred_count[c] += int(p.sum().item())

    def summary(self):
        class_rows = []
        ious = []
        accs = []
        labels = sorted(set(self.union.keys()) | set(self.gt_count.keys()))
        for c in labels:
            union = self.union.get(c, 0)
            gt = self.gt_count.get(c, 0)
            inter = self.inter.get(c, 0)
            iou = float(inter) / float(union) if union > 0 else float("nan")
            acc = float(inter) / float(gt) if gt > 0 else float("nan")
            if union > 0:
                ious.append(iou)
            if gt > 0:
                accs.append(acc)
            class_rows.append({
                "part_category_id": c,
                "iou": iou,
                "acc": acc,
                "inter": inter,
                "union": union,
                "gt_count": gt,
                "pred_count": self.pred_count.get(c, 0),
            })
        return {
            "num_classes_with_union": len(ious),
            "num_classes_with_gt": len(accs),
            "patch_aAcc": float(self.total_correct) / float(max(self.total_valid, 1)),
            "patch_mIoU": float(np.mean(ious)) if ious else float("nan"),
            "patch_mAcc": float(np.mean(accs)) if accs else float("nan"),
            "total_valid_patch_positions": self.total_valid,
            "class_results": class_rows,
        }


def make_target_from_gt(gt_vis_obj: torch.Tensor, part_ids: torch.Tensor) -> torch.Tensor:
    """Create one target part id per object patch; ignore patch if no visible GT part covers it."""
    K, M = gt_vis_obj.shape
    target = torch.full((M,), -1, dtype=torch.long, device=gt_vis_obj.device)
    # If masks overlap after patchization, use the first visible part in dataset order.
    # This keeps the diagnostic deterministic.
    for k in range(K):
        mask = gt_vis_obj[k] & (target < 0)
        target[mask] = part_ids[k]
    return target


@torch.no_grad()
def eval_per_image_gtproto(val_loader, device: torch.device, args):
    acc = IoUAccumulator()
    used_samples = 0
    skipped_samples = 0
    pbar = tqdm(val_loader, desc="eval per-image GT prototypes")
    for step, batch in enumerate(pbar):
        if args.max_eval_batches > 0 and step >= args.max_eval_batches:
            break
        batch = move_batch(batch, device)
        for b in range(batch["patch_tokens"].shape[0]):
            sample = visible_sample_tensors(batch, b)
            if sample is None:
                skipped_samples += 1
                continue
            p_obj, gt_vis_obj, part_ids, _ = sample
            protos = []
            keep_ids = []
            for k in range(gt_vis_obj.shape[0]):
                mask = gt_vis_obj[k]
                if bool(mask.any().item()):
                    protos.append(normalize(p_obj[mask].mean(dim=0), dim=-1))
                    keep_ids.append(part_ids[k])
            if not protos:
                skipped_samples += 1
                continue
            proto = torch.stack(protos, dim=0)
            keep_ids_t = torch.stack(keep_ids, dim=0).long()
            sim = proto @ p_obj.T
            pred = keep_ids_t[sim.argmax(dim=0)]
            target = make_target_from_gt(gt_vis_obj, part_ids)
            acc.update(pred, target)
            used_samples += 1
        s = acc.summary()
        pbar.set_postfix(samples=used_samples, miou=f"{s['patch_mIoU']:.4f}", aacc=f"{s['patch_aAcc']:.4f}")
    out = acc.summary()
    out.update({"used_samples": used_samples, "skipped_samples": skipped_samples})
    return out


@torch.no_grad()
def eval_train_global_gtproto(val_loader, protos: Dict[int, torch.Tensor], device: torch.device, args):
    acc = IoUAccumulator()
    used_samples = 0
    skipped_samples = 0
    missing_part_instances = 0
    pbar = tqdm(val_loader, desc="eval train-global GT prototypes")
    for step, batch in enumerate(pbar):
        if args.max_eval_batches > 0 and step >= args.max_eval_batches:
            break
        batch = move_batch(batch, device)
        for b in range(batch["patch_tokens"].shape[0]):
            sample = visible_sample_tensors(batch, b)
            if sample is None:
                skipped_samples += 1
                continue
            p_obj, gt_vis_obj, part_ids, _ = sample
            proto_list = []
            keep_ids = []
            for k in range(part_ids.numel()):
                pid = int(part_ids[k].detach().cpu().item())
                if pid in protos:
                    proto_list.append(protos[pid].to(device))
                    keep_ids.append(part_ids[k])
                else:
                    missing_part_instances += 1
            if not proto_list:
                skipped_samples += 1
                continue
            proto = torch.stack(proto_list, dim=0)
            keep_ids_t = torch.stack(keep_ids, dim=0).long()
            sim = proto @ p_obj.T
            pred = keep_ids_t[sim.argmax(dim=0)]
            target = make_target_from_gt(gt_vis_obj, part_ids)
            acc.update(pred, target)
            used_samples += 1
        s = acc.summary()
        pbar.set_postfix(samples=used_samples, miou=f"{s['patch_mIoU']:.4f}", aacc=f"{s['patch_aAcc']:.4f}")
    out = acc.summary()
    out.update({
        "used_samples": used_samples,
        "skipped_samples": skipped_samples,
        "missing_part_instances_without_train_proto": missing_part_instances,
    })
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--val_dataset", required=True)
    parser.add_argument("--train_dataset", default=None, help="required for --mode train_global_gtproto")
    parser.add_argument("--mode", choices=["per_image_gtproto", "train_global_gtproto", "both"], default="both")
    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--path_prefix", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode in ("train_global_gtproto", "both") and not args.train_dataset:
        raise ValueError("--train_dataset is required for train_global_gtproto/both")
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")
    print("[setup] This reports object-crop PATCH-LEVEL mIoU, not official full-resolution mIoU.")
    train_loader, val_loader, train_set, val_set = build_loaders(args)
    print(f"[setup] val samples: {len(val_set)}")
    if train_set is not None:
        print(f"[setup] train samples: {len(train_set)}")

    result = {"args": vars(args), "notes": {
        "metric": "object-crop patch-level part mIoU",
        "candidate_parts": "oracle-visible parts from GT masks",
        "per_image_gtproto": "validation GT masks are used to compute prototypes for the same sample; test-time oracle upper bound",
        "train_global_gtproto": "train GT masks are used to build global part prototypes; val GT masks are used only for oracle-visible candidate filtering and evaluation",
    }}

    if args.mode in ("per_image_gtproto", "both"):
        per_img = eval_per_image_gtproto(val_loader, device, args)
        result["per_image_gtproto"] = per_img
        print(json.dumps({"per_image_gtproto": {k: v for k, v in per_img.items() if k != "class_results"}}, indent=2, ensure_ascii=False))

    if args.mode in ("train_global_gtproto", "both"):
        protos, proto_meta = build_train_global_prototypes(train_loader, device, args)
        glob = eval_train_global_gtproto(val_loader, protos, device, args)
        result["train_global_prototype_meta"] = proto_meta
        result["train_global_gtproto"] = glob
        print(json.dumps({"train_global_gtproto": {k: v for k, v in glob.items() if k != "class_results"}}, indent=2, ensure_ascii=False))

    out_path = os.path.join(args.out_dir, "gtproto_patch_miou_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
