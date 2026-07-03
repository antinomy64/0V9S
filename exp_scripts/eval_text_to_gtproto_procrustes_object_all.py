#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real text feature -> train-global GT visual prototype Procrustes upper bound
with object-known / part-visible-unknown candidates.

Purpose
-------
This diagnostic answers:
  If we give the real Stage1 projected text part features the best global orthogonal
  map W to the train-split GT visual part prototype bank, how well can the mapped
  text prototypes classify validation object-crop patches?

Protocol
--------
1) Build train-global visual GT prototypes V_k:
     For each train object crop and each visible GT part, average normalized DINO
     patch tokens inside the GT part mask. Aggregate by global part_category_id.

2) Build real text prototypes T_k:
     For the same visible train part instances, apply the frozen Stage1 text projector
     to part_text_feat and aggregate by global part_category_id.

3) Fit orthogonal Procrustes W:
     min_W || T W - V ||_F, subject to W^T W = I.
     Row-vector convention: z_after = normalize(z_text @ W).

4) Evaluate on validation object crops with object-known candidates:
     For each val crop, candidates are ALL valid parts of the known object category,
     not the GT-visible parts in that crop. Classify object-mask patches by nearest
     prototype among those object-level candidate parts. Report object-crop PATCH-LEVEL
     mIoU. This is not the official full-resolution segmentation evaluator.

Important supervision note
--------------------------
This is an upper-bound diagnostic. Train GT masks are used to build the visual
prototype target bank. Validation GT masks are used only to compute evaluation
targets/metrics, NOT to filter candidate parts. The candidate set is all parts of
the known object category, as encoded by part_valid_mask in DinoClipJointDataset.
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


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_stage1_model(config_path: str, weights_path: str, device: torch.device):
    config = load_config(config_path)
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


@torch.no_grad()
def project_text_batch(model, batch: Dict) -> torch.Tensor:
    z = model.project_clip_txt(batch["part_text_feat"].float())
    return normalize(z.float(), dim=-1)


def visible_indices_and_tensors(batch: Dict, b: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return p_obj [M,D], gt_vis_obj [K,M], part_ids [K], vis_idx [K], obj_idx [M]."""
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
    return p_obj, gt_vis_obj, part_ids, vis_idx, obj_idx


def object_all_candidate_part_ids(batch: Dict, b: int) -> torch.Tensor:
    """Return all valid part_category_id values for the known object category.

    This is the object-known / part-visible-unknown candidate set. It does NOT
    use GT visibility. In DinoClipJointDataset, part_valid_mask marks the parts
    belonging to the current object's class_part_bank.
    """
    valid = batch["part_valid_mask"][b].bool().reshape(-1)
    idx = torch.nonzero(valid, as_tuple=False).flatten()
    if idx.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=valid.device)
    part_ids = batch["part_category_id"][b][idx].long().reshape(-1)

    # Deduplicate defensively while preserving first occurrence order.
    seen = set()
    keep = []
    for x in part_ids.detach().cpu().tolist():
        x = int(x)
        if x not in seen:
            seen.add(x)
            keep.append(x)
    return torch.tensor(keep, dtype=torch.long, device=part_ids.device)


@torch.no_grad()
def build_train_visual_and_text_prototypes(model, train_loader, device: torch.device, args):
    visual_sums: Dict[int, torch.Tensor] = {}
    visual_patch_counts: Dict[int, int] = defaultdict(int)
    text_sums: Dict[int, torch.Tensor] = {}
    text_inst_counts: Dict[int, int] = defaultdict(int)

    total_samples = 0
    total_visible_instances = 0
    dim = None

    pbar = tqdm(train_loader, desc="build train GT visual + text prototypes")
    for step, batch in enumerate(pbar):
        if args.max_train_batches > 0 and step >= args.max_train_batches:
            break
        batch = move_batch(batch, device)
        part_z0 = project_text_batch(model, batch)
        for b in range(batch["patch_tokens"].shape[0]):
            sample = visible_indices_and_tensors(batch, b)
            if sample is None:
                continue
            p_obj, gt_vis_obj, part_ids, vis_idx, _ = sample
            z_vis = part_z0[b][vis_idx]
            dim = int(p_obj.shape[-1])
            total_samples += 1

            for k in range(gt_vis_obj.shape[0]):
                mask = gt_vis_obj[k]
                if not bool(mask.any().item()):
                    continue
                pid = int(part_ids[k].detach().cpu().item())
                if pid not in visual_sums:
                    visual_sums[pid] = torch.zeros((dim,), device=device, dtype=torch.float32)
                    text_sums[pid] = torch.zeros((dim,), device=device, dtype=torch.float32)

                # Visual prototype aggregation is patch-count weighted.
                n_patch = int(mask.float().sum().detach().cpu().item())
                visual_sums[pid] += p_obj[mask].sum(dim=0).float()
                visual_patch_counts[pid] += n_patch

                # Text prototype aggregation is instance-count weighted.
                text_sums[pid] += z_vis[k].float()
                text_inst_counts[pid] += 1
                total_visible_instances += 1

        pbar.set_postfix(samples=total_samples, parts=total_visible_instances, proto=len(visual_sums))

    visual_protos: Dict[int, torch.Tensor] = {}
    text_protos: Dict[int, torch.Tensor] = {}
    for pid in sorted(set(visual_sums.keys()) & set(text_sums.keys())):
        if visual_patch_counts[pid] > 0 and text_inst_counts[pid] > 0:
            visual_protos[pid] = normalize(visual_sums[pid] / float(visual_patch_counts[pid]), dim=-1)
            text_protos[pid] = normalize(text_sums[pid] / float(text_inst_counts[pid]), dim=-1)

    meta = {
        "num_common_part_prototypes": len(set(visual_protos.keys()) & set(text_protos.keys())),
        "num_visual_part_prototypes": len(visual_protos),
        "num_text_part_prototypes": len(text_protos),
        "total_train_samples_used": total_samples,
        "total_train_visible_part_instances": total_visible_instances,
        "visual_patch_counts": {str(k): int(v) for k, v in visual_patch_counts.items()},
        "text_instance_counts": {str(k): int(v) for k, v in text_inst_counts.items()},
    }
    return visual_protos, text_protos, meta


def stack_common(text_protos: Dict[int, torch.Tensor], visual_protos: Dict[int, torch.Tensor], args, device: torch.device):
    ids = sorted(set(text_protos.keys()) & set(visual_protos.keys()))
    if not ids:
        raise ValueError("No common part ids between text prototypes and visual GT prototypes.")
    T = torch.stack([text_protos[i].to(device).float() for i in ids], dim=0)
    V = torch.stack([visual_protos[i].to(device).float() for i in ids], dim=0)
    T = normalize(T, dim=-1)
    V = normalize(V, dim=-1)
    return ids, T, V


@torch.no_grad()
def fit_orthogonal_procrustes(T: torch.Tensor, V: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fit row-vector W such that T @ W approximates V."""
    if weights is not None:
        w = weights.reshape(-1, 1).to(T.device, dtype=T.dtype).clamp_min(0.0).sqrt()
        T_fit = T * w
        V_fit = V * w
    else:
        T_fit = T
        V_fit = V
    M = T_fit.T @ V_fit
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vh
    return W


@torch.no_grad()
def retrieval_metrics(T: torch.Tensor, V: torch.Tensor) -> Dict[str, float]:
    Tn = normalize(T, dim=-1)
    Vn = normalize(V, dim=-1)
    sim = Tn @ Vn.T
    diag = sim.diag()
    if sim.shape[0] > 1:
        other = sim.masked_fill(torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool), -1e9).max(dim=1).values
        margin = diag - other
    else:
        margin = torch.zeros_like(diag)
    top1 = sim.argmax(dim=1) == torch.arange(sim.shape[0], device=sim.device)
    return {
        "num_parts": int(sim.shape[0]),
        "mean_diag_cos": float(diag.mean().detach().cpu().item()),
        "median_diag_cos": float(diag.median().detach().cpu().item()),
        "top1_retrieval_acc": float(top1.float().mean().detach().cpu().item()),
        "mean_diag_minus_best_other_margin": float(margin.mean().detach().cpu().item()),
        "median_diag_minus_best_other_margin": float(margin.median().detach().cpu().item()),
    }


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
        if target_cpu.numel() == 0:
            return
        labels = torch.unique(torch.cat([pred_cpu, target_cpu], dim=0)).tolist()
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
    target = torch.full((gt_vis_obj.shape[1],), -1, dtype=torch.long, device=gt_vis_obj.device)
    for k in range(gt_vis_obj.shape[0]):
        mask = gt_vis_obj[k] & (target < 0)
        target[mask] = part_ids[k]
    return target


@torch.no_grad()
def eval_prototype_bank(val_loader, proto_bank: Dict[int, torch.Tensor], device: torch.device, args, desc: str):
    """Evaluate with object-known but part-visible-unknown candidates.

    Candidates for each crop are all valid parts of that object's class_part_bank
    according to part_valid_mask, not the GT-visible parts. GT masks are used only
    to build target labels over object-mask patches and compute IoU.
    """
    acc = IoUAccumulator()
    used_samples = 0
    skipped_samples = 0
    missing_candidate_parts = 0
    samples_with_no_candidate_proto = 0
    pbar = tqdm(val_loader, desc=desc)
    for step, batch in enumerate(pbar):
        if args.max_eval_batches > 0 and step >= args.max_eval_batches:
            break
        batch = move_batch(batch, device)
        for b in range(batch["patch_tokens"].shape[0]):
            sample = visible_indices_and_tensors(batch, b)
            if sample is None:
                skipped_samples += 1
                continue
            p_obj, gt_vis_obj, visible_part_ids, _, _ = sample

            cand_part_ids = object_all_candidate_part_ids(batch, b)
            proto_list: List[torch.Tensor] = []
            keep_ids: List[torch.Tensor] = []
            for j in range(cand_part_ids.numel()):
                pid = int(cand_part_ids[j].detach().cpu().item())
                if pid in proto_bank:
                    proto_list.append(proto_bank[pid].to(device).float())
                    keep_ids.append(cand_part_ids[j])
                else:
                    missing_candidate_parts += 1

            if not proto_list:
                skipped_samples += 1
                samples_with_no_candidate_proto += 1
                continue

            proto = normalize(torch.stack(proto_list, dim=0), dim=-1)
            keep_ids_t = torch.stack(keep_ids, dim=0).long()
            sim = proto @ p_obj.T
            pred = keep_ids_t[sim.argmax(dim=0)]
            target = make_target_from_gt(gt_vis_obj, visible_part_ids)
            acc.update(pred, target)
            used_samples += 1
        s = acc.summary()
        pbar.set_postfix(samples=used_samples, miou=f"{s['patch_mIoU']:.4f}", aacc=f"{s['patch_aAcc']:.4f}")
    out = acc.summary()
    out.update({
        "used_samples": used_samples,
        "skipped_samples": skipped_samples,
        "missing_candidate_parts_without_proto": missing_candidate_parts,
        "samples_with_no_candidate_proto": samples_with_no_candidate_proto,
        "candidate_protocol": "object_known_all_valid_parts_part_visible_unknown",
    })
    return out


def make_mapped_text_bank(text_protos: Dict[int, torch.Tensor], W: Optional[torch.Tensor], device: torch.device) -> Dict[int, torch.Tensor]:
    bank = {}
    for pid, t in text_protos.items():
        z = t.to(device).float()
        if W is not None:
            z = z @ W.to(device).float()
        bank[pid] = normalize(z, dim=-1).detach().clone()
    return bank


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--weights", required=True, help="Stage1 object projector weights; text projector is frozen and used to project part_text_feat")
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--fit_weighting", choices=["uniform", "text_instance", "visual_patch"], default="uniform")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")
    print("[setup] This reports object-crop PATCH-LEVEL mIoU, not official full-resolution mIoU.")
    print("[setup] Candidate protocol: object-known all valid parts; no GT-visible filtering.")
    print("[setup] Row-vector convention: z_after = normalize(z_text @ W)")

    model, config = load_stage1_model(args.model_config, args.weights, device)
    train_loader, val_loader, train_set, val_set = build_loaders(args, config)
    print(f"[setup] train samples: {len(train_set)}")
    print(f"[setup] val samples: {len(val_set)}")

    visual_protos, text_protos, proto_meta = build_train_visual_and_text_prototypes(model, train_loader, device, args)
    common_ids, T, V = stack_common(text_protos, visual_protos, args, device)

    weights = None
    if args.fit_weighting != "uniform":
        if args.fit_weighting == "text_instance":
            raw = [float(proto_meta["text_instance_counts"][str(pid)]) for pid in common_ids]
        elif args.fit_weighting == "visual_patch":
            raw = [float(proto_meta["visual_patch_counts"][str(pid)]) for pid in common_ids]
        else:
            raise ValueError(args.fit_weighting)
        weights = torch.tensor(raw, device=device, dtype=torch.float32)
        weights = weights / weights.mean().clamp_min(1e-6)

    W = fit_orthogonal_procrustes(T, V, weights=weights)
    text_before = retrieval_metrics(T, V)
    text_after = retrieval_metrics(normalize(T @ W, dim=-1), V)
    orth_err = torch.linalg.norm(W.T @ W - torch.eye(W.shape[0], device=device)) / np.sqrt(W.shape[0])
    fit_metrics = {
        "fit_weighting": args.fit_weighting,
        "num_common_parts": len(common_ids),
        "before_text_to_visual_retrieval": text_before,
        "after_textW_to_visual_retrieval": text_after,
        "orthogonality_error_WtW_minus_I_over_sqrtD": float(orth_err.detach().cpu().item()),
        "mean_cos_TW_V": float(((normalize(T @ W, dim=-1) * V).sum(dim=-1)).mean().detach().cpu().item()),
        "mse_TW_minus_V": float(((normalize(T @ W, dim=-1) - V) ** 2).mean().detach().cpu().item()),
    }
    print(json.dumps({"procrustes_fit_metrics": fit_metrics}, indent=2, ensure_ascii=False))

    visual_bank = {pid: visual_protos[pid].detach().clone() for pid in visual_protos.keys()}
    text_identity_bank = make_mapped_text_bank(text_protos, None, device)
    text_procrustes_bank = make_mapped_text_bank(text_protos, W, device)

    result = {
        "args": vars(args),
        "notes": {
            "metric": "object-crop patch-level part mIoU",
            "candidate_parts": "object-known all valid parts from part_valid_mask; no GT-visible filtering",
            "train_visual_gtproto": "train GT masks build global visual prototype bank; val GT masks used only for target/eval, not candidate filtering",
            "text_identity": "frozen Stage1 projected text prototypes, no Procrustes W",
            "text_procrustes": "frozen Stage1 projected text prototypes mapped by best orthogonal W fitted to train-global GT visual prototypes",
            "W_definition": "z_after = normalize(z_text @ W)",
        },
        "prototype_meta": proto_meta,
        "common_part_ids": common_ids,
        "procrustes_fit_metrics": fit_metrics,
    }

    print("[eval] train-global visual GT prototypes")
    visual_eval = eval_prototype_bank(val_loader, visual_bank, device, args, desc="eval train visual GT prototypes")
    result["train_global_visual_gtproto"] = visual_eval
    print(json.dumps({"train_global_visual_gtproto": {k: v for k, v in visual_eval.items() if k != "class_results"}}, indent=2, ensure_ascii=False))

    print("[eval] Stage1 text prototypes without Procrustes")
    text_id_eval = eval_prototype_bank(val_loader, text_identity_bank, device, args, desc="eval text identity prototypes")
    result["text_identity"] = text_id_eval
    print(json.dumps({"text_identity": {k: v for k, v in text_id_eval.items() if k != "class_results"}}, indent=2, ensure_ascii=False))

    print("[eval] Stage1 text prototypes after Procrustes W")
    text_w_eval = eval_prototype_bank(val_loader, text_procrustes_bank, device, args, desc="eval text Procrustes prototypes")
    result["text_procrustes_to_train_global_gtproto"] = text_w_eval
    print(json.dumps({"text_procrustes_to_train_global_gtproto": {k: v for k, v in text_w_eval.items() if k != "class_results"}}, indent=2, ensure_ascii=False))

    w_path = os.path.join(args.out_dir, "text_to_train_global_gtproto_procrustes_W.pt")
    out_path = os.path.join(args.out_dir, "text_to_gtproto_procrustes_summary.json")
    torch.save({
        "W": W.detach().cpu(),
        "common_part_ids": common_ids,
        "args": vars(args),
        "procrustes_fit_metrics": fit_metrics,
    }, w_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[save] W: {w_path}")
    print(f"[save] summary: {out_path}")
    print("Key numbers [patch-level object-known / part-visible-unknown]:")
    print(f"  train_global_visual_gtproto_mIoU = {visual_eval['patch_mIoU']:.6f}")
    print(f"  text_identity_mIoU              = {text_id_eval['patch_mIoU']:.6f}")
    print(f"  text_procrustes_mIoU            = {text_w_eval['patch_mIoU']:.6f}")
    print(f"  textW mean cos to visual proto  = {fit_metrics['mean_cos_TW_V']:.6f}")


if __name__ == "__main__":
    main()
