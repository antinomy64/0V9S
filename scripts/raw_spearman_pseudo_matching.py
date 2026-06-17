#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw-text Spearman structure matching audit.

Question:
  raw text is 512D, visual pseudo/GT is 768D. Can we use raw text?
Answer:
  Yes. Use dimension-free object-internal pairwise structure:
      S_raw[i,j]    = cos(raw_text_i, raw_text_j)
      S_pseudo[i,j] = cos(pseudo_i, pseudo_j)
      S_gt[i,j]     = cos(gt_i, gt_j)

This script tries to recover a permutation between raw part names and pseudo
visual clusters by maximizing:
      Spearman( upper(S_raw), upper(P S_pseudo P^T) )

Then it evaluates whether the recovered pseudo cluster for each raw part name
is visually close to the corresponding GT prototype.

This is NOT a linear Q. It is a structure-only naming/permutation diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn
from src.loss_joint import JointObjPartLoss


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_projector(config: Dict, checkpoint: str, device: torch.device) -> torch.nn.Module:
    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"])

    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    ret = model.load_state_dict(ckpt, strict=False)
    print(f"[model] checkpoint={checkpoint}")
    print(f"[model] missing_keys={getattr(ret, 'missing_keys', [])}")
    print(f"[model] unexpected_keys={getattr(ret, 'unexpected_keys', [])}")

    model.to(device)
    model.eval()
    return model


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_joint_dataset(args, config: Dict) -> DinoClipJointDataset:
    dataset_cfg = config.get("dataset", {})
    min_obj_area_ratio = float(dataset_cfg.get("min_obj_area_ratio", args.min_obj_area_ratio))
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


def build_name_maps(dataset: DinoClipJointDataset):
    cat_id_to_name = {}
    part_id_to_name = {}

    samples = dataset.data.values() if isinstance(dataset.data, dict) else dataset.data
    for sample in samples:
        if "category_id" in sample:
            cat_id_to_name[int(sample.get("category_id", -1))] = str(sample.get("class_name", ""))
        pids = sample.get("part_category_id", [])
        pnames = sample.get("part_class_name", [])
        if torch.is_tensor(pids):
            pids = pids.detach().cpu().tolist()
        for pid, pname in zip(pids, pnames):
            part_id_to_name[int(pid)] = str(pname)

    for _, bank in getattr(dataset, "class_part_bank", {}).items():
        pids = bank.get("part_ids", [])
        pnames = bank.get("part_names", [])
        if torch.is_tensor(pids):
            pids = pids.detach().cpu().tolist()
        for pid, pname in zip(pids, pnames):
            part_id_to_name[int(pid)] = str(pname)

    return cat_id_to_name, part_id_to_name


def part_name(pid: int, part_id_to_name: Dict[int, str]) -> str:
    return part_id_to_name.get(int(pid), f"part_{pid}")


def cat_name(cid: int, cat_id_to_name: Dict[int, str]) -> str:
    return cat_id_to_name.get(int(cid), f"cat_{cid}")


def make_joint_criterion(config: Dict, model: torch.nn.Module) -> JointObjPartLoss:
    train_cfg = config.get("train", {})
    criterion = JointObjPartLoss(
        model,
        obj_ltype=train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce")),
        obj_margin=float(train_cfg.get("margin", 0.2)),
        obj_max_violation=bool(train_cfg.get("max_violation", True)),
        lambda_obj=float(train_cfg.get("lambda_obj", 1.0)),
        lambda_inst=float(train_cfg.get("lambda_inst", 0.2)),
        lambda_overlap=float(train_cfg.get("lambda_overlap", 0.05)),
        lambda_spear=float(train_cfg.get("lambda_spear", 0.0)),
        patch_temperature=float(train_cfg.get("patch_temperature", 0.07)),
        em_iters=int(train_cfg.get("em_iters", 3)),
        present_only_anchor=bool(train_cfg.get("present_only_anchor", False)),
    )
    criterion.present_only_anchor = bool(train_cfg.get("present_only_anchor", False))
    criterion.eval()
    return criterion


def make_global_criterion(config: Dict, model: torch.nn.Module):
    from src.loss_global import PartLoss

    train_cfg = config.get("train", {})
    criterion = PartLoss(
        model,
        lambda_inst=float(train_cfg.get("lambda_inst", 0.2)),
        lambda_overlap=float(train_cfg.get("lambda_overlap", 0.05)),
        lambda_spear=float(train_cfg.get("lambda_spear", 0.0)),
        topk_ratio=float(train_cfg.get("topk_ratio", 0.1)),
        patch_temperature=float(train_cfg.get("patch_temperature", 0.07)),
        em_iters=int(train_cfg.get("em_iters", 3)),
        present_only_anchor=bool(train_cfg.get("present_only_anchor", False)),
        anchor_matcher=str(train_cfg.get("anchor_matcher", "greedy")),
        anchor_score_type=str(train_cfg.get("anchor_score_type", "relative")),
    )
    criterion.present_only_anchor = bool(train_cfg.get("present_only_anchor", False))
    criterion.eval()
    return criterion


class Store:
    def __init__(self):
        # (cat_id, part_id) -> source -> [sum, count]
        self.data = defaultdict(dict)

    def add(self, cat_id: int, part_id: int, source: str, vec: torch.Tensor):
        key = (int(cat_id), int(part_id))
        vec = safe_normalize(vec.detach().float().cpu(), dim=-1)
        if source not in self.data[key]:
            self.data[key][source] = [torch.zeros_like(vec), 0]
        self.data[key][source][0] += vec
        self.data[key][source][1] += 1

    def by_object(self):
        out = defaultdict(dict)
        for (cat_id, pid), srcs in self.data.items():
            item = {}
            for src, (s, c) in srcs.items():
                if c > 0:
                    item[src] = safe_normalize((s / c).float(), dim=-1)
            out[cat_id][pid] = item
        return out


@torch.no_grad()
def collect_common(model, loader, store: Store, max_batches: int):
    device = next(model.parameters()).device

    for bi, batch in enumerate(tqdm(loader, desc="collect raw/proj/GT")):
        if max_batches >= 0 and bi >= max_batches:
            break
        batch = move_batch_to_device(batch, device)

        patch = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()
        part_gt = batch["part_gt_mask_patch"].bool()
        part_valid = batch["part_valid_mask"].bool()
        part_ids = batch["part_category_id"].long()
        raw = safe_normalize(batch["part_text_feat"].float(), dim=-1)
        proj = safe_normalize(model.project_clip_txt(raw), dim=-1)
        cat_ids = batch["category_id"].long()

        B, K, _ = raw.shape
        for b in range(B):
            cat = int(cat_ids[b].item())
            for k in range(K):
                if not bool(part_valid[b, k].item()):
                    continue
                pid = int(part_ids[b, k].item())
                if pid < 0:
                    continue
                mask = part_gt[b, k] & obj_mask[b]
                if not bool(mask.any().item()):
                    continue
                gt = safe_normalize(patch[b, mask].mean(dim=0), dim=-1)
                store.add(cat, pid, "raw", raw[b, k])
                store.add(cat, pid, "proj", proj[b, k])
                store.add(cat, pid, "gt", gt)


@torch.no_grad()
def collect_joint_pseudo(model, criterion, loader, store: Store, max_batches: int):
    device = next(model.parameters()).device

    for bi, batch in enumerate(tqdm(loader, desc="collect joint pseudo")):
        if max_batches >= 0 and bi >= max_batches:
            break
        batch = move_batch_to_device(batch, device)

        patch = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()
        part_gt = batch["part_gt_mask_patch"].bool()
        part_valid = batch["part_valid_mask"].bool()
        part_ids = batch["part_category_id"].long()
        raw = safe_normalize(batch["part_text_feat"].float(), dim=-1)
        cat_ids = batch["category_id"].long()

        if getattr(criterion, "present_only_anchor", False):
            present = (part_gt & obj_mask[:, None, :]).sum(dim=-1) > 0
            anchor_mask = part_valid & present
        else:
            anchor_mask = part_valid
        anchor_mask = anchor_mask & obj_mask.any(dim=-1, keepdim=True)

        if not bool(anchor_mask.any().item()):
            continue

        proj = safe_normalize(model.project_clip_txt(raw), dim=-1)
        logits = torch.einsum("bkd,bnd->bkn", proj, patch)
        logits = logits / float(criterion.patch_temperature)
        logits = logits.masked_fill(~obj_mask[:, None, :], -1e4)

        z, _, _ = criterion._anchor_proto_em_pool(
            patch_tokens=patch,
            abs_logits=logits,
            obj_mask_patch=obj_mask,
            part_valid_mask=anchor_mask,
            part_gt_mask_patch=part_gt,
            num_iters=int(criterion.em_iters),
        )
        z = safe_normalize(z.float(), dim=-1)

        B, K, _ = z.shape
        for b in range(B):
            cat = int(cat_ids[b].item())
            for k in range(K):
                if not bool(anchor_mask[b, k].item()):
                    continue
                pid = int(part_ids[b, k].item())
                if pid < 0:
                    continue
                # comparable with common GT parts
                mask = part_gt[b, k] & obj_mask[b]
                if not bool(mask.any().item()):
                    continue
                store.add(cat, pid, "pseudo", z[b, k])


@torch.no_grad()
def collect_global_pseudo(model, criterion, pool_loader, store: Store, max_batches: int):
    device = next(model.parameters()).device

    for bi, batch in enumerate(tqdm(pool_loader, desc="collect global pseudo")):
        if max_batches >= 0 and bi >= max_batches:
            break
        batch = move_batch_to_device(batch, device)

        patch = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()
        part_gt = batch["part_gt_mask_patch"].bool()
        part_valid = batch["part_valid_mask"].bool()
        part_ids = batch["part_category_id"].long()
        raw = safe_normalize(batch["part_text_feat"].float(), dim=-1)
        cat_ids = batch["category_id"].long()

        anchor_mask = criterion._build_part_anchor_mask(
            part_valid_mask=part_valid,
            part_gt_mask_patch=part_gt,
            obj_mask_patch=obj_mask,
        )
        if not bool(anchor_mask.any().item()):
            continue

        proj = safe_normalize(model.project_clip_txt(raw), dim=-1)
        logits = torch.einsum("bkd,bnd->bkn", proj, patch)
        logits = logits / float(criterion.patch_temperature)
        logits = logits.masked_fill(~obj_mask[:, None, :], -1e4)

        z, _, _ = criterion._anchor_proto_em_pool(
            patch_tokens=patch,
            abs_logits=logits,
            obj_mask_patch=obj_mask,
            part_valid_mask=anchor_mask,
            part_gt_mask_patch=part_gt,
            num_iters=int(criterion.em_iters),
        )
        z = safe_normalize(z.float(), dim=-1)

        B, K, _ = z.shape
        for b in range(B):
            cat = int(cat_ids[b].item())
            for k in range(K):
                if not bool(anchor_mask[b, k].item()):
                    continue
                pid = int(part_ids[b, k].item())
                if pid >= 0:
                    store.add(cat, pid, "pseudo", z[b, k])


def rankdata_average_ties(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    sorted_a = a[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks[i:j] = 0.5 * ((i + 1) + j)
        i = j
    return ranks[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return float("nan")
    rx, ry = rankdata_average_ties(x), rankdata_average_ties(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cos_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x @ x.T


def cross_cos_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return x @ y.T


def upper(mat: np.ndarray) -> np.ndarray:
    return mat[np.triu_indices(mat.shape[0], k=1)]


def structure_spearman(a: np.ndarray, b: np.ndarray) -> float:
    return spearman(upper(cos_np(a)), upper(cos_np(b)))


def permuted_structure_spearman(s_ref: np.ndarray, s_query: np.ndarray, perm: List[int]) -> float:
    # perm[i] = query index assigned to ref/text index i
    sq = s_query[np.ix_(perm, perm)]
    return spearman(upper(s_ref), upper(sq))


def local_search_perm_spearman(s_ref: np.ndarray, s_query: np.ndarray, restarts: int, max_iter: int, seed: int):
    rng = random.Random(seed)
    n = int(s_ref.shape[0])

    def score(p):
        return permuted_structure_spearman(s_ref, s_query, p)

    best_p = list(range(n))
    best_s = score(best_p)

    init_perms = [list(range(n))]
    for _ in range(max(0, restarts - 1)):
        p = list(range(n))
        rng.shuffle(p)
        init_perms.append(p)

    for p0 in init_perms:
        p = p0[:]
        cur = score(p)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # best 2-swap
            best_swap = None
            best_swap_score = cur
            for i in range(n):
                for j in range(i + 1, n):
                    q = p[:]
                    q[i], q[j] = q[j], q[i]
                    sc = score(q)
                    if sc > best_swap_score + 1e-12:
                        best_swap_score = sc
                        best_swap = (i, j, q)
            if best_swap is not None:
                _, _, p = best_swap
                cur = best_swap_score
                improved = True

        if cur > best_s:
            best_s = cur
            best_p = p[:]

    return best_p, float(best_s)


def analyze(store: Store, cat_names, part_names, label: str, out_dir: Path, restarts: int, max_iter: int, seed: int):
    by_obj = store.by_object()
    summary = []
    mapping = []

    for cat in sorted(by_obj):
        pdict = by_obj[cat]
        pids = sorted([pid for pid, srcs in pdict.items() if all(s in srcs for s in ("raw", "proj", "pseudo", "gt"))])
        if len(pids) < 3:
            continue

        raw = torch.stack([pdict[p]["raw"] for p in pids]).numpy()
        proj = torch.stack([pdict[p]["proj"] for p in pids]).numpy()
        pseudo = torch.stack([pdict[p]["pseudo"] for p in pids]).numpy()
        gt = torch.stack([pdict[p]["gt"] for p in pids]).numpy()

        s_raw = cos_np(raw)
        s_proj = cos_np(proj)
        s_pseudo = cos_np(pseudo)
        s_gt = cos_np(gt)

        raw_pseudo_named = spearman(upper(s_raw), upper(s_pseudo))
        proj_pseudo_named = spearman(upper(s_proj), upper(s_pseudo))
        raw_gt = spearman(upper(s_raw), upper(s_gt))
        pseudo_gt = spearman(upper(s_pseudo), upper(s_gt))

        raw_perm, raw_perm_spear = local_search_perm_spearman(
            s_ref=s_raw,
            s_query=s_pseudo,
            restarts=restarts,
            max_iter=max_iter,
            seed=seed + int(cat),
        )
        proj_perm, proj_perm_spear = local_search_perm_spearman(
            s_ref=s_proj,
            s_query=s_pseudo,
            restarts=restarts,
            max_iter=max_iter,
            seed=seed + 1000 + int(cat),
        )

        # Evaluate whether the raw/proj-structure-derived assignment gives the correct GT part.
        sim_pseudo_gt = cross_cos_np(pseudo, gt)
        identity = np.arange(len(pids))
        named_self = sim_pseudo_gt[identity, identity]

        raw_assigned_cos = np.asarray([sim_pseudo_gt[raw_perm[i], i] for i in range(len(pids))])
        proj_assigned_cos = np.asarray([sim_pseudo_gt[proj_perm[i], i] for i in range(len(pids))])

        raw_identity_acc = float(np.mean(np.asarray(raw_perm) == identity))
        proj_identity_acc = float(np.mean(np.asarray(proj_perm) == identity))

        row = {
            "label": label,
            "cat_id": int(cat),
            "object_name": cat_names.get(int(cat), f"cat_{cat}"),
            "num_parts": len(pids),
            "raw_vs_gt_structure_spearman_named": float(raw_gt),
            "pseudo_vs_gt_structure_spearman_named": float(pseudo_gt),
            "raw_vs_pseudo_structure_spearman_named": float(raw_pseudo_named),
            "raw_vs_pseudo_structure_spearman_bestperm": float(raw_perm_spear),
            "raw_structure_bestperm_identity_acc": raw_identity_acc,
            "raw_structure_assignment_mean_pseudo_to_own_gt_cos": float(raw_assigned_cos.mean()),
            "raw_structure_assignment_gain_vs_named_self_cos": float(raw_assigned_cos.mean() - named_self.mean()),
            "proj_vs_pseudo_structure_spearman_named": float(proj_pseudo_named),
            "proj_vs_pseudo_structure_spearman_bestperm": float(proj_perm_spear),
            "proj_structure_bestperm_identity_acc": proj_identity_acc,
            "proj_structure_assignment_mean_pseudo_to_own_gt_cos": float(proj_assigned_cos.mean()),
            "proj_structure_assignment_gain_vs_named_self_cos": float(proj_assigned_cos.mean() - named_self.mean()),
            "named_pseudo_to_own_gt_mean_cos": float(named_self.mean()),
        }
        summary.append(row)

        for i, pid in enumerate(pids):
            raw_j = int(raw_perm[i])
            proj_j = int(proj_perm[i])
            mapping.append({
                "label": label,
                "cat_id": int(cat),
                "object_name": cat_names.get(int(cat), f"cat_{cat}"),
                "text_part_id": int(pid),
                "text_part_name": part_name(int(pid), part_names),
                "raw_spearman_matched_pseudo_part_id": int(pids[raw_j]),
                "raw_spearman_matched_pseudo_part_name": part_name(int(pids[raw_j]), part_names),
                "raw_spearman_matched_pseudo_to_own_gt_cos": float(sim_pseudo_gt[raw_j, i]),
                "raw_spearman_is_identity": bool(raw_j == i),
                "proj_spearman_matched_pseudo_part_id": int(pids[proj_j]),
                "proj_spearman_matched_pseudo_part_name": part_name(int(pids[proj_j]), part_names),
                "proj_spearman_matched_pseudo_to_own_gt_cos": float(sim_pseudo_gt[proj_j, i]),
                "proj_spearman_is_identity": bool(proj_j == i),
                "named_pseudo_to_own_gt_cos": float(named_self[i]),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{label}_raw_spearman_matching_summary.csv"
    mapping_path = out_dir / f"{label}_raw_spearman_matching_mapping.csv"
    report_path = out_dir / f"{label}_raw_spearman_matching_report.txt"

    if summary:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    if mapping:
        with mapping_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(mapping[0].keys()))
            writer.writeheader()
            writer.writerows(mapping)

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"[raw Spearman structure matching] label={label}\n")
        f.write("Goal: use raw text pairwise structure to rename pseudo visual clusters.\n")
        f.write("If bestperm Spearman improves but assignment_to_own_GT_cos does not, raw structure matches pseudo structure but not GT semantics.\n\n")
        for r in summary:
            f.write("-" * 120 + "\n")
            f.write(f"[object] {r['object_name']} parts={r['num_parts']}\n")
            f.write(
                f"raw-GT named spear={r['raw_vs_gt_structure_spearman_named']:.4f}, "
                f"pseudo-GT named spear={r['pseudo_vs_gt_structure_spearman_named']:.4f}\n"
            )
            f.write(
                f"raw-pseudo named spear={r['raw_vs_pseudo_structure_spearman_named']:.4f}, "
                f"raw-pseudo bestperm spear={r['raw_vs_pseudo_structure_spearman_bestperm']:.4f}, "
                f"raw perm identity={r['raw_structure_bestperm_identity_acc']:.4f}, "
                f"raw assignment own-GT cos={r['raw_structure_assignment_mean_pseudo_to_own_gt_cos']:.4f}, "
                f"gain_vs_named={r['raw_structure_assignment_gain_vs_named_self_cos']:.4f}\n"
            )
            f.write(
                f"proj-pseudo named spear={r['proj_vs_pseudo_structure_spearman_named']:.4f}, "
                f"proj-pseudo bestperm spear={r['proj_vs_pseudo_structure_spearman_bestperm']:.4f}, "
                f"proj perm identity={r['proj_structure_bestperm_identity_acc']:.4f}, "
                f"proj assignment own-GT cos={r['proj_structure_assignment_mean_pseudo_to_own_gt_cos']:.4f}, "
                f"gain_vs_named={r['proj_structure_assignment_gain_vs_named_self_cos']:.4f}\n"
            )

    print(f"[saved] {summary_path}")
    print(f"[saved] {mapping_path}")
    print(f"[saved] {report_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=["joint", "global"], required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out_dir", default="audits/raw_spearman_matching")

    p.add_argument("--obj_feature_name", default="avg_self_attn_out")
    p.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    p.add_argument("--obj_text_name", default="ann_feats")
    p.add_argument("--part_text_name", default="part_ann_feats")
    p.add_argument("--resize_dim", type=int, default=448)
    p.add_argument("--crop_dim", type=int, default=448)
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--with_background", action="store_true", default=False)
    p.add_argument("--path_prefix", default=None)
    p.add_argument("--min_obj_area_ratio", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=-1)
    p.add_argument("--global_sample_patches", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--restarts", type=int, default=256)
    p.add_argument("--max_iter", type=int, default=100)

    args = p.parse_args()
    device = torch.device(args.device)
    config = load_config(args.model_config)
    model = load_projector(config, args.checkpoint, device)

    dataset = build_joint_dataset(args, config)
    cat_names, part_names = build_name_maps(dataset)

    store = Store()

    common_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
    )
    collect_common(model, common_loader, store, args.max_batches)

    if args.pipeline == "joint":
        criterion = make_joint_criterion(config, model).to(device)
        pseudo_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=joint_collate_fn,
        )
        collect_joint_pseudo(model, criterion, pseudo_loader, store, args.max_batches)
    else:
        from src.dataset_global import CategoryPatchPoolDataset, global_pool_collate_fn

        criterion = make_global_criterion(config, model).to(device)
        sample_patches = None if args.global_sample_patches <= 0 else args.global_sample_patches
        pool_dataset = CategoryPatchPoolDataset(
            dataset,
            sample_patches_per_step=sample_patches,
            steps_per_epoch=None,
            store_dtype=torch.float16,
            seed=args.seed,
            fixed_subsample=True,
        )
        pool_loader = DataLoader(
            pool_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=global_pool_collate_fn,
        )
        collect_global_pseudo(model, criterion, pool_loader, store, args.max_batches)

    analyze(
        store=store,
        cat_names=cat_names,
        part_names=part_names,
        label=args.label,
        out_dir=Path(args.out_dir),
        restarts=args.restarts,
        max_iter=args.max_iter,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
