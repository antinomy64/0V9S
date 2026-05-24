#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Spearman structure correlations:

  raw feat spearman(part):         mean over object blocks of Spearman(upper_tri(cos(raw part T)),  upper_tri(cos(GT-mask part V)))
  raw feat spearman(obj):          Spearman(upper_tri(cos(raw obj T)), upper_tri(cos(GT-mask obj V)))
  proj feat spearman(part):        mean over object blocks of Spearman(upper_tri(cos(proj part T)), upper_tri(cos(GT-mask part V)))
  proj feat spearman(obj):         Spearman(upper_tri(cos(proj obj T)), upper_tri(cos(GT-mask obj V)))

  raw feat spearman(part-anchor):  mean over object blocks of Spearman(upper_tri(cos(raw part T)),  upper_tri(cos(Stage2 anchor patch V)))
  proj feat spearman(part-anchor): mean over object blocks of Spearman(upper_tri(cos(proj part T)), upper_tri(cos(Stage2 anchor patch V)))

  raw-proj T spearman(part):       mean over object blocks of Spearman(upper_tri(cos(raw part T)), upper_tri(cos(proj part T)))
  raw-proj T spearman(obj):        Spearman(upper_tri(cos(raw obj T)), upper_tri(cos(proj obj T)))

Important:
  - Part-level Spearman is averaged equally over object-category blocks.
  - Anchor patch selection directly calls JointObjPartLoss._anchor_proto_em_pool(..., return_anchor_tokens=True),
    so the anchor computation follows the Stage2 implementation instead of rewriting greedy assignment here.
  - Only prints final results. It does not save CSV/JSON files.
"""

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


def pairwise_cosine_sim(x: torch.Tensor) -> torch.Tensor:
    x = safe_normalize(x.float(), dim=-1)
    return x @ x.T


def upper_tri_no_diag(x: torch.Tensor) -> np.ndarray:
    k = x.shape[0]
    idx = torch.triu_indices(k, k, offset=1, device=x.device)
    return x[idx[0], idx[1]].detach().cpu().numpy().astype(np.float64)


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    if x.size < 2:
        return float("nan")
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def mean_feat(xs: List[torch.Tensor]) -> torch.Tensor:
    return safe_normalize(torch.stack([x.float().cpu() for x in xs], dim=0).mean(dim=0), dim=-1)


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

    if args.init_weights:
        ckpt = torch.load(args.init_weights, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=False)
    else:
        raise ValueError("--init_weights is required because Stage2 anchor selection depends on the trained projector.")

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
def collect_text_gtmask_and_stage2_anchor_prototypes(args, cfg, model, stage2_helper, device):
    dataset = build_dataset(args, cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=joint_collate_fn,
        pin_memory=True,
    )

    obj_text_bank = defaultdict(list)
    obj_visual_bank = defaultdict(list)
    part_text_bank = defaultdict(list)
    part_visual_bank = defaultdict(list)
    part_anchor_bank = defaultdict(list)

    for batch in tqdm(
        loader,
        total=len(loader),
        desc="collect T, GT-mask V, and Stage2 anchor V",
        disable=not args.show_progress,
    ):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        cat_key = pick_key(batch, ["category_id", "obj_category_id"])
        obj_text_key = pick_key(batch, ["obj_text_feat", "ann_feats", "obj_ann_feats"])
        part_text_key = pick_key(batch, ["part_text_feat", "part_ann_feats"])
        part_id_key = pick_key(batch, ["part_category_id"])
        patch_key = pick_key(batch, ["patch_tokens", args.part_feature_name])
        obj_mask_key = pick_key(batch, ["obj_mask_patch"])
        part_valid_key = pick_key(batch, ["part_valid_mask"])
        part_gt_key = pick_key(batch, ["part_gt_mask_patch"])

        cat_ids = batch[cat_key].long()
        obj_text = batch[obj_text_key].float()
        part_text = batch[part_text_key].float()
        part_ids = batch[part_id_key].long()

        patch_tokens = safe_normalize(batch[patch_key].float(), dim=-1)
        obj_mask = batch[obj_mask_key].bool()
        part_valid = batch[part_valid_key].bool()
        part_gt = batch[part_gt_key].bool()

        # Directly follow Stage2 anchor pipeline:
        # part text -> trained projector -> part-patch logits -> JointObjPartLoss._anchor_proto_em_pool(...)
        part_proj = model.project_clip_txt(part_text.float())
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
        _, _, _, anchor_tokens, anchor_valid = anchor_result

        B = int(cat_ids.shape[0])
        for b in range(B):
            cat = int(cat_ids[b].item())

            obj_text_bank[cat].append(obj_text[b].detach().cpu())

            obj_idx = torch.nonzero(obj_mask[b], as_tuple=False).squeeze(1)
            if obj_idx.numel() > 0:
                obj_visual_bank[cat].append(patch_tokens[b, obj_idx].mean(dim=0).detach().cpu())

            K = int(part_ids.shape[1])
            for k in range(K):
                if not bool(part_valid[b, k]):
                    continue
                pid = int(part_ids[b, k].item())
                if pid < 0 or pid >= int(args.num_parts):
                    continue

                part_key = (cat, pid)
                part_text_bank[part_key].append(part_text[b, k].detach().cpu())

                mask_idx = torch.nonzero(part_gt[b, k], as_tuple=False).squeeze(1)
                if mask_idx.numel() > 0:
                    part_visual_bank[part_key].append(patch_tokens[b, mask_idx].mean(dim=0).detach().cpu())

                if bool(anchor_valid[b, k]):
                    part_anchor_bank[part_key].append(anchor_tokens[b, k].detach().cpu())

    return {
        "obj_text": {cat: mean_feat(xs) for cat, xs in obj_text_bank.items() if len(xs) > 0},
        "obj_visual": {cat: mean_feat(xs) for cat, xs in obj_visual_bank.items() if len(xs) > 0},
        "part_text": {key: mean_feat(xs) for key, xs in part_text_bank.items() if len(xs) > 0},
        "part_visual": {key: mean_feat(xs) for key, xs in part_visual_bank.items() if len(xs) > 0},
        "part_anchor": {key: mean_feat(xs) for key, xs in part_anchor_bank.items() if len(xs) > 0},
    }


def compute_spearman_between_feature_graphs(text_feats: torch.Tensor, visual_feats: torch.Tensor) -> float:
    C_t = pairwise_cosine_sim(text_feats)
    C_v = pairwise_cosine_sim(visual_feats)
    vt = upper_tri_no_diag(C_t)
    vv = upper_tri_no_diag(C_v)
    return spearman_rho(vt, vv)


def compute_part_block_mean_spearman(
    text_feats: Dict[Tuple[int, int], torch.Tensor],
    visual_feats: Dict[Tuple[int, int], torch.Tensor],
) -> float:
    """
    Compute part-level Spearman by object block, then average blocks equally.

    For each object category cat:
      1) collect common part ids pid with both T and V prototypes
      2) compute Spearman(upper_tri(cos(T_cat_parts)), upper_tri(cos(V_cat_parts)))
      3) average valid category-level Spearman values with equal weight

    Categories with fewer than 3 valid parts are skipped because fewer than 3
    parts yield fewer than 2 pairwise similarities, making Spearman unstable/NaN.
    """
    cats = sorted({cat for cat, _ in text_feats.keys()} & {cat for cat, _ in visual_feats.keys()})
    block_scores: List[float] = []

    for cat in cats:
        pids = sorted(
            {pid for c, pid in text_feats.keys() if c == cat}
            & {pid for c, pid in visual_feats.keys() if c == cat}
        )
        if len(pids) < 3:
            continue

        T = torch.stack([text_feats[(cat, pid)] for pid in pids], dim=0)
        V = torch.stack([visual_feats[(cat, pid)] for pid in pids], dim=0)
        score = compute_spearman_between_feature_graphs(T, V)
        if not np.isnan(score):
            block_scores.append(score)

    if len(block_scores) == 0:
        return float("nan")
    return float(np.mean(block_scores))


def stack_common(feat_a: Dict[Any, torch.Tensor], feat_b: Dict[Any, torch.Tensor]):
    keys = sorted(set(feat_a.keys()) & set(feat_b.keys()))
    if len(keys) < 3:
        raise ValueError(f"Need at least 3 common prototypes for stable Spearman, got {len(keys)}")
    A = torch.stack([feat_a[k] for k in keys], dim=0)
    B = torch.stack([feat_b[k] for k in keys], dim=0)
    return keys, A, B


@torch.no_grad()
def project_feature_dict(model, feat_dict: Dict[Any, torch.Tensor], device, batch_size: int) -> Dict[Any, torch.Tensor]:
    keys = list(feat_dict.keys())
    out = {}
    for start in range(0, len(keys), batch_size):
        chunk_keys = keys[start:start + batch_size]
        x = torch.stack([feat_dict[k] for k in chunk_keys], dim=0).to(device)
        try:
            z = model.project_clip_txt(x)
        except TypeError as e:
            raise TypeError(
                "model.project_clip_txt must accept one tensor argument. "
                "Use ProjectionLayer/DoubleMLP checkpoints for this script."
            ) from e
        z = safe_normalize(z.float(), dim=-1).detach().cpu()
        for k, v in zip(chunk_keys, z):
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init_weights", required=True, help="Projector checkpoint for projected text features and Stage2 anchor selection")

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
    parser.add_argument("--num_parts", type=int, default=116)
    parser.add_argument("--project_batch_size", type=int, default=4096)
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

    model = build_projector(args, cfg, device)
    stage2_helper = build_stage2_loss_helper(model, cfg, device)

    protos = collect_text_gtmask_and_stage2_anchor_prototypes(args, cfg, model, stage2_helper, device)

    obj_text = protos["obj_text"]
    obj_visual = protos["obj_visual"]
    part_text = protos["part_text"]
    part_visual = protos["part_visual"]
    part_anchor = protos["part_anchor"]

    _, raw_obj_T, obj_V = stack_common(obj_text, obj_visual)

    proj_obj_text = project_feature_dict(model, obj_text, device, args.project_batch_size)
    proj_part_text = project_feature_dict(model, part_text, device, args.project_batch_size)

    _, proj_obj_T, obj_V_for_proj = stack_common(proj_obj_text, obj_visual)

    raw_part_spear = compute_part_block_mean_spearman(part_text, part_visual)
    raw_obj_spear = compute_spearman_between_feature_graphs(raw_obj_T, obj_V)
    proj_part_spear = compute_part_block_mean_spearman(proj_part_text, part_visual)
    proj_obj_spear = compute_spearman_between_feature_graphs(proj_obj_T, obj_V_for_proj)

    raw_part_anchor_spear = compute_part_block_mean_spearman(part_text, part_anchor)
    proj_part_anchor_spear = compute_part_block_mean_spearman(proj_part_text, part_anchor)

    raw_proj_part_t_spear = compute_part_block_mean_spearman(part_text, proj_part_text)
    _, raw_obj_T_for_tpres, proj_obj_T_for_tpres = stack_common(obj_text, proj_obj_text)
    raw_proj_obj_t_spear = compute_spearman_between_feature_graphs(raw_obj_T_for_tpres, proj_obj_T_for_tpres)

    print(f"raw feat spearman(part): {raw_part_spear:.8f}")
    print(f"raw feat spearman(obj): {raw_obj_spear:.8f}")
    print(f"proj feat spearman(part): {proj_part_spear:.8f}")
    print(f"proj feat spearman(obj): {proj_obj_spear:.8f}")
    print(f"raw feat spearman(part-anchor): {raw_part_anchor_spear:.8f}")
    print(f"proj feat spearman(part-anchor): {proj_part_anchor_spear:.8f}")
    print(f"raw-proj T spearman(part): {raw_proj_part_t_spear:.8f}")
    print(f"raw-proj T spearman(obj): {raw_proj_obj_t_spear:.8f}")


if __name__ == "__main__":
    main()
