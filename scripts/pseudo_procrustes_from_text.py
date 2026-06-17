#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit a Procrustes-style transform from text features to pseudo-label visual prototypes.

Purpose:
  This is the non-GT analogue of oracle_orthogonal_procrustes_part.py.

  Oracle version:
      projected_text(part p) -> GT_visual_proto(part p)

  This script:
      raw/projected_text(part p) -> pseudo_visual_proto(part p)

  Then it evaluates whether the transform learned from pseudo prototypes also
  improves retrieval against GT visual prototypes.

Interpretation:
  - If text->pseudo transform improves text->GT retrieval a lot:
      pseudo labels contain useful correspondence signal.
  - If it fits pseudo well but does not improve GT retrieval:
      pseudo labels are self-consistent but semantically misnamed/mismatched.
  - If it cannot even fit pseudo well:
      pseudo prototypes are noisy/collapsed.

Notes:
  - text_source=projected: source dim normally equals DINO dim, so the map is a
    square orthogonal Q, comparable to the oracle Q.
  - text_source=raw: raw CLIP text dim may differ from DINO dim. Then the script
    fits a rectangular semi-orthogonal R by default. This is not a square Q, but
    it is the closest "no-stretch" analogue for 512 -> 768.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
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


def get_part_names_from_dataset(dataset: DinoClipJointDataset, num_parts: int) -> List[str]:
    names = [f"part_{i}" for i in range(num_parts)]

    samples = dataset.data.values() if isinstance(dataset.data, dict) else dataset.data
    for sample in samples:
        pids = sample.get("part_category_id", [])
        pnames = sample.get("part_class_name", [])
        if torch.is_tensor(pids):
            pids = pids.detach().cpu().tolist()
        for pid, pname in zip(pids, pnames):
            pid = int(pid)
            if 0 <= pid < num_parts:
                names[pid] = str(pname)

    for _, bank in getattr(dataset, "class_part_bank", {}).items():
        pids = bank.get("part_ids", [])
        pnames = bank.get("part_names", [])
        if torch.is_tensor(pids):
            pids = pids.detach().cpu().tolist()
        for pid, pname in zip(pids, pnames):
            pid = int(pid)
            if 0 <= pid < num_parts:
                names[pid] = str(pname)

    return names


def build_joint_dataset(args, config: Dict, dataset_path: str) -> DinoClipJointDataset:
    dataset_cfg = config.get("dataset", {})
    min_obj_area_ratio = float(dataset_cfg.get("min_obj_area_ratio", args.min_obj_area_ratio))
    is_wds = ".tar" in dataset_path

    return DinoClipJointDataset(
        dataset_path,
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


class PartProtoAccumulator:
    def __init__(self, num_parts: int):
        self.num_parts = int(num_parts)
        self.sums: Dict[str, List[torch.Tensor | None]] = {}
        self.counts: Dict[str, torch.Tensor] = {}

    def add(self, source: str, pid: int, vec: torch.Tensor):
        pid = int(pid)
        if pid < 0 or pid >= self.num_parts:
            return
        vec = safe_normalize(vec.detach().float().cpu(), dim=-1)

        if source not in self.sums:
            self.sums[source] = [None for _ in range(self.num_parts)]
            self.counts[source] = torch.zeros(self.num_parts, dtype=torch.long)

        if self.sums[source][pid] is None:
            self.sums[source][pid] = torch.zeros_like(vec)

        self.sums[source][pid] += vec
        self.counts[source][pid] += 1

    def mean_matrix(self, source: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if source not in self.sums:
            raise KeyError(f"source not collected: {source}")

        first = None
        for s in self.sums[source]:
            if s is not None:
                first = s
                break
        if first is None:
            raise RuntimeError(f"no vectors collected for source={source}")

        dim = int(first.numel())
        mat = torch.zeros((self.num_parts, dim), dtype=torch.float32)
        valid = torch.zeros(self.num_parts, dtype=torch.bool)
        count = self.counts[source].clone()

        for pid, s in enumerate(self.sums[source]):
            if s is None or count[pid] <= 0:
                continue
            mat[pid] = s / int(count[pid].item())
            valid[pid] = True

        mat = safe_normalize(mat, dim=-1)
        return mat, valid, count


@torch.no_grad()
def collect_text_and_gt(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    acc: PartProtoAccumulator,
    max_batches: int,
):
    device = next(model.parameters()).device

    for batch_idx, batch in enumerate(tqdm(loader, desc="collect raw/projected text + GT")):
        if max_batches >= 0 and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        patch_tokens = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()
        part_gt = batch["part_gt_mask_patch"].bool()
        part_valid = batch["part_valid_mask"].bool()
        part_ids = batch["part_category_id"].long()
        raw_text = safe_normalize(batch["part_text_feat"].float(), dim=-1)
        proj_text = safe_normalize(model.project_clip_txt(raw_text), dim=-1)

        B, K, _ = raw_text.shape
        for b in range(B):
            for k in range(K):
                if not bool(part_valid[b, k].item()):
                    continue
                pid = int(part_ids[b, k].item())
                if pid < 0:
                    continue

                mask = part_gt[b, k] & obj_mask[b]
                if not bool(mask.any().item()):
                    continue

                gt_proto = safe_normalize(patch_tokens[b, mask].mean(dim=0), dim=-1)
                acc.add("raw_text", pid, raw_text[b, k])
                acc.add("projected_text", pid, proj_text[b, k])
                acc.add("gt", pid, gt_proto)


@torch.no_grad()
def collect_joint_pseudo(
    *,
    model: torch.nn.Module,
    criterion: JointObjPartLoss,
    loader: DataLoader,
    acc: PartProtoAccumulator,
    max_batches: int,
):
    device = next(model.parameters()).device

    for batch_idx, batch in enumerate(tqdm(loader, desc="collect joint pseudo")):
        if max_batches >= 0 and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        patch_tokens = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()
        part_gt = batch["part_gt_mask_patch"].bool()
        part_valid = batch["part_valid_mask"].bool()
        part_ids = batch["part_category_id"].long()
        raw_text = safe_normalize(batch["part_text_feat"].float(), dim=-1)

        if raw_text.shape[1] == 0 or not bool(part_valid.any().item()):
            continue

        if getattr(criterion, "present_only_anchor", False):
            present = (part_gt & obj_mask[:, None, :]).sum(dim=-1) > 0
            anchor_mask = part_valid & present
        else:
            anchor_mask = part_valid
        anchor_mask = anchor_mask & obj_mask.any(dim=-1, keepdim=True)

        if not bool(anchor_mask.any().item()):
            continue

        proj_text = safe_normalize(model.project_clip_txt(raw_text), dim=-1)
        abs_logits = torch.einsum("bkd,bnd->bkn", proj_text, patch_tokens)
        abs_logits = abs_logits / float(criterion.patch_temperature)
        abs_logits = abs_logits.masked_fill(~obj_mask[:, None, :], -1e4)

        z_pseudo, _, _ = criterion._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask,
            part_valid_mask=anchor_mask,
            part_gt_mask_patch=part_gt,
            num_iters=int(criterion.em_iters),
        )
        z_pseudo = safe_normalize(z_pseudo.float(), dim=-1)

        B, K, _ = z_pseudo.shape
        for b in range(B):
            for k in range(K):
                if not bool(anchor_mask[b, k].item()):
                    continue
                pid = int(part_ids[b, k].item())
                if pid < 0:
                    continue
                # Keep comparable with GT: require the target part exists in this crop.
                mask = part_gt[b, k] & obj_mask[b]
                if not bool(mask.any().item()):
                    continue
                acc.add("pseudo", pid, z_pseudo[b, k])


@torch.no_grad()
def collect_global_pseudo(
    *,
    model: torch.nn.Module,
    criterion,
    pool_loader: DataLoader,
    acc: PartProtoAccumulator,
    max_batches: int,
):
    device = next(model.parameters()).device

    for batch_idx, batch in enumerate(tqdm(pool_loader, desc="collect global pseudo")):
        if max_batches >= 0 and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        patch_tokens = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()
        part_gt = batch["part_gt_mask_patch"].bool()
        part_valid = batch["part_valid_mask"].bool()
        part_ids = batch["part_category_id"].long()
        raw_text = safe_normalize(batch["part_text_feat"].float(), dim=-1)

        if raw_text.shape[1] == 0 or not bool(part_valid.any().item()):
            continue

        anchor_mask = criterion._build_part_anchor_mask(
            part_valid_mask=part_valid,
            part_gt_mask_patch=part_gt,
            obj_mask_patch=obj_mask,
        )
        if not bool(anchor_mask.any().item()):
            continue

        proj_text = safe_normalize(model.project_clip_txt(raw_text), dim=-1)
        abs_logits = torch.einsum("bkd,bnd->bkn", proj_text, patch_tokens)
        abs_logits = abs_logits / float(criterion.patch_temperature)
        abs_logits = abs_logits.masked_fill(~obj_mask[:, None, :], -1e4)

        z_pseudo, _, _ = criterion._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask,
            part_valid_mask=anchor_mask,
            part_gt_mask_patch=part_gt,
            num_iters=int(criterion.em_iters),
        )
        z_pseudo = safe_normalize(z_pseudo.float(), dim=-1)

        B, K, _ = z_pseudo.shape
        for b in range(B):
            for k in range(K):
                if not bool(anchor_mask[b, k].item()):
                    continue
                pid = int(part_ids[b, k].item())
                if pid < 0:
                    continue
                acc.add("pseudo", pid, z_pseudo[b, k])


def solve_square_orthogonal(text: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cross = text.T @ target
    u, s, vh = torch.linalg.svd(cross, full_matrices=False)
    q = u @ vh
    return q, s


def solve_rectangular_semi_orthogonal(text: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve min_R ||text @ R - target||_F with an orthogonality constraint.

    If d_text <= d_target, R has row-orthonormal constraint R R^T = I.
    This preserves dot products among text features after mapping:
      (xR)·(yR) = x·y.

    If d_text > d_target, R has column-orthonormal constraint R^T R = I.
    """
    cross = text.T @ target  # [d_text, d_target]
    u, s, vh = torch.linalg.svd(cross, full_matrices=False)
    r = u @ vh
    return r, s


def solve_linear_ls(text: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Unconstrained least squares. This can stretch/shear; use only as a diagnostic.
    sol = torch.linalg.lstsq(text.float(), target.float()).solution
    return sol, torch.empty(0)


def retrieval_metrics(
    query: torch.Tensor,
    target: torch.Tensor,
    part_ids: torch.Tensor,
    part_names: List[str],
    label: str,
    print_per_part: bool,
) -> Dict[str, float]:
    query = safe_normalize(query.float(), dim=-1)
    target = safe_normalize(target.float(), dim=-1)
    sim = query @ target.T

    n = int(sim.shape[0])
    diag = sim.diag()
    order = torch.argsort(sim, dim=1, descending=True)
    target_idx = torch.arange(n, device=sim.device)
    ranks = (order == target_idx[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    top1_idx = order[:, 0]

    top1 = (ranks <= 1).float().mean().item()
    top5 = (ranks <= min(5, n)).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()
    mean_self_cos = diag.mean().item()
    median_rank = ranks.float().median().item()

    if n > 1:
        masked = sim.clone()
        masked[target_idx, target_idx] = -1e9
        best_other = masked.max(dim=1).values
        margin = (diag - best_other).mean().item()
    else:
        margin = float("nan")

    print("")
    print("=" * 100)
    print(f"[{label}] num_parts={n}")
    print(
        f"top1={top1:.4f}, top5={top5:.4f}, mrr={mrr:.4f}, "
        f"mean_self_cos={mean_self_cos:.4f}, mean_self_margin={margin:.4f}, "
        f"median_rank={median_rank:.1f}"
    )

    if print_per_part:
        print("")
        print(f"{'pid':>4}  {'part':<34}  {'rank':>5}  {'self_cos':>9}  {'top1_part':<34}  {'top1_cos':>9}")
        print("-" * 110)
        for i in range(n):
            pid = int(part_ids[i].item())
            top_pid = int(part_ids[int(top1_idx[i].item())].item())
            pname = part_names[pid] if 0 <= pid < len(part_names) else f"part_{pid}"
            top_name = part_names[top_pid] if 0 <= top_pid < len(part_names) else f"part_{top_pid}"
            print(
                f"{pid:>4d}  {pname:<34.34}  {int(ranks[i].item()):>5d}  "
                f"{float(diag[i].item()):>9.4f}  {top_name:<34.34}  "
                f"{float(sim[i, int(top1_idx[i].item())].item()):>9.4f}"
            )

    return {
        "num_parts": n,
        "top1": float(top1),
        "top5": float(top5),
        "mrr": float(mrr),
        "mean_self_cos": float(mean_self_cos),
        "mean_self_margin": float(margin),
        "median_rank": float(median_rank),
    }


def fold_square_q_into_linear_checkpoint(model, q: torch.Tensor, config: Dict, out_path: str):
    model_cfg = config.get("model", {})
    act = model_cfg.get("act", None)
    hidden_layer = model_cfg.get("hidden_layer", False)
    act_is_none = act is None or str(act).lower() in {"none", "null"}
    hidden_is_false = hidden_layer in (False, None, 0, "False", "false")

    if not act_is_none or not hidden_is_false:
        raise RuntimeError(
            "Rotated checkpoint folding only supports a linear projector with "
            f"act=null and hidden_layer=False. Got act={act!r}, hidden_layer={hidden_layer!r}."
        )

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if "linear_layer.weight" not in state:
        raise KeyError("No linear_layer.weight found; cannot fold Q.")

    q_cpu = q.detach().cpu().float()
    w = state["linear_layer.weight"].float()
    if w.shape[0] != q_cpu.shape[0] or q_cpu.shape[0] != q_cpu.shape[1]:
        raise ValueError(f"Shape mismatch: W={tuple(w.shape)}, Q={tuple(q_cpu.shape)}")

    state["linear_layer.weight"] = (q_cpu.T @ w).to(state["linear_layer.weight"].dtype)
    if "linear_layer.bias" in state:
        b = state["linear_layer.bias"].float()
        state["linear_layer.bias"] = (q_cpu.T @ b).to(state["linear_layer.bias"].dtype)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    print(f"[saved rotated checkpoint] {out}")


def collect_all(args, config: Dict, model: torch.nn.Module, dataset_path: str):
    joint_dataset = build_joint_dataset(args, config, dataset_path)
    part_names = get_part_names_from_dataset(joint_dataset, args.num_parts)

    acc = PartProtoAccumulator(args.num_parts)

    common_loader = DataLoader(
        joint_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=joint_collate_fn,
    )
    collect_text_and_gt(
        model=model,
        loader=common_loader,
        acc=acc,
        max_batches=int(args.max_batches),
    )

    if args.pipeline == "joint":
        criterion = make_joint_criterion(config, model).to(next(model.parameters()).device)
        pseudo_loader = DataLoader(
            joint_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=joint_collate_fn,
        )
        collect_joint_pseudo(
            model=model,
            criterion=criterion,
            loader=pseudo_loader,
            acc=acc,
            max_batches=int(args.max_batches),
        )
    else:
        from src.dataset_global import CategoryPatchPoolDataset, global_pool_collate_fn

        criterion = make_global_criterion(config, model).to(next(model.parameters()).device)
        sample_patches = None if int(args.global_sample_patches) <= 0 else int(args.global_sample_patches)
        pool_dataset = CategoryPatchPoolDataset(
            joint_dataset,
            sample_patches_per_step=sample_patches,
            steps_per_epoch=None,
            store_dtype=torch.float16,
            seed=int(args.seed),
            fixed_subsample=True,
        )
        pool_loader = DataLoader(
            pool_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=global_pool_collate_fn,
        )
        collect_global_pseudo(
            model=model,
            criterion=criterion,
            pool_loader=pool_loader,
            acc=acc,
            max_batches=int(args.max_batches),
        )

    protos = {}
    for src in ("raw_text", "projected_text", "pseudo", "gt"):
        mat, valid, count = acc.mean_matrix(src)
        protos[f"{src}_mean"] = mat
        protos[f"{src}_valid"] = valid
        protos[f"{src}_count"] = count

    return protos, part_names


def evaluate_with_transform(
    *,
    split_name: str,
    protos: Dict[str, torch.Tensor],
    transform: torch.Tensor,
    text_source: str,
    part_names: List[str],
    print_per_part: bool,
):
    text_key = "raw_text_mean" if text_source == "raw" else "projected_text_mean"
    valid = (
        protos[text_key.replace("_mean", "_valid")]
        & protos["pseudo_valid"]
        & protos["gt_valid"]
    )
    part_ids = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if part_ids.numel() < 2:
        raise RuntimeError(f"No enough valid parts for {split_name}: {int(part_ids.numel())}")

    text = protos[text_key][valid].float()
    pseudo = protos["pseudo_mean"][valid].float()
    gt = protos["gt_mean"][valid].float()

    text_mapped = safe_normalize(text @ transform.cpu().float(), dim=-1)

    results = {
        "text_vs_pseudo_before": retrieval_metrics(
            text, pseudo, part_ids, part_names,
            label=f"{split_name}: {text_source} text vs pseudo BEFORE pseudo-Procrustes",
            print_per_part=print_per_part,
        ),
        "text_vs_gt_before": retrieval_metrics(
            text, gt, part_ids, part_names,
            label=f"{split_name}: {text_source} text vs GT BEFORE pseudo-Procrustes",
            print_per_part=print_per_part,
        ),
        "mapped_text_vs_pseudo_after": retrieval_metrics(
            text_mapped, pseudo, part_ids, part_names,
            label=f"{split_name}: mapped {text_source} text vs pseudo AFTER pseudo-Procrustes",
            print_per_part=print_per_part,
        ),
        "mapped_text_vs_gt_after": retrieval_metrics(
            text_mapped, gt, part_ids, part_names,
            label=f"{split_name}: mapped {text_source} text vs GT AFTER pseudo-Procrustes",
            print_per_part=print_per_part,
        ),
    }

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=["joint", "global"], required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--fit_dataset", required=True)
    p.add_argument("--eval_dataset", default=None)
    p.add_argument("--checkpoint", "--init_weights", dest="checkpoint", required=True)

    p.add_argument("--text_source", choices=["raw", "projected"], default="projected")
    p.add_argument("--map_type", choices=["auto", "orthogonal", "semi_orthogonal", "linear_ls"], default="auto")

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

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_parts", type=int, default=116)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_batches", type=int, default=-1)
    p.add_argument("--global_sample_patches", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--print_per_part", action="store_true")
    p.add_argument("--out_transform_pth", default=None)
    p.add_argument("--out_rotated_ckpt", default=None, help="Only valid for text_source=projected with square orthogonal Q.")
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.model_config)
    model = load_projector(config, args.checkpoint, device)

    print("[pseudo-Procrustes warning]")
    print("  This fits text -> pseudo-label visual prototypes, not text -> GT prototypes.")
    print("  It tests whether pseudo labels provide a useful correspondence signal.")
    print(f"[pipeline] {args.pipeline}")
    print(f"[text_source] {args.text_source}")
    print(f"[fit_dataset] {args.fit_dataset}")
    print(f"[eval_dataset] {args.eval_dataset}")

    fit_protos, part_names = collect_all(args, config, model, args.fit_dataset)

    text_key = "raw_text_mean" if args.text_source == "raw" else "projected_text_mean"
    fit_valid = (
        fit_protos[text_key.replace("_mean", "_valid")]
        & fit_protos["pseudo_valid"]
        & fit_protos["gt_valid"]
    )
    fit_part_ids = torch.nonzero(fit_valid, as_tuple=False).squeeze(1)
    if fit_part_ids.numel() < 2:
        raise RuntimeError(f"Need >=2 valid parts, got {int(fit_part_ids.numel())}")

    fit_text = fit_protos[text_key][fit_valid].float()
    fit_pseudo = fit_protos["pseudo_mean"][fit_valid].float()

    src_dim = int(fit_text.shape[1])
    tgt_dim = int(fit_pseudo.shape[1])

    map_type = args.map_type
    if map_type == "auto":
        map_type = "orthogonal" if src_dim == tgt_dim else "semi_orthogonal"

    if map_type == "orthogonal":
        if src_dim != tgt_dim:
            raise ValueError(f"orthogonal requires same dims, got text={src_dim}, pseudo={tgt_dim}")
        transform, singular_values = solve_square_orthogonal(fit_text, fit_pseudo)
    elif map_type == "semi_orthogonal":
        transform, singular_values = solve_rectangular_semi_orthogonal(fit_text, fit_pseudo)
    else:
        transform, singular_values = solve_linear_ls(fit_text, fit_pseudo)

    print("")
    print("=" * 100)
    print(f"[transform] map_type={map_type}, shape={tuple(transform.shape)}, fit_parts={int(fit_part_ids.numel())}")
    if transform.shape[0] == transform.shape[1] and map_type == "orthogonal":
        q = transform
        orth_error = ((q.T @ q) - torch.eye(q.shape[0])).abs().max().item()
        det = torch.linalg.det(q).item()
        print(f"[Q check] det={det:.6f}, max_abs(Q^TQ-I)={orth_error:.8f}")
    elif map_type == "semi_orthogonal":
        if transform.shape[0] <= transform.shape[1]:
            err = ((transform @ transform.T) - torch.eye(transform.shape[0])).abs().max().item()
            print(f"[R check] rows orthonormal: max_abs(RR^T-I)={err:.8f}")
        else:
            err = ((transform.T @ transform) - torch.eye(transform.shape[1])).abs().max().item()
            print(f"[R check] cols orthonormal: max_abs(R^TR-I)={err:.8f}")

    results = {
        "pipeline": args.pipeline,
        "model_config": args.model_config,
        "checkpoint": args.checkpoint,
        "fit_dataset": args.fit_dataset,
        "eval_dataset": args.eval_dataset,
        "text_source": args.text_source,
        "map_type": map_type,
        "transform_shape": list(transform.shape),
        "fit_part_ids": fit_part_ids.tolist(),
        "singular_values": singular_values.detach().cpu().tolist() if singular_values.numel() > 0 else [],
    }

    results["fit"] = evaluate_with_transform(
        split_name="fit",
        protos=fit_protos,
        transform=transform,
        text_source=args.text_source,
        part_names=part_names,
        print_per_part=args.print_per_part,
    )

    if args.eval_dataset:
        eval_protos, _ = collect_all(args, config, model, args.eval_dataset)
        results["eval"] = evaluate_with_transform(
            split_name="eval",
            protos=eval_protos,
            transform=transform,
            text_source=args.text_source,
            part_names=part_names,
            print_per_part=args.print_per_part,
        )

    if args.out_transform_pth:
        out = Path(args.out_transform_pth)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "transform": transform.detach().cpu(),
                "map_type": map_type,
                "text_source": args.text_source,
                "pipeline": args.pipeline,
                "fit_part_ids": fit_part_ids.cpu(),
                "singular_values": singular_values.detach().cpu(),
                "model_config": args.model_config,
                "checkpoint": args.checkpoint,
                "fit_dataset": args.fit_dataset,
            },
            out,
        )
        print(f"[saved transform] {out}")

    if args.out_rotated_ckpt:
        if args.text_source != "projected" or map_type != "orthogonal" or transform.shape[0] != transform.shape[1]:
            raise RuntimeError("--out_rotated_ckpt is only valid for text_source=projected + square orthogonal map.")
        fold_square_q_into_linear_checkpoint(model, transform, config, args.out_rotated_ckpt)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[saved json] {out}")


if __name__ == "__main__":
    main()
