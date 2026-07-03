#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 W-only part-level anchor training for OVPS/Talk2DINO test15.

v3_nogttrain rules
------------------
1) GT patch/mask/prototype is NEVER used to choose anchors, filter anchors,
   weight anchors, or compute training loss.
2) If --oracle_visible is used, GT mask is used only to define which parts are
   visible positives. This is the explicit oracle-visible assumption. Anchor
   selection/loss still do not use GT patch locations.
3) In --non_oracle mode, GT mask is not used by training at all. If the GT field
   exists in the pth, it is used only for diagnostic statistics such as
   anchor_hit_rate.
4) Samples with one positive part can train: use all valid class parts as
   competitors (--competition_set valid), so the single positive part still has
   non-GT negative part directions.
5) Multiple positive parts can be assigned anchors by Hungarian/global assignment
   to reduce greedy anchor conflicts.
6) Optional confidence weighting uses only model-computed relative margin, not GT.

No trial/fallback projector method names are used. The exact Stage1 method names
must be given through CLI. If a method or field is missing, the script fails.
"""

import argparse
import importlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Stage2WAnchorDataset(Dataset):
    """
    test15 raw-V + object-mask anchor-search dataset wrapper.

    It builds a class-level part bank:
      class_name -> full part_text / part_class_name

    Each object crop uses the class-level bank as its part directions, so a
    per-annotation K=0 part_ann_feats row does not break training.

    Required for anchor search:
      obj_gt_mask_patch: object foreground mask in the crop patch grid.

    Training/eval rule:
      - anchor search is restricted to obj_gt_mask_patch == True.
      - oracle_visible uses part_gt_mask_patch only to define visible positive
        parts. Competition defaults to visible positive parts.
      - non_oracle uses all class-bank parts as positives/competitors.
      - samples without usable part GT in oracle_visible are skipped by
        forward_one because there is no oracle visible part set.
    """

    def __init__(
        self,
        pth_path: str,
        patch_field: str,
        part_text_field: str,
        obj_mask_patch_field: str,
        part_gt_mask_patch_field: str,
        require_gt_mask_for_oracle_visible: bool,
    ) -> None:
        super().__init__()
        self.pth_path = pth_path
        self.patch_field = patch_field
        self.part_text_field = part_text_field
        self.obj_mask_patch_field = obj_mask_patch_field
        self.part_gt_mask_patch_field = part_gt_mask_patch_field
        self.require_gt_mask_for_oracle_visible = require_gt_mask_for_oracle_visible

        print(f"Loading dataset: {pth_path}")
        data = torch.load(pth_path, map_location="cpu")
        if "annotations" not in data:
            raise KeyError(f"{pth_path} does not contain top-level key 'annotations'.")

        # ------------------------------------------------------------------
        # 1) Build class-level part bank from non-empty annotation rows.
        # ------------------------------------------------------------------
        class_bank: Dict[str, Dict[str, Any]] = {}
        class_bank_sources: Dict[str, int] = {}

        for ann in data["annotations"]:
            if "class_name" not in ann:
                continue
            if part_text_field not in ann:
                continue

            class_name = str(ann.get("class_name", ""))
            part_text = ann[part_text_field]
            if not torch.is_tensor(part_text):
                part_text = torch.as_tensor(part_text)

            if part_text.ndim != 2 or part_text.shape[0] == 0:
                continue

            part_names = ann.get("part_class_name", None)
            if part_names is None:
                part_names = [f"{class_name}_part_{i}" for i in range(part_text.shape[0])]
            part_names = [str(x) for x in part_names]

            if len(part_names) != part_text.shape[0]:
                raise ValueError(
                    f"ann id={ann.get('id','unknown')}: len(part_class_name)={len(part_names)} "
                    f"!= {part_text_field}.shape[0]={part_text.shape[0]}"
                )

            if class_name not in class_bank or part_text.shape[0] > class_bank[class_name]["part_text"].shape[0]:
                class_bank[class_name] = {
                    "part_text": part_text.float().cpu(),
                    "part_class_name": part_names,
                }
                class_bank_sources[class_name] = int(ann.get("id", -1))

        if len(class_bank) == 0:
            raise RuntimeError(
                "Could not build any class-level part bank. "
                f"Check field {part_text_field} and part_class_name."
            )

        self.class_bank = class_bank
        print(f"Built class-level part banks for {len(class_bank)} classes.")
        for cls in sorted(class_bank.keys()):
            print(f"  bank[{cls}]: K={class_bank[cls]['part_text'].shape[0]}, source_ann={class_bank_sources[cls]}")

        # ------------------------------------------------------------------
        # 2) Keep annotations that have patch tokens, object patch mask, and a
        #    class-level part bank.
        # ------------------------------------------------------------------
        self.annotations: List[Dict[str, Any]] = []
        skipped = 0
        skipped_missing_field = 0
        skipped_no_class_bank = 0
        skipped_empty_patch_tokens = 0
        skipped_empty_obj_mask = 0
        usable_part_gt = 0
        no_usable_part_gt = 0

        for ann in data["annotations"]:
            missing = []
            for key in [patch_field, obj_mask_patch_field, "class_name"]:
                if key not in ann:
                    missing.append(key)
            if missing:
                skipped += 1
                skipped_missing_field += 1
                continue

            class_name = str(ann.get("class_name", ""))
            if class_name not in class_bank:
                skipped += 1
                skipped_no_class_bank += 1
                continue

            patch_tokens = ann[patch_field]
            if not torch.is_tensor(patch_tokens):
                patch_tokens = torch.as_tensor(patch_tokens)
            if patch_tokens.ndim != 2:
                raise ValueError(
                    f"ann id={ann.get('id','unknown')}: {patch_field} must be [N,D], "
                    f"got {tuple(patch_tokens.shape)}"
                )
            if patch_tokens.shape[0] == 0:
                skipped += 1
                skipped_empty_patch_tokens += 1
                continue

            obj_mask = ann[obj_mask_patch_field]
            if not torch.is_tensor(obj_mask):
                obj_mask = torch.as_tensor(obj_mask)
            obj_flat = obj_mask.bool().reshape(-1)
            if obj_flat.numel() != patch_tokens.shape[0]:
                raise ValueError(
                    f"ann id={ann.get('id','unknown')}: {obj_mask_patch_field}.numel={obj_flat.numel()} "
                    f"!= patch tokens N={patch_tokens.shape[0]}"
                )
            if obj_flat.sum().item() == 0:
                skipped += 1
                skipped_empty_obj_mask += 1
                continue

            bank_k = class_bank[class_name]["part_text"].shape[0]
            has_usable_gt = False
            if part_gt_mask_patch_field in ann:
                gt = ann[part_gt_mask_patch_field]
                if not torch.is_tensor(gt):
                    gt = torch.as_tensor(gt)
                if gt.ndim >= 1 and gt.shape[0] == bank_k and gt.numel() > 0:
                    has_usable_gt = True

            if has_usable_gt:
                usable_part_gt += 1
            else:
                no_usable_part_gt += 1
                # In oracle_visible, samples with no usable part GT are loaded
                # but forward_one will skip them because no visible positive set
                # is defined. In non_oracle, they can train with all class parts.

            self.annotations.append(ann)

        print(f"Loaded {len(self.annotations)} annotations with object masks. Skipped {skipped} annotations.")
        print(f"  skipped_missing_field={skipped_missing_field}")
        print(f"  skipped_no_class_bank={skipped_no_class_bank}")
        print(f"  skipped_empty_patch_tokens={skipped_empty_patch_tokens}")
        print(f"  skipped_empty_obj_mask={skipped_empty_obj_mask}")
        print(f"  usable_part_gt={usable_part_gt}")
        print(f"  no_usable_part_gt={no_usable_part_gt}")

        if len(self.annotations) == 0:
            raise RuntimeError(
                "No trainable annotations found. Check patch_field/class_name/part bank/object mask construction."
            )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        class_name = str(ann.get("class_name", ""))
        bank = self.class_bank[class_name]
        part_text = bank["part_text"]
        part_names = bank["part_class_name"]
        K = part_text.shape[0]

        sample = {
            "patch_tokens": ann[self.patch_field].float(),
            "obj_mask_patch": ann[self.obj_mask_patch_field].bool(),
            "part_text": part_text.float(),
            "part_valid_mask": torch.ones(K, dtype=torch.bool),
            "ann_id": ann.get("id", idx),
            "image_id": ann.get("image_id", -1),
            "class_name": class_name,
            "part_class_name": part_names,
            "has_usable_part_gt": False,
        }

        if self.part_gt_mask_patch_field in ann:
            gt = ann[self.part_gt_mask_patch_field]
            if not torch.is_tensor(gt):
                gt = torch.as_tensor(gt)
            if gt.ndim >= 1 and gt.shape[0] == K and gt.numel() > 0:
                sample["part_gt_mask_patch"] = gt.bool()
                sample["has_usable_part_gt"] = True

        return sample


def collate_list(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def build_stage1_model(config_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"])

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    ret = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded Stage1 checkpoint: {ckpt_path}")
    print("Missing keys:", getattr(ret, "missing_keys", []))
    print("Unexpected keys:", getattr(ret, "unexpected_keys", []))

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_exact_method(model: nn.Module, method_name: str):
    if not hasattr(model, method_name):
        available = [name for name in dir(model) if not name.startswith("_")]
        raise AttributeError(
            f"Stage1 model has no method '{method_name}'. No fallback is attempted. "
            f"Available public attributes include: {available[:120]}"
        )
    method = getattr(model, method_name)
    if not callable(method):
        raise TypeError(f"Stage1 attribute '{method_name}' exists but is not callable.")
    return method


def orthogonality_loss(W: torch.Tensor) -> torch.Tensor:
    I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
    return ((W.T @ W - I) ** 2).mean()


@torch.no_grad()
def retract_orthogonal_(W: torch.Tensor, mode: str) -> None:
    if mode == "none":
        return
    if mode == "qr":
        q, r = torch.linalg.qr(W.data)
        diag = torch.diagonal(r)
        sign = torch.sign(diag)
        sign[sign == 0] = 1
        q = q * sign.unsqueeze(0)
        W.data.copy_(q)
        return
    if mode == "svd":
        u, _, vh = torch.linalg.svd(W.data, full_matrices=False)
        W.data.copy_(u @ vh)
        return
    raise ValueError(f"Unknown retraction mode: {mode}")


def apply_w(text_z0: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return F.normalize(text_z0 @ W.T, dim=-1)


def compute_relative_against_comp(
    sim_comp_patch: torch.Tensor,
    pos_comp_pos: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    sim_comp_patch: [M, N], similarities for competition parts.
    pos_comp_pos: [P], row positions of positive parts in the competition list.

    rel[p, n] = sim[pos_p, n] - max_{m != pos_p} sim[m, n].
    If M < 2, returns None.
    """
    M, _ = sim_comp_patch.shape
    if M < 2:
        return None
    rel_rows = []
    for pos in pos_comp_pos.tolist():
        mask = torch.ones(M, dtype=torch.bool, device=sim_comp_patch.device)
        mask[pos] = False
        other_max = sim_comp_patch[mask].max(dim=0).values
        rel_rows.append(sim_comp_patch[pos] - other_max)
    return torch.stack(rel_rows, dim=0)


@torch.no_grad()
def select_anchors(score: torch.Tensor, mode: str) -> torch.Tensor:
    """
    score: [P, N]. Returns anchor_idx: [P]. P may be 1.
    Does not use GT.
    """
    P, N = score.shape
    if P == 0:
        raise ValueError("select_anchors got P=0.")
    if mode == "allow_duplicate" or P == 1:
        return score.argmax(dim=1)
    if mode == "greedy_unique":
        best_val, _ = score.max(dim=1)
        order = torch.argsort(best_val, descending=True)
        used = torch.zeros(N, dtype=torch.bool, device=score.device)
        anchor_idx = torch.empty(P, dtype=torch.long, device=score.device)
        for r in order.tolist():
            s = score[r].clone()
            s[used] = -torch.inf
            j = int(torch.argmax(s).item())
            if not torch.isfinite(s[j]):
                raise RuntimeError("No unused patch left during greedy_unique anchor selection.")
            anchor_idx[r] = j
            used[j] = True
        return anchor_idx
    if mode == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as e:
            raise ImportError("--anchor_select hungarian requires scipy. Install scipy or use greedy_unique.") from e
        row_ind, col_ind = linear_sum_assignment((-score.detach().cpu().numpy()))
        anchor_idx_cpu = torch.empty(P, dtype=torch.long)
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            anchor_idx_cpu[r] = c
        return anchor_idx_cpu.to(score.device)
    raise ValueError(f"Unknown anchor selection mode: {mode}")


def weighted_mean(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * w).sum() / (w.sum() + eps)


def compute_anchor_weights(
    selected_rel: Optional[torch.Tensor],
    mode: str,
    margin: float,
    temp: float,
) -> Optional[torch.Tensor]:
    """
    Returns detached per-anchor weights. Uses only model-computed relative margin, never GT.
    None means uniform unweighted mean.
    """
    if mode == "none" or selected_rel is None:
        return None
    if mode == "sigmoid_rel":
        w = torch.sigmoid((selected_rel.detach() - margin) / max(temp, 1e-6))
        return w.clamp_min(1e-4)
    if mode == "hard_rel":
        # Non-GT confidence gate. Use carefully; it can reduce gradients early.
        return (selected_rel.detach() >= margin).float()
    raise ValueError(f"Unknown anchor_weight_mode: {mode}")


@dataclass
class SampleOutput:
    loss: torch.Tensor
    anchor_ce_loss: Optional[torch.Tensor]
    rel_loss: Optional[torch.Tensor]
    sim_loss: torch.Tensor
    selected_sim_mean: float
    selected_rel_mean: Optional[float]
    anchor_weight_mean: Optional[float]
    num_positive: int
    num_competition: int
    num_used_anchors: int
    anchor_hit_rate: Optional[float]
    has_gt_diagnostic: bool


class Stage2WAnchorTrainer(nn.Module):
    def __init__(
        self,
        stage1: nn.Module,
        text_projector_fn: str,
        dim: int,
        temperature: float,
        loss_mode: str,
        rel_weight: float,
        sim_weight_when_no_rel: float,
        min_visible_patches: int,
        anchor_select: str,
        candidate_mode: str,
        competition_set: str,
        anchor_weight_mode: str,
        anchor_weight_margin: float,
        anchor_weight_temp: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.stage1 = stage1
        self.text_projector = get_exact_method(stage1, text_projector_fn)
        self.W = nn.Parameter(torch.eye(dim, device=device))
        self.temperature = temperature
        self.loss_mode = loss_mode
        self.rel_weight = rel_weight
        self.sim_weight_when_no_rel = sim_weight_when_no_rel
        self.min_visible_patches = min_visible_patches
        self.anchor_select = anchor_select
        self.candidate_mode = candidate_mode
        self.competition_set = competition_set
        self.anchor_weight_mode = anchor_weight_mode
        self.anchor_weight_margin = anchor_weight_margin
        self.anchor_weight_temp = anchor_weight_temp

    @torch.no_grad()
    def project_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # V-side cropaug_patch_tokens are already DINO/V-space features in test15.
        # They are not passed through any image projector.
        return F.normalize(patch_tokens.float(), dim=-1)

    @torch.no_grad()
    def project_part_text_before_w(self, part_text: torch.Tensor) -> torch.Tensor:
        part_z0 = self.text_projector(part_text.float())
        return F.normalize(part_z0.float(), dim=-1)

    def build_masks(self, sample: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        valid = sample["part_valid_mask"].to(device=device, dtype=torch.bool).reshape(-1)
        gt_flat = None

        if self.candidate_mode == "non_oracle":
            # No GT affects training in this mode.
            positive_mask = valid
        elif self.candidate_mode == "oracle_visible":
            # Strict oracle-visible:
            # GT is used only to decide which parts are present/positive.
            # Anchor search is still score-based and restricted by obj_mask_patch
            # later in forward_one, not by part GT locations.
            K = valid.numel()
            if "part_gt_mask_patch" not in sample:
                return torch.zeros_like(valid), torch.zeros_like(valid), None
            gt = sample["part_gt_mask_patch"].to(device=device, dtype=torch.bool)
            if gt.shape[0] != K:
                raise ValueError(f"part_gt_mask_patch.shape[0]={gt.shape[0]} != K={K}")
            gt_flat = gt.reshape(K, -1)
            visible = gt_flat.sum(dim=1) >= self.min_visible_patches
            positive_mask = valid & visible
        else:
            raise ValueError(f"Unknown candidate_mode: {self.candidate_mode}")

        if self.competition_set == "positive":
            competition_mask = positive_mask.clone()
        elif self.competition_set == "valid":
            competition_mask = valid.clone()
        else:
            raise ValueError(f"Unknown competition_set: {self.competition_set}")

        competition_mask = competition_mask | positive_mask
        return positive_mask, competition_mask, gt_flat

    def forward_one(self, sample: Dict[str, Any], device: torch.device) -> Optional[SampleOutput]:
        patch_tokens = sample["patch_tokens"].to(device=device, dtype=torch.float32)
        part_text = sample["part_text"].to(device=device, dtype=torch.float32)

        positive_mask, competition_mask, gt_flat = self.build_masks(sample, device)
        pos_idx = torch.nonzero(positive_mask, as_tuple=False).reshape(-1)
        comp_idx = torch.nonzero(competition_mask, as_tuple=False).reshape(-1)
        if pos_idx.numel() < 1 or comp_idx.numel() < 1:
            return None

        patch_z_all = self.project_patch_tokens(patch_tokens)   # [N, D], frozen

        # Anchor search is restricted to object foreground patches.
        if "obj_mask_patch" not in sample:
            raise KeyError("sample is missing obj_mask_patch; run add_obj_gt_mask_patch.py first.")
        obj_mask = sample["obj_mask_patch"].to(device=device, dtype=torch.bool).reshape(-1)
        if obj_mask.numel() != patch_z_all.shape[0]:
            raise ValueError(
                f"obj_mask_patch has {obj_mask.numel()} elements but patch tokens have N={patch_z_all.shape[0]}"
            )
        obj_patch_idx = torch.nonzero(obj_mask, as_tuple=False).reshape(-1)
        if obj_patch_idx.numel() == 0:
            return None
        patch_z = patch_z_all[obj_patch_idx]                    # [N_obj, D], frozen

        part_z0 = self.project_part_text_before_w(part_text)    # [K, D], frozen
        part_z = apply_w(part_z0, self.W)                       # [K, D], trainable through W only

        comp_part_z = part_z[comp_idx]                          # [M, D]
        sim_comp = comp_part_z @ patch_z.T                      # [M, N_obj]

        comp_pos_map = {int(g.item()): i for i, g in enumerate(comp_idx)}
        pos_comp_pos = torch.tensor([comp_pos_map[int(g.item())] for g in pos_idx], dtype=torch.long, device=device)

        sim_pos = sim_comp[pos_comp_pos]                        # [P, N]
        rel_pos = compute_relative_against_comp(sim_comp, pos_comp_pos)  # [P, N] or None
        anchor_score = rel_pos if rel_pos is not None else sim_pos

        # Anchor selection is GT-free.
        anchor_idx = select_anchors(anchor_score, self.anchor_select)

        P = pos_idx.numel()
        rows = torch.arange(P, device=device)
        selected_sim = sim_pos[rows, anchor_idx]
        selected_rel = rel_pos[rows, anchor_idx] if rel_pos is not None else None

        # Diagnostic-only GT hit rate. Does not affect loss, weights, anchor selection, or filtering.
        anchor_hit_rate = None
        if gt_flat is not None:
            if gt_flat.shape[1] != patch_z_all.shape[0]:
                raise ValueError(
                    f"GT patch count {gt_flat.shape[1]} != patch token count {patch_z_all.shape[0]}. "
                    "Check part_gt_mask_patch generation/crop grid."
                )
            gt_pos_flat = gt_flat[pos_idx][:, obj_patch_idx]
            hits = gt_pos_flat[rows, anchor_idx].float()
            anchor_hit_rate = float(hits.mean().detach().cpu().item())

        weights = compute_anchor_weights(
            selected_rel=selected_rel,
            mode=self.anchor_weight_mode,
            margin=self.anchor_weight_margin,
            temp=self.anchor_weight_temp,
        )
        if weights is not None and weights.sum().item() <= 0:
            return None

        sim_loss_vec = 1.0 - selected_sim
        sim_loss = weighted_mean(sim_loss_vec, weights) if weights is not None else sim_loss_vec.mean()

        rel_loss = None
        if selected_rel is not None:
            rel_loss_vec = F.softplus(-selected_rel / self.temperature)
            rel_loss = weighted_mean(rel_loss_vec, weights) if weights is not None else rel_loss_vec.mean()

        anchor_ce_loss = None
        if comp_idx.numel() >= 2:
            logits_for_anchor = sim_comp[:, anchor_idx].T / self.temperature  # [P, M]
            labels = pos_comp_pos
            ce_vec = F.cross_entropy(logits_for_anchor, labels, reduction="none")
            anchor_ce_loss = weighted_mean(ce_vec, weights) if weights is not None else ce_vec.mean()

        if self.loss_mode == "softplus_relative":
            loss = rel_loss if rel_loss is not None else sim_loss
        elif self.loss_mode == "anchor_ce":
            loss = anchor_ce_loss if anchor_ce_loss is not None else sim_loss
        elif self.loss_mode == "hybrid_ce_relative":
            if anchor_ce_loss is not None and rel_loss is not None:
                loss = anchor_ce_loss + self.rel_weight * rel_loss
            elif anchor_ce_loss is not None:
                loss = anchor_ce_loss + self.sim_weight_when_no_rel * sim_loss
            elif rel_loss is not None:
                loss = rel_loss + self.sim_weight_when_no_rel * sim_loss
            else:
                loss = sim_loss
        elif self.loss_mode == "one_minus_similarity":
            loss = sim_loss
        else:
            raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

        return SampleOutput(
            loss=loss,
            anchor_ce_loss=anchor_ce_loss.detach() if anchor_ce_loss is not None else None,
            rel_loss=rel_loss.detach() if rel_loss is not None else None,
            sim_loss=sim_loss.detach(),
            selected_sim_mean=float(selected_sim.mean().detach().cpu().item()),
            selected_rel_mean=float(selected_rel.mean().detach().cpu().item()) if selected_rel is not None else None,
            anchor_weight_mean=float(weights.mean().detach().cpu().item()) if weights is not None else None,
            num_positive=int(pos_idx.numel()),
            num_competition=int(comp_idx.numel()),
            num_used_anchors=int(pos_idx.numel()),
            anchor_hit_rate=anchor_hit_rate,
            has_gt_diagnostic=gt_flat is not None,
        )


def infer_projected_dim(stage1: nn.Module, text_projector_fn: str, sample: Dict[str, Any], device: torch.device) -> int:
    text_projector = get_exact_method(stage1, text_projector_fn)
    with torch.no_grad():
        part_text = sample["part_text"][:1].to(device=device, dtype=torch.float32)
        patch = sample["patch_tokens"][:1].to(device=device, dtype=torch.float32)
        text_z = text_projector(part_text).float()
        patch_z = patch.float()
    if text_z.ndim != 2 or patch_z.ndim != 2:
        raise ValueError(f"Projection outputs must be 2D. text_z={tuple(text_z.shape)}, patch_z={tuple(patch_z.shape)}")
    if text_z.shape[-1] != patch_z.shape[-1]:
        raise ValueError(f"Dim mismatch: projected text dim {text_z.shape[-1]} vs raw V patch dim {patch_z.shape[-1]}")
    return int(text_z.shape[-1])

def mean_or_none(xs: List[float]) -> Optional[float]:
    return None if len(xs) == 0 else float(sum(xs) / len(xs))


def run_epoch(
    model: Stage2WAnchorTrainer,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    ortho_lambda: float,
    retraction: str,
    max_grad_norm: float,
    desc: str,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_vals: List[float] = []
    anchor_ce_vals: List[float] = []
    rel_loss_vals: List[float] = []
    sim_loss_vals: List[float] = []
    selected_sim_vals: List[float] = []
    selected_rel_vals: List[float] = []
    anchor_weight_vals: List[float] = []
    pos_counts: List[float] = []
    comp_counts: List[float] = []
    used_anchor_counts: List[float] = []
    hit_rates: List[float] = []

    used_samples = 0
    skipped_samples = 0
    used_samples_with_gt_diag = 0
    used_samples_without_gt_diag = 0
    num_batches = 0

    count_pos_eq1 = 0
    count_pos_eq2 = 0
    count_pos_ge3 = 0
    hit_pos_eq1: List[float] = []
    hit_pos_eq2: List[float] = []
    hit_pos_ge3: List[float] = []

    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        losses: List[torch.Tensor] = []
        batch_anchor_ce: List[float] = []
        batch_rel_loss: List[float] = []
        batch_sim_loss: List[float] = []
        batch_selected_sim: List[float] = []
        batch_selected_rel: List[float] = []
        batch_anchor_weights: List[float] = []
        batch_pos_counts: List[float] = []
        batch_comp_counts: List[float] = []
        batch_used_anchor_counts: List[float] = []
        batch_hit_rates: List[float] = []

        grad_context = torch.enable_grad() if is_train else torch.no_grad()
        with grad_context:
            for sample in batch:
                out = model.forward_one(sample, device)
                if out is None:
                    skipped_samples += 1
                    continue
                used_samples += 1
                if out.has_gt_diagnostic:
                    used_samples_with_gt_diag += 1
                else:
                    used_samples_without_gt_diag += 1
                losses.append(out.loss)

                if out.anchor_ce_loss is not None:
                    batch_anchor_ce.append(float(out.anchor_ce_loss.cpu().item()))
                if out.rel_loss is not None:
                    batch_rel_loss.append(float(out.rel_loss.cpu().item()))
                batch_sim_loss.append(float(out.sim_loss.cpu().item()))
                batch_selected_sim.append(out.selected_sim_mean)
                if out.selected_rel_mean is not None:
                    batch_selected_rel.append(out.selected_rel_mean)
                if out.anchor_weight_mean is not None:
                    batch_anchor_weights.append(out.anchor_weight_mean)
                batch_pos_counts.append(float(out.num_positive))
                batch_comp_counts.append(float(out.num_competition))
                batch_used_anchor_counts.append(float(out.num_used_anchors))

                if out.num_positive == 1:
                    count_pos_eq1 += 1
                    if out.anchor_hit_rate is not None:
                        hit_pos_eq1.append(out.anchor_hit_rate)
                elif out.num_positive == 2:
                    count_pos_eq2 += 1
                    if out.anchor_hit_rate is not None:
                        hit_pos_eq2.append(out.anchor_hit_rate)
                else:
                    count_pos_ge3 += 1
                    if out.anchor_hit_rate is not None:
                        hit_pos_ge3.append(out.anchor_hit_rate)

                if out.anchor_hit_rate is not None:
                    batch_hit_rates.append(out.anchor_hit_rate)

            if len(losses) == 0:
                continue
            data_loss = torch.stack(losses).mean()
            ortho = orthogonality_loss(model.W)
            loss = data_loss + ortho_lambda * ortho

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([model.W], max_grad_norm)
            optimizer.step()
            retract_orthogonal_(model.W, retraction)

        num_batches += 1
        total_loss_vals.append(float(loss.detach().cpu().item()))
        anchor_ce_vals.extend(batch_anchor_ce)
        rel_loss_vals.extend(batch_rel_loss)
        sim_loss_vals.extend(batch_sim_loss)
        selected_sim_vals.extend(batch_selected_sim)
        selected_rel_vals.extend(batch_selected_rel)
        anchor_weight_vals.extend(batch_anchor_weights)
        pos_counts.extend(batch_pos_counts)
        comp_counts.extend(batch_comp_counts)
        used_anchor_counts.extend(batch_used_anchor_counts)
        hit_rates.extend(batch_hit_rates)

        pbar.set_postfix({
            "loss": f"{mean_or_none(total_loss_vals):.4f}" if total_loss_vals else "nan",
            "rel": f"{mean_or_none(selected_rel_vals):.4f}" if selected_rel_vals else "-",
            "hit": f"{mean_or_none(hit_rates):.3f}" if hit_rates else "-",
            "used": used_samples,
            "skip": skipped_samples,
        })

    total_seen = used_samples + skipped_samples
    return {
        "loss": mean_or_none(total_loss_vals),
        "anchor_ce_loss": mean_or_none(anchor_ce_vals),
        "rel_loss": mean_or_none(rel_loss_vals),
        "sim_loss": mean_or_none(sim_loss_vals),
        "selected_sim": mean_or_none(selected_sim_vals),
        "selected_rel": mean_or_none(selected_rel_vals),
        "anchor_weight": mean_or_none(anchor_weight_vals),
        "positive_parts": mean_or_none(pos_counts),
        "competition_parts": mean_or_none(comp_counts),
        "used_anchors": mean_or_none(used_anchor_counts),
        "anchor_hit_rate": mean_or_none(hit_rates),
        "hit_rate_pos_eq1": mean_or_none(hit_pos_eq1),
        "hit_rate_pos_eq2": mean_or_none(hit_pos_eq2),
        "hit_rate_pos_ge3": mean_or_none(hit_pos_ge3),
        "samples_pos_eq1": count_pos_eq1,
        "samples_pos_eq2": count_pos_eq2,
        "samples_pos_ge3": count_pos_ge3,
        "used_samples": used_samples,
        "used_samples_with_gt_diag": used_samples_with_gt_diag,
        "used_samples_without_gt_diag": used_samples_without_gt_diag,
        "skipped_samples": skipped_samples,
        "skip_rate": float(skipped_samples / max(total_seen, 1)),
        "num_batches": num_batches,
        "orth_loss": float(orthogonality_loss(model.W).detach().cpu().item()),
    }


def save_checkpoint(path: str, model: Stage2WAnchorTrainer, epoch: int, args: argparse.Namespace, metrics: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "W": model.W.detach().cpu(),
            "state_dict": {"W": model.W.detach().cpu()},
            "epoch": epoch,
            "args": vars(args),
            "metrics": metrics,
            "definition": (
                "part_z = normalize(stage1.<text_projector_fn>(part_ann_feats) @ W.T); "
                "patch_z = normalize(cropaug_patch_tokens[obj_gt_mask_patch]); "
                "anchor search is restricted to object-mask patches; "
                "GT part mask is not used for anchor selection/filter/loss/weights; "
                "in oracle_visible, GT part mask defines visible positives only."
            ),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)

    parser.add_argument("--text_projector_fn", type=str, required=True)

    parser.add_argument("--patch_field", type=str, default="cropaug_patch_tokens")
    parser.add_argument("--part_text_field", type=str, default="part_ann_feats")
    parser.add_argument("--obj_mask_patch_field", type=str, default="obj_gt_mask_patch")
    parser.add_argument("--part_gt_mask_patch_field", type=str, default="part_gt_mask_patch")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--oracle_visible", action="store_true",
                      help="Use GT only to define visible positive parts. GT is not used for anchors/loss/weights.")
    mode.add_argument("--non_oracle", action="store_true",
                      help="Use all valid parts as positives. GT, if present, is diagnostic-only.")

    parser.add_argument("--competition_set", type=str, default="positive", choices=["valid", "positive"],
                        help="For oracle_visible use positive: only visible/present parts compete. For non_oracle use valid: all class-bank parts compete.")

    parser.add_argument("--out_dir", type=str, default="weights/stage2_w_anchor_test15")
    parser.add_argument("--name", type=str, default="stage2_w_anchor")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss_mode", type=str, default="hybrid_ce_relative",
                        choices=["softplus_relative", "anchor_ce", "hybrid_ce_relative", "one_minus_similarity"])
    parser.add_argument("--rel_weight", type=float, default=0.25)
    parser.add_argument("--sim_weight_when_no_rel", type=float, default=0.25)
    parser.add_argument("--anchor_select", type=str, default="hungarian",
                        choices=["greedy_unique", "allow_duplicate", "hungarian"])
    parser.add_argument("--anchor_weight_mode", type=str, default="sigmoid_rel", choices=["none", "sigmoid_rel", "hard_rel"],
                        help="Non-GT confidence weighting. sigmoid_rel uses detached selected_rel; hard_rel gates by selected_rel margin.")
    parser.add_argument("--anchor_weight_margin", type=float, default=0.0)
    parser.add_argument("--anchor_weight_temp", type=float, default=0.05)
    parser.add_argument("--min_visible_patches", type=int, default=1)
    parser.add_argument("--ortho_lambda", type=float, default=0.0)
    parser.add_argument("--retraction", type=str, default="qr", choices=["qr", "svd", "none"])
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    candidate_mode = "oracle_visible" if args.oracle_visible else "non_oracle"
    require_gt_for_oracle = args.oracle_visible

    if args.oracle_visible and args.competition_set != "positive":
        print("[Warning] oracle_visible is usually expected to use --competition_set positive.")
    if args.non_oracle and args.competition_set != "valid":
        print("[Warning] non_oracle is usually expected to use --competition_set valid.")

    train_dataset = Stage2WAnchorDataset(
        args.train_dataset,
        patch_field=args.patch_field,
        part_text_field=args.part_text_field,
        obj_mask_patch_field=args.obj_mask_patch_field,
        part_gt_mask_patch_field=args.part_gt_mask_patch_field,
        require_gt_mask_for_oracle_visible=require_gt_for_oracle,
    )
    val_dataset = Stage2WAnchorDataset(
        args.val_dataset,
        patch_field=args.patch_field,
        part_text_field=args.part_text_field,
        obj_mask_patch_field=args.obj_mask_patch_field,
        part_gt_mask_patch_field=args.part_gt_mask_patch_field,
        require_gt_mask_for_oracle_visible=require_gt_for_oracle,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_list, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_list, pin_memory=True)

    stage1 = build_stage1_model(args.model_config, args.stage1_ckpt, device)
    dim = infer_projected_dim(stage1, args.text_projector_fn, train_dataset[0], device)
    print(f"Projected dimension inferred: D={dim}")
    print(f"Candidate mode: {candidate_mode}")
    print(f"competition_set={args.competition_set}, anchor_select={args.anchor_select}")
    print(f"anchor_weight_mode={args.anchor_weight_mode}, margin={args.anchor_weight_margin}, temp={args.anchor_weight_temp}")
    print("GT usage: anchor search is restricted to object-mask patches. In oracle_visible, GT defines visible positive parts only; competition should be positive. In non_oracle, use all class-bank parts.")
    print(f"Loss mode: {args.loss_mode}")

    model = Stage2WAnchorTrainer(
        stage1=stage1,
        text_projector_fn=args.text_projector_fn,
        dim=dim,
        temperature=args.temperature,
        loss_mode=args.loss_mode,
        rel_weight=args.rel_weight,
        sim_weight_when_no_rel=args.sim_weight_when_no_rel,
        min_visible_patches=args.min_visible_patches,
        anchor_select=args.anchor_select,
        candidate_mode=candidate_mode,
        competition_set=args.competition_set,
        anchor_weight_mode=args.anchor_weight_mode,
        anchor_weight_margin=args.anchor_weight_margin,
        anchor_weight_temp=args.anchor_weight_temp,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW([model.W], lr=args.lr, weight_decay=args.weight_decay)

    log_path = os.path.join(args.out_dir, f"{args.name}_log.jsonl")
    summary_path = os.path.join(args.out_dir, f"{args.name}_summary.json")
    best_path = os.path.join(args.out_dir, f"{args.name}_best.pth")
    last_path = os.path.join(args.out_dir, f"{args.name}_last.pth")
    if os.path.exists(log_path):
        os.remove(log_path)

    best_val = math.inf
    history = []
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer,
                                  ortho_lambda=args.ortho_lambda, retraction=args.retraction,
                                  max_grad_norm=args.max_grad_norm, desc=f"train epoch {epoch}")
        val_metrics = run_epoch(model, val_loader, device, optimizer=None,
                                ortho_lambda=args.ortho_lambda, retraction="none",
                                max_grad_norm=0.0, desc=f"val epoch {epoch}")
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "args": vars(args)}
        history.append(record)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, indent=2, ensure_ascii=False))

        save_checkpoint(last_path, model, epoch, args, record)
        val_loss = val_metrics["loss"] if val_metrics["loss"] is not None else math.inf
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, epoch, args, record)
            print(f"Saved best checkpoint to {best_path}")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            ep_path = os.path.join(args.out_dir, f"{args.name}_epoch{epoch:03d}.pth")
            save_checkpoint(ep_path, model, epoch, args, record)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"best_val": best_val, "history": history}, f, indent=2, ensure_ascii=False)

    print(f"Done. Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
