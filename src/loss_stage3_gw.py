"""
Stage 3 GW loss for Talk2DINO.

This file implements the proposed Stage 2 + Stage 3 merged experiment:

Stage 2, in memory:
    Use the initialized projector to extract pseudo part visual features z_part
    from the train set, then mean all pseudo z_part by part_category_id to build
    global visual prototypes.

Stage 3:
    Train with only:
        total = lambda_obj * Lo + lambda_gw * Lgw

    Lo  = object-level InfoNCE, same as existing object branch.
    Lgw = per-object-class GW matching between pre-text part structure and
          global visual prototype structure, followed by soft prototype alignment.

No real part GT masks are used by this loss. Existing datasets may still return
part_gt_mask_patch, but Stage 3 loss never reads it. During Stage-2 extraction,
a dummy all-false part_gt_mask_patch is passed only to reuse the exact Stage-1
_anchor_proto_em_pool implementation.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import ContrastiveLoss
from src.loss_joint import JointObjPartLoss


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def compute_relative_scores(local_scores: torch.Tensor) -> torch.Tensor:
    """
    Compute the same relative part-patch scores used by loss_joint.py.

    local_scores: [K, M], scores for K parts over M object-internal patches.
    relative_score(k, m) = score(k, m) - best_score(other_part, m)
    """
    k, _ = local_scores.shape
    if k <= 1:
        return local_scores

    top2_vals, top2_idx = torch.topk(local_scores, k=min(2, k), dim=0)
    best_vals = top2_vals[0]
    best_idx = top2_idx[0]
    second_vals = top2_vals[1]

    row_ids = torch.arange(k, device=local_scores.device)[:, None]
    is_top1 = row_ids == best_idx[None, :]
    best_other = torch.where(is_top1, second_vals[None, :], best_vals[None, :])
    return local_scores - best_other


@torch.no_grad()
def extract_z_part_from_batch(
    model: nn.Module,
    batch: Dict,
    patch_temperature: float = 0.07,
    em_iters: int = 1,
    anchor_helper: JointObjPartLoss | None = None,
) -> torch.Tensor:
    """
    Extract pseudo part visual features z_part from one batch by directly reusing
    the Stage-1 anchor/prototype routine:

        JointObjPartLoss._anchor_proto_em_pool(...)

    This is intentionally tied to src/loss_joint.py so Stage-2 prototype
    extraction uses the exact same anchor selection and EM pooling logic as the
    current Stage-1 joint training pipeline.

    No real part GT masks are used. A dummy all-false part_gt_mask_patch is
    passed only because _anchor_proto_em_pool also computes anchor audit metrics;
    that mask does not affect anchor selection, EM assignment, or z_part.

    Returns:
        z_part: [B, K, D]
    """
    device = next(model.parameters()).device

    part_text_feat = batch["part_text_feat"].to(device).float()
    patch_tokens = batch["patch_tokens"].to(device).float()
    obj_mask_patch = batch["obj_mask_patch"].to(device).bool()
    part_valid_mask = batch["part_valid_mask"].to(device).bool()

    if anchor_helper is None:
        anchor_helper = JointObjPartLoss(
            sim_model=model,
            obj_ltype="infonce",
            lambda_obj=0.0,
            lambda_inst=0.0,
            lambda_overlap=0.0,
            lambda_spear=0.0,
            patch_temperature=patch_temperature,
            em_iters=em_iters,
        ).to(device)

    part_proj = model.project_clip_txt(part_text_feat)  # [B, K, D]
    part_proj = anchor_helper._safe_normalize(part_proj, dim=-1)
    patch_tokens = anchor_helper._safe_normalize(patch_tokens, dim=-1)

    # Match Stage-1 JointObjPartLoss.forward exactly.
    abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / float(patch_temperature)
    abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

    dummy_part_gt_mask_patch = torch.zeros(
        part_valid_mask.shape[0],
        part_valid_mask.shape[1],
        patch_tokens.shape[1],
        dtype=torch.bool,
        device=device,
    )

    z_part, _, _ = anchor_helper._anchor_proto_em_pool(
        patch_tokens=patch_tokens,
        abs_logits=abs_logits,
        obj_mask_patch=obj_mask_patch,
        part_valid_mask=part_valid_mask,
        part_gt_mask_patch=dummy_part_gt_mask_patch,
        num_iters=em_iters,
    )

    return z_part

@torch.no_grad()
def build_stage2_visual_prototypes(
    model: nn.Module,
    dataloader,
    num_parts: int,
    patch_temperature: float = 0.07,
    em_iters: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Build global visual prototypes in memory.

    This is intentionally the simple Stage 2 baseline requested:
      - no top-k confidence filtering
      - no per-part sample filtering
      - mean all pseudo z_part by part_category_id

    Returns:
        {
            "visual_proto": [num_parts, D], normalized,
            "proto_count":  [num_parts], number of pseudo z_part used,
        }
    """
    device = next(model.parameters()).device
    model.eval()

    proto_sum = None
    proto_count = torch.zeros(num_parts, device=device)

    # Reuse the exact Stage-1 anchor/EM helper once for all batches.
    anchor_helper = JointObjPartLoss(
        sim_model=model,
        obj_ltype="infonce",
        lambda_obj=0.0,
        lambda_inst=0.0,
        lambda_overlap=0.0,
        lambda_spear=0.0,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
    ).to(device)
    anchor_helper.eval()

    for batch in dataloader:
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        batch = moved

        z_part = extract_z_part_from_batch(
            model=model,
            batch=batch,
            patch_temperature=patch_temperature,
            em_iters=em_iters,
            anchor_helper=anchor_helper,
        )

        part_ids = batch["part_category_id"].long()       # [B, K]
        part_valid = batch["part_valid_mask"].bool()      # [B, K]

        if proto_sum is None:
            dim = z_part.shape[-1]
            proto_sum = torch.zeros(num_parts, dim, device=device)

        bsz, max_k = part_ids.shape
        for b in range(bsz):
            for k in range(max_k):
                if not bool(part_valid[b, k]):
                    continue
                pid = int(part_ids[b, k].item())
                if pid < 0 or pid >= num_parts:
                    continue
                proto_sum[pid] += z_part[b, k]
                proto_count[pid] += 1.0

    if proto_sum is None:
        raise RuntimeError("No prototypes were accumulated. Check dataloader and dataset fields.")

    visual_proto = proto_sum / proto_count.clamp_min(1.0)[:, None]
    visual_proto = safe_normalize(visual_proto, dim=-1)

    return {
        "visual_proto": visual_proto.detach(),
        "proto_count": proto_count.detach(),
    }


def build_class_part_blocks_from_dataset(dataset, device: torch.device) -> List[Dict]:
    """
    Build one part block per object category from DinoClipJointDataset.data.

    Each block contains the object category's complete part bank:
      - part ids [K]
      - pre-text part features [K, 512]

    GW is computed separately within each block.
    """
    blocks_by_cat = {}

    if not hasattr(dataset, "data"):
        raise AttributeError("Expected dataset to have .data. This helper is for pth-backed DinoClipJointDataset.")

    for sample in dataset.data:
        category_id = int(sample["category_id"])
        if category_id in blocks_by_cat:
            continue

        part_ids = sample["part_category_id"]
        part_text = sample["part_text_feat"]

        if not torch.is_tensor(part_ids):
            part_ids = torch.tensor(part_ids, dtype=torch.long)
        if not torch.is_tensor(part_text):
            part_text = torch.tensor(part_text)

        if part_ids.numel() == 0:
            continue

        blocks_by_cat[category_id] = {
            "category_id": category_id,
            "class_name": sample.get("class_name", ""),
            "part_ids": part_ids.long().to(device),
            "part_text": part_text.float().to(device),
            "part_names": sample.get("part_class_name", []),
        }

    blocks = list(blocks_by_cat.values())
    blocks.sort(key=lambda x: int(x["category_id"]))
    return blocks


def pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    x = safe_normalize(x, dim=-1)
    sim = x @ x.T
    return (1.0 - sim).clamp_min(0.0)


def sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float = 0.05,
    max_iter: int = 50,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Entropic OT plan for a fixed cost matrix."""
    kernel = torch.exp(-cost / epsilon).clamp_min(eps)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(max_iter):
        u = a / (kernel @ v).clamp_min(eps)
        v = b / (kernel.T @ u).clamp_min(eps)

    return u[:, None] * kernel * v[None, :]


def gw_cost_matrix(C1: torch.Tensor, C2: torch.Tensor, T: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Squared-loss GW cost matrix.

    M_ij = sum_k C1_ik^2 p_k + sum_l C2_jl^2 q_l - 2 * (C1 T C2^T)_ij
    """
    const1 = (C1 ** 2) @ p
    const2 = (C2 ** 2) @ q
    return const1[:, None] + const2[None, :] - 2.0 * C1 @ T @ C2.T


@torch.no_grad()
def entropic_gw(
    C1: torch.Tensor,
    C2: torch.Tensor,
    epsilon: float = 0.05,
    max_iter: int = 20,
    sinkhorn_iter: int = 50,
) -> torch.Tensor:
    """
    Small entropic GW solver for per-object part blocks.

    C1, C2: [K, K] distance matrices.
    Returns a transport plan T: [K, K].
    """
    if C1.shape != C2.shape:
        raise ValueError(f"GW expects same-size blocks, got {C1.shape} and {C2.shape}")

    k = C1.shape[0]
    device = C1.device
    p = torch.full((k,), 1.0 / k, device=device)
    q = torch.full((k,), 1.0 / k, device=device)
    T = p[:, None] * q[None, :]

    for _ in range(max_iter):
        cost = gw_cost_matrix(C1, C2, T, p, q)
        cost = cost - cost.min()
        T = sinkhorn(p, q, cost, epsilon=epsilon, max_iter=sinkhorn_iter)

    return T.detach()


class Stage3GWLoss(nn.Module):
    """
    Stage 3 loss:
        total = lambda_obj * Lo + lambda_gw * Lgw

    Lo:
        object-level InfoNCE through existing ContrastiveLoss.
    Lgw:
        per-object-class GW plan between pre-text part graph and visual prototype graph,
        followed by soft alignment from projected text to matched visual prototypes.
    """

    def __init__(
        self,
        sim_model: nn.Module,
        visual_proto: torch.Tensor,
        class_blocks: List[Dict],
        obj_ltype: str = "infonce",
        obj_margin: float = 0.2,
        obj_max_violation: bool = True,
        lambda_obj: float = 600.0,
        lambda_gw: float = 0.25,
        gw_epsilon: float = 0.05,
        gw_max_iter: int = 20,
        sinkhorn_iter: int = 50,
        min_proto_count: int = 1,
        proto_count: torch.Tensor | None = None,
    ):
        super().__init__()
        self.sim_model = sim_model
        self.visual_proto = safe_normalize(visual_proto.float(), dim=-1)
        self.class_blocks = class_blocks
        self.lambda_obj = float(lambda_obj)
        self.lambda_gw = float(lambda_gw)
        self.gw_epsilon = float(gw_epsilon)
        self.gw_max_iter = int(gw_max_iter)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.min_proto_count = int(min_proto_count)
        self.proto_count = proto_count

        self.obj_criterion = ContrastiveLoss(
            sim_model,
            margin=obj_margin,
            max_violation=obj_max_violation,
            ltype=obj_ltype,
        )

        self.gw_blocks = []
        self._precompute_gw_plans()

    @torch.no_grad()
    def _precompute_gw_plans(self) -> None:
        self.gw_blocks = []

        for block in self.class_blocks:
            part_ids = block["part_ids"]
            part_text = block["part_text"].float()

            if part_ids.numel() < 2:
                continue

            if self.proto_count is not None:
                counts = self.proto_count[part_ids]
                if bool((counts < self.min_proto_count).any()):
                    print(
                        f"[Stage3GWLoss] skip block {block.get('class_name', '')}: "
                        f"prototype count below {self.min_proto_count}"
                    )
                    continue

            visual = self.visual_proto[part_ids]
            if not torch.isfinite(visual).all():
                print(f"[Stage3GWLoss] skip block {block.get('class_name', '')}: non-finite visual proto")
                continue

            C_text = pairwise_cosine_distance(part_text)
            C_visual = pairwise_cosine_distance(visual)

            T = entropic_gw(
                C_text,
                C_visual,
                epsilon=self.gw_epsilon,
                max_iter=self.gw_max_iter,
                sinkhorn_iter=self.sinkhorn_iter,
            )

            self.gw_blocks.append(
                {
                    "category_id": int(block["category_id"]),
                    "class_name": block.get("class_name", ""),
                    "part_ids": part_ids.detach(),
                    "part_text": part_text.detach(),
                    "visual": visual.detach(),
                    "T": T.detach(),
                }
            )

        print(f"[Stage3GWLoss] valid GW blocks: {len(self.gw_blocks)}")
        for block in self.gw_blocks:
            print(
                f"  - {block['class_name']} "
                f"category_id={block['category_id']} parts={block['part_ids'].numel()}"
            )

    def _gw_loss(self) -> torch.Tensor:
        losses = []

        for block in self.gw_blocks:
            part_text = block["part_text"]
            visual = block["visual"]
            transport = block["T"]

            projected_text = self.sim_model.project_clip_txt(part_text)
            projected_text = safe_normalize(projected_text, dim=-1)
            visual = safe_normalize(visual, dim=-1)

            cost = 1.0 - projected_text @ visual.T
            loss = (transport * cost).sum()
            losses.append(loss)

        if len(losses) == 0:
            return self.visual_proto.new_tensor(0.0)

        return torch.stack(losses).mean()

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        obj_feat = batch["obj_feat"]
        obj_text_feat = batch["obj_text_feat"]

        obj_loss = self.obj_criterion(
            obj_feat,
            obj_text_feat,
            return_similarity_mat=False,
            self_attn_maps=None,
            cls=None,
            text_input_mask=None,
            text_argmax=None,
        )

        gw_loss = self._gw_loss()
        total = self.lambda_obj * obj_loss + self.lambda_gw * gw_loss
        zero = total.new_tensor(0.0)

        return {
            "total": total,
            "obj": obj_loss.detach(),
            "gw": gw_loss.detach(),
            "inst": zero.detach(),
            "overlap": zero.detach(),
            "spear": zero.detach(),
            "anchor_hit_rate": zero.detach(),
            "anchor_total_valid_parts": zero.detach(),
            "anchor_total_hits": zero.detach(),
        }
