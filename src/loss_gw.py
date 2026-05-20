from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.loss import ContrastiveLoss
from src.loss_joint import JointObjPartLoss


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    x = safe_normalize(x.float(), dim=-1)
    return (1.0 - x @ x.T).clamp_min(0.0)


def pairwise_cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    x = safe_normalize(x.float(), dim=-1)
    return x @ x.T


def upper_tri_vector(mat: torch.Tensor) -> torch.Tensor:
    k = mat.shape[0]
    if k < 2:
        return mat.new_empty((0,))
    idx = torch.triu_indices(k, k, offset=1, device=mat.device)
    return mat[idx[0], idx[1]]


def rankdata_torch(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.float()
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def spearman_corr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.flatten().float()
    y = y.flatten().float()
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.numel() < 2 or y.numel() < 2:
        return x.new_tensor(float("nan"))

    rx = rankdata_torch(x)
    ry = rankdata_torch(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = rx.norm() * ry.norm()
    if denom <= eps:
        return x.new_tensor(float("nan"))
    return (rx * ry).sum() / denom


def structure_spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sim_a = pairwise_cosine_similarity(a)
    sim_b = pairwise_cosine_similarity(b)
    return spearman_corr_torch(upper_tri_vector(sim_a), upper_tri_vector(sim_b))


def structure_retrieval_metric(feat_1: torch.Tensor, feat_2: torch.Tensor) -> torch.Tensor:
    assert feat_1.shape[0] == feat_2.shape[0]
    n = feat_1.shape[0]
    if n <= 1:
        return feat_1.new_tensor(float("nan"))

    feat_1_ = safe_normalize(feat_1.float(), dim=-1)
    feat_2_ = safe_normalize(feat_2.float(), dim=-1)

    sim_1 = feat_1_ @ feat_1_.T
    sim_2 = feat_2_ @ feat_2_.T

    eye = torch.eye(n, dtype=torch.bool, device=feat_1.device)
    sim_1 = sim_1[~eye].view(n, -1)
    sim_2 = sim_2[~eye].view(n, -1)

    sim_1 = sim_1 - sim_1.mean(dim=-1, keepdim=True)
    sim_2 = sim_2 - sim_2.mean(dim=-1, keepdim=True)

    sim_1_norm = safe_normalize(sim_1, dim=-1)
    sim_2_norm = safe_normalize(sim_2, dim=-1)

    sim_1_2 = sim_1_norm @ sim_2_norm.T
    if torch.isnan(sim_1_2).any():
        return feat_1.new_tensor(float("nan"))

    idx = sim_1_2.argmax(dim=0)
    target = torch.arange(n, device=feat_1.device)
    return (idx == target).float().mean()


# -----------------------------------------------------------------------------
# Stage2 visual prototype construction
# -----------------------------------------------------------------------------


@torch.no_grad()
def extract_z_part_from_batch(
    model: nn.Module,
    batch: Dict,
    patch_temperature: float = 0.07,
    em_iters: int = 1,
    anchor_helper: Optional[JointObjPartLoss] = None,
    return_anchor_tokens: bool = False,
):
    """
    Reuse the Stage2 anchor / pseudo-prototype routine.

    If return_anchor_tokens=True, returns the selected single anchor patch token
    for each valid part. Otherwise returns EM-pooled pseudo part features.
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
        anchor_helper.eval()

    part_proj = model.project_clip_txt(part_text_feat)
    part_proj = anchor_helper._safe_normalize(part_proj, dim=-1)
    patch_tokens = anchor_helper._safe_normalize(patch_tokens, dim=-1)

    abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / float(patch_temperature)
    abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

    dummy_part_gt_mask_patch = torch.zeros(
        part_valid_mask.shape[0],
        part_valid_mask.shape[1],
        patch_tokens.shape[1],
        dtype=torch.bool,
        device=device,
    )

    if return_anchor_tokens:
        z_part, _, _, anchor_tokens, anchor_valid = anchor_helper._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_valid_mask,
            part_gt_mask_patch=dummy_part_gt_mask_patch,
            num_iters=em_iters,
            return_anchor_tokens=True,
        )
        return z_part, anchor_tokens, anchor_valid

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
    visual_source: str = "anchor",
) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    model.eval()

    visual_source = str(visual_source).lower()
    if visual_source not in {"anchor", "zpart"}:
        raise ValueError(f"visual_source must be 'anchor' or 'zpart', got {visual_source}")

    proto_sum = None
    proto_count = torch.zeros(num_parts, device=device)

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

    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Build Stage2 visual prototypes ({visual_source})"):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        if visual_source == "anchor":
            _, anchor_tokens, anchor_valid = extract_z_part_from_batch(
                model=model,
                batch=batch,
                patch_temperature=patch_temperature,
                em_iters=em_iters,
                anchor_helper=anchor_helper,
                return_anchor_tokens=True,
            )
            feat = anchor_tokens
        else:
            feat = extract_z_part_from_batch(
                model=model,
                batch=batch,
                patch_temperature=patch_temperature,
                em_iters=em_iters,
                anchor_helper=anchor_helper,
            )
            anchor_valid = None

        part_ids = batch["part_category_id"].long()
        part_valid = batch["part_valid_mask"].bool()
        if anchor_valid is not None:
            part_valid = part_valid & anchor_valid.bool()

        if proto_sum is None:
            proto_sum = torch.zeros(num_parts, feat.shape[-1], device=device)

        bsz, max_k = part_ids.shape
        for b in range(bsz):
            for k in range(max_k):
                if not bool(part_valid[b, k]):
                    continue
                pid = int(part_ids[b, k].item())
                if 0 <= pid < num_parts:
                    proto_sum[pid] += feat[b, k]
                    proto_count[pid] += 1.0

    if proto_sum is None:
        raise RuntimeError("No prototypes were accumulated. Check dataloader and dataset fields.")

    visual_proto = proto_sum / proto_count.clamp_min(1.0)[:, None]
    visual_proto = safe_normalize(visual_proto, dim=-1)

    return {
        "visual_proto": visual_proto.detach(),
        "proto_count": proto_count.detach(),
        "visual_source": visual_source,
    }


@torch.no_grad()
def build_class_part_blocks_from_dataset(dataset, device: torch.device) -> List[Dict]:
    """
    Build one object-level part block per category.

    Each block contains:
      - category_id
      - class_name
      - part_ids: [K]
      - part_text: [K, text_dim]
    """
    if not hasattr(dataset, "data"):
        raise AttributeError("Expected dataset to have .data. This helper is for pth-backed DinoClipJointDataset.")

    data = dataset.data.values() if isinstance(dataset.data, dict) else dataset.data
    blocks_by_cat = {}

    for sample in tqdm(data, total=len(dataset.data), desc="Build object part blocks"):
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


# -----------------------------------------------------------------------------
# Hard bijective GW matching
# -----------------------------------------------------------------------------


@torch.no_grad()
def make_perm_transport(row_to_col: torch.Tensor, k: int) -> torch.Tensor:
    transport = torch.zeros(k, k, device=row_to_col.device, dtype=torch.float32)
    transport[torch.arange(k, device=row_to_col.device), row_to_col] = 1.0 / float(k)
    return transport


def hard_gw_struct_objective(C1: torch.Tensor, C2: torch.Tensor, row_to_col: torch.Tensor) -> torch.Tensor:
    """
    Hard bijective GW objective for a fixed source->target permutation.

        row_to_col[i] = j
        objective = mean_{i,k} (C1[i,k] - C2[j_i,j_k])^2
    """
    C2_perm = C2[row_to_col][:, row_to_col]
    return F.mse_loss(C1, C2_perm)


def gw_linearized_cost(C1: torch.Tensor, C2: torch.Tensor, transport: torch.Tensor) -> torch.Tensor:
    """
    Squared-loss GW linearization for a current hard transport.
    """
    k = C1.shape[0]
    p = torch.full((k,), 1.0 / float(k), device=C1.device, dtype=C1.dtype)
    q = torch.full((k,), 1.0 / float(k), device=C1.device, dtype=C1.dtype)

    const1 = (C1 ** 2) @ p
    const2 = (C2 ** 2) @ q
    return const1[:, None] + const2[None, :] - 2.0 * C1 @ transport @ C2.T


@torch.no_grad()
def pair_swap_refine(
    C1: torch.Tensor,
    C2: torch.Tensor,
    perm: torch.Tensor,
    max_passes: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    best_perm = perm.detach().clone()
    best_obj = hard_gw_struct_objective(C1, C2, best_perm).detach()
    k = int(best_perm.numel())

    for _ in range(max(1, int(max_passes))):
        improved = False
        for a in range(k):
            for b in range(a + 1, k):
                cand = best_perm.clone()
                cand[a], cand[b] = cand[b].clone(), cand[a].clone()
                obj = hard_gw_struct_objective(C1, C2, cand).detach()
                if obj.item() + 1e-12 < best_obj.item():
                    best_perm = cand
                    best_obj = obj
                    improved = True
        if not improved:
            break

    return best_perm, best_obj


@torch.no_grad()
def hard_bijective_gw_match(
    C1: torch.Tensor,
    C2: torch.Tensor,
    num_iters: int = 20,
    num_restarts: int = 50,
    include_identity: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hard bijective GW solver.

    It finds a source->target permutation P such that:
        C1 ~= C2[P][:, P]

    It uses iterative GW linearization + Hungarian assignment + light pair-swap
    refinement. There is no Sinkhorn / soft transport here.
    """
    if C1.shape != C2.shape:
        raise ValueError(f"hard_bijective_gw_match expects same-size blocks, got {C1.shape} and {C2.shape}")

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:
        raise ImportError("hard_bijective_gw_match requires scipy.optimize.linear_sum_assignment.") from exc

    C1 = C1.detach().float()
    C2 = C2.detach().float()
    k = int(C1.shape[0])
    device = C1.device

    best_perm = None
    best_obj = None

    for restart_id in range(max(1, int(num_restarts))):
        if restart_id == 0 and include_identity:
            perm = torch.arange(k, device=device)
        else:
            perm = torch.randperm(k, device=device)

        transport = make_perm_transport(perm, k).to(device=device, dtype=C1.dtype)

        for _ in range(max(1, int(num_iters))):
            cost = gw_linearized_cost(C1, C2, transport)
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

            new_perm = torch.empty(k, dtype=torch.long, device=device)
            new_perm[torch.as_tensor(row_ind, dtype=torch.long, device=device)] = torch.as_tensor(
                col_ind, dtype=torch.long, device=device
            )

            new_transport = make_perm_transport(new_perm, k).to(device=device, dtype=C1.dtype)
            if torch.equal(new_perm, perm):
                perm = new_perm
                transport = new_transport
                break

            perm = new_perm
            transport = new_transport

        perm, obj = pair_swap_refine(C1, C2, perm, max_passes=max(1, int(num_iters) // 2))

        if best_obj is None or obj.item() < best_obj.item():
            best_perm = perm.detach().clone()
            best_obj = obj.detach()

    return best_perm, best_obj


# -----------------------------------------------------------------------------
# Stage3 loss
# -----------------------------------------------------------------------------


class Stage3GWLoss(nn.Module):
    """
    Stage3 fixed-Z0 hard-GW refinement.

    Initialization:
      1. Project each object's part text with the Stage2-initialized projector:
             Z0 = f_stage2(T)
      2. Build D(Z0) and D(V_anchor).
      3. Use hard bijective GW to find a fixed permutation:
             P = GW(D(Z0), D(V_anchor))

    Training:
      1. Current projected text:
             Zt = f_current(T)
      2. Matched feature alignment:
             L_gw = mean_i [1 - cos(Zt_i, V_anchor[P(i)])]
      3. Structure preservation:
             L_struct = MSE(D(Zt), D(Z0))
    """

    def __init__(
        self,
        sim_model: nn.Module,
        visual_proto: torch.Tensor,
        class_blocks: List[Dict],
        obj_ltype: str = "infonce",
        obj_margin: float = 0.2,
        obj_max_violation: bool = True,
        lambda_obj: float = 0.0,
        lambda_gw: float = 1.0,
        lambda_struct: float = 50.0,
        gw_max_iter: int = 20,
        gw_restarts: int = 50,
        min_proto_count: int = 1,
        proto_count: Optional[torch.Tensor] = None,
        patch_temperature: float = 0.07,
        em_iters: int = 1,
    ):
        super().__init__()

        self.sim_model = sim_model
        self.register_buffer("visual_proto", safe_normalize(visual_proto.float(), dim=-1).detach())

        self.class_blocks = class_blocks
        self.lambda_obj = float(lambda_obj)
        self.lambda_gw = float(lambda_gw)
        self.lambda_struct = float(lambda_struct)
        self.gw_max_iter = int(gw_max_iter)
        self.gw_restarts = int(gw_restarts)
        self.min_proto_count = int(min_proto_count)
        self.proto_count = proto_count
        self.patch_temperature = float(patch_temperature)
        self.em_iters = int(em_iters)

        self.obj_criterion = ContrastiveLoss(
            sim_model,
            margin=obj_margin,
            max_violation=obj_max_violation,
            ltype=obj_ltype,
        )

        self.anchor_helper = JointObjPartLoss(
            sim_model=sim_model,
            obj_ltype=obj_ltype,
            lambda_obj=0.0,
            lambda_inst=0.0,
            lambda_overlap=0.0,
            lambda_spear=0.0,
            patch_temperature=self.patch_temperature,
            em_iters=self.em_iters,
        )

        self.gw_blocks: List[Dict] = []
        self._prepare_gw_blocks()

    @torch.no_grad()
    def _prepare_gw_blocks(self) -> None:
        self.gw_blocks = []

        for block in self.class_blocks:
            part_ids = block["part_ids"].long()
            part_text = block["part_text"].float()

            if part_ids.numel() < 2:
                continue

            if self.proto_count is not None:
                counts = self.proto_count[part_ids]
                if bool((counts < self.min_proto_count).any()):
                    print(
                        f"[Stage3GWLoss] skip {block.get('class_name', '')}: "
                        f"prototype count below {self.min_proto_count}"
                    )
                    continue

            visual = safe_normalize(self.visual_proto[part_ids].float(), dim=-1)
            if not torch.isfinite(visual).all():
                print(f"[Stage3GWLoss] skip {block.get('class_name', '')}: non-finite visual proto")
                continue

            z0 = self.sim_model.project_clip_txt(part_text)
            z0 = safe_normalize(z0.float(), dim=-1)
            if not torch.isfinite(z0).all():
                print(f"[Stage3GWLoss] skip {block.get('class_name', '')}: non-finite Z0")
                continue

            C_z0 = pairwise_cosine_distance(z0).detach()
            C_visual = pairwise_cosine_distance(visual).detach()

            perm, gw_obj = hard_bijective_gw_match(
                C_z0,
                C_visual,
                num_iters=self.gw_max_iter,
                num_restarts=self.gw_restarts,
                include_identity=True,
            )

            self.gw_blocks.append(
                {
                    "category_id": int(block["category_id"]),
                    "class_name": block.get("class_name", ""),
                    "part_ids": part_ids.detach(),
                    "part_text": part_text.detach(),
                    "visual": visual.detach(),
                    "z0": z0.detach(),
                    "C_z0": C_z0.detach(),
                    "gw_perm": perm.detach().long(),
                    "gw_init_obj": gw_obj.detach(),
                }
            )

        print(f"[Stage3GWLoss] valid GW blocks: {len(self.gw_blocks)}")
        for block in self.gw_blocks:
            print(
                f"  - {block['class_name']} "
                f"category_id={block['category_id']} "
                f"parts={block['part_ids'].numel()} "
                f"gw_obj={float(block['gw_init_obj'].detach().cpu().item()):.6f} "
                f"perm={block['gw_perm'].detach().cpu().tolist()}"
            )

    def _gw_alignment_loss(self) -> torch.Tensor:
        losses = []

        for block in self.gw_blocks:
            part_text = block["part_text"]
            visual = block["visual"]
            perm = block["gw_perm"]

            zt = self.sim_model.project_clip_txt(part_text)
            zt = safe_normalize(zt.float(), dim=-1)

            target_visual = visual[perm].detach()
            losses.append((1.0 - (zt * target_visual).sum(dim=-1)).mean())

        if len(losses) == 0:
            return self.visual_proto.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _structure_preservation_loss(self) -> torch.Tensor:
        losses = []

        for block in self.gw_blocks:
            part_text = block["part_text"]
            if part_text.shape[0] < 2:
                continue

            zt = self.sim_model.project_clip_txt(part_text)
            zt = safe_normalize(zt.float(), dim=-1)

            C_zt = pairwise_cosine_distance(zt)
            C_z0 = block["C_z0"].detach()
            losses.append(F.mse_loss(C_zt, C_z0))

        if len(losses) == 0:
            return self.visual_proto.new_tensor(0.0)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def _structure_audit(self) -> Dict[str, torch.Tensor]:
        values: Dict[str, List[torch.Tensor]] = {
            "audit_spear_z0_vs_visual": [],
            "audit_spear_zt_vs_visual": [],
            "audit_spear_zt_vs_z0": [],
            "audit_spear_z0_vs_visual_perm": [],
            "audit_spear_zt_vs_visual_perm": [],
            "audit_strret_z0_vs_visual": [],
            "audit_strret_zt_vs_visual": [],
            "audit_strret_zt_vs_z0": [],
            "audit_strret_z0_vs_visual_perm": [],
            "audit_strret_zt_vs_visual_perm": [],
            "audit_gw_obj_z0_fixed_perm": [],
            "audit_gw_obj_zt_fixed_perm": [],
        }

        for block in self.gw_blocks:
            part_text = block["part_text"]
            visual = safe_normalize(block["visual"], dim=-1)
            z0 = safe_normalize(block["z0"], dim=-1)
            perm = block["gw_perm"]
            visual_perm = visual[perm]

            if part_text.shape[0] < 2:
                continue

            zt = self.sim_model.project_clip_txt(part_text)
            zt = safe_normalize(zt.float(), dim=-1)

            values["audit_spear_z0_vs_visual"].append(structure_spearman(z0, visual))
            values["audit_spear_zt_vs_visual"].append(structure_spearman(zt, visual))
            values["audit_spear_zt_vs_z0"].append(structure_spearman(zt, z0))
            values["audit_spear_z0_vs_visual_perm"].append(structure_spearman(z0, visual_perm))
            values["audit_spear_zt_vs_visual_perm"].append(structure_spearman(zt, visual_perm))

            values["audit_strret_z0_vs_visual"].append(structure_retrieval_metric(z0, visual))
            values["audit_strret_zt_vs_visual"].append(structure_retrieval_metric(zt, visual))
            values["audit_strret_zt_vs_z0"].append(structure_retrieval_metric(zt, z0))
            values["audit_strret_z0_vs_visual_perm"].append(structure_retrieval_metric(z0, visual_perm))
            values["audit_strret_zt_vs_visual_perm"].append(structure_retrieval_metric(zt, visual_perm))

            C_visual = pairwise_cosine_distance(visual)
            C_zt = pairwise_cosine_distance(zt)
            values["audit_gw_obj_z0_fixed_perm"].append(block["gw_init_obj"].to(C_zt.device).float())
            values["audit_gw_obj_zt_fixed_perm"].append(
                hard_gw_struct_objective(C_zt, C_visual, perm).detach()
            )

        out: Dict[str, torch.Tensor] = {}
        device = self.visual_proto.device

        for key, vals in values.items():
            if len(vals) == 0:
                out[key] = torch.tensor(float("nan"), device=device)
                continue

            stacked = torch.stack([v.to(device).float() for v in vals])
            finite = torch.isfinite(stacked)
            out[key] = stacked[finite].mean() if finite.any() else torch.tensor(float("nan"), device=device)

        # Backward-compatible aliases for existing train_util_gw progress/history code.
        out["audit_spear_pre_text_vs_visual"] = out["audit_spear_z0_vs_visual"]
        out["audit_spear_post_text_vs_visual"] = out["audit_spear_zt_vs_visual"]
        out["audit_strret_pre_text_vs_visual"] = out["audit_strret_z0_vs_visual"]
        out["audit_strret_post_text_vs_visual"] = out["audit_strret_zt_vs_visual"]

        return out

    @torch.no_grad()
    def _anchor_audit(self, batch: Dict) -> Dict[str, torch.Tensor]:
        required = ["part_text_feat", "patch_tokens", "obj_mask_patch", "part_valid_mask", "part_gt_mask_patch"]
        if any(k not in batch for k in required):
            z = self.visual_proto.new_tensor(0.0)
            return {
                "anchor_hit_rate": z,
                "anchor_hit_rate_post": z,
                "anchor_total_valid_parts": z,
                "anchor_total_valid_parts_post": z,
                "anchor_total_hits": z,
                "anchor_total_hits_post": z,
            }

        device = self.visual_proto.device
        part_text_feat = batch["part_text_feat"].to(device).float()
        patch_tokens = batch["patch_tokens"].to(device).float()
        obj_mask_patch = batch["obj_mask_patch"].to(device).bool()
        part_valid_mask = batch["part_valid_mask"].to(device).bool()
        part_gt_mask_patch = batch["part_gt_mask_patch"].to(device).bool()

        part_proj = self.sim_model.project_clip_txt(part_text_feat)
        part_proj = self.anchor_helper._safe_normalize(part_proj, dim=-1)
        patch_tokens = self.anchor_helper._safe_normalize(patch_tokens, dim=-1)

        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / float(self.patch_temperature)
        abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        _, _, metrics = self.anchor_helper._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_valid_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=self.em_iters,
        )
        return {
            "anchor_hit_rate": metrics["anchor_hit_rate"],
            "anchor_hit_rate_post": metrics["anchor_hit_rate"],
            "anchor_total_valid_parts": metrics["anchor_total_valid_parts"],
            "anchor_total_valid_parts_post": metrics["anchor_total_valid_parts"],
            "anchor_total_hits": metrics["anchor_total_hits"],
            "anchor_total_hits_post": metrics["anchor_total_hits"],
        }

    def forward(
        self,
        batch: Optional[Dict] = None,
        do_anchor_audit: bool = False,
        do_structure_audit: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = self.visual_proto.device
        zero = torch.tensor(0.0, device=device)

        if batch is not None and self.lambda_obj > 0:
            obj_loss = self.obj_criterion(
                batch["obj_feat"],
                batch["obj_text_feat"],
                return_similarity_mat=False,
                self_attn_maps=None,
                cls=None,
                text_input_mask=None,
                text_argmax=None,
            )
        else:
            obj_loss = zero

        gw_loss = self._gw_alignment_loss() if self.lambda_gw > 0 else zero
        struct_loss = self._structure_preservation_loss() if self.lambda_struct > 0 else zero

        total = (
            self.lambda_obj * obj_loss
            + self.lambda_gw * gw_loss
            + self.lambda_struct * struct_loss
        )

        out = {
            "total": total,
            "obj": obj_loss.detach(),
            "gw": gw_loss.detach(),
            "struct": struct_loss.detach(),
            "inst": zero.detach(),
            "overlap": zero.detach(),
            "spear": zero.detach(),
        }

        if batch is not None and do_anchor_audit:
            out.update(self._anchor_audit(batch))

        if do_structure_audit:
            out.update(self._structure_audit())

        return out
