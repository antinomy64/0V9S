from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn
from src.loss_gw import (
    Stage3GWLoss,
    build_class_part_blocks_from_dataset,
    safe_normalize,
    pairwise_cosine_distance,
    hard_bijective_gw_match,
    hard_gw_struct_objective,
)


# Fixed dataset fields.
OBJ_FEATURE_NAME = "avg_self_attn_out"
PART_FEATURE_NAME = "cropaug_patch_tokens"
OBJ_TEXT_NAME = "ann_feats"
PART_TEXT_NAME = "part_ann_feats"
RESIZE_DIM = 448
CROP_DIM = 448
PATCH_SIZE = 14

# Skip k=2, because 2-point structures are permutation-ambiguous.
MIN_BLOCK_PARTS = 3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DebugProjector(nn.Module):
    """
    Simple affine projector for fakeT -> V recovery.

    If in_dim == out_dim, identity init means:
        Z0 = fakeT

    Stage3GWLoss then computes fixed P = GW(D(Z0), D(V)).
    Training updates this projector so Zt approaches V[P] while preserving D(Z0).
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)
        with torch.no_grad():
            self.proj.weight.zero_()
            m = min(in_dim, out_dim)
            self.proj.weight[:m, :m] = torch.eye(m)
            if self.proj.bias is not None:
                self.proj.bias.zero_()

    def project_clip_txt(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


def load_cfg(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(dataset_path: str, cfg: dict):
    min_obj_area_ratio = float(cfg.get("dataset", {}).get("min_obj_area_ratio", 0.0))
    return DinoClipJointDataset(
        dataset_path,
        obj_feature_name=OBJ_FEATURE_NAME,
        part_feature_name=PART_FEATURE_NAME,
        obj_text_name=OBJ_TEXT_NAME,
        part_text_name=PART_TEXT_NAME,
        resize_dim=RESIZE_DIM,
        crop_dim=CROP_DIM,
        patch_size=PATCH_SIZE,
        with_background=False,
        is_wds=".tar" in dataset_path,
        path_prefix=None,
        min_obj_area_ratio=min_obj_area_ratio,
    )


@torch.no_grad()
def build_maskavg_visual_prototypes(
    dataloader: DataLoader,
    num_parts: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build GT-mask averaged part prototypes.

    This is ONLY for the synthetic recovery test:
        V = part GT maskavg visual prototype

    For every valid part instance:
        local_proto = mean(raw cropaug patch tokens inside its GT part mask)

    Then average local_proto by global part id.
    Final global prototypes are L2-normalized.
    """
    proto_sum = None
    proto_count = torch.zeros(num_parts, device=device)

    for batch in tqdm(dataloader, total=len(dataloader), desc="Build GT-maskavg part prototypes"):
        patch_tokens = batch["patch_tokens"].to(device, dtype=torch.float32)       # [B, N, D]
        part_masks = batch["part_gt_mask_patch"].to(device).bool()                 # [B, K, N]
        part_ids = batch["part_category_id"].to(device).long()                     # [B, K]
        part_valid = batch["part_valid_mask"].to(device).bool()                    # [B, K]

        if proto_sum is None:
            proto_sum = torch.zeros(num_parts, patch_tokens.shape[-1], device=device)

        B, K, _ = part_masks.shape
        for b in range(B):
            for k in range(K):
                if not bool(part_valid[b, k]):
                    continue

                pid = int(part_ids[b, k].item())
                if pid < 0 or pid >= num_parts:
                    continue

                mask = part_masks[b, k]
                if int(mask.sum().item()) <= 0:
                    continue

                local_proto = patch_tokens[b, mask].mean(dim=0)
                proto_sum[pid] += local_proto
                proto_count[pid] += 1.0

    if proto_sum is None:
        raise RuntimeError("No GT-maskavg prototypes accumulated. Check dataset fields.")

    visual_proto = proto_sum / proto_count.clamp_min(1.0)[:, None]
    visual_proto = safe_normalize(visual_proto, dim=-1)

    return {
        "visual_proto": visual_proto.detach(),
        "proto_count": proto_count.detach(),
        "visual_source": "maskavg_gt_part",
    }


@torch.no_grad()
def random_orthogonal_matrix(dim: int, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    A = torch.randn(dim, dim, generator=gen, device=device)
    Q, R = torch.linalg.qr(A)

    # Stabilize QR sign.
    diag = torch.diag(R)
    sign = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
    Q = Q * sign.unsqueeze(0)
    return Q


@torch.no_grad()
def random_semi_orthogonal_matrix(in_dim: int, out_dim: int, device: torch.device, seed: int) -> torch.Tensor:
    """
    Return A with shape [in_dim, out_dim].

    For a clean "rotation-only" test, use out_dim == in_dim.
    If out_dim < in_dim, fakeT compresses V and exact recovery is not guaranteed.
    """
    if in_dim == out_dim:
        return random_orthogonal_matrix(in_dim, device=device, seed=seed)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    M = torch.randn(in_dim, out_dim, generator=gen, device=device)

    if out_dim <= in_dim:
        Q, _ = torch.linalg.qr(M, mode="reduced")
        return Q[:, :out_dim]

    # Expansion case: orthonormal rows via QR on transpose.
    Q, _ = torch.linalg.qr(M.T, mode="reduced")
    return Q.T


@torch.no_grad()
def make_fake_blocks(
    class_blocks: List[Dict[str, Any]],
    visual_proto: torch.Tensor,
    proto_count: torch.Tensor,
    fake_text_dim: int,
    min_proto_count: int,
    seed: int,
    shuffle: bool = True,
    affine_scale: float = 0.0,
    affine_bias: float = 0.0,
):
    """
    Convert GT-maskavg V into synthetic fake text features.

    For each object block:
        V_local = visual_proto[part_ids]

    Global synthetic transform:
        fake_global = normalize(V @ A * scale + bias)

    If shuffle=True:
        fakeT_i comes from V_{perm[i]}, so the known target is target[i]=perm[i].
    If shuffle=False:
        fakeT_i comes from V_i, so the known target is identity.
    """
    device = visual_proto.device
    visual_dim = int(visual_proto.shape[-1])

    A = random_semi_orthogonal_matrix(
        in_dim=visual_dim,
        out_dim=fake_text_dim,
        device=device,
        seed=seed + 991,
    )

    fake_global = visual_proto.float() @ A

    if affine_scale > 0:
        scale = 1.0 + float(affine_scale) * torch.randn(fake_text_dim, device=device)
        fake_global = fake_global * scale[None, :]

    if affine_bias > 0:
        bias = float(affine_bias) * torch.randn(fake_text_dim, device=device)
        fake_global = fake_global + bias[None, :]

    fake_global = safe_normalize(fake_global, dim=-1)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed + 17)

    fake_blocks = []
    true_match = {}

    for block in class_blocks:
        part_ids = block["part_ids"].to(device).long()
        k = int(part_ids.numel())

        if k < MIN_BLOCK_PARTS:
            continue

        if (proto_count[part_ids] < min_proto_count).any():
            continue

        fake_local = fake_global[part_ids]  # [K, fake_text_dim]
        if shuffle:
            perm = torch.randperm(k, generator=rng, device=device)
        else:
            perm = torch.arange(k, device=device)

        new_block = dict(block)
        new_block["part_ids"] = part_ids.detach()
        new_block["part_text"] = fake_local[perm].detach()
        new_block["row_part_ids"] = part_ids[perm].detach()
        fake_blocks.append(new_block)

        # row i in fakeT corresponds to visual local perm[i]
        true_match[int(block["category_id"])] = perm.detach()

    return fake_blocks, true_match


@torch.no_grad()
def evaluate(criterion, projector, true_match, gw_max_iter: int, gw_restarts: int):
    total = 0
    fixed_hit = 0
    dynamic_hit = 0

    fixed_gw_vals, dynamic_gw_vals = [], []
    pre_v_vals, post_v_vals, prepost_vals = [], [], []
    pair_cos_fixed_vals, pair_cos_dynamic_vals, pair_cos_true_vals = [], [], []
    retrieval_true_vals = []

    for block in criterion.gw_blocks:
        cat_id = int(block["category_id"])
        if cat_id not in true_match:
            continue

        T = block["part_text"].float()
        V = safe_normalize(block["visual"].float(), dim=-1)
        target = true_match[cat_id].long().to(V.device)
        fixed_perm = block["gw_perm"].long().to(V.device)

        k = int(T.shape[0])
        Z = safe_normalize(projector.project_clip_txt(T), dim=-1)

        C_t = pairwise_cosine_distance(T)
        C_z = pairwise_cosine_distance(Z)
        C_v = pairwise_cosine_distance(V)

        dyn_perm, _ = hard_bijective_gw_match(
            C_z,
            C_v,
            num_iters=gw_max_iter,
            num_restarts=gw_restarts,
            include_identity=True,
        )
        dyn_perm = dyn_perm.to(V.device).long()

        sim = Z @ V.T
        retrieval_pred = sim.argmax(dim=1)

        pair_cos_fixed_vals.append((Z * V[fixed_perm]).sum(dim=-1).mean().detach())
        pair_cos_dynamic_vals.append((Z * V[dyn_perm]).sum(dim=-1).mean().detach())
        pair_cos_true_vals.append((Z * V[target]).sum(dim=-1).mean().detach())
        retrieval_true_vals.append((retrieval_pred == target).float().mean().detach())

        fixed_hit += int((fixed_perm == target).sum().item())
        dynamic_hit += int((dyn_perm == target).sum().item())
        total += k

        C_v_true = C_v[target][:, target]
        fixed_gw_vals.append(hard_gw_struct_objective(C_z, C_v, fixed_perm).detach())
        dynamic_gw_vals.append(hard_gw_struct_objective(C_z, C_v, dyn_perm).detach())
        pre_v_vals.append(F.mse_loss(C_t, C_v_true).detach())
        post_v_vals.append(F.mse_loss(C_z, C_v_true).detach())
        prepost_vals.append(F.mse_loss(C_z, C_t).detach())

    return {
        "Hacc_fixed": fixed_hit / max(total, 1),
        "Hacc_dynamic": dynamic_hit / max(total, 1),
        "gw_struct_fixed": torch.stack(fixed_gw_vals).mean().item(),
        "gw_struct_dynamic": torch.stack(dynamic_gw_vals).mean().item(),
        "preV": torch.stack(pre_v_vals).mean().item(),
        "postV": torch.stack(post_v_vals).mean().item(),
        "prepost": torch.stack(prepost_vals).mean().item(),
        "paircos_fixed": torch.stack(pair_cos_fixed_vals).mean().item(),
        "paircos_dynamic": torch.stack(pair_cos_dynamic_vals).mean().item(),
        "paircos_true": torch.stack(pair_cos_true_vals).mean().item(),
        "retrieval_true": torch.stack(retrieval_true_vals).mean().item(),
        "num_parts": total,
    }


@torch.no_grad()
def fit_block_pca_basis(V, Z, out_dim=2):
    X = torch.cat([V.detach().float().cpu(), Z.detach().float().cpu()], dim=0)
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    basis = Vh[:out_dim].T.contiguous()
    return mean, basis


@torch.no_grad()
def project_2d(x, mean, basis):
    x = x.detach().float().cpu()
    return (x - mean) @ basis


@torch.no_grad()
def greedy_cosine_bijection(Z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Row-order greedy one-to-one cosine matching for visualization only."""
    sim = (Z @ V.T).detach().float().cpu()
    k = sim.shape[0]

    pred = torch.empty(k, dtype=torch.long)
    used_v = torch.zeros(k, dtype=torch.bool)

    for i in range(k):
        order = torch.argsort(sim[i], descending=True)
        chosen = None
        for j in order.tolist():
            if not used_v[j]:
                chosen = j
                break
        if chosen is None:
            remaining = torch.where(~used_v)[0]
            chosen = int(remaining[0].item())
        pred[i] = chosen
        used_v[chosen] = True

    return pred


def set_square_limits(ax, V2, Z2, pad_ratio=0.08):
    pts = np.concatenate([V2, Z2], axis=0)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min)
    half = max(half * (1.0 + pad_ratio), 1e-6)

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


@torch.no_grad()
def save_all_block_triplet_viz(
    criterion,
    projector,
    true_match,
    step,
    save_root,
    gw_max_iter,
    gw_restarts,
    pca_cache=None,
):
    """
    Save one 3-panel visualization per object block.

    Plot design:
      - Points show ONLY numeric ids: 0, 1, 2, ...
      - No V/T prefix on the points.
      - No part names inside coordinate plots.
      - Bottom has exactly two mapping tables:
          1) V point id -> part name
          2) T point id -> original/source part name
      - No OK/BAD judgment.
      - No per-row matching statistics table.
    """
    save_root = Path(save_root)
    step_dir = save_root / f"step_{step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    def _short_name(name: str, max_len: int = 46) -> str:
        name = str(name) if name is not None else ""
        name = name.replace("\n", " ").strip()
        if len(name) <= max_len:
            return name
        return name[: max_len - 3] + "..."

    def _get_part_names(block, k: int):
        names = block.get("part_names", None)
        if isinstance(names, (list, tuple)) and len(names) >= k:
            return [str(x) for x in names[:k]]
        return [""] * k

    for block_idx, block in enumerate(criterion.gw_blocks):
        cat_id = int(block["category_id"])
        if cat_id not in true_match:
            continue

        V = safe_normalize(block["visual"].float(), dim=-1)
        T = block["part_text"].float()
        Z = safe_normalize(projector.project_clip_txt(T), dim=-1)

        v_part_ids = block["part_ids"].detach().cpu().long().tolist()
        part_names = _get_part_names(block, len(v_part_ids))

        fixed_perm = block["gw_perm"].detach().cpu().long()
        target = true_match[cat_id].detach().cpu().long()

        C_z = pairwise_cosine_distance(Z)
        C_v = pairwise_cosine_distance(V)
        dyn_perm, _ = hard_bijective_gw_match(
            C_z,
            C_v,
            num_iters=gw_max_iter,
            num_restarts=gw_restarts,
            include_identity=True,
        )
        dyn_perm = dyn_perm.detach().cpu().long()

        cos_bij = greedy_cosine_bijection(Z, V).detach().cpu().long()

        cache_key = int(cat_id)
        if pca_cache is not None:
            if cache_key not in pca_cache:
                mean, basis = fit_block_pca_basis(V, Z)
                pca_cache[cache_key] = (mean, basis)
            else:
                mean, basis = pca_cache[cache_key]
        else:
            mean, basis = fit_block_pca_basis(V, Z)

        V2 = project_2d(V, mean, basis).numpy()
        Z2 = project_2d(Z, mean, basis).numpy()

        # Top: 3 plots. Bottom: two id->name mapping tables.
        fig = plt.figure(figsize=(18, 8.9))
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[3.25, 1.45],
            hspace=0.30,
            wspace=0.18,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        table_v_ax = fig.add_subplot(gs[1, 0])
        table_t_ax = fig.add_subplot(gs[1, 1:])
        table_v_ax.axis("off")
        table_t_ax.axis("off")

        label_box = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0)

        # ------------------------------------------------------------------
        # Panel 1: V points, numeric id only.
        # ------------------------------------------------------------------
        ax1.scatter(V2[:, 0], V2[:, 1], s=80, c="blue")
        for i in range(len(V2)):
            ax1.text(
                V2[i, 0],
                V2[i, 1],
                f"{i}",
                fontsize=10,
                color="black",
                fontweight="bold",
                ha="center",
                va="center",
                bbox=label_box,
            )
        ax1.set_title("V: GT maskavg part prototypes")
        set_square_limits(ax1, V2, Z2)

        # ------------------------------------------------------------------
        # Panel 2: T points, numeric id only.
        # T id is original row id, not GW/permutation target id.
        # ------------------------------------------------------------------
        ax2.scatter(Z2[:, 0], Z2[:, 1], s=80, c="red")
        for i in range(len(Z2)):
            ax2.text(
                Z2[i, 0],
                Z2[i, 1],
                f"{i}",
                fontsize=10,
                color="black",
                fontweight="bold",
                ha="center",
                va="center",
                bbox=label_box,
            )
        ax2.set_title("Projected fakeT: original T row ids")
        set_square_limits(ax2, V2, Z2)

        # ------------------------------------------------------------------
        # Panel 3: overlay + matching lines. Points still numeric id only.
        # ------------------------------------------------------------------
        ax3.scatter(V2[:, 0], V2[:, 1], s=80, c="blue", label="V")
        ax3.scatter(Z2[:, 0], Z2[:, 1], s=80, c="red", label="T")

        for i in range(len(V2)):
            ax3.text(
                V2[i, 0],
                V2[i, 1],
                f"{i}",
                fontsize=9,
                color="black",
                fontweight="bold",
                ha="center",
                va="center",
                bbox=label_box,
            )
        for i in range(len(Z2)):
            ax3.text(
                Z2[i, 0],
                Z2[i, 1],
                f"{i}",
                fontsize=9,
                color="black",
                fontweight="bold",
                ha="center",
                va="center",
                bbox=label_box,
            )

        # Black lines: fixed Stage3 GW permutation used by the loss.
        for i in range(len(Z2)):
            j = int(fixed_perm[i].item())
            ax3.plot(
                [Z2[i, 0], V2[j, 0]],
                [Z2[i, 1], V2[j, 1]],
                linewidth=1.4,
                c="black",
                alpha=0.75,
                label="fixed Stage3-GW" if i == 0 else None,
            )

        # Blue dashed: current dynamic hard-GW diagnostic.
        for i in range(len(Z2)):
            j = int(dyn_perm[i].item())
            ax3.plot(
                [Z2[i, 0], V2[j, 0]],
                [Z2[i, 1], V2[j, 1]],
                linewidth=1.1,
                c="blue",
                alpha=0.55,
                linestyle="--",
                label="dynamic hard-GW" if i == 0 else None,
            )

        # Green dotted: greedy cosine baseline.
        for i in range(len(Z2)):
            j = int(cos_bij[i].item())
            ax3.plot(
                [Z2[i, 0], V2[j, 0]],
                [Z2[i, 1], V2[j, 1]],
                linewidth=1.0,
                c="green",
                alpha=0.45,
                linestyle=":",
                label="greedy cosine" if i == 0 else None,
            )

        ax3.set_title("Overlay: numeric ids + matching lines")
        ax3.legend(fontsize=8)
        set_square_limits(ax3, V2, Z2)

        # ------------------------------------------------------------------
        # Bottom table 1: V id -> part name.
        # ------------------------------------------------------------------
        v_rows = []
        for j in range(len(V2)):
            pid = int(v_part_ids[j])
            name = _short_name(part_names[j] if j < len(part_names) else "")
            v_rows.append([str(j), str(pid), name])

        v_table = table_v_ax.table(
            cellText=v_rows,
            colLabels=["V id", "part pid", "part name"],
            cellLoc="center",
            colLoc="center",
            loc="center",
            bbox=[0.01, 0.00, 0.98, 1.00],
        )
        v_table.auto_set_font_size(False)
        v_table.set_fontsize(max(5, min(8, int(70 / max(len(v_rows), 1)))))
        v_table.scale(1.0, 1.12)
        for (row, col), cell in v_table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#eeeeee")

        # ------------------------------------------------------------------
        # Bottom table 2: T id -> original/source part name.
        # This is the only T-name mapping table. No OK/BAD and no matching table.
        # ------------------------------------------------------------------
        t_rows = []
        for i in range(len(Z2)):
            src_j = int(target[i].item())
            pid = int(v_part_ids[src_j]) if 0 <= src_j < len(v_part_ids) else -1
            name = _short_name(part_names[src_j] if 0 <= src_j < len(part_names) else "")
            t_rows.append([str(i), str(src_j), str(pid), name])

        t_table = table_t_ax.table(
            cellText=t_rows,
            colLabels=["T id", "source V id", "source part pid", "source part name"],
            cellLoc="center",
            colLoc="center",
            loc="center",
            bbox=[0.01, 0.00, 0.98, 1.00],
        )
        t_table.auto_set_font_size(False)
        t_table.set_fontsize(max(5, min(8, int(70 / max(len(t_rows), 1)))))
        t_table.scale(1.0, 1.12)
        for (row, col), cell in t_table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#eeeeee")

        class_name = block.get("class_name", str(cat_id))
        fig.suptitle(
            f"step={step} | block={block_idx} | cat_id={cat_id} | class={class_name} | "
            "points use numeric ids only",
            fontsize=12,
        )

        out_path = step_dir / f"block_{block_idx:03d}_cat_{cat_id}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--train_dataset", required=True)

    # Kept for command compatibility. This synthetic test does not need Stage2 weights.
    parser.add_argument("--init_weights", default="")

    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--debug_lr", type=float, default=3e-3)
    parser.add_argument("--lambda_gw_override", type=float, default=None)
    parser.add_argument("--lambda_struct_override", type=float, default=None)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--viz_dir", type=str, default="stage3_debug_viz_maskgt")
    parser.add_argument("--viz_block_idx", type=int, default=0)

    parser.add_argument("--fake_text_dim", type=int, default=0, help="0 means use visual_dim. Cleanest recovery test.")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle")
    parser.add_argument("--affine_scale", type=float, default=0.0, help="Optional non-orthogonal scale noise.")
    parser.add_argument("--affine_bias", type=float, default=0.0, help="Optional bias noise.")
    args = parser.parse_args()

    print_every = max(1, int(args.print_every))
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    cfg = load_cfg(args.model_config)
    train_cfg = cfg["train"]

    batch_size = args.batch_size or int(train_cfg.get("batch_size", 128))
    num_parts = int(train_cfg.get("num_parts", 116))
    min_proto_count = int(train_cfg.get("min_proto_count", 1))
    lambda_gw = float(train_cfg.get("lambda_gw", 1.0))
    lambda_struct = float(train_cfg.get("lambda_struct", 1.0))

    if args.lambda_gw_override is not None:
        lambda_gw = float(args.lambda_gw_override)
    if args.lambda_struct_override is not None:
        lambda_struct = float(args.lambda_struct_override)

    gw_max_iter = int(train_cfg.get("gw_max_iter", 20))
    gw_restarts = int(train_cfg.get("gw_restarts", train_cfg.get("sinkhorn_iter", 50)))

    print(
        f"[debug config] lr={args.debug_lr} lambda_gw={lambda_gw} "
        f"lambda_struct={lambda_struct} print_every={print_every} "
        f"gw_max_iter={gw_max_iter} gw_restarts={gw_restarts} "
        f"shuffle={args.shuffle} affine_scale={args.affine_scale} affine_bias={args.affine_bias}"
    )

    dataset = build_dataset(args.train_dataset, cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
    )

    print("[V] build GT part-mask maskavg prototypes")
    proto = build_maskavg_visual_prototypes(
        dataloader=loader,
        num_parts=num_parts,
        device=device,
    )
    visual_proto = proto["visual_proto"].detach().to(device)
    proto_count = proto["proto_count"].detach().to(device)
    visual_dim = int(visual_proto.shape[-1])

    fake_text_dim = int(args.fake_text_dim) if int(args.fake_text_dim) > 0 else visual_dim
    if fake_text_dim != visual_dim:
        print(
            f"[warning] fake_text_dim={fake_text_dim} != visual_dim={visual_dim}. "
            "This is a compressed/expanded affine test, not a pure rotation recovery."
        )

    print("[V]", tuple(visual_proto.shape), "valid", int((proto_count >= min_proto_count).sum().item()))
    print("[fakeT] dim", fake_text_dim)

    class_blocks = build_class_part_blocks_from_dataset(dataset, device=device)
    fake_blocks, true_match = make_fake_blocks(
        class_blocks=class_blocks,
        visual_proto=visual_proto,
        proto_count=proto_count,
        fake_text_dim=fake_text_dim,
        min_proto_count=min_proto_count,
        seed=args.seed,
        shuffle=bool(args.shuffle),
        affine_scale=float(args.affine_scale),
        affine_bias=float(args.affine_bias),
    )
    print(f"[blocks] {len(fake_blocks)} blocks, fake text dim={fake_text_dim}, visual dim={visual_dim}")

    projector = DebugProjector(fake_text_dim, visual_dim).to(device)
    criterion = Stage3GWLoss(
        sim_model=projector,
        visual_proto=visual_proto,
        class_blocks=fake_blocks,
        lambda_obj=0.0,
        lambda_gw=lambda_gw,
        lambda_struct=lambda_struct,
        gw_max_iter=gw_max_iter,
        gw_restarts=gw_restarts,
        min_proto_count=min_proto_count,
        proto_count=proto_count,
    ).to(device)

    opt = torch.optim.AdamW(projector.parameters(), lr=args.debug_lr)
    initial_weight = projector.proj.weight.detach().clone()

    def print_row(tag, losses=None):
        m = evaluate(criterion, projector, true_match, gw_max_iter, gw_restarts)
        weight_delta = (projector.proj.weight.detach() - initial_weight).norm().item()

        if losses is None:
            print(
                f"{tag} "
                f"Hfixed={m['Hacc_fixed']:.3f} Hdyn={m['Hacc_dynamic']:.3f} "
                f"retr_true={m['retrieval_true']:.3f} "
                f"paircos_fixed={m['paircos_fixed']:.4f} "
                f"paircos_dyn={m['paircos_dynamic']:.4f} "
                f"paircos_true={m['paircos_true']:.4f} "
                f"preV={m['preV']:.3e} postV={m['postV']:.3e} prepost={m['prepost']:.3e} "
                f"wd={weight_delta:.3e} parts={m['num_parts']}"
            )
        else:
            print(
                f"{tag} total={losses['total'].item():.6f} "
                f"gw={losses['gw'].item():.6f} struct={losses['struct'].item():.6f} "
                f"Hfixed={m['Hacc_fixed']:.3f} Hdyn={m['Hacc_dynamic']:.3f} "
                f"retr_true={m['retrieval_true']:.3f} "
                f"paircos_fixed={m['paircos_fixed']:.4f} "
                f"paircos_dyn={m['paircos_dynamic']:.4f} "
                f"paircos_true={m['paircos_true']:.4f} "
                f"preV={m['preV']:.3e} postV={m['postV']:.3e} prepost={m['prepost']:.3e} "
                f"wd={weight_delta:.3e}"
            )

    pca_cache = {}
    print_row("[before]")
    save_all_block_triplet_viz(
        criterion,
        projector,
        true_match,
        0,
        args.viz_dir,
        gw_max_iter,
        gw_restarts,
        pca_cache=pca_cache,
    )

    for step in tqdm(range(args.steps), desc="minimal-stage3-maskgt-affine-recovery"):
        losses = criterion(batch=None, do_anchor_audit=False, do_structure_audit=False)

        if step % print_every == 0 or step == args.steps - 1:
            print_row(f"[step {step} before_update]", losses)

        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        opt.step()

        if step % print_every == 0 or step == args.steps - 1:
            print_row(f"[step {step} after_update]", None)
            save_all_block_triplet_viz(
                criterion,
                projector,
                true_match,
                step + 1,
                args.viz_dir,
                gw_max_iter,
                gw_restarts,
                pca_cache=pca_cache,
            )


if __name__ == "__main__":
    main()
