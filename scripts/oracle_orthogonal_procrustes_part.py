#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

# This script is intended to be placed under scripts/ in the Talk2DINO repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the repository analysis/model/dataset loading utilities.
from anlysis import (  # noqa: E402
    FeatureAnalyser,
    get_part_names,
    mean_features_by_part,
    to_device_batch,
)


def build_analyser(args, dataset: str) -> FeatureAnalyser:
    return FeatureAnalyser(
        model_config=args.model_config,
        dataset=dataset,
        init_weights=args.init_weights,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_parts=args.num_parts,
        device=args.device,
        show_progress=True,
    )


@torch.no_grad()
def collect_gt_visual_features_by_part(analyser: FeatureAnalyser) -> List[torch.Tensor]:
    """
    Collect instance-level GT-mask visual prototypes grouped by global part id.

    This intentionally reuses FeatureAnalyser.loader, device, loss_helper and
    normalization logic, but does not call FeatureAnalyser.collect_vision_feature().
    The existing collect_vision_feature() also runs the current pseudo-label branch;
    an oracle Procrustes test only needs the GT branch.
    """
    num_parts = int(analyser.num_parts)
    dim = int(analyser.cfg["model"].get("dino_embed_dim", 768))
    chunks: List[List[torch.Tensor]] = [[] for _ in range(num_parts)]

    for batch in tqdm(analyser.loader, desc="collect GT-mask visual prototypes"):
        batch = to_device_batch(batch, analyser.device)

        patch_tokens = analyser.loss_helper._safe_normalize(
            batch["patch_tokens"].float(), dim=-1
        )  # [B, N, D]
        obj_mask = batch["obj_mask_patch"].bool()             # [B, N]
        part_gt = batch["part_gt_mask_patch"].bool()          # [B, K, N]
        part_valid = batch["part_valid_mask"].bool()          # [B, K]
        part_ids = batch["part_category_id"].long()           # [B, K]

        gt_mask = part_gt & obj_mask[:, None, :] & part_valid[:, :, None]
        gt_count = gt_mask.sum(dim=-1)                         # [B, K]

        gt_proto = torch.einsum("bkn,bnd->bkd", gt_mask.float(), patch_tokens)
        gt_proto = gt_proto / gt_count.clamp_min(1).float()[:, :, None]
        gt_proto = analyser.loss_helper._safe_normalize(gt_proto, dim=-1)

        flat_pid = part_ids.reshape(-1)
        flat_valid = (
            part_valid.reshape(-1)
            & (gt_count.reshape(-1) > 0)
            & (flat_pid >= 0)
            & (flat_pid < num_parts)
        )
        flat_proto = gt_proto.reshape(-1, dim)

        for pid in range(num_parts):
            keep = flat_valid & (flat_pid == pid)
            if keep.any():
                chunks[pid].append(flat_proto[keep].detach().cpu())

    out: List[torch.Tensor] = []
    for pid in range(num_parts):
        if len(chunks[pid]) == 0:
            out.append(torch.empty((0, dim), dtype=torch.float32))
        else:
            out.append(torch.cat(chunks[pid], dim=0).float().contiguous())
    return out


@torch.no_grad()
def collect_part_mean_prototypes(
    analyser: FeatureAnalyser,
) -> Dict[str, torch.Tensor]:
    dim = int(analyser.cfg["model"].get("dino_embed_dim", 768))

    # Reuse repository text collection logic.
    _, _, _, part_text_proj_by_part = analyser.collect_text_features()
    text_mean, text_valid, text_count = mean_features_by_part(
        part_text_proj_by_part, dim=dim
    )

    gt_visual_by_part = collect_gt_visual_features_by_part(analyser)
    visual_mean, visual_valid, visual_count = mean_features_by_part(
        gt_visual_by_part, dim=dim
    )

    return {
        "text_mean": F.normalize(text_mean.float(), dim=-1),
        "text_valid": text_valid.bool(),
        "text_count": text_count.long(),
        "visual_mean": F.normalize(visual_mean.float(), dim=-1),
        "visual_valid": visual_valid.bool(),
        "visual_count": visual_count.long(),
    }


def solve_orthogonal_procrustes(
    text: torch.Tensor,
    visual: torch.Tensor,
    proper_rotation: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve min_Q || text @ Q - visual ||_F, subject to Q^T Q = I.

    text, visual: [P, D], row-vector convention.
    Returns Q [D, D] and singular values.
    """
    cross = text.T @ visual
    u, s, vh = torch.linalg.svd(cross, full_matrices=False)
    q = u @ vh

    if proper_rotation and torch.linalg.det(q) < 0:
        u = u.clone()
        u[:, -1] *= -1.0
        q = u @ vh

    return q, s


def retrieval_metrics(
    text: torch.Tensor,
    visual: torch.Tensor,
    part_ids: torch.Tensor,
    part_names: List[str],
    label: str,
    print_per_part: bool,
) -> Dict[str, float]:
    """Evaluate same-part retrieval among the provided valid prototypes."""
    text = F.normalize(text.float(), dim=-1)
    visual = F.normalize(visual.float(), dim=-1)
    sim = text @ visual.T

    n = int(sim.shape[0])
    diag = sim.diag()
    order = torch.argsort(sim, dim=1, descending=True)
    target = torch.arange(n)
    ranks = (order == target[:, None]).nonzero(as_tuple=False)[:, 1] + 1

    top1_idx = order[:, 0]
    top1 = (ranks <= 1).float().mean().item()
    top5 = (ranks <= min(5, n)).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()
    mean_self_cos = diag.mean().item()
    median_rank = ranks.float().median().item()

    if n > 1:
        masked = sim.clone()
        masked[target, target] = -1e9
        best_other = masked.max(dim=1).values
        mean_margin = (diag - best_other).mean().item()
    else:
        mean_margin = float("nan")

    print("")
    print("=" * 100)
    print(f"[{label}] num_parts={n}")
    print(
        f"top1={top1:.4f}, top5={top5:.4f}, mrr={mrr:.4f}, "
        f"mean_self_cos={mean_self_cos:.4f}, mean_self_margin={mean_margin:.4f}, "
        f"median_rank={median_rank:.1f}"
    )

    if print_per_part:
        print("")
        print(
            f"{'pid':>4}  {'part':<34}  {'rank':>5}  {'self_cos':>9}  "
            f"{'top1_pid':>8}  {'top1_part':<34}  {'top1_cos':>9}"
        )
        print("-" * 120)
        for i in range(n):
            pid = int(part_ids[i].item())
            top_pid = int(part_ids[top1_idx[i]].item())
            name = part_names[pid] if 0 <= pid < len(part_names) else f"part_{pid}"
            top_name = (
                part_names[top_pid]
                if 0 <= top_pid < len(part_names)
                else f"part_{top_pid}"
            )
            print(
                f"{pid:>4d}  {name:<34.34}  {int(ranks[i].item()):>5d}  "
                f"{float(diag[i].item()):>9.4f}  {top_pid:>8d}  "
                f"{top_name:<34.34}  {float(sim[i, top1_idx[i]].item()):>9.4f}"
            )

    return {
        "num_parts": n,
        "top1": float(top1),
        "top5": float(top5),
        "mrr": float(mrr),
        "mean_self_cos": float(mean_self_cos),
        "mean_self_margin": float(mean_margin),
        "median_rank": float(median_rank),
    }


def evaluate_split(
    split_name: str,
    protos: Dict[str, torch.Tensor],
    q: torch.Tensor,
    part_names: List[str],
    print_per_part: bool,
) -> Dict[str, Dict[str, float]]:
    valid = protos["text_valid"] & protos["visual_valid"]
    part_ids = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if part_ids.numel() == 0:
        raise RuntimeError(f"No common valid text/visual part prototypes for {split_name}.")

    text = protos["text_mean"][valid]
    visual = protos["visual_mean"][valid]
    text_rot = F.normalize(text @ q, dim=-1)

    before = retrieval_metrics(
        text, visual, part_ids, part_names,
        label=f"{split_name} before Procrustes",
        print_per_part=print_per_part,
    )
    after = retrieval_metrics(
        text_rot, visual, part_ids, part_names,
        label=f"{split_name} after Procrustes",
        print_per_part=print_per_part,
    )

    gram_before = text @ text.T
    gram_after = text_rot @ text_rot.T
    gram_max_abs_delta = (gram_before - gram_after).abs().max().item()
    fit_rmse = ((text_rot - visual) ** 2).mean().sqrt().item()

    print(
        f"[{split_name} rotation checks] "
        f"max_abs_Gram_delta={gram_max_abs_delta:.8f}, fit_rmse={fit_rmse:.6f}"
    )

    return {
        "before": before,
        "after": after,
        "rotation_checks": {
            "max_abs_gram_delta": float(gram_max_abs_delta),
            "fit_rmse": float(fit_rmse),
        },
    }


def save_rotated_linear_checkpoint(
    analyser: FeatureAnalyser,
    q: torch.Tensor,
    out_path: str,
) -> None:
    """
    Fold output rotation into a linear ProjectionLayer checkpoint.

    PyTorch Linear uses y_col = W x_col + b.  The row-vector rotation used in
    this script is y_row_new = y_row @ Q, therefore W_new = Q^T W.

    This is only exactly valid when the projector output has no post-linear
    nonlinear activation and no hidden stack after the output layer.
    """
    model_cfg = analyser.cfg.get("model", {})
    act = model_cfg.get("act", None)
    hidden_layer = model_cfg.get("hidden_layer", False)

    act_is_none = act is None or str(act).lower() == "none"
    hidden_is_false = hidden_layer in (False, None, 0, "False", "false")
    if not act_is_none or not hidden_is_false:
        raise RuntimeError(
            "Saving an equivalent rotated checkpoint is only supported for the "
            "linear projector config: act: null and hidden_layer: False. "
            f"Got act={act!r}, hidden_layer={hidden_layer!r}. "
            "The in-memory oracle retrieval metrics are still valid."
        )

    state = {
        k: v.detach().cpu().clone()
        for k, v in analyser.model.state_dict().items()
    }
    if "linear_layer.weight" not in state:
        raise KeyError(
            "Checkpoint/model state has no 'linear_layer.weight'; cannot fold rotation."
        )

    q_cpu = q.detach().cpu().float()
    w = state["linear_layer.weight"].float()
    if w.shape[0] != q_cpu.shape[0]:
        raise ValueError(
            f"Output dim mismatch: linear_layer.weight={tuple(w.shape)}, Q={tuple(q_cpu.shape)}"
        )

    state["linear_layer.weight"] = (q_cpu.T @ w).to(state["linear_layer.weight"].dtype)
    if "linear_layer.bias" in state:
        b = state["linear_layer.bias"].float()
        state["linear_layer.bias"] = (q_cpu.T @ b).to(state["linear_layer.bias"].dtype)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    print(f"[saved rotated checkpoint] {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle Orthogonal Procrustes test: projected part text -> GT visual part prototypes"
    )
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--fit_dataset", required=True, help="Dataset used to fit Q, normally train split")
    parser.add_argument("--eval_dataset", default=None, help="Optional dataset used only for retrieval evaluation, normally val split")
    parser.add_argument("--init_weights", required=True)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_parts", type=int, default=116)
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--proper_rotation",
        action="store_true",
        default=False,
        help="Restrict det(Q)=+1. Default Orthogonal Procrustes allows reflection.",
    )
    parser.add_argument("--print_per_part", action="store_true", default=False)
    parser.add_argument("--out_rotation_pth", default=None)
    parser.add_argument("--out_rotated_ckpt", default=None)
    parser.add_argument("--out_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    part_names = get_part_names(args.num_parts)

    print("[oracle warning] This test uses GT part masks to fit Q. It is diagnostic only.")
    print("[fit dataset]", args.fit_dataset)
    print("[eval dataset]", args.eval_dataset)
    print("[init weights]", args.init_weights)

    fit_analyser = build_analyser(args, args.fit_dataset)
    fit_protos = collect_part_mean_prototypes(fit_analyser)

    fit_valid = fit_protos["text_valid"] & fit_protos["visual_valid"]
    fit_part_ids = torch.nonzero(fit_valid, as_tuple=False).squeeze(1)
    if fit_part_ids.numel() < 2:
        raise RuntimeError(
            f"Need at least 2 common valid parts to fit Procrustes, got {fit_part_ids.numel()}."
        )

    fit_text = fit_protos["text_mean"][fit_valid].float()
    fit_visual = fit_protos["visual_mean"][fit_valid].float()

    q, singular_values = solve_orthogonal_procrustes(
        fit_text,
        fit_visual,
        proper_rotation=args.proper_rotation,
    )

    det_q = torch.linalg.det(q).item()
    orth_error = ((q.T @ q) - torch.eye(q.shape[0])).abs().max().item()
    print("")
    print("=" * 100)
    print(
        f"[Q] shape={tuple(q.shape)}, det={det_q:.6f}, "
        f"max_abs(Q^TQ-I)={orth_error:.8f}, "
        f"fit_common_parts={int(fit_part_ids.numel())}"
    )
    print(
        f"[SVD] nonzero-ish singular values (>1e-6): "
        f"{int((singular_values > 1e-6).sum().item())}/{singular_values.numel()}"
    )

    results: Dict[str, object] = {
        "model_config": args.model_config,
        "fit_dataset": args.fit_dataset,
        "eval_dataset": args.eval_dataset,
        "init_weights": args.init_weights,
        "num_parts": int(args.num_parts),
        "proper_rotation": bool(args.proper_rotation),
        "fit_part_ids": fit_part_ids.tolist(),
        "q_det": float(det_q),
        "q_orthogonality_max_abs_error": float(orth_error),
        "singular_values": singular_values.tolist(),
    }

    results["fit"] = evaluate_split(
        "fit",
        fit_protos,
        q,
        part_names,
        print_per_part=args.print_per_part,
    )

    if args.eval_dataset is not None:
        eval_analyser = build_analyser(args, args.eval_dataset)
        eval_protos = collect_part_mean_prototypes(eval_analyser)
        results["eval"] = evaluate_split(
            "eval",
            eval_protos,
            q,
            part_names,
            print_per_part=args.print_per_part,
        )

    if args.out_rotation_pth is not None:
        out_q = Path(args.out_rotation_pth)
        out_q.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "Q": q.detach().cpu(),
                "fit_part_ids": fit_part_ids.cpu(),
                "singular_values": singular_values.detach().cpu(),
                "proper_rotation": bool(args.proper_rotation),
                "model_config": args.model_config,
                "fit_dataset": args.fit_dataset,
                "init_weights": args.init_weights,
            },
            out_q,
        )
        print(f"[saved rotation] {out_q}")

    if args.out_rotated_ckpt is not None:
        save_rotated_linear_checkpoint(fit_analyser, q, args.out_rotated_ckpt)

    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[saved results] {out_json}")


if __name__ == "__main__":
    main()
