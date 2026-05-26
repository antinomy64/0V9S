#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from anlysis import FeatureAnalyser


def _pool_items(pool_ret):
    if isinstance(pool_ret, (tuple, list)):
        return list(pool_ret)
    return [pool_ret]


def infer_anchor_idx(
    *,
    pool_ret,
    patch_tokens: torch.Tensor,  # [B,N,D], normalized
    obj_mask: torch.Tensor,      # [B,N]
    abs_logits: torch.Tensor,    # [B,K,N]
) -> torch.Tensor:
    """
    Try to recover anchor patch index [B,K] from _anchor_proto_em_pool return.

    Priority:
      1. integer tensor [B,K] with values in [0,N)
      2. anchor token tensor [B,K,D], matched back to nearest patch token
      3. fallback: argmax(abs_logits)
    """
    B, N, D = patch_tokens.shape
    _, K, _ = abs_logits.shape

    items = _pool_items(pool_ret)

    # Case 1: direct integer anchor index
    for t in items:
        if not torch.is_tensor(t):
            continue
        if tuple(t.shape) == (B, K) and t.dtype in (
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8,
        ) and t.dtype != torch.bool:
            idx = t.long()
            if idx.numel() == 0:
                continue
            if int(idx.min().item()) >= 0 and int(idx.max().item()) < N:
                return idx

    # Case 2: anchor token [B,K,D], find nearest patch token.
    # Exclude pool_ret[1] by convention because existing code uses it as proto_part.
    best_idx = None
    best_score = None
    for i, t in enumerate(items):
        if not torch.is_tensor(t):
            continue
        if i == 1:
            continue
        if tuple(t.shape) != (B, K, D):
            continue

        tok = F.normalize(t.float(), dim=-1)
        sims = torch.einsum("bkd,bnd->bkn", tok, patch_tokens.float())
        sims = sims.masked_fill(~obj_mask[:, None, :], -1e4)
        max_sim, idx = sims.max(dim=-1)

        finite = max_sim[max_sim > -1e3]
        score = float(finite.mean().item()) if finite.numel() > 0 else -1e9
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx.long()

    if best_idx is not None:
        return best_idx

    # Case 3: fallback to initial logit argmax.
    return abs_logits.argmax(dim=-1).long()


@torch.no_grad()
def collect_anchor_hit_stats(
    *,
    model_config: str,
    init_weights: str,
    dataset: str,
    obj_feature_name: str,
    part_feature_name: str,
    obj_text_name: str,
    part_text_name: str,
    resize_dim: int,
    crop_dim: int,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    num_parts: int,
    device: str,
    show_progress: bool,
) -> Dict[str, object]:
    analyser = FeatureAnalyser(
        model_config=model_config,
        dataset=dataset,
        init_weights=init_weights,
        obj_feature_name=obj_feature_name,
        part_feature_name=part_feature_name,
        obj_text_name=obj_text_name,
        part_text_name=part_text_name,
        resize_dim=resize_dim,
        crop_dim=crop_dim,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        num_parts=num_parts,
        device=device,
        show_progress=False,
    )

    P = num_parts
    dev = analyser.device

    anchor_count = torch.zeros(P, device=dev, dtype=torch.float32)
    self_hit_count = torch.zeros(P, device=dev, dtype=torch.float32)
    any_gt_hit_count = torch.zeros(P, device=dev, dtype=torch.float32)
    confusion = torch.zeros(P, P, device=dev, dtype=torch.float32)

    for batch in tqdm(
        analyser.loader,
        desc="collect anchor GT hit stats",
        disable=not show_progress,
    ):
        batch = {
            k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        part_text = batch["part_text_feat"].float()       # [B,K,512]
        patch_tokens = analyser.loss_helper._safe_normalize(
            batch["patch_tokens"].float(), dim=-1
        )                                                 # [B,N,D]
        obj_mask = batch["obj_mask_patch"].bool()         # [B,N]
        part_valid = batch["part_valid_mask"].bool()      # [B,K]
        part_gt = batch["part_gt_mask_patch"].bool()      # [B,K,N]
        part_ids = batch["part_category_id"].long()       # [B,K]

        B, K, N = part_gt.shape

        part_proj = analyser.model.project_clip_txt(part_text)
        part_proj = analyser.loss_helper._safe_normalize(part_proj, dim=-1)

        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens)
        abs_logits = abs_logits / float(analyser.loss_helper.patch_temperature)
        abs_logits = abs_logits.masked_fill(~obj_mask[:, None, :], -1e4)

        pool_ret = analyser.loss_helper._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask,
            part_valid_mask=part_valid,
            part_gt_mask_patch=part_gt,
            num_iters=analyser.loss_helper.em_iters,
            return_anchor_tokens=True,
        )

        anchor_valid = pool_ret[-1].bool()  # existing code also uses pool_ret[-1] as anchor_valid
        anchor_idx = infer_anchor_idx(
            pool_ret=pool_ret,
            patch_tokens=patch_tokens,
            obj_mask=obj_mask,
            abs_logits=abs_logits,
        ).clamp_min(0).clamp_max(N - 1)      # [B,K]

        valid_pid = (part_ids >= 0) & (part_ids < P)
        valid_anchor = part_valid & anchor_valid & valid_pid

        # gt_at_anchor[b, src_k, gt_k] = whether source anchor point lies in GT part gt_k.
        gt_at_anchor = torch.gather(
            part_gt,
            dim=2,
            index=anchor_idx[:, None, :].expand(B, K, K),
        ).permute(0, 2, 1)  # [B, src_K, gt_K]
        gt_at_anchor = gt_at_anchor & part_valid[:, None, :]

        hit_any = gt_at_anchor.any(dim=-1)              # [B,src_K]
        hit_local = gt_at_anchor.float().argmax(dim=-1) # [B,src_K], arbitrary if no hit
        hit_pid = part_ids.gather(1, hit_local)         # [B,src_K]
        hit_pid = hit_pid.masked_fill(~hit_any, -1)

        self_hit = hit_any & (hit_pid == part_ids)

        flat_src = part_ids.reshape(-1)
        flat_hit = hit_pid.reshape(-1)
        flat_valid = valid_anchor.reshape(-1)
        flat_self = self_hit.reshape(-1)
        flat_any = hit_any.reshape(-1)

        keep = flat_valid & (flat_src >= 0) & (flat_src < P)

        if keep.any():
            src = flat_src[keep]
            ones = torch.ones_like(src, dtype=torch.float32)
            anchor_count.index_add_(0, src, ones)

            self_keep = keep & flat_self
            if self_keep.any():
                src_self = flat_src[self_keep]
                self_hit_count.index_add_(
                    0,
                    src_self,
                    torch.ones_like(src_self, dtype=torch.float32),
                )

            any_keep = keep & flat_any & (flat_hit >= 0) & (flat_hit < P)
            if any_keep.any():
                src_any = flat_src[any_keep]
                hit_any_pid = flat_hit[any_keep]
                any_gt_hit_count.index_add_(
                    0,
                    src_any,
                    torch.ones_like(src_any, dtype=torch.float32),
                )

                linear = src_any * P + hit_any_pid
                confusion.view(-1).index_add_(
                    0,
                    linear,
                    torch.ones_like(linear, dtype=torch.float32),
                )

    self_hit_rate = self_hit_count / anchor_count.clamp_min(1.0)
    any_gt_hit_rate = any_gt_hit_count / anchor_count.clamp_min(1.0)

    self_hit_rate[anchor_count <= 0] = 0
    any_gt_hit_rate[anchor_count <= 0] = 0

    return {
        "part_names": analyser.part_names,
        "anchor_count": anchor_count.detach().cpu(),
        "self_hit_count": self_hit_count.detach().cpu(),
        "any_gt_hit_count": any_gt_hit_count.detach().cpu(),
        "self_hit_rate": self_hit_rate.detach().cpu(),
        "any_gt_hit_rate": any_gt_hit_rate.detach().cpu(),
        "confusion": confusion.detach().cpu(),
    }


def top_hit_part(confusion_row: torch.Tensor, part_names: List[str]) -> Tuple[str, float]:
    total = float(confusion_row.sum().item())
    if total <= 0:
        return "NO_GT_HIT", 0.0
    pid = int(confusion_row.argmax().item())
    rate = float(confusion_row[pid].item()) / total
    return part_names[pid], rate


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare before/after anchor-point GT hit. "
            "For each part, check whether its selected anchor point falls inside its own GT mask, "
            "and which GT part it most often hits."
        )
    )

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--before_model_config", required=True)
    parser.add_argument("--before_init_weights", required=True)
    parser.add_argument("--after_model_config", required=True)
    parser.add_argument("--after_init_weights", required=True)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show_progress", action="store_true")

    args = parser.parse_args()

    print("[collect before anchor hit]")
    before = collect_anchor_hit_stats(
        model_config=args.before_model_config,
        init_weights=args.before_init_weights,
        dataset=args.dataset,
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
        show_progress=args.show_progress,
    )

    print("[collect after anchor hit]")
    after = collect_anchor_hit_stats(
        model_config=args.after_model_config,
        init_weights=args.after_init_weights,
        dataset=args.dataset,
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
        show_progress=args.show_progress,
    )

    part_names = before["part_names"]
    assert part_names == after["part_names"], "before/after part_names mismatch"

    print("| part_id | part_name | before top-hit GT | after top-hit GT | before self-hit% | after self-hit% | before any-GT-hit% | after any-GT-hit% |")
    print("|---:|---|---|---|---:|---:|---:|---:|")

    before_self_rates = []
    after_self_rates = []

    for pid, pname in enumerate(part_names):
        before_top, before_top_rate = top_hit_part(before["confusion"][pid], part_names)
        after_top, after_top_rate = top_hit_part(after["confusion"][pid], part_names)

        b_self = float(before["self_hit_rate"][pid].item()) * 100.0
        a_self = float(after["self_hit_rate"][pid].item()) * 100.0
        b_any = float(before["any_gt_hit_rate"][pid].item()) * 100.0
        a_any = float(after["any_gt_hit_rate"][pid].item()) * 100.0

        before_self_rates.append(b_self)
        after_self_rates.append(a_self)

        print(
            f"| {pid} | {pname} | "
            f"{before_top} ({before_top_rate*100:.1f}%) | "
            f"{after_top} ({after_top_rate*100:.1f}%) | "
            f"{b_self:.2f} | {a_self:.2f} | {b_any:.2f} | {a_any:.2f} |"
        )

    print("")
    print(f"[summary] before mean self-hit: {sum(before_self_rates)/max(len(before_self_rates),1):.2f}%")
    print(f"[summary] after  mean self-hit: {sum(after_self_rates)/max(len(after_self_rates),1):.2f}%")


if __name__ == "__main__":
    main()
