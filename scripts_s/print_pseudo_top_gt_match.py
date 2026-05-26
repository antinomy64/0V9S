#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from anlysis import FeatureAnalyser


def obj_name_from_part_name(part_name: str) -> str:
    part_name = str(part_name)
    if "'s " in part_name:
        return part_name.split("'s ", 1)[0]
    if "’s " in part_name:
        return part_name.split("’s ", 1)[0]
    return "unknown"


def mean_features_by_part(
    features_by_part: List[torch.Tensor],
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        features_by_part:
            list length P. features_by_part[pid] is [n_pid, dim].

    Returns:
        mean_feat: [P, dim], L2-normalized mean feature. Empty part is zero.
        valid:     [P], bool.
        count:     [P], long.
    """
    means = []
    valid = []
    counts = []

    for x in features_by_part:
        if x is None or x.numel() == 0:
            means.append(torch.zeros(dim, dtype=torch.float32))
            valid.append(False)
            counts.append(0)
        else:
            x = x.float()
            m = x.mean(dim=0)
            m = F.normalize(m[None, :], dim=-1).squeeze(0)
            means.append(m.cpu())
            valid.append(True)
            counts.append(int(x.shape[0]))

    return torch.stack(means, dim=0), torch.tensor(valid, dtype=torch.bool), torch.tensor(counts, dtype=torch.long)


@torch.no_grad()
def collect_fake_gt_mean(
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
        show_progress=show_progress,
    )

    fake_by_part, gt_by_part = analyser.collect_vision_feature()

    dino_dim = int(analyser.cfg["model"].get("dino_embed_dim", 768))

    fake_mean, fake_valid, fake_count = mean_features_by_part(fake_by_part, dino_dim)
    gt_mean, gt_valid, gt_count = mean_features_by_part(gt_by_part, dino_dim)

    return {
        "part_names": analyser.part_names,
        "fake_mean": fake_mean,
        "fake_valid": fake_valid,
        "fake_count": fake_count,
        "gt_mean": gt_mean,
        "gt_valid": gt_valid,
        "gt_count": gt_count,
    }


def build_object_blocks(part_names: List[str]) -> Dict[str, List[int]]:
    blocks: Dict[str, List[int]] = {}
    for pid, pname in enumerate(part_names):
        obj = obj_name_from_part_name(pname)
        blocks.setdefault(obj, []).append(pid)
    return blocks


def top_gt_match_for_run(
    *,
    fake_mean: torch.Tensor,
    fake_valid: torch.Tensor,
    fake_count: torch.Tensor,
    gt_mean: torch.Tensor,
    gt_valid: torch.Tensor,
    gt_count: torch.Tensor,
    part_names: List[str],
    blocks: Dict[str, List[int]],
    tag: str,
) -> Dict[int, Dict[str, object]]:
    """
    For each part pid, search only GT prototypes within the same object block.
    Return best-matched GT part and margin information.
    """
    out: Dict[int, Dict[str, object]] = {}

    for obj, pids in blocks.items():
        block_gt_ids = [pid for pid in pids if bool(gt_valid[pid])]
        if len(block_gt_ids) == 0:
            for pid in pids:
                out[pid] = {
                    f"{tag}_top_gt_part": "NA",
                    f"{tag}_top_gt_cosine": float("nan"),
                    f"{tag}_self_cosine": float("nan"),
                    f"{tag}_max_offdiag_cosine": float("nan"),
                    f"{tag}_margin": float("nan"),
                    f"{tag}_self_rank": -1,
                    f"{tag}_is_self_top1": False,
                    f"{tag}_fake_count": int(fake_count[pid]),
                    f"{tag}_gt_count": int(gt_count[pid]),
                }
            continue

        gt_block = gt_mean[block_gt_ids]  # [K,D]

        for pid in pids:
            if not bool(fake_valid[pid]):
                out[pid] = {
                    f"{tag}_top_gt_part": "NO_FAKE",
                    f"{tag}_top_gt_cosine": float("nan"),
                    f"{tag}_self_cosine": float("nan"),
                    f"{tag}_max_offdiag_cosine": float("nan"),
                    f"{tag}_margin": float("nan"),
                    f"{tag}_self_rank": -1,
                    f"{tag}_is_self_top1": False,
                    f"{tag}_fake_count": int(fake_count[pid]),
                    f"{tag}_gt_count": int(gt_count[pid]),
                }
                continue

            sims = (fake_mean[pid][None, :] @ gt_block.T).squeeze(0)  # [K]
            top_local = int(torch.argmax(sims).item())
            top_pid = int(block_gt_ids[top_local])
            top_cos = float(sims[top_local].item())

            if pid in block_gt_ids:
                self_local = block_gt_ids.index(pid)
                self_cos = float(sims[self_local].item())
                self_rank = int((sims > sims[self_local]).sum().item() + 1)

                if len(block_gt_ids) > 1:
                    mask = torch.ones(len(block_gt_ids), dtype=torch.bool)
                    mask[self_local] = False
                    max_offdiag = float(sims[mask].max().item())
                else:
                    max_offdiag = float("nan")

                margin = self_cos - max_offdiag if len(block_gt_ids) > 1 else float("nan")
                is_self_top1 = (top_pid == pid)
            else:
                self_cos = float("nan")
                self_rank = -1
                max_offdiag = top_cos
                margin = float("nan")
                is_self_top1 = False

            out[pid] = {
                f"{tag}_top_gt_part": part_names[top_pid],
                f"{tag}_top_gt_cosine": top_cos,
                f"{tag}_self_cosine": self_cos,
                f"{tag}_max_offdiag_cosine": max_offdiag,
                f"{tag}_margin": margin,
                f"{tag}_self_rank": self_rank,
                f"{tag}_is_self_top1": is_self_top1,
                f"{tag}_fake_count": int(fake_count[pid]),
                f"{tag}_gt_count": int(gt_count[pid]),
            }

    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare before/after pseudo/fake part prototypes by matching each fake prototype "
            "to the most similar GT prototype inside the same object block."
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

    parser.add_argument("--out_csv", default=None)

    args = parser.parse_args()

    print("[collect before]")
    before = collect_fake_gt_mean(
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

    print("[collect after]")
    after = collect_fake_gt_mean(
        model_config=args.after_model_config,
        init_weights=args.after_init_weights,
        dataset=args.dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        part_text_name=args.part_text_name,
        obj_text_name=args.obj_text_name,
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
    assert part_names == after["part_names"], "before/after part_names are not identical"

    blocks = build_object_blocks(part_names)

    # GT prototypes are dataset-derived. They should be the same for before/after.
    # Use after GT by default.
    gt_mean = after["gt_mean"]
    gt_valid = after["gt_valid"]
    gt_count = after["gt_count"]

    before_rows = top_gt_match_for_run(
        fake_mean=before["fake_mean"],
        fake_valid=before["fake_valid"],
        fake_count=before["fake_count"],
        gt_mean=gt_mean,
        gt_valid=gt_valid,
        gt_count=gt_count,
        part_names=part_names,
        blocks=blocks,
        tag="before",
    )

    after_rows = top_gt_match_for_run(
        fake_mean=after["fake_mean"],
        fake_valid=after["fake_valid"],
        fake_count=after["fake_count"],
        gt_mean=gt_mean,
        gt_valid=gt_valid,
        gt_count=gt_count,
        part_names=part_names,
        blocks=blocks,
        tag="after",
    )

    rows = []
    for pid, pname in enumerate(part_names):
        obj = obj_name_from_part_name(pname)
        row = {
            "part_id": pid,
            "object_name": obj,
            "part_name": pname,
        }
        row.update(before_rows[pid])
        row.update(after_rows[pid])
        row["top_gt_changed"] = (
            row["before_top_gt_part"] != row["after_top_gt_part"]
        )
        row["before_correct_top1"] = row["before_is_self_top1"]
        row["after_correct_top1"] = row["after_is_self_top1"]
        row["correct_top1_changed"] = (
            row["before_correct_top1"] != row["after_correct_top1"]
        )
        rows.append(row)

    simple_cols = [
        "part_id",
        "part_name",
        "before_top_gt_part",
        "after_top_gt_part",
        "before_correct_top1",
        "after_correct_top1",
    ]

    print("| part_id | part_name | before top GT | after top GT | before self-top1 | after self-top1 |")
    print("|---:|---|---|---|---:|---:|")
    for r in rows:
        print(
            f"| {r['part_id']} | {r['part_name']} | "
            f"{r['before_top_gt_part']} | {r['after_top_gt_part']} | "
            f"{int(bool(r['before_correct_top1']))} | {int(bool(r['after_correct_top1']))} |"
        )

    before_acc = sum(bool(r["before_correct_top1"]) for r in rows) / max(len(rows), 1)
    after_acc = sum(bool(r["after_correct_top1"]) for r in rows) / max(len(rows), 1)

    print("")
    print(f"[summary] before self-top1 rate: {before_acc:.4f} ({before_acc*100:.2f}%)")
    print(f"[summary] after  self-top1 rate: {after_acc:.4f} ({after_acc*100:.2f}%)")
    print(f"[summary] top GT changed count: {sum(bool(r['top_gt_changed']) for r in rows)} / {len(rows)}")

    if args.out_csv is not None:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "part_id",
            "object_name",
            "part_name",

            "before_top_gt_part",
            "before_top_gt_cosine",
            "before_self_cosine",
            "before_max_offdiag_cosine",
            "before_margin",
            "before_self_rank",
            "before_is_self_top1",
            "before_fake_count",
            "before_gt_count",

            "after_top_gt_part",
            "after_top_gt_cosine",
            "after_self_cosine",
            "after_max_offdiag_cosine",
            "after_margin",
            "after_self_rank",
            "after_is_self_top1",
            "after_fake_count",
            "after_gt_count",

            "top_gt_changed",
            "before_correct_top1",
            "after_correct_top1",
            "correct_top1_changed",
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
