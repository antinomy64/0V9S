#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from anlysis import FeatureAnalyser, mean_features_by_part


def obj_name_from_part_name(part_name: str) -> str:
    part_name = str(part_name)
    if "'s " in part_name:
        return part_name.split("'s ", 1)[0]
    if "’s " in part_name:
        return part_name.split("’s ", 1)[0]
    return "unknown"


def build_object_blocks(part_names: List[str]) -> Dict[str, List[int]]:
    blocks: Dict[str, List[int]] = {}
    for pid, pname in enumerate(part_names):
        obj = obj_name_from_part_name(pname)
        blocks.setdefault(obj, []).append(pid)
    return blocks


def infer_first_nonempty_dim(xs, fallback=None):
    for x in xs:
        if x is not None and x.shape[0] > 0:
            return int(x.shape[-1])
    if fallback is not None:
        return int(fallback)
    raise RuntimeError("Cannot infer feature dimension.")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity between projected part text features and GT visual part prototypes."
    )
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init_weights", required=True)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--out_csv", default=None)

    args = parser.parse_args()

    analyser = FeatureAnalyser(
        model_config=args.model_config,
        dataset=args.dataset,
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
        show_progress=args.show_progress,
    )

    print("[collect vision GT prototypes]")
    _fake_by_part, gt_by_part = analyser.collect_vision_feature()

    print("[collect projected text features]")
    (
        _obj_text_raw_by_category,
        _obj_text_proj_by_category,
        _part_text_raw_by_part,
        part_text_proj_by_part,
    ) = analyser.collect_text_features()

    dino_dim = int(analyser.cfg["model"].get("dino_embed_dim", 768))

    gt_mean, gt_valid, gt_count = mean_features_by_part(gt_by_part, dino_dim)
    text_proj_mean, text_valid, text_count = mean_features_by_part(part_text_proj_by_part, dino_dim)

    gt_mean = F.normalize(gt_mean.float(), dim=-1, eps=1e-6)
    text_proj_mean = F.normalize(text_proj_mean.float(), dim=-1, eps=1e-6)

    part_names = list(analyser.part_names)
    if len(part_names) != args.num_parts:
        try:
            from src.voc116_part_coarse import COARSE_PART_CLASSES, FINE_PART_CLASSES
            if args.num_parts == len(COARSE_PART_CLASSES):
                part_names = list(COARSE_PART_CLASSES)
            elif args.num_parts == len(FINE_PART_CLASSES):
                part_names = list(FINE_PART_CLASSES)
        except Exception:
            part_names = [f"part_{i}" for i in range(args.num_parts)]

    blocks = build_object_blocks(part_names)

    rows = []
    correct = 0
    valid_total = 0

    print("| part_id | part_name | top GT | self cos | top cos | margin | self rank | self top1 | gt_count | text_count |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for obj, pids in blocks.items():
        gt_ids = [pid for pid in pids if bool(gt_valid[pid])]
        if len(gt_ids) == 0:
            continue

        gt_block = gt_mean[gt_ids]

        for pid in pids:
            if not bool(text_valid[pid]):
                continue

            sims = (text_proj_mean[pid][None, :] @ gt_block.T).squeeze(0)
            top_local = int(torch.argmax(sims).item())
            top_pid = int(gt_ids[top_local])

            top_cos = float(sims[top_local].item())

            if pid in gt_ids:
                self_local = gt_ids.index(pid)
                self_cos = float(sims[self_local].item())
                self_rank = int((sims > sims[self_local]).sum().item() + 1)

                if len(gt_ids) > 1:
                    mask = torch.ones(len(gt_ids), dtype=torch.bool)
                    mask[self_local] = False
                    max_offdiag = float(sims[mask].max().item())
                    margin = self_cos - max_offdiag
                else:
                    max_offdiag = float("nan")
                    margin = float("nan")

                is_self_top1 = top_pid == pid
                valid_total += 1
                correct += int(is_self_top1)
            else:
                self_cos = float("nan")
                self_rank = -1
                margin = float("nan")
                is_self_top1 = False

            row = {
                "part_id": pid,
                "object_name": obj,
                "part_name": part_names[pid],
                "top_gt_part": part_names[top_pid],
                "self_cosine": self_cos,
                "top_gt_cosine": top_cos,
                "margin": margin,
                "self_rank": self_rank,
                "is_self_top1": is_self_top1,
                "gt_count": int(gt_count[pid]),
                "text_count": int(text_count[pid]),
            }
            rows.append(row)

            print(
                f"| {pid} | {part_names[pid]} | {part_names[top_pid]} | "
                f"{self_cos:.4f} | {top_cos:.4f} | {margin:.4f} | "
                f"{self_rank} | {int(is_self_top1)} | {int(gt_count[pid])} | {int(text_count[pid])} |"
            )

    acc = correct / max(valid_total, 1)
    print("")
    print(f"[summary] self-top1 rate: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"[summary] valid parts: {valid_total}")

    if args.out_csv is not None:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "part_id",
            "object_name",
            "part_name",
            "top_gt_part",
            "self_cosine",
            "top_gt_cosine",
            "margin",
            "self_rank",
            "is_self_top1",
            "gt_count",
            "text_count",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
