#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Put this file under scripts/ next to view_anchor.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import view_anchor as VA
from src.dataset_joint import joint_collate_fn
from src.loss_joint import JointObjPartLoss


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def get_pid_name(pid_to_name: Dict[int, str], pid: int) -> str:
    pid = int(pid)
    if pid < 0:
        return "__unlabeled__"
    return pid_to_name.get(pid, f"part_{pid}")


def add_count(d: Dict[Tuple[int, int], int], src: int, tgt: int, n: int = 1):
    d[(int(src), int(tgt))] += int(n)


def write_anchor_confusion(out_csv: Path, anchor_counts, anchor_totals, pid_to_name):
    rows = []
    for (src, gt), cnt in sorted(anchor_counts.items(), key=lambda x: (x[0][0], x[0][1])):
        total = int(anchor_totals.get(src, 0))
        ratio = float(cnt) / max(float(total), 1.0)
        rows.append({
            "source_pid": src,
            "source_part": get_pid_name(pid_to_name, src),
            "anchor_gt_pid": gt,
            "anchor_gt_part": get_pid_name(pid_to_name, gt),
            "count": int(cnt),
            "total_for_source": total,
            "ratio_in_source": ratio,
            "is_identity": int(src == gt),
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "source_pid", "source_part", "anchor_gt_pid", "anchor_gt_part", "count", "total_for_source", "ratio_in_source", "is_identity"
        ])
        writer.writeheader()
        writer.writerows(rows)


def write_pseudo_composition(out_csv: Path, pseudo_counts, pseudo_totals, pid_to_name):
    rows = []
    for (src, gt), cnt in sorted(pseudo_counts.items(), key=lambda x: (x[0][0], x[0][1])):
        total = int(pseudo_totals.get(src, 0))
        ratio = float(cnt) / max(float(total), 1.0)
        rows.append({
            "source_pid": src,
            "source_part": get_pid_name(pid_to_name, src),
            "pseudo_patch_gt_pid": gt,
            "pseudo_patch_gt_part": get_pid_name(pid_to_name, gt),
            "patch_count": int(cnt),
            "total_patches_for_source": total,
            "ratio_in_source_pseudo": ratio,
            "is_identity": int(src == gt),
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "source_pid", "source_part", "pseudo_patch_gt_pid", "pseudo_patch_gt_part", "patch_count", "total_patches_for_source", "ratio_in_source_pseudo", "is_identity"
        ])
        writer.writeheader()
        writer.writerows(rows)


def dominant_from_counts(counts_by_src_tgt, src: int):
    best_tgt, best_cnt = None, 0
    for (s, t), c in counts_by_src_tgt.items():
        if int(s) != int(src):
            continue
        if int(c) > best_cnt:
            best_tgt, best_cnt = int(t), int(c)
    return best_tgt, best_cnt


def write_summary(out_csv: Path, anchor_counts, anchor_totals, pseudo_counts, pseudo_totals, pid_to_name):
    all_src = sorted(set(anchor_totals.keys()) | set(pseudo_totals.keys()) | set(pid_to_name.keys()))
    rows = []
    for src in all_src:
        # Skip global part ids never touched by the current dataset/run.
        anchor_total = int(anchor_totals.get(src, 0))
        pseudo_total = int(pseudo_totals.get(src, 0))
        if anchor_total == 0 and pseudo_total == 0:
            continue

        anchor_self = int(anchor_counts.get((src, src), 0))
        pseudo_self = int(pseudo_counts.get((src, src), 0))

        anchor_dom_gt, anchor_dom_cnt = dominant_from_counts(anchor_counts, src)
        pseudo_dom_gt, pseudo_dom_cnt = dominant_from_counts(pseudo_counts, src)

        rows.append({
            "source_pid": src,
            "source_part": get_pid_name(pid_to_name, src),
            "anchor_total": anchor_total,
            "anchor_self_hit": anchor_self,
            "anchor_self_hit_rate": anchor_self / max(float(anchor_total), 1.0),
            "anchor_dominant_gt_pid": -1 if anchor_dom_gt is None else anchor_dom_gt,
            "anchor_dominant_gt_part": "" if anchor_dom_gt is None else get_pid_name(pid_to_name, anchor_dom_gt),
            "anchor_dominant_ratio": 0.0 if anchor_dom_gt is None else anchor_dom_cnt / max(float(anchor_total), 1.0),
            "pseudo_patch_total": pseudo_total,
            "pseudo_self_patch_count": pseudo_self,
            "pseudo_self_patch_ratio": pseudo_self / max(float(pseudo_total), 1.0),
            "pseudo_dominant_gt_pid": -1 if pseudo_dom_gt is None else pseudo_dom_gt,
            "pseudo_dominant_gt_part": "" if pseudo_dom_gt is None else get_pid_name(pid_to_name, pseudo_dom_gt),
            "pseudo_dominant_ratio": 0.0 if pseudo_dom_gt is None else pseudo_dom_cnt / max(float(pseudo_total), 1.0),
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "source_pid", "source_part", "anchor_total", "anchor_self_hit", "anchor_self_hit_rate",
            "anchor_dominant_gt_pid", "anchor_dominant_gt_part", "anchor_dominant_ratio",
            "pseudo_patch_total", "pseudo_self_patch_count", "pseudo_self_patch_ratio",
            "pseudo_dominant_gt_pid", "pseudo_dominant_gt_part", "pseudo_dominant_ratio",
        ])
        writer.writeheader()
        writer.writerows(rows)


def write_txt_preview(csv_path: Path, txt_path: Path, max_rows: int = 500):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if len(df) > max_rows:
        df = df.head(max_rows)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))
        f.write("\n")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        "Audit which GT parts are selected by anchors and by EM pseudo clusters."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--init_weights", required=True)
    parser.add_argument("--run_name", default="run")

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", default=None)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch_temperature", type=float, default=None)
    parser.add_argument("--em_iters", type=int, default=None)
    parser.add_argument("--present_only_parts", action="store_true", default=False,
                        help="Only mine anchors/pseudo clusters for GT-present parts in each image. Oracle/audit only.")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all dataset samples.")
    parser.add_argument("--out_dir", default="audits/anchor_gt_pseudo_composition")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {args.run_name}")
    print(f"[device] {device}")
    print(f"[present_only_parts] {args.present_only_parts}")

    model, cfg = VA.load_projector_from_config(args.model_config, args.init_weights, device)
    train_cfg = cfg.get("train", {})
    patch_temperature = float(args.patch_temperature if args.patch_temperature is not None else train_cfg.get("patch_temperature", 0.07))
    em_iters = int(args.em_iters if args.em_iters is not None else train_cfg.get("em_iters", 1))
    obj_ltype = train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce"))

    dataset = VA.build_joint_dataset(args, cfg)
    if args.max_samples and args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, list(range(min(int(args.max_samples), len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=joint_collate_fn,
        pin_memory=True,
    )

    anchor_helper = JointObjPartLoss(
        sim_model=model,
        obj_ltype=obj_ltype,
        lambda_obj=0.0,
        lambda_inst=0.0,
        lambda_overlap=0.0,
        lambda_spear=0.0,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
    ).to(device)
    anchor_helper.eval()

    anchor_counts = defaultdict(int)  # (source_pid, anchor_gt_pid) -> count
    anchor_totals = defaultdict(int)  # source_pid -> anchors
    pseudo_counts = defaultdict(int)  # (source_pid, gt_pid) -> patch count
    pseudo_totals = defaultdict(int)  # source_pid -> patch count
    pid_to_name: Dict[int, str] = {}

    total_samples = 0
    skipped_no_pseudo = 0

    for batch in tqdm(loader, desc=f"anchor/pseudo GT audit: {args.run_name}"):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        stage2 = VA.get_stage2_anchor_outputs(
            model=model,
            anchor_helper=anchor_helper,
            batch=batch,
            patch_temperature=patch_temperature,
            em_iters=em_iters,
            present_only_anchor=bool(args.present_only_parts),
        )
        anchor_idx = VA.recover_anchor_patch_indices(
            patch_tokens_norm=stage2["patch_tokens_norm"],
            obj_mask_patch=stage2["obj_mask_patch"],
            anchor_tokens=stage2["anchor_tokens"],
            anchor_valid=stage2["anchor_valid"],
        )

        B = int(batch["patch_tokens"].shape[0])
        for b in range(B):
            total_samples += 1
            meta = VA.get_meta(batch, b)
            part_ids = batch["part_category_id"][b].long()

            # Cache names for source/GT part ids.
            for k in range(int(part_ids.numel())):
                pid = int(part_ids[k].item())
                if pid >= 0 and pid not in pid_to_name:
                    pid_to_name[pid] = VA.get_part_name(meta, k, pid)

            gt_pid_map = VA.build_gt_pid_map(
                obj_mask_patch_b=stage2["obj_mask_patch"][b],
                part_valid_mask_b=batch["part_valid_mask"][b].bool(),
                part_ids_b=part_ids,
                part_gt_mask_patch_b=batch["part_gt_mask_patch"][b].bool(),
            )
            if gt_pid_map is None:
                continue

            # ------------------------------------------------------------
            # Table 1: source text part -> anchor patch GT part.
            # ------------------------------------------------------------
            part_anchor_mask_b = stage2["part_anchor_mask"][b].bool()
            for k in torch.nonzero(part_anchor_mask_b, as_tuple=False).squeeze(1).tolist():
                aidx = int(anchor_idx[b, k].item())
                if aidx < 0:
                    continue
                src_pid = int(part_ids[k].item())
                gt_pid = int(gt_pid_map[aidx].item()) if aidx < int(gt_pid_map.numel()) else -1
                add_count(anchor_counts, src_pid, gt_pid, 1)
                anchor_totals[src_pid] += 1

            # ------------------------------------------------------------
            # Tables 2/3: source pseudo cluster -> composition of GT part ids.
            # ------------------------------------------------------------
            pseudo_pid = VA.build_pseudo_pid_map(
                patch_tokens_norm_b=stage2["patch_tokens_norm"][b],
                obj_mask_patch_b=stage2["obj_mask_patch"][b],
                part_anchor_mask_b=stage2["part_anchor_mask"][b],
                part_ids_b=part_ids,
                anchor_tokens_b=stage2["anchor_tokens"][b],
                anchor_idx_global_b=anchor_idx[b],
                em_iters=em_iters,
            )
            if pseudo_pid is None:
                skipped_no_pseudo += 1
                continue

            valid = pseudo_pid >= 0
            if not valid.any():
                skipped_no_pseudo += 1
                continue

            for src_pid_t in torch.unique(pseudo_pid[valid]).detach().cpu().tolist():
                src_pid = int(src_pid_t)
                mask = pseudo_pid == src_pid
                gt_vals, gt_counts = torch.unique(gt_pid_map[mask], return_counts=True)
                total = int(mask.long().sum().item())
                pseudo_totals[src_pid] += total
                for gt_pid_t, cnt_t in zip(gt_vals.detach().cpu().tolist(), gt_counts.detach().cpu().tolist()):
                    add_count(pseudo_counts, src_pid, int(gt_pid_t), int(cnt_t))

    # Write output tables.
    prefix = args.run_name
    anchor_csv = out_dir / f"{prefix}_anchor_gt_confusion.csv"
    pseudo_csv = out_dir / f"{prefix}_pseudo_cluster_gt_composition.csv"
    summary_csv = out_dir / f"{prefix}_per_part_anchor_pseudo_summary.csv"

    write_anchor_confusion(anchor_csv, anchor_counts, anchor_totals, pid_to_name)
    write_pseudo_composition(pseudo_csv, pseudo_counts, pseudo_totals, pid_to_name)
    write_summary(summary_csv, anchor_counts, anchor_totals, pseudo_counts, pseudo_totals, pid_to_name)

    # Also save readable txt previews.
    for csv_path in [anchor_csv, pseudo_csv, summary_csv]:
        write_txt_preview(csv_path, csv_path.with_suffix(".txt"), max_rows=1000)

    print("[done]")
    print(f"  samples processed     : {total_samples}")
    print(f"  skipped_no_pseudo     : {skipped_no_pseudo}")
    print(f"  anchor table          : {anchor_csv}")
    print(f"  pseudo composition    : {pseudo_csv}")
    print(f"  summary table         : {summary_csv}")


if __name__ == "__main__":
    main()
