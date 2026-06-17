#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

# Put this file under scripts/ or run it from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint import DinoClipJointDataset
from src.dataset_global import CategoryPatchPoolDataset, global_pool_collate_fn
from src.loss_global import PartLoss


def parse_optional_int(value, *, default):
    if value is None or str(value).lower() in {"config", "cfg"}:
        return default
    s = str(value).strip().lower()
    if s in {"none", "null", "full", "all"}:
        return None
    return int(s)


def load_model(config_file: str, init_weights: str, device: torch.device):
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_class_name = cfg["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(cfg["model"])
    model.to(device)

    if init_weights:
        print(f"[model] load {init_weights}")
        ckpt = torch.load(init_weights, map_location="cpu")
        ret = model.load_state_dict(ckpt, strict=False)
        print("[model] missing keys:", getattr(ret, "missing_keys", []))
        print("[model] unexpected keys:", getattr(ret, "unexpected_keys", []))
    model.eval()
    return model, cfg


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def add_count(counter, src_pid: int, gt_pid: int, value: int):
    counter[(int(src_pid), int(gt_pid))] += int(value)


def build_part_name_map(joint_dataset) -> Dict[int, str]:
    samples = joint_dataset.data.values() if isinstance(joint_dataset.data, dict) else joint_dataset.data
    out: Dict[int, str] = {}
    for sample in samples:
        pids = sample.get("part_category_id", [])
        names = sample.get("part_class_name", [])
        if torch.is_tensor(pids):
            pids = pids.detach().cpu().tolist()
        for pid, name in zip(pids, names):
            out[int(pid)] = str(name)
    return out


def part_name(pid: int, part_name_map: Dict[int, str]) -> str:
    pid = int(pid)
    if pid == -1:
        return "__unlabeled__"
    if pid == -2:
        return "__multi__"
    return part_name_map.get(pid, f"part_{pid}")


def gt_pid_map_from_masks(part_gt_mask_local: torch.Tensor, part_ids_local: torch.Tensor) -> torch.Tensor:
    """Assign each patch to one GT pid for composition counting.

    part_gt_mask_local: [K, M] bool
    part_ids_local:     [K]

    Output: [M]
      -1: no GT part mask covers this patch
      -2: multiple part masks cover this patch
      pid: exactly one GT part mask covers this patch
    """
    K, M = part_gt_mask_local.shape
    device = part_gt_mask_local.device
    out = torch.full((M,), -1, dtype=torch.long, device=device)
    if K == 0 or M == 0:
        return out

    hits = part_gt_mask_local.bool()
    hit_count = hits.long().sum(dim=0)
    single = hit_count == 1
    multi = hit_count > 1
    if single.any():
        first_idx = hits[:, single].float().argmax(dim=0).long()
        out[single] = part_ids_local[first_idx].long()
    if multi.any():
        out[multi] = -2
    return out


@torch.no_grad()
def mine_global_pool_with_assignments(model, criterion: PartLoss, batch: Dict):
    """Mirror src.loss_global.PartLoss.forward/_anchor_proto_em_pool, but keep assignments.

    This is global-pool mining, not image-wise mining:
      batch is one category pool from CategoryPatchPoolDataset/global_pool_collate_fn.

    It intentionally calls criterion._build_part_anchor_mask, _anchor_match_scores,
    and _select_anchor_indices so anchor behavior follows the actual loss_global.py
    implementation/configuration loaded in the current repo.
    """
    patch_tokens_raw = batch["patch_tokens"]
    obj_text_feat = batch["obj_text_feat"]
    part_text_feat = batch["part_text_feat"]
    obj_mask_patch = batch["obj_mask_patch"].bool()
    part_valid_mask = batch["part_valid_mask"].bool()
    part_gt_mask_patch = batch["part_gt_mask_patch"].bool()
    part_category_id = batch["part_category_id"].long()

    part_anchor_mask = criterion._build_part_anchor_mask(
        part_valid_mask=part_valid_mask,
        part_gt_mask_patch=part_gt_mask_patch,
        obj_mask_patch=obj_mask_patch,
    )

    if part_text_feat.shape[1] == 0 or not part_anchor_mask.any() or not obj_mask_patch.any():
        return []

    part_proj = model.project_clip_txt(part_text_feat.float())
    part_proj = criterion._safe_normalize(part_proj, dim=-1)
    patch_tokens = criterion._safe_normalize(patch_tokens_raw.float(), dim=-1)

    abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens)
    abs_logits = abs_logits / float(criterion.patch_temperature)
    abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

    records = []
    B, K, _ = abs_logits.shape
    for b in range(B):
        valid_part_idx = torch.nonzero(part_anchor_mask[b], as_tuple=False).squeeze(1)
        valid_patch_idx_global = torch.nonzero(obj_mask_patch[b], as_tuple=False).squeeze(1)
        Kb = int(valid_part_idx.numel())
        Mb = int(valid_patch_idx_global.numel())
        if Kb == 0 or Mb == 0:
            continue
        if Mb < Kb:
            raise RuntimeError(f"Global pool has fewer valid patches than valid parts: Mb={Mb}, Kb={Kb}")

        valid_patch_tokens = patch_tokens[b, valid_patch_idx_global]                    # [Mb, D]
        local_scores = abs_logits[b, valid_part_idx][:, valid_patch_idx_global]         # [Kb, Mb]
        gt_masks_local = part_gt_mask_patch[b, valid_part_idx][:, valid_patch_idx_global]  # [Kb, Mb]
        part_ids_local = part_category_id[b, valid_part_idx]                            # [Kb]

        # Exact anchor behavior comes from current src.loss_global.PartLoss.
        if hasattr(criterion, "_anchor_match_scores") and hasattr(criterion, "_select_anchor_indices"):
            match_scores = criterion._anchor_match_scores(local_scores)
            anchor_idx_local = criterion._select_anchor_indices(match_scores)
        else:
            # Fallback to joint/classic behavior: relative score + greedy one-to-one.
            rel_scores = criterion._compute_relative_scores(local_scores)
            match_scores = rel_scores
            anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=local_scores.device)
            patch_taken = torch.zeros((Mb,), dtype=torch.bool, device=local_scores.device)
            flat_scores = rel_scores.reshape(-1)
            sorted_idx = torch.argsort(flat_scores, descending=True)
            assigned_parts = 0
            for flat_id in sorted_idx:
                p_local = torch.div(flat_id, Mb, rounding_mode="floor")
                n_local = flat_id % Mb
                if anchor_idx_local[p_local] != -1 or patch_taken[n_local]:
                    continue
                anchor_idx_local[p_local] = n_local
                patch_taken[n_local] = True
                assigned_parts += 1
                if assigned_parts == Kb:
                    break
            unassigned = torch.nonzero(anchor_idx_local < 0, as_tuple=False).squeeze(1)
            if unassigned.numel() > 0:
                local_best = rel_scores.argmax(dim=1)
                anchor_idx_local[unassigned] = local_best[unassigned]

        if (anchor_idx_local < 0).any():
            raise RuntimeError("Anchor selection failed to assign one anchor to each valid part.")

        anchor_idx_global = valid_patch_idx_global[anchor_idx_local]
        anchor_tokens = valid_patch_tokens[anchor_idx_local]

        C = anchor_tokens
        assign = None
        for _ in range(max(int(criterion.em_iters), 1)):
            assign_scores = valid_patch_tokens @ C.T
            assign = assign_scores.argmax(dim=1)
            assign[anchor_idx_local] = torch.arange(Kb, device=assign.device)

            proto_sum = valid_patch_tokens.new_zeros((Kb, valid_patch_tokens.shape[-1]))
            proto_sum.index_add_(0, assign, valid_patch_tokens)
            count = torch.bincount(assign, minlength=Kb).to(valid_patch_tokens.dtype).clamp_min(1.0)
            C = proto_sum / count[:, None]
            C = criterion._safe_normalize(C, dim=-1)

        if assign is None:
            raise RuntimeError("EM assignment was not produced.")

        gt_pid_map = gt_pid_map_from_masks(gt_masks_local, part_ids_local)  # [Mb]

        records.append({
            "b": b,
            "valid_part_idx": valid_part_idx.detach().cpu(),
            "valid_patch_idx_global": valid_patch_idx_global.detach().cpu(),
            "part_ids_local": part_ids_local.detach().cpu(),
            "anchor_idx_local": anchor_idx_local.detach().cpu(),
            "anchor_idx_global": anchor_idx_global.detach().cpu(),
            "assign": assign.detach().cpu(),
            "gt_pid_map": gt_pid_map.detach().cpu(),
            "gt_masks_local": gt_masks_local.detach().cpu(),
            "match_scores_anchor": match_scores[torch.arange(Kb, device=match_scores.device), anchor_idx_local].detach().cpu(),
        })
    return records


def write_csv(path: Path, rows: Iterable[Dict], fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_txt_preview(csv_path: Path, txt_path: Path, max_rows: int = 1000):
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(df.head(max_rows).to_string(index=False))
            f.write("\n")
    except Exception as e:
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(f"Failed to write preview for {csv_path}: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        "Audit GLOBAL-POOL anchor GT labels and pseudo-cluster GT composition."
    )
    parser.add_argument("--run_name", required=True)
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
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", default=None)
    parser.add_argument("--min_obj_area_ratio", type=float, default=None)

    parser.add_argument(
        "--sample_patches_per_step",
        default="config",
        help="config: use YAML train.sample_patches_per_step; none/null/full/all: full pool; or integer.",
    )
    parser.add_argument("--fixed_subsample", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="audits/global_anchor_gt_pseudo_composition")

    # Optional explicit override. If omitted, use actual current loss_global defaults/config behavior.
    parser.add_argument("--anchor_matcher", default=None, choices=[None, "greedy", "hungarian"])
    parser.add_argument("--anchor_score_type", default=None, choices=[None, "relative", "absolute"])
    parser.add_argument("--present_only_parts", action="store_true", default=False)
    parser.add_argument("--em_iters", type=int, default=None)
    parser.add_argument("--patch_temperature", type=float, default=None)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model, cfg = load_model(args.model_config, args.init_weights, device)
    train_cfg = cfg.get("train", {})
    dataset_cfg = cfg.get("dataset", {})

    min_obj_area_ratio = float(args.min_obj_area_ratio if args.min_obj_area_ratio is not None else dataset_cfg.get("min_obj_area_ratio", 0.0))

    print("[dataset] building DinoClipJointDataset")
    joint_dataset = DinoClipJointDataset(
        args.dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=(".tar" in args.dataset),
        path_prefix=args.path_prefix,
        min_obj_area_ratio=min_obj_area_ratio,
    )
    part_name_map = build_part_name_map(joint_dataset)

    cfg_sample = train_cfg.get("sample_patches_per_step", 65536)
    sample_patches_per_step = parse_optional_int(args.sample_patches_per_step, default=cfg_sample)

    print("[global pool audit] sample_patches_per_step=", sample_patches_per_step)
    print("[global pool audit] fixed_subsample=", bool(args.fixed_subsample))
    pool_dataset = CategoryPatchPoolDataset(
        joint_dataset,
        sample_patches_per_step=sample_patches_per_step,
        steps_per_epoch=None,
        store_dtype=torch.float16,
        seed=int(args.seed),
        fixed_subsample=bool(args.fixed_subsample),
    )

    patch_temperature = float(args.patch_temperature if args.patch_temperature is not None else train_cfg.get("patch_temperature", 0.07))
    em_iters = int(args.em_iters if args.em_iters is not None else train_cfg.get("em_iters", 3))
    present_only_anchor = bool(args.present_only_parts or train_cfg.get("present_only_anchor", False))

    criterion_kwargs = dict(
        sim_model=model,
        lambda_inst=0.0,
        lambda_overlap=0.0,
        lambda_spear=0.0,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
        present_only_anchor=present_only_anchor,
    )
    if args.anchor_matcher is not None:
        criterion_kwargs["anchor_matcher"] = args.anchor_matcher
    if args.anchor_score_type is not None:
        criterion_kwargs["anchor_score_type"] = args.anchor_score_type

    criterion = PartLoss(**criterion_kwargs).to(device)
    criterion.eval()

    actual_matcher = getattr(criterion, "anchor_matcher", "unknown")
    actual_score_type = getattr(criterion, "anchor_score_type", "unknown")
    print(f"[criterion] patch_temperature={patch_temperature}, em_iters={em_iters}")
    print(f"[criterion] present_only_anchor={present_only_anchor}")
    print(f"[criterion] anchor_matcher={actual_matcher}, anchor_score_type={actual_score_type}")

    anchor_counts = defaultdict(int)
    anchor_totals = defaultdict(int)
    anchor_self_hits = defaultdict(int)
    pseudo_counts = defaultdict(int)
    pseudo_totals = defaultdict(int)
    pseudo_self_counts = defaultdict(int)

    # One item per category when steps_per_epoch=None.
    for idx in tqdm(range(len(pool_dataset)), desc="global anchor/pseudo composition"):
        item = pool_dataset[idx]
        batch = global_pool_collate_fn([item])
        batch = move_batch_to_device(batch, device)

        recs = mine_global_pool_with_assignments(model, criterion, batch)
        for rec in recs:
            part_ids_local = rec["part_ids_local"].long()              # [Kb]
            anchor_idx_local = rec["anchor_idx_local"].long()          # [Kb]
            assign = rec["assign"].long()                              # [Mb]
            gt_pid_map = rec["gt_pid_map"].long()                      # [Mb]
            gt_masks_local = rec["gt_masks_local"].bool()              # [Kb, Mb]

            Kb = int(part_ids_local.numel())
            for k in range(Kb):
                src_pid = int(part_ids_local[k].item())
                aidx = int(anchor_idx_local[k].item())
                gt_pid = int(gt_pid_map[aidx].item()) if aidx >= 0 else -1

                add_count(anchor_counts, src_pid, gt_pid, 1)
                anchor_totals[src_pid] += 1
                if aidx >= 0 and bool(gt_masks_local[k, aidx].item()):
                    anchor_self_hits[src_pid] += 1

                cluster_mask = assign == k
                total_cluster = int(cluster_mask.long().sum().item())
                if total_cluster <= 0:
                    continue
                pseudo_totals[src_pid] += total_cluster
                pseudo_self_counts[src_pid] += int((gt_masks_local[k] & cluster_mask).long().sum().item())

                vals, cnts = torch.unique(gt_pid_map[cluster_mask], return_counts=True)
                for gt_pid_t, cnt_t in zip(vals.tolist(), cnts.tolist()):
                    add_count(pseudo_counts, src_pid, int(gt_pid_t), int(cnt_t))

        del batch, recs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.run_name

    anchor_rows = []
    for (src_pid, gt_pid), count in sorted(anchor_counts.items(), key=lambda x: (x[0][0], -x[1], x[0][1])):
        total = anchor_totals[src_pid]
        anchor_rows.append({
            "source_pid": src_pid,
            "source_part": part_name(src_pid, part_name_map),
            "anchor_gt_pid": gt_pid,
            "anchor_gt_part": part_name(gt_pid, part_name_map),
            "count": count,
            "total_for_source": total,
            "ratio_in_source": 0.0 if total <= 0 else count / total,
            "is_identity_label": int(src_pid == gt_pid),
        })

    pseudo_rows = []
    for (src_pid, gt_pid), count in sorted(pseudo_counts.items(), key=lambda x: (x[0][0], -x[1], x[0][1])):
        total = pseudo_totals[src_pid]
        pseudo_rows.append({
            "source_pid": src_pid,
            "source_part": part_name(src_pid, part_name_map),
            "pseudo_patch_gt_pid": gt_pid,
            "pseudo_patch_gt_part": part_name(gt_pid, part_name_map),
            "patch_count": count,
            "total_patches_for_source": total,
            "ratio_in_source_pseudo": 0.0 if total <= 0 else count / total,
            "is_identity_label": int(src_pid == gt_pid),
        })

    summary_rows = []
    all_src = sorted(set(anchor_totals.keys()) | set(pseudo_totals.keys()))
    for src_pid in all_src:
        # dominant anchor GT
        a_items = [(gt, c) for (s, gt), c in anchor_counts.items() if s == src_pid]
        a_dom_gt, a_dom_cnt = (-1, 0)
        if a_items:
            a_dom_gt, a_dom_cnt = max(a_items, key=lambda x: x[1])
        a_total = anchor_totals[src_pid]

        # dominant pseudo-cluster GT
        p_items = [(gt, c) for (s, gt), c in pseudo_counts.items() if s == src_pid]
        p_dom_gt, p_dom_cnt = (-1, 0)
        if p_items:
            p_dom_gt, p_dom_cnt = max(p_items, key=lambda x: x[1])
        p_total = pseudo_totals[src_pid]

        summary_rows.append({
            "source_pid": src_pid,
            "source_part": part_name(src_pid, part_name_map),
            "anchor_total": a_total,
            "anchor_self_hit": anchor_self_hits[src_pid],
            "anchor_self_hit_rate": 0.0 if a_total <= 0 else anchor_self_hits[src_pid] / a_total,
            "anchor_dominant_gt_pid": a_dom_gt,
            "anchor_dominant_gt_part": part_name(a_dom_gt, part_name_map),
            "anchor_dominant_count": a_dom_cnt,
            "anchor_dominant_ratio": 0.0 if a_total <= 0 else a_dom_cnt / a_total,
            "pseudo_patch_total": p_total,
            "pseudo_self_patch_count": pseudo_self_counts[src_pid],
            "pseudo_self_patch_ratio": 0.0 if p_total <= 0 else pseudo_self_counts[src_pid] / p_total,
            "pseudo_dominant_gt_pid": p_dom_gt,
            "pseudo_dominant_gt_part": part_name(p_dom_gt, part_name_map),
            "pseudo_dominant_count": p_dom_cnt,
            "pseudo_dominant_ratio": 0.0 if p_total <= 0 else p_dom_cnt / p_total,
        })

    anchor_csv = out_dir / f"{prefix}_anchor_gt_confusion.csv"
    pseudo_csv = out_dir / f"{prefix}_pseudo_cluster_gt_composition.csv"
    summary_csv = out_dir / f"{prefix}_per_part_anchor_pseudo_summary.csv"

    write_csv(anchor_csv, anchor_rows, [
        "source_pid", "source_part", "anchor_gt_pid", "anchor_gt_part", "count",
        "total_for_source", "ratio_in_source", "is_identity_label",
    ])
    write_csv(pseudo_csv, pseudo_rows, [
        "source_pid", "source_part", "pseudo_patch_gt_pid", "pseudo_patch_gt_part",
        "patch_count", "total_patches_for_source", "ratio_in_source_pseudo", "is_identity_label",
    ])
    write_csv(summary_csv, summary_rows, [
        "source_pid", "source_part", "anchor_total", "anchor_self_hit", "anchor_self_hit_rate",
        "anchor_dominant_gt_pid", "anchor_dominant_gt_part", "anchor_dominant_count", "anchor_dominant_ratio",
        "pseudo_patch_total", "pseudo_self_patch_count", "pseudo_self_patch_ratio",
        "pseudo_dominant_gt_pid", "pseudo_dominant_gt_part", "pseudo_dominant_count", "pseudo_dominant_ratio",
    ])

    for p in [anchor_csv, pseudo_csv, summary_csv]:
        write_txt_preview(p, p.with_suffix(".txt"), max_rows=2000)
        print("[saved]", p)
        print("[saved]", p.with_suffix(".txt"))


if __name__ == "__main__":
    main()
