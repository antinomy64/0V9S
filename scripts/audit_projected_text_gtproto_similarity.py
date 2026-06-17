#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint import DinoClipJointDataset
from src.dataset_global import CategoryPatchPoolDataset
from src.voc116_part_coarse import FINE_PART_CLASSES, COARSE_PART_CLASSES

try:
    from train_util_global import build_cached_train_gt_prototypes
except Exception:
    from src.train_util_global import build_cached_train_gt_prototypes


def part_names(num_parts: int) -> List[str]:
    if int(num_parts) == 116:
        return list(FINE_PART_CLASSES)
    if int(num_parts) == 59:
        return list(COARSE_PART_CLASSES)
    return [f"part_{i}" for i in range(int(num_parts))]


def parse_run(s: str) -> Tuple[str, str, str]:
    """
    Format: name:model_config:init_weights
    """
    parts = s.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --run {s!r}. Expected format: name:model_config:init_weights"
        )
    return parts[0], parts[1], parts[2]


def load_model(model_config: str, init_weights: str, device: torch.device):
    with open(model_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"].get("model_class", "ProjectionLayer")
    Model = getattr(importlib.import_module("src.model"), model_name)
    model = Model.from_config(cfg["model"]).to(device)

    print(f"[model] load {init_weights}")
    ckpt = torch.load(init_weights, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    msg = model.load_state_dict(ckpt, strict=False)
    print("  missing keys   :", getattr(msg, "missing_keys", []))
    print("  unexpected keys:", getattr(msg, "unexpected_keys", []))
    model.eval()
    return model, cfg


def build_joint_dataset(args, cfg_for_min_area: Dict):
    min_obj_area_ratio = float(
        cfg_for_min_area.get("dataset", {}).get("min_obj_area_ratio", 0.0)
    )
    print(f"[dataset] min_obj_area_ratio={min_obj_area_ratio}")
    return DinoClipJointDataset(
        args.dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=int(args.resize_dim),
        crop_dim=int(args.crop_dim),
        patch_size=int(args.patch_size),
        with_background=False,
        min_obj_area_ratio=min_obj_area_ratio,
    )


def make_category_pool_dataset(joint_dataset, args):
    kwargs = dict(
        sample_patches_per_step=None,  # full pool for GT prototypes
        steps_per_epoch=None,
    )
    sig = inspect.signature(CategoryPatchPoolDataset.__init__)
    if "store_dtype" in sig.parameters:
        kwargs["store_dtype"] = torch.float16
    if "seed" in sig.parameters:
        kwargs["seed"] = int(args.seed)
    if "fixed_subsample" in sig.parameters:
        kwargs["fixed_subsample"] = True

    print("[global pool] building full category patch pools for GT prototypes")
    return CategoryPatchPoolDataset(joint_dataset, **kwargs)


def sort_cache_by_pid(cache: Dict, num_parts: int, device: torch.device):
    raw_text = cache["part_text_feat"].float()
    gt_proto = cache["gt_prototypes"].float()
    pid = cache.get("part_category_id", cache.get("part_ids", None))
    if pid is None:
        raise KeyError("GT prototype cache must contain part_category_id or part_ids")
    pid = pid.long()

    names = part_names(num_parts)
    raw_by_pid = torch.zeros((num_parts, raw_text.shape[-1]), dtype=torch.float32)
    proto_by_pid = torch.zeros((num_parts, gt_proto.shape[-1]), dtype=torch.float32)
    count_by_pid = torch.zeros((num_parts,), dtype=torch.long)
    valid = torch.zeros((num_parts,), dtype=torch.bool)

    # If duplicates ever appear, average them. Normally there is one row per fine part.
    accum_raw = torch.zeros_like(raw_by_pid)
    accum_proto = torch.zeros_like(proto_by_pid)
    seen = torch.zeros((num_parts,), dtype=torch.float32)

    proto_count = cache.get("gt_counts", cache.get("proto_count", None))
    if proto_count is not None:
        proto_count = proto_count.cpu().long()

    for i in range(pid.numel()):
        p = int(pid[i].item())
        if p < 0 or p >= num_parts:
            continue
        accum_raw[p] += raw_text[i].cpu()
        accum_proto[p] += gt_proto[i].cpu()
        seen[p] += 1.0
        if proto_count is not None and i < proto_count.numel():
            count_by_pid[p] += int(proto_count[i].item())

    valid = seen > 0
    raw_by_pid[valid] = accum_raw[valid] / seen[valid, None].clamp_min(1.0)
    proto_by_pid[valid] = accum_proto[valid] / seen[valid, None].clamp_min(1.0)
    raw_by_pid[valid] = F.normalize(raw_by_pid[valid], dim=-1, eps=1e-6)
    proto_by_pid[valid] = F.normalize(proto_by_pid[valid], dim=-1, eps=1e-6)

    if proto_count is None:
        # build_cached_train_gt_prototypes in some versions does not return counts.
        # Keep valid/count info explicit; count=-1 means unknown.
        count_by_pid[valid] = -1

    return {
        "part_text_feat": raw_by_pid.to(device),
        "gt_prototypes": proto_by_pid.to(device),
        "valid": valid.to(device),
        "count": count_by_pid,
        "names": names,
    }


@torch.no_grad()
def compute_one_run(run_name: str, model_config: str, init_weights: str, cache, args, device):
    model, _ = load_model(model_config, init_weights, device)

    part_text = cache["part_text_feat"]
    gt_proto = cache["gt_prototypes"]
    valid = cache["valid"].bool()
    names = cache["names"]
    counts = cache["count"].cpu().tolist()
    num_parts = int(args.num_parts)

    text_proj = model.project_clip_txt(part_text.float())
    text_proj = F.normalize(text_proj.float(), dim=-1, eps=1e-6)
    gt_proto = F.normalize(gt_proto.float(), dim=-1, eps=1e-6)

    sim = text_proj @ gt_proto.T  # [P,P]
    sim_cpu = sim.detach().cpu()
    valid_cpu = valid.detach().cpu()

    long_rows = []
    top_rows = []

    for src in range(num_parts):
        if not bool(valid_cpu[src].item()):
            continue

        row = sim_cpu[src].clone()
        row[~valid_cpu] = float("nan")

        # rank only among valid GT prototypes
        valid_indices = torch.nonzero(valid_cpu, as_tuple=False).squeeze(1)
        valid_scores = row[valid_indices]
        order_local = torch.argsort(valid_scores, descending=True)
        ordered_gt = valid_indices[order_local]
        ordered_scores = valid_scores[order_local]

        rank_map = {int(gt.item()): r + 1 for r, gt in enumerate(ordered_gt)}
        self_cos = float(row[src].item()) if torch.isfinite(row[src]) else float("nan")
        self_rank = rank_map.get(src, -1)

        # best non-self among valid GTs
        nonself_scores = []
        for gt in valid_indices.tolist():
            if int(gt) == src:
                continue
            nonself_scores.append((int(gt), float(row[int(gt)].item())))
        if len(nonself_scores) > 0:
            best_nonself_gt, best_nonself_cos = max(nonself_scores, key=lambda x: x[1])
        else:
            best_nonself_gt, best_nonself_cos = -1, float("nan")

        top = {}
        for k in range(5):
            if k < ordered_gt.numel():
                gt = int(ordered_gt[k].item())
                score = float(ordered_scores[k].item())
                top[f"top{k+1}_gt_pid"] = gt
                top[f"top{k+1}_gt_part"] = names[gt]
                top[f"top{k+1}_cos"] = score
            else:
                top[f"top{k+1}_gt_pid"] = -1
                top[f"top{k+1}_gt_part"] = ""
                top[f"top{k+1}_cos"] = float("nan")

        top_rows.append({
            "run": run_name,
            "source_pid": src,
            "source_part": names[src],
            "gt_token_count": counts[src],
            "self_cos": self_cos,
            "self_rank": self_rank,
            "self_is_top1": bool(self_rank == 1),
            "best_nonself_gt_pid": best_nonself_gt,
            "best_nonself_gt_part": names[best_nonself_gt] if best_nonself_gt >= 0 else "",
            "best_nonself_cos": best_nonself_cos,
            "margin_self_minus_best_nonself": self_cos - best_nonself_cos if best_nonself_gt >= 0 else float("nan"),
            **top,
        })

        for gt in range(num_parts):
            if not bool(valid_cpu[gt].item()):
                continue
            long_rows.append({
                "run": run_name,
                "source_pid": src,
                "source_part": names[src],
                "gt_pid": gt,
                "gt_part": names[gt],
                "cos": float(row[gt].item()),
                "is_self": bool(src == gt),
                "rank_for_source": rank_map.get(gt, -1),
                "gt_token_count": counts[gt],
            })

    # matrix CSV for this run
    matrix_df = pd.DataFrame(sim_cpu.numpy(), index=names, columns=names)
    return pd.DataFrame(long_rows), pd.DataFrame(top_rows), matrix_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", action="append", required=True,
                        help="Repeatable. Format: name:model_config:init_weights")
    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--num_parts", type=int, default=116)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [parse_run(x) for x in args.run]
    # Use the first config only for dataset min_obj_area_ratio. The actual three models
    # are loaded independently below.
    _, first_cfg_path, _ = runs[0]
    with open(first_cfg_path, "r", encoding="utf-8") as f:
        first_cfg = yaml.safe_load(f)

    joint_dataset = build_joint_dataset(args, first_cfg)
    pool_dataset = make_category_pool_dataset(joint_dataset, args)

    raw_cache = build_cached_train_gt_prototypes(
        pool_dataset,
        eps=1e-6,
        chunk_size=int(args.chunk_size),
    )
    cache = sort_cache_by_pid(raw_cache, int(args.num_parts), device)

    meta_rows = []
    for pid, name in enumerate(cache["names"]):
        if bool(cache["valid"].detach().cpu()[pid].item()):
            meta_rows.append({
                "pid": pid,
                "part": name,
                "gt_token_count": int(cache["count"][pid]),
            })
    pd.DataFrame(meta_rows).to_csv(out_dir / "gt_proto_meta.csv", index=False)

    all_long = []
    all_top = []
    for run_name, cfg, weights in runs:
        print("=" * 100)
        print(f"[run] {run_name}")
        long_df, top_df, matrix_df = compute_one_run(run_name, cfg, weights, cache, args, device)
        long_df.to_csv(out_dir / f"{run_name}_projected_text_gtproto_similarity_long.csv", index=False)
        top_df.to_csv(out_dir / f"{run_name}_projected_text_gtproto_top5.csv", index=False)
        matrix_df.to_csv(out_dir / f"{run_name}_projected_text_gtproto_matrix.csv")
        all_long.append(long_df)
        all_top.append(top_df)

    all_long_df = pd.concat(all_long, ignore_index=True)
    all_top_df = pd.concat(all_top, ignore_index=True)
    all_long_df.to_csv(out_dir / "all_runs_projected_text_gtproto_similarity_long.csv", index=False)
    all_top_df.to_csv(out_dir / "all_runs_projected_text_gtproto_top5.csv", index=False)

    # Small JSON summary per run.
    summary = []
    for run_name, _, _ in runs:
        df = all_top_df[all_top_df["run"] == run_name]
        summary.append({
            "run": run_name,
            "num_parts": int(len(df)),
            "top1_acc_self": float(df["self_is_top1"].mean()) if len(df) else float("nan"),
            "mean_self_cos": float(df["self_cos"].mean()) if len(df) else float("nan"),
            "mean_margin": float(df["margin_self_minus_best_nonself"].mean()) if len(df) else float("nan"),
            "mean_self_rank": float(df["self_rank"].replace(-1, float("nan")).mean()) if len(df) else float("nan"),
        })
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[done] outputs written to", out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
