#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from anlysis import (
    DinoClipJointDataset,
    joint_collate_fn,
    COARSE_PART_CLASSES,
    FINE_PART_CLASSES,
)


def parse_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def get_part_names(num_parts: int) -> List[str]:
    if num_parts == 59:
        return list(COARSE_PART_CLASSES)
    if num_parts == 116:
        return list(FINE_PART_CLASSES)
    return [f"part_{i}" for i in range(num_parts)]


def object_name_from_part_name(part_name: str) -> str:
    part_name = str(part_name)
    if "'s " in part_name:
        return part_name.split("'s ", 1)[0]
    if "’s " in part_name:
        return part_name.split("’s ", 1)[0]
    return ""


def get_no_part_objects(num_parts: int) -> set[str]:
    if num_parts == 59:
        part_names = list(COARSE_PART_CLASSES)
    elif num_parts == 116:
        part_names = list(FINE_PART_CLASSES)
    else:
        return set()

    objects_with_parts = {
        object_name_from_part_name(x)
        for x in part_names
        if object_name_from_part_name(x)
    }

    voc20_objects = {
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor",
    }

    return voc20_objects - objects_with_parts


def get_class_name(metadata: Any, b: int, fallback: str) -> str:
    if isinstance(metadata, (list, tuple)) and b < len(metadata):
        meta = metadata[b]
        if isinstance(meta, dict):
            return str(meta.get("class_name", fallback))
    return fallback


@torch.no_grad()
def collect_object_patch_tokens_and_gt_labels(args):
    """
    Returns:
        pools[cat] = {
            "category_id": int,
            "class_name": str,
            "tokens": Tensor [M, D],
            "gt_part_ids": LongTensor [M],
        }

    gt_part_ids[m]:
        -1 means no GT part label for this patch.
        >=0 means coarse/fine part id.
    """
    dataset = DinoClipJointDataset(
        args.dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=False,
        is_wds=".tar" in args.dataset,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=args.min_obj_area_ratio,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
        pin_memory=False,
    )

    store_dtype = parse_dtype(args.store_dtype)

    token_chunks: Dict[int, List[torch.Tensor]] = defaultdict(list)
    gt_chunks: Dict[int, List[torch.Tensor]] = defaultdict(list)
    class_names: Dict[int, str] = {}

    for batch in tqdm(loader, desc="collect object patch tokens + GT part labels"):
        patch_tokens = batch["patch_tokens"]                 # [B, N, D]
        obj_mask = batch["obj_mask_patch"].bool()            # [B, N]
        category_id = batch["category_id"].long()            # [B]

        part_gt = batch["part_gt_mask_patch"].bool()         # [B, K, N]
        part_valid = batch["part_valid_mask"].bool()         # [B, K]
        part_ids = batch["part_category_id"].long()          # [B, K]

        metadata = batch.get("metadata", None)

        B, N, _ = patch_tokens.shape
        K = part_gt.shape[1]

        for b in range(B):
            cat = int(category_id[b].item())

            if args.keep_all_crop_patches:
                valid_patch_mask = torch.ones(N, dtype=torch.bool)
            else:
                valid_patch_mask = obj_mask[b].cpu().bool()

            if valid_patch_mask.sum().item() == 0:
                continue

            gt_patch_ids = torch.full((N,), -1, dtype=torch.long)

            for k in range(K):
                if not bool(part_valid[b, k].item()):
                    continue

                pid = int(part_ids[b, k].item())
                if pid < 0 or pid >= args.num_parts:
                    continue

                m = part_gt[b, k].cpu().bool() & obj_mask[b].cpu().bool()
                gt_patch_ids[m] = pid

            toks = patch_tokens[b][valid_patch_mask].detach().cpu().to(store_dtype).contiguous()
            gt_ids = gt_patch_ids[valid_patch_mask].contiguous()

            token_chunks[cat].append(toks)
            gt_chunks[cat].append(gt_ids)

            class_names.setdefault(
                cat,
                get_class_name(metadata, b, fallback=f"class_{cat}"),
            )

    pools = {}

    for cat in sorted(token_chunks.keys()):
        tokens = torch.cat(token_chunks[cat], dim=0).contiguous()
        gt_ids = torch.cat(gt_chunks[cat], dim=0).contiguous()

        pools[cat] = {
            "category_id": cat,
            "class_name": class_names.get(cat, f"class_{cat}"),
            "tokens": tokens,
            "gt_part_ids": gt_ids,
        }

        valid_gt = int((gt_ids >= 0).sum().item())
        print(
            f"[pool] cat={cat:02d}, class={pools[cat]['class_name']}, "
            f"tokens={tokens.shape[0]}, valid_gt={valid_gt}, "
            f"no_gt={tokens.shape[0] - valid_gt}"
        )

    return pools


@torch.no_grad()
def run_simple_spherical_kmeans(
    tokens_cpu: torch.Tensor,
    *,
    num_centroids: int,
    kmeans_iters: int,
    device: torch.device,
    seed: int,
):
    """
    tokens_cpu: [M, D], CPU tensor.

    Returns:
        assign_cpu: [M]
        centers_cpu: [C, D]
        counts_cpu: [C]
    """
    x = tokens_cpu.float().to(device)
    x = F.normalize(x, dim=-1, eps=1e-6)

    M, D = x.shape
    C = min(int(num_centroids), int(M))

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    perm = torch.randperm(M, device=device, generator=g)[:C]
    centers = x[perm].clone()

    for it in range(int(kmeans_iters)):
        sim = x @ centers.T
        assign = sim.argmax(dim=1)

        counts = torch.bincount(assign, minlength=C).float().to(device)

        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, assign, x)
        new_centers = new_centers / counts.clamp_min(1.0)[:, None]

        empty = counts <= 0
        if empty.any():
            empty_ids = torch.nonzero(empty, as_tuple=False).squeeze(1)
            rand_ids = torch.randint(0, M, (empty_ids.numel(),), device=device)
            new_centers[empty_ids] = x[rand_ids]

        centers = F.normalize(new_centers, dim=-1, eps=1e-6)

        print(
            f"[iter {it:03d}] "
            f"min_count={int(counts.min().item())}, "
            f"max_count={int(counts.max().item())}, "
            f"mean_count={float(counts.mean().item()):.2f}"
        )

        del sim, new_centers, counts

    sim = x @ centers.T
    assign = sim.argmax(dim=1)
    counts = torch.bincount(assign, minlength=C).long()

    assign_cpu = assign.detach().cpu().long()
    centers_cpu = centers.detach().cpu().float()
    counts_cpu = counts.detach().cpu().long()

    del x, centers, sim, assign, counts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return assign_cpu, centers_cpu, counts_cpu


def entropy_from_counts(counts: torch.Tensor) -> float:
    total = counts.sum().item()
    if total <= 0:
        return 0.0
    p = counts.float() / float(total)
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum().item())


def format_top_parts(
    pids: torch.Tensor,
    counts: torch.Tensor,
    part_names: List[str],
    topk: int,
) -> str:
    if pids.numel() == 0:
        return "none"

    order = torch.argsort(counts, descending=True)
    total = int(counts.sum().item())

    items = []
    for j in range(min(topk, int(order.numel()))):
        pid = int(pids[order[j]].item())
        cnt = int(counts[order[j]].item())
        ratio = cnt / max(total, 1)

        if 0 <= pid < len(part_names):
            name = part_names[pid]
        else:
            name = f"part_{pid}"

        # 不再用 "|"，避免和表格分隔符冲突
        items.append(f"{name}:{cnt}({ratio:.3f})")

    return "; ".join(items)


@torch.no_grad()
def print_cluster_gt_purity(
    *,
    class_name: str,
    assign_cpu: torch.Tensor,
    gt_part_ids: torch.Tensor,
    num_centroids: int,
    part_names: List[str],
    topk: int,
):
    """
    Full-cluster GT part purity analysis.
    No truncation is applied.
    """
    print("")
    print(f"[purity table] object={class_name}")

    header_fmt = (
        "{cluster:>7}  "
        "{size:>10}  "
        "{valid_gt:>10}  "
        "{no_gt:>8}  "
        "{top1_part:<28}  "
        "{top1_ratio:>8}  "
        "{entropy:>8}  "
        "{top_parts}"
    )

    row_fmt = (
        "{cluster:>7}  "
        "{size:>10}  "
        "{valid_gt:>10}  "
        "{no_gt:>8.3f}  "
        "{top1_part:<28}  "
        "{top1_ratio:>8.3f}  "
        "{entropy:>8.3f}  "
        "{top_parts}"
    )

    print(
        header_fmt.format(
            cluster="cluster",
            size="size",
            valid_gt="valid_gt",
            no_gt="no_gt",
            top1_part="top1_part",
            top1_ratio="top1",
            entropy="entropy",
            top_parts="top_parts",
        )
    )

    print("-" * 150)

    object_valid_total = 0
    object_top1_total = 0
    weighted_entropy_sum = 0.0

    for c in range(num_centroids):
        idx = torch.nonzero(assign_cpu == c, as_tuple=False).squeeze(1)
        size = int(idx.numel())

        if size == 0:
            print(
                row_fmt.format(
                    cluster=f"{c:03d}",
                    size=0,
                    valid_gt=0,
                    no_gt=1.0,
                    top1_part="none",
                    top1_ratio=0.0,
                    entropy=0.0,
                    top_parts="none",
                )
            )
            continue

        cluster_gt = gt_part_ids[idx].long()
        valid = cluster_gt >= 0
        valid_gt = cluster_gt[valid]

        valid_count = int(valid_gt.numel())
        no_gt_ratio = 1.0 - valid_count / max(size, 1)

        if valid_count == 0:
            print(
                row_fmt.format(
                    cluster=f"{c:03d}",
                    size=size,
                    valid_gt=0,
                    no_gt=no_gt_ratio,
                    top1_part="none",
                    top1_ratio=0.0,
                    entropy=0.0,
                    top_parts="none",
                )
            )
            continue

        pids, pcnts = torch.unique(valid_gt, return_counts=True)
        order = torch.argsort(pcnts, descending=True)

        top_pid = int(pids[order[0]].item())
        top_cnt = int(pcnts[order[0]].item())
        top_ratio = top_cnt / max(valid_count, 1)

        entropy = entropy_from_counts(pcnts)

        if 0 <= top_pid < len(part_names):
            top_name = part_names[top_pid]
        else:
            top_name = f"part_{top_pid}"

        top_parts_str = format_top_parts(pids, pcnts, part_names, topk=topk)

        object_valid_total += valid_count
        object_top1_total += top_cnt
        weighted_entropy_sum += entropy * valid_count

        print(
            row_fmt.format(
                cluster=f"{c:03d}",
                size=size,
                valid_gt=valid_count,
                no_gt=no_gt_ratio,
                top1_part=top_name[:28],
                top1_ratio=top_ratio,
                entropy=entropy,
                top_parts=top_parts_str,
            )
        )

    object_purity = object_top1_total / max(object_valid_total, 1)
    object_entropy = weighted_entropy_sum / max(object_valid_total, 1)

    print("-" * 150)
    print(
        f"[object summary] object={class_name}, "
        f"valid_gt={object_valid_total}, "
        f"weighted_top1_purity={object_purity:.4f}, "
        f"weighted_entropy={object_entropy:.4f}"
    )

    return {
        "valid_gt": object_valid_total,
        "top1_total": object_top1_total,
        "weighted_entropy_sum": weighted_entropy_sum,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--store_dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--min_obj_area_ratio", type=float, default=0.0)
    parser.add_argument("--keep_all_crop_patches", action="store_true", default=False)
    parser.add_argument("--path_prefix", default=None)

    parser.add_argument("--num_parts", type=int, default=59)
    parser.add_argument("--num_centroids", type=int, default=64)
    parser.add_argument("--kmeans_iters", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk_parts", type=int, default=3)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
    )

    part_names = get_part_names(args.num_parts)
    no_part_objects = get_no_part_objects(args.num_parts)

    print("[device]", device)
    print("[dataset]", args.dataset)
    print("[num_parts]", args.num_parts)
    print("[num_centroids]", args.num_centroids)
    print("[kmeans_iters]", args.kmeans_iters)
    print("[no_part_objects]", sorted(list(no_part_objects)))

    pools = collect_object_patch_tokens_and_gt_labels(args)

    global_valid_total = 0
    global_top1_total = 0
    global_entropy_sum = 0.0

    kept_class_names = []
    kept_category_ids = []

    for cat in sorted(pools.keys()):
        item = pools[cat]
        class_name = str(item["class_name"])

        if class_name in no_part_objects:
            print(f"[skip] cat={cat:02d}, class={class_name}, no parts")
            continue

        tokens_cpu = item["tokens"]
        gt_part_ids = item["gt_part_ids"]

        M, D = tokens_cpu.shape
        C = min(int(args.num_centroids), int(M))

        print("")
        print("=" * 100)
        print(
            f"[kmeans] cat={cat:02d}, class={class_name}, "
            f"tokens=({M}, {D}), centroids={C}"
        )

        assign_cpu, centers_cpu, counts_cpu = run_simple_spherical_kmeans(
            tokens_cpu,
            num_centroids=args.num_centroids,
            kmeans_iters=args.kmeans_iters,
            device=device,
            seed=args.seed + int(cat),
        )

        print(
            f"[done object] class={class_name}, "
            f"min_cluster={int(counts_cpu.min().item())}, "
            f"max_cluster={int(counts_cpu.max().item())}, "
            f"mean_cluster={float(counts_cpu.float().mean().item()):.2f}"
        )

        summary = print_cluster_gt_purity(
            class_name=class_name,
            assign_cpu=assign_cpu,
            gt_part_ids=gt_part_ids,
            num_centroids=C,
            part_names=part_names,
            topk=args.topk_parts,
        )

        global_valid_total += int(summary["valid_gt"])
        global_top1_total += int(summary["top1_total"])
        global_entropy_sum += float(summary["weighted_entropy_sum"])

        kept_class_names.append(class_name)
        kept_category_ids.append(int(cat))

        del assign_cpu, centers_cpu, counts_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_purity = global_top1_total / max(global_valid_total, 1)
    global_entropy = global_entropy_sum / max(global_valid_total, 1)

    print("")
    print("=" * 100)
    print("[global summary]")
    print("kept_category_ids:", kept_category_ids)
    print("kept_class_names:", kept_class_names)
    print("global_valid_gt:", global_valid_total)
    print("global_weighted_top1_purity:", f"{global_purity:.4f}")
    print("global_weighted_entropy:", f"{global_entropy:.4f}")


if __name__ == "__main__":
    main()