#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global raw point-set alignment with one shared orthogonal W.

This is the raw-only counterpart of train_global_dectx_pointset_orthogonal_w.py.
The text-to-point-cloud single-anchor logic is unchanged; only the frozen
decontext transform G and the decontext patch pool are removed.

For object class o:
  T_o : all frozen projected part-text features, [K_o, D]
  X_o : every normalized raw foreground patch from the train set, [N_o, D]
  W   : one shared trainable orthogonal matrix, [D, D]

At each epoch:
  Q_o      = normalize(T_o @ W)
  anchor_p = argmax_i cos(Q_o[p], X_o[i])

The single global objective is:
  L(W) = mean_{all objects and all valid parts}
         [1 - cos(Q_o[p], X_o[anchor_p])]

All objects and parts share W. Hard anchors are recomputed at every epoch and
held fixed during the gradient update. W is projected back to the orthogonal
group with SVD after every optimizer step.

GT part masks are used only for anchor-hit auditing when --audit_gt true.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=dim, eps=eps)


def parse_int_set(text: Optional[str]) -> Optional[set[int]]:
    if text is None or str(text).strip() == "":
        return None
    return {int(item.strip()) for item in str(text).split(",") if item.strip()}


def parse_name_set(text: Optional[str]) -> Optional[set[str]]:
    if text is None or str(text).strip() == "":
        return None
    value = str(text).strip()
    if value.lower() in {"all", "*"}:
        return None
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def move_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, non_blocking=True)


def batch_value_at(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        item = value[index]
        if item.ndim == 0:
            return item.item()
        return item
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def metadata_value(
    batch: Dict[str, Any],
    key: str,
    index: int,
    default: Any = None,
) -> Any:
    if key in batch:
        try:
            return batch_value_at(batch[key], index)
        except Exception:
            pass

    metadata = batch.get("metadata")
    if isinstance(metadata, (list, tuple)) and index < len(metadata):
        item = metadata[index]
        if isinstance(item, dict) and key in item:
            return item[key]
    if isinstance(metadata, dict) and key in metadata:
        try:
            return batch_value_at(metadata[key], index)
        except Exception:
            return metadata[key]
    return default


def as_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


def as_str(value: Any) -> str:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return str(value.item())
        return str(value.detach().cpu().tolist())
    return str(value)


def pool_dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(name).strip().lower()
    if key not in table:
        raise ValueError(
            f"unsupported --pool_dtype={name}; choose float16/bfloat16/float32"
        )
    return table[key]


def load_stage1_model(
    config_path: str,
    weights_path: str,
    device: torch.device,
):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    model_class = getattr(importlib.import_module("src.model"), model_class_name)
    model = model_class.from_config(config["model"])

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    result = model.load_state_dict(state_dict, strict=False)
    print(f"[projector] loaded={weights_path}")
    print("[projector] missing keys:", getattr(result, "missing_keys", []))
    print("[projector] unexpected keys:", getattr(result, "unexpected_keys", []))

    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config


def build_dataset(path: str, args: argparse.Namespace):
    kwargs = {
        "obj_feature_name": args.obj_feature_name,
        "part_feature_name": args.part_feature_name,
        "obj_text_name": args.obj_text_name,
        "part_text_name": args.part_text_name,
        "resize_dim": args.resize_dim,
        "crop_dim": args.crop_dim,
        "patch_size": args.patch_size,
        "with_background": False,
        "is_wds": ".tar" in path,
        "path_prefix": args.path_prefix,
        "min_obj_area_ratio": 0.0,
    }
    signature = inspect.signature(DinoClipJointDataset)
    filtered = {
        key: value for key, value in kwargs.items()
        if key in signature.parameters
    }
    return DinoClipJointDataset(path, **filtered)


def build_loader(dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.build_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
        pin_memory=True,
        drop_last=False,
    )


@dataclass
class CpuPoolAccumulator:
    category_id: int
    class_name: str
    raw_chunks: List[torch.Tensor]
    gt_chunks: List[torch.Tensor]
    text_z0: Optional[torch.Tensor] = None
    part_ids: Optional[torch.Tensor] = None
    num_instances: int = 0


@dataclass
class GpuClassPool:
    category_id: int
    class_name: str
    raw: torch.Tensor
    gt_part_ids: Optional[torch.Tensor]
    text_z0: torch.Tensor
    part_ids: torch.Tensor
    num_instances: int

    @property
    def num_patches(self) -> int:
        return int(self.raw.shape[0])

    @property
    def num_parts(self) -> int:
        return int(self.text_z0.shape[0])


def class_is_selected(
    category_id: int,
    class_name: str,
    selected_ids: Optional[set[int]],
    selected_names: Optional[set[str]],
) -> bool:
    if selected_ids is not None and category_id not in selected_ids:
        return False
    if selected_names is not None and class_name.lower() not in selected_names:
        return False
    return True


def make_gt_patch_ids(
    batch: Dict[str, Any],
    batch_index: int,
    foreground_mask: torch.Tensor,
    valid_part_mask: torch.Tensor,
    part_ids: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Audit-only: give each foreground patch one GT part id, or -1."""
    if "part_gt_mask_patch" not in batch:
        return None

    gt = batch["part_gt_mask_patch"][batch_index].detach().cpu().bool()
    if gt.ndim != 2:
        gt = gt.reshape(gt.shape[0], -1)

    foreground_mask = foreground_mask.detach().cpu().bool().reshape(-1)
    if gt.shape[1] != foreground_mask.numel():
        return None

    valid_indices = torch.nonzero(
        valid_part_mask, as_tuple=False
    ).flatten()
    foreground_indices = torch.nonzero(
        foreground_mask, as_tuple=False
    ).flatten()

    labels = torch.full(
        (foreground_indices.numel(),),
        -1,
        dtype=torch.int64,
    )
    if foreground_indices.numel() == 0:
        return labels

    for local_index in valid_indices.tolist():
        part_id = int(part_ids[local_index].item())
        covered = gt[local_index].index_select(0, foreground_indices)
        fill = covered & (labels < 0)
        labels[fill] = part_id

    return labels


@torch.no_grad()
def build_global_pools(
    projector,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> List[GpuClassPool]:
    pool_dtype = pool_dtype_from_name(args.pool_dtype)
    selected_ids = parse_int_set(args.category_ids)
    selected_names = parse_name_set(args.class_names)

    accumulators: Dict[int, CpuPoolAccumulator] = {}
    total_collected_patches = 0

    pbar = tqdm(
        loader,
        desc="build raw global point pools",
        dynamic_ncols=True,
    )

    for batch in pbar:
        patch_batch = batch["patch_tokens"]
        object_mask_batch = batch["obj_mask_patch"].bool()
        part_text_batch = move_tensor(batch["part_text_feat"], device)

        projected_text_batch = normalize(
            projector.project_clip_txt(part_text_batch.float()),
            dim=-1,
        )

        prepared: List[
            Tuple[
                int,
                int,
                str,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []
        raw_foreground_list: List[torch.Tensor] = []

        batch_size = int(patch_batch.shape[0])
        for b in range(batch_size):
            category_id = as_int(
                metadata_value(batch, "category_id", b, -1)
            )
            class_name = as_str(
                metadata_value(batch, "class_name", b, category_id)
            )
            if not class_is_selected(
                category_id,
                class_name,
                selected_ids,
                selected_names,
            ):
                continue

            foreground_mask = (
                object_mask_batch[b]
                .detach()
                .cpu()
                .bool()
                .reshape(-1)
            )
            foreground_index = torch.nonzero(
                foreground_mask, as_tuple=False
            ).flatten()
            if foreground_index.numel() == 0:
                continue

            raw_foreground = (
                patch_batch[b]
                .detach()
                .cpu()
                .index_select(0, foreground_index)
                .contiguous()
            )

            valid_mask = (
                batch["part_valid_mask"][b]
                .detach()
                .cpu()
                .bool()
                .reshape(-1)
            )
            all_part_ids = (
                batch["part_category_id"][b]
                .detach()
                .cpu()
                .long()
                .reshape(-1)
            )
            valid_index = torch.nonzero(
                valid_mask, as_tuple=False
            ).flatten()
            if valid_index.numel() == 0:
                continue

            text_z0 = (
                projected_text_batch[b]
                .detach()
                .cpu()
                .index_select(0, valid_index)
                .float()
            )
            part_ids_valid = all_part_ids.index_select(
                0, valid_index
            ).long()

            gt_ids = None
            if args.audit_gt:
                gt_ids = make_gt_patch_ids(
                    batch=batch,
                    batch_index=b,
                    foreground_mask=foreground_mask,
                    valid_part_mask=valid_mask,
                    part_ids=all_part_ids,
                )

            prepared.append(
                (
                    b,
                    category_id,
                    class_name,
                    raw_foreground,
                    text_z0,
                    part_ids_valid,
                    (
                        gt_ids
                        if gt_ids is not None
                        else torch.empty(0, dtype=torch.int64)
                    ),
                )
            )
            raw_foreground_list.append(raw_foreground)

        if not prepared:
            continue

        raw_all = torch.cat(
            raw_foreground_list, dim=0
        ).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        raw_all_norm = normalize(raw_all, dim=-1)

        offset = 0
        for (
            _,
            category_id,
            class_name,
            raw_cpu,
            text_z0,
            part_ids_valid,
            gt_ids,
        ) in prepared:
            count = int(raw_cpu.shape[0])
            raw_norm_cpu = (
                raw_all_norm[offset: offset + count]
                .to(dtype=pool_dtype)
                .cpu()
                .contiguous()
            )
            offset += count

            if category_id not in accumulators:
                accumulators[category_id] = CpuPoolAccumulator(
                    category_id=category_id,
                    class_name=class_name,
                    raw_chunks=[],
                    gt_chunks=[],
                )

            accumulator = accumulators[category_id]
            accumulator.raw_chunks.append(raw_norm_cpu)
            total_collected_patches += count

            if args.audit_gt:
                if gt_ids.numel() == count:
                    accumulator.gt_chunks.append(gt_ids.contiguous())
                else:
                    accumulator.gt_chunks.append(
                        torch.full(
                            (count,),
                            -1,
                            dtype=torch.int64,
                        )
                    )

            accumulator.num_instances += 1

            if accumulator.text_z0 is None:
                accumulator.text_z0 = text_z0.contiguous()
                accumulator.part_ids = part_ids_valid.contiguous()
            else:
                if not torch.equal(
                    accumulator.part_ids,
                    part_ids_valid,
                ):
                    raise ValueError(
                        f"part taxonomy mismatch inside class "
                        f"{class_name}: "
                        f"first={accumulator.part_ids.tolist()}, "
                        f"current={part_ids_valid.tolist()}"
                    )

                max_diff = float(
                    (
                        accumulator.text_z0 - text_z0
                    ).abs().max().item()
                )
                if max_diff > args.text_consistency_tolerance:
                    raise ValueError(
                        f"projected part text differs across "
                        f"instances for class={class_name}; "
                        f"max_abs_diff={max_diff:.6g}. "
                        "Expected one fixed text point per part."
                    )

        pbar.set_postfix(
            classes=len(accumulators),
            patches=total_collected_patches,
        )
        del raw_all, raw_all_norm

    if not accumulators:
        raise RuntimeError(
            "No object pools were built. "
            "Check feature keys and class filters."
        )

    print("[pool] moving every selected raw object pool to GPU")
    gpu_pools: List[GpuClassPool] = []
    total_bytes = 0

    for category_id in sorted(accumulators):
        accumulator = accumulators[category_id]
        if (
            accumulator.text_z0 is None
            or accumulator.part_ids is None
        ):
            continue

        raw_cpu = torch.cat(
            accumulator.raw_chunks, dim=0
        ).contiguous()

        gt_cpu: Optional[torch.Tensor] = None
        if args.audit_gt and accumulator.gt_chunks:
            gt_cpu = torch.cat(
                accumulator.gt_chunks, dim=0
            ).contiguous()

        raw_gpu = raw_cpu.to(
            device=device,
            non_blocking=True,
        )
        gt_gpu = (
            gt_cpu.to(device=device, non_blocking=True)
            if gt_cpu is not None
            else None
        )
        text_gpu = accumulator.text_z0.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        part_ids_gpu = accumulator.part_ids.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )

        pool = GpuClassPool(
            category_id=category_id,
            class_name=accumulator.class_name,
            raw=raw_gpu,
            gt_part_ids=gt_gpu,
            text_z0=text_gpu,
            part_ids=part_ids_gpu,
            num_instances=accumulator.num_instances,
        )
        gpu_pools.append(pool)

        class_bytes = tensor_bytes(raw_gpu)
        if gt_gpu is not None:
            class_bytes += tensor_bytes(gt_gpu)
        total_bytes += class_bytes

        print(
            f"[pool] {category_id}:{pool.class_name} "
            f"instances={pool.num_instances} "
            f"parts={pool.num_parts} "
            f"patches={pool.num_patches} "
            f"resident={human_bytes(class_bytes)}"
        )

        accumulator.raw_chunks.clear()
        accumulator.gt_chunks.clear()
        del raw_cpu, gt_cpu

    print(
        f"[pool] total classes={len(gpu_pools)}, "
        f"parts={sum(pool.num_parts for pool in gpu_pools)}, "
        f"patches={sum(pool.num_patches for pool in gpu_pools)}, "
        f"GPU resident={human_bytes(total_bytes)}"
    )
    if device.type == "cuda":
        print(
            f"[cuda] allocated="
            f"{human_bytes(torch.cuda.memory_allocated(device))}, "
            f"reserved="
            f"{human_bytes(torch.cuda.memory_reserved(device))}"
        )

    return gpu_pools


@torch.no_grad()
def orthogonalize_(W: torch.Tensor) -> None:
    u, _, vh = torch.linalg.svd(
        W.data.float(),
        full_matrices=False,
    )
    W.data.copy_((u @ vh).to(dtype=W.dtype))


@torch.no_grad()
def orthogonal_stats(W: torch.Tensor) -> Dict[str, float]:
    weight = W.float()
    identity = torch.eye(
        weight.shape[0],
        device=weight.device,
        dtype=weight.dtype,
    )
    return {
        "identity_rms": float(
            (weight - identity)
            .pow(2)
            .mean()
            .sqrt()
            .item()
        ),
        "orthogonality_rms": float(
            (
                weight.t() @ weight - identity
            )
            .pow(2)
            .mean()
            .sqrt()
            .item()
        ),
    }


@torch.no_grad()
def full_pool_argmax(
    query: torch.Tensor,
    pool: torch.Tensor,
    search_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact nearest-neighbour search over the full raw pool."""
    query_for_matmul = query.to(dtype=pool.dtype)
    num_points = int(pool.shape[0])

    if num_points <= 0:
        raise ValueError("cannot search an empty raw patch pool")

    if (
        search_chunk_size <= 0
        or search_chunk_size >= num_points
    ):
        similarity = query_for_matmul @ pool.t()
        best_value, best_index = similarity.max(dim=1)
        return best_index.long(), best_value.float()

    best_value = torch.full(
        (query.shape[0],),
        -float("inf"),
        device=query.device,
        dtype=torch.float32,
    )
    best_index = torch.zeros(
        (query.shape[0],),
        device=query.device,
        dtype=torch.long,
    )

    for start in range(
        0,
        num_points,
        search_chunk_size,
    ):
        stop = min(
            start + search_chunk_size,
            num_points,
        )
        similarity = (
            query_for_matmul @ pool[start:stop].t()
        )
        local_value, local_index = similarity.max(dim=1)
        local_value = local_value.float()
        improve = local_value > best_value
        best_value[improve] = local_value[improve]
        best_index[improve] = (
            local_index[improve].long() + start
        )

    return best_index, best_value


@dataclass
class SelectedClassTargets:
    pool: GpuClassPool
    anchor_index: torch.Tensor
    target_raw: torch.Tensor
    search_similarity: torch.Tensor
    raw_similarity_before_update: torch.Tensor
    gt_hit: Optional[torch.Tensor]
    anchor_stability: float
    duplicate_rate: float


@torch.no_grad()
def select_global_targets(
    pools: Sequence[GpuClassPool],
    W: torch.Tensor,
    args: argparse.Namespace,
    previous_anchors: Optional[
        Dict[int, torch.Tensor]
    ],
) -> Tuple[
    List[SelectedClassTargets],
    Dict[int, torch.Tensor],
]:
    selected: List[SelectedClassTargets] = []
    current_anchors: Dict[int, torch.Tensor] = {}

    for pool in pools:
        query_raw = normalize(
            pool.text_z0 @ W.detach(),
            dim=-1,
        )

        anchor_index, search_similarity = (
            full_pool_argmax(
                query=query_raw,
                pool=pool.raw,
                search_chunk_size=args.search_chunk_size,
            )
        )

        target_raw = (
            pool.raw
            .index_select(0, anchor_index)
            .float()
            .detach()
        )
        raw_similarity = (
            query_raw * target_raw
        ).sum(dim=-1)

        previous = (
            None
            if previous_anchors is None
            else previous_anchors.get(pool.category_id)
        )
        if (
            previous is not None
            and previous.shape == anchor_index.shape
        ):
            anchor_stability = float(
                (
                    previous.to(anchor_index.device)
                    == anchor_index
                )
                .float()
                .mean()
                .item()
            )
        else:
            anchor_stability = float("nan")

        duplicate_rate = (
            1.0
            - float(torch.unique(anchor_index).numel())
            / float(max(anchor_index.numel(), 1))
        )

        gt_hit: Optional[torch.Tensor] = None
        if pool.gt_part_ids is not None:
            selected_gt = pool.gt_part_ids.index_select(
                0,
                anchor_index,
            )
            gt_hit = selected_gt.eq(pool.part_ids)

        selected.append(
            SelectedClassTargets(
                pool=pool,
                anchor_index=anchor_index,
                target_raw=target_raw,
                search_similarity=search_similarity,
                raw_similarity_before_update=raw_similarity,
                gt_hit=gt_hit,
                anchor_stability=anchor_stability,
                duplicate_rate=duplicate_rate,
            )
        )
        current_anchors[pool.category_id] = (
            anchor_index.detach().cpu()
        )

    return selected, current_anchors


def global_alignment_loss(
    selected: Sequence[SelectedClassTargets],
    W: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_sum = W.new_tensor(0.0)
    similarity_sum = W.new_tensor(0.0)
    num_parts = 0

    for item in selected:
        query_raw = normalize(
            item.pool.text_z0 @ W,
            dim=-1,
        )
        selected_similarity = (
            query_raw * item.target_raw
        ).sum(dim=-1)

        loss_sum = (
            loss_sum
            + (1.0 - selected_similarity).sum()
        )
        similarity_sum = (
            similarity_sum
            + selected_similarity.sum()
        )
        num_parts += int(selected_similarity.numel())

    if num_parts == 0:
        raise RuntimeError(
            "No valid part text points were available "
            "for the global loss."
        )

    loss = loss_sum / float(num_parts)
    return loss, {
        "num_parts": float(num_parts),
        "mean_raw_selected_similarity": float(
            (
                similarity_sum.detach()
                / float(num_parts)
            ).item()
        ),
    }


def summarize_selection(
    selected: Sequence[SelectedClassTargets],
) -> Dict[str, float]:
    total_parts = sum(
        item.pool.num_parts for item in selected
    )
    if total_parts <= 0:
        return {}

    search_sum = sum(
        float(item.search_similarity.sum().item())
        for item in selected
    )
    raw_sum = sum(
        float(
            item.raw_similarity_before_update
            .sum()
            .item()
        )
        for item in selected
    )
    duplicate_weighted = sum(
        item.duplicate_rate * item.pool.num_parts
        for item in selected
    )

    stability_numerator = 0.0
    stability_denominator = 0
    for item in selected:
        if np.isfinite(item.anchor_stability):
            stability_numerator += (
                item.anchor_stability
                * item.pool.num_parts
            )
            stability_denominator += (
                item.pool.num_parts
            )

    gt_hits = 0
    gt_count = 0
    for item in selected:
        if item.gt_hit is not None:
            gt_hits += int(item.gt_hit.sum().item())
            gt_count += int(item.gt_hit.numel())

    return {
        "mean_search_similarity": (
            search_sum / float(total_parts)
        ),
        "mean_raw_similarity_before_update": (
            raw_sum / float(total_parts)
        ),
        "duplicate_anchor_rate": (
            duplicate_weighted / float(total_parts)
        ),
        "anchor_stability": (
            stability_numerator
            / float(stability_denominator)
            if stability_denominator > 0
            else float("nan")
        ),
        "gt_anchor_hit_rate": (
            float(gt_hits) / float(gt_count)
            if gt_count > 0
            else float("nan")
        ),
        "gt_anchor_count": float(gt_count),
    }


def class_selection_records(
    selected: Sequence[SelectedClassTargets],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for item in selected:
        selected_gt = None
        if item.pool.gt_part_ids is not None:
            selected_gt = (
                item.pool.gt_part_ids
                .index_select(0, item.anchor_index)
                .detach()
                .cpu()
            )

        for local_part in range(item.pool.num_parts):
            record: Dict[str, Any] = {
                "category_id": item.pool.category_id,
                "class_name": item.pool.class_name,
                "part_category_id": int(
                    item.pool.part_ids[local_part]
                    .detach()
                    .cpu()
                    .item()
                ),
                "anchor_pool_index": int(
                    item.anchor_index[local_part]
                    .detach()
                    .cpu()
                    .item()
                ),
                "search_similarity": float(
                    item.search_similarity[local_part]
                    .detach()
                    .cpu()
                    .item()
                ),
                "raw_similarity_before_update": float(
                    item.raw_similarity_before_update[
                        local_part
                    ]
                    .detach()
                    .cpu()
                    .item()
                ),
            }

            if selected_gt is not None:
                selected_id = int(
                    selected_gt[local_part].item()
                )
                target_id = int(
                    item.pool.part_ids[local_part]
                    .detach()
                    .cpu()
                    .item()
                )
                record["selected_gt_part_id"] = selected_id
                record["gt_hit"] = (
                    selected_id == target_id
                )

            records.append(record)

    return records


def load_initial_W(
    path: Optional[str],
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    if path is None:
        return torch.eye(
            dim,
            device=device,
            dtype=torch.float32,
        )

    checkpoint = torch.load(path, map_location="cpu")
    if torch.is_tensor(checkpoint):
        W = checkpoint
    elif isinstance(checkpoint, dict):
        W = checkpoint.get("W")
        if (
            W is None
            and isinstance(
                checkpoint.get("state_dict"),
                dict,
            )
        ):
            W = checkpoint["state_dict"].get("W")
    else:
        W = None

    if not torch.is_tensor(W):
        raise KeyError(
            f"Could not find tensor W in --init_w={path}"
        )
    if tuple(W.shape) != (dim, dim):
        raise ValueError(
            f"init W shape={tuple(W.shape)}, "
            f"expected={(dim, dim)}"
        )

    W = W.float().to(device)
    u, _, vh = torch.linalg.svd(
        W,
        full_matrices=False,
    )
    print(
        f"[W] initialized from={path} "
        "and projected to orthogonal"
    )
    return u @ vh


def save_checkpoint(
    path: str,
    W: torch.Tensor,
    epoch: int,
    args: argparse.Namespace,
    history: Sequence[Dict[str, Any]],
    pools: Sequence[GpuClassPool],
    selected: Sequence[SelectedClassTargets],
) -> None:
    torch.save(
        {
            "W": W.detach().cpu(),
            "epoch": int(epoch),
            "args": vars(args),
            "history": list(history),
            "classes": [
                {
                    "category_id": pool.category_id,
                    "class_name": pool.class_name,
                    "num_instances": pool.num_instances,
                    "num_parts": pool.num_parts,
                    "num_patches": pool.num_patches,
                    "part_ids": pool.part_ids.detach().cpu(),
                }
                for pool in pools
            ],
            "selected_anchors": (
                class_selection_records(selected)
            ),
            "definition": {
                "query": (
                    "normalize(projected_part_text @ W)"
                ),
                "anchor": (
                    "argmax cosine over every normalized "
                    "raw foreground patch in the object's "
                    "full train pool"
                ),
                "target": (
                    "the selected normalized raw patch token"
                ),
                "loss": (
                    "mean over all object parts of "
                    "1 - cosine(query, selected_raw_anchor)"
                ),
                "orthogonality": (
                    "SVD projection after every optimizer step"
                ),
            },
        },
        path,
    )


def train(
    pools: Sequence[GpuClassPool],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    dim = int(pools[0].text_z0.shape[-1])

    for pool in pools:
        if (
            int(pool.text_z0.shape[-1]) != dim
            or int(pool.raw.shape[-1]) != dim
        ):
            raise ValueError(
                f"dimension mismatch in {pool.class_name}: "
                f"text={pool.text_z0.shape[-1]}, "
                f"raw={pool.raw.shape[-1]}, "
                f"expected={dim}"
            )

    W = nn.Parameter(
        load_initial_W(args.init_w, dim, device)
    )
    optimizer = torch.optim.Adam(
        [W],
        lr=args.lr,
        weight_decay=0.0,
    )

    history: List[Dict[str, Any]] = []
    previous_anchors: Optional[
        Dict[int, torch.Tensor]
    ] = None

    metrics_path = os.path.join(
        args.out_dir,
        "training_metrics.csv",
    )
    fieldnames = [
        "epoch",
        "loss",
        "mean_search_similarity",
        "mean_raw_similarity_before_update",
        "mean_raw_selected_similarity",
        "duplicate_anchor_rate",
        "anchor_stability",
        "gt_anchor_hit_rate",
        "gt_anchor_count",
        "identity_rms",
        "orthogonality_rms",
        "num_parts",
    ]

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        csv.DictWriter(
            file,
            fieldnames=fieldnames,
        ).writeheader()

    selected, previous_anchors = select_global_targets(
        pools=pools,
        W=W,
        args=args,
        previous_anchors=None,
    )
    with torch.no_grad():
        baseline_loss, baseline_info = (
            global_alignment_loss(selected, W)
        )
    baseline_summary = summarize_selection(selected)
    baseline_stats = orthogonal_stats(W)

    print(
        f"[initial] "
        f"loss={float(baseline_loss.item()):.6f} "
        f"search_sim="
        f"{baseline_summary['mean_search_similarity']:.6f} "
        f"raw_sim="
        f"{baseline_info['mean_raw_selected_similarity']:.6f} "
        f"dup="
        f"{baseline_summary['duplicate_anchor_rate']:.6f} "
        f"gt_hit="
        f"{baseline_summary['gt_anchor_hit_rate']:.6f} "
        f"W_id={baseline_stats['identity_rms']:.6e} "
        f"W_orth="
        f"{baseline_stats['orthogonality_rms']:.6e}"
    )

    for epoch in range(1, args.epochs + 1):
        selected, current_anchors = (
            select_global_targets(
                pools=pools,
                W=W,
                args=args,
                previous_anchors=previous_anchors,
            )
        )
        selection_summary = summarize_selection(
            selected
        )

        optimizer.zero_grad(set_to_none=True)
        loss, loss_info = global_alignment_loss(
            selected,
            W,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at epoch={epoch}: "
                f"{loss}"
            )

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [W],
                max_norm=args.grad_clip,
            )

        optimizer.step()
        orthogonalize_(W)

        stats = orthogonal_stats(W)
        record: Dict[str, Any] = {
            "epoch": epoch,
            "loss": float(loss.detach().item()),
            **selection_summary,
            **loss_info,
            **stats,
        }
        history.append(record)
        previous_anchors = current_anchors

        print(
            f"[train {epoch:03d}] "
            f"loss={record['loss']:.6f} "
            f"search_sim="
            f"{record['mean_search_similarity']:.6f} "
            f"raw_sim="
            f"{record['mean_raw_selected_similarity']:.6f} "
            f"dup="
            f"{record['duplicate_anchor_rate']:.6f} "
            f"stability="
            f"{record['anchor_stability']:.6f} "
            f"gt_hit="
            f"{record['gt_anchor_hit_rate']:.6f} "
            f"W_id={record['identity_rms']:.6e} "
            f"W_orth="
            f"{record['orthogonality_rms']:.6e}"
        )

        with open(
            metrics_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writerow(
                {
                    key: record.get(key, "")
                    for key in fieldnames
                }
            )

        if args.save_every_epoch:
            save_checkpoint(
                path=os.path.join(
                    args.out_dir,
                    (
                        "global_raw_pointset_W_"
                        f"epoch_{epoch:03d}.pt"
                    ),
                ),
                W=W,
                epoch=epoch,
                args=args,
                history=history,
                pools=pools,
                selected=selected,
            )

    final_selected, _ = select_global_targets(
        pools=pools,
        W=W,
        args=args,
        previous_anchors=previous_anchors,
    )
    final_path = os.path.join(
        args.out_dir,
        "global_raw_pointset_W_final.pt",
    )

    save_checkpoint(
        path=final_path,
        W=W,
        epoch=args.epochs,
        args=args,
        history=history,
        pools=pools,
        selected=final_selected,
    )

    with open(
        os.path.join(args.out_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "args": vars(args),
                "initial": {
                    "loss": float(
                        baseline_loss.item()
                    ),
                    **baseline_summary,
                    **baseline_info,
                    **baseline_stats,
                },
                "final": (
                    history[-1]
                    if history
                    else None
                ),
                "num_classes": len(pools),
                "num_parts": sum(
                    pool.num_parts for pool in pools
                ),
                "num_patches": sum(
                    pool.num_patches for pool in pools
                ),
                "output_W": final_path,
                "metrics_csv": metrics_path,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[done] W={final_path}")
    print(f"[done] metrics={metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Train one shared orthogonal W by registering "
        "each object's projected part-text points directly "
        "to its full normalized raw foreground patch pool."
    )

    parser.add_argument("--model_config", required=True)
    parser.add_argument(
        "--weights",
        required=True,
        help="Frozen object-level Stage1 projector checkpoint.",
    )
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument(
        "--obj_feature_name",
        default="avg_self_attn_out",
    )
    parser.add_argument(
        "--part_feature_name",
        default="cropaug_patch_tokens",
    )
    parser.add_argument(
        "--obj_text_name",
        default="ann_feats",
    )
    parser.add_argument(
        "--part_text_name",
        default="part_ann_feats",
    )
    parser.add_argument(
        "--resize_dim",
        type=int,
        default=448,
    )
    parser.add_argument(
        "--crop_dim",
        type=int,
        default=448,
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=14,
    )
    parser.add_argument("--path_prefix", default=None)

    parser.add_argument(
        "--class_names",
        default="all",
    )
    parser.add_argument(
        "--category_ids",
        default=None,
    )
    parser.add_argument(
        "--build_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pool_dtype",
        default="float16",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
    )
    parser.add_argument(
        "--search_chunk_size",
        type=int,
        default=0,
        help=(
            "0 computes each KxN raw-pool similarity "
            "matrix in one GPU matmul. Positive values "
            "perform exact chunked argmax."
        ),
    )
    parser.add_argument(
        "--audit_gt",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--text_consistency_tolerance",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--init_w",
        default=None,
    )
    parser.add_argument(
        "--save_every_epoch",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but "
            "torch.cuda.is_available() is False"
        )

    device = torch.device(args.device)

    print(
        "[definition] one shared W for every "
        "object and every part"
    )
    print(
        "[definition] Q=normalize(projected_text@W), "
        "W is SVD-projected orthogonal"
    )
    print(
        "[definition] each discrete part-text point "
        "searches the object's full raw patch cloud"
    )
    print(
        "[definition] one argmax raw anchor per part; "
        "no decontext transform"
    )
    print(
        "[definition] one global mean raw-space cosine "
        "loss; one optimizer update per epoch"
    )

    projector, _ = load_stage1_model(
        args.model_config,
        args.weights,
        device,
    )

    dataset = build_dataset(
        args.train_dataset,
        args,
    )
    loader = build_loader(dataset, args)
    print(f"[dataset] samples={len(dataset)}")

    pools = build_global_pools(
        projector=projector,
        loader=loader,
        device=device,
        args=args,
    )

    train(
        pools=pools,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()
