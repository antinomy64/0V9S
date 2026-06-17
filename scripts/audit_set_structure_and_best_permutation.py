from __future__ import annotations

import argparse
import csv
import importlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn
from src.loss_joint import JointObjPartLoss


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_projector(config: Dict, checkpoint: str, device: torch.device) -> torch.nn.Module:
    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"])

    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    ret = model.load_state_dict(ckpt, strict=False)
    print(f"[model] loaded checkpoint: {checkpoint}")
    print(f"[model] missing_keys={getattr(ret, 'missing_keys', [])}")
    print(f"[model] unexpected_keys={getattr(ret, 'unexpected_keys', [])}")

    model.to(device)
    model.eval()
    return model


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_name_maps(joint_dataset: DinoClipJointDataset) -> Tuple[Dict[int, str], Dict[int, str]]:
    cat_id_to_name: Dict[int, str] = {}
    part_id_to_name: Dict[int, str] = {}

    samples = joint_dataset.data.values() if isinstance(joint_dataset.data, dict) else joint_dataset.data
    for sample in samples:
        if "category_id" in sample:
            cat_id_to_name[int(sample.get("category_id", -1))] = str(sample.get("class_name", ""))
        pids = sample.get("part_category_id", [])
        pnames = sample.get("part_class_name", [])
        if torch.is_tensor(pids):
            pids = pids.detach().cpu().tolist()
        for pid, pname in zip(pids, pnames):
            part_id_to_name[int(pid)] = str(pname)

    for _, bank in getattr(joint_dataset, "class_part_bank", {}).items():
        pids = bank.get("part_ids", [])
        pnames = bank.get("part_names", [])
        if torch.is_tensor(pids):
            pids = pids.detach().cpu().tolist()
        for pid, pname in zip(pids, pnames):
            part_id_to_name[int(pid)] = str(pname)

    return cat_id_to_name, part_id_to_name


def part_name(pid: int, part_id_to_name: Dict[int, str]) -> str:
    return part_id_to_name.get(int(pid), f"part_{int(pid)}")


def cat_name(cid: int, cat_id_to_name: Dict[int, str]) -> str:
    name = cat_id_to_name.get(int(cid), "")
    return name if name else f"cat_{int(cid)}"


def make_joint_criterion(config: Dict, model: torch.nn.Module) -> JointObjPartLoss:
    train_cfg = config.get("train", {})
    criterion = JointObjPartLoss(
        model,
        obj_ltype=train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce")),
        obj_margin=float(train_cfg.get("margin", 0.2)),
        obj_max_violation=bool(train_cfg.get("max_violation", True)),
        lambda_obj=float(train_cfg.get("lambda_obj", 1.0)),
        lambda_inst=float(train_cfg.get("lambda_inst", 0.2)),
        lambda_overlap=float(train_cfg.get("lambda_overlap", 0.05)),
        lambda_spear=float(train_cfg.get("lambda_spear", 0.0)),
        patch_temperature=float(train_cfg.get("patch_temperature", 0.07)),
        em_iters=int(train_cfg.get("em_iters", 3)),
        present_only_anchor=bool(train_cfg.get("present_only_anchor", False)),
    )
    criterion.present_only_anchor = bool(train_cfg.get("present_only_anchor", False))
    criterion.eval()
    return criterion


def make_global_criterion(config: Dict, model: torch.nn.Module):
    from src.loss_global import PartLoss

    train_cfg = config.get("train", {})
    criterion = PartLoss(
        model,
        lambda_inst=float(train_cfg.get("lambda_inst", 0.2)),
        lambda_overlap=float(train_cfg.get("lambda_overlap", 0.05)),
        lambda_spear=float(train_cfg.get("lambda_spear", 0.0)),
        topk_ratio=float(train_cfg.get("topk_ratio", 0.1)),
        patch_temperature=float(train_cfg.get("patch_temperature", 0.07)),
        em_iters=int(train_cfg.get("em_iters", 3)),
        present_only_anchor=bool(train_cfg.get("present_only_anchor", False)),
        anchor_matcher=str(train_cfg.get("anchor_matcher", "greedy")),
        anchor_score_type=str(train_cfg.get("anchor_score_type", "relative")),
    )
    criterion.present_only_anchor = bool(train_cfg.get("present_only_anchor", False))
    criterion.eval()
    return criterion


def rankdata_average_ties(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    n = a.size
    if n == 0:
        return np.asarray([], dtype=np.float64)

    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(n, dtype=np.int64)
    inv[sorter] = np.arange(n)
    sorted_a = a[sorter]

    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * ((i + 1) + j)
        i = j
    return ranks_sorted[inv]


def spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")
    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cosine_matrix_np(x: np.ndarray, y: np.ndarray | None = None, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    if y is None:
        y = x
    else:
        y = np.asarray(y, dtype=np.float64)
        y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), eps)
    return x @ y.T


def upper_triangle_values(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    if n < 2:
        return np.asarray([], dtype=np.float64)
    return mat[np.triu_indices(n, k=1)]


def structure_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return spearman_np(
        upper_triangle_values(cosine_matrix_np(x)),
        upper_triangle_values(cosine_matrix_np(y)),
    )


def best_assignment_dp(score: np.ndarray) -> Tuple[List[int], float]:
    score = np.asarray(score, dtype=np.float64)
    n, m = score.shape
    if n != m:
        raise ValueError(f"best_assignment_dp requires square matrix, got {score.shape}")

    size = 1 << n
    neg = -1e100
    dp = np.full(size, neg, dtype=np.float64)
    dp[0] = 0.0
    parent_j_layers = []
    parent_mask_layers = []

    for i in range(n):
        new_dp = np.full(size, neg, dtype=np.float64)
        parent_j = np.full(size, -1, dtype=np.int64)
        parent_mask = np.full(size, -1, dtype=np.int64)

        for mask in range(size):
            if dp[mask] <= neg / 2 or mask.bit_count() != i:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                val = dp[mask] + score[i, j]
                if val > new_dp[new_mask]:
                    new_dp[new_mask] = val
                    parent_j[new_mask] = j
                    parent_mask[new_mask] = mask

        dp = new_dp
        parent_j_layers.append(parent_j)
        parent_mask_layers.append(parent_mask)

    full = size - 1
    assignment = [-1] * n
    mask = full
    for i in range(n - 1, -1, -1):
        j = int(parent_j_layers[i][mask])
        assignment[i] = j
        mask = int(parent_mask_layers[i][mask])

    return assignment, float(dp[full])


class ProtoStore:
    def __init__(self):
        self.data: Dict[Tuple[int, int], Dict[str, List]] = defaultdict(dict)

    def add(self, cat_id: int, part_id: int, source: str, vec: torch.Tensor):
        key = (int(cat_id), int(part_id))
        vec = safe_normalize(vec.detach().float().cpu(), dim=-1)
        if source not in self.data[key]:
            self.data[key][source] = [torch.zeros_like(vec), 0]
        self.data[key][source][0] += vec
        self.data[key][source][1] += 1

    def as_object_sets(self):
        by_cat: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = defaultdict(dict)
        for (cat_id, pid), srcs in self.data.items():
            out_srcs = {}
            for src, (s, c) in srcs.items():
                if c <= 0:
                    continue
                out_srcs[src] = safe_normalize((s / c).float(), dim=-1)
            by_cat[int(cat_id)][int(pid)] = out_srcs
        return by_cat


def gt_proto_for_slot(
    patch_tokens_b: torch.Tensor,
    obj_mask_b: torch.Tensor,
    part_gt_mask_b: torch.Tensor,
    k: int,
) -> torch.Tensor | None:
    mask = part_gt_mask_b[k].bool() & obj_mask_b.bool()
    if not bool(mask.any().item()):
        return None
    return safe_normalize(patch_tokens_b[mask].mean(dim=0), dim=-1)


@torch.no_grad()
def collect_common_text_gt_sources(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    store: ProtoStore,
    max_units: int,
):
    """
    Collect raw_text, projected_text and GT from the ordinary joint dataset.

    This is deliberately independent of pseudo mining. It uses only:
      part_valid_mask AND actual GT part presence inside object mask.
    """
    device = next(model.parameters()).device

    for unit_idx, batch in enumerate(loader):
        if max_units >= 0 and unit_idx >= max_units:
            break

        batch = move_batch_to_device(batch, device)

        patch_tokens = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask_patch = batch["obj_mask_patch"].bool()
        part_gt_mask_patch = batch["part_gt_mask_patch"].bool()
        part_valid_mask = batch["part_valid_mask"].bool()
        part_category_id = batch["part_category_id"].long()
        raw_text = safe_normalize(batch["part_text_feat"].float(), dim=-1)
        proj_text = safe_normalize(model.project_clip_txt(raw_text), dim=-1)
        category_id = batch["category_id"].long()

        B, K, _ = raw_text.shape
        for b in range(B):
            cat_id = int(category_id[b].item())
            for k in range(K):
                if not bool(part_valid_mask[b, k].item()):
                    continue
                pid = int(part_category_id[b, k].item())
                if pid < 0:
                    continue
                gt = gt_proto_for_slot(
                    patch_tokens_b=patch_tokens[b],
                    obj_mask_b=obj_mask_patch[b],
                    part_gt_mask_b=part_gt_mask_patch[b],
                    k=k,
                )
                if gt is None:
                    continue

                store.add(cat_id, pid, "raw_text", raw_text[b, k])
                store.add(cat_id, pid, "projected_text", proj_text[b, k])
                store.add(cat_id, pid, "gt", gt)


@torch.no_grad()
def collect_joint_pseudo(
    *,
    model: torch.nn.Module,
    criterion: JointObjPartLoss,
    loader: DataLoader,
    store: ProtoStore,
    max_units: int,
):
    device = next(model.parameters()).device

    for unit_idx, batch in enumerate(loader):
        if max_units >= 0 and unit_idx >= max_units:
            break

        batch = move_batch_to_device(batch, device)

        patch_tokens = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask_patch = batch["obj_mask_patch"].bool()
        part_gt_mask_patch = batch["part_gt_mask_patch"].bool()
        part_valid_mask = batch["part_valid_mask"].bool()
        part_category_id = batch["part_category_id"].long()
        raw_text = safe_normalize(batch["part_text_feat"].float(), dim=-1)
        category_id = batch["category_id"].long()

        if raw_text.shape[1] == 0 or not bool(part_valid_mask.any().item()):
            continue

        if getattr(criterion, "present_only_anchor", False):
            present = (part_gt_mask_patch & obj_mask_patch[:, None, :]).sum(dim=-1) > 0
            part_anchor_mask = part_valid_mask & present
        else:
            part_anchor_mask = part_valid_mask
        part_anchor_mask = part_anchor_mask & obj_mask_patch.any(dim=-1, keepdim=True)

        if not bool(part_anchor_mask.any().item()):
            continue

        proj_text = safe_normalize(model.project_clip_txt(raw_text), dim=-1)
        abs_logits = torch.einsum("bkd,bnd->bkn", proj_text, patch_tokens)
        abs_logits = abs_logits / float(criterion.patch_temperature)
        abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        z_pseudo, _, _ = criterion._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_anchor_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=int(criterion.em_iters),
        )
        z_pseudo = safe_normalize(z_pseudo.float(), dim=-1)

        B, K, _ = proj_text.shape
        for b in range(B):
            cat_id = int(category_id[b].item())
            for k in range(K):
                if not bool(part_anchor_mask[b, k].item()):
                    continue
                pid = int(part_category_id[b, k].item())
                if pid < 0:
                    continue
                # Require GT target presence so pseudo is comparable to GT target set.
                gt = gt_proto_for_slot(
                    patch_tokens_b=patch_tokens[b],
                    obj_mask_b=obj_mask_patch[b],
                    part_gt_mask_b=part_gt_mask_patch[b],
                    k=k,
                )
                if gt is None:
                    continue
                store.add(cat_id, pid, "pseudo", z_pseudo[b, k])


@torch.no_grad()
def collect_global_pseudo(
    *,
    model: torch.nn.Module,
    criterion,
    pool_loader: DataLoader,
    store: ProtoStore,
    max_units: int,
):
    device = next(model.parameters()).device

    for unit_idx, batch in enumerate(pool_loader):
        if max_units >= 0 and unit_idx >= max_units:
            break

        batch = move_batch_to_device(batch, device)

        patch_tokens = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask_patch = batch["obj_mask_patch"].bool()
        part_gt_mask_patch = batch["part_gt_mask_patch"].bool()
        part_valid_mask = batch["part_valid_mask"].bool()
        part_category_id = batch["part_category_id"].long()
        raw_text = safe_normalize(batch["part_text_feat"].float(), dim=-1)
        category_id = batch["category_id"].long()

        if raw_text.shape[1] == 0 or not bool(part_valid_mask.any().item()):
            continue

        part_anchor_mask = criterion._build_part_anchor_mask(
            part_valid_mask=part_valid_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            obj_mask_patch=obj_mask_patch,
        )
        if not bool(part_anchor_mask.any().item()):
            continue

        proj_text = safe_normalize(model.project_clip_txt(raw_text), dim=-1)
        abs_logits = torch.einsum("bkd,bnd->bkn", proj_text, patch_tokens)
        abs_logits = abs_logits / float(criterion.patch_temperature)
        abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

        z_pseudo, _, _ = criterion._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_anchor_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=int(criterion.em_iters),
        )
        z_pseudo = safe_normalize(z_pseudo.float(), dim=-1)

        B, K, _ = proj_text.shape
        for b in range(B):
            cat_id = int(category_id[b].item())
            for k in range(K):
                if not bool(part_anchor_mask[b, k].item()):
                    continue
                pid = int(part_category_id[b, k].item())
                if pid < 0:
                    continue
                # In global pool, target presence is part of anchor mask if present_only_anchor
                # or all class parts may be present. We still add only if corresponding common
                # GT source will exist in final merge.
                store.add(cat_id, pid, "pseudo", z_pseudo[b, k])


def compute_object_metrics(by_cat, cat_id_to_name, part_id_to_name, label: str):
    summary_rows = []
    mapping_rows = []

    for cat_id in sorted(by_cat.keys()):
        part_dict = by_cat[cat_id]

        pids = sorted([
            pid for pid, srcs in part_dict.items()
            if all(s in srcs for s in ("raw_text", "projected_text", "pseudo", "gt"))
        ])
        if len(pids) < 2:
            continue

        obj_name = cat_name(cat_id, cat_id_to_name)
        raw = torch.stack([part_dict[pid]["raw_text"] for pid in pids], dim=0).numpy()
        proj = torch.stack([part_dict[pid]["projected_text"] for pid in pids], dim=0).numpy()
        pseudo = torch.stack([part_dict[pid]["pseudo"] for pid in pids], dim=0).numpy()
        gt = torch.stack([part_dict[pid]["gt"] for pid in pids], dim=0).numpy()

        row = {
            "label": label,
            "cat_id": cat_id,
            "object_name": obj_name,
            "num_parts": len(pids),
            "spearman_raw_text_vs_gt_structure": structure_spearman(raw, gt) if len(pids) >= 3 else float("nan"),
            "spearman_projected_text_vs_gt_structure": structure_spearman(proj, gt) if len(pids) >= 3 else float("nan"),
            "spearman_pseudo_vs_gt_structure": structure_spearman(pseudo, gt) if len(pids) >= 3 else float("nan"),
            "spearman_pseudo_vs_projected_text_structure": structure_spearman(pseudo, proj) if len(pids) >= 3 else float("nan"),
        }

        for query_name, query_mat in [("projected_text", proj), ("pseudo", pseudo)]:
            if query_mat.shape[1] != gt.shape[1]:
                continue

            sim = cosine_matrix_np(query_mat, gt)
            diag = np.diag(sim)
            top1 = np.argmax(sim, axis=1)
            assignment, _ = best_assignment_dp(sim)
            assignment = np.asarray(assignment, dtype=np.int64)
            best_cos = sim[np.arange(len(pids)), assignment]
            identity = np.arange(len(pids))

            sim_no_diag = sim.copy()
            sim_no_diag[identity, identity] = -1e9

            row[f"{query_name}_named_top1_acc"] = float(np.mean(top1 == identity))
            row[f"{query_name}_named_mean_self_cos"] = float(np.mean(diag))
            row[f"{query_name}_named_mean_margin"] = float(np.mean(diag - sim_no_diag.max(axis=1)))
            row[f"{query_name}_bestperm_identity_acc"] = float(np.mean(assignment == identity))
            row[f"{query_name}_bestperm_mean_cos"] = float(np.mean(best_cos))
            row[f"{query_name}_bestperm_gain_vs_named"] = float(np.mean(best_cos) - np.mean(diag))
            row[f"{query_name}_top1_unique_ratio"] = float(len(set(top1.tolist())) / len(pids))
            row[f"{query_name}_top1_max_collision_frac"] = float(max(np.bincount(top1, minlength=len(pids))) / len(pids))

            chunks = []
            for i, pid in enumerate(pids):
                gt_j = int(assignment[i])
                gt_pid = int(pids[gt_j])
                chunks.append(f"{part_name(pid, part_id_to_name)}->{part_name(gt_pid, part_id_to_name)}({sim[i, gt_j]:.4f})")

                mapping_rows.append({
                    "label": label,
                    "cat_id": cat_id,
                    "object_name": obj_name,
                    "query_type": query_name,
                    "query_part_id": int(pid),
                    "query_part_name": part_name(pid, part_id_to_name),
                    "named_gt_part_id": int(pid),
                    "named_gt_part_name": part_name(pid, part_id_to_name),
                    "named_cos": float(diag[i]),
                    "top1_gt_part_id": int(pids[int(top1[i])]),
                    "top1_gt_part_name": part_name(int(pids[int(top1[i])]), part_id_to_name),
                    "top1_cos": float(sim[i, int(top1[i])]),
                    "bestperm_gt_part_id": gt_pid,
                    "bestperm_gt_part_name": part_name(gt_pid, part_id_to_name),
                    "bestperm_cos": float(sim[i, gt_j]),
                    "bestperm_is_identity": bool(gt_j == i),
                })

            row[f"{query_name}_bestperm_mapping"] = " ; ".join(chunks)

        summary_rows.append(row)

    return summary_rows, mapping_rows


def write_report(path: Path, rows: List[Dict], label: str):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"[V2 object-internal set-level + best-permutation audit] label={label}\n")
        f.write("=" * 120 + "\n")
        f.write("Correction note: raw_text, projected_text, and GT are collected with a common present-GT mask, independent of pseudo mining.\n")
        f.write("Therefore raw_text_vs_gt_structure should be identical across runs if dataset/filtering is identical.\n\n")

        for r in rows:
            f.write("-" * 120 + "\n")
            f.write(f"[object] {r['object_name']}  cat={r['cat_id']}  parts={r['num_parts']}\n")
            f.write(
                "structure Spearman: "
                f"rawT-GT={r.get('spearman_raw_text_vs_gt_structure', float('nan')):.4f}, "
                f"projT-GT={r.get('spearman_projected_text_vs_gt_structure', float('nan')):.4f}, "
                f"pseudo-GT={r.get('spearman_pseudo_vs_gt_structure', float('nan')):.4f}, "
                f"pseudo-projT={r.get('spearman_pseudo_vs_projected_text_structure', float('nan')):.4f}\n"
            )
            for q in ("projected_text", "pseudo"):
                if f"{q}_bestperm_mean_cos" not in r:
                    continue
                f.write(
                    f"{q}: "
                    f"named_top1={r[f'{q}_named_top1_acc']:.4f}, "
                    f"named_self_cos={r[f'{q}_named_mean_self_cos']:.4f}, "
                    f"named_margin={r[f'{q}_named_mean_margin']:.4f}, "
                    f"bestperm_identity={r[f'{q}_bestperm_identity_acc']:.4f}, "
                    f"bestperm_mean_cos={r[f'{q}_bestperm_mean_cos']:.4f}, "
                    f"bestperm_gain={r[f'{q}_bestperm_gain_vs_named']:.4f}, "
                    f"top1_unique_ratio={r[f'{q}_top1_unique_ratio']:.4f}, "
                    f"top1_max_collision_frac={r[f'{q}_top1_max_collision_frac']:.4f}\n"
                )
                f.write(f"{q} bestperm mapping: {r[f'{q}_bestperm_mapping']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["joint", "global"], required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out_dir", default="audits/set_structure_matching_v2")

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_units", type=int, default=-1)
    parser.add_argument("--global_sample_patches", type=int, default=0, help="0 means full global category pool.")

    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_config(args.model_config)
    dataset_cfg = config.get("dataset", {})
    min_obj_area_ratio = float(dataset_cfg.get("min_obj_area_ratio", 0.0))

    print(f"[audit v2] label={args.label}")
    print(f"[audit v2] pipeline={args.pipeline}")
    print(f"[audit v2] model_config={args.model_config}")
    print(f"[audit v2] checkpoint={args.checkpoint}")
    print(f"[audit v2] dataset={args.dataset}")
    print(f"[audit v2] min_obj_area_ratio={min_obj_area_ratio}")

    model = load_projector(config, args.checkpoint, device)

    is_wds = ".tar" in args.dataset
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
        is_wds=is_wds,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=min_obj_area_ratio,
    )
    cat_id_to_name, part_id_to_name = build_name_maps(joint_dataset)

    print(f"[dataset] taxonomy={getattr(joint_dataset, 'part_taxonomy', 'unknown')}")
    print(f"[dataset] num_samples={len(joint_dataset)}")
    print(f"[dataset] num_object_classes={len(getattr(joint_dataset, 'class_part_bank', {}))}")

    store = ProtoStore()

    # 1) Common collection: raw_text/projected_text/GT, independent of pipeline.
    common_loader = DataLoader(
        joint_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=joint_collate_fn,
    )
    print("[collect] common raw_text/projected_text/GT from joint dataset")
    collect_common_text_gt_sources(
        model=model,
        loader=common_loader,
        store=store,
        max_units=int(args.max_units),
    )

    # 2) Pseudo collection: pipeline-specific.
    print(f"[collect] pseudo via pipeline={args.pipeline}")
    if args.pipeline == "joint":
        criterion = make_joint_criterion(config, model).to(device)
        pseudo_loader = DataLoader(
            joint_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=joint_collate_fn,
        )
        collect_joint_pseudo(
            model=model,
            criterion=criterion,
            loader=pseudo_loader,
            store=store,
            max_units=int(args.max_units),
        )
    else:
        from src.dataset_global import CategoryPatchPoolDataset, global_pool_collate_fn

        criterion = make_global_criterion(config, model).to(device)
        sample_patches = None if int(args.global_sample_patches) <= 0 else int(args.global_sample_patches)
        pool_dataset = CategoryPatchPoolDataset(
            joint_dataset,
            sample_patches_per_step=sample_patches,
            steps_per_epoch=None,
            store_dtype=torch.float16,
            seed=123,
            fixed_subsample=True,
        )
        pool_loader = DataLoader(
            pool_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=global_pool_collate_fn,
        )
        collect_global_pseudo(
            model=model,
            criterion=criterion,
            pool_loader=pool_loader,
            store=store,
            max_units=int(args.max_units),
        )

    by_cat = store.as_object_sets()
    summary_rows, mapping_rows = compute_object_metrics(
        by_cat=by_cat,
        cat_id_to_name=cat_id_to_name,
        part_id_to_name=part_id_to_name,
        label=args.label,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"{args.label}_set_structure_summary.csv"
    mapping_path = out_dir / f"{args.label}_best_permutation_mapping.csv"
    report_path = out_dir / f"{args.label}_set_structure_report.txt"

    if summary_rows:
        fieldnames = sorted(set().union(*(r.keys() for r in summary_rows)))
        preferred = ["label", "cat_id", "object_name", "num_parts"]
        fieldnames = preferred + [c for c in fieldnames if c not in preferred]
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    if mapping_rows:
        fieldnames = [
            "label", "cat_id", "object_name", "query_type",
            "query_part_id", "query_part_name",
            "named_gt_part_id", "named_gt_part_name", "named_cos",
            "top1_gt_part_id", "top1_gt_part_name", "top1_cos",
            "bestperm_gt_part_id", "bestperm_gt_part_name", "bestperm_cos",
            "bestperm_is_identity",
        ]
        with mapping_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mapping_rows)

    write_report(report_path, summary_rows, args.label)

    print(f"[saved] {summary_path}")
    print(f"[saved] {mapping_path}")
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
