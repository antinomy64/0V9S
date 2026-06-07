#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
# When this file is copied into repo/scripts/, SCRIPT_DIR is repo/scripts.
# Keep both paths to make direct execution robust.
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(REPO_ROOT))

try:
    from anlysis import DatasetAnalyser, get_part_names, to_device_batch
except Exception as exc:
    raise ImportError(
        "Cannot import from scripts/anlysis.py. Put this script under repo/scripts/ "
        "next to anlysis.py, or add scripts/ to PYTHONPATH."
    ) from exc


@dataclass
class ScalarStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_val: float = float("inf")
    max_val: float = float("-inf")

    def add(self, value: float) -> None:
        if value is None or math.isnan(float(value)) or math.isinf(float(value)):
            return
        v = float(value)
        self.count += 1
        self.total += v
        self.total_sq += v * v
        self.min_val = min(self.min_val, v)
        self.max_val = max(self.max_val, v)

    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else float("nan")

    def std(self) -> float:
        if self.count <= 1:
            return float("nan")
        m = self.mean()
        var = max(0.0, self.total_sq / self.count - m * m)
        return math.sqrt(var)

    def to_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}count": self.count,
            f"{prefix}mean": self.mean(),
            f"{prefix}std": self.std(),
            f"{prefix}min": self.min_val if self.count > 0 else float("nan"),
            f"{prefix}max": self.max_val if self.count > 0 else float("nan"),
        }


def safe_class_name(metadata: Any, b: int, category_id: int) -> str:
    if isinstance(metadata, (list, tuple)) and b < len(metadata):
        meta = metadata[b]
        if isinstance(meta, dict):
            return str(meta.get("class_name", f"class_{category_id}"))
    return f"class_{category_id}"


def safe_image_ann_id(metadata: Any, b: int, fallback: int) -> Tuple[Any, Any]:
    if isinstance(metadata, (list, tuple)) and b < len(metadata):
        meta = metadata[b]
        if isinstance(meta, dict):
            return meta.get("image_id", fallback), meta.get("annotation_id", fallback)
    return fallback, fallback


def offdiag_mean_cosine(x_norm: torch.Tensor) -> float:
    """Mean pairwise cosine excluding diagonal for already normalized tokens [n, d]."""
    n = int(x_norm.shape[0])
    if n < 2:
        return float("nan")
    s = x_norm.sum(dim=0)
    # sum_{i != j} xi dot xj = ||sum_i xi||^2 - n, because ||xi||=1.
    offdiag_sum = float((s @ s).item()) - float(n)
    return offdiag_sum / float(n * (n - 1))


def normalize_mean(x_norm: torch.Tensor) -> torch.Tensor:
    """Prototype = L2-normalized mean of normalized tokens."""
    return F.normalize(x_norm.mean(dim=0, keepdim=True), dim=-1).squeeze(0)


def float_or_nan(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        "Audit DINOv2 patch-token part separability.\n"
        "For each GT part, compute intra-GT-mask patch-token similarity; "
        "for each object class, compute inter-GT-part-prototype similarities."
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_parts", type=int, default=116)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_obj_area_ratio", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all object instances.")
    parser.add_argument("--out_dir", default="audits/dinov2_part_patch_separability")
    parser.add_argument("--save_instance_rows", action="store_true", default=False)
    parser.add_argument("--save_instance_pair_rows", action="store_true", default=False)
    parser.add_argument(
        "--save_padded_token_bank",
        action="store_true",
        default=False,
        help="Optional debug output. Saves a capped padded tensor [num_obj,max_part,max_patch,D]; can be large.",
    )
    parser.add_argument("--bank_max_objects", type=int, default=256)
    parser.add_argument("--bank_max_parts_per_obj", type=int, default=32)
    parser.add_argument("--bank_max_patches_per_part", type=int, default=128)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyser = DatasetAnalyser(
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
        show_progress=True,
        min_obj_area_ratio=args.min_obj_area_ratio,
    )

    device = analyser.device
    part_names = analyser.part_names
    P = int(args.num_parts)

    # Global per-part accumulators.
    # part_sum_vec[pid] is sum of normalized patch tokens over all GT pixels of that part.
    dino_dim: Optional[int] = None
    part_sum_vec: Optional[torch.Tensor] = None
    part_token_count = torch.zeros(P, dtype=torch.long)
    part_instance_count = torch.zeros(P, dtype=torch.long)
    part_instance_internal_stats = [ScalarStats() for _ in range(P)]
    part_to_category = torch.full((P,), -1, dtype=torch.long)
    category_name: Dict[int, str] = {}

    # Per-category object-level stats.
    cat_internal_stats: Dict[int, ScalarStats] = defaultdict(ScalarStats)
    cat_inter_proto_stats: Dict[int, ScalarStats] = defaultdict(ScalarStats)
    cat_gap_stats: Dict[int, ScalarStats] = defaultdict(ScalarStats)
    cat_pair_lower_than_avg_count: Dict[int, int] = defaultdict(int)
    cat_pair_lower_than_both_count: Dict[int, int] = defaultdict(int)
    cat_pair_total_count: Dict[int, int] = defaultdict(int)
    cat_instance_count: Dict[int, int] = defaultdict(int)

    # Optional row-level outputs.
    instance_rows: List[Dict[str, Any]] = []
    instance_pair_rows: List[Dict[str, Any]] = []

    # Optional padded bank chunks. This is intentionally capped.
    bank_objects: List[torch.Tensor] = []
    bank_part_ids: List[torch.Tensor] = []
    bank_part_mask: List[torch.Tensor] = []
    bank_token_mask: List[torch.Tensor] = []
    bank_category_ids: List[int] = []

    seen_samples = 0

    pbar = tqdm(analyser.loader, desc="audit DINO part patch separability")
    for batch in pbar:
        batch = to_device_batch(batch, device)

        patch_tokens = batch["patch_tokens"].float()                  # [B,N,D]
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()                     # [B,N]
        part_gt = batch["part_gt_mask_patch"].bool()                  # [B,K,N]
        part_valid = batch["part_valid_mask"].bool()                  # [B,K]
        part_ids = batch["part_category_id"].long()                   # [B,K]
        cat_ids = batch["category_id"].long()                         # [B]
        metadata = batch.get("metadata", None)

        if dino_dim is None:
            dino_dim = int(patch_tokens.shape[-1])
            part_sum_vec = torch.zeros((P, dino_dim), dtype=torch.float64)

        B = int(patch_tokens.shape[0])
        for b in range(B):
            if args.max_samples > 0 and seen_samples >= args.max_samples:
                break

            cat = int(cat_ids[b].item())
            category_name.setdefault(cat, safe_class_name(metadata, b, cat))
            cat_instance_count[cat] += 1
            image_id, ann_id = safe_image_ann_id(metadata, b, seen_samples)

            present_items = []
            valid_local_idx = torch.nonzero(part_valid[b], as_tuple=False).squeeze(1)

            for local_idx_t in valid_local_idx:
                local_idx = int(local_idx_t.item())
                pid = int(part_ids[b, local_idx].item())
                if pid < 0 or pid >= P:
                    continue

                mask = part_gt[b, local_idx] & obj_mask[b]
                n_tok = int(mask.sum().item())
                if n_tok <= 0:
                    continue

                toks = patch_tokens[b, mask]                           # [n,D], normalized
                internal = offdiag_mean_cosine(toks)
                proto = normalize_mean(toks)                            # [D]

                part_instance_count[pid] += 1
                part_token_count[pid] += n_tok
                part_to_category[pid] = cat
                part_instance_internal_stats[pid].add(internal)
                cat_internal_stats[cat].add(internal)

                # Global pooled internal/prototype accumulators.
                assert part_sum_vec is not None
                part_sum_vec[pid] += toks.detach().double().cpu().sum(dim=0)

                present_items.append(
                    {
                        "local_idx": local_idx,
                        "pid": pid,
                        "part_name": part_names[pid] if pid < len(part_names) else f"part_{pid}",
                        "num_tokens": n_tok,
                        "internal_cos": internal,
                        "proto": proto.detach(),
                        "tokens": toks.detach() if args.save_padded_token_bank else None,
                    }
                )

            # Per-instance prototype pair similarities.
            pair_cos_values = []
            gap_values = []
            lower_than_avg = 0
            lower_than_both = 0
            total_pairs = 0

            for i in range(len(present_items)):
                for j in range(i + 1, len(present_items)):
                    pi = present_items[i]
                    pj = present_items[j]
                    proto_cos = float((pi["proto"] @ pj["proto"]).item())
                    int_i = float_or_nan(pi["internal_cos"])
                    int_j = float_or_nan(pj["internal_cos"])
                    avg_internal = (
                        0.5 * (int_i + int_j)
                        if not (math.isnan(int_i) or math.isnan(int_j))
                        else float("nan")
                    )
                    gap = avg_internal - proto_cos if not math.isnan(avg_internal) else float("nan")

                    total_pairs += 1
                    pair_cos_values.append(proto_cos)
                    if not math.isnan(gap):
                        gap_values.append(gap)
                        if proto_cos < avg_internal:
                            lower_than_avg += 1
                        if proto_cos < min(int_i, int_j):
                            lower_than_both += 1

                    cat_inter_proto_stats[cat].add(proto_cos)
                    cat_gap_stats[cat].add(gap)
                    cat_pair_total_count[cat] += 1
                    if not math.isnan(avg_internal) and proto_cos < avg_internal:
                        cat_pair_lower_than_avg_count[cat] += 1
                    if not math.isnan(avg_internal) and proto_cos < min(int_i, int_j):
                        cat_pair_lower_than_both_count[cat] += 1

                    if args.save_instance_pair_rows:
                        instance_pair_rows.append(
                            {
                                "sample_index": seen_samples,
                                "image_id": image_id,
                                "annotation_id": ann_id,
                                "category_id": cat,
                                "class_name": category_name[cat],
                                "pid_i": pi["pid"],
                                "part_i": pi["part_name"],
                                "pid_j": pj["pid"],
                                "part_j": pj["part_name"],
                                "num_tokens_i": pi["num_tokens"],
                                "num_tokens_j": pj["num_tokens"],
                                "internal_cos_i": int_i,
                                "internal_cos_j": int_j,
                                "avg_internal_cos": avg_internal,
                                "inter_proto_cos": proto_cos,
                                "avg_internal_minus_inter": gap,
                                "inter_lower_than_avg_internal": int(proto_cos < avg_internal) if not math.isnan(avg_internal) else 0,
                                "inter_lower_than_both_internal": int(proto_cos < min(int_i, int_j)) if not math.isnan(avg_internal) else 0,
                            }
                        )

            if args.save_instance_rows:
                mean_internal = (
                    sum(float_or_nan(x["internal_cos"]) for x in present_items if not math.isnan(float_or_nan(x["internal_cos"])))
                    / max(1, sum(1 for x in present_items if not math.isnan(float_or_nan(x["internal_cos"]))))
                    if len(present_items) > 0
                    else float("nan")
                )
                mean_inter = sum(pair_cos_values) / len(pair_cos_values) if len(pair_cos_values) > 0 else float("nan")
                mean_gap = sum(gap_values) / len(gap_values) if len(gap_values) > 0 else float("nan")
                instance_rows.append(
                    {
                        "sample_index": seen_samples,
                        "image_id": image_id,
                        "annotation_id": ann_id,
                        "category_id": cat,
                        "class_name": category_name[cat],
                        "num_present_parts": len(present_items),
                        "num_part_pairs": total_pairs,
                        "mean_internal_cos": mean_internal,
                        "mean_inter_proto_cos": mean_inter,
                        "mean_internal_minus_inter": mean_gap,
                        "frac_pairs_inter_lower_than_avg_internal": lower_than_avg / total_pairs if total_pairs > 0 else float("nan"),
                        "frac_pairs_inter_lower_than_both_internal": lower_than_both / total_pairs if total_pairs > 0 else float("nan"),
                    }
                )

            if args.save_padded_token_bank and len(bank_objects) < int(args.bank_max_objects):
                max_parts = int(args.bank_max_parts_per_obj)
                max_patches = int(args.bank_max_patches_per_part)
                D = int(patch_tokens.shape[-1])
                obj_bank = torch.zeros((max_parts, max_patches, D), dtype=torch.float16)
                obj_token_mask = torch.zeros((max_parts, max_patches), dtype=torch.bool)
                obj_part_ids = torch.full((max_parts,), -1, dtype=torch.long)
                obj_part_mask = torch.zeros((max_parts,), dtype=torch.bool)

                for slot, item in enumerate(present_items[:max_parts]):
                    toks = item["tokens"]
                    if toks is None:
                        continue
                    n = min(int(toks.shape[0]), max_patches)
                    obj_bank[slot, :n] = toks[:n].detach().cpu().to(torch.float16)
                    obj_token_mask[slot, :n] = True
                    obj_part_ids[slot] = int(item["pid"])
                    obj_part_mask[slot] = True

                bank_objects.append(obj_bank)
                bank_part_ids.append(obj_part_ids)
                bank_part_mask.append(obj_part_mask)
                bank_token_mask.append(obj_token_mask)
                bank_category_ids.append(cat)

            seen_samples += 1

        pbar.set_description(f"audit DINO part patch separability samples={seen_samples}")
        if args.max_samples > 0 and seen_samples >= args.max_samples:
            break

    if dino_dim is None or part_sum_vec is None:
        raise RuntimeError("No samples were processed.")

    # ------------------------------------------------------------
    # Per-part global pooled internal/prototype stats.
    # ------------------------------------------------------------
    part_proto = torch.zeros((P, dino_dim), dtype=torch.float32)
    part_pooled_internal = torch.full((P,), float("nan"), dtype=torch.float32)
    part_valid = part_token_count > 0

    for pid in range(P):
        cnt = int(part_token_count[pid].item())
        if cnt <= 0:
            continue
        s = part_sum_vec[pid].float()
        part_proto[pid] = F.normalize(s[None, :], dim=-1).squeeze(0)
        if cnt >= 2:
            offdiag_sum = float((s.double() @ s.double()).item()) - float(cnt)
            part_pooled_internal[pid] = offdiag_sum / float(cnt * (cnt - 1))

    per_part_rows: List[Dict[str, Any]] = []
    for pid in range(P):
        cat = int(part_to_category[pid].item())
        per_part_rows.append(
            {
                "pid": pid,
                "part_name": part_names[pid] if pid < len(part_names) else f"part_{pid}",
                "category_id": cat,
                "class_name": category_name.get(cat, f"class_{cat}"),
                "token_count": int(part_token_count[pid].item()),
                "instance_count": int(part_instance_count[pid].item()),
                "mean_instance_internal_cos": part_instance_internal_stats[pid].mean(),
                "std_instance_internal_cos": part_instance_internal_stats[pid].std(),
                "pooled_internal_cos": float(part_pooled_internal[pid].item()) if bool(part_valid[pid]) else float("nan"),
            }
        )

    # ------------------------------------------------------------
    # Per-object global prototype pair stats: this matches the user's
    # p1/p2/p3 example at category level.
    # ------------------------------------------------------------
    pair_rows: List[Dict[str, Any]] = []
    per_object_rows: List[Dict[str, Any]] = []

    valid_cats = sorted(int(x) for x in set(part_to_category[part_to_category >= 0].tolist()))
    for cat in valid_cats:
        pids = torch.nonzero((part_to_category == cat) & part_valid, as_tuple=False).squeeze(1).tolist()
        pids = [int(x) for x in pids]
        if len(pids) == 0:
            continue

        obj_internal = ScalarStats()
        for pid in pids:
            obj_internal.add(float(part_pooled_internal[pid].item()))

        obj_inter = ScalarStats()
        obj_gap = ScalarStats()
        lower_avg = 0
        lower_both = 0
        n_pairs = 0
        nearest_other: Dict[int, float] = {pid: float("nan") for pid in pids}

        for a in range(len(pids)):
            for b in range(a + 1, len(pids)):
                pid_i = pids[a]
                pid_j = pids[b]
                cos_ij = float((part_proto[pid_i] @ part_proto[pid_j]).item())
                int_i = float(part_pooled_internal[pid_i].item())
                int_j = float(part_pooled_internal[pid_j].item())
                avg_int = 0.5 * (int_i + int_j)
                gap = avg_int - cos_ij

                n_pairs += 1
                obj_inter.add(cos_ij)
                obj_gap.add(gap)
                if cos_ij < avg_int:
                    lower_avg += 1
                if cos_ij < min(int_i, int_j):
                    lower_both += 1
                if math.isnan(nearest_other[pid_i]) or cos_ij > nearest_other[pid_i]:
                    nearest_other[pid_i] = cos_ij
                if math.isnan(nearest_other[pid_j]) or cos_ij > nearest_other[pid_j]:
                    nearest_other[pid_j] = cos_ij

                pair_rows.append(
                    {
                        "category_id": cat,
                        "class_name": category_name.get(cat, f"class_{cat}"),
                        "pid_i": pid_i,
                        "part_i": part_names[pid_i] if pid_i < len(part_names) else f"part_{pid_i}",
                        "pid_j": pid_j,
                        "part_j": part_names[pid_j] if pid_j < len(part_names) else f"part_{pid_j}",
                        "token_count_i": int(part_token_count[pid_i].item()),
                        "token_count_j": int(part_token_count[pid_j].item()),
                        "pooled_internal_cos_i": int_i,
                        "pooled_internal_cos_j": int_j,
                        "avg_internal_cos": avg_int,
                        "inter_gt_prototype_cos": cos_ij,
                        "avg_internal_minus_inter": gap,
                        "inter_lower_than_avg_internal": int(cos_ij < avg_int),
                        "inter_lower_than_both_internal": int(cos_ij < min(int_i, int_j)),
                    }
                )

        nearest_stats = ScalarStats()
        for v in nearest_other.values():
            nearest_stats.add(v)

        per_object_rows.append(
            {
                "category_id": cat,
                "class_name": category_name.get(cat, f"class_{cat}"),
                "num_parts": len(pids),
                "num_part_pairs": n_pairs,
                "num_object_instances": int(cat_instance_count.get(cat, 0)),
                "mean_pooled_internal_cos": obj_internal.mean(),
                "mean_inter_gt_prototype_cos": obj_inter.mean(),
                "mean_internal_minus_inter": obj_gap.mean(),
                "frac_pairs_inter_lower_than_avg_internal": lower_avg / n_pairs if n_pairs > 0 else float("nan"),
                "frac_pairs_inter_lower_than_both_internal": lower_both / n_pairs if n_pairs > 0 else float("nan"),
                "mean_nearest_other_prototype_cos": nearest_stats.mean(),
                "mean_instance_internal_cos": cat_internal_stats[cat].mean(),
                "mean_instance_inter_proto_cos": cat_inter_proto_stats[cat].mean(),
                "mean_instance_internal_minus_inter": cat_gap_stats[cat].mean(),
                "instance_frac_pairs_inter_lower_than_avg_internal": (
                    cat_pair_lower_than_avg_count[cat] / cat_pair_total_count[cat]
                    if cat_pair_total_count[cat] > 0 else float("nan")
                ),
                "instance_frac_pairs_inter_lower_than_both_internal": (
                    cat_pair_lower_than_both_count[cat] / cat_pair_total_count[cat]
                    if cat_pair_total_count[cat] > 0 else float("nan")
                ),
            }
        )

    # ------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------
    write_csv(
        out_dir / "per_part_internal_similarity.csv",
        per_part_rows,
        [
            "pid", "part_name", "category_id", "class_name",
            "token_count", "instance_count",
            "mean_instance_internal_cos", "std_instance_internal_cos", "pooled_internal_cos",
        ],
    )

    write_csv(
        out_dir / "per_object_pair_prototype_similarity.csv",
        pair_rows,
        [
            "category_id", "class_name",
            "pid_i", "part_i", "pid_j", "part_j",
            "token_count_i", "token_count_j",
            "pooled_internal_cos_i", "pooled_internal_cos_j", "avg_internal_cos",
            "inter_gt_prototype_cos", "avg_internal_minus_inter",
            "inter_lower_than_avg_internal", "inter_lower_than_both_internal",
        ],
    )

    write_csv(
        out_dir / "per_object_separability_summary.csv",
        per_object_rows,
        [
            "category_id", "class_name", "num_parts", "num_part_pairs", "num_object_instances",
            "mean_pooled_internal_cos", "mean_inter_gt_prototype_cos", "mean_internal_minus_inter",
            "frac_pairs_inter_lower_than_avg_internal", "frac_pairs_inter_lower_than_both_internal",
            "mean_nearest_other_prototype_cos",
            "mean_instance_internal_cos", "mean_instance_inter_proto_cos", "mean_instance_internal_minus_inter",
            "instance_frac_pairs_inter_lower_than_avg_internal", "instance_frac_pairs_inter_lower_than_both_internal",
        ],
    )

    if args.save_instance_rows:
        write_csv(
            out_dir / "per_instance_separability_summary.csv",
            instance_rows,
            [
                "sample_index", "image_id", "annotation_id", "category_id", "class_name",
                "num_present_parts", "num_part_pairs",
                "mean_internal_cos", "mean_inter_proto_cos", "mean_internal_minus_inter",
                "frac_pairs_inter_lower_than_avg_internal", "frac_pairs_inter_lower_than_both_internal",
            ],
        )

    if args.save_instance_pair_rows:
        write_csv(
            out_dir / "per_instance_pair_similarity.csv",
            instance_pair_rows,
            [
                "sample_index", "image_id", "annotation_id", "category_id", "class_name",
                "pid_i", "part_i", "pid_j", "part_j",
                "num_tokens_i", "num_tokens_j",
                "internal_cos_i", "internal_cos_j", "avg_internal_cos",
                "inter_proto_cos", "avg_internal_minus_inter",
                "inter_lower_than_avg_internal", "inter_lower_than_both_internal",
            ],
        )

    proto_save = {
        "part_proto": part_proto,
        "part_valid": part_valid,
        "part_token_count": part_token_count,
        "part_instance_count": part_instance_count,
        "part_pooled_internal_cos": part_pooled_internal,
        "part_to_category": part_to_category,
        "part_names": part_names,
        "category_name": category_name,
        "meta": vars(args),
    }
    torch.save(proto_save, out_dir / "gt_part_patch_stats_and_prototypes.pth")

    if args.save_padded_token_bank and len(bank_objects) > 0:
        bank = {
            "tokens": torch.stack(bank_objects, dim=0),
            "token_mask": torch.stack(bank_token_mask, dim=0),
            "part_ids": torch.stack(bank_part_ids, dim=0),
            "part_mask": torch.stack(bank_part_mask, dim=0),
            "category_ids": torch.tensor(bank_category_ids, dtype=torch.long),
            "meta": {
                "shape": "[num_saved_obj, max_parts_per_obj, max_patches_per_part, dino_dim]",
                "bank_max_objects": args.bank_max_objects,
                "bank_max_parts_per_obj": args.bank_max_parts_per_obj,
                "bank_max_patches_per_part": args.bank_max_patches_per_part,
                "note": "This bank is capped/debug-only. Streaming CSV stats use all processed samples.",
            },
        }
        torch.save(bank, out_dir / "debug_padded_part_token_bank.pth")

    summary_internal = ScalarStats()
    summary_inter = ScalarStats()
    summary_gap = ScalarStats()
    frac_avg_stats = ScalarStats()
    frac_both_stats = ScalarStats()

    for row in per_object_rows:
        summary_internal.add(row["mean_pooled_internal_cos"])
        summary_inter.add(row["mean_inter_gt_prototype_cos"])
        summary_gap.add(row["mean_internal_minus_inter"])
        frac_avg_stats.add(row["frac_pairs_inter_lower_than_avg_internal"])
        frac_both_stats.add(row["frac_pairs_inter_lower_than_both_internal"])

    summary = {
        "dataset": args.dataset,
        "num_processed_object_instances": seen_samples,
        "num_parts": P,
        "num_valid_parts": int(part_valid.sum().item()),
        "num_valid_object_categories": len(valid_cats),
        "macro_mean_pooled_internal_cos": summary_internal.mean(),
        "macro_mean_inter_gt_prototype_cos": summary_inter.mean(),
        "macro_mean_internal_minus_inter": summary_gap.mean(),
        "macro_frac_pairs_inter_lower_than_avg_internal": frac_avg_stats.mean(),
        "macro_frac_pairs_inter_lower_than_both_internal": frac_both_stats.mean(),
        "output_files": [
            "per_part_internal_similarity.csv",
            "per_object_pair_prototype_similarity.csv",
            "per_object_separability_summary.csv",
            "gt_part_patch_stats_and_prototypes.pth",
        ],
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[summary]")
    for k, v in summary.items():
        if k != "output_files":
            print(f"{k}: {v}")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
