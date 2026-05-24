from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn
from src.loss_joint import JointObjPartLoss


def load_projector(model_config: str, init_weights: str, device: torch.device):
    with open(model_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"].get("model_class", "ProjectionLayer")
    Model = getattr(importlib.import_module("src.model"), model_name)
    model = Model.from_config(cfg["model"]).to(device)

    print(f"[load projector] {init_weights}")
    ckpt = torch.load(init_weights, map_location="cpu")
    msg = model.load_state_dict(ckpt, strict=False)
    print("  missing keys   :", getattr(msg, "missing_keys", []))
    print("  unexpected keys:", getattr(msg, "unexpected_keys", []))

    model.eval()
    return model, cfg


def build_dataset(args, cfg):
    min_obj_area_ratio = float(cfg.get("dataset", {}).get("min_obj_area_ratio", 0.0))
    return DinoClipJointDataset(
        args.dataset,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=".tar" in args.dataset,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=min_obj_area_ratio,
    )


def clean_name(x: Any) -> str:
    return str(x).replace("\n", " ").strip()


def get_part_name_from_meta(meta: Dict[str, Any], local_idx: int, pid: int) -> str:
    names = meta.get("part_class_name", [])
    if isinstance(names, (list, tuple)) and local_idx < len(names):
        return clean_name(names[local_idx])
    return f"part_{pid}"


def collect_global_part_names(dataset) -> Dict[int, str]:
    """
    Build pid -> readable part name from dataset.data.
    If one pid appears with multiple string variants, keep the most frequent.
    """
    name_counter = defaultdict(Counter)

    data = dataset.data.values() if isinstance(dataset.data, dict) else dataset.data
    for sample in data:
        pids = sample["part_category_id"]
        names = sample.get("part_class_name", [])

        if torch.is_tensor(pids):
            pids_list = [int(x) for x in pids.detach().cpu().tolist()]
        else:
            pids_list = [int(x) for x in pids]

        for i, pid in enumerate(pids_list):
            if isinstance(names, (list, tuple)) and i < len(names):
                name_counter[pid][clean_name(names[i])] += 1
            else:
                name_counter[pid][f"part_{pid}"] += 1

    out = {}
    for pid, ctr in name_counter.items():
        out[int(pid)] = ctr.most_common(1)[0][0]
    return out


@torch.no_grad()
def compute_anchor_tokens_by_original_stage2_function(
    model,
    anchor_helper: JointObjPartLoss,
    batch: Dict[str, Any],
    patch_temperature: float,
    em_iters: int,
):
    """
    Use the original Stage2 anchor routine.

    Important:
      This calls JointObjPartLoss._anchor_proto_em_pool(..., return_anchor_tokens=True).
      We do NOT reimplement the greedy anchor selection here.
      The function returns anchor_tokens, then we recover the selected patch index
      by nearest matching the returned anchor token to normalized patch_tokens
      inside the object mask.
    """
    part_text_feat = batch["part_text_feat"].float()
    patch_tokens = batch["patch_tokens"].float()
    obj_mask_patch = batch["obj_mask_patch"].bool()
    part_valid_mask = batch["part_valid_mask"].bool()
    part_gt_mask_patch = batch["part_gt_mask_patch"].bool()

    part_proj = model.project_clip_txt(part_text_feat)
    part_proj = anchor_helper._safe_normalize(part_proj, dim=-1)
    patch_tokens_norm = anchor_helper._safe_normalize(patch_tokens, dim=-1)

    abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens_norm) / float(patch_temperature)
    abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

    _, _, metrics, anchor_tokens, anchor_valid = anchor_helper._anchor_proto_em_pool(
        patch_tokens=patch_tokens_norm,
        abs_logits=abs_logits,
        obj_mask_patch=obj_mask_patch,
        part_valid_mask=part_valid_mask,
        part_gt_mask_patch=part_gt_mask_patch,
        num_iters=em_iters,
        return_anchor_tokens=True,
    )

    return {
        "patch_tokens_norm": patch_tokens_norm,
        "obj_mask_patch": obj_mask_patch,
        "part_valid_mask": part_valid_mask,
        "part_gt_mask_patch": part_gt_mask_patch,
        "anchor_tokens": anchor_tokens,
        "anchor_valid": anchor_valid,
        "metrics": metrics,
    }


def write_wide_csv(
    out_path: Path,
    confusion: Dict[int, Counter],
    total_by_src: Counter,
    correct_by_src: Counter,
    part_names: Dict[int, str],
    num_parts: int,
    min_show_rate: float = 0.0,
):
    """
    116-row wide table:
      source_part_id, part_name, total, correct, hit_rate,
      NONE_count, NONE_rate,
      <target_part_name>_count, <target_part_name>_rate, ...
    """
    target_ids = list(range(num_parts))
    fieldnames = [
        "source_part_id",
        "part_name",
        "total",
        "correct",
        "hit_rate",
        "NONE_count",
        "NONE_rate",
    ]

    for tid in target_ids:
        tname = part_names.get(tid, f"part_{tid}")
        fieldnames.append(f"{tid}:{tname}_count")
        fieldnames.append(f"{tid}:{tname}_rate")

    rows = []
    for src_pid in range(num_parts):
        total = int(total_by_src.get(src_pid, 0))
        correct = int(correct_by_src.get(src_pid, 0))
        ctr = confusion.get(src_pid, Counter())

        row = {
            "source_part_id": src_pid,
            "part_name": part_names.get(src_pid, f"part_{src_pid}"),
            "total": total,
            "correct": correct,
            "hit_rate": (correct / total) if total > 0 else "",
            "NONE_count": int(ctr.get("NONE", 0)),
            "NONE_rate": (int(ctr.get("NONE", 0)) / total) if total > 0 else "",
        }

        for tid in target_ids:
            c = int(ctr.get(tid, 0))
            r = (c / total) if total > 0 else ""
            tname = part_names.get(tid, f"part_{tid}")
            row[f"{tid}:{tname}_count"] = c
            row[f"{tid}:{tname}_rate"] = r if (not isinstance(r, float) or r >= min_show_rate or c > 0) else 0.0

        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[saved wide 116-row table] {out_path}")


def write_compact_csv(
    out_path: Path,
    confusion: Dict[int, Counter],
    total_by_src: Counter,
    correct_by_src: Counter,
    part_names: Dict[int, str],
    topk: int,
):
    """
    Compact 116-row table:
      one row per source part.
      landed_distribution stores only nonzero targets as:
        target_name:rate(count); target_name:rate(count); ...
    """
    rows = []
    for src_pid in sorted(set(list(range(max(part_names.keys()) + 1 if part_names else 116)))):
        total = int(total_by_src.get(src_pid, 0))
        correct = int(correct_by_src.get(src_pid, 0))
        ctr = confusion.get(src_pid, Counter())

        items = []
        for target, c in ctr.most_common(topk if topk > 0 else None):
            if target == "NONE":
                tname = "NONE"
            else:
                tname = part_names.get(int(target), f"part_{int(target)}")
            rate = c / max(total, 1)
            items.append(f"{tname}:{rate:.4f}({int(c)})")

        rows.append(
            {
                "source_part_id": src_pid,
                "part_name": part_names.get(src_pid, f"part_{src_pid}"),
                "total": total,
                "correct": correct,
                "hit_rate": (correct / total) if total > 0 else "",
                "landed_distribution": "; ".join(items),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_part_id",
                "part_name",
                "total",
                "correct",
                "hit_rate",
                "landed_distribution",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[saved compact 116-row table] {out_path}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

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

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--patch_temperature", type=float, default=None)
    parser.add_argument("--em_iters", type=int, default=None)
    parser.add_argument("--num_parts", type=int, default=116)

    parser.add_argument("--save_dir", default="audits/stage2_anchor_part_gt_table")
    parser.add_argument("--topk_compact", type=int, default=20)
    parser.add_argument("--min_show_rate", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("[device]", device)

    model, cfg = load_projector(args.model_config, args.init_weights, device)

    train_cfg = cfg.get("train", {})
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 128))
    patch_temperature = float(args.patch_temperature if args.patch_temperature is not None else train_cfg.get("patch_temperature", 0.07))
    em_iters = int(args.em_iters if args.em_iters is not None else train_cfg.get("em_iters", 1))
    obj_ltype = train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce"))

    dataset = build_dataset(args, cfg)
    part_names = collect_global_part_names(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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

    confusion: Dict[int, Counter] = defaultdict(Counter)
    total_by_src: Counter = Counter()
    correct_by_src: Counter = Counter()

    # Authoritative global scalar from original function.
    total_valid_from_func = 0.0
    total_hits_from_func = 0.0

    for batch in tqdm(loader, total=len(loader), desc="stage2-anchor-part-gt-table"):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        out = compute_anchor_tokens_by_original_stage2_function(
            model=model,
            anchor_helper=anchor_helper,
            batch=batch,
            patch_temperature=patch_temperature,
            em_iters=em_iters,
        )

        total_valid_from_func += float(out["metrics"]["anchor_total_valid_parts"].detach().cpu().item())
        total_hits_from_func += float(out["metrics"]["anchor_total_hits"].detach().cpu().item())

        patch_tokens_norm = out["patch_tokens_norm"]
        obj_mask_patch = out["obj_mask_patch"]
        part_valid_mask = out["part_valid_mask"]
        part_gt_mask_patch = out["part_gt_mask_patch"]
        anchor_tokens = out["anchor_tokens"]
        anchor_valid = out["anchor_valid"]

        part_ids = batch["part_category_id"].long()

        B = int(part_ids.shape[0])
        for b in range(B):
            valid_patch_idx = torch.nonzero(obj_mask_patch[b], as_tuple=False).squeeze(1)
            if valid_patch_idx.numel() == 0:
                continue

            valid_patch_tokens = patch_tokens_norm[b, valid_patch_idx]  # [M, D]
            K = int(part_ids.shape[1])

            for k in range(K):
                if not bool(part_valid_mask[b, k]):
                    continue
                if not bool(anchor_valid[b, k]):
                    continue

                src_pid = int(part_ids[b, k].item())
                if src_pid < 0 or src_pid >= args.num_parts:
                    continue

                anchor_tok = anchor_tokens[b, k]
                # Recover selected anchor patch index from original returned anchor token.
                sim = valid_patch_tokens @ anchor_tok
                best_local = int(sim.argmax().item())
                anchor_idx = int(valid_patch_idx[best_local].item())

                target_local_idx = torch.nonzero(
                    part_valid_mask[b] & part_gt_mask_patch[b, :, anchor_idx],
                    as_tuple=False,
                ).squeeze(1)

                target_pids = []
                for t in target_local_idx.tolist():
                    pid = int(part_ids[b, int(t)].item())
                    if 0 <= pid < args.num_parts:
                        target_pids.append(pid)

                total_by_src[src_pid] += 1

                if len(target_pids) == 0:
                    confusion[src_pid]["NONE"] += 1
                else:
                    # Usually a patch belongs to at most one part, but keep multi-label safe.
                    for target_pid in target_pids:
                        confusion[src_pid][target_pid] += 1

                if src_pid in target_pids:
                    correct_by_src[src_pid] += 1

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    wide_csv = save_dir / "stage2_anchor_part_gt_distribution_116rows_wide.csv"
    compact_csv = save_dir / "stage2_anchor_part_gt_distribution_116rows_compact.csv"
    summary_json = save_dir / "summary.json"

    write_wide_csv(
        out_path=wide_csv,
        confusion=confusion,
        total_by_src=total_by_src,
        correct_by_src=correct_by_src,
        part_names=part_names,
        num_parts=args.num_parts,
        min_show_rate=args.min_show_rate,
    )
    write_compact_csv(
        out_path=compact_csv,
        confusion=confusion,
        total_by_src=total_by_src,
        correct_by_src=correct_by_src,
        part_names=part_names,
        topk=args.topk_compact,
    )

    total_from_table = sum(total_by_src.values())
    correct_from_table = sum(correct_by_src.values())
    rate_from_table = correct_from_table / max(total_from_table, 1)
    rate_from_func = total_hits_from_func / max(total_valid_from_func, 1.0)

    payload = {
        "dataset": args.dataset,
        "init_weights": args.init_weights,
        "model_config": args.model_config,
        "num_parts": args.num_parts,
        "anchor_hit_from_original_function": {
            "total_valid_parts": total_valid_from_func,
            "total_hits": total_hits_from_func,
            "hit_rate": rate_from_func,
        },
        "anchor_hit_from_recovered_patch_table": {
            "total_valid_parts": total_from_table,
            "total_hits": correct_from_table,
            "hit_rate": rate_from_table,
        },
        "difference_table_minus_original": rate_from_table - rate_from_func,
        "part_names": {str(k): v for k, v in sorted(part_names.items())},
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved summary] {summary_json}")

    print("=" * 120)
    print("[ANCHOR HIT CONSISTENCY]")
    print(
        f"original_function: hits={int(total_hits_from_func)} total={int(total_valid_from_func)} "
        f"rate={rate_from_func:.6f}"
    )
    print(
        f"recovered_table  : hits={int(correct_from_table)} total={int(total_from_table)} "
        f"rate={rate_from_table:.6f}"
    )
    print(f"diff={rate_from_table - rate_from_func:+.8f}")
    print("=" * 120)


if __name__ == "__main__":
    main()
