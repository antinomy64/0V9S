#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_float_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu()


def json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def build_debug_loss(repo_root: str, model, patch_temperature: float):
    sys.path.insert(0, repo_root)
    from src.loss_joint import JointObjPartLoss

    class JointObjPartLossAnchorDebug(JointObjPartLoss):
        @torch.no_grad()
        def debug_anchor_indices(self, batch: Dict[str, torch.Tensor]) -> List[Dict]:
            """
            Reproduce the CURRENT anchor selection logic exactly:
              1) project part text
              2) normalize part text and patch tokens
              3) compute absolute logits inside object
              4) convert to relative scores with _compute_relative_scores()
              5) greedy unique anchor selection
              6) fallback with per-row argmax if still unassigned
            """
            part_text_feat = batch["part_text_feat"]
            patch_tokens = batch["patch_tokens"]
            obj_mask_patch = batch["obj_mask_patch"]
            part_valid_mask = batch["part_valid_mask"]

            part_proj = self.sim_model.project_clip_txt(part_text_feat.float())
            part_proj = self._safe_normalize(part_proj, dim=-1)
            patch_tokens = self._safe_normalize(patch_tokens.float(), dim=-1)

            abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / self.patch_temperature
            abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

            B, _, _ = abs_logits.shape
            results = []
            for b in range(B):
                valid_patch_mask = obj_mask_patch[b]
                valid_part_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)

                if valid_part_idx.numel() == 0 or valid_patch_mask.sum() == 0:
                    results.append({
                        "valid_part_idx": valid_part_idx.cpu(),
                        "anchor_idx_local": torch.empty((0,), dtype=torch.long),
                        "anchor_idx_global": torch.empty((0,), dtype=torch.long),
                        "valid_patch_idx_global": torch.empty((0,), dtype=torch.long),
                    })
                    continue

                local_scores = abs_logits[b][valid_part_idx][:, valid_patch_mask]
                Kb, Mb = local_scores.shape

                rel_scores = self._compute_relative_scores(local_scores)
                flat_scores = rel_scores.reshape(-1)
                sorted_idx = torch.argsort(flat_scores, descending=True)

                anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=local_scores.device)
                patch_taken = torch.zeros((Mb,), dtype=torch.bool, device=local_scores.device)

                assigned_parts = 0
                for flat_id in sorted_idx:
                    p_local = torch.div(flat_id, Mb, rounding_mode='floor')
                    n_local = flat_id % Mb

                    if anchor_idx_local[p_local] != -1:
                        continue
                    if patch_taken[n_local]:
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

                valid_patch_idx_global = torch.nonzero(valid_patch_mask, as_tuple=False).squeeze(1)
                anchor_idx_global = valid_patch_idx_global[anchor_idx_local]

                results.append({
                    "valid_part_idx": valid_part_idx.cpu(),
                    "anchor_idx_local": anchor_idx_local.cpu(),
                    "anchor_idx_global": anchor_idx_global.cpu(),
                    "valid_patch_idx_global": valid_patch_idx_global.cpu(),
                })
            return results

    return JointObjPartLossAnchorDebug(model, patch_temperature=patch_temperature)


def save_heatmap_png(
    matrix: np.ndarray,
    part_names: List[str],
    title: str,
    out_path: Path,
    annotate: bool = True,
):
    fig_w = max(8, 0.6 * len(part_names))
    fig_h = max(7, 0.55 * len(part_names))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("GT mask region part")
    ax.set_ylabel("Text-anchor part")
    ax.set_xticks(np.arange(len(part_names)))
    ax.set_yticks(np.arange(len(part_names)))
    ax.set_xticklabels(part_names, rotation=90)
    ax.set_yticklabels(part_names)

    if annotate and len(part_names) <= 25:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                txt = f"{val:.2f}" if matrix.dtype.kind == "f" else str(int(val))
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("value", rotation=-90, va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Build per-class anchor landing heatmaps: row=text part, col=GT part region."
    )
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--feature_pth", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", type=str, default=None)
    parser.add_argument("--patch_temperature", type=float, default=0.07)
    parser.add_argument("--min_obj_area_ratio", type=float, default=0.0)
    parser.add_argument("--out_dir", default="anchor_part_heatmaps")
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 means use all samples")
    args = parser.parse_args()

    sys.path.insert(0, args.repo_root)
    from src.dataset_joint_with_part_anchoraudit import DinoClipJointDataset, joint_collate_fn
    from src.model import ProjectionLayer, DoubleMLP
    import yaml

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_class_name = cfg["model"].get("model_class", "ProjectionLayer")
    ModelClass = {"ProjectionLayer": ProjectionLayer, "DoubleMLP": DoubleMLP}[model_class_name]
    model = ModelClass.from_config(cfg["model"])
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ret = model.load_state_dict(ckpt, strict=False)
    model.eval()

    criterion = build_debug_loss(args.repo_root, model, args.patch_temperature)

    is_wds = ".tar" in args.feature_pth
    dataset = DinoClipJointDataset(
        args.feature_pth,
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
        min_obj_area_ratio=args.min_obj_area_ratio,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_stats = {}
    n_total = len(dataset) if args.max_samples < 0 else min(len(dataset), args.max_samples)

    for idx in range(n_total):
        sample = dataset[idx]
        batch = joint_collate_fn([sample])
        debug = criterion.debug_anchor_indices(batch)[0]

        class_name = sample["metadata"]["class_name"]
        category_id = int(sample["category_id"].item())
        part_names = list(sample["metadata"]["part_class_name"])
        K = len(part_names)

        if class_name not in class_stats:
            class_stats[class_name] = {
                "category_id": category_id,
                "part_names": part_names,
                "count_matrix": np.zeros((K, K), dtype=np.int64),
                "miss_counts": np.zeros((K,), dtype=np.int64),
                "multi_hit_counts": np.zeros((K,), dtype=np.int64),
                "num_samples": 0,
                "num_anchors_evaluated": np.zeros((K,), dtype=np.int64),
            }

        stats = class_stats[class_name]
        stats["num_samples"] += 1

        gt_masks = to_float_cpu(batch["part_gt_mask_patch"][0]).bool()   # [K, N]
        anchor_idx_global = debug["anchor_idx_global"].tolist()
        valid_part_idx = debug["valid_part_idx"].tolist()

        for local_row, bank_row in enumerate(valid_part_idx):
            anchor_patch = int(anchor_idx_global[local_row])
            hit_cols = torch.nonzero(gt_masks[:, anchor_patch], as_tuple=False).view(-1).tolist()

            stats["num_anchors_evaluated"][bank_row] += 1

            if len(hit_cols) == 0:
                stats["miss_counts"][bank_row] += 1
            else:
                for col in hit_cols:
                    stats["count_matrix"][bank_row, col] += 1
                if len(hit_cols) > 1:
                    stats["multi_hit_counts"][bank_row] += 1

        if (idx + 1) % 200 == 0 or (idx + 1) == n_total:
            print(f"[progress] processed {idx + 1}/{n_total} samples")

    summary = {
        "config": {
            "feature_pth": args.feature_pth,
            "ckpt": args.ckpt,
            "config": args.config,
            "obj_feature_name": args.obj_feature_name,
            "part_feature_name": args.part_feature_name,
            "obj_text_name": args.obj_text_name,
            "part_text_name": args.part_text_name,
            "patch_temperature": args.patch_temperature,
            "max_samples": args.max_samples,
        },
        "classes": {},
    }

    global_diag = 0
    global_total = 0
    global_miss = 0

    for class_name, stats in class_stats.items():
        part_names = stats["part_names"]
        count_matrix = stats["count_matrix"]
        miss_counts = stats["miss_counts"]
        multi_hit_counts = stats["multi_hit_counts"]
        num_anchors_evaluated = stats["num_anchors_evaluated"]

        row_normalized = np.zeros_like(count_matrix, dtype=np.float64)
        for i in range(count_matrix.shape[0]):
            denom = max(int(num_anchors_evaluated[i]), 1)
            row_normalized[i] = count_matrix[i].astype(np.float64) / denom

        diag_hits = np.diag(count_matrix).astype(np.int64)
        diag_rate = np.zeros_like(diag_hits, dtype=np.float64)
        for i in range(len(diag_hits)):
            denom = max(int(num_anchors_evaluated[i]), 1)
            diag_rate[i] = float(diag_hits[i] / denom)

        global_diag += int(diag_hits.sum())
        global_total += int(num_anchors_evaluated.sum())
        global_miss += int(miss_counts.sum())

        class_slug = class_name.replace("/", "_").replace(" ", "_")
        class_dir = out_dir / class_slug
        class_dir.mkdir(parents=True, exist_ok=True)

        np.save(class_dir / "count_matrix.npy", count_matrix)
        np.save(class_dir / "row_normalized.npy", row_normalized)
        np.save(class_dir / "miss_counts.npy", miss_counts)
        np.save(class_dir / "multi_hit_counts.npy", multi_hit_counts)
        np.save(class_dir / "num_anchors_evaluated.npy", num_anchors_evaluated)

        save_heatmap_png(
            count_matrix,
            part_names,
            title=f"{class_name} | anchor->GT part counts",
            out_path=class_dir / "count_heatmap.png",
            annotate=True,
        )
        save_heatmap_png(
            row_normalized,
            part_names,
            title=f"{class_name} | anchor->GT part rate",
            out_path=class_dir / "rate_heatmap.png",
            annotate=True,
        )

        summary["classes"][class_name] = {
            "category_id": stats["category_id"],
            "part_names": part_names,
            "num_samples": int(stats["num_samples"]),
            "count_matrix": count_matrix.tolist(),
            "row_normalized": row_normalized.tolist(),
            "miss_counts": miss_counts.tolist(),
            "multi_hit_counts": multi_hit_counts.tolist(),
            "num_anchors_evaluated": num_anchors_evaluated.tolist(),
            "diag_hits": diag_hits.tolist(),
            "diag_rate": diag_rate.tolist(),
            "class_dir": str(class_dir),
        }

    summary["global"] = {
        "num_classes": len(class_stats),
        "diag_hits_sum": int(global_diag),
        "anchors_total": int(global_total),
        "miss_sum": int(global_miss),
        "global_diag_rate": float(global_diag / max(global_total, 1)),
        "global_miss_rate": float(global_miss / max(global_total, 1)),
        "checkpoint_missing_keys": getattr(ret, "missing_keys", []),
        "checkpoint_unexpected_keys": getattr(ret, "unexpected_keys", []),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)

    print("[done] wrote:", out_dir / "summary.json")
    print("[done] global diag rate:", summary["global"]["global_diag_rate"])
    print("[done] global miss rate:", summary["global"]["global_miss_rate"])


if __name__ == "__main__":
    main()
