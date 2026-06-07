#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
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


def load_hungarian_mapping(
    csv_path: str,
    *,
    mode: str = "hungarian",
) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, float]]:
    """
    Load source pseudo-slot pid -> inferred GT pid mapping.

    Expected CSV is produced by:
      scripts/audit_global_pseudo_gt_correspondence_reuse_anlysis.py

    Required columns for mode='hungarian':
      pid, hungarian_gt_pid, hungarian_gt_part, hungarian_cos

    Optional mode='top1':
      pid, top1_gt_pid, top1_gt_part, top1_cos
    """
    csv_path = str(csv_path)
    mapping: Dict[int, int] = {}
    name_map: Dict[int, str] = {}
    score_map: Dict[int, float] = {}

    if mode not in {"hungarian", "top1", "identity"}:
        raise ValueError(f"Unknown mapping mode: {mode}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "pid" not in row:
                raise ValueError(f"CSV missing 'pid' column: {csv_path}")
            src_pid = int(row["pid"])

            if mode == "identity":
                tgt_pid = src_pid
                tgt_name = row.get("source_part", str(src_pid))
                score = float(row.get("self_cos_identity", "nan"))
            elif mode == "top1":
                tgt_pid = int(float(row["top1_gt_pid"]))
                tgt_name = row.get("top1_gt_part", str(tgt_pid))
                score = float(row.get("top1_cos", "nan"))
            else:
                tgt_pid = int(float(row["hungarian_gt_pid"]))
                tgt_name = row.get("hungarian_gt_part", str(tgt_pid))
                score = float(row.get("hungarian_cos", "nan"))

            if tgt_pid < 0:
                continue
            mapping[src_pid] = tgt_pid
            name_map[src_pid] = tgt_name
            score_map[src_pid] = score

    if len(mapping) == 0:
        raise RuntimeError(f"No mapping rows loaded from {csv_path}")

    return mapping, name_map, score_map


def remap_pseudo_pid_map(
    pseudo_pid: torch.Tensor,
    mapping: Dict[int, int],
    *,
    keep_unmapped: bool = True,
) -> torch.Tensor:
    """Remap source pseudo slot IDs to Hungarian-inferred GT IDs."""
    out = pseudo_pid.clone()
    valid = pseudo_pid >= 0
    if not valid.any():
        return out

    for pid in torch.unique(pseudo_pid[valid]).detach().cpu().tolist():
        pid = int(pid)
        if pid in mapping:
            out[pseudo_pid == pid] = int(mapping[pid])
        elif not keep_unmapped:
            out[pseudo_pid == pid] = -1
    return out


def add_hungarian_legend(
    img: Image.Image,
    meta: Dict[str, Any],
    part_ids: torch.Tensor,
    part_anchor_mask: torch.Tensor,
    part_present_mask: torch.Tensor,
    anchor_idx: torch.Tensor,
    grid_size: int,
    *,
    mapping: Dict[int, int],
    mapped_name: Dict[int, str],
    mapped_score: Dict[int, float],
    mode_name: str,
) -> Image.Image:
    rows = []
    for k in range(int(part_ids.numel())):
        if not bool(part_anchor_mask[k]):
            continue
        src_pid = int(part_ids[k].item())
        dst_pid = int(mapping.get(src_pid, src_pid))
        src_name = VA.get_part_name(meta, k, src_pid)
        dst_name = mapped_name.get(src_pid, f"part_{dst_pid}")
        score = mapped_score.get(src_pid, float("nan"))
        aidx = int(anchor_idx[k].item())
        pos = f"{aidx} ({aidx // grid_size},{aidx % grid_size})" if aidx >= 0 else "invalid"
        present = "P" if bool(part_present_mask[k]) else "A"
        rows.append((k, src_pid, dst_pid, src_name, dst_name, score, pos, present))

    if len(rows) == 0:
        return img

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    legend_w = 760
    line_h = 22
    H = max(img.height, 70 + line_h * len(rows))
    canvas = Image.new("RGB", (img.width + legend_w, H), (255, 255, 255))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    x0 = img.width + 12
    draw.text((x0, 8), f"mode={mode_name}", fill=(0, 0, 0), font=font)
    draw.text(
        (x0, 28),
        f"class={meta.get('class_name', '')} ann={meta.get('annotation_id', '')}",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text(
        (x0, 48),
        "legend color/name = Hungarian-inferred GT label; left name = source text slot",
        fill=(0, 0, 0),
        font=font,
    )

    y = 70
    for local_idx, src_pid, dst_pid, src_name, dst_name, score, pos, present in rows:
        color = VA.color_for_pid(dst_pid).astype("uint8").tolist()
        draw.rectangle((x0, y + 3, x0 + 16, y + 19), fill=tuple(color), outline=(0, 0, 0))
        try:
            score_text = f"{float(score):.3f}"
        except Exception:
            score_text = "nan"
        draw.text(
            (x0 + 24, y),
            f"{local_idx}: [{present}] {src_name}  ->  {dst_name} | cos={score_text} | anchor {pos}",
            fill=(0, 0, 0),
            font=font,
        )
        y += line_h

    return canvas


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        "Visualize Stage2 pseudo labels after Hungarian pseudo-slot -> GT-label remapping."
    )

    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--init_weights", default=None)
    parser.add_argument("--hungarian_csv", required=True, help="per_part_pseudo_to_gt_matching.csv from global pseudo/GT audit")
    parser.add_argument("--mapping_mode", default="hungarian", choices=["hungarian", "top1", "identity"])

    # Backward-compatible aliases from view_anchor.py.
    parser.add_argument("--feature_pth", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", default=None)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patch_temperature", type=float, default=None)
    parser.add_argument("--em_iters", type=int, default=None)

    parser.add_argument("--out_dir", "--save_dir", dest="out_dir", default="visualizations/stage2_hungarian_pseudo")
    parser.add_argument("--max_images", type=int, default=0, help="0 means all.")
    parser.add_argument("--ann_idx", type=int, default=None, help="Visualize one dataset index only.")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--no_anchor", action="store_true", help="Do not draw black anchor circles.")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--present_only_anchor", action="store_true", default=False)
    parser.add_argument(
        "--present_only_parts",
        action="store_true",
        default=False,
        help=(
            "Visualization/audit only: mine pseudo labels only for parts that are "
            "GT-present in the current image/object crop. This is an alias-style "
            "switch on top of the original present_only_anchor logic."
        ),
    )
    parser.add_argument("--no_crop_to_obj", action="store_true", default=False)
    parser.add_argument("--no_resize_crop", action="store_true", default=False)

    # Kept only so old commands do not crash; unused.
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--meta_pth", default=None)
    parser.add_argument("--palette_py", default=None)
    parser.add_argument("--act", default=None)

    args = parser.parse_args()
    args.dataset = args.dataset or args.feature_pth
    args.model_config = args.model_config or args.config
    args.init_weights = args.init_weights or args.ckpt

    if args.dataset is None:
        raise ValueError("Pass --dataset or old alias --feature_pth")
    if args.model_config is None:
        raise ValueError("Pass --model_config or old alias --config")
    if args.init_weights is None:
        raise ValueError("Pass --init_weights or old alias --ckpt")

    mapping, mapped_name, mapped_score = load_hungarian_mapping(
        args.hungarian_csv,
        mode=args.mapping_mode,
    )
    print(f"[mapping] loaded {len(mapping)} source_pid -> inferred_gt_pid entries from {args.hungarian_csv}")
    print(f"[mapping mode] {args.mapping_mode}")

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    present_only_active = bool(args.present_only_anchor or args.present_only_parts)

    print("[device]", device)
    print("[view mode]", "present-only crop anchor" if present_only_active else "all-candidate crop anchor")
    print("[part feature]", args.part_feature_name)

    model, cfg = VA.load_projector_from_config(args.model_config, args.init_weights, device)
    train_cfg = cfg.get("train", {})
    patch_temperature = float(args.patch_temperature if args.patch_temperature is not None else train_cfg.get("patch_temperature", 0.07))
    em_iters = int(args.em_iters if args.em_iters is not None else train_cfg.get("em_iters", 1))
    obj_ltype = train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce"))

    dataset = VA.build_joint_dataset(args, cfg)
    if args.ann_idx is not None:
        subset = torch.utils.data.Subset(dataset, [int(args.ann_idx)])
        loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=joint_collate_fn, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=joint_collate_fn, pin_memory=True)

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_no_image = 0
    skipped_no_pseudo = 0
    mode_name = f"hungarian-{args.mapping_mode}" + ("-present-only" if present_only_active else "")

    pbar = tqdm(loader, total=len(loader), desc="view-hungarian-pseudo")
    for batch in pbar:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        stage2 = VA.get_stage2_anchor_outputs(
            model=model,
            anchor_helper=anchor_helper,
            batch=batch,
            patch_temperature=patch_temperature,
            em_iters=em_iters,
            present_only_anchor=present_only_active,
        )
        anchor_idx = VA.recover_anchor_patch_indices(
            patch_tokens_norm=stage2["patch_tokens_norm"],
            obj_mask_patch=stage2["obj_mask_patch"],
            anchor_tokens=stage2["anchor_tokens"],
            anchor_valid=stage2["anchor_valid"],
        )

        B = int(batch["patch_tokens"].shape[0])
        grid_size = VA.infer_grid_size_from_tokens(
            int(batch["patch_tokens"].shape[1]),
            fallback_crop_dim=int(args.crop_dim),
            patch_size=int(args.patch_size),
        )

        for b in range(B):
            if args.max_images > 0 and saved >= args.max_images:
                break

            meta = VA.get_meta(batch, b)
            base_img = VA.open_stage2_crop_image(
                dataset=dataset,
                meta=meta,
                crop_to_obj=not bool(args.no_crop_to_obj),
                resize_to_crop_dim=not bool(args.no_resize_crop),
                crop_dim=int(args.crop_dim),
            )
            if base_img is None:
                skipped_no_image += 1
                continue

            part_ids = batch["part_category_id"][b].long()
            pseudo_pid_source = VA.build_pseudo_pid_map(
                patch_tokens_norm_b=stage2["patch_tokens_norm"][b],
                obj_mask_patch_b=stage2["obj_mask_patch"][b],
                part_anchor_mask_b=stage2["part_anchor_mask"][b],
                part_ids_b=part_ids,
                anchor_tokens_b=stage2["anchor_tokens"][b],
                anchor_idx_global_b=anchor_idx[b],
                em_iters=em_iters,
            )
            if pseudo_pid_source is None:
                skipped_no_pseudo += 1
                continue

            # This is the only semantic change versus view_anchor.py:
            # source pseudo slot IDs are renamed by global pseudo-vs-GT Hungarian mapping.
            pseudo_pid_hungarian = remap_pseudo_pid_map(pseudo_pid_source, mapping)

            gt_pid = VA.build_gt_pid_map(
                obj_mask_patch_b=stage2["obj_mask_patch"][b],
                part_valid_mask_b=stage2["part_anchor_mask"][b],
                part_ids_b=part_ids,
                part_gt_mask_patch_b=batch["part_gt_mask_patch"][b],
            )
            gt_overlay = VA.overlay_pseudo(
                base_img=base_img,
                pseudo_pid=gt_pid if gt_pid is not None else pseudo_pid_hungarian,
                grid_size=grid_size,
                alpha=float(args.alpha),
                anchor_idx=None,
                draw_anchor=False,
            )

            pseudo_overlay = VA.overlay_pseudo(
                base_img=base_img,
                pseudo_pid=pseudo_pid_hungarian,
                grid_size=grid_size,
                alpha=float(args.alpha),
                anchor_idx=anchor_idx[b],
                draw_anchor=not bool(args.no_anchor),
            )
            composed = VA.compose_gt_and_pseudo_views(gt_overlay, pseudo_overlay)
            overlay = add_hungarian_legend(
                img=composed,
                meta=meta,
                part_ids=part_ids,
                part_anchor_mask=stage2["part_anchor_mask"][b],
                part_present_mask=stage2["part_present_mask"][b],
                anchor_idx=anchor_idx[b],
                grid_size=grid_size,
                mapping=mapping,
                mapped_name=mapped_name,
                mapped_score=mapped_score,
                mode_name=mode_name,
            )

            cls = meta.get("class_name", "unknown")
            image_id = meta.get("image_id", saved)
            ann_id = meta.get("annotation_id", saved)
            out_name = f"{saved:06d}_{mode_name}_{VA.safe_name(cls)}_img{VA.safe_name(image_id)}_ann{VA.safe_name(ann_id)}.png"
            overlay.save(out_dir / out_name)

            saved += 1
            pbar.set_description(f"view-hungarian-pseudo saved={saved}")

        if args.max_images > 0 and saved >= args.max_images:
            break

    print(f"[done] saved={saved} out_dir={out_dir}")
    print(f"[skipped] no_image={skipped_no_image}, no_pseudo={skipped_no_pseudo}")


if __name__ == "__main__":
    main()
