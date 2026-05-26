from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from collect_fake_gt_prototypes import FakeGTPrototypeCollector
from src.voc116_part_coarse import COARSE_PART_CLASSES, FINE_PART_CLASSES


def safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s)).strip("_")
    return s or "unknown"


def get_part_names(num_parts: int):
    if num_parts == 58:
        return list(COARSE_PART_CLASSES)
    if num_parts == 116:
        return list(FINE_PART_CLASSES)
    return [f"part_{i}" for i in range(num_parts)]


def get_obj_name_from_part_name(part_name: str) -> str:
    """
    Examples:
      "cat's head" -> "cat"
      "dog’s eye" -> "dog"
      "pottedplant's pot" -> "pottedplant"
    """
    m = re.match(r"^(.*?)[’']s\s+", str(part_name))
    if m:
        return m.group(1)
    return "unknown"


def write_matrix_csv(path: Path, sim: torch.Tensor, part_names, row_ids, col_ids):
    """
    sim should already be on CPU or GPU. Values are indexed directly.
    CSV writing is naturally row-wise, but all numeric computation is already done by GPU matmul.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fake\\gt"] + [f"{j}:{part_names[j]}" for j in col_ids])

        sim_cpu = sim.detach().cpu()

        for i in row_ids:
            row = [f"{i}:{part_names[i]}"]
            vals = sim_cpu[i, col_ids].tolist()
            row.extend([f"{v:.8f}" for v in vals])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init_weights", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show_progress", action="store_true")

    args = parser.parse_args()

    collector = FakeGTPrototypeCollector(
        model_config=args.model_config,
        dataset=args.dataset,
        init_weights=args.init_weights,

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
        show_progress=args.show_progress,
    )

    ret = collector.collect()

    if isinstance(ret, tuple) and len(ret) == 3:
        fake_proto, gt_proto, meta = ret
    else:
        fake_proto, gt_proto = ret
        meta = {}

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    fake_proto = F.normalize(fake_proto.to(device).float(), dim=-1)
    gt_proto = F.normalize(gt_proto.to(device).float(), dim=-1)

    sim = fake_proto @ gt_proto.T  # [P, P]

    P = args.num_parts
    part_names = meta.get("part_names", None) if isinstance(meta, dict) else None
    if part_names is None:
        part_names = get_part_names(P)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_ids = sorted(range(P), key=lambda i: str(part_names[i]))

    write_matrix_csv(
        out_dir / "global_fake_rows_gt_cols.csv",
        sim=sim,
        part_names=part_names,
        row_ids=sorted_ids,
        col_ids=sorted_ids,
    )

    diag = sim.diag().detach().cpu()

    per_part_csv = out_dir / "per_part_fake_gt_cosine.csv"
    with open(per_part_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["part_id", "part_name", "object_name", "fake_gt_cosine"],
        )
        writer.writeheader()

        for i in sorted_ids:
            writer.writerow({
                "part_id": i,
                "part_name": part_names[i],
                "object_name": get_obj_name_from_part_name(part_names[i]),
                "fake_gt_cosine": f"{float(diag[i]):.8f}",
            })

    obj_to_ids = {}
    for i, name in enumerate(part_names):
        obj = get_obj_name_from_part_name(name)
        obj_to_ids.setdefault(obj, []).append(i)

    matrix_dir = out_dir / "object_block_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for obj_name in sorted(obj_to_ids.keys()):
        ids = sorted(obj_to_ids[obj_name], key=lambda i: str(part_names[i]))
        if len(ids) == 0:
            continue

        matrix_path = matrix_dir / f"{safe_name(obj_name)}_fake_rows_gt_cols.csv"
        write_matrix_csv(
            matrix_path,
            sim=sim,
            part_names=part_names,
            row_ids=ids,
            col_ids=ids,
        )

        block = sim[ids][:, ids]
        diag_vals = block.diag()

        if len(ids) > 1:
            off_mask = ~torch.eye(len(ids), dtype=torch.bool, device=device)
            off_vals = block[off_mask]
            off_mean = float(off_vals.mean().detach().cpu())
        else:
            off_mean = 0.0

        diag_mean = float(diag_vals.mean().detach().cpu())

        summary_rows.append({
            "object_name": obj_name,
            "num_parts": len(ids),
            "diag_mean": f"{diag_mean:.8f}",
            "offdiag_mean": f"{off_mean:.8f}",
            "diag_minus_offdiag": f"{diag_mean - off_mean:.8f}",
            "matrix_csv": str(matrix_path),
        })

    summary_csv = out_dir / "object_block_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "object_name",
                "num_parts",
                "diag_mean",
                "offdiag_mean",
                "diag_minus_offdiag",
                "matrix_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("[done]")
    print("fake_proto:", tuple(fake_proto.shape))
    print("gt_proto:", tuple(gt_proto.shape))
    print("[saved]", out_dir / "global_fake_rows_gt_cols.csv")
    print("[saved]", per_part_csv)
    print("[saved]", summary_csv)
    print("[saved matrices]", matrix_dir)


if __name__ == "__main__":
    main()
