#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Stage3-GW audit curves for two Talk2DINO history JSON files.

Usage:
  python plot_stage3_gw_audit_curves.py \
    --zpart weights/vitb_mlp_infonce_stage3_gw_voc116_obj_with_part_test8_gw_1st_history.json \
    --anchor weights/vitb_mlp_infonce_stage3_gw_voc116_obj_with_part_test8_gw_anchor_1st_history.json \
    --outdir audit_curve_plots

This script plots validation curves by default:
  1) post Spearman: audit_spear_post_text_vs_visual
  2) post STR:      audit_strret_post_text_vs_visual
  3) anchor hit:    anchor_hit_rate_post
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


METRICS = [
    ("audit_spear_post_text_vs_visual", "Post Spearman vs V", "post_spearman_curve.png"),
    ("audit_strret_post_text_vs_visual", "Post STR retrieval vs V", "post_str_curve.png"),
    ("anchor_hit_rate_post", "Anchor hit rate post", "anchor_hit_rate_post_curve.png"),
]


def load_history(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_series(history: Dict[str, Any], split: str, key: str) -> List[float]:
    rows = history.get(split, [])
    values = []
    for row in rows:
        val = row.get(key, float("nan"))
        if val is None:
            val = float("nan")
        values.append(float(val))
    return values


def finite_argmax(values: List[float]) -> Tuple[int, float]:
    best_i = -1
    best_v = -float("inf")
    for i, v in enumerate(values):
        if math.isfinite(v) and v > best_v:
            best_i, best_v = i, v
    return best_i, best_v


def finite_last(values: List[float]) -> float:
    for v in reversed(values):
        if math.isfinite(v):
            return v
    return float("nan")


def plot_metric(
    z_values: List[float],
    a_values: List[float],
    title: str,
    ylabel: str,
    outpath: Path,
    z_label: str,
    a_label: str,
) -> None:
    z_epochs = list(range(len(z_values)))
    a_epochs = list(range(len(a_values)))

    plt.figure(figsize=(9, 5))
    plt.plot(z_epochs, z_values, marker="o", markersize=2, linewidth=1.5, label=z_label)
    plt.plot(a_epochs, a_values, marker="s", markersize=2, linewidth=1.5, label=a_label)

    zi, zv = finite_argmax(z_values)
    ai, av = finite_argmax(a_values)

    if zi >= 0:
        plt.scatter([zi], [zv], s=60)
        plt.annotate(f"zpart max\n{zv:.4f}@{zi}", (zi, zv), textcoords="offset points", xytext=(8, 8))
    if ai >= 0:
        plt.scatter([ai], [av], s=60)
        plt.annotate(f"anchor max\n{av:.4f}@{ai}", (ai, av), textcoords="offset points", xytext=(8, -22))

    plt.title(title)
    plt.xlabel("Epoch / validation index")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zpart", required=True, type=Path, help="z_part prototype history JSON")
    parser.add_argument("--anchor", required=True, type=Path, help="anchor prototype history JSON")
    parser.add_argument("--outdir", default=Path("audit_curve_plots"), type=Path)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--z-label", default="z_part prototype GW")
    parser.add_argument("--anchor-label", default="anchor-token prototype GW")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    z_hist = load_history(args.zpart)
    a_hist = load_history(args.anchor)

    summary_rows = []
    for key, title, filename in METRICS:
        z_values = get_series(z_hist, args.split, key)
        a_values = get_series(a_hist, args.split, key)

        plot_metric(
            z_values,
            a_values,
            title=f"{title} ({args.split})",
            ylabel=title,
            outpath=args.outdir / filename,
            z_label=args.z_label,
            a_label=args.anchor_label,
        )

        zi, zv = finite_argmax(z_values)
        ai, av = finite_argmax(a_values)
        summary_rows.append(
            {
                "metric": key,
                "zpart_peak": zv,
                "zpart_peak_epoch": zi,
                "zpart_final": finite_last(z_values),
                "anchor_peak": av,
                "anchor_peak_epoch": ai,
                "anchor_final": finite_last(a_values),
            }
        )

    csv_path = args.outdir / "audit_curve_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("metric,zpart_peak,zpart_peak_epoch,zpart_final,anchor_peak,anchor_peak_epoch,anchor_final\n")
        for r in summary_rows:
            f.write(
                f"{r['metric']},{r['zpart_peak']:.8f},{r['zpart_peak_epoch']},"
                f"{r['zpart_final']:.8f},{r['anchor_peak']:.8f},"
                f"{r['anchor_peak_epoch']},{r['anchor_final']:.8f}\n"
            )

    print(f"Saved plots to: {args.outdir.resolve()}")
    print(f"Saved summary to: {csv_path.resolve()}")
    print("\nSummary:")
    for r in summary_rows:
        print(
            f"- {r['metric']}: "
            f"zpart peak={r['zpart_peak']:.4f}@{r['zpart_peak_epoch']}, final={r['zpart_final']:.4f}; "
            f"anchor peak={r['anchor_peak']:.4f}@{r['anchor_peak_epoch']}, final={r['anchor_final']:.4f}"
        )


if __name__ == "__main__":
    main()
