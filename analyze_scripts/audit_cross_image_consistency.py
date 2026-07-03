#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-image part consistency audit v2 for OVPS / Talk2DINO.

This version supports your current test15 annotation schema:

    ['id', 'image_id', 'category_id', 'class_name',
     'part_category_id', 'part_class_name', 'caption', 'part_caption',
     'cropaug_patch_tokens', 'cropaug_box_xyxy',
     'ann_feats', 'part_ann_feats']

It does NOT require part_gt_mask_patch.

Important:
    This is a maskless per-annotation audit. Each annotation is treated as
    one object-part sample. The part prototype is computed by mean-pooling
    its cropaug_patch_tokens after optional context suppression.

Use this when each annotation already corresponds to a target part.
If cropaug_patch_tokens are actually identical object-crop tokens repeated
for different parts, then this audit will not be meaningful; in that case
you need a feature file that also stores part_gt_mask_patch or patch-level
part masks.
"""

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--feature_pth", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/test15_cic_audit_v2")

    p.add_argument("--ann_key", type=str, default="annotations")
    p.add_argument("--patch_key", type=str, default="cropaug_patch_tokens")
    p.add_argument("--obj_key", type=str, default="category_id")
    p.add_argument("--obj_name_key", type=str, default="class_name")
    p.add_argument("--part_key", type=str, default="part_category_id")
    p.add_argument("--part_name_key", type=str, default="part_class_name")
    p.add_argument("--instance_keys", type=str, nargs="*", default=["image_id", "category_id", "cropaug_box_xyxy"])

    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inspect", type=int, default=0)
    p.add_argument("--max_items", type=int, default=-1)
    p.add_argument("--progress_every", type=int, default=500)

    p.add_argument("--mean_alphas", type=float, nargs="*", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--pca_ranks", type=int, nargs="*", default=[1, 2, 4])
    p.add_argument("--combined_alpha", type=float, default=0.5)
    p.add_argument("--combined_pca_ranks", type=int, nargs="*", default=[1, 2])

    p.add_argument("--structure_pairs", type=int, default=1024)
    p.add_argument("--max_pair_samples", type=int, default=30000)
    p.add_argument("--max_nn_total", type=int, default=5000)

    return p.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def describe_value(v):
    if isinstance(v, torch.Tensor):
        return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
    if isinstance(v, np.ndarray):
        return f"ndarray(shape={v.shape}, dtype={v.dtype})"
    if isinstance(v, list):
        if len(v) == 0:
            return "list(len=0)"
        return f"list(len={len(v)}, first={describe_value(v[0])})"
    if isinstance(v, tuple):
        return f"tuple(len={len(v)})"
    if isinstance(v, dict):
        return f"dict(keys={list(v.keys())[:30]})"
    return f"{type(v).__name__}: {repr(v)[:160]}"


def load_data(path):
    return torch.load(path, map_location="cpu")


def get_annotations(data, ann_key):
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported feature file type: {type(data)}")
    if ann_key not in data:
        raise KeyError(f"ann_key={ann_key!r} not found. Top-level keys={list(data.keys())}")
    anns = data[ann_key]
    if not isinstance(anns, list):
        raise TypeError(f"data[{ann_key!r}] should be list, got {type(anns)}")
    return anns


def inspect_data(data, ann_key, n):
    print("=" * 100)
    print("[INSPECT] top-level:", describe_value(data))
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"  - {k}: {describe_value(v)}")

    anns = get_annotations(data, ann_key)
    print("=" * 100)
    print(f"[INSPECT] num annotations = {len(anns)}")

    for i, ann in enumerate(anns[:n]):
        print("=" * 100)
        print(f"[INSPECT] annotation #{i}")
        if not isinstance(ann, dict):
            print(describe_value(ann))
            continue
        for k, v in ann.items():
            print(f"  - {k}: {describe_value(v)}")


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (list, tuple)):
        return torch.tensor(x)
    raise TypeError(f"Cannot convert {type(x)} to tensor")


def l2n(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def scalar_to_str(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return str(x.item())
        return str(x.detach().cpu().tolist())
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return str(x.item())
        return str(x.tolist())
    return str(x)


def field_to_stable_str(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        if v.numel() <= 16:
            return str(v.reshape(-1).tolist())
        return f"Tensor{tuple(v.shape)}_" + "_".join([f"{float(x):.1f}" for x in v.reshape(-1)[:8]])
    if isinstance(v, np.ndarray):
        if v.size <= 16:
            return str(v.reshape(-1).tolist())
        return f"ndarray{v.shape}_" + "_".join([f"{float(x):.1f}" for x in v.reshape(-1)[:8]])
    if isinstance(v, (list, tuple)):
        if len(v) <= 16:
            return str(v)
        return str(v[:8])
    return str(v)


def get_instance_id(ann, keys):
    parts = []
    for k in keys:
        if k in ann:
            parts.append(f"{k}={field_to_stable_str(ann[k])}")
    if not parts:
        # Fallback: each annotation is its own instance.
        return f"ann_id={field_to_stable_str(ann.get('id', 'unknown'))}"
    return "|".join(parts)


def flatten_patch_tokens(x, embed_dim):
    """
    Supports:
        [N, D]
        [H, W, D]
        [D, H, W]
        [A, N, D]
        [A, H, W, D]
        [A, D, H, W]
    Returns [N_all, D].
    """
    x = to_tensor(x)

    if x.ndim == 2:
        if x.shape[-1] != embed_dim:
            raise ValueError(f"Expected last dim {embed_dim}, got {tuple(x.shape)}")
        return x

    if x.ndim == 3:
        # [A, N, D] or [H, W, D]
        if x.shape[-1] == embed_dim:
            return x.reshape(-1, embed_dim)

        # [D, H, W]
        if x.shape[0] == embed_dim:
            return x.permute(1, 2, 0).reshape(-1, embed_dim)

    if x.ndim == 4:
        # [A, H, W, D]
        if x.shape[-1] == embed_dim:
            return x.reshape(-1, embed_dim)

        # [A, D, H, W]
        if x.shape[1] == embed_dim:
            return x.permute(0, 2, 3, 1).reshape(-1, embed_dim)

    raise ValueError(
        f"Unsupported {tuple(x.shape)} for embed_dim={embed_dim}. "
        "Expected [N,D], [H,W,D], [D,H,W], [A,N,D], [A,H,W,D], or [A,D,H,W]."
    )


def orthonormalize_rows(basis):
    if basis.numel() == 0:
        return basis
    basis = l2n(basis, dim=-1)
    q, _ = torch.linalg.qr(basis.t(), mode="reduced")
    out = q.t()
    keep = out.norm(dim=-1) > 1e-6
    return out[keep]


def compute_mean_basis(x):
    if x.shape[0] == 0:
        return torch.empty((0, x.shape[-1]), dtype=x.dtype, device=x.device)
    return orthonormalize_rows(x.mean(dim=0, keepdim=True))


def compute_pca_basis(x, max_rank):
    if max_rank <= 0 or x.shape[0] < 3:
        return torch.empty((0, x.shape[-1]), dtype=x.dtype, device=x.device)

    max_rank = min(int(max_rank), x.shape[0] - 1, x.shape[1])
    if max_rank <= 0:
        return torch.empty((0, x.shape[-1]), dtype=x.dtype, device=x.device)

    xc = x - x.mean(dim=0, keepdim=True)
    try:
        _, _, v = torch.pca_lowrank(xc, q=max_rank, center=False, niter=2)
        basis = v[:, :max_rank].t().contiguous()
    except RuntimeError:
        _, _, vh = torch.linalg.svd(xc.detach().float().cpu(), full_matrices=False)
        basis = vh[:max_rank].to(x.device, dtype=x.dtype)

    return orthonormalize_rows(basis)


def remove_basis(x, basis, alpha):
    if basis is None or basis.numel() == 0:
        return x
    basis = basis.to(x.device, dtype=x.dtype)
    proj = (x @ basis.t()) @ basis
    return l2n(x - float(alpha) * proj, dim=-1)


def mode_names(args):
    names = ["raw"]
    for a in args.mean_alphas:
        names.append(f"mean_a{a:g}")
    for r in args.pca_ranks:
        names.append(f"pca_r{int(r)}")
    for r in args.combined_pca_ranks:
        names.append(f"mean_a{args.combined_alpha:g}_pca_r{int(r)}")
    return names


def transform_tokens(x, args):
    out = {"raw": x}

    mean_basis = compute_mean_basis(x)

    max_rank = 0
    if args.pca_ranks:
        max_rank = max(max_rank, max(args.pca_ranks))
    if args.combined_pca_ranks:
        max_rank = max(max_rank, max(args.combined_pca_ranks))
    pca_basis = compute_pca_basis(x, max_rank)

    for a in args.mean_alphas:
        out[f"mean_a{a:g}"] = remove_basis(x, mean_basis, a)

    for r in args.pca_ranks:
        r = int(r)
        out[f"pca_r{r}"] = remove_basis(x, pca_basis[:r], 1.0)

    for r in args.combined_pca_ranks:
        r = int(r)
        if pca_basis.shape[0] > 0:
            basis = torch.cat([mean_basis, pca_basis[:r]], dim=0)
        else:
            basis = mean_basis
        basis = orthonormalize_rows(basis)
        out[f"mean_a{args.combined_alpha:g}_pca_r{r}"] = remove_basis(x, basis, args.combined_alpha)

    return out


def prototype_from_tokens(z):
    proto = z.mean(dim=0)
    return l2n(proto, dim=-1)


def update_structure(stats, mode, obj, x, z, num_pairs):
    if num_pairs <= 0 or x.shape[0] < 4:
        return

    n = x.shape[0]
    m = min(int(num_pairs), n * (n - 1) // 2)
    if m <= 0:
        return

    a = torch.randint(0, n, (m,), device=x.device)
    b = torch.randint(0, n, (m,), device=x.device)
    keep = a.ne(b)
    a, b = a[keep], b[keep]
    if a.numel() < 4:
        return

    sx = (x[a] * x[b]).sum(dim=-1).detach().float().cpu().numpy()
    sz = (z[a] * z[b]).sum(dim=-1).detach().float().cpu().numpy()

    mse = float(np.mean((sx - sz) ** 2))
    corr = float("nan")
    if np.std(sx) >= 1e-8 and np.std(sz) >= 1e-8:
        corr = float(np.corrcoef(sx, sz)[0, 1])

    stats[mode][obj]["sum_mse"] += mse
    stats[mode][obj]["n_mse"] += 1
    if not math.isnan(corr):
        stats[mode][obj]["sum_corr"] += corr
        stats[mode][obj]["n_corr"] += 1


def avg_same_part(vectors, max_pair_samples, rng):
    n = len(vectors)
    if n < 2:
        return float("nan")
    v = np.stack(vectors).astype(np.float32)
    if n <= 2000:
        s = v @ v.T
        iu = np.triu_indices(n, k=1)
        return float(s[iu].mean())
    vals = []
    for _ in range(max_pair_samples):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        vals.append(float(np.dot(v[i], v[j])))
    return float(np.mean(vals)) if vals else float("nan")


def avg_diff_cross(entries, max_pair_samples, rng):
    n = len(entries)
    if n < 2:
        return float("nan")
    vals = []
    trials = 0
    max_trials = max_pair_samples * 20
    while len(vals) < max_pair_samples and trials < max_trials:
        trials += 1
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        pi, ii, vi = entries[i]
        pj, ij, vj = entries[j]
        if pi == pj or ii == ij:
            continue
        vals.append(float(np.dot(vi, vj)))
    return float(np.mean(vals)) if vals else float("nan")


def avg_same_instance_diff(instance_entries):
    vals = []
    for _, plist in instance_entries.items():
        if len(plist) < 2:
            continue
        for i in range(len(plist)):
            pi, vi = plist[i]
            for j in range(i + 1, len(plist)):
                pj, vj = plist[j]
                if pi == pj:
                    continue
                vals.append(float(np.dot(vi, vj)))
    return float(np.mean(vals)) if vals else float("nan")


def nn_purity(entries, max_nn_total, rng):
    n = len(entries)
    if n < 2:
        return float("nan")

    if n > max_nn_total:
        keep = rng.sample(range(n), max_nn_total)
        entries = [entries[i] for i in keep]
        n = len(entries)

    parts = np.array([e[0] for e in entries])
    insts = np.array([e[1] for e in entries])
    v = np.stack([e[2] for e in entries]).astype(np.float32)

    s = v @ v.T
    same_inst = insts[:, None] == insts[None, :]
    s[same_inst] = -np.inf

    valid = np.isfinite(s).any(axis=1)
    if valid.sum() == 0:
        return float("nan")

    nn = np.argmax(s, axis=1)
    ok = parts[nn] == parts
    return float(ok[valid].mean())


def safe_mean(vals):
    out = []
    for x in vals:
        try:
            x = float(x)
        except Exception:
            continue
        if not math.isnan(x) and not math.isinf(x):
            out.append(x)
    if not out:
        return float("nan")
    return float(np.mean(out))


def fmt(x):
    try:
        x = float(x)
    except Exception:
        return x
    if math.isnan(x) or math.isinf(x):
        return ""
    return f"{x:.6f}"


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    ensure_dir(args.out_dir)

    print(f"[INFO] loading: {args.feature_pth}")
    data = load_data(args.feature_pth)

    if args.inspect > 0:
        inspect_data(data, args.ann_key, args.inspect)
        return

    anns = get_annotations(data, args.ann_key)
    if args.max_items > 0:
        anns = anns[: args.max_items]

    print(f"[INFO] num annotations = {len(anns)}")
    print(f"[INFO] patch_key = {args.patch_key}")
    print(f"[INFO] obj_key = {args.obj_key}")
    print(f"[INFO] part_key = {args.part_key}")
    print(f"[INFO] instance_keys = {args.instance_keys}")

    modes = mode_names(args)
    print(f"[INFO] modes = {modes}")

    schema_path = os.path.join(args.out_dir, "schema_first_annotation.txt")
    with open(schema_path, "w", encoding="utf-8") as f:
        if len(anns) > 0 and isinstance(anns[0], dict):
            for k, v in anns[0].items():
                f.write(f"{k}: {describe_value(v)}\n")
        else:
            f.write("No valid first annotation.\n")

    # records[mode][obj][part] = list of (instance_id, vector)
    records = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    instance_records = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    part_name_map = {}
    obj_name_map = {}
    token_count_stats = defaultdict(lambda: defaultdict(list))
    structure_stats = defaultdict(lambda: defaultdict(lambda: {"sum_corr": 0.0, "n_corr": 0, "sum_mse": 0.0, "n_mse": 0}))

    skipped = 0
    used = 0

    for idx, ann in enumerate(anns):
        if not isinstance(ann, dict):
            skipped += 1
            continue

        missing = [k for k in [args.patch_key, args.obj_key, args.part_key] if k not in ann]
        if missing:
            raise KeyError(f"Annotation #{idx} missing keys={missing}. Available keys={list(ann.keys())}")

        obj = scalar_to_str(ann[args.obj_key])
        part = scalar_to_str(ann[args.part_key])
        obj_name = scalar_to_str(ann.get(args.obj_name_key, obj))
        part_name = scalar_to_str(ann.get(args.part_name_key, part))
        inst = get_instance_id(ann, args.instance_keys)

        obj_name_map[obj] = obj_name
        part_name_map[(obj, part)] = part_name

        x = flatten_patch_tokens(ann[args.patch_key], args.embed_dim)
        x = x.detach().float().to(device)
        if x.shape[0] < 1:
            skipped += 1
            continue
        x = l2n(x, dim=-1)

        transformed = transform_tokens(x, args)

        for mode in modes:
            z = transformed[mode]
            proto = prototype_from_tokens(z).detach().float().cpu().numpy().astype(np.float32)

            records[mode][obj][part].append((inst, proto))
            instance_records[mode][obj][inst].append((part, proto))
            token_count_stats[mode][obj].append(int(x.shape[0]))

            update_structure(structure_stats, mode, obj, x, z, args.structure_pairs)

        used += 1

        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            print(f"[PROGRESS] {idx + 1}/{len(anns)} annotations processed")

    print(f"[INFO] used = {used}, skipped = {skipped}")

    object_rows = []
    part_rows = []

    for mode in modes:
        for obj in sorted(records[mode].keys()):
            part_dict = records[mode][obj]

            entries = []
            same_values = []

            for part, plist in sorted(part_dict.items(), key=lambda kv: kv[0]):
                vectors = [v for _, v in plist]
                sp = avg_same_part(vectors, args.max_pair_samples, rng)
                if not math.isnan(sp):
                    same_values.append(sp)

                part_rows.append({
                    "mode": mode,
                    "object": obj,
                    "object_name": obj_name_map.get(obj, ""),
                    "part": part,
                    "part_name": part_name_map.get((obj, part), ""),
                    "n_samples": len(plist),
                    "same_part_cross": fmt(sp),
                })

                for inst, vec in plist:
                    entries.append((part, inst, vec))

            same_part_cross = safe_mean(same_values)
            diff_part_cross = avg_diff_cross(entries, args.max_pair_samples, rng)
            same_instance_diff = avg_same_instance_diff(instance_records[mode][obj])
            purity = nn_purity(entries, args.max_nn_total, rng)
            cross_gap = same_part_cross - diff_part_cross if not (math.isnan(same_part_cross) or math.isnan(diff_part_cross)) else float("nan")

            st = structure_stats[mode][obj]
            struct_corr = st["sum_corr"] / st["n_corr"] if st["n_corr"] > 0 else float("nan")
            struct_mse = st["sum_mse"] / st["n_mse"] if st["n_mse"] > 0 else float("nan")

            n_instances = len({inst for _, inst, _ in entries})
            n_samples = len(entries)

            object_rows.append({
                "mode": mode,
                "object": obj,
                "object_name": obj_name_map.get(obj, ""),
                "n_instances": n_instances,
                "n_parts": len(part_dict),
                "n_samples": n_samples,
                "mean_tokens_per_sample": fmt(safe_mean(token_count_stats[mode][obj])),
                "same_part_cross": fmt(same_part_cross),
                "diff_part_cross": fmt(diff_part_cross),
                "same_instance_diff": fmt(same_instance_diff),
                "cross_gap": fmt(cross_gap),
                "nn_purity": fmt(purity),
                "structure_corr": fmt(struct_corr),
                "structure_mse": fmt(struct_mse),
            })

    object_csv = os.path.join(args.out_dir, "per_object_summary.csv")
    part_csv = os.path.join(args.out_dir, "per_part_summary.csv")

    write_csv(
        object_csv,
        object_rows,
        [
            "mode", "object", "object_name",
            "n_instances", "n_parts", "n_samples", "mean_tokens_per_sample",
            "same_part_cross", "diff_part_cross", "same_instance_diff",
            "cross_gap", "nn_purity", "structure_corr", "structure_mse",
        ],
    )

    write_csv(
        part_csv,
        part_rows,
        [
            "mode", "object", "object_name", "part", "part_name",
            "n_samples", "same_part_cross",
        ],
    )

    summary = {
        "feature_pth": args.feature_pth,
        "schema": "maskless_per_annotation_part_prototype",
        "num_annotations": len(anns),
        "used": used,
        "skipped": skipped,
        "notes": [
            "Each annotation is treated as one object-part sample.",
            "The prototype is mean-pooled from cropaug_patch_tokens after optional context suppression.",
            "If cropaug_patch_tokens are repeated object-crop tokens rather than part-specific tokens, this audit is not reliable.",
        ],
        "modes": {},
    }

    for mode in modes:
        rows = [r for r in object_rows if r["mode"] == mode]

        def col(name):
            vals = []
            for r in rows:
                v = r[name]
                if v == "":
                    continue
                vals.append(float(v))
            return vals

        summary["modes"][mode] = {
            "num_objects": len(rows),
            "macro_same_part_cross": safe_mean(col("same_part_cross")),
            "macro_diff_part_cross": safe_mean(col("diff_part_cross")),
            "macro_same_instance_diff": safe_mean(col("same_instance_diff")),
            "macro_cross_gap": safe_mean(col("cross_gap")),
            "macro_nn_purity": safe_mean(col("nn_purity")),
            "macro_structure_corr": safe_mean(col("structure_corr")),
            "macro_structure_mse": safe_mean(col("structure_mse")),
        }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    console_path = os.path.join(args.out_dir, "console_summary.txt")
    lines = []
    lines.append("=" * 100)
    lines.append("[SUMMARY] maskless per-annotation cross-image consistency audit")
    lines.append(f"feature_pth: {args.feature_pth}")
    lines.append(f"num_annotations: {len(anns)}")
    lines.append(f"used: {used}")
    lines.append(f"skipped: {skipped}")
    lines.append(f"out_dir: {args.out_dir}")
    lines.append("-" * 100)

    for mode in modes:
        m = summary["modes"][mode]
        lines.append(
            f"{mode:>20s} | "
            f"same={m['macro_same_part_cross']:.6f} | "
            f"diff={m['macro_diff_part_cross']:.6f} | "
            f"same_inst_diff={m['macro_same_instance_diff']:.6f} | "
            f"gap={m['macro_cross_gap']:.6f} | "
            f"nn={m['macro_nn_purity']:.6f} | "
            f"struct_corr={m['macro_structure_corr']:.6f} | "
            f"struct_mse={m['macro_structure_mse']:.6f}"
        )

    text = "\n".join(lines)
    print(text)
    with open(console_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print("=" * 100)
    print(f"[DONE] wrote {summary_path}")
    print(f"[DONE] wrote {object_csv}")
    print(f"[DONE] wrote {part_csv}")
    print(f"[DONE] wrote {console_path}")
    print(f"[DONE] wrote {schema_path}")


if __name__ == "__main__":
    main()
