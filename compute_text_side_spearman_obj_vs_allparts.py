import argparse
import importlib
from collections import defaultdict

import numpy as np
import torch
import yaml


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / a.norm().clamp_min(1e-12)
    b = b / b.norm().clamp_min(1e-12)
    return float((a * b).sum().item())


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    if x.size < 2:
        return float("nan")
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def build_part_bank(annotations, part_field="part_ann_feats"):
    bucket = defaultdict(list)
    class_name_map = {}
    for ann in annotations:
        cat = ann.get("category_id", None)
        if cat is None:
            continue
        cat = int(cat)
        class_name_map[cat] = ann.get("class_name", str(cat))

        part_ids = ann.get("part_category_id", []) or []
        part_names = ann.get("part_class_name", []) or []
        part_feats = ann.get(part_field, None)
        if part_feats is None or (not torch.is_tensor(part_feats)) or part_feats.ndim < 2:
            continue

        k = min(len(part_ids), len(part_names), part_feats.shape[0])
        for i in range(k):
            key = (cat, int(part_ids[i]), str(part_names[i]))
            bucket[key].append(part_feats[i].detach().float().cpu().view(-1))

    class_to_parts = defaultdict(list)
    for (cat, pid, pname), vals in bucket.items():
        feat = torch.stack(vals, dim=0).mean(dim=0)
        class_to_parts[cat].append((pid, pname, feat))
    for cat in class_to_parts:
        class_to_parts[cat] = sorted(class_to_parts[cat], key=lambda t: t[0])
    return class_to_parts, class_name_map


def build_class_obj_map(annotations, obj_field="ann_feats"):
    out = {}
    for ann in annotations:
        cat = ann.get("category_id", None)
        if cat is None or obj_field not in ann:
            continue
        cat = int(cat)
        if cat not in out:
            out[cat] = {
                "class_name": ann.get("class_name", str(cat)),
                "obj_feat": ann[obj_field].detach().float().cpu().view(-1),
            }
    return out


def cosine_scores(obj_feat: torch.Tensor, part_feats: torch.Tensor) -> np.ndarray:
    obj_feat = normalize_rows(obj_feat.view(1, -1))[0]
    part_feats = normalize_rows(part_feats)
    return (part_feats @ obj_feat).detach().cpu().numpy()


def upper_triangle_vector_from_cosine_matrix(feats: torch.Tensor) -> np.ndarray:
    feats = normalize_rows(feats.float().cpu())
    sim = feats @ feats.t()
    k = sim.shape[0]
    idx = torch.triu_indices(k, k, offset=1)
    return sim[idx[0], idx[1]].detach().cpu().numpy().astype(np.float64)


def load_projector(model_config_path: str, projector_ckpt: str, device: str):
    with open(model_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"])
    ckpt = torch.load(projector_ckpt, map_location="cpu")
    ret = model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    return model, ret


@torch.no_grad()
def project_feats(model, feats: torch.Tensor, device: str) -> torch.Tensor:
    x = feats.to(device=device, dtype=torch.float32)
    z = model.project_clip_txt(x)
    return z.detach().float().cpu()


def print_obj_part_spearman(args, anns_a, anns_b):
    bank_annotations = anns_a if args.bank_source == "a" else anns_b
    class_to_parts, _ = build_part_bank(bank_annotations, part_field=args.part_field)
    class_obj_a = build_class_obj_map(anns_a, obj_field=args.obj_field)
    class_obj_b = build_class_obj_map(anns_b, obj_field=args.obj_field)
    common_cats = sorted(set(class_obj_a.keys()) & set(class_obj_b.keys()) & set(class_to_parts.keys()))

    print("=" * 110)
    print("PER-CLASS OBJ-vs-ALLPARTS TEXT-SIDE COMPARISON")
    print(f"A (obj source): {args.pth_a}")
    print(f"B (obj source): {args.pth_b}")
    print(f"Fixed part bank source: {'A' if args.bank_source == 'a' else 'B'}")
    print(f"Object field: {args.obj_field}")
    print(f"Part field: {args.part_field}")
    print(f"Common object classes: {len(common_cats)}")
    print("=" * 110)

    rows = []
    for cat in common_cats:
        part_rows = class_to_parts[cat]
        class_name = class_obj_a[cat]["class_name"]
        obj_a = class_obj_a[cat]["obj_feat"]
        obj_b = class_obj_b[cat]["obj_feat"]
        obj_cos = cosine(obj_a, obj_b)
        if len(part_rows) >= 2:
            part_feats = torch.stack([feat for _, _, feat in part_rows], dim=0)
            rho = spearman_rho(cosine_scores(obj_a, part_feats), cosine_scores(obj_b, part_feats))
        else:
            rho = float("nan")
        rows.append((class_name, cat, len(part_rows), rho, obj_cos))

    valid_spearman = np.asarray([r[3] for r in rows if not np.isnan(r[3])], dtype=np.float64)
    valid_obj_cos = np.asarray([r[4] for r in rows], dtype=np.float64)

    print("GLOBAL SUMMARY")
    print(f"num classes (obj cosine)             : {len(rows)}")
    print(f"num classes (valid spearman)         : {len(valid_spearman)}")
    if len(valid_spearman) > 0:
        print(f"mean Spearman                        : {float(np.mean(valid_spearman)):.8f}")
    print(f"mean obj cosine                      : {float(np.mean(valid_obj_cos)):.8f}")
    print("-" * 110)
    print(f"{'class_name':<26} {'category_id':<11} {'num_parts':<10} {'spearman':<14} {'obj_cos':<14}")
    for class_name, cat, num_parts, rho, obj_cos in rows:
        rho_str = "nan" if np.isnan(rho) else f"{rho:.8f}"
        print(f"{class_name:<26} {cat:<11d} {num_parts:<10d} {rho_str:<14} {obj_cos:.8f}")
    print("-" * 110)


def print_projector_prepost_metrics(args, annotations):
    print("=" * 110)
    print("PER-CLASS PROJECTOR PRE/POST METRICS")
    print(f"Projector source: {args.projector_source.upper()}")
    print(f"Part field      : {args.part_field}")
    print(f"Object field    : {args.obj_field}")
    print(f"Model config    : {args.model_config}")
    print(f"Projector       : {args.projector_ckpt}")
    print("=" * 110)

    model, ret = load_projector(args.model_config, args.projector_ckpt, args.device)
    if ret is not None:
        print("Missing keys   :", getattr(ret, "missing_keys", []))
        print("Unexpected keys:", getattr(ret, "unexpected_keys", []))

    class_to_parts, class_name_map = build_part_bank(annotations, part_field=args.part_field)
    class_obj_map = build_class_obj_map(annotations, obj_field=args.obj_field)

    common_cats = sorted(set(class_to_parts.keys()) & set(class_obj_map.keys()))
    rows = []
    for cat in common_cats:
        class_name = class_name_map.get(cat, class_obj_map[cat]["class_name"])
        part_rows = class_to_parts[cat]
        obj_pre = class_obj_map[cat]["obj_feat"]                  # [D_text]
        obj_post = project_feats(model, obj_pre.view(1, -1), args.device)[0]  # [D_proj]
        num_parts = len(part_rows)

        if num_parts == 0:
            rows.append((class_name, cat, 0, float("nan"), float("nan")))
            continue

        part_pre = torch.stack([feat for _, _, feat in part_rows], dim=0)      # [K, D_text]
        part_post = project_feats(model, part_pre, args.device)                # [K, D_proj]

        # 1) part-part graph pre/post Spearman
        if num_parts >= 2:
            graph_rho = spearman_rho(
                upper_triangle_vector_from_cosine_matrix(part_pre),
                upper_triangle_vector_from_cosine_matrix(part_post),
            )
        else:
            graph_rho = float("nan")

        # 2) part-obj relation pre/post Spearman
        if num_parts >= 2:
            pre_scores = cosine_scores(obj_pre, part_pre)
            post_scores = cosine_scores(obj_post, part_post)
            objrel_rho = spearman_rho(pre_scores, post_scores)
        else:
            objrel_rho = float("nan")

        rows.append((class_name, cat, num_parts, graph_rho, objrel_rho))

    valid_graph = np.asarray([r[3] for r in rows if not np.isnan(r[3])], dtype=np.float64)
    valid_objrel = np.asarray([r[4] for r in rows if not np.isnan(r[4])], dtype=np.float64)

    print("GLOBAL SUMMARY")
    print(f"num classes total                    : {len(rows)}")
    print(f"num classes valid graph spearman     : {len(valid_graph)}")
    print(f"num classes valid objrel spearman    : {len(valid_objrel)}")
    if len(valid_graph) > 0:
        print(f"mean graph spearman(pre vs post)     : {float(np.mean(valid_graph)):.8f}")
    if len(valid_objrel) > 0:
        print(f"mean objrel spearman(pre vs post)    : {float(np.mean(valid_objrel)):.8f}")
    print("-" * 110)
    print(
        f"{'class_name':<26} {'category_id':<11} {'num_parts':<10} "
        f"{'graph_spear':<16} {'objrel_spear':<16}"
    )
    for class_name, cat, num_parts, graph_rho, objrel_rho in rows:
        graph_str = "nan" if np.isnan(graph_rho) else f"{graph_rho:.8f}"
        objrel_str = "nan" if np.isnan(objrel_rho) else f"{objrel_rho:.8f}"
        print(f"{class_name:<26} {cat:<11d} {num_parts:<10d} {graph_str:<16} {objrel_str:<16}")
    print("-" * 110)
    print("NOTE")
    print("graph_spear  = Spearman( upper-tri(cos(part_pre, part_pre)), upper-tri(cos(part_post, part_post)) )")
    print("objrel_spear = Spearman( cos(part_pre, obj_pre),            cos(part_post, obj_post) )")
    print("There is NO direct cosine between obj_pre and obj_post, because they live in different dimensions/spaces.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_a", required=True)
    parser.add_argument("--pth_b", default=None)
    parser.add_argument("--obj_field", default="ann_feats")
    parser.add_argument("--part_field", default="part_ann_feats")
    parser.add_argument("--bank_source", choices=["a", "b"], default="a")
    parser.add_argument("--projector_source", choices=["a", "b"], default="a")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_obj_compare", action="store_true", default=False)
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--projector_ckpt", default=None)
    args = parser.parse_args()

    data_a = torch.load(args.pth_a, map_location="cpu")
    anns_a = data_a["annotations"]

    anns_b = None
    if args.pth_b is not None:
        data_b = torch.load(args.pth_b, map_location="cpu")
        anns_b = data_b["annotations"]

    if not args.skip_obj_compare:
        if anns_b is None:
            raise ValueError("Original obj-vs-allparts comparison needs --pth_b unless you pass --skip_obj_compare.")
        print_obj_part_spearman(args, anns_a, anns_b)

    if args.model_config is not None and args.projector_ckpt is not None:
        projector_annotations = anns_a if args.projector_source == "a" else anns_b
        if projector_annotations is None:
            raise ValueError("projector_source=b requires --pth_b.")
        print_projector_prepost_metrics(args, projector_annotations)


if __name__ == "__main__":
    main()
