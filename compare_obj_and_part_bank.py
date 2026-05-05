import argparse
from collections import defaultdict
import torch


def to_float_cpu(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.detach().float().cpu()


def ann_key(ann, fallback_idx):
    if "id" in ann:
        return ("id", int(ann["id"]))
    return ("fallback", (ann.get("image_id"), ann.get("category_id"), ann.get("class_name"), fallback_idx))


def cosine(a, b):
    a = a / a.norm().clamp_min(1e-12)
    b = b / b.norm().clamp_min(1e-12)
    return float((a * b).sum().item())


def compare_vectors(a, b):
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "l2": float(torch.norm(a - b, p=2).item()),
        "cos": cosine(a, b),
        "identical": bool(torch.equal(a, b)),
    }


def build_part_bank(annotations, part_field="part_ann_feats"):
    bank = defaultdict(list)
    for ann in annotations:
        cat = ann.get("category_id", None)
        part_ids = ann.get("part_category_id", []) or []
        part_names = ann.get("part_class_name", []) or []
        part_feats = ann.get(part_field, None)
        if cat is None or part_feats is None:
            continue
        if not torch.is_tensor(part_feats) or part_feats.ndim < 2:
            continue
        k = min(len(part_ids), len(part_names), part_feats.shape[0])
        for i in range(k):
            key = (int(cat), int(part_ids[i]), str(part_names[i]))
            bank[key].append(to_float_cpu(part_feats[i]).view(-1))
    return bank


def mean_bank(bank):
    out = {}
    for k, vals in bank.items():
        out[k] = torch.stack(vals, dim=0).mean(dim=0)
    return out


def summarize_rows(rows, title, show_topk=10, is_obj=True):
    print("=" * 100)
    print(title)
    print("=" * 100)
    if len(rows) == 0:
        print("No comparable rows.")
        return
    valid = [r for r in rows if not r.get("shape_mismatch", False)]
    if len(valid) == 0:
        print("All rows had shape mismatch.")
        return

    identical = sum(1 for r in valid if r["identical"])
    print(f"count                           : {len(valid)}")
    print(f"identical rows                  : {identical}/{len(valid)}")
    print(f"global max abs diff             : {max(r['max_abs'] for r in valid):.8f}")
    print(f"mean(abs diff)                  : {sum(r['mean_abs'] for r in valid) / len(valid):.8f}")
    print(f"mean L2 distance                : {sum(r['l2'] for r in valid) / len(valid):.8f}")
    print(f"mean cosine similarity          : {sum(r['cos'] for r in valid) / len(valid):.8f}")
    print(f"min cosine similarity           : {min(r['cos'] for r in valid):.8f}")
    print("-" * 100)
    top_rows = sorted(valid, key=lambda r: (r["max_abs"], r["l2"]), reverse=True)[:show_topk]
    print(f"TOP {len(top_rows)} MOST-DIFFERENT")
    for i, r in enumerate(top_rows, 1):
        if is_obj:
            prefix = f"class={r['class_name']} ann_id={r['ann_id']} image_id={r['image_id']}"
        else:
            prefix = f"category_id={r['category_id']} part_id={r['part_id']} part_name={r['part_name']}"
        print(f"[{i}] {prefix} max_abs={r['max_abs']:.8f} mean_abs={r['mean_abs']:.8f} l2={r['l2']:.8f} cos={r['cos']:.8f}")


def compare_obj_features(anns_a, anns_b, field):
    map_a, map_b = {}, {}
    for i, ann in enumerate(anns_a):
        if field in ann:
            map_a[ann_key(ann, i)] = ann
    for i, ann in enumerate(anns_b):
        if field in ann:
            map_b[ann_key(ann, i)] = ann

    common = [k for k in map_a.keys() if k in map_b]
    rows = []
    for key in common:
        ann_a, ann_b = map_a[key], map_b[key]
        fa = to_float_cpu(ann_a[field]).view(-1)
        fb = to_float_cpu(ann_b[field]).view(-1)
        if fa.shape != fb.shape:
            rows.append({"shape_mismatch": True, "class_name": ann_a.get("class_name", ""), "ann_id": ann_a.get("id"), "image_id": ann_a.get("image_id")})
            continue
        stats = compare_vectors(fa, fb)
        stats.update({"shape_mismatch": False, "class_name": ann_a.get("class_name", ""), "ann_id": ann_a.get("id"), "image_id": ann_a.get("image_id")})
        rows.append(stats)
    return rows, len(map_a), len(map_b), len(common)


def compare_part_bank(anns_a, anns_b, part_field):
    bank_a = mean_bank(build_part_bank(anns_a, part_field=part_field))
    bank_b = mean_bank(build_part_bank(anns_b, part_field=part_field))
    keys_a, keys_b = set(bank_a.keys()), set(bank_b.keys())
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    rows = []
    for key in common:
        fa = bank_a[key].view(-1)
        fb = bank_b[key].view(-1)
        cat, pid, pname = key
        if fa.shape != fb.shape:
            rows.append({"shape_mismatch": True, "category_id": cat, "part_id": pid, "part_name": pname})
            continue
        stats = compare_vectors(fa, fb)
        stats.update({"shape_mismatch": False, "category_id": cat, "part_id": pid, "part_name": pname})
        rows.append(stats)
    return rows, bank_a, bank_b, only_a, only_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_a", required=True)
    parser.add_argument("--pth_b", required=True)
    parser.add_argument("--obj_field", default="ann_feats")
    parser.add_argument("--part_field", default="part_ann_feats")
    parser.add_argument("--show_topk", type=int, default=10)
    args = parser.parse_args()

    data_a = torch.load(args.pth_a, map_location="cpu")
    data_b = torch.load(args.pth_b, map_location="cpu")
    anns_a = data_a["annotations"]
    anns_b = data_b["annotations"]

    print("# FILE INFO")
    print(f"A: {args.pth_a}")
    print(f"B: {args.pth_b}")
    print(f"A annotations: {len(anns_a)}")
    print(f"B annotations: {len(anns_b)}")

    obj_rows, obj_count_a, obj_count_b, obj_matched = compare_obj_features(anns_a, anns_b, field=args.obj_field)
    print("\n# OBJ FEATURE COMPARE")
    print(f"A annotations with {args.obj_field}: {obj_count_a}")
    print(f"B annotations with {args.obj_field}: {obj_count_b}")
    print(f"Matched annotations: {obj_matched}")
    summarize_rows(obj_rows, f"OBJ FIELD = {args.obj_field}", show_topk=args.show_topk, is_obj=True)

    part_rows, bank_a, bank_b, only_a, only_b = compare_part_bank(anns_a, anns_b, part_field=args.part_field)
    print("\n# PART ALL-PARTS BANK COMPARE")
    print(f"A bank object classes: {len(sorted({k[0] for k in bank_a.keys()}))}")
    print(f"B bank object classes: {len(sorted({k[0] for k in bank_b.keys()}))}")
    print(f"A bank unique parts   : {len(bank_a)}")
    print(f"B bank unique parts   : {len(bank_b)}")
    print(f"Common bank entries   : {len(part_rows)}")
    print(f"Only in A             : {len(only_a)}")
    print(f"Only in B             : {len(only_b)}")
    if len(only_a) > 0:
        print("Only in A examples:", only_a[: min(10, len(only_a))])
    if len(only_b) > 0:
        print("Only in B examples:", only_b[: min(10, len(only_b))])

    summarize_rows(part_rows, f"PART BANK FIELD = {args.part_field}", show_topk=args.show_topk, is_obj=False)

    obj_valid = [r for r in obj_rows if not r.get("shape_mismatch", False)]
    part_valid = [r for r in part_rows if not r.get("shape_mismatch", False)]
    obj_identical = all(r["identical"] for r in obj_valid) if obj_valid else False
    part_identical = all(r["identical"] for r in part_valid) if part_valid else False

    print("\n# QUICK JUDGEMENT")
    print(f"OBJ identical?  {obj_identical}")
    print(f"PART identical? {part_identical}")
    if (not obj_identical) and part_identical:
        print("PASS: obj differs, part bank is identical.")
    elif obj_identical and part_identical:
        print("OBJ and PART are both identical.")
    elif (not obj_identical) and (not part_identical):
        print("OBJ and PART both differ.")
    else:
        print("OBJ identical but PART differs.")


if __name__ == "__main__":
    main()
