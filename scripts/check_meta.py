import torch
from pprint import pprint

path = "feature/voc116_obj_part_test11/train_voc116_obj_with_text.pth"
data = torch.load(path, map_location="cpu")

img0 = data["images"][0]
ann0 = data["annotations"][0]

print("=" * 120)
print("[PTH]", path)
print("=" * 120)

print("\n[top-level]")
for k, v in data.items():
    if k in ["images", "annotations"]:
        print(f"{k}: list len={len(v)}")
    elif isinstance(v, dict):
        print(f"{k}: dict len={len(v)}")
        pprint(v)
    elif isinstance(v, list):
        print(f"{k}: list len={len(v)}")
        pprint(v)
    else:
        print(f"{k}: {repr(v)}")

def print_value(k, v, indent=""):
    # Tensor: feature 太长，只打印前几个
    if torch.is_tensor(v):
        print(f"{indent}{k}: tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        flat = v.flatten()
        n = min(20, flat.numel())
        print(f"{indent}  first_{n}: {flat[:n]}")
        return

    # list 里如果全是 tensor，也只摘要；否则原样打印
    if isinstance(v, list):
        print(f"{indent}{k}: list(len={len(v)})")
        if len(v) > 0 and all(torch.is_tensor(x) for x in v):
            for i, x in enumerate(v[:5]):
                flat = x.flatten()
                n = min(10, flat.numel())
                print(f"{indent}  [{i}]: tensor(shape={tuple(x.shape)}, dtype={x.dtype}, first_{n}={flat[:n]})")
            if len(v) > 5:
                print(f"{indent}  ... ({len(v)-5} more tensor items)")
        else:
            pprint(v, width=160)
        return

    # dict 原样递归打印
    if isinstance(v, dict):
        print(f"{indent}{k}: dict")
        for kk, vv in v.items():
            print_value(kk, vv, indent + "  ")
        return

    # 其他类型原样 repr
    print(f"{indent}{k}: {repr(v)}")

print("\n" + "=" * 120)
print("[images[0]]")
print("=" * 120)
for k, v in img0.items():
    print_value(k, v)

print("\n" + "=" * 120)
print("[annotations[0]]")
print("=" * 120)
for k, v in ann0.items():
    print_value(k, v)

print("\n" + "=" * 120)
print("[sanity]")
print("=" * 120)
print("part_taxonomy:", data.get("part_taxonomy", None))
print("num_parts:", data.get("num_parts", None))
print("image id:", img0.get("id", None))
print("ann id:", ann0.get("id", None))
print("class_name:", ann0.get("class_name", None))
print("part_category_id:", ann0.get("part_category_id", None))
print("part_class_name:", ann0.get("part_class_name", None))