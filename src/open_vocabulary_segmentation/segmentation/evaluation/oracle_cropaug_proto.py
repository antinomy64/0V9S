from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IGNORE_LABEL = -1
NUM_PARTS = 116


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def load_pth(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


def load_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def resolve_part_mask_path(img_info: Dict, part_mask_dir: str) -> str:
    split = img_info.get("split", "train")
    stem = Path(img_info["file_name"]).stem
    path = Path(part_mask_dir) / split / f"{stem}.png"
    if not path.exists():
        raise FileNotFoundError(f"Part GT mask not found: {path}")
    return str(path)


def crop_by_box(arr: np.ndarray, box_xyxy) -> np.ndarray:
    if torch.is_tensor(box_xyxy):
        box_xyxy = box_xyxy.tolist()
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    return arr[y1:y2, x1:x2]


def resize_mask_nearest(mask: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.int32).astype(np.uint8))
    pil = pil.resize((out_hw[1], out_hw[0]), resample=Image.NEAREST)
    return np.array(pil)


def maybe_shift_part_ids(mask: np.ndarray, one_based_part_ids: bool) -> np.ndarray:
    if not one_based_part_ids:
        return mask.astype(np.int64)

    out = np.full(mask.shape, fill_value=IGNORE_LABEL, dtype=np.int64)
    fg = mask > 0
    out[fg] = mask[fg].astype(np.int64) - 1
    return out


def build_patch_part_gt(
    part_mask: np.ndarray,
    cropaug_box_xyxy,
    grid_size: int,
    legal_part_ids: List[int],
    one_based_part_ids: bool = False,
) -> np.ndarray:
    part_mask = maybe_shift_part_ids(part_mask, one_based_part_ids)
    crop = crop_by_box(part_mask, cropaug_box_xyxy)

    crop_safe = crop.copy()
    crop_safe[crop_safe == IGNORE_LABEL] = 255
    patch_gt = resize_mask_nearest(crop_safe, (grid_size, grid_size)).reshape(-1).astype(np.int64)
    patch_gt[patch_gt == 255] = IGNORE_LABEL

    legal_set = set(int(x) for x in legal_part_ids)
    valid = np.isin(patch_gt, list(legal_set))
    out = np.full_like(patch_gt, fill_value=IGNORE_LABEL, dtype=np.int64)
    out[valid] = patch_gt[valid]
    return out


def build_cropaug_visual_prototypes(
    train_pth: str,
    part_mask_dir: str,
    grid_size: int = 32,
    one_based_part_ids: bool = False,
):
    train_data = load_pth(train_pth)
    images = {img["id"]: img for img in train_data["images"]}

    feat_bank = defaultdict(list)
    feat_dim = None

    for ann in tqdm(train_data["annotations"], desc="Building cropaug oracle prototypes"):
        if "cropaug_patch_tokens" not in ann:
            continue
        if "cropaug_box_xyxy" not in ann:
            continue

        img_info = images[ann["image_id"]]
        part_mask_path = resolve_part_mask_path(img_info, part_mask_dir)
        part_mask = load_mask(part_mask_path)

        patch_tokens = ann["cropaug_patch_tokens"].float()
        patch_tokens = l2norm(patch_tokens, dim=-1)
        feat_dim = patch_tokens.shape[-1]

        legal_part_ids = ann.get("part_category_id", [])
        if len(legal_part_ids) == 0:
            continue

        patch_gt = build_patch_part_gt(
            part_mask=part_mask,
            cropaug_box_xyxy=ann["cropaug_box_xyxy"],
            grid_size=grid_size,
            legal_part_ids=legal_part_ids,
            one_based_part_ids=one_based_part_ids,
        )
        patch_gt_t = torch.from_numpy(patch_gt)

        for pid in legal_part_ids:
            pid = int(pid)
            mask = patch_gt_t == pid
            if mask.sum().item() == 0:
                continue
            proto = patch_tokens[mask].mean(dim=0)
            proto = l2norm(proto, dim=0)
            feat_bank[pid].append(proto)

    if feat_dim is None:
        raise RuntimeError("No cropaug_patch_tokens found in train_pth.")

    prototypes = torch.zeros(NUM_PARTS, feat_dim, dtype=torch.float32)
    valid_proto = torch.zeros(NUM_PARTS, dtype=torch.bool)

    for pid, feats in feat_bank.items():
        feats = torch.stack(feats, dim=0)
        proto = feats.mean(dim=0)
        proto = l2norm(proto, dim=0)
        prototypes[pid] = proto
        valid_proto[pid] = True

    return prototypes, valid_proto
