import os
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode



try:
    from src.voc116_part_coarse import COARSE_ID_TO_FINE_IDS
except Exception:
    COARSE_ID_TO_FINE_IDS = {}

class DinoClipJointDataset(Dataset):
    """
    Joint object-level + part-level dataset with GT object mask and GT part masks on patch grid.

    Returned keys:
      - obj_feat: [Dv]
      - patch_tokens: [N, Dv]
      - obj_text_feat: [Dt]
      - part_text_feat: [K, Dt]
      - obj_mask_patch: [N] bool
      - part_gt_mask_patch: [K, N] bool
      - category_id: scalar
      - part_category_id: [K]
      - part_valid_mask is added by collate_fn
      - metadata
    """

    def __init__(
        self,
        features_file: str,
        obj_feature_name: str = "avg_self_attn_out",
        part_feature_name: str = "patch_tokens",
        obj_text_name: str = "ann_feats",
        part_text_name: str = "part_ann_feats",
        resize_dim: int = 448,
        crop_dim: int = 448,
        patch_size: int = 14,
        with_background: bool = True,
        is_wds: bool = False,
        path_prefix: Optional[str] = None,
        min_obj_area_ratio: float = 0.0,
    ):
        self.obj_feature_name = obj_feature_name
        self.part_feature_name = part_feature_name
        self.obj_text_name = obj_text_name
        self.part_text_name = part_text_name
        self.resize_dim = resize_dim
        self.crop_dim = crop_dim
        self.patch_size = patch_size
        self.with_background = with_background
        self.path_prefix = path_prefix
        self.min_obj_area_ratio = float(min_obj_area_ratio)
        self.grid_size = crop_dim // patch_size
        self.class_part_bank = {}

        self.part_taxonomy = "fine"

        if is_wds:
            self._load_wds_dataset(features_file)
        else:
            self._load_pth_dataset(features_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        return {
            "obj_feat": sample["obj_feat"],
            "patch_tokens": sample["patch_tokens"],
            "obj_text_feat": sample["obj_text_feat"],
            "part_text_feat": sample["part_text_feat"],
            "obj_mask_patch": sample["obj_mask_patch"],
            "part_gt_mask_patch": sample["part_gt_mask_patch"],
            "category_id": torch.tensor(sample["category_id"], dtype=torch.long),
            "part_category_id": sample["part_category_id"],
            "metadata": {
                "annotation_id": sample["annotation_id"],
                "image_id": sample["image_id"],
                "class_name": sample["class_name"],
                "part_class_name": sample["part_class_name"],
                "seg_path": sample["seg_path"],
                "cropaug_box_xyxy": sample.get("cropaug_box_xyxy", None),
                "obj_area_ratio": sample.get("obj_area_ratio", None),
            },
        }

    def _resolve_path(self, path_str: str) -> str:
        if os.path.exists(path_str):
            return path_str
        if self.path_prefix is not None:
            candidate = os.path.join(self.path_prefix, path_str)
            if os.path.exists(candidate):
                return candidate
        return path_str

    def _read_mask(self, seg_path: str) -> np.ndarray:
        seg_path = self._resolve_path(seg_path)
        mask = np.array(Image.open(seg_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask

    def _obj_to_part_seg_path(self, obj_seg_path: str) -> str:
        return obj_seg_path.replace("annotations_detectron2_obj", "annotations_detectron2_part")


    def _part_binary_mask(self, mask: np.ndarray, part_id: int) -> np.ndarray:
        """Return uint8 binary mask for either fine or coarse part id."""
        if getattr(self, "part_taxonomy", "fine") == "coarse_pascalpart116_v1":
            fine_ids = COARSE_ID_TO_FINE_IDS.get(int(part_id), [int(part_id)])
            binary = np.isin(mask, fine_ids).astype(np.uint8) * 255
        else:
            binary = (mask == int(part_id)).astype(np.uint8) * 255
        return binary

    def _compute_obj_area_ratio(self, seg_path: str, category_id: int) -> float:
        mask = self._read_mask(seg_path)
        h, w = mask.shape[:2]
        total_pixels = max(int(h * w), 1)
        mask_value = category_id + 1 if self.with_background else category_id
        obj_pixels = int((mask == mask_value).sum())
        return float(obj_pixels / total_pixels)

    def _build_obj_mask_patch(self, seg_path: str, category_id: int) -> torch.Tensor:
        mask = self._read_mask(seg_path)
        mask_value = category_id + 1 if self.with_background else category_id
        binary = (mask == mask_value).astype(np.uint8) * 255

        pil_mask = Image.fromarray(binary)
        pil_mask = TF.resize(pil_mask, self.resize_dim, interpolation=InterpolationMode.NEAREST)
        pil_mask = TF.center_crop(pil_mask, [self.crop_dim, self.crop_dim])
        pil_mask = TF.resize(pil_mask, [self.grid_size, self.grid_size], interpolation=InterpolationMode.NEAREST)

        obj_mask_patch = torch.from_numpy((np.array(pil_mask) > 0).astype(np.bool_)).view(-1)
        return obj_mask_patch

    def _build_cropaug_obj_mask_patch(self, seg_path: str, category_id: int, crop_box_xyxy) -> torch.Tensor:
        mask = self._read_mask(seg_path)
        mask_value = category_id + 1 if self.with_background else category_id
        binary = (mask == mask_value).astype(np.uint8) * 255

        if torch.is_tensor(crop_box_xyxy):
            crop_box_xyxy = crop_box_xyxy.tolist()
        x1, y1, x2, y2 = [int(v) for v in crop_box_xyxy]
        binary = binary[y1:y2, x1:x2]

        pil_mask = Image.fromarray(binary)
        pil_mask = TF.resize(
            pil_mask,
            [self.grid_size, self.grid_size],
            interpolation=InterpolationMode.NEAREST,
        )
        obj_mask_patch = torch.from_numpy((np.array(pil_mask) > 0).astype(np.bool_)).view(-1)
        return obj_mask_patch

    def _build_cropaug_part_mask_patch(self, obj_seg_path: str, part_id: int, crop_box_xyxy) -> torch.Tensor:
        part_seg_path = self._obj_to_part_seg_path(obj_seg_path)
        mask = self._read_mask(part_seg_path)        
        binary = self._part_binary_mask(mask, int(part_id))

        if torch.is_tensor(crop_box_xyxy):
            crop_box_xyxy = crop_box_xyxy.tolist()
        x1, y1, x2, y2 = [int(v) for v in crop_box_xyxy]
        binary = binary[y1:y2, x1:x2]

        pil_mask = Image.fromarray(binary)
        pil_mask = TF.resize(
            pil_mask,
            [self.grid_size, self.grid_size],
            interpolation=InterpolationMode.NEAREST,
        )
        return torch.from_numpy((np.array(pil_mask) > 0).astype(np.bool_)).view(-1)

    def _build_part_mask_stack(self, obj_seg_path: str, part_category_id: torch.Tensor, crop_box_xyxy=None) -> torch.Tensor:
        if part_category_id.numel() == 0:
            return torch.zeros((0, self.grid_size * self.grid_size), dtype=torch.bool)

        masks = []
        use_cropaug = (self.part_feature_name == "cropaug_patch_tokens" and crop_box_xyxy is not None)
        if use_cropaug:
            for pid in part_category_id.tolist():
                masks.append(self._build_cropaug_part_mask_patch(obj_seg_path, int(pid), crop_box_xyxy))
        else:
            part_seg_path = self._obj_to_part_seg_path(obj_seg_path)
            full_mask = self._read_mask(part_seg_path)
            for pid in part_category_id.tolist():
                binary = self._part_binary_mask(full_mask, int(pid))
                pil_mask = Image.fromarray(binary)
                pil_mask = TF.resize(pil_mask, self.resize_dim, interpolation=InterpolationMode.NEAREST)
                pil_mask = TF.center_crop(pil_mask, [self.crop_dim, self.crop_dim])
                pil_mask = TF.resize(pil_mask, [self.grid_size, self.grid_size], interpolation=InterpolationMode.NEAREST)
                masks.append(torch.from_numpy((np.array(pil_mask) > 0).astype(np.bool_)).view(-1))
        return torch.stack(masks, dim=0).bool()

    def _normalize_empty_part_tensor(self, part_text_feat: torch.Tensor, obj_text_feat: torch.Tensor) -> torch.Tensor:
        if part_text_feat.ndim == 1 and part_text_feat.numel() == 0:
            return obj_text_feat.new_zeros((0, obj_text_feat.shape[-1]))
        if part_text_feat.ndim == 0:
            return obj_text_feat.new_zeros((0, obj_text_feat.shape[-1]))
        return part_text_feat

    def _build_common_sample(
        self,
        ann_or_pth: Dict,
        img_or_pth: Dict,
        obj_feat_tensor: torch.Tensor,
        part_feat_tensor: torch.Tensor,
        obj_text_feat: torch.Tensor,
        part_text_feat: torch.Tensor,
        part_category_id: torch.Tensor,
        obj_area_ratio: Optional[float] = None,
    ):
        cropaug_box_xyxy = ann_or_pth.get("cropaug_box_xyxy", img_or_pth.get("cropaug_box_xyxy", None))
        if self.part_feature_name == "cropaug_patch_tokens" and cropaug_box_xyxy is not None:
            obj_mask_patch = self._build_cropaug_obj_mask_patch(
                seg_path=img_or_pth["seg_file_name"],
                category_id=ann_or_pth["category_id"],
                crop_box_xyxy=cropaug_box_xyxy,
            )
        else:
            obj_mask_patch = self._build_obj_mask_patch(
                seg_path=img_or_pth["seg_file_name"],
                category_id=ann_or_pth["category_id"],
            )

        part_gt_mask_patch = self._build_part_mask_stack(
            obj_seg_path=img_or_pth["seg_file_name"],
            part_category_id=part_category_id,
            crop_box_xyxy=cropaug_box_xyxy,
        )

        return {
            "annotation_id": ann_or_pth["id"],
            "image_id": ann_or_pth["image_id"],
            "class_name": ann_or_pth.get("class_name", ""),
            "part_class_name": ann_or_pth.get("part_class_name", []),
            "category_id": ann_or_pth["category_id"],
            "seg_path": img_or_pth["seg_file_name"],
            "obj_feat": obj_feat_tensor,
            "patch_tokens": part_feat_tensor,
            "obj_text_feat": obj_text_feat,
            "part_text_feat": part_text_feat,
            "part_category_id": part_category_id,
            "obj_mask_patch": obj_mask_patch,
            "part_gt_mask_patch": part_gt_mask_patch,
            "cropaug_box_xyxy": cropaug_box_xyxy,
            "obj_area_ratio": obj_area_ratio,
        }


    def _build_class_part_bank_from_annotations(self, annotations: List[Dict], obj_text_dim: int) -> None:
        """
        Build category -> all known parts bank.

        We do NOT use 'which parts appear in this image' as training input.
        We only use the object category to retrieve the full part list.
        """
        tmp_bank: Dict[int, Dict[int, List[torch.Tensor]]] = {}
        tmp_name_bank: Dict[int, Dict[int, str]] = {}

        for ann in annotations:
            cat = int(ann["category_id"])
            part_ids = ann.get("part_category_id", [])
            part_feats = ann.get(self.part_text_name, None)
            part_names = ann.get("part_class_name", [])

            if part_feats is None or len(part_ids) == 0:
                continue
            if not torch.is_tensor(part_feats) or part_feats.ndim < 2:
                continue

            if cat not in tmp_bank:
                tmp_bank[cat] = {}
                tmp_name_bank[cat] = {}

            for j, pid in enumerate(part_ids):
                pid = int(pid)
                feat_j = part_feats[j].float().clone()
                tmp_bank[cat].setdefault(pid, []).append(feat_j)
                if j < len(part_names):
                    tmp_name_bank[cat][pid] = part_names[j]

        self.class_part_bank = {}
        for cat, pid2feat_list in tmp_bank.items():
            sorted_pids = sorted(pid2feat_list.keys())
            feat_rows = []
            part_names = []
            for pid in sorted_pids:
                feat = torch.stack(pid2feat_list[pid], dim=0).mean(dim=0)
                feat_rows.append(feat)
                part_names.append(tmp_name_bank.get(cat, {}).get(pid, str(pid)))
            self.class_part_bank[cat] = {
                "part_ids": torch.tensor(sorted_pids, dtype=torch.long),
                "part_feats": torch.stack(feat_rows, dim=0) if len(feat_rows) > 0 else torch.zeros((0, obj_text_dim)),
                "part_names": part_names,
            }

    def _get_all_parts_for_category(self, category_id: int, obj_text_feat: torch.Tensor):
        bank = self.class_part_bank.get(int(category_id), None)
        if bank is None:
            return (
                obj_text_feat.new_zeros((0, obj_text_feat.shape[-1])),
                torch.zeros((0,), dtype=torch.long),
                [],
            )
        return (
            bank["part_feats"].to(dtype=obj_text_feat.dtype),
            bank["part_ids"].clone(),
            list(bank.get("part_names", [])),
        )

    def _load_pth_dataset(self, features_file: str) -> None:
        print("Loading joint dataset...")
        data = torch.load(features_file, map_location="cpu")

        self.part_taxonomy = str(data.get("part_taxonomy", "fine"))
        print(f"Part taxonomy: {self.part_taxonomy}")
        print("Joint dataset loaded!")

        images = {imm["id"]: imm for imm in data["images"]}
        annotations = data["annotations"]

        example_obj_text = None
        for ann in annotations:
            if self.obj_text_name in ann and torch.is_tensor(ann[self.obj_text_name]):
                example_obj_text = ann[self.obj_text_name]
                break
        obj_text_dim = int(example_obj_text.shape[-1]) if example_obj_text is not None else 0

        # Pass 1: build category -> all-parts bank
        self._build_class_part_bank_from_annotations(annotations, obj_text_dim=obj_text_dim)
        print(f"Built class_part_bank for {len(self.class_part_bank)} object classes")

        # Pass 2: build samples; each sample uses ALL parts of its object category
        self.data = {}
        skipped = 0
        skipped_small_obj = 0

        for ann in annotations:
            imm_id = ann["image_id"]
            imm = images[imm_id]

            obj_feat_src = ann if self.obj_feature_name in ann else imm if self.obj_feature_name in imm else None
            part_feat_src = ann if self.part_feature_name in ann else imm if self.part_feature_name in imm else None

            if obj_feat_src is None:
                skipped += 1
                continue
            if part_feat_src is None:
                skipped += 1
                continue
            if self.obj_text_name not in ann:
                skipped += 1
                continue

            obj_area_ratio = self._compute_obj_area_ratio(imm["seg_file_name"], int(ann["category_id"]))
            if obj_area_ratio < self.min_obj_area_ratio:
                skipped_small_obj += 1
                continue

            obj_text_feat = ann[self.obj_text_name]
            part_text_feat, part_category_id, part_class_name = self._get_all_parts_for_category(
                int(ann["category_id"]), obj_text_feat
            )

            sample = self._build_common_sample(
                ann_or_pth=ann,
                img_or_pth=imm,
                obj_feat_tensor=obj_feat_src[self.obj_feature_name],
                part_feat_tensor=part_feat_src[self.part_feature_name],
                obj_text_feat=obj_text_feat,
                part_text_feat=part_text_feat,
                part_category_id=part_category_id,
                obj_area_ratio=obj_area_ratio,
            )
            sample["part_class_name"] = part_class_name
            self.data[len(self.data)] = sample

        print(f"Joint samples: {len(self.data)}")
        print(f"Skipped annotations due to missing fields: {skipped}")
        print(f"Skipped annotations due to small object area ratio (< {self.min_obj_area_ratio}): {skipped_small_obj}")

    def _load_wds_dataset(self, features_file: str) -> None:
        print("Loading joint WebDataset...")

        def my_decoder(key, value):
            if not key.endswith(".pth"):
                return None
            return torch.load(BytesIO(value))

        dataset = wds.WebDataset(features_file).decode(my_decoder)
        records = []
        for obj in dataset:
            if "pth" in obj:
                records.append(obj["pth"])

        example_obj_text = None
        for pth in records:
            if self.obj_text_name in pth and torch.is_tensor(pth[self.obj_text_name]):
                example_obj_text = pth[self.obj_text_name]
                break
        obj_text_dim = int(example_obj_text.shape[-1]) if example_obj_text is not None else 0

        # Pass 1: build category -> all-parts bank
        self._build_class_part_bank_from_annotations(records, obj_text_dim=obj_text_dim)
        print(f"Built class_part_bank for {len(self.class_part_bank)} object classes")

        # Pass 2: build samples; each sample uses ALL parts of its object category
        self.data = {}
        skipped = 0
        skipped_small_obj = 0

        for pth in records:
            obj_feat_src = pth if self.obj_feature_name in pth else None
            part_feat_src = pth if self.part_feature_name in pth else None

            if obj_feat_src is None:
                skipped += 1
                continue
            if part_feat_src is None:
                skipped += 1
                continue
            if self.obj_text_name not in pth:
                skipped += 1
                continue

            if "seg_file_name" not in pth:
                skipped += 1
                continue
            if "image_id" not in pth:
                pth["image_id"] = pth.get("id", len(self.data))

            obj_area_ratio = self._compute_obj_area_ratio(pth["seg_file_name"], int(pth["category_id"]))
            if obj_area_ratio < self.min_obj_area_ratio:
                skipped_small_obj += 1
                continue

            obj_text_feat = pth[self.obj_text_name]
            part_text_feat, part_category_id, part_class_name = self._get_all_parts_for_category(
                int(pth["category_id"]), obj_text_feat
            )

            sample = self._build_common_sample(
                ann_or_pth=pth,
                img_or_pth=pth,
                obj_feat_tensor=obj_feat_src[self.obj_feature_name],
                part_feat_tensor=part_feat_src[self.part_feature_name],
                obj_text_feat=obj_text_feat,
                part_text_feat=part_text_feat,
                part_category_id=part_category_id,
                obj_area_ratio=obj_area_ratio,
            )
            sample["part_class_name"] = part_class_name
            self.data[len(self.data)] = sample

        print(f"Joint WebDataset samples: {len(self.data)}")
        print(f"Skipped annotations due to missing fields: {skipped}")
        print(f"Skipped annotations due to small object area ratio (< {self.min_obj_area_ratio}): {skipped_small_obj}")


def joint_collate_fn(batch: List[Dict]) -> Dict:
    batch_size = len(batch)

    obj_feat = torch.stack([b["obj_feat"].float() for b in batch], dim=0)
    patch_tokens = torch.stack([b["patch_tokens"].float() for b in batch], dim=0)
    obj_text_feat = torch.stack([b["obj_text_feat"].float() for b in batch], dim=0)
    obj_mask_patch = torch.stack([b["obj_mask_patch"].bool() for b in batch], dim=0)
    category_id = torch.stack([b["category_id"].long() for b in batch], dim=0)

    feat_dim = obj_text_feat.shape[-1]
    num_patches = patch_tokens.shape[1]
    max_k = max((b["part_text_feat"].shape[0] for b in batch), default=0)

    part_text_feat = obj_text_feat.new_zeros((batch_size, max_k, feat_dim))
    part_category_id = torch.full((batch_size, max_k), -1, dtype=torch.long)
    part_valid_mask = torch.zeros((batch_size, max_k), dtype=torch.bool)
    part_gt_mask_patch = torch.zeros((batch_size, max_k, num_patches), dtype=torch.bool)

    metadata = []
    for i, b in enumerate(batch):
        k = b["part_text_feat"].shape[0]
        if k > 0:
            part_text_feat[i, :k] = b["part_text_feat"].float()
            part_category_id[i, :k] = b["part_category_id"].long()
            part_valid_mask[i, :k] = True
            part_gt_mask_patch[i, :k] = b["part_gt_mask_patch"].bool()
        metadata.append(b["metadata"])

    return {
        "obj_feat": obj_feat,
        "patch_tokens": patch_tokens,
        "obj_text_feat": obj_text_feat,
        "part_text_feat": part_text_feat,
        "obj_mask_patch": obj_mask_patch,
        "part_gt_mask_patch": part_gt_mask_patch,
        "category_id": category_id,
        "part_category_id": part_category_id,
        "part_valid_mask": part_valid_mask,
        "metadata": metadata,
    }
