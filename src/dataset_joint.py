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


class DinoClipJointDataset(Dataset):
    """
    Additive dataset for joint object-level + part-level training.

    One sample returns:
      - obj_feat: pooled visual feature for the object-level branch
      - patch_tokens: patch tokens for the part-level branch
      - obj_text_feat: object text feature (ann_feats)
      - part_text_feat: part text features (part_ann_feats), variable K x C
      - obj_mask_patch: patch-level GT object mask
      - part_category_id: global part ids aligned with part_text_feat
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
        self.grid_size = crop_dim // patch_size

        if is_wds:
            self._load_wds_dataset(features_file)
        else:
            self._load_pth_dataset(features_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        obj_mask_patch = sample["obj_mask_patch"]

        return {
            "obj_feat": sample["obj_feat"],
            "patch_tokens": sample["patch_tokens"],
            "obj_text_feat": sample["obj_text_feat"],
            "part_text_feat": sample["part_text_feat"],
            "obj_mask_patch": obj_mask_patch,
            "category_id": torch.tensor(sample["category_id"], dtype=torch.long),
            "part_category_id": sample["part_category_id"],
            "metadata": {
                "annotation_id": sample["annotation_id"],
                "image_id": sample["image_id"],
                "class_name": sample["class_name"],
                "part_class_name": sample["part_class_name"],
                "seg_path": sample["seg_path"],
                "cropaug_box_xyxy": sample.get("cropaug_box_xyxy", None),
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

    def _build_obj_mask_patch(self, seg_path: str, category_id: int) -> torch.Tensor:
        mask = self._read_mask(seg_path)
        mask_value = category_id + 1 if self.with_background else category_id
        binary = (mask == mask_value).astype(np.uint8) * 255

        pil_mask = Image.fromarray(binary)

        # match patch_tokens extraction geometry: direct Resize((resize_dim, resize_dim))
        pil_mask = TF.resize(
            pil_mask,
            [self.resize_dim, self.resize_dim],
            interpolation=InterpolationMode.NEAREST,
        )

        # optional: if crop_dim != resize_dim, keep this crop; otherwise it is a no-op
        if self.crop_dim != self.resize_dim:
            pil_mask = TF.center_crop(pil_mask, [self.crop_dim, self.crop_dim])

        pil_mask = TF.resize(
            pil_mask,
            [self.grid_size, self.grid_size],
            interpolation=InterpolationMode.NEAREST,
        )

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
        # cropaug geometry: object square crop -> Resize((resize_dim, resize_dim))
        pil_mask = TF.resize(
            pil_mask,
            [self.grid_size, self.grid_size],
            interpolation=InterpolationMode.NEAREST,
        )

        obj_mask_patch = torch.from_numpy((np.array(pil_mask) > 0).astype(np.bool_)).view(-1)
        return obj_mask_patch

    def _normalize_empty_part_tensor(self, part_text_feat: torch.Tensor, obj_text_feat: torch.Tensor) -> torch.Tensor:
        if part_text_feat.ndim == 1 and part_text_feat.numel() == 0:
            return obj_text_feat.new_zeros((0, obj_text_feat.shape[-1]))
        if part_text_feat.ndim == 0:
            return obj_text_feat.new_zeros((0, obj_text_feat.shape[-1]))
        return part_text_feat

    def _load_pth_dataset(self, features_file: str) -> None:
        print("Loading joint dataset...")
        data = torch.load(features_file, map_location="cpu")
        print("Joint dataset loaded!")

        images = {imm["id"]: imm for imm in data["images"]}
        self.data = {}

        skipped = 0
        for ann in data["annotations"]:
            imm_id = ann["image_id"]
            imm = images[imm_id]

            if self.obj_feature_name not in imm:
                skipped += 1
                continue
            if self.obj_text_name not in ann:
                skipped += 1
                continue

            # minimal additive change:
            # part feature can come from annotations (cropaug) or images (original patch_tokens)
            if self.part_feature_name in ann:
                patch_tokens = ann[self.part_feature_name]
            elif self.part_feature_name in imm:
                patch_tokens = imm[self.part_feature_name]
            else:
                skipped += 1
                continue

            obj_text_feat = ann[self.obj_text_name]
            part_text_feat = ann.get(self.part_text_name, None)
            if part_text_feat is None:
                part_text_feat = obj_text_feat.new_zeros((0, obj_text_feat.shape[-1]))
            part_text_feat = self._normalize_empty_part_tensor(part_text_feat, obj_text_feat)

            part_category_id = ann.get("part_category_id", [])
            part_category_id = torch.tensor(part_category_id, dtype=torch.long)

            cropaug_box_xyxy = ann.get("cropaug_box_xyxy", None)
            if self.part_feature_name == "cropaug_patch_tokens" and cropaug_box_xyxy is not None:
                obj_mask_patch = self._build_cropaug_obj_mask_patch(
                    seg_path=imm["seg_file_name"],
                    category_id=ann["category_id"],
                    crop_box_xyxy=cropaug_box_xyxy,
                )
            else:
                obj_mask_patch = self._build_obj_mask_patch(
                    seg_path=imm["seg_file_name"],
                    category_id=ann["category_id"],
                )

            self.data[len(self.data)] = {
                "annotation_id": ann["id"],
                "image_id": imm_id,
                "class_name": ann.get("class_name", ""),
                "part_class_name": ann.get("part_class_name", []),
                "category_id": ann["category_id"],
                "seg_path": imm["seg_file_name"],
                "obj_feat": imm[self.obj_feature_name],
                "patch_tokens": patch_tokens,
                "obj_text_feat": obj_text_feat,
                "part_text_feat": part_text_feat,
                "part_category_id": part_category_id,
                "obj_mask_patch": obj_mask_patch,
                "cropaug_box_xyxy": cropaug_box_xyxy,
            }

        print(f"Joint samples: {len(self.data)}")
        print(f"Skipped annotations due to missing fields: {skipped}")

    def _load_wds_dataset(self, features_file: str) -> None:
        print("Loading joint WebDataset...")

        def my_decoder(key, value):
            if not key.endswith(".pth"):
                return None
            return torch.load(BytesIO(value))

        dataset = wds.WebDataset(features_file).decode(my_decoder)
        self.data = {}
        skipped = 0

        for obj in dataset:
            pth = obj["pth"]
            if self.obj_feature_name not in pth:
                skipped += 1
                continue
            if self.obj_text_name not in pth:
                skipped += 1
                continue
            if self.part_feature_name not in pth:
                skipped += 1
                continue

            obj_text_feat = pth[self.obj_text_name]
            part_text_feat = pth.get(self.part_text_name, None)
            if part_text_feat is None:
                part_text_feat = obj_text_feat.new_zeros((0, obj_text_feat.shape[-1]))
            part_text_feat = self._normalize_empty_part_tensor(part_text_feat, obj_text_feat)

            part_category_id = torch.tensor(pth.get("part_category_id", []), dtype=torch.long)

            cropaug_box_xyxy = pth.get("cropaug_box_xyxy", None)
            if self.part_feature_name == "cropaug_patch_tokens" and cropaug_box_xyxy is not None:
                obj_mask_patch = self._build_cropaug_obj_mask_patch(
                    seg_path=pth["seg_file_name"],
                    category_id=pth["category_id"],
                    crop_box_xyxy=cropaug_box_xyxy,
                )
            else:
                obj_mask_patch = self._build_obj_mask_patch(
                    seg_path=pth["seg_file_name"],
                    category_id=pth["category_id"],
                )

            self.data[len(self.data)] = {
                "annotation_id": pth["id"],
                "image_id": pth["image_id"],
                "class_name": pth.get("class_name", ""),
                "part_class_name": pth.get("part_class_name", []),
                "category_id": pth["category_id"],
                "seg_path": pth["seg_file_name"],
                "obj_feat": pth[self.obj_feature_name],
                "patch_tokens": pth[self.part_feature_name],
                "obj_text_feat": obj_text_feat,
                "part_text_feat": part_text_feat,
                "part_category_id": part_category_id,
                "obj_mask_patch": obj_mask_patch,
                "cropaug_box_xyxy": cropaug_box_xyxy,
            }

        print(f"Joint WebDataset samples: {len(self.data)}")
        print(f"Skipped annotations due to missing fields: {skipped}")


def joint_collate_fn(batch: List[Dict]) -> Dict:
    batch_size = len(batch)

    obj_feat = torch.stack([b["obj_feat"].float() for b in batch], dim=0)
    patch_tokens = torch.stack([b["patch_tokens"].float() for b in batch], dim=0)
    obj_text_feat = torch.stack([b["obj_text_feat"].float() for b in batch], dim=0)
    obj_mask_patch = torch.stack([b["obj_mask_patch"].bool() for b in batch], dim=0)
    category_id = torch.stack([b["category_id"].long() for b in batch], dim=0)

    feat_dim = obj_text_feat.shape[-1]
    max_k = max((b["part_text_feat"].shape[0] for b in batch), default=0)

    part_text_feat = obj_text_feat.new_zeros((batch_size, max_k, feat_dim))
    part_valid_mask = torch.zeros((batch_size, max_k), dtype=torch.bool)
    part_category_id = torch.full((batch_size, max_k), -1, dtype=torch.long)

    metadata = {
        "annotation_id": [],
        "image_id": [],
        "class_name": [],
        "part_class_name": [],
        "seg_path": [],
        "cropaug_box_xyxy": [],
    }

    for i, sample in enumerate(batch):
        k = sample["part_text_feat"].shape[0]
        if k > 0:
            part_text_feat[i, :k] = sample["part_text_feat"].float()
            part_valid_mask[i, :k] = True
            if sample["part_category_id"].numel() > 0:
                part_category_id[i, :k] = sample["part_category_id"].long()

        metadata["annotation_id"].append(sample["metadata"]["annotation_id"])
        metadata["image_id"].append(sample["metadata"]["image_id"])
        metadata["class_name"].append(sample["metadata"]["class_name"])
        metadata["part_class_name"].append(sample["metadata"]["part_class_name"])
        metadata["seg_path"].append(sample["metadata"]["seg_path"])
        metadata["cropaug_box_xyxy"].append(sample["metadata"]["cropaug_box_xyxy"])

    return {
        "obj_feat": obj_feat,
        "patch_tokens": patch_tokens,
        "obj_text_feat": obj_text_feat,
        "part_text_feat": part_text_feat,
        "obj_mask_patch": obj_mask_patch,
        "category_id": category_id,
        "part_category_id": part_category_id,
        "part_valid_mask": part_valid_mask,
        "metadata": metadata,
    }
