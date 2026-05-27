from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn
from src.loss_joint import JointObjPartLoss
from src.voc116_part_coarse import COARSE_PART_CLASSES, FINE_PART_CLASSES


def get_part_names(num_parts: int) -> List[str]:
    if num_parts == 58:
        return list(COARSE_PART_CLASSES)
    if num_parts == 116:
        return list(FINE_PART_CLASSES)
    return [f"part_{i}" for i in range(num_parts)]


def to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def cat_or_empty(chunks: List[torch.Tensor], shape, dtype=torch.float32) -> torch.Tensor:
    if len(chunks) == 0:
        return torch.empty(shape, dtype=dtype)
    return torch.cat(chunks, dim=0)


class FeatureAnalyser:
    """
    Feature/prototype analyser.

    This class loads:
      - model_config
      - projector/model
      - init_weights/checkpoint
      - JointObjPartLoss helper
      - DinoClipJointDataset

    It only returns feature tensors. It does NOT compute dataset statistics.

    Part-level feature lists are aligned with self.part_names:
        features_by_part[pid] <=> self.part_names[pid]
    """

    def __init__(
        self,
        model_config: str,
        dataset: str,
        init_weights: str,
        obj_feature_name: str = "avg_self_attn_out",
        part_feature_name: str = "cropaug_patch_tokens",
        obj_text_name: str = "ann_feats",
        part_text_name: str = "part_ann_feats",
        resize_dim: int = 448,
        crop_dim: int = 448,
        patch_size: int = 14,
        batch_size: int = 64,
        num_workers: int = 0,
        num_parts: int = 58,
        device: str = "cuda",
        show_progress: bool = True,
    ):
        self.model_config = model_config
        self.dataset_path = dataset
        self.init_weights = init_weights

        self.obj_feature_name = obj_feature_name
        self.part_feature_name = part_feature_name
        self.obj_text_name = obj_text_name
        self.part_text_name = part_text_name

        self.resize_dim = resize_dim
        self.crop_dim = crop_dim
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_parts = num_parts
        self.show_progress = show_progress

        self.device = torch.device(
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )

        with open(self.model_config, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        ModelClass = getattr(
            importlib.import_module("src.model"),
            self.cfg["model"].get("model_class", "ProjectionLayer"),
        )
        self.model = ModelClass.from_config(self.cfg["model"]).to(self.device)

        ckpt = torch.load(self.init_weights, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()

        self.dataset = DinoClipJointDataset(
            self.dataset_path,
            obj_feature_name=self.obj_feature_name,
            part_feature_name=self.part_feature_name,
            obj_text_name=self.obj_text_name,
            part_text_name=self.part_text_name,
            resize_dim=self.resize_dim,
            crop_dim=self.crop_dim,
            patch_size=self.patch_size,
            with_background=False,
            min_obj_area_ratio=float(self.cfg.get("dataset", {}).get("min_obj_area_ratio", 0.0)),
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=joint_collate_fn,
            pin_memory=False,
        )

        train_cfg = self.cfg.get("train", {})
        self.loss_helper = JointObjPartLoss(
            sim_model=self.model,
            obj_ltype=train_cfg.get("ltype", "infonce"),
            lambda_obj=0.0,
            lambda_inst=0.0,
            lambda_overlap=0.0,
            lambda_spear=0.0,
            patch_temperature=float(train_cfg.get("patch_temperature", 0.07)),
            em_iters=int(train_cfg.get("em_iters", 1)),
        ).to(self.device)
        self.loss_helper.eval()

    @property
    def part_names(self) -> List[str]:
        return get_part_names(self.num_parts)

    def _group_by_part(
        self,
        features: torch.Tensor,
        part_ids: torch.Tensor,
        dim: int,
    ) -> List[torch.Tensor]:
        grouped: List[torch.Tensor] = []
        for pid in range(self.num_parts):
            grouped.append(features[part_ids == pid].contiguous())
        return grouped

    def _group_by_category(
        self,
        features: torch.Tensor,
        category_ids: torch.Tensor,
        dim: int,
        max_obj_slots: int,
    ) -> List[torch.Tensor]:
        grouped: List[torch.Tensor] = []
        for cid in range(max_obj_slots):
            grouped.append(features[category_ids == cid].contiguous())
        return grouped

    @torch.no_grad()
    def collect_vision_feature(self):
        """
        Return only visual feature lists:

            fake_features_by_part, gt_features_by_part

        fake_features_by_part[pid]: [num_fake_instances_of_pid, dino_dim]
        gt_features_by_part[pid]:   [num_gt_instances_of_pid, dino_dim]

        No mean is computed here.
        """
        D = int(self.cfg["model"].get("dino_embed_dim", 768))
        P = self.num_parts

        fake_feat_chunks: List[torch.Tensor] = []
        fake_pid_chunks: List[torch.Tensor] = []

        gt_feat_chunks: List[torch.Tensor] = []
        gt_pid_chunks: List[torch.Tensor] = []

        for batch in tqdm(
            self.loader,
            desc="collect instance-level fake/GT features by part",
            disable=not self.show_progress,
        ):
            batch = to_device_batch(batch, self.device)

            part_text = batch["part_text_feat"].float()       # [B,K,512]
            patch_tokens = self.loss_helper._safe_normalize(
                batch["patch_tokens"].float(), dim=-1
            )                                                 # [B,N,D]
            obj_mask = batch["obj_mask_patch"].bool()         # [B,N]
            part_valid = batch["part_valid_mask"].bool()      # [B,K]
            part_gt = batch["part_gt_mask_patch"].bool()      # [B,K,N]
            part_ids = batch["part_category_id"].long()       # [B,K]

            part_proj = self.model.project_clip_txt(part_text)
            part_proj = self.loss_helper._safe_normalize(part_proj, dim=-1)

            abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens)
            abs_logits = abs_logits / float(self.loss_helper.patch_temperature)
            abs_logits = abs_logits.masked_fill(~obj_mask[:, None, :], -1e4)

            pool_ret = self.loss_helper._anchor_proto_em_pool(
                patch_tokens=patch_tokens,
                abs_logits=abs_logits,
                obj_mask_patch=obj_mask,
                part_valid_mask=part_valid,
                part_gt_mask_patch=part_gt,
                num_iters=self.loss_helper.em_iters,
                return_anchor_tokens=True,
            )

            proto_part = self.loss_helper._safe_normalize(pool_ret[1].float(), dim=-1)  # [B,K,D]
            anchor_valid = pool_ret[-1].bool()                                         # [B,K]

            gt_mask = part_gt & obj_mask[:, None, :] & part_valid[:, :, None]
            gt_pix_cnt = gt_mask.sum(dim=-1).float()                                   # [B,K]
            gt_proto = torch.einsum("bkn,bnd->bkd", gt_mask.float(), patch_tokens)
            gt_proto = gt_proto / gt_pix_cnt.clamp_min(1.0)[:, :, None]
            gt_proto = self.loss_helper._safe_normalize(gt_proto, dim=-1)

            flat_pid = part_ids.reshape(-1)                                            # [B*K]
            valid_pid = (flat_pid >= 0) & (flat_pid < P)

            fake_valid = (part_valid & anchor_valid).reshape(-1) & valid_pid
            gt_valid = (part_valid & (gt_pix_cnt > 0)).reshape(-1) & valid_pid

            flat_fake = proto_part.reshape(-1, D)
            flat_gt = gt_proto.reshape(-1, D)

            if fake_valid.any():
                fake_feat_chunks.append(flat_fake[fake_valid].detach().cpu())
                fake_pid_chunks.append(flat_pid[fake_valid].detach().cpu())

            if gt_valid.any():
                gt_feat_chunks.append(flat_gt[gt_valid].detach().cpu())
                gt_pid_chunks.append(flat_pid[gt_valid].detach().cpu())

        fake_features = cat_or_empty(fake_feat_chunks, (0, D), torch.float32)
        fake_part_ids = cat_or_empty(fake_pid_chunks, (0,), torch.long)

        gt_features = cat_or_empty(gt_feat_chunks, (0, D), torch.float32)
        gt_part_ids = cat_or_empty(gt_pid_chunks, (0,), torch.long)

        fake_features_by_part = self._group_by_part(fake_features, fake_part_ids, D)
        gt_features_by_part = self._group_by_part(gt_features, gt_part_ids, D)

        return fake_features_by_part, gt_features_by_part

    @torch.no_grad()
    def collect_text_features(self, max_obj_slots: int = 256):
        """
        Return only text feature lists:

            obj_text_raw_by_category,
            obj_text_proj_by_category,
            part_text_raw_by_part,
            part_text_proj_by_part

        Object-level:
            obj_text_raw_by_category[category_id]:  [num_obj_instances, clip_dim]
            obj_text_proj_by_category[category_id]: [num_obj_instances, dino_dim]

        Part-level:
            part_text_raw_by_part[pid]:  [num_valid_slots_of_pid, clip_dim]
            part_text_proj_by_part[pid]: [num_valid_slots_of_pid, dino_dim]

        No mean is computed here.
        """
        P = self.num_parts

        obj_raw_chunks: List[torch.Tensor] = []
        obj_proj_chunks: List[torch.Tensor] = []
        obj_cat_chunks: List[torch.Tensor] = []

        part_raw_chunks: List[torch.Tensor] = []
        part_proj_chunks: List[torch.Tensor] = []
        part_pid_chunks: List[torch.Tensor] = []

        clip_dim = None
        dino_dim = int(self.cfg["model"].get("dino_embed_dim", 768))

        for batch in tqdm(
            self.loader,
            desc="collect instance-level object/part text features",
            disable=not self.show_progress,
        ):
            batch = to_device_batch(batch, self.device)

            cat_ids = batch["category_id"].long()          # [B]
            obj_text = batch["obj_text_feat"].float()      # [B,C]
            part_text = batch["part_text_feat"].float()    # [B,K,C]
            part_ids = batch["part_category_id"].long()    # [B,K]
            part_valid = batch["part_valid_mask"].bool()   # [B,K]

            clip_dim = int(obj_text.shape[-1])

            obj_text_raw = self.loss_helper._safe_normalize(obj_text, dim=-1)
            obj_text_proj = self.model.project_clip_txt(obj_text_raw)
            obj_text_proj = self.loss_helper._safe_normalize(obj_text_proj.float(), dim=-1)

            obj_raw_chunks.append(obj_text_raw.detach().cpu())
            obj_proj_chunks.append(obj_text_proj.detach().cpu())
            obj_cat_chunks.append(cat_ids.detach().cpu())

            flat_pid = part_ids.reshape(-1)
            flat_text = part_text.reshape(-1, clip_dim)
            keep = (flat_pid >= 0) & (flat_pid < P) & part_valid.reshape(-1)

            if keep.any():
                part_text_raw = self.loss_helper._safe_normalize(flat_text[keep], dim=-1)
                part_text_proj = self.model.project_clip_txt(part_text_raw)
                part_text_proj = self.loss_helper._safe_normalize(part_text_proj.float(), dim=-1)

                part_raw_chunks.append(part_text_raw.detach().cpu())
                part_proj_chunks.append(part_text_proj.detach().cpu())
                part_pid_chunks.append(flat_pid[keep].detach().cpu())

        if clip_dim is None:
            clip_dim = 0

        obj_text_raw = cat_or_empty(obj_raw_chunks, (0, clip_dim), torch.float32)
        obj_text_proj = cat_or_empty(obj_proj_chunks, (0, dino_dim), torch.float32)
        obj_category_ids = cat_or_empty(obj_cat_chunks, (0,), torch.long)

        part_text_raw = cat_or_empty(part_raw_chunks, (0, clip_dim), torch.float32)
        part_text_proj = cat_or_empty(part_proj_chunks, (0, dino_dim), torch.float32)
        part_ids = cat_or_empty(part_pid_chunks, (0,), torch.long)

        obj_text_raw_by_category = self._group_by_category(
            obj_text_raw, obj_category_ids, clip_dim, max_obj_slots
        )
        obj_text_proj_by_category = self._group_by_category(
            obj_text_proj, obj_category_ids, dino_dim, max_obj_slots
        )

        part_text_raw_by_part = self._group_by_part(part_text_raw, part_ids, clip_dim)
        part_text_proj_by_part = self._group_by_part(part_text_proj, part_ids, dino_dim)

        return obj_text_raw_by_category, obj_text_proj_by_category, part_text_raw_by_part, part_text_proj_by_part


class DatasetAnalyser:
    """
    Lightweight dataset statistics analyser.

    This class does NOT load:
      - model_config
      - projector/model
      - init_weights/checkpoint
      - JointObjPartLoss

    It only loads DinoClipJointDataset and reads:
      - category_id
      - part_category_id
      - part_valid_mask
      - obj_mask_patch
      - part_gt_mask_patch

    Returned tensors are aligned with self.part_names:
        output[pid] <=> self.part_names[pid]
    """

    def __init__(
        self,
        dataset: str,
        obj_feature_name: str = "avg_self_attn_out",
        part_feature_name: str = "cropaug_patch_tokens",
        obj_text_name: str = "ann_feats",
        part_text_name: str = "part_ann_feats",
        resize_dim: int = 448,
        crop_dim: int = 448,
        patch_size: int = 14,
        batch_size: int = 128,
        num_workers: int = 0,
        num_parts: int = 58,
        device: str = "cuda",
        show_progress: bool = True,
        min_obj_area_ratio: float = 0.0,
    ):
        self.dataset_path = dataset

        self.obj_feature_name = obj_feature_name
        self.part_feature_name = part_feature_name
        self.obj_text_name = obj_text_name
        self.part_text_name = part_text_name

        self.resize_dim = resize_dim
        self.crop_dim = crop_dim
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_parts = num_parts
        self.show_progress = show_progress
        self.min_obj_area_ratio = min_obj_area_ratio

        self.device = torch.device(
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )

        self.dataset = DinoClipJointDataset(
            self.dataset_path,
            obj_feature_name=self.obj_feature_name,
            part_feature_name=self.part_feature_name,
            obj_text_name=self.obj_text_name,
            part_text_name=self.part_text_name,
            resize_dim=self.resize_dim,
            crop_dim=self.crop_dim,
            patch_size=self.patch_size,
            with_background=False,
            min_obj_area_ratio=self.min_obj_area_ratio,
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=joint_collate_fn,
            pin_memory=False,
        )

    @property
    def part_names(self) -> List[str]:
        return get_part_names(self.num_parts)

    @torch.no_grad()
    def compute_area_and_frequency(
        self,
        max_obj_slots: int = 256,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute area and occurrence frequency in one dataset pass.

        Returns:
            {
                "avg_pixel_count_per_obj_crop": Tensor [num_parts],
                "part_occurrence_freq_per_obj_crop": Tensor [num_parts],
            }

        Both tensors are CPU tensors and aligned with self.part_names.
        """
        P = self.num_parts

        obj_crop_cnt = torch.zeros(max_obj_slots, device=self.device, dtype=torch.float32)
        part_total_patch_cnt = torch.zeros(P, device=self.device, dtype=torch.float32)
        part_occurrence_cnt = torch.zeros(P, device=self.device, dtype=torch.float32)

        part_to_cat = torch.full((P,), -1, dtype=torch.long, device=self.device)

        for batch in tqdm(
            self.loader,
            desc="compute part area/frequency stats",
            disable=not self.show_progress,
        ):
            batch = to_device_batch(batch, self.device)

            cat_ids = batch["category_id"].long()             # [B]
            obj_mask = batch["obj_mask_patch"].bool()         # [B,N]
            part_gt = batch["part_gt_mask_patch"].bool()      # [B,K,N]
            part_valid = batch["part_valid_mask"].bool()      # [B,K]
            part_ids = batch["part_category_id"].long()       # [B,K]

            B, K, _ = part_gt.shape

            if int(cat_ids.max().item()) >= max_obj_slots:
                raise ValueError(
                    f"category_id max={int(cat_ids.max().item())} >= max_obj_slots={max_obj_slots}. "
                    f"Increase max_obj_slots."
                )

            obj_crop_cnt.index_add_(
                0,
                cat_ids,
                torch.ones_like(cat_ids, dtype=torch.float32),
            )

            gt_mask = part_gt & obj_mask[:, None, :] & part_valid[:, :, None]
            part_patch_cnt = gt_mask.sum(dim=-1).float()                       # [B,K]

            flat_pid = part_ids.reshape(-1)
            flat_cnt = part_patch_cnt.reshape(-1)
            flat_cat = cat_ids[:, None].expand(B, K).reshape(-1)
            flat_valid = part_valid.reshape(-1)

            valid = (flat_pid >= 0) & (flat_pid < P) & flat_valid

            if valid.any():
                part_to_cat[flat_pid[valid]] = flat_cat[valid]

            has_area = valid & (flat_cnt > 0)
            if has_area.any():
                part_total_patch_cnt.index_add_(0, flat_pid[has_area], flat_cnt[has_area])

                # Count each part at most once per object crop.
                flat_b = torch.arange(B, device=self.device)[:, None].expand(B, K).reshape(-1)
                present = torch.zeros(B, P, device=self.device, dtype=torch.bool)
                present[flat_b[has_area], flat_pid[has_area]] = True
                part_occurrence_cnt += present.float().sum(dim=0)

        safe_cat = part_to_cat.clamp_min(0)
        part_obj_crop_cnt = obj_crop_cnt[safe_cat]
        part_obj_crop_cnt[part_to_cat < 0] = 0

        avg_patch_count_per_obj_crop = (
            part_total_patch_cnt / part_obj_crop_cnt.clamp_min(1.0)
        )
        avg_patch_count_per_obj_crop[part_obj_crop_cnt <= 0] = 0

        avg_pixel_count_per_obj_crop = (
            avg_patch_count_per_obj_crop * float(self.patch_size * self.patch_size)
        )

        part_occurrence_freq_per_obj_crop = (
            part_occurrence_cnt / part_obj_crop_cnt.clamp_min(1.0)
        )
        part_occurrence_freq_per_obj_crop[part_obj_crop_cnt <= 0] = 0

        return {
            "avg_pixel_count_per_obj_crop": avg_pixel_count_per_obj_crop.detach().cpu(),
            "part_occurrence_freq_per_obj_crop": part_occurrence_freq_per_obj_crop.detach().cpu(),
        }

    @torch.no_grad()
    def compute_part_avg_pixel_count_per_obj_crop(
        self,
        max_obj_slots: int = 256,
    ) -> torch.Tensor:
        return self.compute_area_and_frequency(max_obj_slots=max_obj_slots)[
            "avg_pixel_count_per_obj_crop"
        ]

    @torch.no_grad()
    def compute_part_occurrence_freq_per_obj_crop(
        self,
        max_obj_slots: int = 256,
    ) -> torch.Tensor:
        return self.compute_area_and_frequency(max_obj_slots=max_obj_slots)[
            "part_occurrence_freq_per_obj_crop"
        ]
