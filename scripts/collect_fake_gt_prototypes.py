from __future__ import annotations

import importlib
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn
from src.loss_joint import JointObjPartLoss
from src.voc116_part_coarse import COARSE_PART_CLASSES, FINE_PART_CLASSES


class FeatureAnlyser:
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
    def part_names(self):
        if self.num_parts == 58:
            return list(COARSE_PART_CLASSES)
        if self.num_parts == 116:
            return list(FINE_PART_CLASSES)
        return [f"part_{i}" for i in range(self.num_parts)]

    def collect(self):
        P = self.num_parts
        D = int(self.cfg["model"].get("dino_embed_dim", 768))

        fake_sum = torch.zeros(P, D, device=self.device)
        fake_cnt = torch.zeros(P, device=self.device)

        gt_sum = torch.zeros(P, D, device=self.device)
        gt_cnt = torch.zeros(P, device=self.device)

        with torch.no_grad():
            for batch in tqdm(
                self.loader,
                desc="collect fake/GT prototypes",
                disable=not self.show_progress,
            ):
                batch = {
                    k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }

                part_text = batch["part_text_feat"].float()
                patch_tokens = self.loss_helper._safe_normalize(
                    batch["patch_tokens"].float(), dim=-1
                )
                obj_mask = batch["obj_mask_patch"].bool()
                part_valid = batch["part_valid_mask"].bool()
                part_gt = batch["part_gt_mask_patch"].bool()
                part_ids = batch["part_category_id"].long()

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

                # fake prototype from Stage2 pseudo/EM prototype.
                proto_part = self.loss_helper._safe_normalize(pool_ret[1].float(), dim=-1)
                anchor_valid = pool_ret[-1].bool()

                # GT prototype from GT part mask.
                gt_mask = part_gt & obj_mask[:, None, :] & part_valid[:, :, None]
                gt_pix_cnt = gt_mask.sum(dim=-1).float()

                gt_proto = torch.einsum("bkn,bnd->bkd", gt_mask.float(), patch_tokens)
                gt_proto = gt_proto / gt_pix_cnt.clamp_min(1.0)[:, :, None]
                gt_proto = self.loss_helper._safe_normalize(gt_proto, dim=-1)

                fake_valid = part_valid & anchor_valid
                gt_valid = part_valid & (gt_pix_cnt > 0)

                for b in range(part_ids.shape[0]):
                    for k in range(part_ids.shape[1]):
                        pid = int(part_ids[b, k].item())
                        if pid < 0 or pid >= P:
                            continue

                        if bool(fake_valid[b, k]):
                            fake_sum[pid] += proto_part[b, k]
                            fake_cnt[pid] += 1

                        if bool(gt_valid[b, k]):
                            gt_sum[pid] += gt_proto[b, k]
                            gt_cnt[pid] += 1

        fake_proto = fake_sum / fake_cnt.clamp_min(1.0)[:, None]
        fake_proto = self.loss_helper._safe_normalize(fake_proto, dim=-1)
        fake_proto[fake_cnt <= 0] = 0

        gt_proto = gt_sum / gt_cnt.clamp_min(1.0)[:, None]
        gt_proto = self.loss_helper._safe_normalize(gt_proto, dim=-1)
        gt_proto[gt_cnt <= 0] = 0

        return fake_proto, gt_proto