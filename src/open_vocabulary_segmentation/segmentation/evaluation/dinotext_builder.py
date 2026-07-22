# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import torch
import torch.nn.functional as F
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.datasets.pipelines import Compose
from omegaconf import OmegaConf
from datasets import get_template

from .dinotext_seg import DINOTextSegInference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_precomputed_obj_text_from_annos(model, config, classnames):
    """
    Load object-level precomputed text features.

    Expected annotation format:
        category_id: int
        ann_feats: [D]

    All annotations belonging to the same object category are averaged,
    producing [num_classes, D].
    """
    pth_path = config.model.get("precomputed_text_pth", "")
    text_key = config.model.get("precomputed_text_key", "ann_feats")
    id_key = config.model.get("precomputed_part_id_key", "category_id")

    if not pth_path:
        raise ValueError("model.precomputed_text_pth is empty.")

    data = torch.load(pth_path, map_location="cpu")
    if "annotations" not in data:
        raise KeyError(
            f"{pth_path} has no top-level key 'annotations'."
        )

    num_classes = len(classnames)
    sums = {}
    counts = {}

    total_ann = 0
    used_ann = 0

    for ann_idx, ann in enumerate(data["annotations"]):
        total_ann += 1

        if id_key not in ann or text_key not in ann:
            continue

        class_id = ann[id_key]
        if torch.is_tensor(class_id):
            if class_id.numel() != 1:
                raise ValueError(
                    f"ann[{id_key}] must contain one object id, "
                    f"got shape {tuple(class_id.shape)}"
                )
            class_id = int(class_id.item())
        else:
            class_id = int(class_id)

        feat = ann[text_key]
        if not torch.is_tensor(feat):
            feat = torch.as_tensor(feat)

        feat = feat.float()

        if feat.ndim != 1:
            raise ValueError(
                f"ann[{text_key}] must be [D] for object-level eval, "
                f"got {tuple(feat.shape)} at annotation {ann_idx}"
            )

        if class_id < 0 or class_id >= num_classes:
            raise ValueError(
                f"object category id {class_id} is outside "
                f"[0, {num_classes - 1}]"
            )

        if class_id not in sums:
            sums[class_id] = feat.clone()
            counts[class_id] = 1
        else:
            sums[class_id] += feat
            counts[class_id] += 1

        used_ann += 1

    missing = [
        class_id
        for class_id in range(num_classes)
        if class_id not in sums
    ]
    if missing:
        raise ValueError(
            f"Missing object text features for "
            f"{len(missing)}/{num_classes} classes. "
            f"Missing ids: {missing[:30]}"
        )

    raw_feats = torch.stack(
        [
            sums[class_id] / counts[class_id]
            for class_id in range(num_classes)
        ],
        dim=0,
    ).float().to(device)

    print(
        "[PrecomputedObjectText]",
        f"pth={pth_path},",
        f"text_key={text_key},",
        f"id_key={id_key},",
        f"total_ann={total_ann},",
        f"used_ann={used_ann},",
        f"raw_shape={tuple(raw_feats.shape)}",
    )

    text_embedding = model.proj.project_clip_txt(raw_feats)
    text_embedding = F.normalize(
        text_embedding.float(),
        dim=-1,
    )

    print(
        "[PrecomputedObjectText] projected:",
        tuple(text_embedding.shape),
    )

    return text_embedding

def _load_precomputed_part_text_from_annos(model, config, classnames):
    pth_path = config.model.get("precomputed_text_pth", "")
    part_text_key = config.model.get("precomputed_text_key", "part_ann_feats")
    part_id_key = config.model.get("precomputed_part_id_key", "part_category_id")

    if not pth_path:
        raise ValueError("model.precomputed_text_pth is empty.")

    data = torch.load(pth_path, map_location="cpu")
    if "annotations" not in data:
        raise KeyError(f"{pth_path} has no top-level key 'annotations'.")

    num_parts = len(classnames)
    sums = {}
    counts = {}

    total_ann = 0
    used_ann = 0
    total_part_feats = 0

    for ann in data["annotations"]:
        total_ann += 1

        if part_id_key not in ann:
            continue
        if part_text_key not in ann:
            continue

        part_ids = ann[part_id_key]
        part_feats = ann[part_text_key]

        if not torch.is_tensor(part_feats):
            part_feats = torch.as_tensor(part_feats)
        part_feats = part_feats.float()

        if part_feats.ndim != 2:
            raise ValueError(
                f"ann[{part_text_key}] must be [K,D], got {tuple(part_feats.shape)}"
            )

        if len(part_ids) != part_feats.shape[0]:
            raise ValueError(
                f"len(ann[{part_id_key}])={len(part_ids)} != "
                f"ann[{part_text_key}].shape[0]={part_feats.shape[0]}"
            )

        used_ann += 1
        total_part_feats += int(part_feats.shape[0])

        for pid, feat in zip(part_ids, part_feats):
            pid = int(pid)

            if pid < 0 or pid >= num_parts:
                raise ValueError(
                    f"part id {pid} out of range [0, {num_parts - 1}]. "
                    f"Check whether dataset CLASSES order matches pth part_category_id."
                )

            if pid not in sums:
                sums[pid] = feat.clone()
                counts[pid] = 1
            else:
                sums[pid] += feat
                counts[pid] += 1

    missing = [i for i in range(num_parts) if i not in sums]
    if missing:
        raise ValueError(
            f"Missing part text features for {len(missing)} / {num_parts} parts. "
            f"First missing ids: {missing[:30]}"
        )

    raw_feats = torch.stack(
        [sums[i] / counts[i] for i in range(num_parts)],
        dim=0,
    ).float().to(device)

    print(
        "[PrecomputedText] loaded from annotations:",
        f"pth={pth_path},",
        f"key={part_text_key},",
        f"part_id_key={part_id_key},",
        f"total_ann={total_ann},",
        f"used_ann={used_ann},",
        f"total_part_feats={total_part_feats},",
        f"raw_shape={tuple(raw_feats.shape)}",
    )

    text_embedding = model.proj.project_clip_txt(raw_feats)
    text_embedding = F.normalize(text_embedding.float(), dim=-1)

    print(
        "[PrecomputedText] projected:",
        f"shape={tuple(text_embedding.shape)}"
    )

    return text_embedding

def build_dinotext_seg_inference(
    model,
    dataset,
    config,
    seg_config,
):
    dset_cfg = mmcv.Config.fromfile(seg_config)  # dataset config
    with_bg = dataset.dataset.CLASSES[0] == "background"
    if with_bg:
        classnames = dataset.dataset.CLASSES[1:]
    else:
        classnames = dataset.dataset.CLASSES
    if config.model.get("precomputed_text_pth", ""):
        precomputed_text_level = config.model.get(
            "precomputed_text_level",
            "part",
        )

        if precomputed_text_level == "obj":
            text_embedding = _load_precomputed_obj_text_from_annos(
                model=model,
                config=config,
                classnames=classnames,
            )
        elif precomputed_text_level == "part":
            text_embedding = _load_precomputed_part_text_from_annos(
                model=model,
                config=config,
                classnames=classnames,
            )
        else:
            raise ValueError(
                "model.precomputed_text_level must be 'obj' or 'part', "
                f"got {precomputed_text_level}"
            )
    else:
        # 原版 Talk2DINO 逻辑，保持不变
        text_tokens = model.build_dataset_class_tokens(
            config.evaluate.template,
            classnames,
        )
        text_embedding = model.build_text_embedding(text_tokens)
    kwargs = dict(with_bg=with_bg)
    if hasattr(dset_cfg, "test_cfg"):
        kwargs["test_cfg"] = dset_cfg.test_cfg

    model_type = config.model.type
    if model_type == "DINOText":
        seg_model = DINOTextSegInference(model, text_embedding, classnames, **kwargs, **config.evaluate)
    else:
        raise ValueError(model_type)

    seg_model.CLASSES = dataset.dataset.CLASSES
    seg_model.PALETTE = dataset.dataset.PALETTE

    return seg_model
