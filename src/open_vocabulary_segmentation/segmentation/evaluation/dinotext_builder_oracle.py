# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
import mmcv
import torch
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.datasets.pipelines import Compose
from omegaconf import OmegaConf
from datasets import get_template

from .dinotext_seg_oracle import DINOTextSegInference, OracleCropAugSegInference
from .oracle_cropaug_proto import build_cropaug_visual_prototypes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    kwargs = dict(with_bg=with_bg)
    if hasattr(dset_cfg, "test_cfg"):
        kwargs["test_cfg"] = dset_cfg.test_cfg

    # ------------------------------------------------------------------
    # Oracle mode: use cropaug visual prototypes instead of text embeddings
    # Activate by adding these keys under cfg.evaluate:
    #   oracle_train_pth: path to train pth with cropaug_patch_tokens
    #   oracle_part_mask_dir: root dir of GT part masks
    # Optional:
    #   oracle_grid_size: default 32
    #   oracle_one_based_part_ids: default False
    # ------------------------------------------------------------------
    if hasattr(config.evaluate, "oracle_train_pth") and config.evaluate.oracle_train_pth is not None:
        grid_size = int(getattr(config.evaluate, "oracle_grid_size", 32))
        one_based = bool(getattr(config.evaluate, "oracle_one_based_part_ids", False))

        visual_prototypes, valid_proto = build_cropaug_visual_prototypes(
            train_pth=config.evaluate.oracle_train_pth,
            part_mask_dir=config.evaluate.oracle_part_mask_dir,
            grid_size=grid_size,
            one_based_part_ids=one_based,
        )

        # dataset class order is assumed to be identical to global part-id order
        if visual_prototypes.shape[0] != len(classnames):
            raise ValueError(
                f"Prototype count {visual_prototypes.shape[0]} does not match dataset classes {len(classnames)}."
            )

        seg_model = OracleCropAugSegInference(
            model,
            visual_prototypes.to(device),
            classnames,
            **kwargs,
            **config.evaluate,
        )
    else:
        text_tokens = model.build_dataset_class_tokens(config.evaluate.template, classnames)
        text_embedding = model.build_text_embedding(text_tokens)
        model_type = config.model.type
        if model_type == "DINOText":
            seg_model = DINOTextSegInference(model, text_embedding, classnames, **kwargs, **config.evaluate)
        else:
            raise ValueError(model_type)

    seg_model.CLASSES = dataset.dataset.CLASSES
    seg_model.PALETTE = dataset.dataset.PALETTE

    return seg_model
