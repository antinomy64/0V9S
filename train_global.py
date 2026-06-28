import argparse
import importlib
import json
import os

import torch
import yaml

from src.dataset_joint import DinoClipJointDataset
from src.train_util_global import do_train


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_and_eval(
    config_file,
    train_dataset,
    val_dataset,
    optimizer="AdamW",
    weight_decay=0.05,
    scheduler='linear',
    warmup=0,
    name_pedix='',
    init_weights='',
    audit_out_txt='',
    train_select_dataset=None,
    val_select_dataset=None,
):
    out_dir = 'weights'
    os.makedirs(out_dir, exist_ok=True)

    proj_class = os.path.basename(config_file).split('.')[0]
    model_name = proj_class
    if name_pedix:
        model_name += f"_{name_pedix}"
    out_path = os.path.join(out_dir, model_name)

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    model_class_name = config['model'].get('model_class', 'ProjectionLayer')
    ModelClass = getattr(importlib.import_module('src.model'), model_class_name)
    model = ModelClass.from_config(config['model'])
    model.to(device)

    if init_weights:
        print(f"Loading init weights from {init_weights}")
        ckpt = torch.load(init_weights, map_location='cpu')
        ret = model.load_state_dict(ckpt, strict=False)
        if ret is not None:
            print("Missing keys:", getattr(ret, "missing_keys", []))
            print("Unexpected keys:", getattr(ret, "unexpected_keys", []))

    print(model)

    model, train_history, val_history = do_train(
        model,
        train_dataset,
        val_dataset,
        config['train'],
        optimizer_name=optimizer,
        weight_decay=weight_decay,
        scheduler_name=scheduler,
        warmup=warmup,
        eval_proj_name=model_name,
        audit_out_txt=audit_out_txt,
        train_select_dataset=train_select_dataset,
        val_select_dataset=val_select_dataset,
    )

    torch.save(model.state_dict(), f"{out_path}.pth")
    print(f"Saved model at {out_path}.pth")

    with open(f"{out_path}_history.json", 'w') as f:
        json.dump({"train": train_history, "val": val_history}, f, indent=2)
    print(f"Saved training history at {out_path}_history.json")


def _build_joint_dataset(
    dataset_path,
    args,
    part_feature_name,
    min_obj_area_ratio,
    is_wds,
    class_part_bank=None,
):
    return DinoClipJointDataset(
        dataset_path,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=is_wds,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=min_obj_area_ratio,
        class_part_bank=class_part_bank,
    )


def _samples(dataset):
    return list(dataset.data.values()) if isinstance(dataset.data, dict) else list(dataset.data)


def _sample_has_parts(sample):
    part_ids = sample.get("part_category_id", None)

    if part_ids is None:
        return False

    if torch.is_tensor(part_ids):
        return part_ids.numel() > 0

    if isinstance(part_ids, (list, tuple)):
        return len(part_ids) > 0

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, required=True, help='Path to train .pth/.tar dataset')
    parser.add_argument('--val_dataset', type=str, required=True, help='Path to val .pth/.tar dataset')
    parser.add_argument('--model_config', type=str, required=True, help='Path to YAML model config')

    parser.add_argument('--obj_feature_name', type=str, default='avg_self_attn_out', help='Image feature field for object branch')
    parser.add_argument('--part_feature_name', type=str, default='patch_tokens', help='Legacy patch token field. Used when select/target names are omitted.')
    parser.add_argument('--part_select_feature_name', type=str, default=None, help='Patch token field for anchor/EM selection, e.g. cropaug_patch_tokens_coral_a075')
    parser.add_argument('--part_target_feature_name', type=str, default=None, help='Patch token field for final learning target, e.g. cropaug_patch_tokens')
    parser.add_argument('--obj_text_name', type=str, default='ann_feats', help='Object text feature field')
    parser.add_argument('--part_text_name', type=str, default='part_ann_feats', help='Part text feature field')

    parser.add_argument('--resize_dim', type=int, default=448, help='Resize dim used during DINO extraction')
    parser.add_argument('--crop_dim', type=int, default=448, help='Crop dim used during DINO extraction')
    parser.add_argument('--patch_size', type=int, default=14, help='Patch size for ViT patch grid')
    parser.add_argument('--with_background', action='store_true', default=False, help='Set true if segmentation masks are 1..C with 0 as background')
    parser.add_argument('--path_prefix', type=str, default=None, help='Optional prefix to resolve image/mask relative paths')

    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to be used')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay to be used')
    parser.add_argument('--scheduler', type=str, default='linear', help='Scheduler to be used')
    parser.add_argument('--warmup', type=int, default=0, help='Number of warmup steps')
    parser.add_argument('--name_pedix', type=str, default='', help='String appended to output model name')
    parser.add_argument('--init_weights', type=str, default='', help='Path to existing projector checkpoint used to initialize finetuning')
    parser.add_argument(
        '--audit_out_txt',
        type=str,
        default='',
        help='Optional output txt for per-epoch full-pool anchor/prototype audit.',
    )

    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        config = yaml.safe_load(f)
    dataset_cfg = config.get('dataset', {})
    min_obj_area_ratio = float(dataset_cfg.get('min_obj_area_ratio', 0.0))

    is_train_wds = '.tar' in args.train_dataset
    is_val_wds = '.tar' in args.val_dataset

    part_select_feature_name = args.part_select_feature_name or args.part_feature_name
    part_target_feature_name = args.part_target_feature_name or args.part_feature_name

    print("[dual global features]")
    print(f"  part_select_feature_name = {part_select_feature_name}")
    print(f"  part_target_feature_name = {part_target_feature_name}")
    print("  selection feature drives anchor/EM; target feature drives pseudo prototype learning.")

    train_dataset = _build_joint_dataset(
        args.train_dataset,
        args,
        part_feature_name=part_target_feature_name,
        min_obj_area_ratio=min_obj_area_ratio,
        is_wds=is_train_wds,
        class_part_bank=None,
    )
    val_dataset = _build_joint_dataset(
        args.val_dataset,
        args,
        part_feature_name=part_target_feature_name,
        min_obj_area_ratio=0.0,
        is_wds=is_val_wds,
        class_part_bank=train_dataset.class_part_bank,
    )

    # Build a parallel select dataset only when the select key differs from target key.
    if part_select_feature_name == part_target_feature_name:
        train_select_dataset = None
        val_select_dataset = None
    else:
        train_select_dataset = _build_joint_dataset(
            args.train_dataset,
            args,
            part_feature_name=part_select_feature_name,
            min_obj_area_ratio=min_obj_area_ratio,
            is_wds=is_train_wds,
            class_part_bank=train_dataset.class_part_bank,
        )
        val_select_dataset = _build_joint_dataset(
            args.val_dataset,
            args,
            part_feature_name=part_select_feature_name,
            min_obj_area_ratio=0.0,
            is_wds=is_val_wds,
            class_part_bank=train_dataset.class_part_bank,
        )

    if train_dataset.part_taxonomy != val_dataset.part_taxonomy:
        raise ValueError(
            "Train/val part taxonomy mismatch: "
            f"train={train_dataset.part_taxonomy}, val={val_dataset.part_taxonomy}"
        )
    if train_select_dataset is not None:
        if train_dataset.part_taxonomy != train_select_dataset.part_taxonomy:
            raise ValueError(
                "Target/select train taxonomy mismatch: "
                f"target={train_dataset.part_taxonomy}, select={train_select_dataset.part_taxonomy}"
            )
        if val_dataset.part_taxonomy != val_select_dataset.part_taxonomy:
            raise ValueError(
                "Target/select val taxonomy mismatch: "
                f"target={val_dataset.part_taxonomy}, select={val_select_dataset.part_taxonomy}"
            )
        if len(_samples(train_dataset)) != len(_samples(train_select_dataset)):
            raise ValueError("Target/select train dataset sample count mismatch.")
        if len(_samples(val_dataset)) != len(_samples(val_select_dataset)):
            raise ValueError("Target/select val dataset sample count mismatch.")

    train_categories = set(int(x) for x in train_dataset.class_part_bank.keys())
    val_samples = _samples(val_dataset)

    val_categories = {
        int(sample["category_id"])
        for sample in val_samples
    }

    val_categories_with_parts = {
        int(sample["category_id"])
        for sample in val_samples
        if _sample_has_parts(sample)
    }

    missing_val_part_categories = sorted(
        val_categories_with_parts - train_categories
    )

    if missing_val_part_categories:
        raise ValueError(
            "Validation categories with part annotations are missing from "
            "train class_part_bank: "
            f"{missing_val_part_categories}"
        )

    val_categories_without_parts = sorted(
        val_categories - val_categories_with_parts
    )

    if val_categories_without_parts:
        print(
            "[dataset] validation categories without part annotations will "
            "be skipped by the global part pool: "
            f"{val_categories_without_parts}"
        )

    print(f"[dataset] with_background={args.with_background}")
    print(f"[dataset] train/val taxonomy={train_dataset.part_taxonomy}")
    print(f"[dataset] fixed class_part_bank categories={len(train_categories)}")

    train_and_eval(
        args.model_config,
        train_dataset,
        val_dataset,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup=args.warmup,
        name_pedix=args.name_pedix,
        init_weights=args.init_weights,
        audit_out_txt=args.audit_out_txt,
        train_select_dataset=train_select_dataset,
        val_select_dataset=val_select_dataset,
    )
