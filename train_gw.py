from __future__ import annotations

import argparse
import importlib
import json
import os

import torch
import yaml

from src.dataset_joint import DinoClipJointDataset
from src.train_util_gw import do_train_gw


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(config: dict, device: str):
    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"])
    model.to(device)
    return model


def load_init_weights(model, init_weights: str):
    if not init_weights:
        return

    print(f"[load init] {init_weights}")
    ckpt = torch.load(init_weights, map_location="cpu")
    ret = model.load_state_dict(ckpt, strict=False)
    if ret is not None:
        print("  missing keys   :", getattr(ret, "missing_keys", []))
        print("  unexpected keys:", getattr(ret, "unexpected_keys", []))


def build_dataset(args, config: dict, split: str):
    dataset_path = args.train_dataset if split == "train" else args.val_dataset
    dataset_cfg = config.get("dataset", {})
    min_obj_area_ratio = float(dataset_cfg.get("min_obj_area_ratio", 0.0)) if split == "train" else 0.0

    return DinoClipJointDataset(
        dataset_path,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        with_background=args.with_background,
        is_wds=".tar" in dataset_path,
        path_prefix=args.path_prefix,
        min_obj_area_ratio=min_obj_area_ratio,
    )


def train_gw(
    args,
):
    os.makedirs("weights", exist_ok=True)

    with open(args.model_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    proj_class = os.path.basename(args.model_config).split(".")[0]
    model_name = proj_class
    if args.name_pedix:
        model_name += f"_{args.name_pedix}"

    out_path = os.path.join("weights", model_name)

    model = build_model(config, DEVICE)
    load_init_weights(model, args.init_weights)
    print(model)

    train_dataset = build_dataset(args, config, split="train")
    val_dataset = build_dataset(args, config, split="val")

    model, train_history, val_history = do_train_gw(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_cfg=config["train"],
        seed=args.seed,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        warmup=args.warmup,
    )

    torch.save(model.state_dict(), f"{out_path}.pth")
    print(f"[saved model] {out_path}.pth")

    with open(f"{out_path}_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "proj_class": proj_class,
                "proj_name": model_name,
                "init_weights": args.init_weights,
                "train": train_history,
                "val": val_history,
            },
            f,
            indent=2,
        )
    print(f"[saved history] {out_path}_history.json")
    print(f"[eval reminder] use: --opts model.proj_class={proj_class} model.proj_name={model_name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)

    parser.add_argument("--obj_feature_name", type=str, default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", type=str, default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", type=str, default="ann_feats")
    parser.add_argument("--part_text_name", type=str, default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", type=str, default=None)

    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--name_pedix", type=str, default="")
    parser.add_argument("--init_weights", type=str, default="")

    args = parser.parse_args()
    train_gw(args)


if __name__ == "__main__":
    main()
