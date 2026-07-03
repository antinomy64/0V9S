#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bake an orthogonal W checkpoint into the Stage1 text projector.

Current W training definition:
    text_z0 = stage1.<text_projector_fn>(part_ann_feats)
    text_z  = normalize(text_z0 @ W)

For a Linear projector:
    text_z0 = x @ A.T + b

Baked projector should output:
    text_z_baked = text_z0 @ W
                 = x @ A.T @ W + b @ W

Therefore:
    A_new = W.T @ A
    b_new = b @ W

This script:
  1. loads Stage1 model from config + base weights
  2. loads W from W checkpoint
  3. finds the Linear module used by --text_projector_module, or auto-detects one
  4. bakes W into that Linear module
  5. saves a checkpoint compatible with existing eval scripts

Recommended explicit usage:
  python scripts/bake_orthogonal_w_into_projector.py \
    --model_config configs/vitb_mlp_infonce.yaml \
    --base_weights weights/vitb_mlp_infonce_voc116_obj_test15.pth \
    --w_ckpt weights/stage2_w_anchor_test15_oracle_objmask/w_oracle_objmask_best.pth \
    --out_weights weights/vitb_mlp_infonce_voc116_obj_test15_oracle_objmaskW_best.pth \
    --text_projector_fn project_clip_txt

If auto-detection fails or finds multiple candidates, rerun with:
    --text_projector_module <module_name>

The script will print all candidate Linear modules.
"""

import argparse
import copy
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import yaml
from torch import nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def load_stage1_model(config_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"])

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    ret = model.load_state_dict(state_dict, strict=False)

    print(f"[Stage1] loaded base weights: {ckpt_path}")
    print("[Stage1] missing keys:", getattr(ret, "missing_keys", []))
    print("[Stage1] unexpected keys:", getattr(ret, "unexpected_keys", []))

    model.to(device)
    model.eval()
    return model


def load_w(w_ckpt_path: str) -> torch.Tensor:
    ckpt = torch.load(w_ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "W" in ckpt:
            W = ckpt["W"]
        elif "state_dict" in ckpt and "W" in ckpt["state_dict"]:
            W = ckpt["state_dict"]["W"]
        else:
            keys = list(ckpt.keys())
            raise KeyError(f"Cannot find W in checkpoint. Top-level keys: {keys}")
    elif torch.is_tensor(ckpt):
        W = ckpt
    else:
        raise TypeError(f"Unsupported W checkpoint type: {type(ckpt)}")

    if not torch.is_tensor(W):
        W = torch.as_tensor(W)
    W = W.float().cpu()

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square [D,D], got shape={tuple(W.shape)}")

    eye = torch.eye(W.shape[0], dtype=W.dtype)
    ortho_err = ((W.T @ W - eye) ** 2).mean().sqrt().item()
    print(f"[W] loaded: {w_ckpt_path}")
    print(f"[W] shape={tuple(W.shape)}, rms_orth_error={ortho_err:.8e}")
    return W


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if module_name not in modules:
        available = list(modules.keys())
        raise KeyError(
            f"Module '{module_name}' not found. Available module names include:\n"
            + "\n".join(available[:200])
        )
    return modules[module_name]


def list_linear_candidates(model: nn.Module, W_dim: int) -> List[Tuple[str, nn.Linear]]:
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.out_features == W_dim:
                candidates.append((name, module))
    return candidates


def auto_select_linear(model: nn.Module, W_dim: int, text_projector_fn: str = "") -> Tuple[str, nn.Linear]:
    candidates = list_linear_candidates(model, W_dim)

    print("\n[Candidates] Linear modules with out_features == W_dim")
    if len(candidates) == 0:
        print("  <none>")
    for name, module in candidates:
        print(f"  {name}: Linear(in={module.in_features}, out={module.out_features}, bias={module.bias is not None})")

    if len(candidates) == 0:
        raise RuntimeError(
            f"No Linear module with out_features={W_dim} found. "
            "Pass --text_projector_module explicitly if the projector is not a simple nn.Linear."
        )

    # Prefer module names that look text/clip-related.
    preferred_tokens = ["clip", "txt", "text", "ann"]
    scored = []
    for name, module in candidates:
        low = name.lower()
        score = sum(tok in low for tok in preferred_tokens)
        # Mild boost if method name shares tokens.
        if text_projector_fn:
            fn_low = text_projector_fn.lower()
            for tok in preferred_tokens:
                if tok in fn_low and tok in low:
                    score += 1
        scored.append((score, name, module))

    best_score = max(s for s, _, _ in scored)
    best = [(name, module) for s, name, module in scored if s == best_score]

    if len(best) == 1 and best_score > 0:
        print(f"[Auto] selected text projector module: {best[0][0]}")
        return best[0]

    if len(candidates) == 1:
        print(f"[Auto] only one candidate, selected: {candidates[0][0]}")
        return candidates[0]

    raise RuntimeError(
        "Auto-detection is ambiguous. Please rerun with --text_projector_module.\n"
        "Candidate module names:\n" + "\n".join([name for name, _ in candidates])
    )


def verify_method_output_changed(
    model_before: nn.Module,
    model_after: nn.Module,
    method_name: str,
    W: torch.Tensor,
    device: torch.device,
    text_dim: int,
) -> None:
    if not method_name:
        return
    if not hasattr(model_before, method_name) or not hasattr(model_after, method_name):
        print(f"[Verify] Skip method check: model has no method {method_name}")
        return

    fn_before = getattr(model_before, method_name)
    fn_after = getattr(model_after, method_name)
    if not callable(fn_before) or not callable(fn_after):
        print(f"[Verify] Skip method check: {method_name} is not callable")
        return

    x = torch.randn(4, text_dim, device=device)
    Wd = W.to(device=device)

    with torch.no_grad():
        y_before = fn_before(x).float()
        y_after = fn_after(x).float()
        y_expected = y_before @ Wd

    max_err = (y_after - y_expected).abs().max().item()
    mean_err = (y_after - y_expected).abs().mean().item()
    print(f"[Verify] {method_name}: max|after - before@W|={max_err:.8e}, mean={mean_err:.8e}")


def save_baked_checkpoint(
    out_path: str,
    base_weights_path: str,
    w_ckpt_path: str,
    model: nn.Module,
    module_name: str,
    W: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    base_ckpt = torch.load(base_weights_path, map_location="cpu")
    baked_state = model.state_dict()

    if isinstance(base_ckpt, dict) and "state_dict" in base_ckpt:
        out_ckpt = copy.deepcopy(base_ckpt)
        out_ckpt["state_dict"] = baked_state
        out_ckpt["baked_W_info"] = {
            "w_ckpt": w_ckpt_path,
            "text_projector_module": module_name,
            "text_projector_fn": args.text_projector_fn,
            "formula": "linear.weight <- W.T @ weight; linear.bias <- bias @ W",
            "W_shape": list(W.shape),
        }
    else:
        # Existing eval code often accepts a raw state_dict. Save raw state_dict if the base was raw.
        out_ckpt = baked_state

    torch.save(out_ckpt, out_path)
    print(f"[Save] baked checkpoint: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--base_weights", required=True)
    parser.add_argument("--w_ckpt", required=True)
    parser.add_argument("--out_weights", required=True)
    parser.add_argument("--text_projector_fn", default="project_clip_txt")
    parser.add_argument("--text_projector_module", default="",
                        help="Exact nn.Linear module name to bake W into. If empty, auto-detect.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    W = load_w(args.w_ckpt)
    W_dim = W.shape[0]

    model_before = load_stage1_model(args.model_config, args.base_weights, device)
    model_after = load_stage1_model(args.model_config, args.base_weights, device)

    if args.text_projector_module:
        module_name = args.text_projector_module
        module = get_module_by_name(model_after, module_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"--text_projector_module must point to nn.Linear, got {type(module)}")
        if module.out_features != W_dim:
            raise ValueError(
                f"Module out_features={module.out_features} but W_dim={W_dim}. "
                "This does not match current W training definition."
            )
    else:
        module_name, module = auto_select_linear(model_after, W_dim, args.text_projector_fn)

    W_device = W.to(device=module.weight.device, dtype=module.weight.dtype)
    with torch.no_grad():
        old_weight = module.weight.data.clone()
        module.weight.data.copy_(W_device.T @ module.weight.data)
        if module.bias is not None:
            old_bias = module.bias.data.clone()
            module.bias.data.copy_(module.bias.data @ W_device)

    print(f"[Bake] baked W into module: {module_name}")
    print(f"[Bake] weight shape: {tuple(module.weight.shape)}")
    print(f"[Bake] formula: weight_new = W.T @ weight_old; bias_new = bias_old @ W")

    # Verification uses module input dim.
    verify_method_output_changed(
        model_before=model_before,
        model_after=model_after,
        method_name=args.text_projector_fn,
        W=W,
        device=device,
        text_dim=module.in_features,
    )

    save_baked_checkpoint(
        out_path=args.out_weights,
        base_weights_path=args.base_weights,
        w_ckpt_path=args.w_ckpt,
        model=model_after.cpu(),
        module_name=module_name,
        W=W,
        args=args,
    )


if __name__ == "__main__":
    main()
