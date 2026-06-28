
from copy import deepcopy
import json
import os
import random
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_joint import joint_collate_fn
from src.loss_joint import JointObjPartLoss


def set_seed(seed: int):
    print(f'Setting seed {seed}...')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def _move_joint_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _detach_metric_dict(metrics: Dict) -> Dict:
    """Detach tensors before storing epoch statistics.

    Keeping the original loss dictionary would retain the computation graph
    referenced by ``total`` for the whole epoch.
    """
    out = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            out[key] = value.detach().float().cpu()
        else:
            out[key] = value
    return out


def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _mean_dict(list_of_dicts):
    if len(list_of_dicts) == 0:
        return {}
    keys = list(list_of_dicts[0].keys())
    out = {}

    # Anchor metrics should be aggregated by counts, not by simple batch mean.
    if "anchor_total_valid_parts" in keys and "anchor_total_hits" in keys:
        total_valid = 0.0
        total_hits = 0.0
        for d in list_of_dicts:
            total_valid += _as_float(d["anchor_total_valid_parts"])
            total_hits += _as_float(d["anchor_total_hits"])

        out["anchor_total_valid_parts"] = total_valid
        out["anchor_total_hits"] = total_hits
        out["anchor_hit_rate"] = 0.0 if total_valid <= 0 else total_hits / total_valid

    # Dustbin audit metrics: aggregate counts globally, then derive rates.
    dustbin_count_keys = {
        "dustbin_valid_parts",
        "dustbin_active_parts",
        "dustbin_dropped_parts",
        "dustbin_fallback_images",
        "dustbin_score_count",
        "dustbin_score_sum",
        "dustbin_score_sq_sum",
        "dustbin_gt_present_total",
        "dustbin_gt_present_active",
        "dustbin_gt_present_dropped",
        "dustbin_gt_absent_total",
        "dustbin_gt_absent_dropped",
        "dustbin_gt_absent_kept",
    }
    dustbin_derived_keys = {
        "dustbin_active_ratio",
        "dustbin_score_mean",
        "dustbin_score_std",
        "dustbin_score_min",
        "dustbin_score_max",
        "dustbin_gt_present_keep_rate",
        "dustbin_gt_absent_drop_rate",
    }
    if "dustbin_valid_parts" in keys:
        for k in dustbin_count_keys:
            if k in keys:
                out[k] = sum(_as_float(d[k]) for d in list_of_dicts)

        # min/max need special aggregation.
        if "dustbin_score_min" in keys:
            mins = [_as_float(d["dustbin_score_min"]) for d in list_of_dicts if _as_float(d.get("dustbin_score_count", 0.0)) > 0]
            out["dustbin_score_min"] = min(mins) if len(mins) > 0 else 0.0
        if "dustbin_score_max" in keys:
            maxs = [_as_float(d["dustbin_score_max"]) for d in list_of_dicts if _as_float(d.get("dustbin_score_count", 0.0)) > 0]
            out["dustbin_score_max"] = max(maxs) if len(maxs) > 0 else 0.0

        valid = out.get("dustbin_valid_parts", 0.0)
        active = out.get("dustbin_active_parts", 0.0)
        out["dustbin_active_ratio"] = 0.0 if valid <= 0 else active / valid

        score_count = out.get("dustbin_score_count", 0.0)
        score_sum = out.get("dustbin_score_sum", 0.0)
        score_sq_sum = out.get("dustbin_score_sq_sum", 0.0)
        if score_count > 0:
            score_mean = score_sum / score_count
            score_var = max(score_sq_sum / score_count - score_mean ** 2, 0.0)
            out["dustbin_score_mean"] = score_mean
            out["dustbin_score_std"] = float(score_var ** 0.5)
        else:
            out["dustbin_score_mean"] = 0.0
            out["dustbin_score_std"] = 0.0

        present_total = out.get("dustbin_gt_present_total", 0.0)
        present_active = out.get("dustbin_gt_present_active", 0.0)
        out["dustbin_gt_present_keep_rate"] = 0.0 if present_total <= 0 else present_active / present_total

        absent_total = out.get("dustbin_gt_absent_total", 0.0)
        absent_dropped = out.get("dustbin_gt_absent_dropped", 0.0)
        out["dustbin_gt_absent_drop_rate"] = 0.0 if absent_total <= 0 else absent_dropped / absent_total

        # Average static settings for logging.
        if "dustbin_enabled" in keys:
            out["dustbin_enabled"] = sum(_as_float(d["dustbin_enabled"]) for d in list_of_dicts) / len(list_of_dicts)
        if "dustbin_tau" in keys:
            out["dustbin_tau"] = sum(_as_float(d["dustbin_tau"]) for d in list_of_dicts) / len(list_of_dicts)
        if "dustbin_topk" in keys:
            out["dustbin_topk"] = sum(_as_float(d["dustbin_topk"]) for d in list_of_dicts) / len(list_of_dicts)

    skip_keys = {
        "anchor_hit_rate", "anchor_total_valid_parts", "anchor_total_hits",
        *dustbin_count_keys, *dustbin_derived_keys,
        "dustbin_enabled", "dustbin_tau", "dustbin_topk",
    }

    for k in keys:
        if k in skip_keys:
            continue
        vals = []
        for d in list_of_dicts:
            v = d[k]
            if torch.is_tensor(v):
                vals.append(v.detach().float().cpu())
            else:
                vals.append(torch.tensor(float(v)))
        out[k] = torch.stack(vals).mean().item()
    return out


def train_joint(model, train_dataloader, criterion, optimizer, scheduler=None, epoch=0):
    model.train()
    device = next(model.parameters()).device
    prev_iter = epoch * len(train_dataloader)

    running = []
    pbar = tqdm(train_dataloader)
    for n_batch, batch in enumerate(pbar):
        batch = _move_joint_batch_to_device(batch, device)

        if scheduler is not None:
            scheduler(n_batch + prev_iter)

        losses = criterion(batch)
        total_loss = losses["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running.append(_detach_metric_dict(losses))
        db_active = losses.get("dustbin_active_ratio", torch.tensor(0.0, device=total_loss.device))
        db_keep = losses.get("dustbin_gt_present_keep_rate", torch.tensor(0.0, device=total_loss.device))
        db_drop = losses.get("dustbin_gt_absent_drop_rate", torch.tensor(0.0, device=total_loss.device))
        pbar.set_description(
            f"train total={losses['total'].item():.4f} obj={losses['obj'].item():.4f} "
            f"inst={losses['inst'].item():.4f} overlap={losses['overlap'].item():.4f} "
            f"spear={losses['spear'].item():.4f} anchor={losses['anchor_hit_rate'].item():.4f} "
            f"db_active={float(db_active):.3f} gt_keep={float(db_keep):.3f} absent_drop={float(db_drop):.3f}"
        )

    return _mean_dict(running)


@torch.no_grad()
def validate_joint(model, val_dataloader, criterion):
    model.eval()
    device = next(model.parameters()).device

    running = []
    pbar = tqdm(val_dataloader)
    for batch in pbar:
        batch = _move_joint_batch_to_device(batch, device)
        losses = criterion(batch)
        running.append(_detach_metric_dict(losses))
        db_active = losses.get("dustbin_active_ratio", torch.tensor(0.0, device=device))
        db_keep = losses.get("dustbin_gt_present_keep_rate", torch.tensor(0.0, device=device))
        db_drop = losses.get("dustbin_gt_absent_drop_rate", torch.tensor(0.0, device=device))
        pbar.set_description(
            f"val total={losses['total'].item():.4f} obj={losses['obj'].item():.4f} "
            f"inst={losses['inst'].item():.4f} overlap={losses['overlap'].item():.4f} "
            f"spear={losses['spear'].item():.4f} anchor={losses['anchor_hit_rate'].item():.4f} "
            f"db_active={float(db_active):.3f} gt_keep={float(db_keep):.3f} absent_drop={float(db_drop):.3f}"
        )

    return _mean_dict(running)

def do_train_joint(
    model,
    train_dataset,
    val_dataset,
    train_cfg,
    seed: int = 123,
    optimizer_name: str = "Adam",
    weight_decay: float = 0.05,
    scheduler_name: str = 'linear',
    warmup: int = 0,
):
    set_seed(seed)

    lr = train_cfg['lr']
    num_epochs = train_cfg['num_epochs']
    batch_size = train_cfg['batch_size']
    shuffle = train_cfg.get('shuffle', True)

    obj_ltype = train_cfg.get('obj_ltype', train_cfg.get('ltype', 'infonce'))
    obj_margin = train_cfg.get('margin', 0.2)
    obj_max_violation = train_cfg.get('max_violation', True)

    lambda_obj = train_cfg.get('lambda_obj', 1.0)
    lambda_inst = train_cfg.get('lambda_inst', 0.2)
    lambda_overlap = train_cfg.get('lambda_overlap', 0.05)
    lambda_spear = train_cfg.get('lambda_spear', 0.0)
    patch_temperature = train_cfg.get('patch_temperature', 0.07)
    em_iters = int(train_cfg.get('em_iters', 3))
    present_only_anchor = train_cfg.get('present_only_anchor', False)

    use_dustbin_gate = bool(train_cfg.get('use_dustbin_gate', False))
    dustbin_topk = int(train_cfg.get('dustbin_topk', 8))
    dustbin_tau = float(train_cfg.get('dustbin_tau', 0.04))
    dustbin_min_active_parts = int(train_cfg.get('dustbin_min_active_parts', 1))

    print(
        "[joint config] "
        f"lambda_obj={lambda_obj}, "
        f"lambda_inst={lambda_inst}, "
        f"lambda_overlap={lambda_overlap}, "
        f"lambda_spear={lambda_spear}, "
        f"em_iters={em_iters}, "
        f"present_only_anchor={present_only_anchor}, "
        f"use_dustbin_gate={use_dustbin_gate}, "
        f"dustbin_topk={dustbin_topk}, "
        f"dustbin_tau={dustbin_tau}, "
        f"dustbin_min_active_parts={dustbin_min_active_parts}, "
        f"min_obj_area_ratio={getattr(train_dataset, 'min_obj_area_ratio', 0.0)}"
    )
    if present_only_anchor:
        print(
            "[oracle] present_only_anchor=True: only GT-present parts are used "
            "for Stage2 anchor / pseudo part losses."
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=joint_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=joint_collate_fn,
    )

    criterion = JointObjPartLoss(
        model,
        obj_ltype=obj_ltype,
        obj_margin=obj_margin,
        obj_max_violation=obj_max_violation,
        lambda_obj=lambda_obj,
        lambda_inst=lambda_inst,
        lambda_overlap=lambda_overlap,
        lambda_spear=lambda_spear,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
        use_dustbin_gate=use_dustbin_gate,
        dustbin_topk=dustbin_topk,
        dustbin_tau=dustbin_tau,
        dustbin_min_active_parts=dustbin_min_active_parts,
    )
    criterion.present_only_anchor = present_only_anchor

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented")

    total_steps = len(train_dataloader) * num_epochs
    if scheduler_name == 'linear' and warmup == 0:
        scheduler = None
    elif scheduler_name == 'linear' and warmup > 0:
        scheduler = const_lr(optimizer, lr, warmup, total_steps)
    elif scheduler_name == 'cosine':
        scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    else:
        scheduler = None

    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} / {num_epochs - 1}")
        train_metrics = train_joint(model, train_dataloader, criterion, optimizer, scheduler=scheduler, epoch=epoch)
        val_metrics = validate_joint(model, val_dataloader, criterion)

        val_metrics = {
            **val_metrics,
        }

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch}: "
            f"train_total={train_metrics['total']:.4f}, val_total={val_metrics['total']:.4f}, "
            f"anchor_hit_rate={val_metrics.get('anchor_hit_rate', 0.0):.4f}, "
            f"db_active={val_metrics.get('dustbin_active_ratio', 0.0):.4f}, "
            f"gt_present_keep={val_metrics.get('dustbin_gt_present_keep_rate', 0.0):.4f}, "
            f"gt_absent_drop={val_metrics.get('dustbin_gt_absent_drop_rate', 0.0):.4f}, "
            f"fallback={val_metrics.get('dustbin_fallback_images', 0.0):.0f}"
        )
    
    return model, train_history, val_history
