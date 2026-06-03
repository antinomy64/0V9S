
from copy import deepcopy
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_global import CategoryPatchPoolDataset, global_pool_collate_fn
from src.loss_global import PartLoss


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


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


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
            v_valid = d["anchor_total_valid_parts"]
            v_hits = d["anchor_total_hits"]
            if torch.is_tensor(v_valid):
                total_valid += float(v_valid.detach().float().cpu().item())
            else:
                total_valid += float(v_valid)
            if torch.is_tensor(v_hits):
                total_hits += float(v_hits.detach().float().cpu().item())
            else:
                total_hits += float(v_hits)

        out["anchor_total_valid_parts"] = total_valid
        out["anchor_total_hits"] = total_hits
        out["anchor_hit_rate"] = 0.0 if total_valid <= 0 else total_hits / total_valid

    for k in keys:
        if k in {"anchor_hit_rate", "anchor_total_valid_parts", "anchor_total_hits"}:
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


def train(model, train_dataloader, criterion, optimizer, scheduler=None, epoch=0):
    model.train()
    device = next(model.parameters()).device
    prev_iter = epoch * len(train_dataloader)

    running = []
    pbar = tqdm(train_dataloader)
    for n_batch, batch in enumerate(pbar):
        batch = _move_batch_to_device(batch, device)

        if scheduler is not None:
            scheduler(n_batch + prev_iter)

        losses = criterion(batch)
        total_loss = losses["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running.append(losses)
        pbar.set_description(
            f"train total={losses['total'].item():.4f} "
            f"inst={losses['inst'].item():.4f} overlap={losses['overlap'].item():.4f} "
            f"spear={losses['spear'].item():.4f} anchor={losses['anchor_hit_rate'].item():.4f}"
        )

    return _mean_dict(running)


@torch.no_grad()
def validate(model, val_dataloader, criterion):
    model.eval()
    device = next(model.parameters()).device

    running = []
    pbar = tqdm(val_dataloader)
    for batch in pbar:
        batch = _move_batch_to_device(batch, device)
        losses = criterion(batch)
        running.append(losses)
        pbar.set_description(
            f"val total={losses['total'].item():.4f} "
            f"inst={losses['inst'].item():.4f} overlap={losses['overlap'].item():.4f} "
            f"spear={losses['spear'].item():.4f} anchor={losses['anchor_hit_rate'].item():.4f}"
        )

    return _mean_dict(running)

def do_train(
    model,
    train_dataset,
    val_dataset,
    train_cfg,
    seed: int = 123,
    optimizer_name: str = "Adam",
    weight_decay: float = 0.05,
    scheduler_name: str = 'linear',
    warmup: int = 0,
    eval_proj_name: str = "",
):
    set_seed(seed)

    lr = train_cfg['lr']
    num_epochs = train_cfg['num_epochs']
    batch_size = train_cfg['batch_size']
    shuffle = train_cfg.get('shuffle', True)

    lambda_inst = train_cfg.get('lambda_inst', 0.2)
    lambda_overlap = train_cfg.get('lambda_overlap', 0.05)
    lambda_spear = train_cfg.get('lambda_spear', 0.0)
    topk_ratio = train_cfg.get('topk_ratio', 0.1)
    patch_temperature = train_cfg.get('patch_temperature', 0.07)
    em_iters = int(train_cfg.get('em_iters', 3))
    present_only_anchor = bool(train_cfg.get('present_only_anchor', False))
    sample_patches_per_step = train_cfg.get("sample_patches_per_step", 65536)
    global_steps_per_epoch = train_cfg.get("global_steps_per_epoch", None)

    if not eval_proj_name:
        raise ValueError("eval_proj_name must be provided for mIoU evaluation.")
    
    print(
        "[config] "
        f"lambda_inst={lambda_inst}, "
        f"lambda_overlap={lambda_overlap}, "
        f"lambda_spear={lambda_spear}, "
        f"em_iters={em_iters}, "
        f"present_only_anchor={present_only_anchor}, "
        f"min_obj_area_ratio={getattr(train_dataset, 'min_obj_area_ratio', 0.0)}"
    )
    if present_only_anchor:
        print(
            "[oracle] present_only_anchor=True: only GT-present parts are used "
            "for Stage2 anchor / pseudo part losses."
        )

    train_pool_dataset = CategoryPatchPoolDataset(
        train_dataset,
        sample_patches_per_step=sample_patches_per_step,
        steps_per_epoch=global_steps_per_epoch,
        store_dtype=torch.float16,
        seed=seed,
    )
    val_pool_dataset = CategoryPatchPoolDataset(
        val_dataset,
        sample_patches_per_step=sample_patches_per_step,
        steps_per_epoch=None,
        store_dtype=torch.float16,
        seed=seed,
    )

    train_dataloader = DataLoader(
        train_pool_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=global_pool_collate_fn,
    )
    val_dataloader = DataLoader(
        val_pool_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=global_pool_collate_fn,
    )

    criterion = PartLoss(
        model,
        lambda_inst=lambda_inst,
        lambda_overlap=lambda_overlap,
        lambda_spear=lambda_spear,
        topk_ratio=topk_ratio,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
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
        train_metrics = train(model, train_dataloader, criterion, optimizer, scheduler=scheduler, epoch=epoch)
        val_metrics = validate(model, val_dataloader, criterion)

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch}: "
            f"train_total={train_metrics['total']:.4f}, val_total={val_metrics['total']:.4f}, "
            f"anchor_hit_rate={val_metrics.get('anchor_hit_rate', 0.0):.4f}"
        )

    return model, train_history, val_history
