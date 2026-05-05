from copy import deepcopy
import os
import random
from typing import Dict, Tuple

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


def _mean_dict(list_of_dicts):
    if len(list_of_dicts) == 0:
        return {}
    keys = list(list_of_dicts[0].keys())
    out = {}
    for k in keys:
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

        running.append(losses)
        pbar.set_description(
            f"train total={losses['total'].item():.4f} obj={losses['obj'].item():.4f} "
            f"inst={losses['inst'].item():.4f} overlap={losses['overlap'].item():.4f} "
            f"spear={losses['spear'].item():.4f}"
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
        running.append(losses)
        pbar.set_description(
            f"val total={losses['total'].item():.4f} obj={losses['obj'].item():.4f} "
            f"inst={losses['inst'].item():.4f} overlap={losses['overlap'].item():.4f} "
            f"spear={losses['spear'].item():.4f}"
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
    device = next(model.parameters()).device
    set_seed(seed)

    lr = train_cfg['lr']
    num_epochs = train_cfg['num_epochs']
    batch_size = train_cfg['batch_size']
    shuffle = train_cfg.get('shuffle', True)
    save_best_model = train_cfg.get('save_best_model', True)

    obj_ltype = train_cfg.get('obj_ltype', train_cfg.get('ltype', 'infonce'))
    obj_margin = train_cfg.get('margin', 0.2)
    obj_max_violation = train_cfg.get('max_violation', True)

    lambda_obj = train_cfg.get('lambda_obj', 1.0)
    lambda_inst = train_cfg.get('lambda_inst', 0.2)
    lambda_overlap = train_cfg.get('lambda_overlap', 0.05)
    lambda_spear = train_cfg.get('lambda_spear', 0.0)
    topk_ratio = train_cfg.get('topk_ratio', 0.1)
    patch_temperature = train_cfg.get('patch_temperature', 0.07)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=joint_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
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
        topk_ratio=topk_ratio,
        patch_temperature=patch_temperature,
    )

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
    best_model = deepcopy(model)
    best_val = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} / {num_epochs - 1}")
        train_metrics = train_joint(model, train_dataloader, criterion, optimizer, scheduler=scheduler, epoch=epoch)
        val_metrics = validate_joint(model, val_dataloader, criterion)

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch}: "
            f"train_total={train_metrics['total']:.4f}, val_total={val_metrics['total']:.4f}, "
            f"train_obj={train_metrics['obj']:.4f}, val_obj={val_metrics['obj']:.4f}, "
            f"train_inst={train_metrics['inst']:.4f}, val_inst={val_metrics['inst']:.4f}, "
            f"train_overlap={train_metrics['overlap']:.4f}, val_overlap={val_metrics['overlap']:.4f}, "
            f"train_spear={train_metrics['spear']:.4f}, val_spear={val_metrics['spear']:.4f}"
        )

        if save_best_model:
            if best_val is None or val_metrics['total'] < best_val:
                best_val = val_metrics['total']
                best_model = deepcopy(model)
                print("Best validation total loss, saving current best model in memory.")

    model = best_model if save_best_model else model
    return model, train_history, val_history
