from __future__ import annotations

import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_joint import joint_collate_fn
from src.loss_gw import (
    Stage3GWLoss,
    build_class_part_blocks_from_dataset,
    build_stage2_visual_prototypes,
)
from src.loss_joint import JointObjPartLoss


# -----------------------------------------------------------------------------
# Small training utilities
# -----------------------------------------------------------------------------


def set_seed(seed):
    print(f"Setting seed {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def to_float(v) -> float:
    if torch.is_tensor(v):
        if v.numel() == 0:
            return float("nan")
        return float(v.detach().float().cpu().reshape(-1)[0].item())
    try:
        return float(v)
    except Exception:
        return float("nan")


def mean_metrics(records: List[Dict]) -> Dict[str, float]:
    if not records:
        return {}

    keys = sorted({k for r in records for k in r.keys()})
    out: Dict[str, float] = {}

    # For anchor hit, use micro aggregation rather than mean of batch rates.
    if "anchor_total_valid_parts" in keys and "anchor_total_hits" in keys:
        total = sum(to_float(r.get("anchor_total_valid_parts", 0.0)) for r in records)
        hits = sum(to_float(r.get("anchor_total_hits", 0.0)) for r in records)
        out["anchor_total_valid_parts"] = total
        out["anchor_total_hits"] = hits
        out["anchor_hit_rate"] = 0.0 if total <= 0 else hits / total

    for k in keys:
        if k in {"anchor_hit_rate", "anchor_total_valid_parts", "anchor_total_hits"}:
            continue
        vals = [to_float(r[k]) for r in records if k in r]
        vals = [v for v in vals if np.isfinite(v)]
        out[k] = float(np.mean(vals)) if vals else float("nan")

    return out


def iter_dataset_samples(dataset):
    if not hasattr(dataset, "data"):
        raise AttributeError("Expected dataset to have .data")
    data = dataset.data
    return data.values() if isinstance(data, dict) else data


def infer_num_parts(train_dataset) -> int:
    max_pid = -1
    for sample in iter_dataset_samples(train_dataset):
        pids = sample["part_category_id"]
        if torch.is_tensor(pids):
            if pids.numel() > 0:
                max_pid = max(max_pid, int(pids.max().item()))
        elif len(pids) > 0:
            max_pid = max(max_pid, int(max(pids)))

    if max_pid < 0:
        raise RuntimeError("Could not infer num_parts from train_dataset.data")
    return max_pid + 1


# -----------------------------------------------------------------------------
# Direct anchor audit: reuse JointObjPartLoss, do not reimplement anchor hit.
# -----------------------------------------------------------------------------


@torch.no_grad()
def eval_anchor_hit_rate_direct(
    model,
    dataloader,
    obj_ltype: str = "infonce",
    obj_margin: float = 0.2,
    obj_max_violation: bool = True,
    patch_temperature: float = 0.07,
    em_iters: int = 1,
    desc: str = "anchor-audit",
) -> Dict[str, float]:
    """
    Eval-only anchor hit rate.

    This directly reuses JointObjPartLoss.forward(), whose anchor_hit_rate is
    computed by the existing _anchor_proto_em_pool() routine. No extra anchor
    matching/statistics logic is duplicated here.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    anchor_loss = JointObjPartLoss(
        sim_model=model,
        obj_ltype=obj_ltype,
        obj_margin=obj_margin,
        obj_max_violation=obj_max_violation,
        lambda_obj=0.0,
        lambda_inst=0.0,
        lambda_overlap=0.0,
        lambda_spear=0.0,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
    ).to(device)
    anchor_loss.eval()

    total_hits = 0.0
    total_valid = 0.0

    pbar = tqdm(dataloader, desc=desc)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        out = anchor_loss(batch)

        hits = to_float(out.get("anchor_total_hits", 0.0))
        valid = to_float(out.get("anchor_total_valid_parts", 0.0))

        total_hits += hits
        total_valid += valid
        hit_rate = 0.0 if total_valid <= 0 else total_hits / total_valid

        pbar.set_description(
            f"{desc} anchor={hit_rate:.6f} "
            f"hits={int(total_hits)} total={int(total_valid)}"
        )

    hit_rate = 0.0 if total_valid <= 0 else total_hits / total_valid
    print(
        f"[{desc}] anchor_hit_rate={hit_rate:.6f} "
        f"hits={int(total_hits)} total={int(total_valid)}"
    )

    model.train(mode=was_training)

    return {
        "anchor_hit_rate": float(hit_rate),
        "anchor_total_hits": float(total_hits),
        "anchor_total_valid_parts": float(total_valid),
    }


# -----------------------------------------------------------------------------
# Epoch loops
# -----------------------------------------------------------------------------


def run_global_stage3_epoch(
    model,
    criterion: Stage3GWLoss,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    steps_per_epoch: int = 1,
    audit_structure_every: int = 1,
    train: bool = True,
):
    model.train(mode=train)
    records = []
    steps_per_epoch = max(1, int(steps_per_epoch))
    prev_iter = epoch * steps_per_epoch

    pbar = tqdm(range(steps_per_epoch), desc="train-gw-global" if train else "val-gw-global")
    for step in pbar:
        if train and scheduler is not None:
            scheduler(step + prev_iter)

        do_structure_audit = audit_structure_every > 0 and (step % audit_structure_every == 0)

        with torch.set_grad_enabled(train):
            losses = criterion(
                batch=None,
                do_anchor_audit=False,
                do_structure_audit=do_structure_audit,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()

        records.append(losses)

        desc = (
            f"{'train' if train else 'val'} "
            f"total={losses['total'].item():.4f} "
            f"gw={losses['gw'].item():.4f} "
            f"struct={losses['struct'].item():.6f}"
        )
        if do_structure_audit and "audit_spear_zt_vs_z0" in losses:
            desc += (
                f" sp_zt_z0={losses['audit_spear_zt_vs_z0'].item():.3f}"
                f" sp_zt_vp={losses['audit_spear_zt_vs_visual_perm'].item():.3f}"
                f" sr_zt_z0={losses['audit_strret_zt_vs_z0'].item():.3f}"
                f" sr_zt_vp={losses['audit_strret_zt_vs_visual_perm'].item():.3f}"
            )
        pbar.set_description(desc)

    return mean_metrics(records)


def run_batch_stage3_epoch(
    model,
    dataloader,
    criterion: Stage3GWLoss,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    audit_anchor_every: int = 0,
    audit_structure_every: int = 1,
    train: bool = True,
):
    model.train(mode=train)
    device = next(model.parameters()).device
    records = []
    prev_iter = epoch * len(dataloader)

    pbar = tqdm(dataloader, desc="train-gw-batch" if train else "val-gw-batch")
    for n_batch, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device)

        if train and scheduler is not None:
            scheduler(n_batch + prev_iter)

        do_anchor_audit = audit_anchor_every > 0 and (n_batch % audit_anchor_every == 0)
        do_structure_audit = audit_structure_every > 0 and (n_batch % audit_structure_every == 0)

        with torch.set_grad_enabled(train):
            losses = criterion(
                batch=batch,
                do_anchor_audit=do_anchor_audit,
                do_structure_audit=do_structure_audit,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()

        records.append(losses)

        desc = (
            f"{'train' if train else 'val'} "
            f"total={losses['total'].item():.4f} "
            f"obj={losses['obj'].item():.6f} "
            f"gw={losses['gw'].item():.4f} "
            f"struct={losses['struct'].item():.6f}"
        )
        if do_anchor_audit and "anchor_hit_rate" in losses:
            desc += f" anchor={losses['anchor_hit_rate'].item():.4f}"
        if do_structure_audit and "audit_spear_zt_vs_z0" in losses:
            desc += (
                f" sp_zt_z0={losses['audit_spear_zt_vs_z0'].item():.3f}"
                f" sp_zt_vp={losses['audit_spear_zt_vs_visual_perm'].item():.3f}"
            )
        pbar.set_description(desc)

    return mean_metrics(records)


# -----------------------------------------------------------------------------
# Main training entry
# -----------------------------------------------------------------------------


def do_train_gw(
    model,
    train_dataset,
    val_dataset,
    train_cfg,
    seed: int = 123,
    optimizer_name: str = "AdamW",
    weight_decay: float = 0.05,
    scheduler_name: str = "linear",
    warmup: int = 0,
):
    device = next(model.parameters()).device
    set_seed(seed)

    lr = float(train_cfg["lr"])
    num_epochs = int(train_cfg["num_epochs"])
    batch_size = int(train_cfg["batch_size"])
    shuffle = bool(train_cfg.get("shuffle", True))
    num_workers = int(train_cfg.get("num_workers", 8))

    obj_ltype = train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce"))
    obj_margin = float(train_cfg.get("margin", 0.2))
    obj_max_violation = bool(train_cfg.get("max_violation", True))

    lambda_obj = float(train_cfg.get("lambda_obj", 0.0))
    lambda_gw = float(train_cfg.get("lambda_gw", 1.0))
    lambda_struct = float(train_cfg.get("lambda_struct", 50.0))

    patch_temperature = float(train_cfg.get("patch_temperature", 0.07))
    em_iters = int(train_cfg.get("em_iters", 1))

    gw_max_iter = int(train_cfg.get("gw_max_iter", 20))
    gw_restarts = int(train_cfg.get("gw_restarts", 50))
    min_proto_count = int(train_cfg.get("min_proto_count", 1))

    num_parts_cfg = train_cfg.get("num_parts", None)
    num_parts = infer_num_parts(train_dataset) if num_parts_cfg is None else int(num_parts_cfg)

    stage2_visual_source = str(train_cfg.get("stage2_visual_source", "anchor")).lower()
    audit_anchor_every = int(train_cfg.get("audit_anchor_every", 0))
    audit_structure_every = int(train_cfg.get("audit_structure_every", 1))
    gw_only_steps_per_epoch = int(train_cfg.get("gw_only_steps_per_epoch", 1))

    # Direct before/after anchor audit can be disabled from yaml if needed.
    audit_anchor_before_after = bool(train_cfg.get("audit_anchor_before_after", True))

    gw_only_mode = lambda_obj <= 0.0

    print(
        "[Stage3 config] "
        f"lambda_obj={lambda_obj}, lambda_gw={lambda_gw}, lambda_struct={lambda_struct}, "
        f"gw_max_iter={gw_max_iter}, gw_restarts={gw_restarts}, "
        f"visual_source={stage2_visual_source}, gw_only_mode={gw_only_mode}, "
        f"steps_per_epoch={gw_only_steps_per_epoch}, min_proto_count={min_proto_count}, "
        f"audit_anchor_before_after={audit_anchor_before_after}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=joint_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=joint_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    print(f"[Stage3] Building visual prototypes: num_parts={num_parts}, source={stage2_visual_source}")
    proto_pack = build_stage2_visual_prototypes(
        model=model,
        dataloader=train_loader,
        num_parts=num_parts,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
        visual_source=stage2_visual_source,
    )

    class_blocks = build_class_part_blocks_from_dataset(train_dataset, device=device)

    criterion = Stage3GWLoss(
        sim_model=model,
        visual_proto=proto_pack["visual_proto"],
        class_blocks=class_blocks,
        obj_ltype=obj_ltype,
        obj_margin=obj_margin,
        obj_max_violation=obj_max_violation,
        lambda_obj=lambda_obj,
        lambda_gw=lambda_gw,
        lambda_struct=lambda_struct,
        gw_max_iter=gw_max_iter,
        gw_restarts=gw_restarts,
        min_proto_count=min_proto_count,
        proto_count=proto_pack["proto_count"],
        patch_temperature=patch_temperature,
        em_iters=em_iters,
    )

    anchor_before = None
    if audit_anchor_before_after:
        print("[Stage3-GW] Anchor hit rate BEFORE GW training:")
        anchor_before = eval_anchor_hit_rate_direct(
            model=model,
            dataloader=val_loader,
            obj_ltype=obj_ltype,
            obj_margin=obj_margin,
            obj_max_violation=obj_max_violation,
            patch_temperature=patch_temperature,
            em_iters=em_iters,
            desc="anchor-before-gw",
        )

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented")

    if gw_only_mode:
        total_steps = max(1, gw_only_steps_per_epoch) * num_epochs
    else:
        total_steps = max(1, len(train_loader)) * num_epochs

    if scheduler_name == "linear" and warmup > 0:
        scheduler = const_lr(optimizer, lr, warmup, total_steps)
    elif scheduler_name == "cosine":
        scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    else:
        scheduler = None

    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} / {num_epochs - 1}")

        if gw_only_mode:
            train_metrics = run_global_stage3_epoch(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                steps_per_epoch=gw_only_steps_per_epoch,
                audit_structure_every=audit_structure_every,
                train=True,
            )
            val_metrics = run_global_stage3_epoch(
                model=model,
                criterion=criterion,
                optimizer=None,
                scheduler=None,
                epoch=epoch,
                steps_per_epoch=1,
                audit_structure_every=audit_structure_every,
                train=False,
            )
        else:
            train_metrics = run_batch_stage3_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                audit_anchor_every=audit_anchor_every,
                audit_structure_every=audit_structure_every,
                train=True,
            )
            val_metrics = run_batch_stage3_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                optimizer=None,
                scheduler=None,
                epoch=epoch,
                audit_anchor_every=audit_anchor_every,
                audit_structure_every=audit_structure_every,
                train=False,
            )

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch}: "
            f"train_total={train_metrics.get('total', float('nan')):.4f}, "
            f"train_gw={train_metrics.get('gw', float('nan')):.4f}, "
            f"train_struct={train_metrics.get('struct', float('nan')):.6f}, "
            f"val_total={val_metrics.get('total', float('nan')):.4f}, "
            f"val_gw={val_metrics.get('gw', float('nan')):.4f}, "
            f"val_struct={val_metrics.get('struct', float('nan')):.6f}"
        )

    if audit_anchor_before_after:
        print("[Stage3-GW] Anchor hit rate AFTER GW training:")
        anchor_after = eval_anchor_hit_rate_direct(
            model=model,
            dataloader=val_loader,
            obj_ltype=obj_ltype,
            obj_margin=obj_margin,
            obj_max_violation=obj_max_violation,
            patch_temperature=patch_temperature,
            em_iters=em_iters,
            desc="anchor-after-gw",
        )

        before_rate = anchor_before["anchor_hit_rate"] if anchor_before is not None else float("nan")
        after_rate = anchor_after["anchor_hit_rate"]
        delta = after_rate - before_rate

        print(
            "[Stage3-GW] Anchor hit delta: "
            f"{before_rate:.6f} -> {after_rate:.6f} "
            f"delta={delta:+.6f}"
        )

    return model, train_history, val_history
