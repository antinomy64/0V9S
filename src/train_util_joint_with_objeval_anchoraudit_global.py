from copy import deepcopy
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_joint_with_part_anchoraudit_global import joint_collate_fn
from src.loss_joint_hungarian_global import JointObjPartLoss


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


def _compute_relative_scores_fallback(local_scores: torch.Tensor) -> torch.Tensor:
    Kb, Mb = local_scores.shape
    if Kb <= 1:
        return local_scores
    top2_vals, top2_idx = torch.topk(local_scores, k=min(2, Kb), dim=0)
    best_vals = top2_vals[0]
    best_idx = top2_idx[0]
    second_vals = top2_vals[1]
    row_ids = torch.arange(Kb, device=local_scores.device)[:, None]
    is_top1 = row_ids == best_idx[None, :]
    best_other = torch.where(is_top1, second_vals[None, :], best_vals[None, :])
    return local_scores - best_other


@torch.no_grad()
def audit_anchor_hit_rate(criterion: JointObjPartLoss, batch: Dict) -> Dict:
    """
    Training-consistent anchor audit:
      - use current projector
      - use current patch_temperature
      - mask by obj_mask_patch
      - greedily assign one unique anchor patch per valid part
      - check whether each anchor falls inside the corresponding GT part mask
    """
    sim_model = criterion.sim_model
    patch_temperature = criterion.patch_temperature

    part_text_feat = batch["part_text_feat"].float()
    patch_tokens = batch["patch_tokens"].float()
    obj_mask_patch = batch["obj_mask_patch"].bool()
    part_valid_mask = batch["part_valid_mask"].bool()
    part_gt_mask_patch = batch["part_gt_mask_patch"].bool()

    part_proj = sim_model.project_clip_txt(part_text_feat)
    part_proj = F.normalize(part_proj, dim=-1)
    patch_tokens = F.normalize(patch_tokens, dim=-1)

    logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / patch_temperature
    logits = logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

    total_valid_parts = 0
    total_anchor_hits = 0

    B, K, N = logits.shape
    for b in range(B):
        valid_patch_mask = obj_mask_patch[b]
        valid_part_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)
        if valid_part_idx.numel() == 0:
            continue

        local_scores = logits[b][valid_part_idx][:, valid_patch_mask]   # [Kb, Mb]
        Kb, Mb = local_scores.shape
        if Mb == 0:
            continue

        if hasattr(criterion, "_compute_relative_scores"):
            try:
                anchor_scores = criterion._compute_relative_scores(local_scores)
            except Exception:
                anchor_scores = _compute_relative_scores_fallback(local_scores)
        else:
            anchor_scores = _compute_relative_scores_fallback(local_scores)

        flat_scores = anchor_scores.reshape(-1)
        sorted_idx = torch.argsort(flat_scores, descending=True)

        anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=local_scores.device)
        patch_taken = torch.zeros((Mb,), dtype=torch.bool, device=local_scores.device)

        assigned_parts = 0
        for flat_id in sorted_idx:
            p_local = torch.div(flat_id, Mb, rounding_mode='floor')
            n_local = flat_id % Mb
            if anchor_idx_local[p_local] != -1:
                continue
            if patch_taken[n_local]:
                continue
            anchor_idx_local[p_local] = n_local
            patch_taken[n_local] = True
            assigned_parts += 1
            if assigned_parts == Kb:
                break

        unassigned = torch.nonzero(anchor_idx_local < 0, as_tuple=False).squeeze(1)
        if unassigned.numel() > 0:
            local_best = anchor_scores.argmax(dim=1)
            anchor_idx_local[unassigned] = local_best[unassigned]

        valid_patch_idx_global = torch.nonzero(valid_patch_mask, as_tuple=False).squeeze(1)
        anchor_idx_global = valid_patch_idx_global[anchor_idx_local]   # [Kb]

        gt_masks = part_gt_mask_patch[b, valid_part_idx]               # [Kb, N]
        hit_vec = gt_masks[torch.arange(Kb, device=gt_masks.device), anchor_idx_global]  # [Kb]

        total_valid_parts += int(Kb)
        total_anchor_hits += int(hit_vec.long().sum().item())

    hit_rate = 0.0 if total_valid_parts == 0 else total_anchor_hits / float(total_valid_parts)
    return {
        "audit_anchor_hit_rate": float(hit_rate),
        "audit_total_valid_parts": float(total_valid_parts),
        "audit_total_anchor_hits": float(total_anchor_hits),
    }


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


@torch.no_grad()
def audit_joint(model, dataloader, criterion):
    model.eval()
    device = next(model.parameters()).device
    running = []

    pbar = tqdm(dataloader)
    for batch in pbar:
        batch = _move_joint_batch_to_device(batch, device)
        running.append(audit_anchor_hit_rate(criterion, batch))

    return _mean_dict(running)


def _extract_miou_from_result_json(result_json_path: str, bench_key: Optional[str] = None) -> float:
    if not os.path.exists(result_json_path):
        raise FileNotFoundError(f"mIoU result json not found: {result_json_path}")

    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if bench_key is not None:
        if bench_key not in data:
            raise KeyError(f"bench key '{bench_key}' not found in {result_json_path}; keys={list(data.keys())}")
        return float(data[bench_key])

    if len(data) == 1:
        return float(next(iter(data.values())))

    if "avg_miou" in data:
        return float(data["avg_miou"])

    # fallback: first numeric value
    if "voc116_obj" in data:
        return float(data["voc116_obj"])
    if "voc116_part" in data:
        return float(data["voc116_part"])
    for v in data.values():
        try:
            return float(v)
        except Exception:
            pass

    raise RuntimeError(f"Could not extract numeric mIoU from {result_json_path}: {data}")


def evaluate_object_miou_subprocess(
    model,
    proj_name: str,
    eval_script: str,
    eval_cfg: str,
    eval_base_cfg: str,
    result_dir: str = "segmentation_results",
    result_json_name: Optional[str] = None,
    bench_key: Optional[str] = None,
    extra_opts: Optional[List[str]] = None,
):
    """
    Evaluate object mIoU using the SAME evaluation script path as the user-provided main.py flow:
      python -m torch.distributed.run --nproc_per_node=1 <eval_script> --eval --eval_cfg ... --eval_base_cfg ... --opts model.proj_name=<proj_name> ...
    """
    # Save current projector weights to the path expected by the segmentation evaluator.
    os.makedirs("weights", exist_ok=True)
    ckpt_path = os.path.join("weights", f"{proj_name}.pth")
    torch.save(model.state_dict(), ckpt_path)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port=29517",
        eval_script,
        "--eval",
        "--eval_cfg",
        eval_cfg,
        "--eval_base_cfg",
        eval_base_cfg,
        "--opts",
        f"model.proj_name={proj_name}",
    ]
    if extra_opts:
        cmd.extend(extra_opts)

    print("[mIoU eval cmd]", " ".join(cmd))
    proc = subprocess.run(cmd, check=True)

    json_name = result_json_name if result_json_name is not None else proj_name
    result_json_path = os.path.join(result_dir, f"{json_name}.json")
    miou = _extract_miou_from_result_json(result_json_path, bench_key=bench_key)
    return {
        "obj_eval_miou": float(miou),
        "obj_eval_ckpt_path": ckpt_path,
        "obj_eval_result_json": result_json_path,
        "obj_eval_subprocess_returncode": int(proc.returncode),
    }


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
    eval_proj_name: str = "",
    miou_eval_script: Optional[str] = None,
    miou_eval_cfg: Optional[str] = None,
    miou_eval_base_cfg: Optional[str] = None,
    miou_result_dir: str = "segmentation_results",
    miou_result_json_name: Optional[str] = None,
    miou_bench_key: Optional[str] = None,
    miou_extra_opts: Optional[List[str]] = None,
):
    device = next(model.parameters()).device
    set_seed(seed)

    lr = train_cfg['lr']
    num_epochs = train_cfg['num_epochs']
    batch_size = train_cfg['batch_size']
    shuffle = train_cfg.get('shuffle', True)
    save_best_model = train_cfg.get('save_best_model', True)

    # mIoU guardrail
    object_miou_max_drop = float(train_cfg.get('object_miou_max_drop', 0.5))
    select_best_by_miou = bool(train_cfg.get('select_best_by_miou', True))

    obj_ltype = train_cfg.get('obj_ltype', train_cfg.get('ltype', 'infonce'))
    obj_margin = train_cfg.get('margin', 0.2)
    obj_max_violation = train_cfg.get('max_violation', True)

    lambda_obj = train_cfg.get('lambda_obj', 1.0)
    lambda_inst = train_cfg.get('lambda_inst', 0.2)
    lambda_overlap = train_cfg.get('lambda_overlap', 0.05)
    lambda_spear = train_cfg.get('lambda_spear', 0.0)
    topk_ratio = train_cfg.get('topk_ratio', 0.1)
    patch_temperature = train_cfg.get('patch_temperature', 0.07)

    if not eval_proj_name:
        raise ValueError("eval_proj_name must be provided for mIoU evaluation.")
    if miou_eval_script is None or miou_eval_cfg is None or miou_eval_base_cfg is None:
        raise ValueError("miou_eval_script / miou_eval_cfg / miou_eval_base_cfg must all be provided.")

    print(
        "[joint config] "
        f"lambda_obj={lambda_obj}, "
        f"lambda_inst={lambda_inst}, "
        f"lambda_overlap={lambda_overlap}, "
        f"lambda_spear={lambda_spear}, "
        f"min_obj_area_ratio={getattr(train_dataset, 'min_obj_area_ratio', 0.0)}"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        collate_fn=joint_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
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
        class_part_bank=getattr(train_dataset, "class_part_bank", None),
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
    best_obj_miou = None

    # Baseline object mIoU before joint training
    baseline_obj_eval = evaluate_object_miou_subprocess(
        model=model,
        proj_name=eval_proj_name,
        eval_script=miou_eval_script,
        eval_cfg=miou_eval_cfg,
        eval_base_cfg=miou_eval_base_cfg,
        result_dir=miou_result_dir,
        result_json_name=miou_result_json_name,
        bench_key=miou_bench_key,
        extra_opts=miou_extra_opts,
    )
    baseline_obj_miou = baseline_obj_eval["obj_eval_miou"]
    print(f"[baseline object mIoU] miou={baseline_obj_miou:.4f}")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} / {num_epochs - 1}")
        train_metrics = train_joint(model, train_dataloader, criterion, optimizer, scheduler=scheduler, epoch=epoch)
        val_metrics = validate_joint(model, val_dataloader, criterion)

        # Added: per-epoch object mIoU evaluation via same segmentation eval script
        obj_eval_metrics = evaluate_object_miou_subprocess(
            model=model,
            proj_name=eval_proj_name,
            eval_script=miou_eval_script,
            eval_cfg=miou_eval_cfg,
            eval_base_cfg=miou_eval_base_cfg,
            result_dir=miou_result_dir,
            result_json_name=miou_result_json_name,
            bench_key=miou_bench_key,
            extra_opts=miou_extra_opts,
        )
        obj_eval_metrics["obj_eval_miou_delta_vs_baseline"] = float(obj_eval_metrics["obj_eval_miou"] - baseline_obj_miou)

        # Added: part GT anchor audit
        audit_metrics = audit_joint(model, val_dataloader, criterion)

        val_metrics = {
            **val_metrics,
            **obj_eval_metrics,
            **audit_metrics,
        }

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch}: "
            f"train_total={train_metrics['total']:.4f}, val_total={val_metrics['total']:.4f}, "
            f"obj_eval_miou={val_metrics['obj_eval_miou']:.4f}, "
            f"miou_delta_vs_baseline={val_metrics['obj_eval_miou_delta_vs_baseline']:.4f}, "
            f"anchor_hit_rate={val_metrics['audit_anchor_hit_rate']:.4f}"
        )

        # Best model selection with mIoU guardrail
        current_obj_miou = val_metrics["obj_eval_miou"]
        obj_ok = current_obj_miou >= (baseline_obj_miou - object_miou_max_drop)

        if save_best_model:
            if select_best_by_miou:
                if obj_ok and (best_obj_miou is None or current_obj_miou > best_obj_miou):
                    best_obj_miou = current_obj_miou
                    best_val = val_metrics['total']
                    best_model = deepcopy(model)
                    print("Best model updated by object mIoU under guardrail.")
                elif not obj_ok:
                    print(
                        f"Skip best update because object mIoU dropped too much: "
                        f"{current_obj_miou:.4f} < {baseline_obj_miou - object_miou_max_drop:.4f}"
                    )
            else:
                if obj_ok and (best_val is None or val_metrics['total'] < best_val):
                    best_val = val_metrics['total']
                    best_model = deepcopy(model)
                    print("Best validation total loss under object mIoU guardrail, saving current best model in memory.")
                elif not obj_ok:
                    print(
                        f"Skip best update because object mIoU dropped too much: "
                        f"{current_obj_miou:.4f} < {baseline_obj_miou - object_miou_max_drop:.4f}"
                    )

    model = best_model if save_best_model else model
    return model, train_history, val_history
