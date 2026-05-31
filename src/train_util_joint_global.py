from copy import deepcopy
from typing import Dict, List, Optional

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_joint import joint_collate_fn
from src.loss_joint import JointObjPartLoss
from src.train_util_joint import (
    set_seed,
    const_lr,
    cosine_lr,
    _move_joint_batch_to_device,
    _mean_dict,
    validate_joint,
    evaluate_object_miou_subprocess,
)


# -----------------------------------------------------------------------------
# Minimal epoch-level part-update pipeline.
#
# This file intentionally reuses the existing dataset, collate function,
# JointObjPartLoss, _anchor_proto_em_pool, instance/overlap/spear losses, and
# mIoU evaluation utilities. The only change is the training schedule:
#   - object loss: batch-wise optimizer.step(), same as before
#   - part loss: one optimizer.step() at the end of the epoch
#
# epoch_part_mode:
#   local_stack:
#       each image/batch computes pseudo labels exactly as before; gradients of
#       original part losses are accumulated over the whole epoch, then one step.
#   global_obj_bank:
#       all object-mask patch tokens of the same object category are concatenated
#       into a class-level patch bank; a fake batch is built for each object class;
#       original part losses are computed on these object-level banks, accumulated,
#       then one step.
# -----------------------------------------------------------------------------


def _zero_like_from_model(model):
    return next(model.parameters()).new_tensor(0.0)


def compute_object_only_losses(model, criterion: JointObjPartLoss, batch: Dict) -> Dict:
    """Compute only the original object loss, with original lambda_obj."""
    obj_feat = batch["obj_feat"]
    obj_text_feat = batch["obj_text_feat"]

    obj_loss = criterion.obj_criterion(
        obj_feat,
        obj_text_feat,
        return_similarity_mat=False,
        self_attn_maps=None,
        cls=None,
        text_input_mask=None,
        text_argmax=None,
    )

    zero = obj_loss.new_tensor(0.0)
    total = criterion.lambda_obj * obj_loss

    return {
        "total": total,
        "obj": obj_loss.detach(),
        "inst": zero.detach(),
        "overlap": zero.detach(),
        "spear": zero.detach(),
        "anchor_hit_rate": zero.detach(),
        "anchor_total_valid_parts": zero.detach(),
        "anchor_total_hits": zero.detach(),
    }


def compute_part_only_losses(
    criterion: JointObjPartLoss,
    batch: Dict,
    use_present_only_anchor: Optional[bool] = None,
) -> Dict:
    """
    Compute ONLY the original part losses from JointObjPartLoss.forward().

    This function deliberately keeps the original part pipeline:
      part text -> projector -> abs_logits -> _anchor_proto_em_pool()
      -> inst / overlap / spear.

    The object loss is not included here.
    """
    patch_tokens = batch["patch_tokens"]
    obj_text_feat = batch["obj_text_feat"]
    part_text_feat = batch["part_text_feat"]
    obj_mask_patch = batch["obj_mask_patch"].bool()
    part_valid_mask = batch["part_valid_mask"].bool()
    part_gt_mask_patch = batch["part_gt_mask_patch"].bool()

    if use_present_only_anchor is None:
        use_present_only_anchor = bool(
            getattr(criterion, "present_only_anchor", False)
            or getattr(criterion, "oracle_present_only_anchor", False)
        )

    if use_present_only_anchor:
        part_present_mask = (part_gt_mask_patch & obj_mask_patch[:, None, :]).sum(dim=-1) > 0
        part_anchor_mask = part_valid_mask & part_present_mask
    else:
        part_anchor_mask = part_valid_mask

    zero = _zero_like_from_model(criterion.sim_model)

    if part_text_feat.shape[1] == 0 or not part_anchor_mask.any():
        return {
            "total": zero,
            "obj": zero.detach(),
            "inst": zero.detach(),
            "overlap": zero.detach(),
            "spear": zero.detach(),
            "anchor_hit_rate": zero.detach(),
            "anchor_total_valid_parts": zero.detach(),
            "anchor_total_hits": zero.detach(),
        }

    # Same projection and normalization as original JointObjPartLoss.forward().
    part_proj = criterion.sim_model.project_clip_txt(part_text_feat.float())   # [B, K, D]
    obj_proj = criterion.sim_model.project_clip_txt(obj_text_feat.float())     # [B, D]
    part_proj = criterion._safe_normalize(part_proj, dim=-1)
    obj_proj = criterion._safe_normalize(obj_proj, dim=-1)
    patch_tokens = criterion._safe_normalize(patch_tokens.float(), dim=-1)

    # Same absolute part-patch score map as original.
    abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens) / criterion.patch_temperature
    abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

    z_part, proto_part, anchor_metrics = criterion._anchor_proto_em_pool(
        patch_tokens=patch_tokens,
        abs_logits=abs_logits,
        obj_mask_patch=obj_mask_patch,
        part_valid_mask=part_anchor_mask,
        part_gt_mask_patch=part_gt_mask_patch,
        num_iters=criterion.em_iters,
    )

    inst_loss = criterion._instance_consistency_loss(part_proj, z_part, part_anchor_mask)

    overlap_loss = (
        criterion._soft_part_overlap_loss(
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_anchor_mask,
        )
        if criterion.lambda_overlap > 0
        else zero
    )

    spear_loss = (
        criterion._combined_structure_spearman_surrogate_loss(
            obj_text_feat=obj_text_feat,
            part_text_feat=part_text_feat,
            obj_proj=obj_proj,
            part_proj=part_proj,
            part_valid_mask=part_anchor_mask,
        )
        if criterion.lambda_spear > 0
        else zero
    )

    total = (
        criterion.lambda_inst * inst_loss
        + criterion.lambda_overlap * overlap_loss
        + criterion.lambda_spear * spear_loss
    )

    return {
        "total": total,
        "obj": zero.detach(),
        "inst": inst_loss.detach(),
        "overlap": overlap_loss.detach(),
        "spear": spear_loss.detach(),
        "anchor_hit_rate": anchor_metrics["anchor_hit_rate"].detach(),
        "anchor_total_valid_parts": anchor_metrics["anchor_total_valid_parts"].detach(),
        "anchor_total_hits": anchor_metrics["anchor_total_hits"].detach(),
    }


def train_object_batchwise(model, train_dataloader, criterion, optimizer, scheduler=None, epoch=0):
    """Original object branch, still batch-wise optimizer.step()."""
    model.train()
    device = next(model.parameters()).device
    prev_iter = epoch * len(train_dataloader)

    running = []
    pbar = tqdm(train_dataloader, desc="obj-train")
    for n_batch, batch in enumerate(pbar):
        batch = _move_joint_batch_to_device(batch, device)

        if scheduler is not None:
            scheduler(n_batch + prev_iter)

        losses = compute_object_only_losses(model, criterion, batch)
        total_loss = losses["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running.append(losses)
        pbar.set_description(
            f"obj train total={losses['total'].item():.4f} obj={losses['obj'].item():.4f}"
        )

    return _mean_dict(running)


def train_part_local_stack_epoch(model, train_dataloader, criterion, optimizer):
    """
    Pipeline 2: local_stack.

    Each batch/image computes pseudo labels exactly as before. The original part
    losses are backpropagated with gradient accumulation over all train batches,
    then one optimizer.step() is performed at the epoch end.
    """
    model.train()
    device = next(model.parameters()).device

    optimizer.zero_grad()
    running = []
    num_batches = max(len(train_dataloader), 1)

    pbar = tqdm(train_dataloader, desc="part-local-stack")
    for batch in pbar:
        batch = _move_joint_batch_to_device(batch, device)
        losses = compute_part_only_losses(criterion, batch)

        # Mean over epoch, equivalent to optimizing average part loss once.
        if losses["total"].requires_grad:
            (losses["total"] / float(num_batches)).backward()

        running.append(losses)
        pbar.set_description(
            f"part local total={losses['total'].item():.4f} "
            f"inst={losses['inst'].item():.4f} overlap={losses['overlap'].item():.4f} "
            f"spear={losses['spear'].item():.4f} anchor={losses['anchor_hit_rate'].item():.4f}"
        )

    optimizer.step()
    optimizer.zero_grad()
    return _mean_dict(running)


@torch.no_grad()
def _collect_object_patch_banks(train_dataloader, device, max_patches_per_obj: int = 100000):
    """
    Collect object-mask patch tokens per object category from existing batches.
    This does not change dataset logic.
    """
    obj_patch_bank = {}
    obj_part_bank = {}
    obj_text_bank = {}

    for batch in tqdm(train_dataloader, desc="collect-global-obj-bank"):
        batch = _move_joint_batch_to_device(batch, device)

        patch_tokens = batch["patch_tokens"].float()            # [B, N, D]
        obj_mask = batch["obj_mask_patch"].bool()               # [B, N]
        category_id = batch["category_id"].long()               # [B]
        obj_text_feat = batch["obj_text_feat"].float()          # [B, Dt]
        part_text_feat = batch["part_text_feat"].float()        # [B, K, Dt]
        part_category_id = batch["part_category_id"].long()     # [B, K]
        part_valid_mask = batch["part_valid_mask"].bool()       # [B, K]

        B = patch_tokens.shape[0]
        for b in range(B):
            cid = int(category_id[b].item())
            x = patch_tokens[b][obj_mask[b]].detach()
            if x.numel() == 0:
                continue

            # Store on CPU to reduce GPU memory while collecting.
            obj_patch_bank.setdefault(cid, []).append(x.cpu())

            # For this object category, all samples should share the same all-part bank.
            if cid not in obj_part_bank:
                keep = part_valid_mask[b]
                obj_part_bank[cid] = {
                    "part_text_feat": part_text_feat[b][keep].detach().cpu(),
                    "part_category_id": part_category_id[b][keep].detach().cpu(),
                }
                obj_text_bank[cid] = obj_text_feat[b].detach().cpu()

    fake_batches = []
    for cid in sorted(obj_patch_bank.keys()):
        if cid not in obj_part_bank:
            continue

        X = torch.cat(obj_patch_bank[cid], dim=0)
        if max_patches_per_obj is not None and max_patches_per_obj > 0 and X.shape[0] > max_patches_per_obj:
            perm = torch.randperm(X.shape[0])[:max_patches_per_obj]
            X = X[perm]

        part_text = obj_part_bank[cid]["part_text_feat"]
        part_ids = obj_part_bank[cid]["part_category_id"]
        obj_text = obj_text_bank[cid]

        K = int(part_ids.numel())
        M = int(X.shape[0])
        if K == 0 or M == 0:
            continue

        fake_batch = {
            # compute_part_only_losses only needs these fields.
            "patch_tokens": X.unsqueeze(0).to(device),
            "obj_mask_patch": torch.ones((1, M), dtype=torch.bool, device=device),
            "part_gt_mask_patch": torch.zeros((1, K, M), dtype=torch.bool, device=device),
            "obj_text_feat": obj_text.unsqueeze(0).to(device),
            "part_text_feat": part_text.unsqueeze(0).to(device),
            "part_category_id": part_ids.unsqueeze(0).to(device),
            "part_valid_mask": torch.ones((1, K), dtype=torch.bool, device=device),
            "category_id": torch.tensor([cid], dtype=torch.long, device=device),
        }
        fake_batches.append(fake_batch)

    return fake_batches


def train_part_global_obj_bank_epoch(
    model,
    train_dataloader,
    criterion,
    optimizer,
    max_patches_per_obj: int = 100000,
):
    """
    Pipeline 1: global_obj_bank.

    For each object category, concatenate all object-mask patch tokens into one
    large fake sample, then call the original part-loss pipeline on it.
    Gradients from all object categories are accumulated, then one optimizer.step().
    """
    device = next(model.parameters()).device
    model.train()

    fake_batches = _collect_object_patch_banks(
        train_dataloader=train_dataloader,
        device=device,
        max_patches_per_obj=max_patches_per_obj,
    )

    optimizer.zero_grad()
    running = []
    denom = max(len(fake_batches), 1)

    pbar = tqdm(fake_batches, desc="part-global-obj-bank")
    for fake_batch in pbar:
        # present_only_anchor must be disabled here because fake global banks use
        # dummy GT masks. The anchor method/losses remain original otherwise.
        losses = compute_part_only_losses(
            criterion,
            fake_batch,
            use_present_only_anchor=False,
        )

        if losses["total"].requires_grad:
            (losses["total"] / float(denom)).backward()

        running.append(losses)
        pbar.set_description(
            f"part global total={losses['total'].item():.4f} "
            f"inst={losses['inst'].item():.4f} overlap={losses['overlap'].item():.4f} "
            f"spear={losses['spear'].item():.4f}"
        )

    if len(fake_batches) > 0:
        optimizer.step()
    optimizer.zero_grad()

    if len(running) == 0:
        z = _zero_like_from_model(model)
        return {
            "total": 0.0,
            "obj": 0.0,
            "inst": 0.0,
            "overlap": 0.0,
            "spear": 0.0,
            "anchor_hit_rate": 0.0,
            "anchor_total_valid_parts": 0.0,
            "anchor_total_hits": 0.0,
        }
    return _mean_dict(running)


def do_train_joint_epoch_part(
    model,
    train_dataset,
    val_dataset,
    train_cfg,
    seed: int = 123,
    optimizer_name: str = "Adam",
    weight_decay: float = 0.05,
    scheduler_name: str = 'linear',
    warmup: int = 0,
    eval_proj_class: str = "",
    eval_proj_name: str = "",
    miou_eval_script: Optional[str] = None,
    miou_eval_cfg: Optional[str] = None,
    miou_eval_base_cfg: Optional[str] = None,
    miou_result_dir: str = "segmentation_results",
    miou_result_json_name: Optional[str] = None,
    miou_bench_key: Optional[str] = None,
    miou_extra_opts: Optional[List[str]] = None,
    miou_eval_port: int = 29517,
):
    device = next(model.parameters()).device
    set_seed(seed)

    lr = train_cfg['lr']
    num_epochs = train_cfg['num_epochs']
    batch_size = train_cfg['batch_size']
    shuffle = train_cfg.get('shuffle', True)
    save_best_model = train_cfg.get('save_best_model', True)

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
    em_iters = int(train_cfg.get('em_iters', 3))
    present_only_anchor = bool(
        train_cfg.get('present_only_anchor', train_cfg.get('oracle_present_only_anchor', False))
    )

    epoch_part_mode = str(train_cfg.get('epoch_part_mode', 'local_stack'))
    max_patches_per_obj = int(train_cfg.get('max_patches_per_obj', 100000))

    if epoch_part_mode not in {"local_stack", "global_obj_bank"}:
        raise ValueError(f"Unknown epoch_part_mode={epoch_part_mode}. Use local_stack or global_obj_bank.")

    if not eval_proj_name:
        raise ValueError("eval_proj_name must be provided for mIoU evaluation.")
    if miou_eval_script is None or miou_eval_cfg is None or miou_eval_base_cfg is None:
        raise ValueError("miou_eval_script / miou_eval_cfg / miou_eval_base_cfg must all be provided.")

    print(
        "[joint epoch-part config] "
        f"lambda_obj={lambda_obj}, "
        f"lambda_inst={lambda_inst}, "
        f"lambda_overlap={lambda_overlap}, "
        f"lambda_spear={lambda_spear}, "
        f"em_iters={em_iters}, "
        f"present_only_anchor={present_only_anchor}, "
        f"epoch_part_mode={epoch_part_mode}, "
        f"max_patches_per_obj={max_patches_per_obj}, "
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
        em_iters=em_iters,
    )
    criterion.present_only_anchor = present_only_anchor
    criterion.oracle_present_only_anchor = present_only_anchor

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

    baseline_obj_eval = evaluate_object_miou_subprocess(
        model=model,
        proj_class=eval_proj_class,
        proj_name=eval_proj_name,
        eval_script=miou_eval_script,
        eval_cfg=miou_eval_cfg,
        eval_base_cfg=miou_eval_base_cfg,
        result_dir=miou_result_dir,
        result_json_name=miou_result_json_name,
        bench_key=miou_bench_key,
        extra_opts=miou_extra_opts,
        miou_eval_port=miou_eval_port,
    )
    baseline_obj_miou = baseline_obj_eval["obj_eval_miou"]
    print(f"[baseline object mIoU] miou={baseline_obj_miou:.4f}")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} / {num_epochs - 1}")

        obj_train_metrics = train_object_batchwise(
            model,
            train_dataloader,
            criterion,
            optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )

        if epoch_part_mode == "local_stack":
            part_train_metrics = train_part_local_stack_epoch(
                model,
                train_dataloader,
                criterion,
                optimizer,
            )
        else:
            part_train_metrics = train_part_global_obj_bank_epoch(
                model,
                train_dataloader,
                criterion,
                optimizer,
                max_patches_per_obj=max_patches_per_obj,
            )

        train_metrics = {
            "total": float(obj_train_metrics.get("total", 0.0)) + float(part_train_metrics.get("total", 0.0)),
            "obj": float(obj_train_metrics.get("obj", 0.0)),
            "inst": float(part_train_metrics.get("inst", 0.0)),
            "overlap": float(part_train_metrics.get("overlap", 0.0)),
            "spear": float(part_train_metrics.get("spear", 0.0)),
            "anchor_hit_rate": float(part_train_metrics.get("anchor_hit_rate", 0.0)),
            "anchor_total_valid_parts": float(part_train_metrics.get("anchor_total_valid_parts", 0.0)),
            "anchor_total_hits": float(part_train_metrics.get("anchor_total_hits", 0.0)),
            "obj_total": float(obj_train_metrics.get("total", 0.0)),
            "part_total": float(part_train_metrics.get("total", 0.0)),
            "epoch_part_mode": epoch_part_mode,
        }

        val_metrics = validate_joint(model, val_dataloader, criterion)

        obj_eval_metrics = evaluate_object_miou_subprocess(
            model=model,
            proj_class=eval_proj_class,
            proj_name=eval_proj_name,
            eval_script=miou_eval_script,
            eval_cfg=miou_eval_cfg,
            eval_base_cfg=miou_eval_base_cfg,
            result_dir=miou_result_dir,
            result_json_name=miou_result_json_name,
            bench_key=miou_bench_key,
            extra_opts=miou_extra_opts,
            miou_eval_port=miou_eval_port,
        )
        obj_eval_metrics["obj_eval_miou_delta_vs_baseline"] = float(
            obj_eval_metrics["obj_eval_miou"] - baseline_obj_miou
        )

        val_metrics = {**val_metrics, **obj_eval_metrics}

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch}: "
            f"train_total={train_metrics['total']:.4f}, "
            f"obj_total={train_metrics['obj_total']:.4f}, "
            f"part_total={train_metrics['part_total']:.4f}, "
            f"inst={train_metrics['inst']:.4f}, "
            f"overlap={train_metrics['overlap']:.4f}, "
            f"spear={train_metrics['spear']:.4f}, "
            f"val_total={val_metrics['total']:.4f}, "
            f"obj_eval_miou={val_metrics['obj_eval_miou']:.4f}, "
            f"miou_delta_vs_baseline={val_metrics['obj_eval_miou_delta_vs_baseline']:.4f}, "
            f"anchor_hit_rate={train_metrics.get('anchor_hit_rate', 0.0):.4f}"
        )

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
