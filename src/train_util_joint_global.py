from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
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
    evaluate_object_miou_subprocess,
)


def _object_only_losses(batch: Dict, criterion: JointObjPartLoss) -> Dict:
    obj_loss = criterion.obj_criterion(
        batch["obj_feat"],
        batch["obj_text_feat"],
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


def train_joint_object_only(model, train_dataloader, criterion, optimizer, scheduler=None, epoch=0):
    model.train()
    device = next(model.parameters()).device
    prev_iter = epoch * len(train_dataloader)

    running = []
    pbar = tqdm(train_dataloader)
    for n_batch, batch in enumerate(pbar):
        batch = _move_joint_batch_to_device(batch, device)

        if scheduler is not None:
            scheduler(n_batch + prev_iter)

        losses = _object_only_losses(batch, criterion)
        total_loss = losses["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running.append(losses)
        pbar.set_description(f"train-obj total={losses['total'].item():.4f} obj={losses['obj'].item():.4f}")

    return _mean_dict(running)


@torch.no_grad()
def validate_joint_object_only(model, val_dataloader, criterion):
    model.eval()
    device = next(model.parameters()).device

    running = []
    pbar = tqdm(val_dataloader)
    for batch in pbar:
        batch = _move_joint_batch_to_device(batch, device)
        losses = _object_only_losses(batch, criterion)
        running.append(losses)
        pbar.set_description(f"val-obj total={losses['total'].item():.4f} obj={losses['obj'].item():.4f}")

    return _mean_dict(running)


@torch.no_grad()
def collect_part_text_proto(train_dataloader, num_parts: int, device: torch.device):
    text_sum = None
    text_count = torch.zeros(num_parts, device=device)

    for batch in tqdm(train_dataloader, desc="collect part text proto"):
        batch = _move_joint_batch_to_device(batch, device)
        part_text = batch["part_text_feat"].float()
        part_ids = batch["part_category_id"].long()
        part_valid = batch["part_valid_mask"].bool()

        if text_sum is None:
            text_sum = torch.zeros(num_parts, int(part_text.shape[-1]), device=device)

        B, K = part_ids.shape
        for b in range(B):
            valid_k = torch.nonzero(part_valid[b], as_tuple=False).squeeze(1)
            for k in valid_k.tolist():
                pid = int(part_ids[b, k].item())
                if 0 <= pid < num_parts:
                    text_sum[pid] += part_text[b, k]
                    text_count[pid] += 1.0

    if text_sum is None:
        raise RuntimeError("No part text features were collected from train_dataloader.")

    valid_text = text_count > 0
    T_raw = text_sum / text_count.clamp_min(1.0)[:, None]
    return T_raw.detach(), valid_text.detach(), text_count.detach()


@torch.no_grad()
def build_local_pseudo_mean_pool(
    model,
    criterion: JointObjPartLoss,
    train_dataloader,
    num_parts: int,
    dino_dim: int,
    device: torch.device,
    present_only_anchor: bool = False,
    min_pool_count: int = 1,
):
    """Version B: each crop finds pseudo labels locally; epoch end averages prototypes by part id."""
    model.eval()
    criterion.eval()

    pool_sum = torch.zeros(num_parts, dino_dim, device=device)
    pool_count = torch.zeros(num_parts, device=device)
    total_valid_parts = torch.tensor(0.0, device=device)
    total_anchor_hits = torch.tensor(0.0, device=device)

    pbar = tqdm(train_dataloader, desc="build local_pseudo_mean pool")
    for batch in pbar:
        batch = _move_joint_batch_to_device(batch, device)

        part_text = batch["part_text_feat"].float()
        patch_tokens = criterion._safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()
        part_valid = batch["part_valid_mask"].bool()
        part_gt = batch["part_gt_mask_patch"].bool()
        part_ids = batch["part_category_id"].long()

        if present_only_anchor:
            part_present = (part_gt & obj_mask[:, None, :]).sum(dim=-1) > 0
            anchor_part_valid = part_valid & part_present
        else:
            anchor_part_valid = part_valid

        if part_text.shape[1] == 0 or not anchor_part_valid.any():
            continue

        part_proj = model.project_clip_txt(part_text)
        part_proj = criterion._safe_normalize(part_proj.float(), dim=-1)

        abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens)
        abs_logits = abs_logits / float(criterion.patch_temperature)
        abs_logits = abs_logits.masked_fill(~obj_mask[:, None, :], -1e4)

        _, proto_part, anchor_metrics, _, anchor_valid = criterion._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask,
            part_valid_mask=anchor_part_valid,
            part_gt_mask_patch=part_gt,
            num_iters=criterion.em_iters,
            return_anchor_tokens=True,
        )
        proto_part = criterion._safe_normalize(proto_part.float(), dim=-1)
        valid = anchor_part_valid & anchor_valid

        total_valid_parts += anchor_metrics["anchor_total_valid_parts"].detach()
        total_anchor_hits += anchor_metrics["anchor_total_hits"].detach()

        B, K = part_ids.shape
        for b in range(B):
            valid_k = torch.nonzero(valid[b], as_tuple=False).squeeze(1)
            for k in valid_k.tolist():
                pid = int(part_ids[b, k].item())
                if 0 <= pid < num_parts:
                    pool_sum[pid] += proto_part[b, k]
                    pool_count[pid] += 1.0

    valid_pool = pool_count >= float(min_pool_count)
    V_pool = pool_sum / pool_count.clamp_min(1.0)[:, None]
    V_pool = criterion._safe_normalize(V_pool.float(), dim=-1)

    metrics = {
        "pool_valid_parts": int(valid_pool.sum().item()),
        "pool_mean_count": float(pool_count[valid_pool].mean().item()) if valid_pool.any() else 0.0,
        "pool_max_count": float(pool_count.max().item()) if pool_count.numel() > 0 else 0.0,
        "pool_anchor_total_valid_parts": float(total_valid_parts.item()),
        "pool_anchor_total_hits": float(total_anchor_hits.item()),
        "pool_anchor_hit_rate": float((total_anchor_hits / total_valid_parts.clamp_min(1.0)).item()),
    }
    return V_pool.detach(), valid_pool.detach(), pool_count.detach(), metrics


@torch.no_grad()
def build_global_patch_bank_pool(
    model,
    criterion: JointObjPartLoss,
    train_dataloader,
    num_parts: int,
    dino_dim: int,
    device: torch.device,
    max_patches_per_obj: int = 100000,
    min_pool_count: int = 1,
):
    """Version A: each object class builds one global object-mask patch bank, then reuses _anchor_proto_em_pool()."""
    model.eval()
    criterion.eval()

    obj_patch_bank = {}
    obj_part_ids_bank = {}
    obj_part_text_bank = {}

    pbar = tqdm(train_dataloader, desc="collect global patch banks")
    for batch in pbar:
        batch = _move_joint_batch_to_device(batch, device)
        patch_tokens = criterion._safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()
        category_id = batch["category_id"].long()
        part_ids = batch["part_category_id"].long()
        part_text = batch["part_text_feat"].float()
        part_valid = batch["part_valid_mask"].bool()

        B = patch_tokens.shape[0]
        for b in range(B):
            cid = int(category_id[b].item())
            valid_patch = obj_mask[b]
            if valid_patch.sum() == 0:
                continue

            obj_patch_bank.setdefault(cid, []).append(patch_tokens[b][valid_patch].detach().cpu())
            if cid not in obj_part_ids_bank:
                keep = part_valid[b]
                obj_part_ids_bank[cid] = part_ids[b][keep].detach().cpu()
                obj_part_text_bank[cid] = part_text[b][keep].detach().cpu()

    pool_sum = torch.zeros(num_parts, dino_dim, device=device)
    pool_count = torch.zeros(num_parts, device=device)
    total_valid_parts = torch.tensor(0.0, device=device)

    for cid in tqdm(sorted(obj_patch_bank.keys()), desc="global anchor per object"):
        if cid not in obj_part_ids_bank:
            continue

        X_cpu = torch.cat(obj_patch_bank[cid], dim=0)
        if X_cpu.numel() == 0:
            continue
        if max_patches_per_obj is not None and max_patches_per_obj > 0 and X_cpu.shape[0] > max_patches_per_obj:
            perm = torch.randperm(X_cpu.shape[0])[:max_patches_per_obj]
            X_cpu = X_cpu[perm]

        X = criterion._safe_normalize(X_cpu.to(device=device, dtype=torch.float32), dim=-1)
        part_ids_obj = obj_part_ids_bank[cid].to(device=device, dtype=torch.long)
        part_text_obj = obj_part_text_bank[cid].to(device=device, dtype=torch.float32)

        K = int(part_ids_obj.shape[0])
        M = int(X.shape[0])
        if K == 0 or M == 0:
            continue

        part_proj = model.project_clip_txt(part_text_obj[None])
        part_proj = criterion._safe_normalize(part_proj.float(), dim=-1)
        abs_logits = torch.einsum("bkd,nd->bkn", part_proj, X) / float(criterion.patch_temperature)

        obj_mask_global = torch.ones(1, M, device=device, dtype=torch.bool)
        part_valid_global = torch.ones(1, K, device=device, dtype=torch.bool)
        dummy_gt = torch.zeros(1, K, M, device=device, dtype=torch.bool)

        _, proto_part, anchor_metrics, _, anchor_valid = criterion._anchor_proto_em_pool(
            patch_tokens=X[None],
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_global,
            part_valid_mask=part_valid_global,
            part_gt_mask_patch=dummy_gt,
            num_iters=criterion.em_iters,
            return_anchor_tokens=True,
        )
        proto_obj = criterion._safe_normalize(proto_part.float(), dim=-1)[0]
        valid_obj = anchor_valid.bool()[0]
        total_valid_parts += anchor_metrics["anchor_total_valid_parts"].detach()

        for k in torch.nonzero(valid_obj, as_tuple=False).squeeze(1).tolist():
            pid = int(part_ids_obj[k].item())
            if 0 <= pid < num_parts:
                pool_sum[pid] += proto_obj[k]
                pool_count[pid] += 1.0

    valid_pool = pool_count >= float(min_pool_count)
    V_pool = pool_sum / pool_count.clamp_min(1.0)[:, None]
    V_pool = criterion._safe_normalize(V_pool.float(), dim=-1)

    metrics = {
        "pool_valid_parts": int(valid_pool.sum().item()),
        "pool_mean_count": float(pool_count[valid_pool].mean().item()) if valid_pool.any() else 0.0,
        "pool_max_count": float(pool_count.max().item()) if pool_count.numel() > 0 else 0.0,
        "pool_anchor_total_valid_parts": float(total_valid_parts.item()),
        "pool_anchor_total_hits": 0.0,
        "pool_anchor_hit_rate": 0.0,
        "max_patches_per_obj": int(max_patches_per_obj) if max_patches_per_obj is not None else -1,
    }
    return V_pool.detach(), valid_pool.detach(), pool_count.detach(), metrics


def epoch_part_contrastive_loss(
    model,
    criterion: JointObjPartLoss,
    T_raw: torch.Tensor,
    V_pool: torch.Tensor,
    valid_pool: torch.Tensor,
    temperature: float = 0.07,
):
    T_proj = model.project_clip_txt(T_raw.float())
    T_proj = criterion._safe_normalize(T_proj.float(), dim=-1)

    valid = valid_pool.bool()
    valid = valid & torch.isfinite(V_pool).all(dim=-1)
    valid = valid & (V_pool.norm(dim=-1) > 1e-8)

    T = T_proj[valid]
    V = V_pool[valid].detach()
    if T.shape[0] < 2:
        return None, int(T.shape[0])

    logits = torch.matmul(T, V.T) / float(temperature)
    labels = torch.arange(T.shape[0], device=T.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return loss, int(T.shape[0])


def do_train_joint_epoch_pool(
    model,
    train_dataset,
    val_dataset,
    train_cfg,
    seed: int = 123,
    optimizer_name: str = "Adam",
    weight_decay: float = 0.05,
    scheduler_name: str = "linear",
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

    lr = train_cfg["lr"]
    num_epochs = train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]
    shuffle = train_cfg.get("shuffle", True)
    save_best_model = train_cfg.get("save_best_model", True)
    object_miou_max_drop = float(train_cfg.get("object_miou_max_drop", 0.5))
    select_best_by_miou = bool(train_cfg.get("select_best_by_miou", True))

    obj_ltype = train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce"))
    obj_margin = train_cfg.get("margin", 0.2)
    obj_max_violation = train_cfg.get("max_violation", True)
    lambda_obj = float(train_cfg.get("lambda_obj", 1.0))
    topk_ratio = train_cfg.get("topk_ratio", 0.1)
    patch_temperature = train_cfg.get("patch_temperature", 0.07)
    em_iters = int(train_cfg.get("em_iters", 3))
    present_only_anchor = bool(train_cfg.get("present_only_anchor", train_cfg.get("oracle_present_only_anchor", False)))

    epoch_pool_mode = train_cfg.get("epoch_pool_mode", "local_pseudo_mean")
    if epoch_pool_mode not in {"local_pseudo_mean", "global_patch_bank"}:
        raise ValueError(f"epoch_pool_mode must be local_pseudo_mean or global_patch_bank, got {epoch_pool_mode}")

    num_parts = int(train_cfg.get("num_parts", 116))
    lambda_epoch_part = float(train_cfg.get("lambda_epoch_part", 1.0))
    epoch_part_temperature = float(train_cfg.get("epoch_part_temperature", 0.07))
    min_pool_count = int(train_cfg.get("min_pool_count", 1))
    max_patches_per_obj = int(train_cfg.get("max_patches_per_obj", 100000))
    pool_update_every = int(train_cfg.get("pool_update_every", 1))
    warmup_epochs = int(train_cfg.get("epoch_pool_warmup_epochs", 0))

    if not eval_proj_name:
        raise ValueError("eval_proj_name must be provided for mIoU evaluation.")
    if miou_eval_script is None or miou_eval_cfg is None or miou_eval_base_cfg is None:
        raise ValueError("miou_eval_script / miou_eval_cfg / miou_eval_base_cfg must all be provided.")

    print(
        "[joint epoch-pool config] "
        f"lambda_obj={lambda_obj}, epoch_pool_mode={epoch_pool_mode}, "
        f"lambda_epoch_part={lambda_epoch_part}, epoch_part_temperature={epoch_part_temperature}, "
        f"num_parts={num_parts}, min_pool_count={min_pool_count}, pool_update_every={pool_update_every}, "
        f"warmup_epochs={warmup_epochs}, present_only_anchor={present_only_anchor}, "
        f"em_iters={em_iters}, max_patches_per_obj={max_patches_per_obj}, "
        f"min_obj_area_ratio={getattr(train_dataset, 'min_obj_area_ratio', 0.0)}"
    )
    print("[epoch-pool] Batch-level part losses are disabled. Object loss is batch-wise; part loss is epoch-level.")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, collate_fn=joint_collate_fn)
    pool_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=joint_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=joint_collate_fn)

    criterion = JointObjPartLoss(
        model,
        obj_ltype=obj_ltype,
        obj_margin=obj_margin,
        obj_max_violation=obj_max_violation,
        lambda_obj=lambda_obj,
        lambda_inst=0.0,
        lambda_overlap=0.0,
        lambda_spear=0.0,
        topk_ratio=topk_ratio,
        patch_temperature=patch_temperature,
        em_iters=em_iters,
        present_only_anchor=present_only_anchor,
    )
    criterion.present_only_anchor = present_only_anchor
    criterion.oracle_present_only_anchor = present_only_anchor

    one = next(iter(pool_dataloader))
    dino_dim = int(one["patch_tokens"].shape[-1])
    print(f"[epoch-pool] inferred dino_dim={dino_dim}")

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented")

    total_steps = len(train_dataloader) * num_epochs
    if scheduler_name == "linear" and warmup == 0:
        scheduler = None
    elif scheduler_name == "linear" and warmup > 0:
        scheduler = const_lr(optimizer, lr, warmup, total_steps)
    elif scheduler_name == "cosine":
        scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    else:
        scheduler = None

    T_raw, valid_text, text_count = collect_part_text_proto(pool_dataloader, num_parts=num_parts, device=device)
    print(
        "[epoch-pool] text proto: "
        f"valid_parts={int(valid_text.sum().item())}/{num_parts}, "
        f"mean_count={float(text_count[valid_text].mean().item()) if valid_text.any() else 0.0:.2f}"
    )

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

    cached_V_pool = None
    cached_valid_pool = None
    cached_pool_metrics = {}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} / {num_epochs - 1}")
        train_metrics = train_joint_object_only(model, train_dataloader, criterion, optimizer, scheduler=scheduler, epoch=epoch)

        do_pool_update = (epoch >= warmup_epochs) and ((epoch - warmup_epochs) % max(pool_update_every, 1) == 0)
        if do_pool_update:
            if epoch_pool_mode == "local_pseudo_mean":
                cached_V_pool, cached_valid_pool, _, cached_pool_metrics = build_local_pseudo_mean_pool(
                    model=model,
                    criterion=criterion,
                    train_dataloader=pool_dataloader,
                    num_parts=num_parts,
                    dino_dim=dino_dim,
                    device=device,
                    present_only_anchor=present_only_anchor,
                    min_pool_count=min_pool_count,
                )
            else:
                cached_V_pool, cached_valid_pool, _, cached_pool_metrics = build_global_patch_bank_pool(
                    model=model,
                    criterion=criterion,
                    train_dataloader=pool_dataloader,
                    num_parts=num_parts,
                    dino_dim=dino_dim,
                    device=device,
                    max_patches_per_obj=max_patches_per_obj,
                    min_pool_count=min_pool_count,
                )
        else:
            print("[epoch-pool] skip pool update due to warmup/update schedule.")

        epoch_part_loss_value = 0.0
        epoch_part_valid_num = 0
        if cached_V_pool is not None and cached_valid_pool is not None and lambda_epoch_part > 0 and epoch >= warmup_epochs:
            model.train()
            valid_pool = cached_valid_pool.to(device) & valid_text.to(device)
            part_loss, valid_num = epoch_part_contrastive_loss(
                model=model,
                criterion=criterion,
                T_raw=T_raw.to(device),
                V_pool=cached_V_pool.to(device),
                valid_pool=valid_pool,
                temperature=epoch_part_temperature,
            )
            epoch_part_valid_num = int(valid_num)
            if part_loss is not None:
                optimizer.zero_grad()
                (lambda_epoch_part * part_loss).backward()
                optimizer.step()
                epoch_part_loss_value = float(part_loss.detach().cpu().item())
                print(f"[epoch-pool] part_loss={epoch_part_loss_value:.4f}, valid_parts={epoch_part_valid_num}")
            else:
                print(f"[epoch-pool] skip part loss because valid_parts={epoch_part_valid_num}")

        train_metrics.update({
            "epoch_part_loss": epoch_part_loss_value,
            "epoch_part_valid_num": epoch_part_valid_num,
            **{f"epoch_pool/{k}": v for k, v in cached_pool_metrics.items()},
        })

        val_metrics = validate_joint_object_only(model, val_dataloader, criterion)
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
        obj_eval_metrics["obj_eval_miou_delta_vs_baseline"] = float(obj_eval_metrics["obj_eval_miou"] - baseline_obj_miou)
        val_metrics = {**val_metrics, **obj_eval_metrics}

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch}: train_total={train_metrics['total']:.4f}, "
            f"epoch_part_loss={train_metrics.get('epoch_part_loss', 0.0):.4f}, "
            f"pool_valid={train_metrics.get('epoch_part_valid_num', 0)}, "
            f"val_total={val_metrics['total']:.4f}, obj_eval_miou={val_metrics['obj_eval_miou']:.4f}, "
            f"miou_delta_vs_baseline={val_metrics['obj_eval_miou_delta_vs_baseline']:.4f}"
        )

        current_obj_miou = val_metrics["obj_eval_miou"]
        obj_ok = current_obj_miou >= (baseline_obj_miou - object_miou_max_drop)

        if save_best_model:
            if select_best_by_miou:
                if obj_ok and (best_obj_miou is None or current_obj_miou > best_obj_miou):
                    best_obj_miou = current_obj_miou
                    best_val = val_metrics["total"]
                    best_model = deepcopy(model)
                    print("Best model updated by object mIoU under guardrail.")
                elif not obj_ok:
                    print(
                        f"Skip best update because object mIoU dropped too much: "
                        f"{current_obj_miou:.4f} < {baseline_obj_miou - object_miou_max_drop:.4f}"
                    )
            else:
                if obj_ok and (best_val is None or val_metrics["total"] < best_val):
                    best_val = val_metrics["total"]
                    best_model = deepcopy(model)
                    print("Best validation total loss under object mIoU guardrail, saving current best model in memory.")
                elif not obj_ok:
                    print(
                        f"Skip best update because object mIoU dropped too much: "
                        f"{current_obj_miou:.4f} < {baseline_obj_miou - object_miou_max_drop:.4f}"
                    )

    model = best_model if save_best_model else model
    return model, train_history, val_history
