
from copy import deepcopy
import os
from pathlib import Path
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


def _build_part_name_map(joint_dataset) -> Dict[int, str]:
    """Build part-id -> part-name mapping from the existing dataset metadata."""
    samples = (
        joint_dataset.data.values()
        if isinstance(joint_dataset.data, dict)
        else joint_dataset.data
    )

    part_name_map: Dict[int, str] = {}

    for sample in samples:
        part_ids = sample.get("part_category_id", [])
        part_names = sample.get("part_class_name", [])

        if torch.is_tensor(part_ids):
            part_ids = part_ids.detach().cpu().tolist()

        for part_id, part_name in zip(part_ids, part_names):
            part_name_map[int(part_id)] = str(part_name)

    return part_name_map


def _part_name(part_id: int, part_name_map: Dict[int, str]) -> str:
    if int(part_id) < 0:
        return "NONE"
    return part_name_map.get(int(part_id), f"part_{int(part_id)}")


def _anchor_gt_name(part_ids, part_name_map: Dict[int, str]) -> str:
    if len(part_ids) == 0:
        return "NONE"
    return "|".join(_part_name(pid, part_name_map) for pid in part_ids)


# @torch.no_grad()
# def audit_full_train_pool(
#     model,
#     train_pool_dataset,
#     criterion,
#     epoch: int,
#     out_txt: str,
#     part_name_map: Dict[int, str],
# ):
#     """
#     Audit the complete category pools after one training epoch.

#     For each projected part text feature, print:
#       target text -> anchor patch GT -> nearest GT prototype to EM pseudo prototype.
#     """
#     model.eval()
#     criterion.eval()
#     device = next(model.parameters()).device

#     lines = []
#     lines.append("=" * 150)
#     lines.append(f"[epoch {epoch:03d}] full_train_global_pool")
#     lines.append("")
#     lines.append(
#         f"{'cat':>4}  "
#         f"{'target_id':>9}  "
#         f"{'target_text':<38}  "
#         f"{'anchor_patch_GT':<38}  "
#         f"{'pseudo_proto_nearest_GT':<38}  "
#         f"{'cos':>8}"
#     )
#     lines.append("-" * 150)

#     for cat in train_pool_dataset.categories:
#         # Use the stored complete pool directly, not __getitem__(), because
#         # __getitem__() may randomly subsample sample_patches_per_step patches.
#         full_item = train_pool_dataset.pools[cat]
#         batch = global_pool_collate_fn([full_item])
#         batch = _move_batch_to_device(batch, device)

#         records = criterion.audit_anchor_and_proto(batch)

#         for record in records:
#             target_id = int(record["target_part_id"])
#             nearest_id = int(record["pseudo_proto_nearest_gt_part_id"])
#             nearest_cos = float(record["pseudo_proto_nearest_gt_cos"])

#             target_name = _part_name(target_id, part_name_map)
#             anchor_name = _anchor_gt_name(
#                 record["anchor_gt_part_ids"],
#                 part_name_map,
#             )
#             nearest_name = _part_name(nearest_id, part_name_map)

#             cos_text = "nan" if not np.isfinite(nearest_cos) else f"{nearest_cos:.4f}"

#             lines.append(
#                 f"{int(record['category_id']):>4d}  "
#                 f"{target_id:>9d}  "
#                 f"{target_name:<38.38}  "
#                 f"{anchor_name:<38.38}  "
#                 f"{nearest_name:<38.38}  "
#                 f"{cos_text:>8}"
#             )

#         del batch, records
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     lines.append("=" * 150)
#     text = "\n".join(lines)

#     print("")
#     print(text)

#     out_path = Path(out_txt)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with out_path.open("a", encoding="utf-8") as f:
#         f.write(text)
#         f.write("\n\n")
#         f.flush()


def build_cached_train_gt_prototypes(
    train_pool_dataset,
    eps: float = 1e-6,
    chunk_size: int = 65536,
):
    """Precompute train GT visual part prototypes exactly once.

    This uses the already-built complete category pools, so it does not reload
    the dataset or change the training pipeline.  For every semantic part that
    has at least one GT-covered patch, cache:

      * its original text feature;
      * its GT visual prototype (mean of normalized GT-covered patch tokens,
        followed by L2 normalization);
      * its global part id.

    The calculation is chunked to avoid converting a very large category pool
    to float32 all at once.  The returned tensors are small (about 116 rows for
    fine VOC116) and can be kept on GPU during training.
    """
    text_features = []
    gt_prototypes = []
    part_ids = []

    for cat in train_pool_dataset.categories:
        item = train_pool_dataset.pools[cat]
        patch_tokens = item["patch_tokens"]          # [M, D], normally CPU float16
        part_gt_mask = item["part_gt_mask_patch"].bool()  # [K, M]
        part_text_feat = item["part_text_feat"].float()   # [K, Dt]
        part_category_id = item["part_category_id"].long()

        if patch_tokens.ndim != 2 or part_gt_mask.ndim != 2:
            raise ValueError(
                f"Invalid pool shapes for category {cat}: "
                f"patch_tokens={tuple(patch_tokens.shape)}, "
                f"part_gt_mask_patch={tuple(part_gt_mask.shape)}"
            )

        K = int(part_gt_mask.shape[0])
        M = int(patch_tokens.shape[0])
        D = int(patch_tokens.shape[1])

        proto_sum = torch.zeros((K, D), dtype=torch.float32)
        proto_count = torch.zeros((K,), dtype=torch.float32)

        for start in range(0, M, int(chunk_size)):
            end = min(start + int(chunk_size), M)
            patch_chunk = patch_tokens[start:end].float()
            patch_chunk = torch.nn.functional.normalize(
                patch_chunk,
                dim=-1,
                eps=eps,
            )
            mask_chunk = part_gt_mask[:, start:end].float()

            proto_sum += mask_chunk @ patch_chunk
            proto_count += mask_chunk.sum(dim=1)

        valid = proto_count > 0
        if not bool(valid.any().item()):
            continue

        proto = proto_sum[valid] / proto_count[valid, None].clamp_min(1.0)
        proto = torch.nn.functional.normalize(proto, dim=-1, eps=eps)

        text_features.append(part_text_feat[valid].cpu())
        gt_prototypes.append(proto.cpu())
        part_ids.append(part_category_id[valid].cpu())

    if len(text_features) == 0:
        raise RuntimeError("No valid GT part prototypes could be built from train pools.")

    cache = {
        "part_text_feat": torch.cat(text_features, dim=0).contiguous(),
        "gt_prototypes": torch.cat(gt_prototypes, dim=0).contiguous(),
        "part_category_id": torch.cat(part_ids, dim=0).contiguous(),
    }

    print(
        "[gt proto cache] cached "
        f"{cache['part_text_feat'].shape[0]} GT part prototypes once "
        f"for per-epoch cosine audit"
    )
    return cache


@torch.no_grad()
def cached_projected_text_gt_proto_cosine(model, gt_proto_cache) -> float:
    """Project cached part text features and average paired GT-prototype cosine.

    This performs no patch-pool scan.  Per epoch, the work is only one text
    projection over the cached semantic parts and one row-wise cosine mean.
    """
    model.eval()
    device = next(model.parameters()).device

    part_text_feat = gt_proto_cache["part_text_feat"]
    gt_prototypes = gt_proto_cache["gt_prototypes"]

    if part_text_feat.device != device:
        part_text_feat = part_text_feat.to(device)
        gt_prototypes = gt_prototypes.to(device)
        gt_proto_cache["part_text_feat"] = part_text_feat
        gt_proto_cache["gt_prototypes"] = gt_prototypes
        gt_proto_cache["part_category_id"] = gt_proto_cache["part_category_id"].to(device)

    part_proj = model.project_clip_txt(part_text_feat.float())
    part_proj = torch.nn.functional.normalize(part_proj, dim=-1, eps=1e-6)
    gt_prototypes = torch.nn.functional.normalize(gt_prototypes.float(), dim=-1, eps=1e-6)

    return float((part_proj * gt_prototypes).sum(dim=-1).mean().item())


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

        running.append(_detach_metric_dict(losses))
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
        running.append(_detach_metric_dict(losses))
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
    audit_out_txt: str = "",
):
    set_seed(seed)

    lr = train_cfg['lr']
    num_epochs = train_cfg['num_epochs']
    batch_size = train_cfg['batch_size']
    shuffle = train_cfg.get('shuffle', True)

    if int(batch_size) != 1:
        raise ValueError(
            f"Global category-pool training requires train.batch_size=1, got {batch_size}."
        )

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

    # if not audit_out_txt:
    #     audit_out_txt = os.path.join(
    #         "audits",
    #         f"{eval_proj_name}_anchor_proto_audit.txt",
    #     )

    # audit_path = Path(audit_out_txt)
    # audit_path.parent.mkdir(parents=True, exist_ok=True)
    # with audit_path.open("w", encoding="utf-8") as f:
    #     f.write(
    #         "target text -> anchor patch GT -> EM pseudo prototype nearest GT prototype\n\n"
    #     )
    # print(f"[audit] anchor/prototype audit will be saved to: {audit_out_txt}")

    # part_name_map = _build_part_name_map(train_dataset)
    
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
        seed=seed + 1,
        fixed_subsample=True,
    )

    # Build GT prototypes once.  This is an audit-only cache and never enters loss.
    gt_proto_cache = build_cached_train_gt_prototypes(train_pool_dataset)

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

    initial_full_train_gt_proto_cos = cached_projected_text_gt_proto_cosine(
        model,
        gt_proto_cache,
    )

    print(
        "[gt proto cache] initial_full_train_gt_proto_cos="
        f"{initial_full_train_gt_proto_cos:.4f}"
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} / {num_epochs - 1}")
        train_metrics = train(model, train_dataloader, criterion, optimizer, scheduler=scheduler, epoch=epoch)
        val_metrics = validate(model, val_dataloader, criterion)

        # Almost-free oracle audit metric: cached GT prototypes, no pool scan.
        full_train_gt_proto_cos = cached_projected_text_gt_proto_cosine(
            model,
            gt_proto_cache,
        )
        train_metrics["full_train_gt_proto_cos"] = full_train_gt_proto_cos

        # audit_full_train_pool(
        #     model=model,
        #     train_pool_dataset=train_pool_dataset,
        #     criterion=criterion,
        #     epoch=epoch,
        #     out_txt=audit_out_txt,
        #     part_name_map=part_name_map,
        # )

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch}: "
            f"train_total={train_metrics['total']:.4f}, val_total={val_metrics['total']:.4f}, "
            f"anchor_hit_rate={val_metrics.get('anchor_hit_rate', 0.0):.4f}, "
            f"full_train_gt_proto_cos={full_train_gt_proto_cos:.4f}"
        )

    return model, train_history, val_history
