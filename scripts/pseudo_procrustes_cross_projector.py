from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn
from src.loss_joint import JointObjPartLoss


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_projector(config: Dict, checkpoint: str, device: torch.device, tag: str) -> torch.nn.Module:
    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"])

    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    ret = model.load_state_dict(ckpt, strict=False)
    print(f"[{tag}] checkpoint={checkpoint}")
    print(f"[{tag}] missing_keys={getattr(ret, 'missing_keys', [])}")
    print(f"[{tag}] unexpected_keys={getattr(ret, 'unexpected_keys', [])}")

    model.to(device)
    model.eval()
    return model


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_dataset(args, config: Dict, dataset_path: str) -> DinoClipJointDataset:
    dataset_cfg = config.get("dataset", {})
    min_obj_area_ratio = float(dataset_cfg.get("min_obj_area_ratio", args.min_obj_area_ratio))
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


def get_part_names(dataset: DinoClipJointDataset, num_parts: int) -> List[str]:
    names = [f"part_{i}" for i in range(num_parts)]
    samples = dataset.data.values() if isinstance(dataset.data, dict) else dataset.data
    for sample in samples:
        pids = sample.get("part_category_id", [])
        pnames = sample.get("part_class_name", [])
        if torch.is_tensor(pids):
            pids = pids.detach().cpu().tolist()
        for pid, pname in zip(pids, pnames):
            pid = int(pid)
            if 0 <= pid < num_parts:
                names[pid] = str(pname)
    for _, bank in getattr(dataset, "class_part_bank", {}).items():
        pids = bank.get("part_ids", [])
        pnames = bank.get("part_names", [])
        if torch.is_tensor(pids):
            pids = pids.detach().cpu().tolist()
        for pid, pname in zip(pids, pnames):
            pid = int(pid)
            if 0 <= pid < num_parts:
                names[pid] = str(pname)
    return names


def make_joint_criterion(config: Dict, model: torch.nn.Module) -> JointObjPartLoss:
    train_cfg = config.get("train", {})
    criterion = JointObjPartLoss(
        model,
        obj_ltype=train_cfg.get("obj_ltype", train_cfg.get("ltype", "infonce")),
        obj_margin=float(train_cfg.get("margin", 0.2)),
        obj_max_violation=bool(train_cfg.get("max_violation", True)),
        lambda_obj=float(train_cfg.get("lambda_obj", 1.0)),
        lambda_inst=float(train_cfg.get("lambda_inst", 0.2)),
        lambda_overlap=float(train_cfg.get("lambda_overlap", 0.05)),
        lambda_spear=float(train_cfg.get("lambda_spear", 0.0)),
        patch_temperature=float(train_cfg.get("patch_temperature", 0.07)),
        em_iters=int(train_cfg.get("em_iters", 3)),
        present_only_anchor=bool(train_cfg.get("present_only_anchor", False)),
    )
    criterion.present_only_anchor = bool(train_cfg.get("present_only_anchor", False))
    criterion.eval()
    return criterion


def make_global_criterion(config: Dict, model: torch.nn.Module):
    from src.loss_global import PartLoss

    train_cfg = config.get("train", {})
    criterion = PartLoss(
        model,
        lambda_inst=float(train_cfg.get("lambda_inst", 0.2)),
        lambda_overlap=float(train_cfg.get("lambda_overlap", 0.05)),
        lambda_spear=float(train_cfg.get("lambda_spear", 0.0)),
        topk_ratio=float(train_cfg.get("topk_ratio", 0.1)),
        patch_temperature=float(train_cfg.get("patch_temperature", 0.07)),
        em_iters=int(train_cfg.get("em_iters", 3)),
        present_only_anchor=bool(train_cfg.get("present_only_anchor", False)),
        anchor_matcher=str(train_cfg.get("anchor_matcher", "greedy")),
        anchor_score_type=str(train_cfg.get("anchor_score_type", "relative")),
    )
    criterion.present_only_anchor = bool(train_cfg.get("present_only_anchor", False))
    criterion.eval()
    return criterion


class Accumulator:
    def __init__(self, num_parts: int):
        self.num_parts = int(num_parts)
        self.sums = {}
        self.counts = {}

    def add(self, source: str, pid: int, vec: torch.Tensor):
        pid = int(pid)
        if pid < 0 or pid >= self.num_parts:
            return
        vec = safe_normalize(vec.detach().float().cpu(), dim=-1)
        if source not in self.sums:
            self.sums[source] = [None for _ in range(self.num_parts)]
            self.counts[source] = torch.zeros(self.num_parts, dtype=torch.long)
        if self.sums[source][pid] is None:
            self.sums[source][pid] = torch.zeros_like(vec)
        self.sums[source][pid] += vec
        self.counts[source][pid] += 1

    def mean(self, source: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if source not in self.sums:
            raise KeyError(source)
        first = next((x for x in self.sums[source] if x is not None), None)
        if first is None:
            raise RuntimeError(f"no vectors for source={source}")
        mat = torch.zeros((self.num_parts, first.numel()), dtype=torch.float32)
        valid = torch.zeros(self.num_parts, dtype=torch.bool)
        count = self.counts[source].clone()
        for i, s in enumerate(self.sums[source]):
            if s is not None and count[i] > 0:
                mat[i] = s / int(count[i].item())
                valid[i] = True
        return safe_normalize(mat, dim=-1), valid, count


@torch.no_grad()
def collect_text_and_gt(
    text_model: torch.nn.Module,
    loader: DataLoader,
    acc: Accumulator,
    max_batches: int,
):
    device = next(text_model.parameters()).device
    for bi, batch in enumerate(tqdm(loader, desc="collect text-projector projected text + GT")):
        if max_batches >= 0 and bi >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        patch = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()
        part_gt = batch["part_gt_mask_patch"].bool()
        part_valid = batch["part_valid_mask"].bool()
        part_ids = batch["part_category_id"].long()
        raw = safe_normalize(batch["part_text_feat"].float(), dim=-1)
        text_proj = safe_normalize(text_model.project_clip_txt(raw), dim=-1)

        B, K, _ = raw.shape
        for b in range(B):
            for k in range(K):
                if not bool(part_valid[b, k].item()):
                    continue
                pid = int(part_ids[b, k].item())
                if pid < 0:
                    continue
                mask = part_gt[b, k] & obj_mask[b]
                if not bool(mask.any().item()):
                    continue
                gt = safe_normalize(patch[b, mask].mean(dim=0), dim=-1)
                acc.add("text_proj", pid, text_proj[b, k])
                acc.add("gt", pid, gt)


@torch.no_grad()
def collect_joint_pseudo(
    pseudo_model: torch.nn.Module,
    criterion,
    loader: DataLoader,
    acc: Accumulator,
    max_batches: int,
):
    device = next(pseudo_model.parameters()).device
    for bi, batch in enumerate(tqdm(loader, desc="collect joint pseudo from pseudo-projector")):
        if max_batches >= 0 and bi >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        patch = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()
        part_gt = batch["part_gt_mask_patch"].bool()
        part_valid = batch["part_valid_mask"].bool()
        part_ids = batch["part_category_id"].long()
        raw = safe_normalize(batch["part_text_feat"].float(), dim=-1)

        if getattr(criterion, "present_only_anchor", False):
            present = (part_gt & obj_mask[:, None, :]).sum(dim=-1) > 0
            anchor_mask = part_valid & present
        else:
            anchor_mask = part_valid
        anchor_mask = anchor_mask & obj_mask.any(dim=-1, keepdim=True)
        if not bool(anchor_mask.any().item()):
            continue

        pseudo_proj = safe_normalize(pseudo_model.project_clip_txt(raw), dim=-1)
        logits = torch.einsum("bkd,bnd->bkn", pseudo_proj, patch)
        logits = logits / float(criterion.patch_temperature)
        logits = logits.masked_fill(~obj_mask[:, None, :], -1e4)

        z, _, _ = criterion._anchor_proto_em_pool(
            patch_tokens=patch,
            abs_logits=logits,
            obj_mask_patch=obj_mask,
            part_valid_mask=anchor_mask,
            part_gt_mask_patch=part_gt,
            num_iters=int(criterion.em_iters),
        )
        z = safe_normalize(z.float(), dim=-1)

        B, K, _ = z.shape
        for b in range(B):
            for k in range(K):
                if not bool(anchor_mask[b, k].item()):
                    continue
                pid = int(part_ids[b, k].item())
                if pid < 0:
                    continue
                # compare only parts that actually exist in this object crop
                mask = part_gt[b, k] & obj_mask[b]
                if not bool(mask.any().item()):
                    continue
                acc.add("pseudo", pid, z[b, k])


@torch.no_grad()
def collect_global_pseudo(
    pseudo_model: torch.nn.Module,
    criterion,
    pool_loader: DataLoader,
    acc: Accumulator,
    max_batches: int,
):
    device = next(pseudo_model.parameters()).device
    for bi, batch in enumerate(tqdm(pool_loader, desc="collect global pseudo from pseudo-projector")):
        if max_batches >= 0 and bi >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        patch = safe_normalize(batch["patch_tokens"].float(), dim=-1)
        obj_mask = batch["obj_mask_patch"].bool()
        part_gt = batch["part_gt_mask_patch"].bool()
        part_valid = batch["part_valid_mask"].bool()
        part_ids = batch["part_category_id"].long()
        raw = safe_normalize(batch["part_text_feat"].float(), dim=-1)

        anchor_mask = criterion._build_part_anchor_mask(
            part_valid_mask=part_valid,
            part_gt_mask_patch=part_gt,
            obj_mask_patch=obj_mask,
        )
        if not bool(anchor_mask.any().item()):
            continue

        pseudo_proj = safe_normalize(pseudo_model.project_clip_txt(raw), dim=-1)
        logits = torch.einsum("bkd,bnd->bkn", pseudo_proj, patch)
        logits = logits / float(criterion.patch_temperature)
        logits = logits.masked_fill(~obj_mask[:, None, :], -1e4)

        z, _, _ = criterion._anchor_proto_em_pool(
            patch_tokens=patch,
            abs_logits=logits,
            obj_mask_patch=obj_mask,
            part_valid_mask=anchor_mask,
            part_gt_mask_patch=part_gt,
            num_iters=int(criterion.em_iters),
        )
        z = safe_normalize(z.float(), dim=-1)

        B, K, _ = z.shape
        for b in range(B):
            for k in range(K):
                if not bool(anchor_mask[b, k].item()):
                    continue
                pid = int(part_ids[b, k].item())
                if pid >= 0:
                    acc.add("pseudo", pid, z[b, k])


def solve_orthogonal_q(x: torch.Tensor, y: torch.Tensor):
    cross = x.T @ y
    u, s, vh = torch.linalg.svd(cross, full_matrices=False)
    q = u @ vh
    return q, s


def retrieval_metrics(
    query: torch.Tensor,
    target: torch.Tensor,
    part_ids: torch.Tensor,
    label: str,
):
    query = safe_normalize(query.float(), dim=-1)
    target = safe_normalize(target.float(), dim=-1)
    sim = query @ target.T
    n = sim.shape[0]
    order = torch.argsort(sim, dim=1, descending=True)
    idx = torch.arange(n)
    ranks = (order.cpu() == idx[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    diag = sim.diag().cpu()
    top1 = (ranks <= 1).float().mean().item()
    top5 = (ranks <= min(5, n)).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()
    if n > 1:
        masked = sim.cpu().clone()
        masked[idx, idx] = -1e9
        margin = (diag - masked.max(dim=1).values).mean().item()
    else:
        margin = float("nan")
    out = {
        "num_parts": int(n),
        "top1": float(top1),
        "top5": float(top5),
        "mrr": float(mrr),
        "mean_self_cos": float(diag.mean().item()),
        "mean_self_margin": float(margin),
        "median_rank": float(ranks.float().median().item()),
    }
    print("")
    print("=" * 100)
    print(f"[{label}]")
    print(
        f"top1={out['top1']:.4f}, top5={out['top5']:.4f}, mrr={out['mrr']:.4f}, "
        f"mean_self_cos={out['mean_self_cos']:.4f}, margin={out['mean_self_margin']:.4f}, "
        f"median_rank={out['median_rank']:.1f}"
    )
    return out


def fold_q_into_linear_checkpoint(model, q: torch.Tensor, config: Dict, out_path: str):
    model_cfg = config.get("model", {})
    act = model_cfg.get("act", None)
    hidden_layer = model_cfg.get("hidden_layer", False)
    act_is_none = act is None or str(act).lower() in {"none", "null"}
    hidden_is_false = hidden_layer in (False, None, 0, "False", "false")
    if not act_is_none or not hidden_is_false:
        raise RuntimeError(
            "Can only fold Q into linear projector with act=null and hidden_layer=False. "
            f"Got act={act!r}, hidden_layer={hidden_layer!r}."
        )

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if "linear_layer.weight" not in state:
        raise KeyError("linear_layer.weight not found in text projector state_dict.")

    q = q.detach().cpu().float()
    w = state["linear_layer.weight"].float()
    if w.shape[0] != q.shape[0] or q.shape[0] != q.shape[1]:
        raise ValueError(f"shape mismatch: W={tuple(w.shape)}, Q={tuple(q.shape)}")

    # original output: y = W x + b. desired output: y' = y @ Q.
    # With PyTorch linear output x @ W.T + b, new W = Q.T @ W, new b = Q.T @ b.
    state["linear_layer.weight"] = (q.T @ w).to(state["linear_layer.weight"].dtype)
    if "linear_layer.bias" in state:
        state["linear_layer.bias"] = (q.T @ state["linear_layer.bias"].float()).to(state["linear_layer.bias"].dtype)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    print(f"[saved rotated text checkpoint] {out}")


def collect_split(args, split_name: str, dataset_path: str, text_config, text_model, pseudo_config, pseudo_model):
    dataset = build_dataset(args, pseudo_config, dataset_path)
    acc = Accumulator(args.num_parts)

    common_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=joint_collate_fn,
    )
    collect_text_and_gt(text_model, common_loader, acc, args.max_batches)

    if args.pipeline == "joint":
        criterion = make_joint_criterion(pseudo_config, pseudo_model).to(next(pseudo_model.parameters()).device)
        pseudo_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=joint_collate_fn,
        )
        collect_joint_pseudo(pseudo_model, criterion, pseudo_loader, acc, args.max_batches)
    else:
        from src.dataset_global import CategoryPatchPoolDataset, global_pool_collate_fn

        criterion = make_global_criterion(pseudo_config, pseudo_model).to(next(pseudo_model.parameters()).device)
        sample_patches = None if args.global_sample_patches <= 0 else args.global_sample_patches
        pool_dataset = CategoryPatchPoolDataset(
            dataset,
            sample_patches_per_step=sample_patches,
            steps_per_epoch=None,
            store_dtype=torch.float16,
            seed=args.seed,
            fixed_subsample=True,
        )
        pool_loader = DataLoader(
            pool_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=global_pool_collate_fn,
        )
        collect_global_pseudo(pseudo_model, criterion, pool_loader, acc, args.max_batches)

    text, text_valid, text_count = acc.mean("text_proj")
    pseudo, pseudo_valid, pseudo_count = acc.mean("pseudo")
    gt, gt_valid, gt_count = acc.mean("gt")
    valid = text_valid & pseudo_valid & gt_valid
    print(f"[{split_name}] valid parts={int(valid.sum().item())}")
    return {
        "text_proj": text[valid],
        "pseudo": pseudo[valid],
        "gt": gt[valid],
        "part_ids": torch.nonzero(valid, as_tuple=False).squeeze(1),
        "counts": {
            "text": text_count[valid],
            "pseudo": pseudo_count[valid],
            "gt": gt_count[valid],
        },
    }


def evaluate_split(split_name: str, data: Dict, q: torch.Tensor):
    text = data["text_proj"]
    pseudo = data["pseudo"]
    gt = data["gt"]
    ids = data["part_ids"]
    mapped = safe_normalize(text @ q.cpu().float(), dim=-1)
    return {
        "text_to_pseudo_before": retrieval_metrics(text, pseudo, ids, f"{split_name}: Stage1/source text -> Stage2 pseudo BEFORE Q"),
        "mapped_text_to_pseudo_after": retrieval_metrics(mapped, pseudo, ids, f"{split_name}: Stage1/source text @ Q -> Stage2 pseudo AFTER Q"),
        "text_to_gt_before": retrieval_metrics(text, gt, ids, f"{split_name}: Stage1/source text -> GT BEFORE Q"),
        "mapped_text_to_gt_after": retrieval_metrics(mapped, gt, ids, f"{split_name}: Stage1/source text @ Q -> GT AFTER Q"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=["joint", "global"], required=True)

    p.add_argument("--text_model_config", required=True)
    p.add_argument("--text_checkpoint", required=True)
    p.add_argument("--pseudo_model_config", required=True)
    p.add_argument("--pseudo_checkpoint", required=True)

    p.add_argument("--fit_dataset", required=True)
    p.add_argument("--eval_dataset", default=None)

    p.add_argument("--obj_feature_name", default="avg_self_attn_out")
    p.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    p.add_argument("--obj_text_name", default="ann_feats")
    p.add_argument("--part_text_name", default="part_ann_feats")
    p.add_argument("--resize_dim", type=int, default=448)
    p.add_argument("--crop_dim", type=int, default=448)
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--with_background", action="store_true", default=False)
    p.add_argument("--path_prefix", default=None)
    p.add_argument("--min_obj_area_ratio", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_parts", type=int, default=116)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_batches", type=int, default=-1)
    p.add_argument("--global_sample_patches", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--out_transform_pth", default=None)
    p.add_argument("--out_rotated_text_ckpt", default=None)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    text_config = load_config(args.text_model_config)
    pseudo_config = load_config(args.pseudo_model_config)

    text_model = load_projector(text_config, args.text_checkpoint, device, tag="text_projector")
    pseudo_model = load_projector(pseudo_config, args.pseudo_checkpoint, device, tag="pseudo_projector")

    print("")
    print("[cross pseudo-Procrustes]")
    print(f"  text projector:   {args.text_checkpoint}")
    print(f"  pseudo projector: {args.pseudo_checkpoint}")
    print(f"  pipeline:         {args.pipeline}")
    print("  fit: source projected text -> target pseudo prototype")

    fit_data = collect_split(args, "fit", args.fit_dataset, text_config, text_model, pseudo_config, pseudo_model)
    if fit_data["text_proj"].shape[1] != fit_data["pseudo"].shape[1]:
        raise ValueError(
            f"source text dim and pseudo dim differ: "
            f"{fit_data['text_proj'].shape[1]} vs {fit_data['pseudo'].shape[1]}"
        )

    q, s = solve_orthogonal_q(fit_data["text_proj"].float(), fit_data["pseudo"].float())
    q_err = ((q.T @ q) - torch.eye(q.shape[0])).abs().max().item()
    det = torch.linalg.det(q).item()
    print("")
    print("=" * 100)
    print(f"[Q] shape={tuple(q.shape)} det={det:.6f} max_abs(Q^TQ-I)={q_err:.8f} fit_parts={fit_data['text_proj'].shape[0]}")

    results = {
        "pipeline": args.pipeline,
        "text_model_config": args.text_model_config,
        "text_checkpoint": args.text_checkpoint,
        "pseudo_model_config": args.pseudo_model_config,
        "pseudo_checkpoint": args.pseudo_checkpoint,
        "fit_dataset": args.fit_dataset,
        "eval_dataset": args.eval_dataset,
        "q_shape": list(q.shape),
        "q_det": float(det),
        "q_orth_error": float(q_err),
        "fit_part_ids": fit_data["part_ids"].tolist(),
        "singular_values": s.detach().cpu().tolist(),
        "fit": evaluate_split("fit", fit_data, q),
    }

    if args.eval_dataset:
        eval_data = collect_split(args, "eval", args.eval_dataset, text_config, text_model, pseudo_config, pseudo_model)
        results["eval_part_ids"] = eval_data["part_ids"].tolist()
        results["eval"] = evaluate_split("eval", eval_data, q)

    if args.out_transform_pth:
        out = Path(args.out_transform_pth)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "Q": q.detach().cpu(),
                "singular_values": s.detach().cpu(),
                "pipeline": args.pipeline,
                "text_model_config": args.text_model_config,
                "text_checkpoint": args.text_checkpoint,
                "pseudo_model_config": args.pseudo_model_config,
                "pseudo_checkpoint": args.pseudo_checkpoint,
                "fit_dataset": args.fit_dataset,
                "fit_part_ids": fit_data["part_ids"].cpu(),
            },
            out,
        )
        print(f"[saved Q] {out}")

    if args.out_rotated_text_ckpt:
        fold_q_into_linear_checkpoint(text_model, q, text_config, args.out_rotated_text_ckpt)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[saved json] {out}")


if __name__ == "__main__":
    main()
