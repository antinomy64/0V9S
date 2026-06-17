import argparse
import importlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset_joint import DinoClipJointDataset, joint_collate_fn
from src.loss_joint import JointObjPartLoss



class TeePrinter:
    def __init__(self, out_txt: str):
        self.out_path = Path(out_txt)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.out_path.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str):
        self.stdout.write(text)
        self.f.write(text)

    def flush(self):
        self.stdout.flush()
        self.f.flush()

    def print(self, *args, **kwargs):
        sep = kwargs.pop("sep", " ")
        end = kwargs.pop("end", "\n")
        if kwargs:
            # Keep this audit script simple: support ordinary print-style output.
            pass
        text = sep.join(str(a) for a in args) + end
        self.write(text)
        self.flush()

    def close(self):
        self.f.close()


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_projector(config: Dict, checkpoint: str, device: torch.device) -> torch.nn.Module:
    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"])

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        # common DDP cleanup
        if isinstance(ckpt, dict):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        ret = model.load_state_dict(ckpt, strict=False)
        print(f"[model] loaded checkpoint: {checkpoint}")
        print(f"[model] missing_keys={getattr(ret, 'missing_keys', [])}")
        print(f"[model] unexpected_keys={getattr(ret, 'unexpected_keys', [])}")
    else:
        print("[model] WARNING: no checkpoint is given; auditing randomly initialized projector.")

    model.to(device)
    model.eval()
    return model


def build_name_maps(joint_dataset: DinoClipJointDataset) -> Tuple[Dict[int, str], Dict[int, str]]:
    cat_id_to_name: Dict[int, str] = {}
    part_id_to_name: Dict[int, str] = {}

    samples = joint_dataset.data.values() if isinstance(joint_dataset.data, dict) else joint_dataset.data
    for sample in samples:
        cat_id_to_name[int(sample.get("category_id", -1))] = str(sample.get("class_name", ""))
        pids = sample.get("part_category_id", [])
        pnames = sample.get("part_class_name", [])
        if torch.is_tensor(pids):
            pids = pids.detach().cpu().tolist()
        for pid, pname in zip(pids, pnames):
            part_id_to_name[int(pid)] = str(pname)

    # class_part_bank is the canonical all-parts bank used by DinoClipJointDataset.
    for _, bank in getattr(joint_dataset, "class_part_bank", {}).items():
        pids = bank.get("part_ids", [])
        pnames = bank.get("part_names", [])
        if torch.is_tensor(pids):
            pids = pids.detach().cpu().tolist()
        for pid, pname in zip(pids, pnames):
            part_id_to_name[int(pid)] = str(pname)

    return cat_id_to_name, part_id_to_name


def part_name(pid: int, part_id_to_name: Dict[int, str]) -> str:
    return part_id_to_name.get(int(pid), f"part_{int(pid)}")


def cat_name(cid: int, cat_id_to_name: Dict[int, str]) -> str:
    name = cat_id_to_name.get(int(cid), "")
    return name if name else f"cat_{int(cid)}"


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def build_gt_proto_for_one_unit(
    patch_tokens_b: torch.Tensor,          # [N, D], already normalized
    obj_mask_b: torch.Tensor,              # [N]
    part_gt_mask_b: torch.Tensor,          # [K, N]
    part_valid_b: torch.Tensor,            # [K]
    part_ids_b: torch.Tensor,              # [K]
    eps: float = 1e-6,
) -> Tuple[List[int], List[int], torch.Tensor]:
    """Return valid GT part slots, IDs, and normalized GT visual prototypes."""
    valid_slots: List[int] = []
    valid_ids: List[int] = []
    protos: List[torch.Tensor] = []

    K = int(part_gt_mask_b.shape[0])
    for j in range(K):
        if not bool(part_valid_b[j].item()):
            continue
        gt_mask = part_gt_mask_b[j].bool() & obj_mask_b.bool()
        if not bool(gt_mask.any().item()):
            continue
        proto = patch_tokens_b[gt_mask].mean(dim=0)
        proto = safe_normalize(proto, dim=-1, eps=eps)
        valid_slots.append(j)
        valid_ids.append(int(part_ids_b[j].item()))
        protos.append(proto)

    if len(protos) == 0:
        return [], [], patch_tokens_b.new_zeros((0, patch_tokens_b.shape[-1]))
    return valid_slots, valid_ids, torch.stack(protos, dim=0)


def ranking_string(
    query_vec: torch.Tensor,
    gt_proto_mat: torch.Tensor,
    gt_part_ids: List[int],
    part_id_to_name: Dict[int, str],
    digits: int = 4,
) -> str:
    """Format all GT parts sorted by cosine desc."""
    if gt_proto_mat.numel() == 0:
        return "EMPTY_GT"
    query_vec = safe_normalize(query_vec.float(), dim=-1)
    gt_proto_mat = safe_normalize(gt_proto_mat.float(), dim=-1)
    sims = query_vec @ gt_proto_mat.T
    order = torch.argsort(sims, descending=True).detach().cpu().tolist()
    chunks = []
    for idx in order:
        pid = int(gt_part_ids[idx])
        chunks.append(f"{part_name(pid, part_id_to_name)}({float(sims[idx].item()):.{digits}f})")
    return " \\ ".join(chunks)


def update_aggregate(
    agg: Dict,
    key: Tuple[int, int],
    gt_part_ids: List[int],
    proj_sims: torch.Tensor,
    anchor_sims: torch.Tensor,
    pseudo_sims: torch.Tensor,
):
    """Accumulate mean similarities by (category_id, target_part_id, gt_part_id)."""
    for local_idx, gt_pid in enumerate(gt_part_ids):
        gt_pid = int(gt_pid)
        agg[key]["proj"][gt_pid][0] += float(proj_sims[local_idx].item())
        agg[key]["proj"][gt_pid][1] += 1
        agg[key]["anchor"][gt_pid][0] += float(anchor_sims[local_idx].item())
        agg[key]["anchor"][gt_pid][1] += 1
        agg[key]["pseudo"][gt_pid][0] += float(pseudo_sims[local_idx].item())
        agg[key]["pseudo"][gt_pid][1] += 1


def mean_ranking_from_aggregate(
    d: Dict[int, List[float]],
    part_id_to_name: Dict[int, str],
    digits: int = 4,
) -> str:
    items = []
    for gt_pid, pair in d.items():
        s, c = pair
        if c <= 0:
            continue
        items.append((int(gt_pid), s / c))
    items.sort(key=lambda x: x[1], reverse=True)
    return " \\ ".join(f"{part_name(pid, part_id_to_name)}({val:.{digits}f})" for pid, val in items)


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


@torch.no_grad()
def compute_batch_rankings(
    *,
    pipeline: str,
    model: torch.nn.Module,
    criterion,
    batch: Dict,
    cat_id_to_name: Dict[int, str],
    part_id_to_name: Dict[int, str],
    printer: TeePrinter,
    aggregate: Dict,
    detail: bool,
    require_target_present: bool,
    digits: int,
):
    batch = move_batch_to_device(batch, next(model.parameters()).device)

    patch_tokens = safe_normalize(batch["patch_tokens"].float(), dim=-1)
    obj_mask_patch = batch["obj_mask_patch"].bool()
    part_gt_mask_patch = batch["part_gt_mask_patch"].bool()
    part_valid_mask = batch["part_valid_mask"].bool()
    part_category_id = batch["part_category_id"].long()
    part_text_feat = batch["part_text_feat"].float()
    category_id = batch["category_id"].long()

    if part_text_feat.shape[1] == 0 or not bool(part_valid_mask.any().item()):
        return

    # Use exactly the corresponding pipeline's part-anchor mask logic.
    if pipeline == "global":
        part_anchor_mask = criterion._build_part_anchor_mask(
            part_valid_mask=part_valid_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            obj_mask_patch=obj_mask_patch,
        )
    else:
        if getattr(criterion, "present_only_anchor", False):
            part_present_mask = (part_gt_mask_patch & obj_mask_patch[:, None, :]).sum(dim=-1) > 0
            part_anchor_mask = part_valid_mask & part_present_mask
        else:
            part_anchor_mask = part_valid_mask
        part_anchor_mask = part_anchor_mask & obj_mask_patch.any(dim=-1, keepdim=True)

    if not bool(part_anchor_mask.any().item()):
        return

    part_proj = model.project_clip_txt(part_text_feat)
    part_proj = safe_normalize(part_proj, dim=-1)

    abs_logits = torch.einsum("bkd,bnd->bkn", part_proj, patch_tokens)
    abs_logits = abs_logits / float(criterion.patch_temperature)
    abs_logits = abs_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

    if pipeline == "joint":
        z_pseudo, _, _, anchor_tokens, anchor_valid = criterion._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_anchor_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=int(criterion.em_iters),
            return_anchor_tokens=True,
        )
    elif pipeline == "global":
        z_pseudo, _, _, anchor_tokens, anchor_valid = criterion._anchor_proto_em_pool(
            patch_tokens=patch_tokens,
            abs_logits=abs_logits,
            obj_mask_patch=obj_mask_patch,
            part_valid_mask=part_anchor_mask,
            part_gt_mask_patch=part_gt_mask_patch,
            num_iters=int(criterion.em_iters),
            return_anchor_tokens=True,
        )
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    B, K, _ = part_proj.shape
    for b in range(B):
        cat_id = int(category_id[b].item())
        valid_gt_slots, valid_gt_ids, gt_proto_mat = build_gt_proto_for_one_unit(
            patch_tokens_b=patch_tokens[b],
            obj_mask_b=obj_mask_patch[b],
            part_gt_mask_b=part_gt_mask_patch[b],
            part_valid_b=part_valid_mask[b],
            part_ids_b=part_category_id[b],
        )
        if len(valid_gt_ids) == 0:
            continue

        valid_gt_slot_set = set(int(x) for x in valid_gt_slots)

        if detail:
            meta = batch.get("metadata", [{}])[b] if isinstance(batch.get("metadata", None), list) else {}
            ann_id = meta.get("annotation_id", "NA") if isinstance(meta, dict) else "NA"
            img_id = meta.get("image_id", "NA") if isinstance(meta, dict) else "NA"
            printer.print("=" * 180)
            printer.print(
                f"[unit] pipeline={pipeline} cat={cat_id}({cat_name(cat_id, cat_id_to_name)}) "
                f"ann_id={ann_id} image_id={img_id} valid_gt_parts={len(valid_gt_ids)}"
            )
            printer.print(f"{'part name':<42}\t{'projected_text_vs_GT':<40}\t{'anchor_patch_vs_GT':<40}\t{'pseudo_label_vs_GT'}")
            printer.print("-" * 180)

        for k in range(K):
            if not bool(anchor_valid[b, k].item()):
                continue
            if not bool(part_anchor_mask[b, k].item()):
                continue
            if require_target_present and k not in valid_gt_slot_set:
                continue

            target_pid = int(part_category_id[b, k].item())
            if target_pid < 0:
                continue

            proj_vec = part_proj[b, k]
            anchor_vec = anchor_tokens[b, k]
            pseudo_vec = z_pseudo[b, k]

            proj_sims = safe_normalize(proj_vec.float(), dim=-1) @ safe_normalize(gt_proto_mat.float(), dim=-1).T
            anchor_sims = safe_normalize(anchor_vec.float(), dim=-1) @ safe_normalize(gt_proto_mat.float(), dim=-1).T
            pseudo_sims = safe_normalize(pseudo_vec.float(), dim=-1) @ safe_normalize(gt_proto_mat.float(), dim=-1).T

            if detail:
                proj_rank = ranking_string(proj_vec, gt_proto_mat, valid_gt_ids, part_id_to_name, digits=digits)
                anchor_rank = ranking_string(anchor_vec, gt_proto_mat, valid_gt_ids, part_id_to_name, digits=digits)
                pseudo_rank = ranking_string(pseudo_vec, gt_proto_mat, valid_gt_ids, part_id_to_name, digits=digits)
                printer.print(
                    f"{part_name(target_pid, part_id_to_name):<42}\t{proj_rank}\t{anchor_rank}\t{pseudo_rank}"
                )
            else:
                update_aggregate(
                    aggregate,
                    key=(cat_id, target_pid),
                    gt_part_ids=valid_gt_ids,
                    proj_sims=proj_sims,
                    anchor_sims=anchor_sims,
                    pseudo_sims=pseudo_sims,
                )


def print_aggregate(
    aggregate: Dict,
    cat_id_to_name: Dict[int, str],
    part_id_to_name: Dict[int, str],
    printer: TeePrinter,
    digits: int,
):
    by_cat = defaultdict(list)
    for (cat_id, target_pid), vals in aggregate.items():
        by_cat[int(cat_id)].append((int(target_pid), vals))

    printer.print("=" * 180)
    printer.print("[AGGREGATED MEAN RANKING]")
    printer.print("Each cosine is averaged over all audited units before sorting.")
    printer.print("Columns: part name | projected text feature vs GT prototypes | anchor patch vs GT prototypes | pseudo label prototype vs GT prototypes")
    printer.print("=" * 180)

    for cat_id in sorted(by_cat.keys()):
        rows = sorted(by_cat[cat_id], key=lambda x: part_name(x[0], part_id_to_name))
        printer.print("")
        printer.print("=" * 180)
        printer.print(f"[object] cat={cat_id}  name={cat_name(cat_id, cat_id_to_name)}  num_target_parts={len(rows)}")
        printer.print(f"{'part name':<42}\t{'projected_text_vs_GT':<40}\t{'anchor_patch_vs_GT':<40}\t{'pseudo_label_vs_GT'}")
        printer.print("-" * 180)
        for target_pid, vals in rows:
            proj_rank = mean_ranking_from_aggregate(vals["proj"], part_id_to_name, digits=digits)
            anchor_rank = mean_ranking_from_aggregate(vals["anchor"], part_id_to_name, digits=digits)
            pseudo_rank = mean_ranking_from_aggregate(vals["pseudo"], part_id_to_name, digits=digits)
            printer.print(
                f"{part_name(target_pid, part_id_to_name):<42}\t{proj_rank}\t{anchor_rank}\t{pseudo_rank}"
            )


def make_default_aggregate():
    return defaultdict(lambda: {
        "proj": defaultdict(lambda: [0.0, 0]),
        "anchor": defaultdict(lambda: [0.0, 0]),
        "pseudo": defaultdict(lambda: [0.0, 0]),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=str, choices=["joint", "global"], required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="Dataset .pth/.tar to audit. For global, this builds the category patch pool from this split.")
    parser.add_argument("--out_txt", type=str, required=True)

    parser.add_argument("--obj_feature_name", type=str, default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", type=str, default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", type=str, default="ann_feats")
    parser.add_argument("--part_text_name", type=str, default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--batch_size", type=int, default=1, help="Only used for joint pipeline. Keep 1 for clean per-instance diagnosis.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_units", type=int, default=-1, help="Limit audited image instances / category pools. -1 means all.")
    parser.add_argument("--detail", action="store_true", help="Print every image/pool unit. Default prints aggregate mean per object part.")
    parser.add_argument("--include_absent_targets", action="store_true", help="By default, target part must have a GT mask in the current unit.")
    parser.add_argument("--global_sample_patches", type=int, default=0, help="For global pipeline: 0 means use the full category pool; otherwise subsample this many patches per category.")
    parser.add_argument("--digits", type=int, default=4)

    args = parser.parse_args()
    device = torch.device(args.device)
    config = load_config(args.model_config)
    dataset_cfg = config.get("dataset", {})
    min_obj_area_ratio = float(dataset_cfg.get("min_obj_area_ratio", 0.0))

    printer = TeePrinter(args.out_txt)
    old_stdout = sys.stdout
    sys.stdout = printer
    try:
        printer.print(f"[audit] pipeline={args.pipeline}")
        printer.print(f"[audit] model_config={args.model_config}")
        printer.print(f"[audit] checkpoint={args.checkpoint}")
        printer.print(f"[audit] dataset={args.dataset}")
        printer.print(f"[audit] out_txt={args.out_txt}")
        printer.print(f"[audit] require_target_present={not args.include_absent_targets}")
        printer.print("")

        model = load_projector(config, args.checkpoint, device)

        is_wds = ".tar" in args.dataset
        joint_dataset = DinoClipJointDataset(
            args.dataset,
            obj_feature_name=args.obj_feature_name,
            part_feature_name=args.part_feature_name,
            obj_text_name=args.obj_text_name,
            part_text_name=args.part_text_name,
            resize_dim=args.resize_dim,
            crop_dim=args.crop_dim,
            patch_size=args.patch_size,
            with_background=args.with_background,
            is_wds=is_wds,
            path_prefix=args.path_prefix,
            min_obj_area_ratio=min_obj_area_ratio,
        )
        cat_id_to_name, part_id_to_name = build_name_maps(joint_dataset)
        printer.print(f"[dataset] taxonomy={getattr(joint_dataset, 'part_taxonomy', 'unknown')}")
        printer.print(f"[dataset] num_samples={len(joint_dataset)}")
        printer.print(f"[dataset] num_object_classes={len(getattr(joint_dataset, 'class_part_bank', {}))}")
        printer.print("")

        aggregate = make_default_aggregate()

        if args.pipeline == "joint":
            criterion = make_joint_criterion(config, model).to(device)
            loader = DataLoader(
                joint_dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                collate_fn=joint_collate_fn,
            )
            for unit_idx, batch in enumerate(loader):
                if args.max_units >= 0 and unit_idx >= args.max_units:
                    break
                compute_batch_rankings(
                    pipeline="joint",
                    model=model,
                    criterion=criterion,
                    batch=batch,
                    cat_id_to_name=cat_id_to_name,
                    part_id_to_name=part_id_to_name,
                    printer=printer,
                    aggregate=aggregate,
                    detail=bool(args.detail),
                    require_target_present=not bool(args.include_absent_targets),
                    digits=int(args.digits),
                )

        else:
            from src.dataset_global import CategoryPatchPoolDataset, global_pool_collate_fn

            criterion = make_global_criterion(config, model).to(device)
            sample_patches = None if int(args.global_sample_patches) <= 0 else int(args.global_sample_patches)
            pool_dataset = CategoryPatchPoolDataset(
                joint_dataset,
                sample_patches_per_step=sample_patches,
                steps_per_epoch=None,
                store_dtype=torch.float16,
                seed=123,
                fixed_subsample=True,
            )
            # Use DataLoader for the same [B, ...] format as training.
            loader = DataLoader(
                pool_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=global_pool_collate_fn,
            )
            for unit_idx, batch in enumerate(loader):
                if args.max_units >= 0 and unit_idx >= args.max_units:
                    break
                compute_batch_rankings(
                    pipeline="global",
                    model=model,
                    criterion=criterion,
                    batch=batch,
                    cat_id_to_name=cat_id_to_name,
                    part_id_to_name=part_id_to_name,
                    printer=printer,
                    aggregate=aggregate,
                    detail=bool(args.detail),
                    require_target_present=not bool(args.include_absent_targets),
                    digits=int(args.digits),
                )

        if not args.detail:
            print_aggregate(
                aggregate=aggregate,
                cat_id_to_name=cat_id_to_name,
                part_id_to_name=part_id_to_name,
                printer=printer,
                digits=int(args.digits),
            )

        printer.print("")
        printer.print(f"[done] saved printed output to: {args.out_txt}")
    finally:
        sys.stdout = old_stdout
        printer.close()


if __name__ == "__main__":
    main()
