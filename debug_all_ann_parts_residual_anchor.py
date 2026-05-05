from pathlib import Path
import argparse
import importlib
import json
from typing import Dict, List

import torch
import yaml

from src.dataset_joint_with_part_anchoraudit_residual_anchor import DinoClipJointDataset, joint_collate_fn
from src.loss_joint_residual_anchor import JointObjPartLoss


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def build_model(model_config: str, weights: str, device: torch.device):
    with open(model_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)
    model = ModelClass.from_config(config["model"]).to(device)

    if weights:
        ckpt = torch.load(weights, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model


@torch.no_grad()
def compute_debug_pseudolabels(loss_obj: JointObjPartLoss, batch: Dict) -> List[Dict]:
    device = next(loss_obj.sim_model.parameters()).device
    batch = move_batch_to_device(batch, device)

    patch_tokens = batch["patch_tokens"]
    obj_text_feat = batch["obj_text_feat"]
    part_text_feat = batch["part_text_feat"]
    obj_mask_patch = batch["obj_mask_patch"]
    part_valid_mask = batch["part_valid_mask"]
    part_gt_mask_patch = batch["part_gt_mask_patch"]
    part_category_id = batch["part_category_id"]
    metadata = batch["metadata"]

    obj_proj = loss_obj.sim_model.project_clip_txt(obj_text_feat.float())
    part_proj = loss_obj.sim_model.project_clip_txt(part_text_feat.float())

    obj_proj = loss_obj._safe_normalize(obj_proj, dim=-1)
    part_proj = loss_obj._safe_normalize(part_proj, dim=-1)
    patch_tokens = loss_obj._safe_normalize(patch_tokens.float(), dim=-1)

    obj_origin = obj_proj[:, None, :]
    text_residual = loss_obj._safe_normalize(part_proj - obj_origin, dim=-1)
    vision_residual = loss_obj._safe_normalize(patch_tokens - obj_origin, dim=-1)

    residual_logits = torch.einsum("bkd,bnd->bkn", text_residual, vision_residual) / loss_obj.patch_temperature
    residual_logits = residual_logits.masked_fill(~obj_mask_patch[:, None, :], -1e4)

    outputs: List[Dict] = []
    B = patch_tokens.shape[0]

    for b in range(B):
        valid_patch_mask = obj_mask_patch[b]
        valid_part_idx = torch.nonzero(part_valid_mask[b], as_tuple=False).squeeze(1)

        sample_out = {
            "annotation_id": metadata[b].get("annotation_id"),
            "image_id": metadata[b].get("image_id"),
            "class_name": metadata[b].get("class_name", ""),
            "part_class_name": metadata[b].get("part_class_name", []),
            "num_all_parts": int(valid_part_idx.numel()),
            "num_obj_patches": int(valid_patch_mask.sum().item()),
            "part_ids": part_category_id[b, valid_part_idx].detach().cpu().tolist() if valid_part_idx.numel() > 0 else [],
        }

        if valid_part_idx.numel() == 0 or valid_patch_mask.sum() == 0:
            sample_out.update({
                "anchor_patch_idx_global": [],
                "anchor_hit_vec": [],
                "anchor_hit_rate": 0.0,
                "patch_assignment_global": [],
                "patch_count_per_part": [],
            })
            outputs.append(sample_out)
            continue

        valid_patch_tokens = patch_tokens[b][valid_patch_mask]
        valid_patch_residual = vision_residual[b][valid_patch_mask]
        local_scores = residual_logits[b][valid_part_idx][:, valid_patch_mask]

        Kb, Mb = local_scores.shape
        rel_scores = loss_obj._compute_relative_scores(local_scores)

        flat_scores = rel_scores.reshape(-1)
        sorted_idx = torch.argsort(flat_scores, descending=True)

        anchor_idx_local = torch.full((Kb,), -1, dtype=torch.long, device=local_scores.device)
        patch_taken = torch.zeros((Mb,), dtype=torch.bool, device=local_scores.device)

        assigned_parts = 0
        for flat_id in sorted_idx:
            p_local = torch.div(flat_id, Mb, rounding_mode="floor")
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
            local_best = rel_scores.argmax(dim=1)
            anchor_idx_local[unassigned] = local_best[unassigned]

        valid_patch_idx_global = torch.nonzero(valid_patch_mask, as_tuple=False).squeeze(1)
        anchor_idx_global = valid_patch_idx_global[anchor_idx_local]

        gt_masks = part_gt_mask_patch[b, valid_part_idx]
        hit_vec = gt_masks[torch.arange(Kb, device=gt_masks.device), anchor_idx_global]
        anchor_hit_rate = float(hit_vec.float().mean().item()) if Kb > 0 else 0.0

        C = valid_patch_residual[anchor_idx_local]
        assign = torch.zeros((Mb,), dtype=torch.long, device=local_scores.device)

        for _ in range(max(int(loss_obj.em_iters), 1)):
            assign_scores = valid_patch_residual @ C.T
            assign = assign_scores.argmax(dim=1)
            assign[anchor_idx_local] = torch.arange(Kb, device=assign.device)

            onehot = torch.nn.functional.one_hot(assign, num_classes=Kb).float()
            count = onehot.sum(dim=0).clamp_min(1.0)
            proto_sum = onehot.T @ valid_patch_residual
            C = proto_sum / count[:, None]
            C = loss_obj._safe_normalize(C, dim=-1)

        patch_assignment_global = [-1] * int(obj_mask_patch.shape[1])
        for local_n, global_n in enumerate(valid_patch_idx_global.detach().cpu().tolist()):
            patch_assignment_global[global_n] = int(assign[local_n].detach().cpu().item())

        onehot = torch.nn.functional.one_hot(assign, num_classes=Kb).float()
        patch_count_per_part = onehot.sum(dim=0).detach().cpu().int().tolist()

        sample_out.update({
            "anchor_patch_idx_global": anchor_idx_global.detach().cpu().tolist(),
            "anchor_hit_vec": hit_vec.detach().cpu().int().tolist(),
            "anchor_hit_rate": anchor_hit_rate,
            "patch_assignment_global": patch_assignment_global,
            "patch_count_per_part": patch_count_per_part,
        })
        outputs.append(sample_out)

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--obj_feature_name", type=str, default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", type=str, default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", type=str, default="ann_feats")
    parser.add_argument("--part_text_name", type=str, default="part_ann_feats")
    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--with_background", action="store_true", default=False)
    parser.add_argument("--path_prefix", type=str, default=None)
    parser.add_argument("--min_obj_area_ratio", type=float, default=0.0)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_wds = ".tar" in args.dataset_path
    dataset = DinoClipJointDataset(
        args.dataset_path,
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
        min_obj_area_ratio=args.min_obj_area_ratio,
    )

    model = build_model(args.model_config, args.weights, device)
    loss_obj = JointObjPartLoss(model).to(device).eval()

    start = max(int(args.sample_idx), 0)
    end = min(start + max(int(args.num_samples), 1), len(dataset))
    samples = [dataset[i] for i in range(start, end)]
    batch = joint_collate_fn(samples)

    outputs = compute_debug_pseudolabels(loss_obj, batch)
    result = {
        "sample_idx_start": start,
        "sample_idx_end_exclusive": end,
        "num_samples": len(outputs),
        "outputs": outputs,
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
