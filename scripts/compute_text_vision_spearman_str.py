#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from scripts.anlysis import FeatureAnalyser
from scripts.anlysis import mean_features_by_part

try:
    # Prefer existing project implementation if available.
    from utils.metric import structure_retrieval as imported_structure_retrieval
except Exception:
    imported_structure_retrieval = None


VOC_OBJECTS = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def resolve_part_names(num_parts: int, analyser_part_names):
    """
    Use repo-defined class-name list when available.
    This fixes coarse59, because older scripts/anlysis.py may only know 58/116.
    """
    try:
        from src.voc116_part_coarse import COARSE_PART_CLASSES, FINE_PART_CLASSES
        if num_parts == len(COARSE_PART_CLASSES):
            return list(COARSE_PART_CLASSES)
        if num_parts == len(FINE_PART_CLASSES):
            return list(FINE_PART_CLASSES)
    except Exception:
        pass

    if analyser_part_names is not None and len(analyser_part_names) == num_parts:
        return list(analyser_part_names)

    return [f"part_{i}" for i in range(num_parts)]


def structure_retrieval_legacy(feat_1, feat_2, ret_sim=False, use_HM=False, ret_idx=False):
    """
    Legacy STR implementation from the user's previous pipeline.

    Args:
        feat_1: [N_cls, D2]
        feat_2: [N_cls, D1]
        D1 and D2 do not have to be equal.
    """
    if imported_structure_retrieval is not None:
        return imported_structure_retrieval(
            feat_1,
            feat_2,
            ret_sim=ret_sim,
            use_HM=use_HM,
            ret_idx=ret_idx,
        )

    feat_1 = feat_1.float().cpu()
    feat_2 = feat_2.float().cpu()

    feat_1_ = feat_1 / feat_1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    feat_2_ = feat_2 / feat_2.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    sim_1 = feat_1_ @ feat_1_.transpose(1, 0)
    sim_2 = feat_2_ @ feat_2_.transpose(1, 0)

    N = feat_1_.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=sim_1.device)

    sim_1 = sim_1[~eye].view(N, -1)
    sim_2 = sim_2[~eye].view(N, -1)

    sim_1 = sim_1 - sim_1.mean(-1).unsqueeze(1)
    sim_2 = sim_2 - sim_2.mean(-1).unsqueeze(1)

    sim_1_norm = sim_1 / sim_1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sim_2_norm = sim_2 / sim_2.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    sim_1_2 = sim_1_norm @ sim_2_norm.transpose(1, 0)

    if not use_HM:
        idx = sim_1_2.argmax(0).numpy()
        ret_flag = idx == np.arange(len(sim_1_2))
        retrieval_structure = ret_flag.sum() / len(sim_1_2)

        if ret_idx:
            return retrieval_structure, np.where(ret_flag)[0]
        if ret_sim:
            return sim_1_2
        return retrieval_structure

    row_ind, col_ind = linear_sum_assignment(1.0 - sim_1_2.cpu().numpy())

    # Match the old `linear_assignment` return convention m[1].
    order = np.argsort(row_ind)
    matched_cols = col_ind[order]

    retrieval_ratio_HM = (matched_cols == np.arange(len(sim_1_2))).sum() / len(sim_1_2)
    if ret_idx:
        return retrieval_ratio_HM, matched_cols
    return retrieval_ratio_HM


def object_to_indices(part_names):
    obj_to_part_indices = {obj: [] for obj in VOC_OBJECTS}
    for idx, name in enumerate(part_names):
        if "'s" not in name:
            continue
        obj_name = name.split("'s")[0]
        if obj_name in obj_to_part_indices:
            obj_to_part_indices[obj_name].append(idx)
    return obj_to_part_indices


def prep_feature_for_structure(x):
    if torch.is_tensor(x) and x.dim() == 3 and x.shape[1] == 1:
        x = x.squeeze(1)
    x = x.detach().cpu().float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = F.normalize(x, dim=-1, eps=1e-6)
    return x


def print_spearman_per_object(feat_vision, feat_text, part_names, feat1_name, feat2_name, min_parts=3):
    V = prep_feature_for_structure(feat_vision)
    T = prep_feature_for_structure(feat_text)

    obj_to_part_indices = object_to_indices(part_names)

    print(f"feat1_name: {feat1_name}")
    print(f"feat2_name: {feat2_name}")
    print(f"feat1 shape: {tuple(V.shape)}")
    print(f"feat2 shape: {tuple(T.shape)}")
    print("")
    print("==================== Spearman(T vs V) per Object ====================")

    rhos = []
    for obj in VOC_OBJECTS:
        indices = obj_to_part_indices.get(obj, [])
        P = len(indices)

        if P < min_parts:
            print(f"{obj:<20} | rho=nan  (parts={P})")
            continue

        subT = T[indices]
        subV = V[indices]

        valid = (
            torch.isfinite(subT).all(dim=-1)
            & torch.isfinite(subV).all(dim=-1)
            & (subT.norm(dim=-1) > 1e-8)
            & (subV.norm(dim=-1) > 1e-8)
        )
        subT = subT[valid]
        subV = subV[valid]
        P_valid = subT.shape[0]

        if P_valid < min_parts:
            print(f"{obj:<20} | rho=nan  (parts={P_valid})")
            continue

        simT = (subT @ subT.T).numpy()
        simV = (subV @ subV.T).numpy()

        iu = np.triu_indices(P_valid, k=1)
        rho = spearmanr(simT[iu], simV[iu]).correlation

        if np.isnan(rho):
            print(f"{obj:<20} | rho=nan  (parts={P_valid})")
        else:
            rhos.append(rho)
            print(f"{obj:<20} | rho={rho:.6f}  (parts={P_valid})")

    mean_rho = float(np.mean(rhos)) if len(rhos) > 0 else float("nan")
    print("------------------------------------------------------------")
    print(f"{'Average (unweighted)':<20} | rho={mean_rho:.6f}")


def print_local_structure_retrieval(feat_text, feat_vision, part_names, feat_text_name, feat_vision_name, min_parts=3):
    T = prep_feature_for_structure(feat_text)
    V = prep_feature_for_structure(feat_vision)

    obj_to_part_indices = object_to_indices(part_names)

    avg_str_t2v = 0.0
    avg_str_HM_t2v = 0.0
    avg_str_v2t = 0.0
    avg_str_HM_v2t = 0.0
    valid_obj_count = 0

    print("")
    print(f"feat_text_name: {feat_text_name}")
    print(f"feat_vision_name: {feat_vision_name}")
    print("")

    for obj_name in VOC_OBJECTS:
        indices = obj_to_part_indices.get(obj_name, [])

        if len(indices) < min_parts:
            continue

        sub_feat_text = T[indices].float().cpu()
        sub_feat_vision = V[indices].float().cpu()

        valid = (
            torch.isfinite(sub_feat_text).all(dim=-1)
            & torch.isfinite(sub_feat_vision).all(dim=-1)
            & (sub_feat_text.norm(dim=-1) > 1e-8)
            & (sub_feat_vision.norm(dim=-1) > 1e-8)
        )
        sub_feat_text = sub_feat_text[valid]
        sub_feat_vision = sub_feat_vision[valid]

        if sub_feat_text.shape[0] < min_parts:
            continue

        str_t2v = structure_retrieval_legacy(sub_feat_text, sub_feat_vision)
        str_HM_t2v = structure_retrieval_legacy(sub_feat_text, sub_feat_vision, use_HM=True)
        str_v2t = structure_retrieval_legacy(sub_feat_vision, sub_feat_text)
        str_HM_v2t = structure_retrieval_legacy(sub_feat_vision, sub_feat_text, use_HM=True)

        avg_str_t2v += str_t2v
        avg_str_HM_t2v += str_HM_t2v
        avg_str_v2t += str_v2t
        avg_str_HM_v2t += str_HM_v2t
        valid_obj_count += 1

        print(f"Object: {obj_name}")
        print(f"text vs vision: retrieval[str/str_HM]: {str_t2v:.4f}/{str_HM_t2v:.4f}")
        print(f"vision vs text: retrieval[str/str_HM]: {str_v2t:.4f}/{str_HM_v2t:.4f}")
        print("")

    if valid_obj_count == 0:
        print("Average text vs vision: retrieval[str/str_HM]: nan/nan")
        print("Average vision vs text: retrieval[str/str_HM]: nan/nan")
        return

    print(
        "Average text vs vision: retrieval[str/str_HM]: "
        f"{avg_str_t2v / valid_obj_count:.4f}/{avg_str_HM_t2v / valid_obj_count:.4f}"
    )
    print(
        "Average vision vs text: retrieval[str/str_HM]: "
        f"{avg_str_v2t / valid_obj_count:.4f}/{avg_str_HM_v2t / valid_obj_count:.4f}"
    )


def find_first_nonempty_dim(features_by_part, fallback=None):
    for x in features_by_part:
        if x is not None and x.shape[0] > 0:
            return int(x.shape[-1])
    if fallback is not None:
        return int(fallback)
    raise RuntimeError("Cannot infer feature dimension from empty feature list.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init_weights", required=True)

    parser.add_argument("--obj_feature_name", default="avg_self_attn_out")
    parser.add_argument("--part_feature_name", default="cropaug_patch_tokens")
    parser.add_argument("--obj_text_name", default="ann_feats")
    parser.add_argument("--part_text_name", default="part_ann_feats")

    parser.add_argument("--resize_dim", type=int, default=448)
    parser.add_argument("--crop_dim", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--num_parts", type=int, default=116)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show_progress", action="store_true")

    parser.add_argument("--feat_text_name", default="feat_clip_part")
    parser.add_argument("--feat_vision_name", default="feat_dinov2_part")
    parser.add_argument("--min_parts", type=int, default=3)

    args = parser.parse_args()

    analyser = FeatureAnalyser(
        model_config=args.model_config,
        dataset=args.dataset,
        init_weights=args.init_weights,
        obj_feature_name=args.obj_feature_name,
        part_feature_name=args.part_feature_name,
        obj_text_name=args.obj_text_name,
        part_text_name=args.part_text_name,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_parts=args.num_parts,
        device=args.device,
        show_progress=args.show_progress,
    )

    fake_features_by_part, gt_features_by_part = analyser.collect_vision_feature()

    (
        _obj_text_raw_by_category,
        _obj_text_proj_by_category,
        part_text_raw_by_part,
        part_text_proj_by_part,
    ) = analyser.collect_text_features()

    dino_dim = int(analyser.cfg["model"].get("dino_embed_dim"))
    raw_text_dim = find_first_nonempty_dim(part_text_raw_by_part)

    feat_pseudo_mean, _, _pseudo_count = mean_features_by_part(
        fake_features_by_part,
        dim=dino_dim,
    )

    feat_gt_mean, _, _gt_count = mean_features_by_part(
        gt_features_by_part,
        dim=dino_dim,
    )

    feat_text_raw_mean, _, _text_raw_count = mean_features_by_part(
        part_text_raw_by_part,
        dim=raw_text_dim,
    )

    feat_text_proj_mean, _, _text_proj_count = mean_features_by_part(
        part_text_proj_by_part,
        dim=dino_dim,
    )

    # Requested metric: raw text feature vs GT visual prototype.
    part_names = resolve_part_names(args.num_parts, analyser.part_names)

    print_spearman_per_object(
        feat_vision=feat_gt_mean,
        feat_text=feat_text_proj_mean,
        part_names=part_names,
        feat1_name=args.feat_vision_name,
        feat2_name=args.feat_text_name,
        min_parts=args.min_parts,
    )

    print_local_structure_retrieval(
        feat_text=feat_text_proj_mean,
        feat_vision=feat_gt_mean,
        part_names=part_names,
        feat_text_name=args.feat_text_name,
        feat_vision_name=args.feat_vision_name,
        min_parts=args.min_parts,
    )


if __name__ == "__main__":
    main()
