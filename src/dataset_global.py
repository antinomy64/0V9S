# src/dataset_global.py

from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset


class CategoryPatchPoolDataset(Dataset):
    """
    Build one global patch-token pool per object category.

    This version supports dual V-side tokens:
      - patch_tokens_select: used by loss_global.py for anchor / EM / assignment.
      - patch_tokens_target: used by loss_global.py for the final pseudo-prototype target.

    Backward compatibility:
      - If select_joint_dataset is None, select == target.
      - The old key "patch_tokens" is kept as an alias of patch_tokens_target.

    Each item is one object category:
        patch_tokens_select: [M, D]
        patch_tokens_target: [M, D]
        patch_tokens:        [M, D] alias of target
        obj_mask_patch:      [M] all True
        part_gt_mask_patch:  [K, M]
        part_text_feat:      [K, Dt]
        obj_text_feat:       [Dt]
        part_valid_mask:     [K]
    """

    def __init__(
        self,
        joint_dataset,
        select_joint_dataset=None,
        sample_patches_per_step: Optional[int] = 65536,
        steps_per_epoch: Optional[int] = None,
        store_dtype: torch.dtype = torch.float16,
        seed: int = 123,
        fixed_subsample: bool = False,
    ):
        if sample_patches_per_step is not None and int(sample_patches_per_step) <= 0:
            raise ValueError("sample_patches_per_step must be positive or None.")
        if steps_per_epoch is not None and int(steps_per_epoch) <= 0:
            raise ValueError("steps_per_epoch must be positive or None.")

        self.sample_patches_per_step = sample_patches_per_step
        self.steps_per_epoch = steps_per_epoch
        self.store_dtype = store_dtype
        self.seed = int(seed)
        self.fixed_subsample = bool(fixed_subsample)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.pools: Dict[int, Dict] = {}
        self.categories: List[int] = []
        self.fixed_indices: Dict[int, torch.Tensor] = {}

        self._build_pools(joint_dataset, select_joint_dataset=select_joint_dataset)
        self._build_fixed_indices()

    @staticmethod
    def _dataset_samples(joint_dataset):
        return (
            list(joint_dataset.data.values())
            if isinstance(joint_dataset.data, dict)
            else list(joint_dataset.data)
        )

    @staticmethod
    def _same_part_order(sample_a: Dict, sample_b: Dict) -> bool:
        ids_a = sample_a["part_category_id"].long()
        ids_b = sample_b["part_category_id"].long()
        return torch.equal(ids_a, ids_b)

    def _build_pools(self, joint_dataset, select_joint_dataset=None):
        tmp = {}

        target_samples = self._dataset_samples(joint_dataset)
        if select_joint_dataset is None:
            select_samples = target_samples
            using_dual = False
        else:
            select_samples = self._dataset_samples(select_joint_dataset)
            using_dual = True
            if len(select_samples) != len(target_samples):
                raise ValueError(
                    "target/select datasets must contain the same number of samples: "
                    f"target={len(target_samples)}, select={len(select_samples)}"
                )

        for sample_idx, (target_sample, select_sample) in enumerate(zip(target_samples, select_samples)):
            cat = int(target_sample["category_id"])
            select_cat = int(select_sample["category_id"])
            if select_cat != cat:
                raise ValueError(
                    f"target/select category mismatch at sample {sample_idx}: "
                    f"target={cat}, select={select_cat}"
                )
            if not self._same_part_order(target_sample, select_sample):
                raise ValueError(
                    f"target/select part_category_id order mismatch at sample {sample_idx}, "
                    f"category={cat}"
                )

            patch_tokens_target = target_sample["patch_tokens"].float()  # [N, D]
            patch_tokens_select = select_sample["patch_tokens"].float()  # [N, D]
            obj_mask = target_sample["obj_mask_patch"].bool()            # [N]
            part_gt = target_sample["part_gt_mask_patch"].bool()         # [K, N]

            if patch_tokens_target.ndim != 2:
                raise ValueError(
                    f"Expected target patch_tokens [N, D], got {tuple(patch_tokens_target.shape)}"
                )
            if patch_tokens_select.ndim != 2:
                raise ValueError(
                    f"Expected select patch_tokens [N, D], got {tuple(patch_tokens_select.shape)}"
                )
            if patch_tokens_select.shape != patch_tokens_target.shape:
                raise ValueError(
                    f"select/target patch token shape mismatch at sample {sample_idx}: "
                    f"select={tuple(patch_tokens_select.shape)}, "
                    f"target={tuple(patch_tokens_target.shape)}"
                )
            if obj_mask.ndim != 1 or obj_mask.shape[0] != patch_tokens_target.shape[0]:
                raise ValueError(
                    f"obj_mask_patch shape {tuple(obj_mask.shape)} does not match "
                    f"patch_tokens shape {tuple(patch_tokens_target.shape)}"
                )
            if part_gt.ndim != 2 or part_gt.shape[1] != patch_tokens_target.shape[0]:
                raise ValueError(
                    f"part_gt_mask_patch shape {tuple(part_gt.shape)} does not match "
                    f"patch_tokens shape {tuple(patch_tokens_target.shape)}"
                )

            if obj_mask.sum().item() == 0:
                continue

            # Keep only object-mask patches in the global pool.
            patch_obj_select = patch_tokens_select[obj_mask].to(self.store_dtype)  # [Mi, D]
            patch_obj_target = patch_tokens_target[obj_mask].to(self.store_dtype)  # [Mi, D]
            part_gt_obj = part_gt[:, obj_mask].contiguous()                        # [K, Mi]

            if cat not in tmp:
                tmp[cat] = {
                    "patch_select_chunks": [],
                    "patch_target_chunks": [],
                    "part_gt_chunks": [],
                    "obj_text_feats": [],
                    "part_text_feat": target_sample["part_text_feat"].float().clone(),
                    "part_category_id": target_sample["part_category_id"].long().clone(),
                    "part_class_name": target_sample.get("part_class_name", []),
                }
            else:
                ref_ids = tmp[cat]["part_category_id"]
                cur_ids = target_sample["part_category_id"].long()
                if not torch.equal(ref_ids, cur_ids):
                    raise ValueError(
                        f"Inconsistent part_category_id order for category {cat}: "
                        f"expected {ref_ids.tolist()}, got {cur_ids.tolist()}"
                    )
                if int(part_gt_obj.shape[0]) != int(ref_ids.numel()):
                    raise ValueError(
                        f"part_gt rows ({part_gt_obj.shape[0]}) do not match "
                        f"part bank size ({ref_ids.numel()}) for category {cat}"
                    )

            tmp[cat]["patch_select_chunks"].append(patch_obj_select.cpu())
            tmp[cat]["patch_target_chunks"].append(patch_obj_target.cpu())
            tmp[cat]["part_gt_chunks"].append(part_gt_obj.cpu())
            tmp[cat]["obj_text_feats"].append(target_sample["obj_text_feat"].float().cpu())

        self.pools = {}
        for cat, item in tmp.items():
            patch_select_pool = torch.cat(item["patch_select_chunks"], dim=0)  # [M, D]
            patch_target_pool = torch.cat(item["patch_target_chunks"], dim=0)  # [M, D]
            part_gt_pool = torch.cat(item["part_gt_chunks"], dim=1)            # [K, M]
            obj_text_feat = torch.stack(item["obj_text_feats"], dim=0).mean(dim=0)

            K = item["part_text_feat"].shape[0]
            if K == 0:
                continue

            self.pools[cat] = {
                "category_id": torch.tensor(cat, dtype=torch.long),
                "patch_tokens_select": patch_select_pool,
                "patch_tokens_target": patch_target_pool,
                "patch_tokens": patch_target_pool,  # backward-compatible alias: target/raw tokens
                "obj_text_feat": obj_text_feat,
                "part_text_feat": item["part_text_feat"],
                "obj_mask_patch": torch.ones(patch_target_pool.shape[0], dtype=torch.bool),
                "part_gt_mask_patch": part_gt_pool,
                "part_valid_mask": torch.ones(K, dtype=torch.bool),
                "part_category_id": item["part_category_id"],
                "metadata": {
                    "category_id": cat,
                    "num_patches": int(patch_target_pool.shape[0]),
                    "num_parts": int(K),
                    "dual_select_target": bool(using_dual),
                },
            }

        self.categories = sorted(self.pools.keys())
        if len(self.categories) == 0:
            raise RuntimeError("No non-empty category patch pools were built.")

        mode = "dual select/target" if using_dual else "single target-as-select"
        print(f"[global pool] built {len(self.categories)} category pools ({mode})")
        for cat in self.categories:
            p = self.pools[cat]
            print(
                f"[global pool] cat={cat}, "
                f"patches={p['patch_tokens_target'].shape[0]}, "
                f"parts={p['part_text_feat'].shape[0]}"
            )

    def _build_fixed_indices(self) -> None:
        """Precompute deterministic subsets for validation-style usage."""
        if not self.fixed_subsample or self.sample_patches_per_step is None:
            return

        for cat in self.categories:
            M = int(self.pools[cat]["patch_tokens_target"].shape[0])
            if M > int(self.sample_patches_per_step):
                self.fixed_indices[cat] = torch.randperm(
                    M,
                    generator=self.generator,
                )[: int(self.sample_patches_per_step)]

    def __len__(self):
        if self.steps_per_epoch is not None:
            return int(self.steps_per_epoch)
        return len(self.categories)

    def __getitem__(self, idx: int):
        cat = self.categories[idx % len(self.categories)]
        item = self.pools[cat]

        patch_tokens_select = item["patch_tokens_select"]
        patch_tokens_target = item["patch_tokens_target"]
        part_gt_mask_patch = item["part_gt_mask_patch"]
        M = patch_tokens_target.shape[0]

        if self.sample_patches_per_step is not None and M > self.sample_patches_per_step:
            if self.fixed_subsample:
                perm = self.fixed_indices[cat]
            else:
                perm = torch.randperm(M, generator=self.generator)[: self.sample_patches_per_step]
            patch_tokens_select = patch_tokens_select[perm]
            patch_tokens_target = patch_tokens_target[perm]
            part_gt_mask_patch = part_gt_mask_patch[:, perm]

        return {
            "category_id": item["category_id"],
            "patch_tokens_select": patch_tokens_select,
            "patch_tokens_target": patch_tokens_target,
            "patch_tokens": patch_tokens_target,  # backward-compatible alias
            "obj_text_feat": item["obj_text_feat"],
            "part_text_feat": item["part_text_feat"],
            "obj_mask_patch": torch.ones(patch_tokens_target.shape[0], dtype=torch.bool),
            "part_gt_mask_patch": part_gt_mask_patch,
            "part_valid_mask": item["part_valid_mask"],
            "part_category_id": item["part_category_id"],
            "metadata": item["metadata"],
        }


def global_pool_collate_fn(batch: List[Dict]) -> Dict:
    """
    Global pool training uses batch_size=1.
    Convert category-level tensors into the [B, ...] format expected by PartLoss.
    """
    assert len(batch) == 1, "Use batch_size=1 for global category patch pool training."

    b = batch[0]
    return {
        "patch_tokens_select": b["patch_tokens_select"].float().unsqueeze(0),        # [1, M, D]
        "patch_tokens_target": b["patch_tokens_target"].float().unsqueeze(0),        # [1, M, D]
        "patch_tokens": b["patch_tokens"].float().unsqueeze(0),                      # [1, M, D], alias target
        "obj_text_feat": b["obj_text_feat"].float().unsqueeze(0),                    # [1, Dt]
        "part_text_feat": b["part_text_feat"].float().unsqueeze(0),                  # [1, K, Dt]
        "obj_mask_patch": b["obj_mask_patch"].bool().unsqueeze(0),                   # [1, M]
        "part_gt_mask_patch": b["part_gt_mask_patch"].bool().unsqueeze(0),           # [1, K, M]
        "part_valid_mask": b["part_valid_mask"].bool().unsqueeze(0),                 # [1, K]
        "category_id": b["category_id"].view(1),
        "part_category_id": b["part_category_id"].long().unsqueeze(0),               # [1, K]
        "metadata": [b["metadata"]],
    }
