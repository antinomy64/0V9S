# src/dataset_global_pool.py

from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset


class CategoryPatchPoolDataset(Dataset):
    """
    Build one global patch-token pool per object category.

    Each item is one object category:
        patch_tokens:        [M, D]
        obj_mask_patch:      [M] all True
        part_gt_mask_patch:  [K, M]
        part_text_feat:      [K, Dt]
        obj_text_feat:       [Dt]
        part_valid_mask:     [K]
    """

    def __init__(
        self,
        joint_dataset,
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

        # Use an independent torch.Generator so pool sampling is reproducible
        # and does not consume the global RNG used by model training.
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.pools: Dict[int, Dict] = {}
        self.categories: List[int] = []
        self.fixed_indices: Dict[int, torch.Tensor] = {}

        self._build_pools(joint_dataset)
        self._build_fixed_indices()

    def _build_pools(self, joint_dataset):
        tmp = {}

        # joint_dataset.data 是你 DinoClipJointDataset 构造好的 sample dict
        samples = joint_dataset.data.values() if isinstance(joint_dataset.data, dict) else joint_dataset.data

        for sample in samples:
            cat = int(sample["category_id"])

            patch_tokens = sample["patch_tokens"].float()          # [N, D]
            obj_mask = sample["obj_mask_patch"].bool()             # [N]
            part_gt = sample["part_gt_mask_patch"].bool()          # [K, N]

            if patch_tokens.ndim != 2:
                raise ValueError(f"Expected patch_tokens [N, D], got {tuple(patch_tokens.shape)}")
            if obj_mask.ndim != 1 or obj_mask.shape[0] != patch_tokens.shape[0]:
                raise ValueError(
                    f"obj_mask_patch shape {tuple(obj_mask.shape)} does not match "
                    f"patch_tokens shape {tuple(patch_tokens.shape)}"
                )
            if part_gt.ndim != 2 or part_gt.shape[1] != patch_tokens.shape[0]:
                raise ValueError(
                    f"part_gt_mask_patch shape {tuple(part_gt.shape)} does not match "
                    f"patch_tokens shape {tuple(patch_tokens.shape)}"
                )

            if obj_mask.sum().item() == 0:
                continue

            # 只保留 object 内 patch，避免背景进入全局池
            patch_obj = patch_tokens[obj_mask].to(self.store_dtype)        # [Mi, D]
            part_gt_obj = part_gt[:, obj_mask].contiguous()                # [K, Mi]

            if cat not in tmp:
                tmp[cat] = {
                    "patch_chunks": [],
                    "part_gt_chunks": [],
                    "obj_text_feats": [],
                    "part_text_feat": sample["part_text_feat"].float().clone(),
                    "part_category_id": sample["part_category_id"].long().clone(),
                    "part_class_name": sample.get("part_class_name", []),
                }
            else:
                ref_ids = tmp[cat]["part_category_id"]
                cur_ids = sample["part_category_id"].long()
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

            tmp[cat]["patch_chunks"].append(patch_obj.cpu())
            tmp[cat]["part_gt_chunks"].append(part_gt_obj.cpu())
            tmp[cat]["obj_text_feats"].append(sample["obj_text_feat"].float().cpu())

        self.pools = {}
        for cat, item in tmp.items():
            patch_pool = torch.cat(item["patch_chunks"], dim=0)          # [M, D]
            part_gt_pool = torch.cat(item["part_gt_chunks"], dim=1)      # [K, M]
            obj_text_feat = torch.stack(item["obj_text_feats"], dim=0).mean(dim=0)

            K = item["part_text_feat"].shape[0]
            if K == 0:
                continue

            self.pools[cat] = {
                "category_id": torch.tensor(cat, dtype=torch.long),
                "patch_tokens": patch_pool,
                "obj_text_feat": obj_text_feat,
                "part_text_feat": item["part_text_feat"],
                "obj_mask_patch": torch.ones(patch_pool.shape[0], dtype=torch.bool),
                "part_gt_mask_patch": part_gt_pool,
                "part_valid_mask": torch.ones(K, dtype=torch.bool),
                "part_category_id": item["part_category_id"],
                "metadata": {
                    "category_id": cat,
                    "num_patches": int(patch_pool.shape[0]),
                    "num_parts": int(K),
                },
            }

        self.categories = sorted(self.pools.keys())
        if len(self.categories) == 0:
            raise RuntimeError("No non-empty category patch pools were built.")

        print(f"[global pool] built {len(self.categories)} category pools")
        for cat in self.categories:
            p = self.pools[cat]
            print(
                f"[global pool] cat={cat}, "
                f"patches={p['patch_tokens'].shape[0]}, "
                f"parts={p['part_text_feat'].shape[0]}"
            )

    def _build_fixed_indices(self) -> None:
        """Precompute deterministic subsets for validation-style usage."""
        if not self.fixed_subsample or self.sample_patches_per_step is None:
            return

        for cat in self.categories:
            M = int(self.pools[cat]["patch_tokens"].shape[0])
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
        # 如果 steps_per_epoch > 类别数，就循环类别
        cat = self.categories[idx % len(self.categories)]
        item = self.pools[cat]

        patch_tokens = item["patch_tokens"]
        part_gt_mask_patch = item["part_gt_mask_patch"]
        M = patch_tokens.shape[0]

        # 每次 step 从类别全局池里采样一部分 patch，避免显存爆炸
        if self.sample_patches_per_step is not None and M > self.sample_patches_per_step:
            if self.fixed_subsample:
                perm = self.fixed_indices[cat]
            else:
                perm = torch.randperm(M, generator=self.generator)[: self.sample_patches_per_step]
            patch_tokens = patch_tokens[perm]
            part_gt_mask_patch = part_gt_mask_patch[:, perm]

        return {
            "category_id": item["category_id"],
            "patch_tokens": patch_tokens,
            "obj_text_feat": item["obj_text_feat"],
            "part_text_feat": item["part_text_feat"],
            "obj_mask_patch": torch.ones(patch_tokens.shape[0], dtype=torch.bool),
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
        "patch_tokens": b["patch_tokens"].float().unsqueeze(0),             # [1, M, D]
        "obj_text_feat": b["obj_text_feat"].float().unsqueeze(0),           # [1, Dt]
        "part_text_feat": b["part_text_feat"].float().unsqueeze(0),         # [1, K, Dt]
        "obj_mask_patch": b["obj_mask_patch"].bool().unsqueeze(0),          # [1, M]
        "part_gt_mask_patch": b["part_gt_mask_patch"].bool().unsqueeze(0),  # [1, K, M]
        "part_valid_mask": b["part_valid_mask"].bool().unsqueeze(0),        # [1, K]
        "category_id": b["category_id"].view(1),
        "part_category_id": b["part_category_id"].long().unsqueeze(0),      # [1, K]
        "metadata": [b["metadata"]],
    }