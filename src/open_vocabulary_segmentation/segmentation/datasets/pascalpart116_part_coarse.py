# ------------------------------------------------------------------------------
# Coarse PascalPart116 dataset for 0V9S/Talk2DINO.
# Keeps original fine mask files, but remaps fine part ids to coarse part ids at evaluation time.
# ------------------------------------------------------------------------------

from __future__ import annotations

import os
import os.path as osp

import mmcv
import numpy as np
from mmseg.datasets import DATASETS
from mmseg.datasets import CustomDataset

from src.voc116_part_coarse import COARSE_PART_CLASSES, COARSE_PALETTE, FINE_TO_COARSE_LUT


@DATASETS.register_module(force=True)
class PascalPart116_PART_COARSE(CustomDataset):
    CLASSES = tuple(COARSE_PART_CLASSES)
    PALETTE = COARSE_PALETTE

    def __init__(self, split=None, **kwargs):
        super(PascalPart116_PART_COARSE, self).__init__(
            img_suffix=".jpg",
            seg_map_suffix=".png",
            split=split,
            reduce_zero_label=False,
            **kwargs,
        )
        assert os.path.exists(self.img_dir)

    @staticmethod
    def _remap(gt):
        gt = np.asarray(gt)
        if gt.ndim == 3:
            gt = gt[..., 0]
        return FINE_TO_COARSE_LUT[gt.astype(np.uint8)]

    def get_gt_seg_map_by_idx(self, index):
        ann_info = self.get_ann_info(index)
        seg_map = ann_info["seg_map"]
        seg_map_path = osp.join(self.ann_dir, seg_map)
        gt = mmcv.imread(seg_map_path, flag="unchanged", backend="pillow")
        return self._remap(gt)

    def get_gt_seg_maps(self, efficient_test=None):
        return [self.get_gt_seg_map_by_idx(i) for i in range(len(self.img_infos))]
