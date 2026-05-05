# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import os

from mmseg.datasets import DATASETS
from mmseg.datasets import CustomDataset

@DATASETS.register_module(force=True)
class PascalPart116_OBJ(CustomDataset):

    CLASSES = ("aeroplane", 
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
               "tvmonitor",)

    PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    def __init__(self, split=None, **kwargs):
        super(PascalPart116_OBJ, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)
        assert os.path.exists(self.img_dir) 
