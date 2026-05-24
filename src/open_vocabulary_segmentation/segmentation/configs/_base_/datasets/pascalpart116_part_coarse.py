# ------------------------------------------------------------------------------
# Coarse PascalPart116 part dataset.
# It reads original fine masks from annotations_detectron2_part/val and remaps ids in the dataset class.
# ------------------------------------------------------------------------------

_base_ = ["../custom_import.py"]

dataset_type = "PascalPart116_PART_COARSE"
data_root = "./data/PascalPart116"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="FloatImage"),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/val",
        ann_dir="annotations_detectron2_part/val",
        pipeline=test_pipeline,
    )
)

test_cfg = dict(mode="whole")
