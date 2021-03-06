_base_ = "./yolact_r50_1x8_coco.py"
dataset_type = "CocoDataset"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        ann_file="/home/daehan/project/mmcv/mmdetection/data/train.json",
        img_prefix="/home/daehan/project/mmcv/mmdetection/data/images/train/"),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        ann_file="/home/daehan/project/mmcv/mmdetection/data/val.json",
        img_prefix="/home/daehan/project/mmcv/mmdetection/data/images/val/"),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        ann_file="/home/daehan/project/mmcv/mmdetection/data/val.json",
        img_prefix="/home/daehan/project/mmcv/mmdetection/data/images/val/"))


model = dict(
    backbone=dict()
        ,
    bbox_head=dict(
        num_classes=1,
        ),
    mask_head=dict(
        num_classes=1,
        ),
    segm_head=dict(
            num_classes=1,
        ),)


