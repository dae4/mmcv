_base_ = './yolact_r50_1x8_coco.py'

dataset_type = 'CocoDataset'
classes = ('food')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/dae4/mmcv/mmdetection/train.json',
        img_prefix='/disk2/daehan/v5/OD/images/train_v3/'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/dae4/mmcv/mmdetection/val.json',
        img_prefix='/disk2/daehan/v5/OD/images/val/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/dae4/mmcv/mmdetection/val.json',
        img_prefix='/disk2/daehan/v5/OD/images/val/'))


model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    bbox_head=dict(
        num_classes=7,
        ),
    mask_head=dict(
        num_classes=7,
        ),
    segm_head=dict(
            num_classes=7,
        ),)


