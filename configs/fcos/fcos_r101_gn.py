_base_ = [
    "../_base_/models/fcos_r50_caffe_fpn_gn-head_4x4.py",
    "../_base_/datasets/wheat_detection_mstrain_hard.py",
    "../_base_/schedules/schedule_4x.py",
    "../_base_/default_runtime.py",
]
model = dict(
    pretrained='open-mmlab://detectron/resnet101_caffe',
    backbone=dict(depth=101),
    bbox_head = dict(
    type='FCOSHead_TTA',
    num_classes=1,
    in_channels=256,
    stacked_convs=4,
    feat_channels=256,
    strides=[8, 16, 32, 64, 128],
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0),
    loss_bbox=dict(type='IoULoss', loss_weight=1.0),
    loss_centerness=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
)

data = dict(samples_per_gpu=4)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# resume_from='/home/ubuntu/PycharmProjects/mmdetection/tools/work_dirs/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_4x4_2x_coco/epoch_16.pth'

lr_config = dict(step=[30, 60])
total_epochs = 30
