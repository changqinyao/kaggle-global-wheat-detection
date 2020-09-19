_base_ = [
    "../_base_/models/fcos_r50_caffe_fpn_gn-head_4x4.py",
    "../_base_/datasets/wheat_detection_mstrain_hard.py",
    "../_base_/schedules/schedule_4x.py",
    "../_base_/default_runtime.py",
]
model = dict(
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    bbox_head=dict(
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)))

data = dict(samples_per_gpu=4)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=None)
lr_config = dict(warmup='linear')