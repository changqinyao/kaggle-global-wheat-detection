_base_ = [
    "../_base_/models/detectors_r50_ga.py",
    "../_base_/datasets/wheat_detection_mstrain_hard.py",
    "../_base_/schedules/schedule_4x.py",
    "../_base_/default_runtime.py",
]

data = dict(samples_per_gpu=2)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
resume_from = "/home/ubuntu/PycharmProjects/kaggle-global-wheat-detection/mosaic/work_dirs/detectors_r50_ga_mstrain_stage0/0/epoch_40.pth"
# load_from='/home/ubuntu/PycharmProjects/kaggle-global-wheat-detection/dumps/DetectoRS_R50-0f1c8080.pth'

