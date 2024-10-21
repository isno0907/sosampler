import os
file_name = os.path.basename(__file__).split(".")[0]
# model settings
print(file_name)
total_epochs = 50
work_dir = f'./work_dirs/{file_name}'  # noqa: E501
batch_size = 128
num_gpu = 1
lr = 1e-1
total_seg = 6
sampled_seg = 6
tcp = 29708
num_class = 487

model = dict(
    type='SOSampler2DRecognizer2D',
    label_conf=False,
    use_sampler=True,
    resize_px=128,
    #resize_px=96,
    loss='hinge',
    ce_loss=False,
    ft_loss=False,
    loss_lambda=0.99,
    gamma=0.03,
    num_segments=total_seg,
    num_test_segments=sampled_seg,
    return_logit=False,
    softmax=True,
    use_bb_head=False,
    temperature=0.3,
    dropout_ratio=0.0,
    sampler=dict(
        type='FlexibleMobileNetV2TSM',
        #type='MobileNetV2TSM',
        pretrained='mmcls://mobilenet_v2',
        is_sampler=False,
        shift_div=10,
        num_segments=10,
        total_segments=total_seg),
    backbone=dict(
        type='ResNet',
        frozen_stages=5,
        depth=50,
        norm_eval=True),
    cls_head=dict(
        type='TSMHead',
        num_classes=num_class,
        in_channels=2048,
        spatial_type='avg',
        num_segments=total_seg,
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=False,
        frozen=True,
        ))

train_cfg=None,
test_cfg=dict(average_clips='prob')

# model training and testing settings
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/sports-1m/rawframes'
data_root_val = 'data/sports-1m/rawframes'
ann_file_train = 'data/sports-1m/annotations/train_middle_sports1m.txt'
ann_file_val = 'data/sports-1m/annotations/test_minisports1m.txt'
ann_file_test = 'data/sports-1m/annotations/test_minisports1m.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=total_seg, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=total_seg, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    # follow FrameExit
    # https://github.com/Qualcomm-AI-research/FrameExit/blob/main/config/activitynet_inference_2d.yml#L20-L21
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=total_seg, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    # follow FrameExit
    # https://github.com/Qualcomm-AI-research/FrameExit/blob/main/config/activitynet_inference_2d.yml#L20-L21
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=batch_size,
    workers_per_gpu=8,
    val_dataloader=dict(videos_per_gpu=batch_size),
    test_dataloader=dict(videos_per_gpu=batch_size),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        num_classes=num_class,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        num_classes=num_class,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
dev_check = dict(
    check = True,
    size = 224,
    input_format='NCHW'
)
# optimizer
optimizer = dict(type='SGD', lr=(lr / 8) * (batch_size / 40 * num_gpu / 8), momentum=0.9, weight_decay=0.0001)
# this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)

checkpoint_config = dict(interval=1, max_keep_ckpts=5)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# runtime sett ngs
#dist_params = dict(backend='nccl', port=tcp)
dist_params = dict(backend='nccl')
log_level = 'INFO'

work_dir = './work_dirs/middle_sports_kd_mbnv2_resnet50_default'  # noqa: E501
adjust_parameters = dict(base_ratio=0.0, min_ratio=0., by_epoch=False, style='step')
evaluation = dict(interval=1, metrics=['mean_average_precision'],gpu_collect=True)

# runtime settings
checkpoint_config = dict(interval=5)

load_from = 'modelzoo/middlesports_resnet50.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
