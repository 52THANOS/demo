default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))
custom_hooks = [dict(type='SyncBuffersHook')]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False
backend_args = dict(backend='local')
train_cfg = dict(by_epoch=True, max_epochs=420, val_interval=10)
val_cfg = dict()
test_cfg = dict()
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.8,
        custom_keys=dict(
            bias=dict(decay_multi=0.0),
            pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1.0, norm_type=2))
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]
auto_scale_lr = dict(base_batch_size=512)
codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmcls.VisionTransformer',
        arch=dict(
            embed_dims=384,
            num_layers=12,
            num_heads=12,
            feedforward_channels=1536),
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.1,
        with_cls_token=False,
        output_cls_token=False,
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_small.pth'
        )),
    head=dict(
        type='HeatmapHead',
        in_channels=384,
        out_channels=2,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='UDPHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2)),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False))
data_root = 'qiaqianshangji/'
dataset_type = 'qiaqianshangjiDataset'
data_mode = 'topdown'
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='UDPHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2)),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='qiaqianshangjiDataset',
        data_root='qiaqianshangji/',
        data_mode='topdown',
        ann_file='annotations/train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomHalfBody'),
            dict(type='RandomBBoxTransform'),
            dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='UDPHeatmap',
                    input_size=(192, 256),
                    heatmap_size=(48, 64),
                    sigma=2)),
            dict(type='PackPoseInputs')
        ]))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='qiaqianshangjiDataset',
        data_root='qiaqianshangji/',
        data_mode='topdown',
        ann_file='annotations/val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
            dict(type='PackPoseInputs')
        ]))
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='qiaqianshangjiDataset',
        data_root='qiaqianshangji/',
        data_mode='topdown',
        ann_file='annotations/val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
            dict(type='PackPoseInputs')
        ]))
val_evaluator = dict(
    type='CocoMetric', ann_file='qiaqianshangji/annotations/val2017.json')
test_evaluator = dict(
    type='CocoMetric', ann_file='qiaqianshangji/annotations/val2017.json')
