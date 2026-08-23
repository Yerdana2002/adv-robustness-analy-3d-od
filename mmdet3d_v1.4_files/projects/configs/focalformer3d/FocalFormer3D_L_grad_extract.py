# =============================================================================
# FocalFormer3D LiDAR-only — Gradient Extraction on NuScenes Val
# =============================================================================
# Standalone config. Runs 1 epoch of "training" on val set with NoOpOptimizer
# (weights frozen, gradients flow) for extraction via the hook.
# =============================================================================

# ---------------------------------------------------------------------------
# Custom imports
# ---------------------------------------------------------------------------

custom_imports = dict(
    imports=[
        'projects.mmdet3d_plugin',
        'projects.mmdet3d_plugin.hooks.focalformer_gradient_hook', #FocalFormerGradientHook
        'projects.mmdet3d_plugin.hooks.noop_optimizer',
    ],
    allow_failed_imports=False)

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=999),
    sampler_seed=dict(type='DistSamplerSeedHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ---------------------------------------------------------------------------
# Gradient extraction settings
# ---------------------------------------------------------------------------
gradients_output_dir = ''  # Override via --cfg-options gradients_output_dir=...

custom_hooks = [
    dict(type='DisableObjectSampleHook', disable_after_epoch=0),
    dict(
        type='FocalFormerGradientHook',
        target_layer='neck',
        save_path=gradients_output_dir,
        normalize=True,
        save_interval=100),
]

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)
voxel_size = [0.075, 0.075, 0.2]
out_size_factor = 8

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

inter_channel = 128
multistage_heatmap = 1
extra_feat = True

# ---------------------------------------------------------------------------
# Model (FocalFormer3D_L_v14)
# ---------------------------------------------------------------------------
model = dict(
    type='FocalFormer3D',
    freeze_img=True,
    freeze_pts=False,  # We want gradients to flow through the point-based backbone and neck
    input_img=False,
    data_preprocessor=dict(
        type='mmdet3d.Det3DDataPreprocessor',
        voxel=False),
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    imgpts_neck=dict(
        type='FocalEncoder',
        num_layers=multistage_heatmap,
        in_channels_img=256,
        in_channels_pts=sum([256, 256]),
        hidden_channel=inter_channel,
        bn_momentum=0.1,
        max_points_height=10,
        bias='auto',
        iterbev='bevfusionmb2',
        input_img=False,
        iterbev_wo_img=True,
        multistage_heatmap=multistage_heatmap,
        extra_feat=extra_feat),
    pts_bbox_head=dict(
        type='FocalDecoder',
        reuse_first_heatmap=True,
        extra_feat=extra_feat,
        roi_feats=7,
        roi_dropout_rate=0.1,
        roi_based_reg=True,
        roi_expand_ratio=1.2,
        heatmap_box=False,
        thin_heatmap_box=False,
        multiscale=True,
        multistage_heatmap=multistage_heatmap,
        mask_heatmap_mode='poscls',
        input_img=False,
        iterbev_wo_img=True,
        add_gt_groups=3,
        add_gt_groups_noise='box,1',
        add_gt_groups_noise_box='gtnoise',
        add_gt_pos_thresh=5.,
        add_gt_pos_boxnoise_thresh=0.75,
        gt_center_limit=5,
        bevpos=True,
        loss_weight_heatmap=1.,
        loss_weight_separate_heatmap=0.,
        loss_weight_separate_bbox=0.3,
        num_proposals=300,
        hidden_channel=inter_channel,
        num_classes=len(class_names),
        num_decoder_layers=2,
        num_heads=8,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2),
            rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10),
        loss_cls=dict(
            type='mmdet.FocalLoss', use_sigmoid=True,
            gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        decoder_cfg=dict(
            type='DeformableDetrTransformerDecoder',
            num_layers=3,
            return_intermediate=False,
            post_norm_cfg=None,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=inter_channel,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=inter_channel,
                    num_levels=3,
                    num_points=4,
                    num_heads=8,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=inter_channel,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True))))),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None)))

# ---------------------------------------------------------------------------
# Data pipelines — NO augmentation for gradient extraction
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # No ObjectSample — disabled for clean gradient extraction
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 0.0],
        scale_ratio_range=[1.0, 1.0],
        translation_std=[0, 0, 0]),
    # No RandomFlip3D
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    # No PointShuffle
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(800, 448),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points']),
]

# ---------------------------------------------------------------------------
# Dataloaders — FLAT NuScenesDataset (NO CBGSDataset wrapper)
# ---------------------------------------------------------------------------
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='NuScenesDataset',
#         data_root=data_root,
#         ann_file='nuscenes_infos_val.pkl',
#         pipeline=test_pipeline,
#         metainfo=metainfo,
#         modality=input_modality,
#         test_mode=True,
#         data_prefix=data_prefix,
#         box_type_3d='LiDAR',
#         backend_args=backend_args))

#test_dataloader = val_dataloader

# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------
val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# ---------------------------------------------------------------------------
# Optimizer — NoOp (gradients flow, weights frozen)
# ---------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='NoOpOptimizer', lr=0.0))

# ---------------------------------------------------------------------------
# Schedule — 1 epoch, no LR changes
# ---------------------------------------------------------------------------
epoch_num = 1
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0, begin=0, end=epoch_num)]

#train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch_num, val_interval=999)
#val_cfg = dict(type='ValLoop')
#test_cfg = dict(type='TestLoop')

# disable evaluation loops entirely
val_dataloader = None
val_evaluator = None
val_cfg = None

test_dataloader = None
test_evaluator = None
test_cfg = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1)



# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
load_from = None  # Set via --cfg-options or bash script
resume = False
find_unused_parameters = True
auto_scale_lr = dict(enable=False, base_batch_size=16)