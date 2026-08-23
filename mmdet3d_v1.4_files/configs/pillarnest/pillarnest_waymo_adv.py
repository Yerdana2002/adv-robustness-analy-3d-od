# ============================================================
# PillarNeSt Waymo Adversarial/Gradient Config (MMDet3D v1.4)
# Single-file, copy-paste ready
# ============================================================

        

# PillarNeSt Waymo (v0.18-style, MMDet3D v1.4-compatible)
# _f variants mapped to standard CenterHead + CenterPointBBoxCoder.

default_scope = 'mmdet3d'
backend_args = None

# Runtime
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=999),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'),
)
gradients_output_dir=''

# custom_hooks = [dict(type='GradientCaptureHook', module_name='pts_middle_encoder', save_path=gradients_output_dir, start_epoch=0, normalize=True)]

custom_hooks=[]


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Data constants
dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
waymo_metric_root = 'data/waymo/waymo_format/'
class_names = ['Car', 'Pedestrian', 'Cyclist']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)

point_cloud_range = [-74.88, -74.88, -2.0, 74.88, 74.88, 4.0]
voxel_size = [0.36, 0.36, 6.0]
grid_size = [416, 416, 1]
out_size_factor = 4

# Model
model = dict(
    type='CenterPoint',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(120000, 160000),
        )),
    pts_voxel_encoder=dict(
        type='PillarNestHeightFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='maxavg',
        legacy=True,
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=(416, 416),
    ),
    pts_backbone=dict(
        type='PillarNestConvNeXt',
        arch='base',
        in_channels=64,
        stem_patch_size=4,
        norm_cfg=dict(type='LN2d', eps=1e-6),
        act_cfg=dict(type='GELU'),
        linear_pw_conv=True,
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[2, 3, 4],
        frozen_stages=0,
        gap_before_final_norm=False,
        first_downsample=1,
        large_arch=None,
        init_cfg=None,
    ),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[384, 384, 384],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True,
    ),
    pts_bbox_head=dict(
        type='CenterHead_f',  # _f -> standard
        debug=True,
        in_channels=384,
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
        ],
        common_heads=dict(
            reg=(2, 2),
            height=(1, 2),
            dim=(3, 2),
            rot=(2, 2),
            iou=(1, 2),  # kept to match legacy checkpoint layout
        ),
        share_conv_channel=64,
        separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
        bbox_coder=dict(
            type='CenterPointBBoxCoder_f',  # _f -> standard
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-80, -80, -10, 80, 80, 10],
            max_num=500,
            score_threshold=0.01,  # debug-friendly; raise later
            code_size=7,
        ),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0] * 8,
        )),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_limit_range=point_cloud_range,
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 4, 4],
            score_threshold=0.01,  # debug-friendly; raise later
            nms_type='rotate',
            pre_max_size=4096,
            post_max_size=512,
            nms_thr=0.2,
        )),
)

# Pipelines
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='GlobalRotScaleTrans', rot_range=[0.0, 0.0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ConvertToLegacyCoords'),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5, backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='GlobalRotScaleTrans', rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='ConvertToLegacyCoords'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=['box_type_3d', 'sample_idx', 'context_name', 'timestamp']),
]

# Dataloaders
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        box_type_3d='LiDAR', 
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='WaymoMetric',
    metric='mAP',
    load_type='frame_based',
    result_prefix='work_dirs/waymo_preds',
    waymo_bin_file=waymo_metric_root + 'gt.bin',
)
test_evaluator = val_evaluator

# Keep eval clean (no attack/training hooks)

# Dummy optimizer section (not used for test but keeps config complete)

custom_imports = dict(
    imports=[
        'mmdet3d.engine.optimizers.my_optimizer',
        'mmdet3d.datasets.transforms.convert_legacy_coords',
    ],
    allow_failed_imports=False
)

optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='MyOptimizer', lr=0.0))
param_scheduler = [dict(type='LinearLR', start_factor=1.0, begin=0, end=1)]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=999)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

auto_scale_lr = dict(enable=False, base_batch_size=16)
load_from = None
resume = False
work_dir = ''
