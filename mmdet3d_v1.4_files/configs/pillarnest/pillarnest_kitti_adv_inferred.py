
#inferred

auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None
class_names = [
    'Pedestrian',
    'Cyclist',
    'Car',
]
custom_hooks = [
    dict(disable_after_epoch=0, type='DisableObjectSampleHook'),
    dict(module_name='pts_middle_encoder', save_path='', type='myHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet3d.engine.optimizers.my_optimizer',
    ])
data_root = 'data/kitti/'
dataset_type = 'KittiDataset'
default_hooks = dict(
    checkpoint=dict(interval=999, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
gradients_output_dir = ''
input_modality = dict(use_camera=False, use_lidar=True)
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.001
metainfo = dict(classes=[
    'Pedestrian',
    'Cyclist',
    'Car',
])
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            max_voxels=(
                16000,
                40000,
            ),
            point_cloud_range=[
                0,
                -40,
                -3,
                70.4,
                40,
                1,
            ],
            voxel_size=[
                0.16,
                0.16,
                4.0,
            ])),
    pts_backbone=dict(
        act_cfg=dict(type='GELU'),
        arch='base',
        drop_path_rate=0.0,
        first_downsample=1,
        frozen_stages=0,
        gap_before_final_norm=False,
        in_channels=64,
        init_cfg=None,
        large_arch=None,
        layer_scale_init_value=1.0,
        linear_pw_conv=True,
        norm_cfg=dict(eps=1e-06, type='LN2d'),
        out_indices=[
            2,
            3,
            4,
        ],
        stem_patch_size=4,
        type='PillarNestConvNeXt'),
    pts_bbox_head=dict(
        bbox_coder=dict(
            code_size=7,
            max_num=500,
            out_size_factor=4,
            pc_range=[
                0,
                -40,
            ],
            post_center_range=[
                -10,
                -50,
                -10,
                80.4,
                50,
                10,
            ],
            score_threshold=0.1,
            type='PillarNestBBoxCoder',
            voxel_size=[
                0.16,
                0.16,
            ]),
        common_heads=dict(
            dim=(
                3,
                2,
            ),
            height=(
                1,
                2,
            ),
            iou=(
                1,
                2,
            ),
            reg=(
                2,
                2,
            ),
            rot=(
                2,
                2,
            )),
        in_channels=384,
        iou_score=dict(coordinate='lidar', type='BboxOverlaps3D'),
        iou_score_weight=1.0,
        loss_bbox=dict(
            loss_weight=0.25, reduction='mean', type='mmdet.L1Loss'),
        loss_cls=dict(reduction='mean', type='mmdet.GaussianFocalLoss'),
        loss_iou_score=dict(
            loss_weight=1.0, reduction='mean', type='mmdet.L1Loss'),
        norm_bbox=True,
        separate_head=dict(
            final_kernel=3, init_bias=-2.19, type='SeparateHead'),
        share_conv_channel=64,
        tasks=[
            dict(class_names=[
                'Pedestrian',
            ], num_class=1),
            dict(class_names=[
                'Cyclist',
            ], num_class=1),
            dict(class_names=[
                'Car',
            ], num_class=1),
        ],
        type='PillarNestCenterHead'),
    pts_middle_encoder=dict(
        in_channels=64, output_shape=(
            500,
            440,
        ), type='PointPillarsScatter'),
    pts_neck=dict(
        in_channels=[
            128,
            384,
            384,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            384,
            128,
            128,
        ],
        type='SECONDFPN',
        upsample_cfg=dict(bias=False, type='deconv'),
        upsample_strides=[
            1,
            2,
            4,
        ],
        use_conv_for_no_stride=True),
    pts_voxel_encoder=dict(
        feat_channels=[
            64,
        ],
        in_channels=4,
        legacy=False,
        mode='max',
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN1d'),
        point_cloud_range=[
            0,
            -40,
            -3,
            70.4,
            40,
            1,
        ],
        type='PillarNestFeatureNet',
        voxel_size=[
            0.16,
            0.16,
            4.0,
        ],
        with_cluster_center=True,
        with_distance=True,
        with_voxel_center=True),
    test_cfg=dict(
        pts=dict(
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[
                4,
                4,
                4,
            ],
            nms_thr=0.2,
            nms_type='rotate',
            out_size_factor=4,
            pc_range=[
                0,
                -40,
            ],
            post_center_limit_range=[
                -10,
                -50,
                -10,
                80.4,
                50,
                10,
            ],
            post_max_size=200,
            pre_max_size=1000,
            score_threshold=0.1,
            voxel_size=[
                0.16,
                0.16,
            ])),
    train_cfg=dict(
        pts=dict(
            code_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dense_reg=1,
            gaussian_overlap=0.1,
            grid_size=[
                440,
                500,
                1,
            ],
            max_objs=500,
            min_radius=2,
            out_size_factor=4,
            point_cloud_range=[
                0,
                -40,
                -3,
                70.4,
                40,
                1,
            ],
            voxel_size=[
                0.16,
                0.16,
                4.0,
            ])),
    type='CenterPoint')
optim_wrapper = dict(
    optimizer=dict(lr=0.0, type='MyOptimizer'), type='OptimWrapper')
out_size_factor = 4
param_scheduler = [
    dict(begin=0, end=1, start_factor=1.0, type='LinearLR'),
]
point_cloud_range = [
    0,
    -40,
    -3,
    70.4,
    40,
    1,
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            0,
                            -40,
                            -3,
                            70.4,
                            40,
                            1,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(
                rot_range=[
                    0,
                    0,
                ],
                scale_ratio_range=[
                    1.0,
                    1.0,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    0,
                    -40,
                    -3,
                    70.4,
                    40,
                    1,
                ],
                type='PointsRangeFilter'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=999)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                point_cloud_range=[
                    0,
                    -40,
                    -3,
                    70.4,
                    40,
                    1,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    0,
                    -40,
                    -3,
                    70.4,
                    40,
                    1,
                ],
                type='ObjectRangeFilter'),
            dict(
                classes=[
                    'Pedestrian',
                    'Cyclist',
                    'Car',
                ],
                type='ObjectNameFilter'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='KittiDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        point_cloud_range=[
            0,
            -40,
            -3,
            70.4,
            40,
            1,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            0,
            -40,
            -3,
            70.4,
            40,
            1,
        ],
        type='ObjectRangeFilter'),
    dict(classes=[
        'Pedestrian',
        'Cyclist',
        'Car',
    ], type='ObjectNameFilter'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            0,
                            -40,
                            -3,
                            70.4,
                            40,
                            1,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.16,
    0.16,
    4.0,
]
work_dir = ''
