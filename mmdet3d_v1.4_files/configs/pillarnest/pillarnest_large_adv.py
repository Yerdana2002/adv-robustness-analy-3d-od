_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/models/centerpoint_pillar02_second_secfpn_nus_pillarnest.py',
    '../_base_/schedules/cyclic-20e.py', 
    '../_base_/default_runtime.py'
]
gradients_output_dir=''
# this is my base
# Override base config values - don't redefine at top level
# Instead, override within the model dict

voxel_size = [0.15, 0.15, 8]
# Remove this line - it conflicts with base configs:
# mmdet3d/configs/pillarnest/pillarnest_large.py
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=0)]  # disables object sample, aka augmentation
#custom_imports = dict(imports=['mmdet3d.engine.optimizers.my_optimizer'], allow_failed_imports=False)  # this might be very important

point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')

custom_imports = dict(
    imports=[
        'mmdet3d.models.backbones.pillarnest_convnext',
        'mmdet3d.models.voxel_encoders.pillarnest_pillar_encoder',
        'mmdet3d.models.dense_heads.pillarnest_center_head',
        'mmdet3d.models.task_modules.coders.pillarnest_bbox_coder',
        'mmdet3d.engine.optimizers.my_optimizer'
    ],
    allow_failed_imports=False)

model = dict(
    type='CenterPoint',
    data_preprocessor=dict(
        _delete_=True,
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(90000, 120000))),
    
    pts_voxel_encoder=dict(
        _delete_=True,
        type='PillarNestHeightFeatureNet',
        in_channels=5,
        feat_channels=[96],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='maxavg',
        encoder_layer='PFNLayer',
        legacy=False),
    
    pts_middle_encoder=dict(
        _delete_=True,
        type='PointPillarsScatter',
        in_channels=96,
        output_shape=(720, 720)),
    
    # Backbone - ONLY use parameters that PillarNestConvNeXt accepts
    pts_backbone=dict(
        _delete_=True,
        type='PillarNestConvNeXt',
        arch='large',
        in_channels=96,
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
        init_cfg=None),
    
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        in_channels=[384, 384, 384],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    pts_bbox_head=dict(
        _delete_=True,
        type='PillarNestCenterHead',
        in_channels=384,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2),
            height=(1, 2),
            dim=(3, 2),
            rot=(2, 2),
            vel=(2, 2),
            iou=(1, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='PillarNestBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        iou_score=dict(type='BboxOverlaps3D', coordinate='lidar'),
        loss_iou_score=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=1.0),
        iou_score_weight=1.0,
        norm_bbox=True),
    
    train_cfg=dict(
        _delete_=True,
        pts=dict(
            grid_size=[720, 720, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    
    test_cfg=dict(
        _delete_=True,
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=point_cloud_range[:2],
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            iou_score_beta=0.5)))

# Dataset settings
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
backend_args = None

# Add this to your config before train_dataloader

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    # ============== AUGMENTATIONS (disable as needed) ==============
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root=data_root,
            info_path=data_root + 'nuscenes_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)),
            classes=class_names,
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4],
                backend_args=backend_args),
            backend_args=backend_args)),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 0.0],
        scale_ratio_range=[1.0, 1.0],
        translation_std=[0, 0, 0]),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    # ============== END AUGMENTATIONS ==============
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=class_names),
    # dict(
    #     type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_evaluator = dict(
    _delete_=True,
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator

lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='MyOptimizer', 
        lr=0.0) # LR doesn't matter since step is pass, but 0 is safe
)
epoch_num = 1
param_scheduler = [dict(type='LinearLR', start_factor=1.0, begin=0, end=epoch_num)]