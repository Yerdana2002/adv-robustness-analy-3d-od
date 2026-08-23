# =============================================================================
# FocalFormer3D Waymo Config — Refactored for MMDetection3D >= 1.1
# =============================================================================
#FocalFormer3D_Waymo_L.py
# ---------------------------------------------------------------------------
# 1. Custom imports
# ---------------------------------------------------------------------------
custom_imports = dict(
    imports=['projects.mmdet3d_plugin'],
    allow_failed_imports=False)

# ---------------------------------------------------------------------------
# 2. Runtime defaults
# ---------------------------------------------------------------------------
default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=7),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# ---------------------------------------------------------------------------
# 3. Shared constants (Waymo Specific)
# ---------------------------------------------------------------------------
# Waymo range is larger than NuScenes
point_cloud_range = [-76.8, -76.8, -2, 76.8, 76.8, 4]
class_names = ['Car', 'Pedestrian', 'Cyclist']
metainfo = dict(classes=class_names)
voxel_size = [0.1, 0.1, 0.15]
out_size_factor = 8

dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
waymo_metric_root = 'data/waymo/waymo_format/'
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

multistage_heatmap = 2
inter_channel = 128
extra_feat = True

# ---------------------------------------------------------------------------
# 4. Database sampler
# ---------------------------------------------------------------------------
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args))

# ---------------------------------------------------------------------------
# 5. Data pipelines
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,  # Waymo has 6 dims
        use_dim=5,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        # Waymo rotation range is smaller than NuScenes (approx -45 to 45 deg)
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]), # Source config had 0 translation std
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800), # Standard placeholder scale
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
    #dict(type='Pack3DDetInputs', keys=['points']),
    dict(
    type='Pack3DDetInputs',
    keys=['points'],
    meta_keys=['box_type_3d', 'sample_idx', 'context_name', 'timestamp']),

]

# ---------------------------------------------------------------------------
# 6. Dataloaders
# ---------------------------------------------------------------------------
# Waymo config used RepeatDataset(times=1), effectively standard loading
train_dataloader = dict(
    batch_size=4, # Source had 4
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='waymo_infos_val.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            # Waymo data prefix usually needs specific setting depending on generation
            data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = val_dataloader  # For test submission generation, we can use the same dataloader with test annotations

# test_dataloader = dict(
#         batch_size=1,
#         num_workers=6,
#         persistent_workers=True,
#         drop_last=False,
#         sampler=dict(type='DefaultSampler', shuffle=False),
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             ann_file='waymo_infos_test.pkl',
#             pipeline=test_pipeline,
#             metainfo=metainfo,
#             modality=input_modality,
#             test_mode=True,
#             data_prefix=dict(pts='testing/velodyne', sweeps='testing/velodyne'),
#             box_type_3d='LiDAR',
#             backend_args=backend_args))

# ---------------------------------------------------------------------------
# 7. Evaluators
# ---------------------------------------------------------------------------
# val_evaluator = dict(
#     type='WaymoMetric',
#     ann_file=data_root + 'waymo_infos_val.pkl',
#     waymo_bin_file=data_root + 'waymo_infos_val.pkl', # Adjust based on data generation
#     data_root=data_root,
#     metric='LET_mAP',
#     backend_args=backend_args)

val_evaluator = dict(
    type='WaymoMetric',  # <--- Change this
    #ann_file=data_root + 'waymo_infos_val.pkl',
    metric='mAP',       # Standard 3D bounding box mAP
    waymo_bin_file=waymo_metric_root + 'gt.bin',
    result_prefix='work_dirs/waymo_preds',
    load_type='frame_based')
    #backend_args=backend_args)

    #'./data/waymo/waymo_format/gt.bin'
    # data_root = 'data/waymo/kitti_format/'

test_evaluator = val_evaluator  # For test submission generation, we can use the same evaluator with test annotations
# For generating Test Submission (No metrics calculated)
# test_evaluator = dict(
#     type='WaymoMetric',
#     ann_file=data_root + 'waymo_infos_test.pkl',
#     # waymo_bin_file is NOT needed for test submission generation
#     metric='LET_mAP',
#     backend_args=backend_args)


# ---------------------------------------------------------------------------
# 8. Model
# ---------------------------------------------------------------------------
model = dict(
    type='FocalFormer3D',
    freeze_img=True,
    freeze_pts=False,  # We want gradients to flow through the point-based backbone and neck
    input_img=False,
    data_preprocessor=dict(
        type='mmdet3d.Det3DDataPreprocessor',
        voxel=False),
    # Waymo specific voxel settings
    pts_voxel_layer=dict(
        max_num_points=5, # Source: 5 (NuScenes was 10)
        voxel_size=voxel_size,
        max_voxels=150000, # Source: 150k
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        with_cluster_center=False,
        with_voxel_center=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1536, 1536], # Calculated from Waymo range/voxel_size
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
        num_proposals=200, # Source: 200 (NuScenes was 300)
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
            rot=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-80, -80, -10.0, 80, 80, 10.0], # Waymo Range
            score_threshold=0.0,
            code_size=8), # Waymo code size 10
        loss_cls=dict(
            type='mmdet.FocalLoss', use_sigmoid=True,
            gamma=2, alpha=0.25, reduction='mean', loss_weight=0.6), # Source: 0.6
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=2.0), # Source: 2.0
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
            dataset='Waymo',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='FocalLossCost', gamma=2, alpha=0.25, weight=0.6),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=2.0),
                iou_cost=dict(type='IoU3DCost', weight=2.0)),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1536, 1536, 40], # Waymo grid size
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # 8 weights
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='Waymo',
            grid_size=[1536, 1536, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None)))

# ---------------------------------------------------------------------------
# 9. Optimizer wrapper
# ---------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2))

# ---------------------------------------------------------------------------
# 10. Param scheduler (Adapting Source 11 epochs to 1.x syntax)
# ---------------------------------------------------------------------------
total_epochs = 11
up_ratio = 0.4
lr = 0.0001

param_scheduler = [
    # Phase 1: Ramp up
    dict(
        type='CosineAnnealingLR',
        T_max=total_epochs * up_ratio,
        eta_min=lr * 10,
        begin=0,
        end=total_epochs * up_ratio,
        by_epoch=True,
        convert_to_iter_based=True),
    # Phase 2: Ramp down
    dict(
        type='CosineAnnealingLR',
        T_max=total_epochs * (1 - up_ratio),
        eta_min=lr * 0.0001,
        begin=total_epochs * up_ratio,
        end=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    # Momentum (Optional, matches NuScenes example)
    dict(
        type='CosineAnnealingMomentum',
        T_max=total_epochs * up_ratio,
        eta_min=0.8947368421052632,
        begin=0,
        end=total_epochs * up_ratio,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=total_epochs * (1 - up_ratio),
        eta_min=1,
        begin=total_epochs * up_ratio,
        end=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

# ---------------------------------------------------------------------------
# 11. Training / val / test loop config
# ---------------------------------------------------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ---------------------------------------------------------------------------
# 12. Loading & resuming
# ---------------------------------------------------------------------------
# Source checkpoint from Waymo config
load_from = './work_dirs/DeformFormer3D_Waymo_L/epoch_36.pth'
resume = False
find_unused_parameters = True

# ---------------------------------------------------------------------------
# 13. Hooks
# ---------------------------------------------------------------------------
# Source Waymo config used `fade_epoch=5`.
# In old configs this usually meant "turn off at epoch 5".
custom_hooks = [
    dict(type='DisableObjectSampleHook', disable_after_epoch=5)
]

auto_scale_lr = dict(enable=False, base_batch_size=16)