_base_ = [
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/schedules/cyclic_80e.py',
    '../_base_/default_runtime.py'
]

device = 'cuda'
file_client_args = dict(backend='disk')

# === Point Cloud Settings ===
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
voxel_size = [0.36, 0.36, 6.0]
grid_size = [416, 416, 1]  # X/Y/Z voxel grid

dataset_type = 'WaymoReducedDataset'
class_names = ['Car', 'Pedestrian', 'Cyclist']

# === Model ===
model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=5,
    ),
    pts_voxel_encoder=dict(
        type='HeightPillarFeatureNet',
        in_channels=5, 
        feat_channels=[64],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        mode='maxavg'
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=(416, 416)
    ),
    pts_backbone=dict(
        type="ConvNeXt_PC",
        arch="base",
        in_channels=64,
        out_indices=[2, 3, 4],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
    ),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[384, 384, 384],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],  # adjust so output = 116x116 for out_size_factor=4
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
    pts_bbox_head=dict(
        type='CenterHead_f',
        in_channels=128 * 3,  # 3 scales concatenated
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist'])
        ],
        common_heads=dict(
            reg=(2, 2),
            height=(1, 2),
            dim=(3, 2),
            rot=(2, 2),
            # vel=(2, 2),
            iou=(1, 2)
        ),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-80, -80, -10, 80, 80, 10],
            max_num=100,
            score_threshold=0.1,
            out_size_factor=4,  # matches SECONDFPN output
            voxel_size=voxel_size[:2],
            code_size=7
        ),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='L1Loss',
            reduction='mean',
            loss_weight=0.25),
        norm_bbox=True
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            out_size_factor=4,  # match bbox_coder
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0] * 7  # match code_size
        )
    ),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=4,  # match training
            iou_score_beta=0.5,
            post_center_limit_range=point_cloud_range,
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            nms_type='rotate',
            pre_max_size=4096,
            post_max_size=512,
            nms_thr=0.2
        )
    )
)

# === Training Pipeline ===
train_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=6,
         use_dim=5,
         file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.78539816, 0.78539816],
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])]

# === Optimizer & Data ===
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    test=dict(
        type=dataset_type,
        samples_per_scene=5,
        seed=42,
        pipeline=test_pipeline, 
        adv_mode=True,
        classes=class_names))