_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic_80e.py', 
    '../_base_/default_runtime.py'
]
device = 'cuda'  
file_client_args = dict(backend='disk')
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size = [0.1, 0.1, 0.2]
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'

class_names = ['Car', 'Pedestrian', 'Cyclist']

model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=5,
    ),
    pts_voxel_encoder=dict(
        type='HeightPillarFeatureNet',
        feat_channels=[64],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        mode='maxavg'
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=(800, 704)  # now (800, 704)
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
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
    pts_bbox_head=dict(
        type='CenterHead_f',# trained without centerpoint_plus head because of kitti compability
        in_channels=128*3,
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
        ],
        bbox_coder=dict(
            type='CenterPointBBoxCoder_PN',
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            out_size_factor=4,
            post_center_range=[-10, -50, -10, 80.4, 50, 10],
            max_num=500,
            code_size=7,
        ),
        common_heads=dict(
            reg=(2, 2),
            height=(1, 2),
            dim=(3, 2),
            rot=(2, 2),
            #vel=(2, 2),
            iou=(1, 2)
        ),
        #iou_score=dict(type='BboxOverlaps3D', coordinate='lidar'),
        #loss_iou_score=dict(type='L1Loss', reduction='mean', loss_weight=1.0),
        #iou_score_weight=1.0
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=[704, 800, 20],  # now [704, 800, 20]
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
    ),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=4,
            iou_score_beta=0.5,
            post_center_limit_range=[-10, -50, -10, 80.4, 50, 10],
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


db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        pipeline=test_pipeline, 
        adv_mode=True,
        classes=class_names),
    test=dict(
        pipeline=test_pipeline, 
        adv_mode=True,
        classes=class_names))
optimizer = dict(type='AdamW', lr=10e-4, weight_decay=0.01)

