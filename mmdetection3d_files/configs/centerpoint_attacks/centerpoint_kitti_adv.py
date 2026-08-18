_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/models/centerpoint_005voxel_second_secfpn_kitti.py',
    '../_base_/schedules/cyclic_80e.py', '../_base_/default_runtime.py'
]
device = 'cuda'  
file_client_args = dict(backend='disk')
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size = [0.05, 0.05, 0.1]
class_names = ['Pedestrian', 'Cyclist', 'Car']
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range, max_num_points=5, voxel_size=voxel_size, max_voxels=(16000, 40000)),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

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
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
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
    val=dict(
        pipeline=test_pipeline, 
        adv_mode=True,
        classes=class_names),
    test=dict(
        pipeline=test_pipeline, 
        adv_mode=True,
        classes=class_names))