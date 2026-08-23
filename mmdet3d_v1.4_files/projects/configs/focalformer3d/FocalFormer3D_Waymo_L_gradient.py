# =============================================================================
# FocalFormer3D Waymo Gradient Extraction Config
# =============================================================================
# Inherits from the main FocalFormer3D Waymo config and overrides:
#   - Disables all data augmentations (flip, rotate, scale, shuffle, ObjectSample)
#   - Uses NoOpOptimizer to freeze weights
#   - Batch size 1, sequential loading from val set
#   - Registers FocalFormerGradientHook for gradient capture
#   - Trains for 1 epoch over val set to extract gradients
#
# Usage:
#   python tools/train.py <this_config> --work-dir <work_dir> \
#       --cfg-options \
#           "custom_hooks.1.target_layer=neck" \
#           "custom_hooks.1.save_path=/path/to/gradients"
# =============================================================================

_base_ = ['./FocalFormer3D_Waymo_L.py']

# ---------------------------------------------------------------------------
# Custom imports: hook + no-op optimizer
# ---------------------------------------------------------------------------
custom_imports = dict(
    imports=[
        'projects.mmdet3d_plugin',
        'projects.mmdet3d_plugin.hooks.noop_optimizer',
    ],
    allow_failed_imports=False)

# ---------------------------------------------------------------------------
# Gradient output directory (override via --cfg-options)
# ---------------------------------------------------------------------------
#gradients_output_dir = './work_dirs/focalformer_waymo_gradients'

# ---------------------------------------------------------------------------
# Hooks: disable augmentation + gradient extraction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# No-op optimizer: forward + backward run, but weights never change
# ---------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='NoOpOptimizer', lr=0.0))

# ---------------------------------------------------------------------------
# 1 epoch, no validation (just gradient extraction)
# ---------------------------------------------------------------------------
epoch_num = 1
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch_num, val_interval=999)
param_scheduler = [dict(type='LinearLR', start_factor=1.0, begin=0, end=epoch_num)]

# ---------------------------------------------------------------------------
# Data pipeline: NO augmentations (Waymo-specific)
#   - load_dim=6, use_dim=5 (Waymo has 6 dims in point cloud)
#   - No LoadPointsFromMultiSweeps (Waymo doesn't use sweeps)
#   - No ObjectSample
#   - No rotation/scaling/translation (identity only)
#   - No RandomFlip3D
#   - No PointShuffle
#   - Sequential order (shuffle=False) on val set
# ---------------------------------------------------------------------------
point_cloud_range = [-76.8, -76.8, -2, 76.8, 76.8, 4]
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        backend_args=backend_args),
    # NO LoadPointsFromMultiSweeps (Waymo doesn't use sweeps)
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # NO ObjectSample
    # Identity transform only (no augmentation)
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 0.0],
        scale_ratio_range=[1.0, 1.0],
        translation_std=[0, 0, 0]),
    # NO RandomFlip3D
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # NO PointShuffle
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

# ---------------------------------------------------------------------------
# Dataloader: batch_size=1, sequential, val set as train
# ---------------------------------------------------------------------------
data_root = 'data/waymo/kitti_format/'

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='WaymoDataset',
        data_root=data_root,
        ann_file='waymo_infos_val.pkl',
        pipeline=train_pipeline,
        metainfo=dict(classes=['Car', 'Pedestrian', 'Cyclist']),
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=False,
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        box_type_3d='LiDAR',
        backend_args=backend_args))

# ---------------------------------------------------------------------------
# Load pretrained checkpoint (override via --cfg-options or bash script)
# ---------------------------------------------------------------------------
load_from = None
resume = False