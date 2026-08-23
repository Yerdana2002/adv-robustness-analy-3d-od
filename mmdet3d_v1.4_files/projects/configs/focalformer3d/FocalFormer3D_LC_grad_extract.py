# =============================================================================
# FocalFormer3D LiDAR+Camera -- gradient extraction on nuScenes val
# =============================================================================
# Inherits the model from FocalFormer3D_LC_test.py rather than restating it.
# That config is VALIDATED: job 19039335 reproduced the checkpoint's own
# reference to within 0.3 of a point (NDS 0.7282 vs 0.7310, mAP 0.7025 vs
# 0.7050). Re-deriving the eight camera-branch settings here would mean
# re-earning that guarantee, so don't -- change the base if something is wrong.
#
# What this adds on top of the base: a train loop that runs one epoch over the
# VAL split with a no-op optimizer, so backward reaches pts_neck and
# FocalFormerGradientHook can persist dL/df per frame. Weights never move.
#
# Augmentation is off, deliberately and completely
# ------------------------------------------------
# No ObjectSample, no RandomFlip3D, no PointShuffle, no GlobalRotScaleTrans,
# and FFImageAug3D runs with is_train=False so the resize is deterministic.
# A gradient extracted under augmentation does not correspond to the frame the
# attack will later perturb.
#
# The point filters are IDENTICAL to FocalFormer3D_L_grad_extract.py --
# PointsRangeFilter, ObjectRangeFilter, ObjectNameFilter, use_valid_flag=True.
# That is what makes the LC gradient set cover the same 5980 frames as the L
# one (6019 minus the 39 that end the pipeline with no GT), so the two are
# comparable frame for frame. Changing any of them breaks that correspondence
# silently.
#
# NOTE the asymmetry this sets up: the attack perturbs POINTS only. Camera
# images stay pristine. So an LC result measures the camera branch rescuing a
# corrupted LiDAR input, not fusion robustness in general -- the same one-sided
# caveat that applies to the BEVFusion lidar-cam set.
# =============================================================================
_base_ = ['./FocalFormer3D_LC_test.py']

custom_imports = dict(
    imports=[
        'projects.mmdet3d_plugin',
        'projects.mmdet3d_plugin.datasets.pipelines.focalformer_img',
        'projects.mmdet3d_plugin.hooks.focalformer_gradient_hook',
        'projects.mmdet3d_plugin.hooks.noop_optimizer',
    ],
    allow_failed_imports=False)

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None
final_dim = [448, 800]
resize_lim = [0.5, 0.5]

gradients_output_dir = ''  # set via --cfg-options custom_hooks.1.save_path=...

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=999),
    sampler_seed=dict(type='DistSamplerSeedHook'))

# Index 1 is the gradient hook; the extraction script addresses it as
# custom_hooks.1.*, so do not reorder these.
custom_hooks = [
    dict(type='DisableObjectSampleHook', disable_after_epoch=0),
    dict(
        type='FocalFormerGradientHook',
        target_layer='neck',
        save_path=gradients_output_dir,
        normalize=True,
        save_interval=100),
]

train_pipeline = [
    dict(
        type='FFLoadMultiViewImage',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
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
    dict(
        type='FFImageAug3D',
        final_dim=final_dim,
        resize_lim=resize_lim,
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        # lidar2img and img_aug_matrix are what LiftSplatShoot needs; without
        # them the camera features splat to the wrong BEV cells.
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ]),
]

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT')

# shuffle=False: the attack resumes by filename via --skip_existing, and a
# deterministic order also makes a partial run reproducible.
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        # _delete_ is required, not decoration. The base chain's
        # train_dataloader is a CBGSDataset WRAPPING a NuScenesDataset, and
        # mmengine merges dicts rather than replacing them -- so without this
        # the merged dataset keeps type='CBGSDataset' and the base's inner
        # `dataset=` key, and NuScenesDataset is handed a `dataset` argument
        # it does not accept. CBGS resampling would also be wrong here: it
        # rebalances by class frequency, which changes both the frame order
        # and the frame COUNT, breaking the 1:1 correspondence with the L
        # gradient set that the identical filters exist to preserve.
        _delete_=True,
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_val_bevfusion.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

# Weights must not move: this is extraction, not training.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='NoOpOptimizer', lr=0.0))

# Flat schedule, replacing the base's training one. The base is a real
# training config: its param_scheduler anneals both LR and MOMENTUM, and a
# momentum scheduler against NoOpOptimizer raises "optimizer must support
# momentum when using momentum scheduler" before the first iteration. A list
# assignment replaces the base list outright, which is what is wanted -- there
# is nothing to schedule when no weights move.
epoch_num = 1
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0, begin=0, end=epoch_num)]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1)

val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None
test_cfg = None
