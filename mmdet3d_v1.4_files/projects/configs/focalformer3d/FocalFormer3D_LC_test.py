# =============================================================================
# FocalFormer3D LiDAR + Camera -- evaluation on nuScenes val
# =============================================================================
# Overrides the LiDAR-only test config rather than copying it. Everything the
# two share -- voxelization, SparseEncoder, SECOND/SECONDFPN, FocalDecoder,
# dataloaders, evaluator -- stays defined in exactly one place.
#
# Where this config came from
# ---------------------------
# There was no LC config in this project; the camera blocks in
# FocalFormer3D_L_v14.py are commented out. Every value below is either
# recovered from that commented block or read directly off
# FocalFormer3D_LC_ep6_mAP705_NDS731.pth, so nothing here is guesswork:
#
#   img_backbone  ResNet-50   conv1 is (64,3,7,7); layer1..4 are 3/4/6/3
#                             bottleneck blocks -- that is R50, not R18/R101.
#   img_neck      FPN         lateral_convs in_channels are 256/512/1024/2048,
#                             all out 256. Only 4 fpn_convs carry weights,
#                             which is consistent with num_outs=5: the fifth
#                             level is a weightless max-pool.
#   img_scale     448 x 800   the commented block writes this (800, 448); it
#                             is transposed below to (H, W) because that is
#                             what LiftSplatShoot expects. See the note there.
#
# The checkpoint also had to be converted -- it shipped in upstream key naming
# (pts_bbox_head.decoder.*.attentions.0.attn.*), the same form the L
# checkpoint was in before convert_focalformer_ckpt.py. Use
# FocalFormer3D_LC_ep6_converted.pth, NOT the original.
#
# How the camera actually reaches the BEV
# ---------------------------------------
# FocalEncoder.forward does not consume images directly. With cam_lss truthy
# (and not the literal string 'proj') it builds per-camera rotations and
# translations by INVERTING img_metas['lidar2img'], then hands those to
# LiftSplatShoot. LiftSplatShoot.get_geometry separately checks for
# 'img_aug_matrix' to undo the resize/crop this pipeline applies. So both keys
# must survive into metainfo -- dropping either one leaves the camera features
# splatted to the wrong BEV cells, which costs mAP without raising anything.
# That is the single most likely way this config goes quietly wrong.
#
# VALIDATE BEFORE USING: the clean eval must land near mAP 0.705 / NDS 0.731,
# the checkpoint's own reference. Anything much below that means the image
# branch is mis-wired, and gradients or an attack built on it would be
# meaningless. (For calibration: the L config reproduces its own reference to
# within 0.6 of a point, so treat a gap larger than ~1 point as a failure.)
# =============================================================================
_base_ = ['./FocalFormer3d_L_test.py']

# The base imports only projects.mmdet3d_plugin. The camera loader and resize
# live in a module of their own -- see focalformer_img.py for why they could
# not simply be taken from projects/BEVFusion (8 registry name collisions) or
# from the plugin's own legacy ImageAug3D (asserts on mmdet 2.x keys).
custom_imports = dict(
    imports=[
        'projects.mmdet3d_plugin',
        'projects.mmdet3d_plugin.datasets.pipelines.focalformer_img',
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

# (H, W), NOT (W, H). The commented block wrote this as (800, 448), but
# LiftSplatShoot is height-first -- its own default is (900, 1600), nuScenes
# HxW. Passing (800, 448) builds a frustum of (41, 200, 112, 3) against the
# checkpoint's (41, 112, 200, 3), i.e. the image plane transposed. It loads
# with a shape warning and splats every camera feature to the wrong BEV cell.
img_scale = (448, 800)
final_dim = [448, 800]          # (H, W), the order ImageAug3D wants too
# nuScenes images are 1600x900. 0.5 gives 800x450; ImageAug3D then crops 2
# rows off the top to reach 448, which is why bot_pct_lim is [0, 0].
resize_lim = [0.5, 0.5]

multistage_heatmap = 1
inter_channel = 128
extra_feat = True

# ---------------------------------------------------------------------------
# Model: add the image branch, switch the fusion neck on
# ---------------------------------------------------------------------------
model = dict(
    input_img=True,
    # bgr_to_rgb=False, deliberately. FFLoadMultiViewImage decodes with
    # channel_order='rgb', so the images are ALREADY RGB here. The original
    # img_norm_cfg's to_rgb=True describes the mmdet 2.x pipeline doing that
    # conversion; repeating it would feed BGR to a model trained on RGB.
    data_preprocessor=dict(
        type='mmdet3d.Det3DDataPreprocessor',
        voxel=False,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    imgpts_neck=dict(
        input_img=True,
        iterbev_wo_img=False,
        cam_lss=True,
        # Two fusion blocks, not one. The L config sets num_layers to
        # multistage_heatmap (=1); the LC checkpoint carries fusion_blocks.0
        # AND fusion_blocks.1, which is the FocalEncoder default of 2. These
        # are separate knobs that happen to coincide for L -- multistage_heatmap
        # stays 1 here, because the decoder head matched at that value.
        num_layers=2,
        # 'bevfusion', not the L config's 'bevfusionmb2'. The two branch in
        # FocalEncoder.__init__: 'bevfusionmb2' builds P_IML/P_integration as
        # MobileNetV2 InvertedResiduals (conv.0.0, conv.0.1, ...), 'bevfusion'
        # builds them as LocalContextAttentionBlock + ConvBNReLU, which is what
        # carries key_project/query_project. The LC checkpoint has the latter,
        # so the two checkpoints genuinely use different fusion variants --
        # inheriting the L value silently mismatches 146 tensors.
        iterbev='bevfusion',
        # Required whenever cam_lss is on, and it costs no parameters.
        # FocalEncoder passes need_projbev=not cam_lss, so with LSS the I2P
        # block is deliberately not built -- the camera features are already in
        # BEV and there is nothing to project. The checkpoint agrees: it has no
        # I2P_block weights. But FocalEncoderLayer.forward only honours that in
        # its iter_bev_cam branch; the other path calls self.I2P_block(...)
        # unguarded and dies on 'NoneType' object is not callable. With
        # iter_bev_cam=True both layers take `I2P_feat = img_feat`, i.e. use the
        # LSS BEV output directly, which is the intended behaviour.
        iter_bev_cam=True,
        pc_range=point_cloud_range,
        img_scale=img_scale),
    pts_bbox_head=dict(
        # The checkpoint carries heatmap_head_img.0.* AND .1.*, both real.
        # FocalDecoder.__init__ does `if reuse_first_heatmap: multistage_heatmap
        # += 1` and then appends None at index 0, so the L settings
        # (multistage_heatmap=1, reuse=True) build [None, copy] -- weights only
        # at .1, with .0 left unclaimed. Two real stages means the LC head was
        # built with multistage_heatmap=2 and reuse off.
        #
        # Note this is the HEAD's copy of multistage_heatmap only. imgpts_neck
        # keeps 1 (it already matches the checkpoint exactly at that value) and
        # gets its two fusion blocks from the explicit num_layers above.
        multistage_heatmap=2,
        reuse_first_heatmap=False),
)

# ---------------------------------------------------------------------------
# Pipeline: load the six views, resize, and keep the projection metadata
# ---------------------------------------------------------------------------
# Deliberately NOT MultiScaleFlipAug3D, which the LiDAR-only test pipeline
# uses: it wraps the point transforms in a nested structure that carries no
# image branch, so the camera metadata would never reach the model. This
# mirrors the BEVFusion lidar-cam pipeline, which is the known-good camera path
# in this repo.
test_pipeline = [
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
    dict(
        type='FFImageAug3D',
        final_dim=final_dim,
        resize_lim=resize_lim,
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points'],
        # lidar2img and img_aug_matrix are load-bearing -- see the header.
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ]),
]

# ---------------------------------------------------------------------------
# Dataloader: camera modality + the six CAM_* prefixes
# ---------------------------------------------------------------------------
# data_prefix must name every camera, or BEVLoadMultiViewImageFromFiles cannot
# resolve img_path and the run dies at the first sample.
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT')

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        modality=input_modality,
        data_prefix=data_prefix))
test_dataloader = val_dataloader
