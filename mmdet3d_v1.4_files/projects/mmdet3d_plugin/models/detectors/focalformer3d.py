'''
# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

# =============================================================================
# focalformer3d.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - DETECTORS from mmdet.models          → MODELS from mmdet3d.registry
#   - builder.build_neck(cfg)              → MODELS.build(cfg)
#   - Voxelization from mmdet3d.ops        → mmcv.ops.Voxelization
#   - force_fp32 from mmcv.runner          → REMOVED (AmpOptimWrapper handles)
#   - Box3DMode, Coord3DMode, bbox3d2result,
#     show_result from mmdet3d.core        → mmdet3d.structures
#   - multi_apply from mmdet.core          → mmdet.models.utils
#   - mmcv.parallel.DataContainer          → REMOVED (not needed in 1.1+)
#   - import mmcv (general)                → REMOVED (unused)
#   - import pdb                           → REMOVED
#   - forward_train()                      → loss(batch_inputs_dict, batch_data_samples)
#   - simple_test()                        → predict(batch_inputs_dict, batch_data_samples)
#   - aug_test()                           → aug_test (internal, called from predict)
#   - print() statements                   → print/warning
#   - merge_aug_bboxes_3d from mmdet3d.core→ from local plugin
# =============================================================================
import logging
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)
print('[focalformer3d] Loading module...')

# --- mmcv.ops (Voxelization moved from mmdet3d.ops) ---
try:
    from mmcv.ops import Voxelization
    print('[focalformer3d] ✓ Imported Voxelization from mmcv.ops')
except ImportError as e:
    print(f'[focalformer3d] ✗ Voxelization: {e}')
    raise

# --- Registry: DETECTORS → MODELS ---
try:
    from mmdet3d.registry import MODELS
    print('[focalformer3d] ✓ Imported MODELS from mmdet3d.registry')
except ImportError as e:
    print(f'[focalformer3d] ✗ Registry imports: {e}')
    raise

# --- MVXTwoStageDetector base class ---
try:
    from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
    print('[focalformer3d] ✓ Imported MVXTwoStageDetector')
except ImportError as e:
    print(f'[focalformer3d] ✗ MVXTwoStageDetector: {e}')
    raise

# --- Structures (old: mmdet3d.core) ---
try:
    from mmdet3d.structures import bbox3d2result
    print('[focalformer3d] ✓ Imported bbox3d2result '
                'from mmdet3d.structures')
except ImportError:
    # bbox3d2result may have been removed in some mmdet3d versions
    # since Det3DDataSample handles results directly.
    print('[focalformer3d] bbox3d2result not found — using fallback')

    def bbox3d2result(bboxes, scores, labels, attrs=None):
        """Convert detection results to a list of numpy arrays."""
        result_dict = dict(
            boxes_3d=bboxes,
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu())
        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()
        return result_dict

# --- Det3DDataSample / InstanceData for result packaging ---
try:
    from mmdet3d.structures import Det3DDataSample
    print('[focalformer3d] ✓ Imported Det3DDataSample')
except ImportError as e:
    print(f'[focalformer3d] ✗ Det3DDataSample: {e}')
    raise

try:
    from mmengine.structures import InstanceData
    print('[focalformer3d] ✓ Imported InstanceData from mmengine')
except ImportError as e:
    print(f'[focalformer3d] ✗ InstanceData: {e}')
    raise

# --- Local project imports (unchanged — these are project-internal) ---
try:
    from projects.mmdet3d_plugin.models.utils.time_utils import T
    print('[focalformer3d] ✓ Imported T from time_utils')
except ImportError:
    # T is optional — only used for timing in simple_test (commented out)
    T = None
    print('[focalformer3d] time_utils.T not available — timing disabled')

try:
    from projects.mmdet3d_plugin.core.post_processing.merge_augs import merge_aug_bboxes_3d
    print('[focalformer3d] ✓ Imported merge_aug_bboxes_3d')
except ImportError as e:
    print(f'[focalformer3d] ✗ merge_aug_bboxes_3d: {e}')
    raise


# ===================================================================
# FocalFormer3D Detector
# ===================================================================
@MODELS.register_module()
class FocalFormer3D(MVXTwoStageDetector):
    """FocalFormer3D: Focusing on Hard Instance for 3D Object Detection.

    Refactored for mmdet3d >= 1.1 (mmengine-based).

    Key changes from old version:
      - ``forward_train()`` → ``loss()``
      - ``simple_test()`` → ``predict()``
      - Data comes via ``batch_inputs_dict`` and ``batch_data_samples``
      - ``@force_fp32`` removed (use AmpOptimWrapper)
      - ``builder.build_neck`` → ``MODELS.build``
    """

    def __init__(self,
                 freeze_img=False,
                 freeze_img_level=None,
                 freeze_camlss=False,
                 freeze_pts=False,
                 trainneck_ms=False,
                 train_middle_encoder=False,
                 pts_pillar_layer=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 imgpts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_img=True,
                 use_grid_mask=False,
                 input_pts=True,
                 init_cfg=None,
                 data_preprocessor=None,
                 **kwargs):
        # NOTE: In mmdet3d 1.4, MVXTwoStageDetector no longer accepts
        # pts_voxel_layer as an __init__ kwarg (it's handled by
        # data_preprocessor in the standard pipeline). Since FocalFormer3D
        # does its own voxelization, we intercept pts_voxel_layer here
        # and build it ourselves, then call super() without it.
        super(FocalFormer3D, self).__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs)

        print(f'[FocalFormer3D] Building: '
                    f'freeze_img={freeze_img}, freeze_pts={freeze_pts}, '
                    f'input_img={input_img}, input_pts={input_pts}, '
                    f'use_grid_mask={use_grid_mask}')

        # Build voxel layers ourselves (removed from super() call)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
            print(f'[FocalFormer3D] Built pts_voxel_layer')

        if pts_pillar_layer:
            self.pts_pillar_layer = Voxelization(**pts_pillar_layer)
            print(f'[FocalFormer3D] Built pts_pillar_layer')

        self.freeze_img_level = freeze_img_level
        self.freeze_camlss = freeze_camlss

        # --- builder.build_neck → MODELS.build ---
        self.imgpts_neck = MODELS.build(imgpts_neck)
        print(f'[FocalFormer3D] Built imgpts_neck: '
                     f'{imgpts_neck["type"]}')

        self.freeze_img = freeze_img
        self.freeze_pts = freeze_pts
        self.trainneck_ms = trainneck_ms
        self.train_middle_encoder = train_middle_encoder

        self.input_img = input_img
        self.input_pts = input_pts

        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
            print('[FocalFormer3D] GridMask enabled')

        self.apply_dynamic_voxelize = 'Dynamic' in pts_voxel_encoder['type']
        print(f'[FocalFormer3D] apply_dynamic_voxelize='
                     f'{self.apply_dynamic_voxelize}')

        print('[FocalFormer3D] ✓ Built successfully')

    def init_weights(self):
        """Initialize model weights."""
        super(FocalFormer3D, self).init_weights()

    def load_state_dict(self, state_dict, strict=True):
        """Override to remap old mmcv decoder keys to new mmdet decoder keys.

        Old (mmcv 0.x / mmdet 2.x):
            decoder.N.layers.L.attentions.0  -> self_attn
            decoder.N.layers.L.attentions.1  -> cross_attn
            decoder.N.layers.L.ffns.0        -> ffn

        New (mmdet 3.x):
            decoder.N.layers.L.self_attn
            decoder.N.layers.L.cross_attn
            decoder.N.layers.L.ffn
        """
        new_state_dict = {}
        remapped = 0
        for key, value in state_dict.items():
            new_key = key
            if 'pts_bbox_head.decoder.' in key:
                # attentions.0 -> self_attn
                new_key = new_key.replace('.attentions.0.', '.self_attn.')
                # attentions.1 -> cross_attn
                new_key = new_key.replace('.attentions.1.', '.cross_attn.')
                # ffns.0 -> ffn
                new_key = new_key.replace('.ffns.0.', '.ffn.')
                if new_key != key:
                    remapped += 1
            new_state_dict[new_key] = value

        if remapped > 0:
            print(f'[FocalFormer3D] Remapped {remapped} decoder keys '
                        f'(old mmcv -> new mmdet format)')

        return super().load_state_dict(new_state_dict, strict=strict)

        if self.input_img and self.freeze_img:
            if self.with_img_backbone:
                if self.freeze_img_level:
                    param_levels = [
                        ['conv1', 'bn1'], ['layer1'], ['layer2'],
                        ['layer3'], ['layer4']
                    ]
                    for i in range(self.freeze_img_level):
                        for pn in param_levels[i]:
                            # print → print
                            print(f'[FocalFormer3D] Freezing image {pn}')
                            for param in self.img_backbone.get_submodule(pn).parameters():
                                param.requires_grad = False
                else:
                    print('[FocalFormer3D] Freezing entire img_backbone')
                    for param in self.img_backbone.parameters():
                        param.requires_grad = False

            if self.with_img_neck:
                print('[FocalFormer3D] Freezing img_neck')
                for param in self.img_neck.parameters():
                    param.requires_grad = False

            if self.freeze_camlss and hasattr(self.imgpts_neck, 'cam_lss'):
                print('[FocalFormer3D] Freezing imgpts_neck.cam_lss')
                for param in self.imgpts_neck.cam_lss.parameters():
                    param.requires_grad = False

        if self.freeze_pts:
            print('[FocalFormer3D] Freezing pts branches (partial)')
            for name, param in self.named_parameters():
                if 'pts' in name and 'pts_bbox_head' not in name and 'imgpts_neck' not in name:
                    if self.trainneck_ms:
                        if 'pts_backbone' in name:
                            continue
                        if 'pts_neck' in name:
                            continue
                    if self.train_middle_encoder:
                        if 'pts' in name:
                            continue
                    param.requires_grad = False

            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

            if not self.train_middle_encoder:
                self.pts_voxel_layer.apply(fix_bn)
                self.pts_voxel_encoder.apply(fix_bn)
                self.pts_middle_encoder.apply(fix_bn)

            if not self.trainneck_ms:
                self.pts_backbone.apply(fix_bn)
                if self.with_pts_neck:
                    self.pts_neck.apply(fix_bn)

        if not self.input_pts:
            print('[FocalFormer3D] No pts input — nullifying pts modules')
            self.voxelize = None
            self.pts_voxel_encoder = None
            self.pts_middle_encoder = None
            self.pts_backbone = None
            if self.with_pts_neck:
                self.pts_neck = None

        print('[FocalFormer3D] init_weights complete')

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            if self.use_grid_mask and self.training:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img.float())
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        print(f'[FocalFormer3D.extract_img_feat] '
                     f'img_feats type={type(img_feats).__name__}, '
                     f'len={len(img_feats) if isinstance(img_feats, (list, tuple)) else "N/A"}')
        return img_feats

    def extract_pts_feat(self, pts, img_feats=None, img_metas=None):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None

        if self.apply_dynamic_voxelize:
            voxels, coors = self.dynamic_voxelize(pts)
            voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors)
            batch_size = coors[-1, 0] + 1
            coors = feature_coors  # update
        else:
            voxels, num_points, coors = self.voxelize(pts, voxel_type='voxel')
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0] + 1

        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)

        if self.with_pts_neck:
            x = self.pts_neck(x)
        else:
            x = [x]

        print(f'[FocalFormer3D.extract_pts_feat] '
                     f'num_features={len(x)}, '
                     f'feat0 shape={x[0].shape if isinstance(x[0], torch.Tensor) else "N/A"}')
        return x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points.

        This is the shared feature extraction used by both loss() and predict().
        """
            
        # ... rest of extract_feat runs normally ...
        if self.input_img:
            img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = [None]

        if self.input_pts:
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        else:
            pts_feats = [None]

        new_img_feat, new_pts_feat = self.imgpts_neck(
            img_feats[0], pts_feats[0], img_metas)

        print(f'[FocalFormer3D.extract_feat] '
                     f'new_img_feat={type(new_img_feat).__name__}, '
                     f'new_pts_feat={type(new_pts_feat).__name__}')
        return (new_img_feat, new_pts_feat)

    @torch.no_grad()
    # NOTE: @force_fp32() REMOVED — handled by AmpOptimWrapper in mmdet3d 1.1+
    def voxelize(self, points, voxel_type='voxel'):
        """Apply voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            voxel_type (str): 'voxel' or 'pillar'.

        Returns:
            tuple: voxels, num_points, coors_batch
        """
        assert voxel_type == 'voxel' or voxel_type == 'pillar'
        voxels, coors, num_points = [], [], []

        for res in points:
            if voxel_type == 'voxel':
                res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            elif voxel_type == 'pillar':
                res_voxels, res_coors, res_num_points = self.pts_pillar_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []

        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)

        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    # NOTE: @force_fp32() REMOVED — handled by AmpOptimWrapper in mmdet3d 1.1+
    def dynamic_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provides a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)

        points = torch.cat(points, dim=0)
        coors_batch = []

        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)

        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    # =================================================================
    # NEW API: loss() replaces forward_train()
    # =================================================================
    # def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
    #     """Calculate losses from a batch of inputs and data samples.

    #     This replaces the old ``forward_train()`` method.

    #     Args:
    #         batch_inputs_dict (dict): Contains 'points', 'imgs', etc.
    #         batch_data_samples (list[:obj:`Det3DDataSample`]): Each
    #             contains gt_instances_3d with bboxes_3d and labels_3d,
    #             and metainfo (the old img_metas).

    #     Returns:
    #         dict: A dictionary of loss components.
    #     """
    #     # --- Unpack inputs (new API) ---
    #     points = batch_inputs_dict.get('points', None)
    #     imgs = batch_inputs_dict.get('imgs', None)

    #     # --- Unpack ground truths and metadata from data samples ---
    #     gt_bboxes_3d = []
    #     gt_labels_3d = []
    #     img_metas = []

    #     for data_sample in batch_data_samples:
    #         img_metas.append(data_sample.metainfo)
    #         gt_bboxes_3d.append(data_sample.gt_instances_3d.bboxes_3d)
    #         gt_labels_3d.append(data_sample.gt_instances_3d.labels_3d)

    #     print(f'[FocalFormer3D.loss] batch_size={len(img_metas)}, '
    #                  f'num_gt_per_sample='
    #                  f'{[len(l) for l in gt_labels_3d]}')

    #     # --- Feature extraction (same as before) ---
    #     img_feats, pts_feats = self.extract_feat(
    #         points, img=imgs, img_metas=img_metas)

    #     # --- Compute losses ---
    #     losses = dict()
    #     losses_pts = self.forward_pts_train(
    #         pts_feats, img_feats, gt_bboxes_3d,
    #         gt_labels_3d, img_metas)
    #     losses.update(losses_pts)

    #     return losses

    # def forward_pts_train(self,
    #                       pts_feats,
    #                       img_feats,
    #                       gt_bboxes_3d,
    #                       gt_labels_3d,
    #                       img_metas,
    #                       gt_bboxes_ignore=None):
    #     """Forward function for point cloud branch.

    #     Args:
    #         pts_feats (list[torch.Tensor]): Features of point cloud branch
    #         img_feats: Image features after fusion.
    #         gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
    #             boxes for each sample.
    #         gt_labels_3d (list[torch.Tensor]): Ground truth labels for
    #             boxes of each sample.
    #         img_metas (list[dict]): Meta information of samples.
    #         gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
    #             boxes to be ignored. Defaults to None.

    #     Returns:
    #         dict: Losses of each branch.
    #     """
    #     outs = self.pts_bbox_head(
    #         pts_feats, img_feats, img_metas,
    #         gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
    #     loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
    #     losses = self.pts_bbox_head.loss(*loss_inputs)

    #     print(f'[FocalFormer3D.forward_pts_train] '
    #                  f'loss keys: {list(losses.keys())}')
    #     return losses

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs and data samples."""
        from mmengine.structures import InstanceData # Ensure this is imported

        points = batch_inputs_dict.get('points', None)
        imgs = batch_inputs_dict.get('imgs', None)

        # --- v1.x Compliant GT Unpacking ---
        batch_gt_instances_3d = []
        img_metas = []


        # =====================================================================
        # 🚨 TIME MACHINE (Input Boundary): v1.x GTs -> v0.x Model
        # =====================================================================
        # Detect dataset: NuScenes needs dim swap, Waymo does not
        _is_nuscenes = (hasattr(self, 'test_cfg') and self.test_cfg is not None
                        and self.test_cfg.get('pts', {}).get('dataset', '') == 'nuScenes')

        for data_sample in batch_data_samples:
            bboxes = data_sample.gt_instances_3d.bboxes_3d
            if bboxes is not None and len(bboxes) > 0:
                tensor = bboxes.tensor
                # NuScenes only: mmdet3d 1.4 stores dims as (l, w, h) but
                # FocalFormer3D (trained on 0.x) expects (w, l, h)
                if _is_nuscenes:
                    l_v1 = tensor[:, 3].clone()
                    w_v1 = tensor[:, 4].clone()
                    tensor[:, 3] = w_v1  # Now it's width
                    tensor[:, 4] = l_v1  # Now it's length
            
            img_metas.append(data_sample.metainfo)
            
            # Extract raw boxes and labels
            #bboxes = data_sample.gt_instances_3d.bboxes_3d
            labels = data_sample.gt_instances_3d.labels_3d
            
            # Wrap them in InstanceData (v1.x requirement for Assigners)
            gt_instances = InstanceData()
            gt_instances.bboxes_3d = bboxes
            gt_instances.labels_3d = labels
            batch_gt_instances_3d.append(gt_instances)

        print(f'[FocalFormer3D.loss] batch_size={len(img_metas)}')

        # Feature extraction
        img_feats, pts_feats = self.extract_feat(
            points, img=imgs, img_metas=img_metas)

        # Compute losses (Pass the InstanceData list, NOT raw box lists)
        losses = dict()
        losses_pts = self.forward_pts_train(
            pts_feats, img_feats, 
            batch_gt_instances_3d, # <--- Pass the wrapped instances here
            None, # gt_labels_3d is now inside batch_gt_instances_3d
            img_metas)
        losses.update(losses_pts)

        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          batch_gt_instances_3d, # <--- Updated argument
                          gt_labels_3d,          # <--- Will be None
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch."""
        
        # Unpack for the head (Some old heads still want separate lists)
        # We try passing them separated first, as your DeepInteractionDecoder 
        # get_targets() expects lists of boxes and labels.
        gt_bboxes_3d = [inst.bboxes_3d for inst in batch_gt_instances_3d]
        gt_labels_3d = [inst.labels_3d for inst in batch_gt_instances_3d]

        outs = self.pts_bbox_head(
            pts_feats, img_feats, img_metas,
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
            
        # 🚨 CRITICAL FIX for Assigner: 
        # Pass the wrapped batch_gt_instances_3d to the loss function, 
        # NOT the raw gt_bboxes_3d/gt_labels_3d lists.
        loss_inputs = [batch_gt_instances_3d, outs] 
        # Note: If your DeepInteractionDecoder.loss() strictly expects 
        # (gt_bboxes_3d, gt_labels_3d, preds_dicts), you will need to update 
        # DeepInteractionDecoder.loss() to accept batch_gt_instances_3d instead.
        
        try:
            # Try v1.x standard format first
            losses = self.pts_bbox_head.loss_by_feat(*loss_inputs)
        except AttributeError:
             # Fallback if your custom head uses .loss() instead of .loss_by_feat()
             try:
                 losses = self.pts_bbox_head.loss(batch_gt_instances_3d, outs)
             except TypeError:
                 # If it strictly requires separate arguments, we fall back to raw lists
                 # (But this is what caused the batch_valid_gt_mask error)
                 losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)

        return losses
    # =================================================================
    # NEW API: predict() replaces simple_test()
    # =================================================================
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples.

        This replaces the old ``simple_test()`` method.

        Args:
            batch_inputs_dict (dict): Contains 'points', 'imgs', etc.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each
                contains metainfo (the old img_metas).

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results with
                pred_instances_3d populated.
        """
        # --- Unpack inputs ---
        points = batch_inputs_dict.get('points', None)
        imgs = batch_inputs_dict.get('imgs', None)
        img_metas = [ds.metainfo for ds in batch_data_samples]


        # Optionally unpack gt for oracle/debug testing
        gt_bboxes_3d = None
        gt_labels_3d = None
        if hasattr(batch_data_samples[0], 'gt_instances_3d'):
            try:
                gt_bboxes_3d = [ds.gt_instances_3d.bboxes_3d for ds in batch_data_samples]
                gt_labels_3d = [ds.gt_instances_3d.labels_3d for ds in batch_data_samples]
            except AttributeError:
                pass

        print(f'[FocalFormer3D.predict] batch_size={len(img_metas)}')

        # --- Feature extraction ---
        img_feats, pts_feats = self.extract_feat(
            points, img=imgs, img_metas=img_metas)

        # --- Point cloud branch inference ---
        bbox_pts = self.simple_test_pts(
            pts_feats, img_feats, img_metas,
            rescale=kwargs.get('rescale', False),
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d)

        # --- Package results into Det3DDataSample ---
        results_list = self.add_pred_to_datasample(
            batch_data_samples, bbox_pts)

        return results_list

    

    def simple_test_pts(self, x, x_img, img_metas, rescale=False,
                        gt_bboxes_3d=None, gt_labels_3d=None, **kwargs):
        """Test function of point cloud branch.

        Args:
            x: Point cloud features.
            x_img: Image features.
            img_metas (list[dict]): Meta information.
            rescale (bool): Whether to rescale results.
            gt_bboxes_3d: Optional GT boxes for debug/oracle testing.
            gt_labels_3d: Optional GT labels for debug/oracle testing.

        Returns:
            list[dict]: Detection results per sample.
        """
        outs = self.pts_bbox_head(
            x, x_img, img_metas,
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, **kwargs)

        if True:
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
        # In simple_test_pts, after bbox_list = self.pts_bbox_head.get_bboxes(...)
        for bboxes, scores, labels in bbox_list:
            print(f'  [DIAG] After get_bboxes: {len(bboxes)} detections out of 300 proposals')
            print(f'  [DIAG] Score range: [{scores.min():.4f}, {scores.max():.4f}]')
            if len(bboxes) > 0:
                centers = bboxes.gravity_center
                print(f'  [DIAG] Center X range: [{centers[:,0].min():.1f}, {centers[:,0].max():.1f}]')
                print(f'  [DIAG] Center Y range: [{centers[:,1].min():.1f}, {centers[:,1].max():.1f}]')
                print(f'  [DIAG] Center Z range: [{centers[:,2].min():.1f}, {centers[:,2].max():.1f}]')
            break  # Only first sample
        else:
            bbox_list = self.pts_bbox_head.get_heatmap_bboxes(
                outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        print(f'[FocalFormer3D.simple_test_pts] '
                     f'{len(bbox_results)} samples')
        return bbox_results

    def aug_test(self, batch_inputs_dict_list, batch_data_samples_list,
                 rescale=False):
        """Test function with augmentation.

        In mmdet3d 1.1+, aug_test is called differently than in 1.0.
        This method handles both patterns.

        Args:
            batch_inputs_dict_list: List of input dicts per augmentation.
            batch_data_samples_list: List of data samples per augmentation.
            rescale (bool): Whether to rescale results.

        Returns:
            list[dict]: Merged detection results.
        """
        precompute = False

        if not precompute:
            print('[FocalFormer3D.aug_test] Precomputing aug_test ...')

            # Extract points and images for each augmentation
            all_pts_feats = []
            all_img_feats = []
            all_img_metas = []

            for batch_inputs, batch_samples in zip(
                    batch_inputs_dict_list, batch_data_samples_list):
                points = batch_inputs.get('points', None)
                imgs = batch_inputs.get('imgs', None)
                img_metas = [ds.metainfo for ds in batch_samples]

                img_feats, pts_feats = self.extract_feat(
                    points, img=imgs, img_metas=img_metas)
                all_pts_feats.append(pts_feats)
                all_img_feats.append(img_feats)
                all_img_metas.append(img_metas)

            bbox_list = dict()
            if self.with_pts_bbox:
                bbox_pts = self.aug_test_pts(
                    all_pts_feats, all_img_feats, all_img_metas,
                    rescale=rescale)
                bbox_list.update(pts_bbox=bbox_pts)
        else:
            print('[FocalFormer3D.aug_test] Using precomputed results ...')
            all_img_metas = [
                [ds.metainfo for ds in batch_samples]
                for batch_samples in batch_data_samples_list
            ]
            bbox_list = dict()
            bbox_pts = self.aug_test_pts(
                None, None, all_img_metas, rescale=rescale)
            bbox_list.update(pts_bbox=bbox_pts)

        return [bbox_list]

    def aug_test_pts(self, xs, x_imgs, img_metas, rescale=False):
        """Test function of point cloud branch with augmentation.

        Args:
            xs (list): Point features per augmentation (or None if precomputed).
            x_imgs (list): Image features per augmentation (or None).
            img_metas (list[list[dict]]): Meta info per augmentation.
            rescale (bool): Whether to rescale results.

        Returns:
            dict: Merged bounding box results.
        """
        if xs is not None:
            # only support aug_test for one sample
            aug_bboxes = []
            for x, x_img, img_meta in zip(xs, x_imgs, img_metas):
                outs = self.pts_bbox_head(x, x_img, img_meta)
                bbox_list = self.pts_bbox_head.get_bboxes(
                    outs, img_meta, rescale=rescale)
                bbox_list = [
                    dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                    for bboxes, scores, labels in bbox_list
                ]
                aug_bboxes.append(bbox_list[0])

            # after merging, bboxes will be rescaled to the original image size
            merged_bboxes = merge_aug_bboxes_3d(
                aug_bboxes, img_metas,
                self.pts_bbox_head.test_cfg)
        else:
            merged_bboxes = merge_aug_bboxes_3d(
                None, img_metas,
                self.pts_bbox_head.test_cfg)

        print(f'[FocalFormer3D.aug_test_pts] '
                     f'merged {len(aug_bboxes) if xs else 0} augmentations')
        return merged_bboxes

    # def add_pred_to_datasample(self, batch_data_samples, bbox_results):
    #     """Add prediction results to Det3DDataSample.

    #     This converts the old-style bbox_results dicts into the new
    #     Det3DDataSample format expected by mmdet3d 1.1+.

    #     Args:
    #         batch_data_samples (list[:obj:`Det3DDataSample`]): Data samples.
    #         bbox_results (list[dict]): Prediction results from
    #             simple_test_pts.

    #     Returns:
    #         list[:obj:`Det3DDataSample`]: Updated data samples.
    #     """
    #     for data_sample, bbox_result in zip(batch_data_samples, bbox_results):
    #         pred_instances_3d = InstanceData()

    #         # Handle both old-style and new-style keys
    #         bboxes_3d = bbox_result.get('bboxes_3d', bbox_result.get('boxes_3d', bbox_result.get('bboxes', None)))
    #         scores_3d = bbox_result.get('scores_3d', bbox_result.get('scores', None))
    #         labels_3d = bbox_result.get('labels_3d', bbox_result.get('labels', None))

    #         print(f'[add_pred_to_datasample] bbox_result keys: {bbox_result.keys()}, '
    #                      f'bboxes_3d type: {type(bboxes_3d)}, '
    #                      f'has __len__: {hasattr(bboxes_3d, "__len__") if bboxes_3d is not None else "None"}')
    #         # One-time debug: log label distribution for first sample
    #         if labels_3d is not None and hasattr(labels_3d, 'unique'):
    #             if not hasattr(self, '_logged_label_info'):
    #                 self._logged_label_info = True
    #                 print(f'[add_pred_to_datasample] FIRST SAMPLE label info: '
    #                                f'unique={labels_3d.unique().tolist()}, '
    #                                f'dtype={labels_3d.dtype}, '
    #                                f'min={labels_3d.min().item()}, max={labels_3d.max().item()}, '
    #                                f'count={len(labels_3d)}')

    #         if bboxes_3d is not None and hasattr(bboxes_3d, '__len__') and len(bboxes_3d) > 0:
    #             # Ensure labels are long type and in valid range [0, num_classes)
    #             labels_3d = labels_3d.long()
    #             num_classes = 10  # nuScenes
    #             valid_mask = (labels_3d >= 0) & (labels_3d < num_classes)
    #             if not valid_mask.all():
    #                 n_invalid = (~valid_mask).sum().item()
    #                 invalid_labels = labels_3d[~valid_mask].unique().tolist()
    #                 print(f'[add_pred_to_datasample] Filtering {n_invalid} '
    #                                f'predictions with out-of-range labels: {invalid_labels}')
    #                 valid_idx = valid_mask.nonzero(as_tuple=True)[0]
    #                 bboxes_3d = bboxes_3d[valid_idx]
    #                 scores_3d = scores_3d[valid_idx]
    #                 labels_3d = labels_3d[valid_idx]
    #             pred_instances_3d.bboxes_3d = bboxes_3d
    #             pred_instances_3d.scores_3d = scores_3d
    #             pred_instances_3d.labels_3d = labels_3d
    #         else:
    #             # Empty prediction fallback
    #             import torch
    #             from mmdet3d.structures import LiDARInstance3DBoxes
    #             pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
    #                 torch.zeros((0, 9), dtype=torch.float32))
    #             pred_instances_3d.scores_3d = torch.zeros((0,), dtype=torch.float32)
    #             pred_instances_3d.labels_3d = torch.zeros((0,), dtype=torch.int64)

    #         data_sample.pred_instances_3d = pred_instances_3d

    #         # Add empty 2D predictions if not present (required by evaluator)
    #         if not hasattr(data_sample, 'pred_instances') or data_sample.pred_instances is None:
    #             data_sample.pred_instances = InstanceData()

    #     return batch_data_samples

    def add_pred_to_datasample(self, batch_data_samples, bbox_results):
        """Add prediction results to Det3DDataSample."""
        # =====================================================================
        # Detect dataset: NuScenes needs dim swap + yaw fix, Waymo does not
        _is_nuscenes = (hasattr(self, 'test_cfg') and self.test_cfg is not None
                        and self.test_cfg.get('pts', {}).get('dataset', '') == 'nuScenes')

        for result in bbox_results: 
            bboxes_3d = result.get('bboxes_3d', result.get('boxes_3d', result.get('bboxes', None)))
            if bboxes_3d is not None and len(bboxes_3d) > 0:
                tensor = bboxes_3d.tensor
                if _is_nuscenes:
                    # NuScenes: swap (w, l) -> (l, w) for evaluator
                    w_v0 = tensor[:, 3].clone()
                    l_v0 = tensor[:, 4].clone()
                    tensor[:, 3] = l_v0  # Now it's length
                    tensor[:, 4] = w_v0  # Now it's width
                    # NuScenes: yaw convention fix (v0.x -> v1.x evaluator)
                    tensor[:, 6] = -tensor[:, 6] - (np.pi / 2)
        # =====================================================================

        from mmengine.structures import InstanceData

        for data_sample, bbox_result in zip(batch_data_samples, bbox_results):
            pred_instances_3d = InstanceData()

            # Now we extract the mathematically corrected boxes
            bboxes_3d = bbox_result.get('bboxes_3d', bbox_result.get('boxes_3d', bbox_result.get('bboxes', None))) # 2
            scores_3d = bbox_result.get('scores_3d', bbox_result.get('scores', None))
            labels_3d = bbox_result.get('labels_3d', bbox_result.get('labels', None))

            if bboxes_3d is not None and hasattr(bboxes_3d, '__len__') and len(bboxes_3d) > 0:
                labels_3d = labels_3d.long()
                num_classes = 10  # nuScenes
                valid_mask = (labels_3d >= 0) & (labels_3d < num_classes)
                if not valid_mask.all():
                    n_invalid = (~valid_mask).sum().item()
                    invalid_labels = labels_3d[~valid_mask].unique().tolist()
                    print(f'[add_pred_to_datasample] Filtering {n_invalid} '
                                   f'predictions with out-of-range labels: {invalid_labels}')
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                    bboxes_3d = bboxes_3d[valid_idx]
                    scores_3d = scores_3d[valid_idx]
                    labels_3d = labels_3d[valid_idx]
                pred_instances_3d.bboxes_3d = bboxes_3d
                print(f'[add_pred_to_datasample] Adding {len(bboxes_3d)} predictions to data sample')
                pred_instances_3d.scores_3d = scores_3d
                pred_instances_3d.labels_3d = labels_3d
            else:
                # Empty prediction fallback
                import torch
                from mmdet3d.structures import LiDARInstance3DBoxes
                pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
                    torch.zeros((0, 9), dtype=torch.float32))
                print(f'[add_pred_to_datasample] No valid predictions, adding empty boxes')
                pred_instances_3d.scores_3d = torch.zeros((0,), dtype=torch.float32)
                pred_instances_3d.labels_3d = torch.zeros((0,), dtype=torch.int64)

            data_sample.pred_instances_3d = pred_instances_3d

            # Add empty 2D predictions if not present (required by evaluator)
            if not hasattr(data_sample, 'pred_instances') or data_sample.pred_instances is None:
                data_sample.pred_instances = InstanceData()

        return batch_data_samples


print('[focalformer3d] ✓ Registered FocalFormer3D to MODELS')
print('[focalformer3d] ✓ Module fully loaded')

'''





#----------------------------------------------------------------------------------------------------------------------------


# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

# =============================================================================
# focalformer3d.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - DETECTORS from mmdet.models          → MODELS from mmdet3d.registry
#   - builder.build_neck(cfg)              → MODELS.build(cfg)
#   - Voxelization from mmdet3d.ops        → mmcv.ops.Voxelization
#   - force_fp32 from mmcv.runner          → REMOVED (AmpOptimWrapper handles)
#   - Box3DMode, Coord3DMode, bbox3d2result,
#     show_result from mmdet3d.core        → mmdet3d.structures
#   - multi_apply from mmdet.core          → mmdet.models.utils
#   - mmcv.parallel.DataContainer          → REMOVED (not needed in 1.1+)
#   - import mmcv (general)                → REMOVED (unused)
#   - import pdb                           → REMOVED
#   - forward_train()                      → loss(batch_inputs_dict, batch_data_samples)
#   - simple_test()                        → predict(batch_inputs_dict, batch_data_samples)
#   - aug_test()                           → aug_test (internal, called from predict)
#   - print() statements                   → print/warning
#   - merge_aug_bboxes_3d from mmdet3d.core→ from local plugin
# =============================================================================
import logging
import pickle
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)
print('[focalformer3d] Loading module...')

# --- mmcv.ops (Voxelization moved from mmdet3d.ops) ---
try:
    from mmcv.ops import Voxelization
    print('[focalformer3d] ✓ Imported Voxelization from mmcv.ops')
except ImportError as e:
    print(f'[focalformer3d] ✗ Voxelization: {e}')
    raise

# --- Registry: DETECTORS → MODELS ---
try:
    from mmdet3d.registry import MODELS
    print('[focalformer3d] ✓ Imported MODELS from mmdet3d.registry')
except ImportError as e:
    print(f'[focalformer3d] ✗ Registry imports: {e}')
    raise

# --- MVXTwoStageDetector base class ---
try:
    from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
    print('[focalformer3d] ✓ Imported MVXTwoStageDetector')
except ImportError as e:
    print(f'[focalformer3d] ✗ MVXTwoStageDetector: {e}')
    raise

# --- Structures (old: mmdet3d.core) ---
try:
    from mmdet3d.structures import bbox3d2result
    print('[focalformer3d] ✓ Imported bbox3d2result '
                'from mmdet3d.structures')
except ImportError:
    # bbox3d2result may have been removed in some mmdet3d versions
    # since Det3DDataSample handles results directly.
    print('[focalformer3d] bbox3d2result not found — using fallback')

    def bbox3d2result(bboxes, scores, labels, attrs=None):
        """Convert detection results to a list of numpy arrays."""
        result_dict = dict(
            boxes_3d=bboxes,
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu())
        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()
        return result_dict

# --- Det3DDataSample / InstanceData for result packaging ---
try:
    from mmdet3d.structures import Det3DDataSample
    print('[focalformer3d] ✓ Imported Det3DDataSample')
except ImportError as e:
    print(f'[focalformer3d] ✗ Det3DDataSample: {e}')
    raise

try:
    from mmengine.structures import InstanceData
    print('[focalformer3d] ✓ Imported InstanceData from mmengine')
except ImportError as e:
    print(f'[focalformer3d] ✗ InstanceData: {e}')
    raise

# --- Local project imports (unchanged — these are project-internal) ---
try:
    from projects.mmdet3d_plugin.models.utils.time_utils import T
    print('[focalformer3d] ✓ Imported T from time_utils')
except ImportError:
    # T is optional — only used for timing in simple_test (commented out)
    T = None
    print('[focalformer3d] time_utils.T not available — timing disabled')

try:
    from projects.mmdet3d_plugin.core.post_processing.merge_augs import merge_aug_bboxes_3d
    print('[focalformer3d] ✓ Imported merge_aug_bboxes_3d')
except ImportError as e:
    print(f'[focalformer3d] ✗ merge_aug_bboxes_3d: {e}')
    raise


# =============================================================================
# DEBUG HELPERS — pred vs GT comparison for first N samples
# =============================================================================

# Counter persists across calls within a single evaluation run
_SAMPLE_DEBUG_COUNT = 0
_PKL_CACHE = {}  # Cache for loaded pkl files

# ── Debug output file ────────────────────────────────────────────────────────
# All _debug_* output goes here instead of relying on terminal which clips.
# Override path via env var: export DEBUG_OUT_FILE=/path/to/debug.txt
import os as _os
_DEBUG_OUT_FILE = _os.environ.get(
    'DEBUG_OUT_FILE',
    _os.path.join(_os.getcwd(), 'focalformer_debug.txt'))
_debug_fh = None  # opened lazily on first write

def _dprint(*args):
    """Write to both the debug file and stdout (both flushed immediately)."""
    global _debug_fh
    import sys
    msg = ' '.join(str(a) for a in args)
    if _debug_fh is None:
        try:
            _debug_fh = open(_DEBUG_OUT_FILE, 'w', buffering=1)
            _debug_fh.write(f'# FocalFormer3D debug output\n')
            print(f'[DEBUG] Writing debug output to: {_DEBUG_OUT_FILE}', flush=True)
        except Exception as exc:
            print(f'[DEBUG] Could not open debug file: {exc}', flush=True)
    if _debug_fh is not None:
        _debug_fh.write(msg + '\n')
        _debug_fh.flush()
    print(msg, flush=True)
# ─────────────────────────────────────────────────────────────────────────────


def _debug_gt_from_pkl(data_sample,
                       ann_file='data/waymo/kitti_format/waymo_infos_val.pkl'):
    """Load GT boxes from the val pkl file using sample_idx.

    Useful when gt_instances_3d is not populated during test mode.
    Caches the pkl on first load to avoid repeated I/O.

    Args:
        data_sample: Det3DDataSample with metainfo containing sample_idx.
        ann_file (str): Path to the annotation pkl.
    """
    global _PKL_CACHE

    if ann_file not in _PKL_CACHE:
        _dprint(f'  [GT_PKL] Loading {ann_file} ...')
        try:
            with open(ann_file, 'rb') as f:
                raw = pickle.load(f)
            # Support both list-of-dicts and {'data_list': [...]} formats
            data_list = raw['data_list'] if isinstance(raw, dict) else raw
            _PKL_CACHE[ann_file] = {
                str(d['sample_idx']): d for d in data_list
            }
            _dprint(f'  [GT_PKL] Loaded {len(_PKL_CACHE[ann_file])} entries')
        except Exception as exc:
            _dprint(f'  [GT_PKL] Failed to load pkl: {exc}')
            _PKL_CACHE[ann_file] = {}

    sample_idx = str(data_sample.metainfo.get('sample_idx', ''))
    info = _PKL_CACHE[ann_file].get(sample_idx, None)

    if info is None:
        _dprint(f'  [GT_PKL] sample_idx={sample_idx} not found in pkl '
              f'(total keys: {len(_PKL_CACHE[ann_file])})')
        return

    instances = info.get('instances', [])
    _dprint(f'  [GT_PKL] sample_idx={sample_idx}  '
          f'num_instances={len(instances)}')
    for j, inst in enumerate(instances[:10]):
        bbox = inst.get('bbox_3d', [])
        lbl  = inst.get('bbox_label_3d', -1)
        name = inst.get('bbox_label', lbl)
        fmt  = [f'{v:.2f}' for v in bbox]
        _dprint(f'    GT_pkl[{j}] label={lbl}({name})  bbox_3d={fmt}')


def _debug_predictions_vs_gt(batch_data_samples, max_samples=5,
                              ann_file='data/waymo/kitti_format/waymo_infos_val.pkl'):
    """Print detailed pred vs GT comparison for the first ``max_samples`` calls.

    Reads GT from:
      1. data_sample.gt_instances_3d  (populated during train/val mode)
      2. data_sample.eval_ann_info    (sometimes available in test mode)
      3. The val pkl file via _debug_gt_from_pkl (fallback)

    Output goes to focalformer_debug.txt (or $DEBUG_OUT_FILE) AND stdout.

    Args:
        batch_data_samples (list[Det3DDataSample]): Samples AFTER
            add_pred_to_datasample() has been called so pred_instances_3d
            is populated.
        max_samples (int): Stop after this many samples total across all calls.
        ann_file (str): Path to annotation pkl for GT fallback.
    """
    global _SAMPLE_DEBUG_COUNT

    CLASS_NAMES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}

    for data_sample in batch_data_samples:
        if _SAMPLE_DEBUG_COUNT >= max_samples:
            return

        _SAMPLE_DEBUG_COUNT += 1
        sep = '=' * 72
        _dprint(f'\n{sep}')
        _dprint(f'[PRED_GT_DEBUG] Sample {_SAMPLE_DEBUG_COUNT} / {max_samples}')

        # ── Metadata ────────────────────────────────────────────────────────
        meta = data_sample.metainfo if hasattr(data_sample, 'metainfo') else {}
        _dprint(f'  sample_idx   : {meta.get("sample_idx", "N/A")}')
        _dprint(f'  context_name : {meta.get("context_name", "N/A")}')
        _dprint(f'  timestamp    : {meta.get("timestamp", "N/A")}')
        _dprint(f'  box_type_3d  : {meta.get("box_type_3d", "N/A")}')

        # ── Ground Truth ─────────────────────────────────────────────────────
        _dprint(f'\n  [GT BOXES]')
        gt_found = False

        # Source 1: gt_instances_3d (train/val mode)
        if hasattr(data_sample, 'gt_instances_3d'):
            gt = data_sample.gt_instances_3d
            gt_boxes  = getattr(gt, 'bboxes_3d', None)
            gt_labels = getattr(gt, 'labels_3d', None)
            if gt_boxes is not None and len(gt_boxes) > 0:
                gt_found = True
                _dprint(f'  Source: gt_instances_3d')
                _dprint(f'  num_gt     : {len(gt_boxes)}')
                _dprint(f'  tensor shape: {gt_boxes.tensor.shape}')
                _dprint(f'  tensor dtype: {gt_boxes.tensor.dtype}')
                for j in range(min(10, len(gt_boxes))):
                    b   = gt_boxes.tensor[j].cpu().numpy()
                    lbl = int(gt_labels[j].cpu()) if gt_labels is not None else -1
                    cname = CLASS_NAMES.get(lbl, f'cls{lbl}')
                    vel = (f'  vel=({b[7]:.2f},{b[8]:.2f})'
                           if len(b) > 8 else '')
                    _dprint(f'    GT[{j:02d}] {cname:12s} (label={lbl}) '
                          f'x={b[0]:7.2f} y={b[1]:7.2f} z={b[2]:6.2f}  '
                          f'dx={b[3]:.2f} dy={b[4]:.2f} dz={b[5]:.2f}  '
                          f'yaw={b[6]:.3f}{vel}')

        # Source 2: eval_ann_info
        if not gt_found and hasattr(data_sample, 'eval_ann_info') \
                and data_sample.eval_ann_info is not None:
            ann = data_sample.eval_ann_info
            if isinstance(ann, dict):
                _dprint(f'  Source: eval_ann_info  keys={list(ann.keys())}')
                gb = ann.get('gt_bboxes_3d', None)
                gl = ann.get('gt_labels_3d', None)
                if gb is not None and len(gb) > 0:
                    gt_found = True
                    _dprint(f'  num_gt: {len(gb)}')
                    for j in range(min(10, len(gb))):
                        b   = gb.tensor[j].cpu().numpy() \
                              if hasattr(gb, 'tensor') else gb[j]
                        lbl = int(gl[j]) if gl is not None else -1
                        cname = CLASS_NAMES.get(lbl, f'cls{lbl}')
                        _dprint(f'    GT[{j:02d}] {cname:12s} (label={lbl}) '
                              f'x={b[0]:7.2f} y={b[1]:7.2f} z={b[2]:6.2f}  '
                              f'dx={b[3]:.2f} dy={b[4]:.2f} dz={b[5]:.2f}  '
                              f'yaw={b[6]:.3f}')

        # Source 3: pkl fallback
        if not gt_found:
            _dprint(f'  Source: pkl fallback ({ann_file})')
            _debug_gt_from_pkl(data_sample, ann_file=ann_file)

        # ── Predictions ──────────────────────────────────────────────────────
        _dprint(f'\n  [PRED BOXES] (top 10 by score)')
        pred = getattr(data_sample, 'pred_instances_3d', None)

        if pred is None:
            _dprint('  pred_instances_3d not set')
        else:
            pred_boxes  = getattr(pred, 'bboxes_3d', None)
            pred_scores = getattr(pred, 'scores_3d', None)
            pred_labels = getattr(pred, 'labels_3d', None)

            if pred_boxes is None or len(pred_boxes) == 0:
                _dprint('  No predictions')
            else:
                _dprint(f'  num_preds   : {len(pred_boxes)}')
                _dprint(f'  tensor shape: {pred_boxes.tensor.shape}')
                _dprint(f'  tensor dtype: {pred_boxes.tensor.dtype}')

                # Sort by score descending
                if pred_scores is not None:
                    order      = pred_scores.argsort(descending=True)
                    top_boxes  = pred_boxes.tensor[order[:10]].cpu().numpy()
                    top_scores = pred_scores[order[:10]].cpu().numpy()
                    top_labels = (pred_labels[order[:10]].cpu().numpy()
                                  if pred_labels is not None
                                  else [-1] * 10)
                else:
                    top_boxes  = pred_boxes.tensor[:10].cpu().numpy()
                    top_scores = ['N/A'] * 10
                    top_labels = (pred_labels[:10].cpu().numpy()
                                  if pred_labels is not None
                                  else [-1] * 10)

                for j in range(len(top_boxes)):
                    b     = top_boxes[j]
                    lbl   = int(top_labels[j])
                    cname = CLASS_NAMES.get(lbl, f'cls{lbl}')
                    score = (f'{top_scores[j]:.4f}'
                             if top_scores[j] != 'N/A'
                             else 'N/A')
                    vel   = (f'  vel=({b[7]:.2f},{b[8]:.2f})'
                             if len(b) > 8 else '')
                    _dprint(f'    PRED[{j:02d}] {cname:12s} (label={lbl}) '
                          f'score={score}  '
                          f'x={b[0]:7.2f} y={b[1]:7.2f} z={b[2]:6.2f}  '
                          f'dx={b[3]:.2f} dy={b[4]:.2f} dz={b[5]:.2f}  '
                          f'yaw={b[6]:.3f}{vel}')

                # Per-class summary
                _dprint(f'\n  [PRED CLASS SUMMARY]')
                if pred_labels is not None:
                    for cls_id, cls_name in CLASS_NAMES.items():
                        mask = (pred_labels == cls_id)
                        n = mask.sum().item()
                        if n > 0 and pred_scores is not None:
                            s = pred_scores[mask]
                            _dprint(f'    {cls_name:12s}: {n:3d} preds  '
                                  f'score [{s.min():.3f}, {s.max():.3f}]  '
                                  f'mean={s.mean():.3f}')
                        elif n > 0:
                            _dprint(f'    {cls_name:12s}: {n:3d} preds')
                        else:
                            _dprint(f'    {cls_name:12s}:   0 preds')

        _dprint(f'{sep}\n')


# ===================================================================
# FocalFormer3D Detector
# ===================================================================
@MODELS.register_module()
class FocalFormer3D(MVXTwoStageDetector):
    """FocalFormer3D: Focusing on Hard Instance for 3D Object Detection.

    Refactored for mmdet3d >= 1.1 (mmengine-based).

    Key changes from old version:
      - ``forward_train()`` → ``loss()``
      - ``simple_test()`` → ``predict()``
      - Data comes via ``batch_inputs_dict`` and ``batch_data_samples``
      - ``@force_fp32`` removed (use AmpOptimWrapper)
      - ``builder.build_neck`` → ``MODELS.build``
    """

    def __init__(self,
                 freeze_img=False,
                 freeze_img_level=None,
                 freeze_camlss=False,
                 freeze_pts=False,
                 trainneck_ms=False,
                 train_middle_encoder=False,
                 pts_pillar_layer=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 imgpts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_img=True,
                 use_grid_mask=False,
                 input_pts=True,
                 init_cfg=None,
                 data_preprocessor=None,
                 **kwargs):
        # NOTE: In mmdet3d 1.4, MVXTwoStageDetector no longer accepts
        # pts_voxel_layer as an __init__ kwarg (it's handled by
        # data_preprocessor in the standard pipeline). Since FocalFormer3D
        # does its own voxelization, we intercept pts_voxel_layer here
        # and build it ourselves, then call super() without it.
        super(FocalFormer3D, self).__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs)

        print(f'[FocalFormer3D] Building: '
                    f'freeze_img={freeze_img}, freeze_pts={freeze_pts}, '
                    f'input_img={input_img}, input_pts={input_pts}, '
                    f'use_grid_mask={use_grid_mask}')

        # Build voxel layers ourselves (removed from super() call)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
            print(f'[FocalFormer3D] Built pts_voxel_layer')

        if pts_pillar_layer:
            self.pts_pillar_layer = Voxelization(**pts_pillar_layer)
            print(f'[FocalFormer3D] Built pts_pillar_layer')

        self.freeze_img_level = freeze_img_level
        self.freeze_camlss = freeze_camlss

        # --- builder.build_neck → MODELS.build ---
        self.imgpts_neck = MODELS.build(imgpts_neck)
        print(f'[FocalFormer3D] Built imgpts_neck: '
                     f'{imgpts_neck["type"]}')

        self.freeze_img = freeze_img
        self.freeze_pts = freeze_pts
        self.trainneck_ms = trainneck_ms
        self.train_middle_encoder = train_middle_encoder

        self.input_img = input_img
        self.input_pts = input_pts

        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
            print('[FocalFormer3D] GridMask enabled')

        self.apply_dynamic_voxelize = 'Dynamic' in pts_voxel_encoder['type']
        print(f'[FocalFormer3D] apply_dynamic_voxelize='
                     f'{self.apply_dynamic_voxelize}')

        print('[FocalFormer3D] ✓ Built successfully')

    def init_weights(self):
        """Initialize model weights."""
        super(FocalFormer3D, self).init_weights()

    def load_state_dict(self, state_dict, strict=True):
        """Override to remap old mmcv decoder keys to new mmdet decoder keys.

        Old (mmcv 0.x / mmdet 2.x):
            decoder.N.layers.L.attentions.0  -> self_attn
            decoder.N.layers.L.attentions.1  -> cross_attn
            decoder.N.layers.L.ffns.0        -> ffn

        New (mmdet 3.x):
            decoder.N.layers.L.self_attn
            decoder.N.layers.L.cross_attn
            decoder.N.layers.L.ffn
        """
        new_state_dict = {}
        remapped = 0
        for key, value in state_dict.items():
            new_key = key
            if 'pts_bbox_head.decoder.' in key:
                # attentions.0 -> self_attn
                new_key = new_key.replace('.attentions.0.', '.self_attn.')
                # attentions.1 -> cross_attn
                new_key = new_key.replace('.attentions.1.', '.cross_attn.')
                # ffns.0 -> ffn
                new_key = new_key.replace('.ffns.0.', '.ffn.')
                if new_key != key:
                    remapped += 1
            new_state_dict[new_key] = value

        if remapped > 0:
            print(f'[FocalFormer3D] Remapped {remapped} decoder keys '
                        f'(old mmcv -> new mmdet format)')

        return super().load_state_dict(new_state_dict, strict=strict)

        if self.input_img and self.freeze_img:
            if self.with_img_backbone:
                if self.freeze_img_level:
                    param_levels = [
                        ['conv1', 'bn1'], ['layer1'], ['layer2'],
                        ['layer3'], ['layer4']
                    ]
                    for i in range(self.freeze_img_level):
                        for pn in param_levels[i]:
                            # print → print
                            print(f'[FocalFormer3D] Freezing image {pn}')
                            for param in self.img_backbone.get_submodule(pn).parameters():
                                param.requires_grad = False
                else:
                    print('[FocalFormer3D] Freezing entire img_backbone')
                    for param in self.img_backbone.parameters():
                        param.requires_grad = False

            if self.with_img_neck:
                print('[FocalFormer3D] Freezing img_neck')
                for param in self.img_neck.parameters():
                    param.requires_grad = False

            if self.freeze_camlss and hasattr(self.imgpts_neck, 'cam_lss'):
                print('[FocalFormer3D] Freezing imgpts_neck.cam_lss')
                for param in self.imgpts_neck.cam_lss.parameters():
                    param.requires_grad = False

        if self.freeze_pts:
            print('[FocalFormer3D] Freezing pts branches (partial)')
            for name, param in self.named_parameters():
                if 'pts' in name and 'pts_bbox_head' not in name and 'imgpts_neck' not in name:
                    if self.trainneck_ms:
                        if 'pts_backbone' in name:
                            continue
                        if 'pts_neck' in name:
                            continue
                    if self.train_middle_encoder:
                        if 'pts' in name:
                            continue
                    param.requires_grad = False

            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

            if not self.train_middle_encoder:
                self.pts_voxel_layer.apply(fix_bn)
                self.pts_voxel_encoder.apply(fix_bn)
                self.pts_middle_encoder.apply(fix_bn)

            if not self.trainneck_ms:
                self.pts_backbone.apply(fix_bn)
                if self.with_pts_neck:
                    self.pts_neck.apply(fix_bn)

        if not self.input_pts:
            print('[FocalFormer3D] No pts input — nullifying pts modules')
            self.voxelize = None
            self.pts_voxel_encoder = None
            self.pts_middle_encoder = None
            self.pts_backbone = None
            if self.with_pts_neck:
                self.pts_neck = None

        print('[FocalFormer3D] init_weights complete')

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            if self.use_grid_mask and self.training:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img.float())
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        print(f'[FocalFormer3D.extract_img_feat] '
                     f'img_feats type={type(img_feats).__name__}, '
                     f'len={len(img_feats) if isinstance(img_feats, (list, tuple)) else "N/A"}')
        return img_feats

    def extract_pts_feat(self, pts, img_feats=None, img_metas=None):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None

        if self.apply_dynamic_voxelize:
            voxels, coors = self.dynamic_voxelize(pts)
            voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors)
            batch_size = coors[-1, 0] + 1
            coors = feature_coors  # update
        else:
            voxels, num_points, coors = self.voxelize(pts, voxel_type='voxel')
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0] + 1

        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)

        if self.with_pts_neck:
            x = self.pts_neck(x)
        else:
            x = [x]

        print(f'[FocalFormer3D.extract_pts_feat] '
                     f'num_features={len(x)}, '
                     f'feat0 shape={x[0].shape if isinstance(x[0], torch.Tensor) else "N/A"}')
        return x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points.

        This is the shared feature extraction used by both loss() and predict().
        """
            
        # ... rest of extract_feat runs normally ...
        if self.input_img:
            img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = [None]

        if self.input_pts:
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        else:
            pts_feats = [None]

        new_img_feat, new_pts_feat = self.imgpts_neck(
            img_feats[0], pts_feats[0], img_metas)

        print(f'[FocalFormer3D.extract_feat] '
                     f'new_img_feat={type(new_img_feat).__name__}, '
                     f'new_pts_feat={type(new_pts_feat).__name__}')
        return (new_img_feat, new_pts_feat)

    @torch.no_grad()
    # NOTE: @force_fp32() REMOVED — handled by AmpOptimWrapper in mmdet3d 1.1+
    def voxelize(self, points, voxel_type='voxel'):
        """Apply voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            voxel_type (str): 'voxel' or 'pillar'.

        Returns:
            tuple: voxels, num_points, coors_batch
        """
        assert voxel_type == 'voxel' or voxel_type == 'pillar'
        voxels, coors, num_points = [], [], []

        for res in points:
            if voxel_type == 'voxel':
                res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            elif voxel_type == 'pillar':
                res_voxels, res_coors, res_num_points = self.pts_pillar_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []

        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)

        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    # NOTE: @force_fp32() REMOVED — handled by AmpOptimWrapper in mmdet3d 1.1+
    def dynamic_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provides a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)

        points = torch.cat(points, dim=0)
        coors_batch = []

        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)

        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    # =================================================================
    # NEW API: loss() replaces forward_train()
    # =================================================================
    # def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
    #     """Calculate losses from a batch of inputs and data samples.

    #     This replaces the old ``forward_train()`` method.

    #     Args:
    #         batch_inputs_dict (dict): Contains 'points', 'imgs', etc.
    #         batch_data_samples (list[:obj:`Det3DDataSample`]): Each
    #             contains gt_instances_3d with bboxes_3d and labels_3d,
    #             and metainfo (the old img_metas).

    #     Returns:
    #         dict: A dictionary of loss components.
    #     """
    #     # --- Unpack inputs (new API) ---
    #     points = batch_inputs_dict.get('points', None)
    #     imgs = batch_inputs_dict.get('imgs', None)

    #     # --- Unpack ground truths and metadata from data samples ---
    #     gt_bboxes_3d = []
    #     gt_labels_3d = []
    #     img_metas = []

    #     for data_sample in batch_data_samples:
    #         img_metas.append(data_sample.metainfo)
    #         gt_bboxes_3d.append(data_sample.gt_instances_3d.bboxes_3d)
    #         gt_labels_3d.append(data_sample.gt_instances_3d.labels_3d)

    #     print(f'[FocalFormer3D.loss] batch_size={len(img_metas)}, '
    #                  f'num_gt_per_sample='
    #                  f'{[len(l) for l in gt_labels_3d]}')

    #     # --- Feature extraction (same as before) ---
    #     img_feats, pts_feats = self.extract_feat(
    #         points, img=imgs, img_metas=img_metas)

    #     # --- Compute losses ---
    #     losses = dict()
    #     losses_pts = self.forward_pts_train(
    #         pts_feats, img_feats, gt_bboxes_3d,
    #         gt_labels_3d, img_metas)
    #     losses.update(losses_pts)

    #     return losses

    # def forward_pts_train(self,
    #                       pts_feats,
    #                       img_feats,
    #                       gt_bboxes_3d,
    #                       gt_labels_3d,
    #                       img_metas,
    #                       gt_bboxes_ignore=None):
    #     """Forward function for point cloud branch.

    #     Args:
    #         pts_feats (list[torch.Tensor]): Features of point cloud branch
    #         img_feats: Image features after fusion.
    #         gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
    #             boxes for each sample.
    #         gt_labels_3d (list[torch.Tensor]): Ground truth labels for
    #             boxes of each sample.
    #         img_metas (list[dict]): Meta information of samples.
    #         gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
    #             boxes to be ignored. Defaults to None.

    #     Returns:
    #         dict: Losses of each branch.
    #     """
    #     outs = self.pts_bbox_head(
    #         pts_feats, img_feats, img_metas,
    #         gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
    #     loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
    #     losses = self.pts_bbox_head.loss(*loss_inputs)

    #     print(f'[FocalFormer3D.forward_pts_train] '
    #                  f'loss keys: {list(losses.keys())}')
    #     return losses

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs and data samples."""
        from mmengine.structures import InstanceData # Ensure this is imported

        points = batch_inputs_dict.get('points', None)
        imgs = batch_inputs_dict.get('imgs', None)

        # --- v1.x Compliant GT Unpacking ---
        batch_gt_instances_3d = []
        img_metas = []


        # =====================================================================
        # 🚨 TIME MACHINE (Input Boundary): v1.x GTs -> v0.x Model
        # =====================================================================
        # Detect dataset: NuScenes needs dim swap, Waymo does not
        _is_nuscenes = (hasattr(self, 'test_cfg') and self.test_cfg is not None
                        and self.test_cfg.get('pts', {}).get('dataset', '') == 'nuScenes')

        for data_sample in batch_data_samples:
            bboxes = data_sample.gt_instances_3d.bboxes_3d
            if bboxes is not None and len(bboxes) > 0:
                tensor = bboxes.tensor
                # NuScenes only: mmdet3d 1.4 stores dims as (l, w, h) but
                # FocalFormer3D (trained on 0.x) expects (w, l, h)
                if _is_nuscenes:
                    l_v1 = tensor[:, 3].clone()
                    w_v1 = tensor[:, 4].clone()
                    tensor[:, 3] = w_v1  # Now it's width
                    tensor[:, 4] = l_v1  # Now it's length
            
            img_metas.append(data_sample.metainfo)
            
            # Extract raw boxes and labels
            #bboxes = data_sample.gt_instances_3d.bboxes_3d
            labels = data_sample.gt_instances_3d.labels_3d
            
            # Wrap them in InstanceData (v1.x requirement for Assigners)
            gt_instances = InstanceData()
            gt_instances.bboxes_3d = bboxes
            gt_instances.labels_3d = labels
            batch_gt_instances_3d.append(gt_instances)

        print(f'[FocalFormer3D.loss] batch_size={len(img_metas)}')

        # Feature extraction
        img_feats, pts_feats = self.extract_feat(
            points, img=imgs, img_metas=img_metas)

        # Compute losses (Pass the InstanceData list, NOT raw box lists)
        losses = dict()
        losses_pts = self.forward_pts_train(
            pts_feats, img_feats, 
            batch_gt_instances_3d, # <--- Pass the wrapped instances here
            None, # gt_labels_3d is now inside batch_gt_instances_3d
            img_metas)
        losses.update(losses_pts)

        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          batch_gt_instances_3d, # <--- Updated argument
                          gt_labels_3d,          # <--- Will be None
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch."""
        
        # Unpack for the head (Some old heads still want separate lists)
        # We try passing them separated first, as your DeepInteractionDecoder 
        # get_targets() expects lists of boxes and labels.
        gt_bboxes_3d = [inst.bboxes_3d for inst in batch_gt_instances_3d]
        gt_labels_3d = [inst.labels_3d for inst in batch_gt_instances_3d]

        outs = self.pts_bbox_head(
            pts_feats, img_feats, img_metas,
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
            
        # 🚨 CRITICAL FIX for Assigner: 
        # Pass the wrapped batch_gt_instances_3d to the loss function, 
        # NOT the raw gt_bboxes_3d/gt_labels_3d lists.
        loss_inputs = [batch_gt_instances_3d, outs] 
        # Note: If your DeepInteractionDecoder.loss() strictly expects 
        # (gt_bboxes_3d, gt_labels_3d, preds_dicts), you will need to update 
        # DeepInteractionDecoder.loss() to accept batch_gt_instances_3d instead.
        
        try:
            # Try v1.x standard format first
            losses = self.pts_bbox_head.loss_by_feat(*loss_inputs)
        except AttributeError:
             # Fallback if your custom head uses .loss() instead of .loss_by_feat()
             try:
                 losses = self.pts_bbox_head.loss(batch_gt_instances_3d, outs)
             except TypeError:
                 # If it strictly requires separate arguments, we fall back to raw lists
                 # (But this is what caused the batch_valid_gt_mask error)
                 losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)

        return losses

    # =================================================================
    # NEW API: predict() replaces simple_test()
    # =================================================================
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples.

        This replaces the old ``simple_test()`` method.

        Args:
            batch_inputs_dict (dict): Contains 'points', 'imgs', etc.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each
                contains metainfo (the old img_metas).

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results with
                pred_instances_3d populated.
        """
        # --- Unpack inputs ---
        points = batch_inputs_dict.get('points', None)
        imgs = batch_inputs_dict.get('imgs', None)
        img_metas = [ds.metainfo for ds in batch_data_samples]


        # Optionally unpack gt for oracle/debug testing
        gt_bboxes_3d = None
        gt_labels_3d = None
        if hasattr(batch_data_samples[0], 'gt_instances_3d'):
            try:
                gt_bboxes_3d = [ds.gt_instances_3d.bboxes_3d for ds in batch_data_samples]
                gt_labels_3d = [ds.gt_instances_3d.labels_3d for ds in batch_data_samples]
            except AttributeError:
                pass

        print(f'[FocalFormer3D.predict] batch_size={len(img_metas)}')

        # --- Feature extraction ---
        img_feats, pts_feats = self.extract_feat(
            points, img=imgs, img_metas=img_metas)

        # --- Point cloud branch inference ---
        bbox_pts = self.simple_test_pts(
            pts_feats, img_feats, img_metas,
            rescale=kwargs.get('rescale', False),
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d)

        # --- Package results into Det3DDataSample ---
        results_list = self.add_pred_to_datasample(
            batch_data_samples, bbox_pts)

        # ── Debug: print pred vs GT for first 5 samples ──────────────────────
        # Set _SAMPLE_DEBUG_COUNT = 0 at module level to re-enable after reset.
        # Change max_samples to control how many frames are printed.
        # Change ann_file to point to your actual val pkl if GT is not in
        # batch_data_samples (test mode typically has no gt_instances_3d).
        _debug_predictions_vs_gt(
            results_list,
            max_samples=5,
            ann_file='data/waymo/kitti_format/waymo_infos_val.pkl')
        # ─────────────────────────────────────────────────────────────────────

        return results_list

    

    def simple_test_pts(self, x, x_img, img_metas, rescale=False,
                        gt_bboxes_3d=None, gt_labels_3d=None, **kwargs):
        """Test function of point cloud branch.

        Args:
            x: Point cloud features.
            x_img: Image features.
            img_metas (list[dict]): Meta information.
            rescale (bool): Whether to rescale results.
            gt_bboxes_3d: Optional GT boxes for debug/oracle testing.
            gt_labels_3d: Optional GT labels for debug/oracle testing.

        Returns:
            list[dict]: Detection results per sample.
        """
        outs = self.pts_bbox_head(
            x, x_img, img_metas,
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, **kwargs)

        if True:
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
        # In simple_test_pts, after bbox_list = self.pts_bbox_head.get_bboxes(...)
        for bboxes, scores, labels in bbox_list:
            print(f'  [DIAG] After get_bboxes: {len(bboxes)} detections out of 300 proposals')
            print(f'  [DIAG] Score range: [{scores.min():.4f}, {scores.max():.4f}]')
            if len(bboxes) > 0:
                centers = bboxes.gravity_center
                print(f'  [DIAG] Center X range: [{centers[:,0].min():.1f}, {centers[:,0].max():.1f}]')
                print(f'  [DIAG] Center Y range: [{centers[:,1].min():.1f}, {centers[:,1].max():.1f}]')
                print(f'  [DIAG] Center Z range: [{centers[:,2].min():.1f}, {centers[:,2].max():.1f}]')
            break  # Only first sample
        else:
            bbox_list = self.pts_bbox_head.get_heatmap_bboxes(
                outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        print(f'[FocalFormer3D.simple_test_pts] '
                     f'{len(bbox_results)} samples')
        return bbox_results

    def aug_test(self, batch_inputs_dict_list, batch_data_samples_list,
                 rescale=False):
        """Test function with augmentation.

        In mmdet3d 1.1+, aug_test is called differently than in 1.0.
        This method handles both patterns.

        Args:
            batch_inputs_dict_list: List of input dicts per augmentation.
            batch_data_samples_list: List of data samples per augmentation.
            rescale (bool): Whether to rescale results.

        Returns:
            list[dict]: Merged detection results.
        """
        precompute = False

        if not precompute:
            print('[FocalFormer3D.aug_test] Precomputing aug_test ...')

            # Extract points and images for each augmentation
            all_pts_feats = []
            all_img_feats = []
            all_img_metas = []

            for batch_inputs, batch_samples in zip(
                    batch_inputs_dict_list, batch_data_samples_list):
                points = batch_inputs.get('points', None)
                imgs = batch_inputs.get('imgs', None)
                img_metas = [ds.metainfo for ds in batch_samples]

                img_feats, pts_feats = self.extract_feat(
                    points, img=imgs, img_metas=img_metas)
                all_pts_feats.append(pts_feats)
                all_img_feats.append(img_feats)
                all_img_metas.append(img_metas)

            bbox_list = dict()
            if self.with_pts_bbox:
                bbox_pts = self.aug_test_pts(
                    all_pts_feats, all_img_feats, all_img_metas,
                    rescale=rescale)
                bbox_list.update(pts_bbox=bbox_pts)
        else:
            print('[FocalFormer3D.aug_test] Using precomputed results ...')
            all_img_metas = [
                [ds.metainfo for ds in batch_samples]
                for batch_samples in batch_data_samples_list
            ]
            bbox_list = dict()
            bbox_pts = self.aug_test_pts(
                None, None, all_img_metas, rescale=rescale)
            bbox_list.update(pts_bbox=bbox_pts)

        return [bbox_list]

    def aug_test_pts(self, xs, x_imgs, img_metas, rescale=False):
        """Test function of point cloud branch with augmentation.

        Args:
            xs (list): Point features per augmentation (or None if precomputed).
            x_imgs (list): Image features per augmentation (or None).
            img_metas (list[list[dict]]): Meta info per augmentation.
            rescale (bool): Whether to rescale results.

        Returns:
            dict: Merged bounding box results.
        """
        if xs is not None:
            # only support aug_test for one sample
            aug_bboxes = []
            for x, x_img, img_meta in zip(xs, x_imgs, img_metas):
                outs = self.pts_bbox_head(x, x_img, img_meta)
                bbox_list = self.pts_bbox_head.get_bboxes(
                    outs, img_meta, rescale=rescale)
                bbox_list = [
                    dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                    for bboxes, scores, labels in bbox_list
                ]
                aug_bboxes.append(bbox_list[0])

            # after merging, bboxes will be rescaled to the original image size
            merged_bboxes = merge_aug_bboxes_3d(
                aug_bboxes, img_metas,
                self.pts_bbox_head.test_cfg)
        else:
            merged_bboxes = merge_aug_bboxes_3d(
                None, img_metas,
                self.pts_bbox_head.test_cfg)

        print(f'[FocalFormer3D.aug_test_pts] '
                     f'merged {len(aug_bboxes) if xs else 0} augmentations')
        return merged_bboxes

    # def add_pred_to_datasample(self, batch_data_samples, bbox_results):
    #     """Add prediction results to Det3DDataSample.

    #     This converts the old-style bbox_results dicts into the new
    #     Det3DDataSample format expected by mmdet3d 1.1+.

    #     Args:
    #         batch_data_samples (list[:obj:`Det3DDataSample`]): Data samples.
    #         bbox_results (list[dict]): Prediction results from
    #             simple_test_pts.

    #     Returns:
    #         list[:obj:`Det3DDataSample`]: Updated data samples.
    #     """
    #     for data_sample, bbox_result in zip(batch_data_samples, bbox_results):
    #         pred_instances_3d = InstanceData()

    #         # Handle both old-style and new-style keys
    #         bboxes_3d = bbox_result.get('bboxes_3d', bbox_result.get('boxes_3d', bbox_result.get('bboxes', None)))
    #         scores_3d = bbox_result.get('scores_3d', bbox_result.get('scores', None))
    #         labels_3d = bbox_result.get('labels_3d', bbox_result.get('labels', None))

    #         print(f'[add_pred_to_datasample] bbox_result keys: {bbox_result.keys()}, '
    #                      f'bboxes_3d type: {type(bboxes_3d)}, '
    #                      f'has __len__: {hasattr(bboxes_3d, "__len__") if bboxes_3d is not None else "None"}')
    #         # One-time debug: log label distribution for first sample
    #         if labels_3d is not None and hasattr(labels_3d, 'unique'):
    #             if not hasattr(self, '_logged_label_info'):
    #                 self._logged_label_info = True
    #                 print(f'[add_pred_to_datasample] FIRST SAMPLE label info: '
    #                                f'unique={labels_3d.unique().tolist()}, '
    #                                f'dtype={labels_3d.dtype}, '
    #                                f'min={labels_3d.min().item()}, max={labels_3d.max().item()}, '
    #                                f'count={len(labels_3d)}')

    #         if bboxes_3d is not None and hasattr(bboxes_3d, '__len__') and len(bboxes_3d) > 0:
    #             # Ensure labels are long type and in valid range [0, num_classes)
    #             labels_3d = labels_3d.long()
    #             num_classes = 10  # nuScenes
    #             valid_mask = (labels_3d >= 0) & (labels_3d < num_classes)
    #             if not valid_mask.all():
    #                 n_invalid = (~valid_mask).sum().item()
    #                 invalid_labels = labels_3d[~valid_mask].unique().tolist()
    #                 print(f'[add_pred_to_datasample] Filtering {n_invalid} '
    #                                f'predictions with out-of-range labels: {invalid_labels}')
    #                 valid_idx = valid_mask.nonzero(as_tuple=True)[0]
    #                 bboxes_3d = bboxes_3d[valid_idx]
    #                 scores_3d = scores_3d[valid_idx]
    #                 labels_3d = labels_3d[valid_idx]
    #             pred_instances_3d.bboxes_3d = bboxes_3d
    #             pred_instances_3d.scores_3d = scores_3d
    #             pred_instances_3d.labels_3d = labels_3d
    #         else:
    #             # Empty prediction fallback
    #             import torch
    #             from mmdet3d.structures import LiDARInstance3DBoxes
    #             pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
    #                 torch.zeros((0, 9), dtype=torch.float32))
    #             pred_instances_3d.scores_3d = torch.zeros((0,), dtype=torch.float32)
    #             pred_instances_3d.labels_3d = torch.zeros((0,), dtype=torch.int64)

    #         data_sample.pred_instances_3d = pred_instances_3d

    #         # Add empty 2D predictions if not present (required by evaluator)
    #         if not hasattr(data_sample, 'pred_instances') or data_sample.pred_instances is None:
    #             data_sample.pred_instances = InstanceData()

    #     return batch_data_samples

    def add_pred_to_datasample(self, batch_data_samples, bbox_results):
        """Add prediction results to Det3DDataSample."""
        # =====================================================================
        # Both NuScenes and Waymo checkpoints were trained with the v0.x
        # convention where the head outputs (w, l) and yaw in the old sense.
        # mmdet3d v1.x evaluators expect (l, w) and yaw = -yaw_old - pi/2.
        # Apply the same fix for both datasets.
        _dataset = ''
        if hasattr(self, 'test_cfg') and self.test_cfg is not None:
            _dataset = self.test_cfg.get('pts', {}).get('dataset', '')
        _needs_fix = _dataset in ('nuScenes', 'Waymo')

        for result in bbox_results:
            bboxes_3d = result.get('bboxes_3d', result.get('boxes_3d', result.get('bboxes', None)))
            if bboxes_3d is not None and len(bboxes_3d) > 0:
                tensor = bboxes_3d.tensor
                if _needs_fix:
                    # swap predicted (w, l) -> (l, w) to match v1.x evaluator
                    w_v0 = tensor[:, 3].clone()
                    l_v0 = tensor[:, 4].clone()
                    tensor[:, 3] = l_v0  # index 3 = length
                    tensor[:, 4] = w_v0  # index 4 = width
                    # yaw convention: v0 head outputs sin/cos of old yaw,
                    # v1.x evaluator expects yaw_new = -yaw_old - pi/2
                    tensor[:, 6] = -tensor[:, 6] - (np.pi / 2)
        # =====================================================================

        from mmengine.structures import InstanceData

        for data_sample, bbox_result in zip(batch_data_samples, bbox_results):
            pred_instances_3d = InstanceData()

            # Now we extract the mathematically corrected boxes
            bboxes_3d = bbox_result.get('bboxes_3d', bbox_result.get('boxes_3d', bbox_result.get('bboxes', None))) # 2
            scores_3d = bbox_result.get('scores_3d', bbox_result.get('scores', None))
            labels_3d = bbox_result.get('labels_3d', bbox_result.get('labels', None))

            if bboxes_3d is not None and hasattr(bboxes_3d, '__len__') and len(bboxes_3d) > 0:
                labels_3d = labels_3d.long()
                # Derive num_classes from dataset rather than hardcoding 10 (which silently
                # passes Waymo labels 0/1/2 but would wrongly pass spurious high labels)
                _is_waymo = (hasattr(self, 'test_cfg') and self.test_cfg is not None
                             and self.test_cfg.get('pts', {}).get('dataset', '') == 'Waymo')
                num_classes = 3 if _is_waymo else 10
                valid_mask = (labels_3d >= 0) & (labels_3d < num_classes)
                if not valid_mask.all():
                    n_invalid = (~valid_mask).sum().item()
                    invalid_labels = labels_3d[~valid_mask].unique().tolist()
                    print(f'[add_pred_to_datasample] Filtering {n_invalid} '
                                   f'predictions with out-of-range labels: {invalid_labels}')
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                    bboxes_3d = bboxes_3d[valid_idx]
                    scores_3d = scores_3d[valid_idx]
                    labels_3d = labels_3d[valid_idx]
                pred_instances_3d.bboxes_3d = bboxes_3d
                print(f'[add_pred_to_datasample] Adding {len(bboxes_3d)} predictions to data sample')
                pred_instances_3d.scores_3d = scores_3d
                pred_instances_3d.labels_3d = labels_3d
            else:
                # Empty prediction fallback
                import torch
                from mmdet3d.structures import LiDARInstance3DBoxes
                pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
                    torch.zeros((0, 9), dtype=torch.float32))
                print(f'[add_pred_to_datasample] No valid predictions, adding empty boxes')
                pred_instances_3d.scores_3d = torch.zeros((0,), dtype=torch.float32)
                pred_instances_3d.labels_3d = torch.zeros((0,), dtype=torch.int64)

            data_sample.pred_instances_3d = pred_instances_3d

            # Add empty 2D predictions if not present (required by evaluator)
            if not hasattr(data_sample, 'pred_instances') or data_sample.pred_instances is None:
                data_sample.pred_instances = InstanceData()

        return batch_data_samples


print('[focalformer3d] ✓ Registered FocalFormer3D to MODELS')
print('[focalformer3d] ✓ Module fully loaded')