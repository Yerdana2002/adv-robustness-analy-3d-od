# =============================================================================
# deep_interaction.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - DETECTORS from mmdet.models          → MODELS from mmdet3d.registry
#   - builder.build_neck(cfg)              → MODELS.build(cfg)
#   - Voxelization from mmdet3d.ops        → mmcv.ops.Voxelization
#   - force_fp32 from mmcv.runner          → REMOVED (AmpOptimWrapper handles)
#   - Box3DMode, Coord3DMode, bbox3d2result,
#     show_result from mmdet3d.core        → mmdet3d.structures
#   - merge_aug_bboxes_3d from mmdet3d.core → mmdet3d.structures (or removed)
#   - multi_apply from mmdet.core          → mmdet.models.utils
#   - mmcv.parallel.DataContainer          → REMOVED (not needed in 1.1+)
#   - import pdb                           → REMOVED
#   - forward_train()                      → loss(batch_inputs_dict, batch_data_samples)
#   - simple_test()                        → predict(batch_inputs_dict, batch_data_samples)
#   - MVXTwoStageDetector forward signature adapted
# =============================================================================
import logging

import torch
from torch import nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)
logger.info('[deep_interaction] Loading module...')

# --- mmcv.ops (Voxelization moved from mmdet3d.ops) ---
try:
    from mmcv.ops import Voxelization
    logger.info('[deep_interaction] ✓ Imported Voxelization from mmcv.ops')
except ImportError as e:
    logger.error(f'[deep_interaction] ✗ Voxelization: {e}')
    raise

# --- Registry: DETECTORS → MODELS ---
try:
    from mmdet3d.registry import MODELS
    logger.info('[deep_interaction] ✓ Imported MODELS from mmdet3d.registry')
except ImportError as e:
    logger.error(f'[deep_interaction] ✗ Registry imports: {e}')
    raise

# --- MVXTwoStageDetector base class ---
try:
    from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
    logger.info('[deep_interaction] ✓ Imported MVXTwoStageDetector '
                'from mmdet3d.models.detectors.mvx_two_stage')
except ImportError as e:
    logger.error(f'[deep_interaction] ✗ MVXTwoStageDetector: {e}')
    raise

# --- Structures (old: mmdet3d.core) ---
try:
    from mmdet3d.structures import bbox3d2result
    logger.info('[deep_interaction] ✓ Imported bbox3d2result '
                'from mmdet3d.structures')
except ImportError:
    # In some mmdet3d versions, bbox3d2result may have been removed
    # since Det3DDataSample handles results directly.
    # Provide a fallback.
    logger.warning('[deep_interaction] bbox3d2result not found in '
                   'mmdet3d.structures — using inline fallback')

    def bbox3d2result(bboxes, scores, labels, attrs=None):
        """Convert detection results to a list of numpy arrays."""
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu())
        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()
        return result_dict

# --- Det3DDataSample for result packaging ---
try:
    from mmdet3d.structures import Det3DDataSample
    from mmdet3d.structures.det3d_data_sample import SampleList
    logger.info('[deep_interaction] ✓ Imported Det3DDataSample')
except ImportError:
    try:
        from mmdet3d.structures import Det3DDataSample
        SampleList = list
        logger.info('[deep_interaction] ✓ Imported Det3DDataSample '
                    '(SampleList fallback to list)')
    except ImportError as e:
        logger.error(f'[deep_interaction] ✗ Det3DDataSample: {e}')
        raise

# --- InstanceData for structured results ---
try:
    from mmengine.structures import InstanceData
    logger.info('[deep_interaction] ✓ Imported InstanceData from mmengine')
except ImportError as e:
    logger.error(f'[deep_interaction] ✗ InstanceData: {e}')
    raise


# ===================================================================
# DeepInteraction Detector
# ===================================================================
@MODELS.register_module()
class DeepInteraction(MVXTwoStageDetector):
    """Multi-modality VoxelNet with Deep Interaction.

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
                 freeze_pts=False,
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
                 init_cfg=None,
                 data_preprocessor=None,
                 **kwargs):
        super(DeepInteraction, self).__init__(
            pts_voxel_layer=pts_voxel_layer,
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

        logger.info(f'[DeepInteraction] Building: '
                    f'freeze_img={freeze_img}, freeze_pts={freeze_pts}')

        # --- Voxelization for pillar branch ---
        self.pts_pillar_layer = Voxelization(**pts_pillar_layer)
        logger.debug(f'[DeepInteraction] Built pts_pillar_layer: '
                     f'{pts_pillar_layer}')

        # --- builder.build_neck → MODELS.build ---
        self.imgpts_neck = MODELS.build(imgpts_neck)
        logger.debug(f'[DeepInteraction] Built imgpts_neck: '
                     f'{imgpts_neck["type"]}')

        self.freeze_img = freeze_img
        self.freeze_pts = freeze_pts

        logger.info('[DeepInteraction] ✓ Built successfully')

    def init_weights(self):
        """Initialize model weights."""
        super(DeepInteraction, self).init_weights()

        if self.freeze_img:
            logger.info('[DeepInteraction] Freezing image backbone/neck')
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

        if self.freeze_pts:
            logger.info('[DeepInteraction] Freezing pts branches (partial)')
            for name, param in self.named_parameters():
                if 'pts' in name and 'pts_bbox_head' not in name and 'imgpts_neck' not in name:
                    param.requires_grad = False
                if 'pts_bbox_head.decoder.0' in name:
                    param.requires_grad = False
                if 'imgpts_neck.shared_conv_pts' in name:
                    param.requires_grad = False
                if 'pts_bbox_head.heatmap_head' in name and 'pts_bbox_head.heatmap_head_img' not in name:
                    param.requires_grad = False
                if 'pts_bbox_head.prediction_heads.0' in name:
                    param.requires_grad = False
                if 'pts_bbox_head.class_encoding' in name:
                    param.requires_grad = False

            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

            self.pts_voxel_layer.apply(fix_bn)
            self.pts_voxel_encoder.apply(fix_bn)
            self.pts_middle_encoder.apply(fix_bn)
            self.pts_backbone.apply(fix_bn)
            self.pts_neck.apply(fix_bn)
            self.pts_bbox_head.heatmap_head.apply(fix_bn)
            self.pts_bbox_head.class_encoding.apply(fix_bn)
            self.pts_bbox_head.decoder[0].apply(fix_bn)
            self.pts_bbox_head.prediction_heads[0].apply(fix_bn)
            self.imgpts_neck.shared_conv_pts.apply(fix_bn)

        logger.debug('[DeepInteraction] init_weights complete')

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

            img_feats = self.img_backbone(img.float())
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        logger.debug(f'[DeepInteraction.extract_img_feat] '
                     f'img_feats type={type(img_feats)}, '
                     f'len={len(img_feats) if isinstance(img_feats, (list, tuple)) else "N/A"}')
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None

        voxels, num_points, coors = self.voxelize(pts, voxel_type='voxel')
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)

        if self.with_pts_neck:
            x = self.pts_neck(x)

        pillars, pillars_num_points, pillar_coors = self.voxelize(pts, voxel_type='pillar')
        pillar_features = self.pts_voxel_encoder(pillars, pillars_num_points, pillar_coors)

        pts_metas = {}
        pts_metas['pillar_center'] = pillar_features
        pts_metas['pillars'] = pillars
        pts_metas['pillars_num_points'] = pillars_num_points
        pts_metas['pillar_coors'] = pillar_coors
        pts_metas['pts'] = pts

        logger.debug(f'[DeepInteraction.extract_pts_feat] '
                     f'voxels={voxels.shape}, pillars={pillars.shape}')
        return x, pts_metas

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points.

        This is the shared feature extraction used by both loss() and predict().
        """
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats, pts_metas = self.extract_pts_feat(points, img_feats, img_metas)
        new_img_feat, new_pts_feat = self.imgpts_neck(
            img_feats[0], pts_feats[0], img_metas, pts_metas)

        logger.debug(f'[DeepInteraction.extract_feat] '
                     f'new_img_feat={new_img_feat.shape}, '
                     f'new_pts_feat shape={[x.shape for x in new_pts_feat] if isinstance(new_pts_feat, (list, tuple)) else new_pts_feat.shape}')
        return (new_img_feat, new_pts_feat)

    @torch.no_grad()
    # NOTE: @force_fp32() REMOVED — handled by AmpOptimWrapper in mmdet3d 1.1+
    def voxelize(self, points, voxel_type='voxel'):
        """Apply dynamic voxelization to points.

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

    # =================================================================
    # NEW API: loss() replaces forward_train()
    # =================================================================
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        This replaces the old ``forward_train()`` method.

        Args:
            batch_inputs_dict (dict): Contains 'points', 'imgs', etc.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each
                contains gt_instances_3d with bboxes_3d and labels_3d,
                and metainfo (the old img_metas).

        Returns:
            dict: A dictionary of loss components.
        """
        # --- Unpack inputs (new API) ---
        points = batch_inputs_dict.get('points', None)
        imgs = batch_inputs_dict.get('imgs', None)

        # --- Unpack ground truths and metadata from data samples ---
        gt_bboxes_3d = []
        gt_labels_3d = []
        img_metas = []

        for data_sample in batch_data_samples:
            img_metas.append(data_sample.metainfo)
            gt_bboxes_3d.append(data_sample.gt_instances_3d.bboxes_3d)
            gt_labels_3d.append(data_sample.gt_instances_3d.labels_3d)

        logger.debug(f'[DeepInteraction.loss] batch_size={len(img_metas)}, '
                     f'num_gt_per_sample='
                     f'{[len(l) for l in gt_labels_3d]}')

        # --- Feature extraction (same as before) ---
        img_feats, pts_feats = self.extract_feat(
            points, img=imgs, img_metas=img_metas)

        # --- Compute losses ---
        losses = dict()
        losses_pts = self.forward_pts_train(
            pts_feats, img_feats, gt_bboxes_3d,
            gt_labels_3d, img_metas)
        losses.update(losses_pts)

        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            img_feats: Image features after fusion.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sample.
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        logger.debug(f'[DeepInteraction.forward_pts_train] '
                     f'loss keys: {list(losses.keys())}')
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

        logger.debug(f'[DeepInteraction.predict] batch_size={len(img_metas)}')

        # --- Feature extraction ---
        img_feats, pts_feats = self.extract_feat(
            points, img=imgs, img_metas=img_metas)

        # --- Point cloud branch inference ---
        bbox_pts = self.simple_test_pts(
            pts_feats, img_feats, img_metas, rescale=kwargs.get('rescale', False))

        # --- Package results into Det3DDataSample ---
        results_list = self.add_pred_to_datasample(
            batch_data_samples, bbox_pts)

        return results_list

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch.

        Args:
            x: Point cloud features.
            x_img: Image features.
            img_metas (list[dict]): Meta information.
            rescale (bool): Whether to rescale results.

        Returns:
            list[dict]: Detection results per sample.
        """
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        logger.debug(f'[DeepInteraction.simple_test_pts] '
                     f'{len(bbox_results)} samples, '
                     f'boxes per sample: '
                     f'{[len(r.get("boxes_3d", r.get("bboxes", []))) for r in bbox_results]}')
        return bbox_results

    def add_pred_to_datasample(self, batch_data_samples, bbox_results):
        """Add prediction results to Det3DDataSample.

        This converts the old-style bbox_results dicts into the new
        Det3DDataSample format expected by mmdet3d 1.1+.

        Args:
            batch_data_samples (list[:obj:`Det3DDataSample`]): Data samples.
            bbox_results (list[dict]): Prediction results from
                simple_test_pts.

        Returns:
            list[:obj:`Det3DDataSample`]: Updated data samples.
        """
        for data_sample, bbox_result in zip(batch_data_samples, bbox_results):
            pred_instances_3d = InstanceData()

            # Handle both old-style ('bboxes') and new-style ('boxes_3d') keys
            if 'boxes_3d' in bbox_result:
                pred_instances_3d.bboxes_3d = bbox_result['boxes_3d']
                pred_instances_3d.scores_3d = bbox_result['scores_3d']
                pred_instances_3d.labels_3d = bbox_result['labels_3d']
            else:
                # Fallback for old-style bbox3d2result output
                pred_instances_3d.bboxes_3d = bbox_result['bboxes']
                pred_instances_3d.scores_3d = bbox_result['scores']
                pred_instances_3d.labels_3d = bbox_result['labels']

            data_sample.pred_instances_3d = pred_instances_3d

        return batch_data_samples


logger.info('[deep_interaction] ✓ Registered DeepInteraction to MODELS')
logger.info('[deep_interaction] ✓ Module fully loaded')