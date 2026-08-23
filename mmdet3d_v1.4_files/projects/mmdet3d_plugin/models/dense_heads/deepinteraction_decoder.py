# =============================================================================
# deep_interaction_decoder.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - HEADS from mmdet3d.models.builder      → MODELS from mmdet3d.registry
#   - build_loss from mmdet3d.models.builder  → MODELS.build(loss_cfg)
#   - build_bbox_coder from mmdet.core        → TASK_UTILS.build(coder_cfg)
#   - build_assigner from mmdet.core          → TASK_UTILS.build(assigner_cfg)
#   - build_sampler from mmdet.core           → TASK_UTILS.build(sampler_cfg)
#   - multi_apply from mmdet.core             → mmdet.models.utils.multi_apply
#   - AssignResult from mmdet.core            → mmdet.models.task_modules.assigners
#   - force_fp32 from mmcv.runner             → REMOVED (use AmpOptimWrapper)
#   - circle_nms from mmdet3d.core            → mmdet3d.models.layers
#   - draw_heatmap_gaussian, gaussian_radius
#     from mmdet3d.core                       → mmdet3d.models.utils.gaussian
#   - xywhr2xyxyr from mmdet3d.core           → mmdet3d.structures
#   - PseudoSampler from mmdet3d.core         → mmdet3d.models.task_modules
#     or mmdet.models.task_modules.samplers
#   - clip_sigmoid from mmdet3d.models.utils  → mmdet3d.models.utils.clip_sigmoid
#   - nms_gpu from mmdet3d.ops                → nms_bev from mmdet3d.models.layers
#   - ConvModule, build_conv_layer from mmcv.cnn → still in mmcv.cnn (unchanged)
#   - import pdb                              → REMOVED
# =============================================================================
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.info('[deep_interaction_decoder] Loading module...')

# --- mmcv.cnn (unchanged in mmcv 2.x) ---
try:
    from mmcv.cnn import ConvModule, build_conv_layer
    logger.info('[deep_interaction_decoder] ✓ Imported ConvModule, '
                'build_conv_layer from mmcv.cnn')
except ImportError as e:
    logger.error(f'[deep_interaction_decoder] ✗ mmcv.cnn imports: {e}')
    raise

# --- Registry: HEADS → MODELS ---
try:
    from mmdet3d.registry import MODELS, TASK_UTILS
    logger.info('[deep_interaction_decoder] ✓ Imported MODELS, TASK_UTILS '
                'from mmdet3d.registry')
except ImportError as e:
    logger.error(f'[deep_interaction_decoder] ✗ Registry imports: {e}')
    raise

# --- Structures (old: mmdet3d.core) ---
try:
    from mmdet3d.structures import xywhr2xyxyr
    logger.info('[deep_interaction_decoder] ✓ Imported xywhr2xyxyr '
                'from mmdet3d.structures')
except ImportError as e:
    logger.error(f'[deep_interaction_decoder] ✗ xywhr2xyxyr: {e}')
    raise

# --- NMS: nms_gpu → nms_bev, circle_nms ---
try:
    from mmdet3d.models.layers import nms_bev, circle_nms
    logger.info('[deep_interaction_decoder] ✓ Imported nms_bev, circle_nms '
                'from mmdet3d.models.layers')
except ImportError as e:
    logger.error(f'[deep_interaction_decoder] ✗ NMS imports: {e}')
    logger.error('  → nms_gpu moved to nms_bev, circle_nms moved from '
                 'mmdet3d.core to mmdet3d.models.layers')
    raise

# --- Gaussian heatmap utils ---
try:
    from mmdet3d.models.utils.gaussian import (draw_heatmap_gaussian,
                                                gaussian_radius)
    logger.info('[deep_interaction_decoder] ✓ Imported draw_heatmap_gaussian, '
                'gaussian_radius from mmdet3d.models.utils.gaussian')
except ImportError as e:
    logger.error(f'[deep_interaction_decoder] ✗ Gaussian utils: {e}')
    raise

# --- clip_sigmoid ---
try:
    from mmdet3d.models.utils.clip_sigmoid import clip_sigmoid
    logger.info('[deep_interaction_decoder] ✓ Imported clip_sigmoid')
except ImportError:
    try:
        from mmdet3d.models.utils import clip_sigmoid
        logger.info('[deep_interaction_decoder] ✓ Imported clip_sigmoid '
                    '(from mmdet3d.models.utils)')
    except ImportError as e:
        logger.error(f'[deep_interaction_decoder] ✗ clip_sigmoid: {e}')
        raise

# --- multi_apply ---
try:
    from mmdet.models.utils import multi_apply
    logger.info('[deep_interaction_decoder] ✓ Imported multi_apply '
                'from mmdet.models.utils')
except ImportError as e:
    logger.error(f'[deep_interaction_decoder] ✗ multi_apply: {e}')
    raise

# --- AssignResult ---
try:
    from mmdet.models.task_modules.assigners import AssignResult
    logger.info('[deep_interaction_decoder] ✓ Imported AssignResult '
                'from mmdet.models.task_modules.assigners')
except ImportError as e:
    logger.error(f'[deep_interaction_decoder] ✗ AssignResult: {e}')
    raise

# --- PseudoSampler ---
try:
    from mmdet.models.task_modules.samplers import PseudoSampler
    logger.info('[deep_interaction_decoder] ✓ Imported PseudoSampler '
                'from mmdet.models.task_modules.samplers')
except ImportError:
    try:
        from mmdet3d.models.task_modules import PseudoSampler
        logger.info('[deep_interaction_decoder] ✓ Imported PseudoSampler '
                    '(from mmdet3d fallback)')
    except ImportError as e:
        logger.error(f'[deep_interaction_decoder] ✗ PseudoSampler: {e}')
        raise

# --- Local decoder utils (unchanged — these are project-internal) ---
try:
    from projects.mmdet3d_plugin.models.utils.decoder_utils import (
        ImageRCNNBlock, PointRCNNBlock, PositionEmbeddingLearned,
        TransformerDecoderLayer, FFN)
    logger.info('[deep_interaction_decoder] ✓ Imported decoder_utils')
except ImportError as e:
    logger.error(f'[deep_interaction_decoder] ✗ decoder_utils: {e}')
    raise


# ===================================================================
# DeepInteractionDecoder
# ===================================================================
@MODELS.register_module()
class DeepInteractionDecoder(nn.Module):
    def __init__(self,
                 num_views=0,
                 out_size_factor_img=4,
                 num_proposals=128,
                 auxiliary=True,
                 hidden_channel=128,
                 num_classes=4,
                 # config for Transformer
                 num_mmpi=4,
                 num_decoder_layers=1,
                 num_heads=8,
                 learnable_query_pos=False,
                 initialize_by_heatmap=False,
                 nms_kernel_size=1,
                 ffn_channel=256,
                 dropout=0.1,
                 bn_momentum=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(
                     type='GaussianFocalLoss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 ret_idx=None,
                 ):
        super(DeepInteractionDecoder, self).__init__()

        logger.info(f'[DeepInteractionDecoder] Building: '
                    f'num_proposals={num_proposals}, '
                    f'num_classes={num_classes}, '
                    f'num_mmpi={num_mmpi}, '
                    f'num_decoder_layers={num_decoder_layers}, '
                    f'initialize_by_heatmap={initialize_by_heatmap}')

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.learnable_query_pos = learnable_query_pos
        self.initialize_by_heatmap = initialize_by_heatmap
        self.nms_kernel_size = nms_kernel_size
        if self.initialize_by_heatmap is True:
            assert self.learnable_query_pos is False, \
                ("initialized by heatmap is conflicting with "
                 "learnable query position")
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        # --- Build losses via MODELS.build() ---
        logger.debug(f'[DeepInteractionDecoder] Building loss_cls: '
                     f'{loss_cls["type"]}')
        self.loss_cls = MODELS.build(loss_cls)
        logger.debug(f'[DeepInteractionDecoder] Building loss_bbox: '
                     f'{loss_bbox["type"]}')
        self.loss_bbox = MODELS.build(loss_bbox)
        logger.debug(f'[DeepInteractionDecoder] Building loss_heatmap: '
                     f'{loss_heatmap["type"]}')
        self.loss_heatmap = MODELS.build(loss_heatmap)

        # --- Build bbox_coder via TASK_UTILS.build() ---
        logger.debug(f'[DeepInteractionDecoder] Building bbox_coder: '
                     f'{bbox_coder["type"] if bbox_coder else None}')
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.sampling = False

        if self.initialize_by_heatmap:
            layers = []
            layers.append(ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
            layers.append(build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
            self.heatmap_head = nn.Sequential(*layers)
            self.heatmap_head_img = copy.deepcopy(self.heatmap_head)
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        else:
            self.query_feat = nn.Parameter(
                torch.randn(1, hidden_channel, self.num_proposals))
            self.query_pos = nn.Parameter(
                torch.rand([1, self.num_proposals, 2]),
                requires_grad=learnable_query_pos)

        # transformer decoder layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout,
                    activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(
                        2, hidden_channel),
                ))

        # Prediction Heads
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(
                dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                FFN(hidden_channel, heads, conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg, bias=bias))

        self.decode_head = nn.ModuleList()
        self.pred_head = nn.ModuleList()
        self.num_mmpi = num_mmpi
        assert self.num_mmpi % 2 == 0
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        for i in range(int(self.num_mmpi / 2)):
            heads = copy.deepcopy(common_heads)
            heads.update(
                dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.decode_head.append(
                ImageRCNNBlock(
                    self.num_views, self.num_proposals,
                    self.out_size_factor_img, self.test_cfg,
                    self.bbox_coder,
                    hidden_channel, num_heads, dropout
                )
            )
            self.pred_head.append(
                FFN(hidden_channel * 2, heads, conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg, bias=bias))

            self.decode_head.append(
                PointRCNNBlock(
                    hidden_channel, num_heads, dropout, self.bbox_coder
                )
            )
            self.pred_head.append(
                FFN(hidden_channel * 2, heads, conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg, bias=bias))

        x_size = (self.test_cfg['grid_size'][0] //
                  self.test_cfg['out_size_factor'])
        y_size = (self.test_cfg['grid_size'][1] //
                  self.test_cfg['out_size_factor'])
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

        self.init_weights()
        self._init_assigner_sampler()

        self.ret_idx = ret_idx

        logger.info(f'[DeepInteractionDecoder] ✓ Built successfully')

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat(
            [batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()
        logger.debug('[DeepInteractionDecoder] init_weights complete')

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        # --- build_sampler → TASK_UTILS.build() ---
        if self.sampling:
            logger.debug('[DeepInteractionDecoder] Building bbox_sampler '
                         'via TASK_UTILS')
            self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()

        # --- build_assigner → TASK_UTILS.build() ---
        if isinstance(self.train_cfg.assigner, dict):
            logger.debug(f'[DeepInteractionDecoder] Building bbox_assigner: '
                         f'{self.train_cfg.assigner["type"]}')
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                TASK_UTILS.build(res) for res in self.train_cfg.assigner
            ]
            logger.debug(f'[DeepInteractionDecoder] Built '
                         f'{len(self.bbox_assigner)} bbox_assigners')

    def forward(self, pts_inputs, img_inputs, img_metas):
        """Forward function.

        Args:
            pts_inputs: Point cloud features.
            img_inputs: Image features.
            img_metas: Image metadata.

        Returns:
            list[dict]: Output results.
        """
        lidar_feat = pts_inputs[0]
        batch_size = lidar_feat.shape[0]
        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        img_feat = img_inputs
        BN, I_C, I_H, I_W = img_feat.shape
        img_h = I_H
        img_w = I_W
        new_lidar_feat = pts_inputs[1]
        bev_feat = new_lidar_feat.view(
            batch_size, new_lidar_feat.shape[1], -1)

        logger.debug(f'[DeepInteractionDecoder.forward] batch_size={batch_size}, '
                     f'lidar_feat={lidar_feat.shape}, '
                     f'img_feat={img_feat.shape}')

        dense_heatmap = self.heatmap_head(lidar_feat)
        dense_heatmap_img = self.heatmap_head_img(
            bev_feat.view(lidar_feat.shape))
        heatmap = (dense_heatmap.detach().sigmoid() +
                   dense_heatmap_img.detach().sigmoid()) / 2
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                  padding:(-padding)] = local_max_inner

        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(
                heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(
                heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg['dataset'] == 'Waymo':
            local_max[:, 1, ] = F.max_pool2d(
                heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(
                heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, lidar_feat_flatten.shape[1], -1),
            dim=-1)
        self.query_labels = top_proposals_class

        logger.debug(f'[DeepInteractionDecoder.forward] '
                     f'top_proposals_class unique: '
                     f'{top_proposals_class.unique().tolist()}')

        one_hot = F.one_hot(
            top_proposals_class,
            num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1)

        ret_dicts = []
        for i in range(self.num_decoder_layers):
            query_feat = self.decoder[i](
                query_feat, lidar_feat_flatten, query_pos, bev_pos)
            res_layer = self.prediction_heads[i](query_feat)
            res_layer['center'] = (res_layer['center'] +
                                   query_pos.permute(0, 2, 1))
            first_res_layer = res_layer
            query_pos = (res_layer['center'].detach().clone()
                         .permute(0, 2, 1))

        img_feat_flatten = img_feat.view(
            batch_size, self.num_views, img_feat.shape[1], -1)

        if self.img_feat_pos is None:
            (h, w) = img_inputs.shape[-2], img_inputs.shape[-1]
            img_feat_pos = self.img_feat_pos = self.create_2D_grid(
                h, w).to(img_feat_flatten.device)
        else:
            img_feat_pos = self.img_feat_pos

        self.on_the_image_mask = []

        for layer_idx in range(self.num_mmpi):
            prev_query_feat = query_feat.clone()
            query_pos = (res_layer['center'].detach().clone()
                         .permute(0, 2, 1))
            query_feat, on_the_image = self.decode_head[layer_idx](
                query_feat=prev_query_feat, res_layer=res_layer,
                new_lidar_feat=new_lidar_feat,
                img_feat_flatten=img_feat_flatten,
                img_metas=img_metas, img_h=img_h, img_w=img_w
            )
            res_layer = self.pred_head[layer_idx](
                torch.cat([query_feat, prev_query_feat], dim=1))
            res_layer['center'] = (res_layer['center'] +
                                   query_pos.permute(0, 2, 1))
            if layer_idx % 2 == 0:
                self.on_the_image_mask.append(on_the_image != -1)
                for key, value in res_layer.items():
                    pred_dim = value.shape[1]
                    mask = ~self.on_the_image_mask[-1].unsqueeze(1).repeat(
                        1, pred_dim, 1)
                    res_layer[key][mask] = first_res_layer[key][mask]
            ret_dicts.append(res_layer)

        if self.initialize_by_heatmap:
            ret_dicts[0]['query_heatmap_score'] = heatmap.gather(
                index=top_proposals_index[:, None, :].expand(
                    -1, self.num_classes, -1),
                dim=-1)
            ret_dicts[0]['dense_heatmap'] = dense_heatmap_img

        if self.auxiliary is False:
            return [ret_dicts[-1]]

        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ['dense_heatmap', 'dense_heatmap_old',
                           'query_heatmap_score']:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]

        logger.debug(f'[DeepInteractionDecoder.forward] '
                     f'Returning {len(new_res)} keys, '
                     f'center shape={new_res["center"].shape}')
        return [[new_res]]

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets."""
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d,
            list_of_pred_dict, np.arange(len(gt_labels_3d)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])

        logger.debug(f'[DeepInteractionDecoder.get_targets] '
                     f'num_pos={num_pos}, matched_ious={matched_ious:.4f}')

        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            return (labels, label_weights, bbox_targets, bbox_weights,
                    ious, num_pos, matched_ious, heatmap)
        else:
            return (labels, label_weights, bbox_targets, bbox_weights,
                    ious, num_pos, matched_ious)

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d,
                           preds_dict, batch_idx):
        """Generate training targets for a single sample."""
        num_proposals = preds_dict['center'].shape[-1]

        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height, vel)
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)

        if self.auxiliary:
            num_layer = self.num_mmpi
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[
                self.num_proposals * idx_layer:
                self.num_proposals * (idx_layer + 1), :]
            score_layer = score[
                ..., self.num_proposals * idx_layer:
                self.num_proposals * (idx_layer + 1)]

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer, gt_bboxes_tensor,
                    gt_labels_3d, score_layer, self.train_cfg)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer, gt_bboxes_tensor,
                    None, gt_labels_3d,
                    self.query_labels[batch_idx])
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat(
                [res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat(
                [res.max_overlaps for res in assign_result_list]),
            labels=torch.cat(
                [res.labels for res in assign_result_list]),
        )

        sampling_result = self.bbox_sampler.sample(
            assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        logger.debug(f'[get_targets_single] batch={batch_idx}, '
                     f'pos={len(pos_inds)}, neg={len(neg_inds)}, '
                     f'gt_boxes={len(gt_bboxes_tensor)}')

        bbox_targets = torch.zeros(
            [num_proposals, self.bbox_coder.code_size]).to(center.device)
        bbox_weights = torch.zeros(
            [num_proposals, self.bbox_coder.code_size]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(
            num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:
            labels += self.num_classes

        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        if self.initialize_by_heatmap:
            device = labels.device
            gt_bboxes_3d = torch.cat(
                [gt_bboxes_3d.gravity_center,
                 gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device)
            grid_size = torch.tensor(self.train_cfg['grid_size'])
            pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
            voxel_size = torch.tensor(self.train_cfg['voxel_size'])
            feature_map_size = (grid_size[:2] //
                                self.train_cfg['out_size_factor'])
            heatmap = gt_bboxes_3d.new_zeros(
                self.num_classes, feature_map_size[1], feature_map_size[0])
            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3]
                length = gt_bboxes_3d[idx][4]
                width = (width / voxel_size[0] /
                         self.train_cfg['out_size_factor'])
                length = (length / voxel_size[1] /
                          self.train_cfg['out_size_factor'])
                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                    coor_x = ((x - pc_range[0]) / voxel_size[0] /
                              self.train_cfg['out_size_factor'])
                    coor_y = ((y - pc_range[1]) / voxel_size[1] /
                              self.train_cfg['out_size_factor'])

                    center = torch.tensor(
                        [coor_x, coor_y], dtype=torch.float32,
                        device=device)
                    center_int = center.to(torch.int32)
                    draw_heatmap_gaussian(
                        heatmap[gt_labels_3d[idx]], center_int, radius)

            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return (labels[None], label_weights[None], bbox_targets[None],
                    bbox_weights[None], ious[None], int(pos_inds.shape[0]),
                    float(mean_iou), heatmap[None])

        else:
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return (labels[None], label_weights[None], bbox_targets[None],
                    bbox_weights[None], ious[None], int(pos_inds.shape[0]),
                    float(mean_iou))

    # NOTE: @force_fp32 is REMOVED in mmdet3d 1.1+.
    # FP32 enforcement is now handled by AmpOptimWrapper.
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for DeepInteractionDecoder.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): GT boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss dict.
        """
        if self.initialize_by_heatmap:
            (labels, label_weights, bbox_targets, bbox_weights,
             ious, num_pos, matched_ious, heatmap) = self.get_targets(
                gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        else:
            (labels, label_weights, bbox_targets, bbox_weights,
             ious, num_pos, matched_ious) = self.get_targets(
                gt_bboxes_3d, gt_labels_3d, preds_dicts[0])

        num_pos = []
        if hasattr(self, 'on_the_image_mask'):
            for idx_layer in range(self.num_mmpi):
                s = idx_layer * self.num_proposals
                e = (idx_layer + 1) * self.num_proposals
                if idx_layer % 2 == 0:
                    mask = self.on_the_image_mask[idx_layer // 2]
                    label_weights[..., s:e] = (
                        label_weights[..., s:e] * mask)
                    bbox_weights[:, s:e, :] = (
                        bbox_weights[:, s:e, :] * mask[:, :, None])
                num_pos.append(
                    bbox_weights.max(-1).values[..., s:e].sum())
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        if self.initialize_by_heatmap:
            loss_heatmap = self.loss_heatmap(
                clip_sigmoid(preds_dict['dense_heatmap']),
                heatmap,
                avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            loss_dict['loss_heatmap'] = loss_heatmap

        for idx_layer in range(self.num_mmpi):
            prefix = f'layer_{idx_layer}'
            s = idx_layer * self.num_proposals
            e = (idx_layer + 1) * self.num_proposals

            layer_labels = labels[..., s:e].reshape(-1)
            layer_label_weights = label_weights[..., s:e].reshape(-1)
            layer_score = preds_dict['heatmap'][..., s:e]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(
                -1, self.num_classes)
            layer_loss_cls = self.loss_cls(
                layer_cls_score, layer_labels, layer_label_weights,
                avg_factor=max(num_pos[idx_layer], 1))

            layer_center = preds_dict['center'][..., s:e]
            layer_height = preds_dict['height'][..., s:e]
            layer_rot = preds_dict['rot'][..., s:e]
            layer_dim = preds_dict['dim'][..., s:e]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot],
                dim=1).permute(0, 2, 1)
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., s:e]
                preds = torch.cat(
                    [layer_center, layer_height, layer_dim,
                     layer_rot, layer_vel],
                    dim=1).permute(0, 2, 1)
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, s:e, :]
            layer_reg_weights = (layer_bbox_weights *
                                 layer_bbox_weights.new_tensor(code_weights))
            layer_bbox_targets = bbox_targets[:, s:e, :]
            layer_loss_bbox = self.loss_bbox(
                preds, layer_bbox_targets, layer_reg_weights,
                avg_factor=max(num_pos[idx_layer], 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox

            logger.debug(f'[loss] {prefix}: cls={layer_loss_cls.item():.4f}, '
                         f'bbox={layer_loss_bbox.item():.4f}, '
                         f'num_pos={num_pos[idx_layer]}')

        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None,
                   rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[list[dict]]: Decoded bbox, scores and labels.
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][
                ..., -self.num_proposals:].sigmoid()
            one_hot = F.one_hot(
                self.query_labels,
                num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = (batch_score *
                           preds_dict[0]['query_heatmap_score'] *
                           one_hot)

            batch_center = preds_dict[0]['center'][
                ..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][
                ..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][
                    ..., -self.num_proposals:]

            temp = self.bbox_coder.decode(
                batch_score, batch_rot, batch_dim,
                batch_center, batch_height, batch_vel, filter=True)

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(num_class=8, class_names=[],
                         indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                    dict(num_class=1, class_names=['pedestrian'],
                         indices=[8], radius=0.175),
                    dict(num_class=1, class_names=['traffic_cone'],
                         indices=[9], radius=0.175),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(num_class=1, class_names=['Car'],
                         indices=[0], radius=0.7),
                    dict(num_class=1, class_names=['Pedestrian'],
                         indices=[1], radius=0.7),
                    dict(num_class=1, class_names=['Cyclist'],
                         indices=[2], radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']

                logger.debug(f'[get_bboxes] batch[{i}]: '
                             f'{len(boxes3d)} boxes before NMS')

                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat(
                                    [boxes3d[task_mask][:, :2],
                                     scores[:, None][task_mask]], dim=1)
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    img_metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                # nms_gpu → nms_bev
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg[
                                        'pre_maxsize'],
                                    post_max_size=self.test_cfg[
                                        'post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(
                                task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(
                                task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(bboxes=boxes3d[keep_mask],
                               scores=scores[keep_mask],
                               labels=labels[keep_mask])

                    logger.debug(f'[get_bboxes] batch[{i}]: '
                                 f'{keep_mask.sum().item()} boxes '
                                 f'after NMS')
                else:
                    ret = dict(bboxes=boxes3d, scores=scores,
                               labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)

        assert len(rets) == 1
        assert len(rets[0]) == 1
        res = [[
            img_metas[0]['box_type_3d'](
                rets[0][0]['bboxes'],
                box_dim=rets[0][0]['bboxes'].shape[-1]),
            rets[0][0]['scores'],
            rets[0][0]['labels'].int()
        ]]
        return res


logger.info('[deep_interaction_decoder] ✓ Registered '
            'DeepInteractionDecoder to MODELS')
logger.info('[deep_interaction_decoder] ✓ Module fully loaded')
