# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for PillarNeSt from mmdet3d 0.18 to 1.x
#pillarnest_center_head.py
import copy
import os
import sys
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor, nn
import numpy as np
from mmengine.structures import InstanceData
from mmdet3d.models.dense_heads import CenterHead
from mmdet3d.models.layers import circle_nms
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.registry import MODELS, TASK_UTILS


# =============================================================================
# File-backed debug printer (mirrors focalformer3d._dprint)
# =============================================================================
_debug_file = None

def _dprint(msg: str) -> None:
    """Print to stdout AND to DEBUG_OUT_FILE (if set)."""
    global _debug_file
    print(msg, flush=True)
    if _debug_file is None:
        path = os.environ.get('DEBUG_OUT_FILE', '')
        if path:
            _debug_file = open(path, 'a', buffering=1)
            _debug_file.write(f'[PillarNestCenterHead debug file opened]\n')
            _debug_file.flush()
    if _debug_file is not None:
        _debug_file.write(msg + '\n')
        _debug_file.flush()


@MODELS.register_module()
class PillarNestCenterHead(CenterHead):
    """CenterHead for PillarNeSt with IoU-aware scoring and debug prints."""

    def __init__(
        self,
        in_channels: Union[List[int], int] = [128],
        tasks: Optional[List[Dict]] = None,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        bbox_coder: Optional[Dict] = None,
        common_heads: Dict = dict(),
        legacy_iou_transform: bool = False,
        loss_cls: Dict = dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox: Dict = dict(type='L1Loss', reduction='none', loss_weight=0.25),
        loss_iou_reg: Optional[str] = None,
        iou_reg: Dict = dict(type='BboxOverlaps3D', coordinate='lidar'),
        iou_reg_weight: float = 0.25,
        iou_score: Dict = dict(type='BboxOverlaps3D', coordinate='lidar'),
        loss_iou_score: Dict = dict(type='L1Loss', reduction='none', loss_weight=1.0),
        iou_score_weight: float = 1.0,
        separate_head: Dict = dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
        share_conv_channel: int = 64,
        num_heatmap_convs: int = 2,
        conv_cfg: Dict = dict(type='Conv2d'),
        norm_cfg: Dict = dict(type='BN2d'),
        bias: str = 'auto',
        norm_bbox: bool = True,
        init_cfg: Optional[Dict] = None,
        debug: bool = False,
        debug_max_print: int = 200
    ) -> None:
        assert init_cfg is None, (
            'To prevent abnormal initialization behavior, init_cfg is not allowed to be set'
        )

        super().__init__(
            in_channels=in_channels,
            tasks=tasks,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
            share_conv_channel=share_conv_channel,
            num_heatmap_convs=num_heatmap_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias,
            norm_bbox=norm_bbox,
            init_cfg=init_cfg)

        self.debug = debug
        self.debug_max_print = debug_max_print
        self._debug_count = 0
        self.legacy_iou_transform = legacy_iou_transform

        self.with_iou_score = 'iou' in common_heads
        if self.with_iou_score:
            self.iou_score_calculator = TASK_UTILS.build(iou_score)
            self.iou_score_beta = test_cfg.get('iou_score_beta', 0.5) if test_cfg else 0.5
            self.loss_iou_score = MODELS.build(loss_iou_score)
            self.iou_score_weight = iou_score_weight

        if loss_iou_reg is not None:
            self.with_iou_reg = True
            self.iou_reg_calculator = TASK_UTILS.build(iou_reg)
            self.loss_iou_reg = loss_iou_reg
            self.loss_reg_weight = iou_reg_weight
        else:
            self.with_iou_reg = False

        self._dbg(
            f'init with_iou_score={self.with_iou_score}, '
            f'with_iou_reg={self.with_iou_reg}, num_tasks={len(self.num_classes)}',
            force=True)

    def _dbg(self, msg: str, force: bool = False) -> None:
        """Write to stdout + DEBUG_OUT_FILE."""
        if self.debug and (force or self._debug_count < self.debug_max_print):
            _dprint(f'[PillarNestCenterHead] {msg}')
            self._debug_count += 1

    def _gather_feat(self, feat: Tensor, ind: Tensor) -> Tensor:
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        return feat

    def loss(self, pts_feats, batch_data_samples, **kwargs):
        preds_dicts = self(pts_feats)
        batch_gt_instances_3d = [data_sample.gt_instances_3d for data_sample in batch_data_samples]
        self._dbg(
            f'loss called: num_tasks={len(preds_dicts)}, '
            f'batch={len(batch_gt_instances_3d)}')
        return self.loss_by_feat(preds_dicts, batch_gt_instances_3d)

    def get_really_bboxes(self, pred: Tensor, ind: Tensor) -> Tensor:
        """Convert encoded predictions to world-space boxes [x,y,z,dx,dy,dz,yaw]."""
        out_size_factor = self.train_cfg['out_size_factor']
        voxel_size = self.train_cfg['voxel_size']
        pc_range = self.train_cfg['point_cloud_range']
        grid_size = torch.tensor(self.train_cfg['grid_size'], device=pred.device)

        feature_map_size = grid_size[:2] // out_size_factor
        xs = ind % feature_map_size[1]
        ys = ind // feature_map_size[1]

        pred_core = pred[:, :, :8].clone()

        xs = xs[:, :, None].float() + pred_core[:, :, 0:1]
        ys = ys[:, :, None].float() + pred_core[:, :, 1:2]

        xs = xs * out_size_factor * voxel_size[0] + pc_range[0]
        ys = ys * out_size_factor * voxel_size[1] + pc_range[1]

        z = pred_core[:, :, 2:3]
        dims_log = pred_core[:, :, 3:6]
        rot_sin = pred_core[:, :, 6:7]
        rot_cos = pred_core[:, :, 7:8]

        if self.legacy_iou_transform:
            dims_log = dims_log[:, :, [1, 0, 2]]
            yaw = -torch.atan2(rot_sin, rot_cos) - (np.pi / 2)
        else:
            yaw = torch.atan2(rot_sin, rot_cos)

        dim = torch.exp(torch.clamp(dims_log, min=-5, max=5))
        return torch.cat([xs, ys, z, dim, yaw], dim=2)

    def loss_by_feat(self, preds_dicts, batch_gt_instances_3d, *args, **kwargs):
        heatmaps, anno_boxes, inds, masks = self.get_targets(batch_gt_instances_3d)
        loss_dict = dict()

        for task_id, preds_dict in enumerate(preds_dicts):
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))

            head_tensors = [
                preds_dict[0]['reg'],
                preds_dict[0]['height'],
                preds_dict[0]['dim'],
                preds_dict[0]['rot'],
            ]
            if 'vel' in preds_dict[0]:
                head_tensors.append(preds_dict[0]['vel'])

            if self.with_iou_score:
                preds_dict[0]['iou'] = clip_sigmoid(preds_dict[0]['iou'])
                head_tensors.append(preds_dict[0]['iou'])

            preds_dict[0]['anno_box'] = torch.cat(head_tensors, dim=1)

            target_box = anno_boxes[task_id]
            ind = inds[task_id]
            num = masks[task_id].float().sum()

            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)

            if self.with_iou_score or self.with_iou_reg:
                batch_size = pred.size(0)
                real_pred_bbox = self.get_really_bboxes(pred[:, :, :8], ind).view(-1, 7)
                real_target_bbox = self.get_really_bboxes(target_box[:, :, :8], ind).view(-1, 7)

                if self.with_iou_score:
                    iou_score = self.iou_score_calculator(real_target_bbox, real_pred_bbox)
                    iou_score = torch.diag(iou_score).view(batch_size, -1, 1).detach()

                    iou_score_pred = pred[:, :, -1:]
                    iou_mask = masks[task_id].unsqueeze(2).expand_as(iou_score).float()
                    iou_mask *= (~torch.isnan(iou_score)).float()

                    loss_iou_score = self.loss_iou_score(
                        iou_score_pred, iou_score, iou_mask, avg_factor=(num + 1e-4))
                    loss_iou_score *= self.iou_score_weight
                    loss_dict[f'task{task_id}.loss_iou_score'] = loss_iou_score
                    pred = pred[:, :, :-1]

                if self.with_iou_reg:
                    iou = self.iou_reg_calculator(real_target_bbox, real_pred_bbox)
                    iou = torch.diag(iou).view(batch_size, -1)
                    if self.loss_iou_reg in ['IoU', 'iou']:
                        loss_iou_reg = ((1. - iou) * masks[task_id]).sum() / (masks[task_id].sum() + 1e-4)
                        loss_iou_reg *= self.loss_reg_weight
                        loss_dict[f'task{task_id}.loss_iou_reg'] = loss_iou_reg

            if pred.size(-1) > target_box.size(-1):
                pred = pred[:, :, :target_box.size(-1)]

            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)

            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights,
                avg_factor=(num + 1e-4))

            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox

            self._dbg(
                f'loss task={task_id}, pos={int(num_pos)}, '
                f'heatmap={float(loss_heatmap):.4f}, bbox={float(loss_bbox):.4f}')

        return loss_dict

    def predict_by_feat(self,
                        preds_dicts: List[Dict],
                        batch_input_metas: List[Dict],
                        rescale: bool = False,
                        **kwargs) -> List[InstanceData]:
        """Generate bboxes from bbox head predictions (mmdet3d 1.x API)."""
        self._dbg(
            f'predict_by_feat called: tasks={len(preds_dicts)}, '
            f'batch={preds_dicts[0][0]["heatmap"].shape[0] if preds_dicts else 0}',
            force=True)

        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]

            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()
            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']
            batch_dim = torch.exp(preds_dict[0]['dim']) if self.norm_bbox else preds_dict[0]['dim']
            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)
            batch_vel = preds_dict[0]['vel'] if 'vel' in preds_dict[0] else None
            batch_iou = preds_dict[0]['iou'].sigmoid() if self.with_iou_score else None

            self._dbg(f"task={task_id} batch_dim shape before decode: {tuple(batch_dim.shape)}", force=True)

            decoded = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                iou_scores=batch_iou,
                task_id=task_id)

            if len(decoded) > 0:
                n = decoded[0]['scores'].numel()
                self._dbg(f'task={task_id} decoded sample0 boxes={n}')

            if self.with_iou_score:
                for i in range(len(decoded)):
                    if 'iou_scores' in decoded[i]:
                        decoded[i]['scores'] = (
                            torch.pow(decoded[i]['scores'], 1 - self.iou_score_beta) *
                            torch.pow(decoded[i]['iou_scores'], self.iou_score_beta)
                        )

            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = decoded[i]['bboxes']
                    scores = decoded[i]['scores']
                    labels = decoded[i]['labels']

                    if boxes3d.numel() == 0:
                        ret_task.append(dict(bboxes=boxes3d, scores=scores, labels=labels))
                        continue

                    centers = boxes3d[:, [0, 1]]
                    boxes_for_nms = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = circle_nms(
                        boxes_for_nms.detach().cpu().numpy(),
                        self.test_cfg['min_radius'][task_id],
                        post_max_size=self.test_cfg['post_max_size'])
                    keep = torch.tensor(keep, dtype=torch.long, device=boxes3d.device)

                    ret_task.append(
                        dict(
                            bboxes=boxes3d[keep],
                            scores=scores[keep],
                            labels=labels[keep]))
                    self._dbg(f'task={task_id}, sample={i}, after_circle_nms={int(keep.numel())}')
                rets.append(ret_task)
            else:
                batch_reg_preds = [box['bboxes'] for box in decoded]
                batch_cls_preds = [box['scores'] for box in decoded]
                batch_cls_labels = [box['labels'] for box in decoded]
                rets.append(
                    self.get_task_detections(
                        num_class_with_bg,
                        batch_cls_preds,
                        batch_reg_preds,
                        batch_cls_labels,
                        batch_input_metas))

        # merge all tasks
        num_samples = len(rets[0])
        ret_list = []
        for i in range(num_samples):
            temp_instances = InstanceData()

            bboxes = torch.cat([ret[i]['bboxes'] for ret in rets], dim=0)
            scores = torch.cat([ret[i]['scores'] for ret in rets], dim=0)

            flag = 0
            merged_labels = []
            for j, num_class in enumerate(self.num_classes):
                labels_j = rets[j][i]['labels'].int() + flag
                merged_labels.append(labels_j)
                flag += num_class
            labels = torch.cat(merged_labels, dim=0)

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = batch_input_metas[i]['box_type_3d'](
                bboxes, self.bbox_coder.code_size)

            temp_instances.bboxes_3d = bboxes
            temp_instances.scores_3d = scores
            temp_instances.labels_3d = labels
            ret_list.append(temp_instances)

            self._dbg(
                f'final sample={i}, boxes={scores.numel()}, '
                f'score_max={scores.max().item():.4f}' if scores.numel() > 0
                else f'final sample={i}, boxes=0')

            # ----------------------------------------------------------------
            # Top-5 predictions (mirrors focalformer3d debug output)
            # ----------------------------------------------------------------
            if self.debug and scores.numel() > 0:
                top_k = min(5, scores.numel())
                top_scores, top_idx = torch.topk(scores, top_k)
                top_boxes = bboxes.tensor[top_idx]
                top_labels = labels[top_idx]
                label_names = ['Car', 'Pedestrian', 'Cyclist']
                _dprint(f'[PillarNestCenterHead] --- Sample {i} Top-{top_k} predictions ---')
                for rank, (box, score, label) in enumerate(
                        zip(top_boxes, top_scores, top_labels)):
                    lname = label_names[int(label)] if int(label) < len(label_names) else str(int(label))
                    x, y, z, dx, dy, dz, yaw = [float(v) for v in box[:7]]
                    _dprint(
                        f'[PillarNestCenterHead]   PRED[{rank:02d}] {lname}'
                        f'  score={float(score):.4f}'
                        f'  x={x:.2f} y={y:.2f} z={z:.2f}'
                        f'  dx={dx:.2f} dy={dy:.2f} dz={dz:.2f}'
                        f'  yaw={yaw:.3f}'
                    )

        return ret_list
