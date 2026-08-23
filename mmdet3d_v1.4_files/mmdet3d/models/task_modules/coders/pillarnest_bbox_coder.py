# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
from mmdet.models.task_modules import BaseBBoxCoder
from torch import Tensor

from mmdet3d.registry import TASK_UTILS
import numpy as np


@TASK_UTILS.register_module()
class PillarNestBBoxCoder(BaseBBoxCoder):
    """BBox coder for PillarNeSt (CenterPoint-style with optional IoU scores)."""

    def __init__(self,
                 pc_range: List[float],
                 out_size_factor: int,
                 voxel_size: List[float],
                 post_center_range: Optional[List[float]] = None,
                 max_num: int = 100,
                 legacy_dim_swap: bool = False,
                 legacy_yaw_transform: bool = False,
                 score_threshold: Optional[float] = None,
                 code_size: int = 9,
                 debug: bool = False,
                 debug_max_print: int = 10) -> None:
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size
        self.legacy_dim_swap = legacy_dim_swap
        self.legacy_yaw_transform = legacy_yaw_transform

        # Debug controls
        self.debug = debug
        self.debug_max_print = debug_max_print
        self._debug_count = 0

    def _dbg(self, msg: str) -> None:
        if self.debug and self._debug_count < self.debug_max_print:
            print(f'[PillarNestBBoxCoder] {msg}', flush=True)
            self._debug_count += 1

    def _gather_feat(self,
                     feats: Tensor,
                     inds: Tensor,
                     feat_masks: Optional[Tensor] = None) -> Tensor:
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores: Tensor, K: int = 80) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get top-k scores and corresponding indices."""
        batch, cat, height, width = scores.size()
        K = min(K, height * width)

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).to(dtype=scores.dtype)
        topk_xs = (topk_inds % width).to(dtype=scores.dtype)

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind // K).to(dtype=torch.long)

        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat: Tensor, ind: Tensor) -> Tensor:
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def encode(self, *args, **kwargs):
        raise NotImplementedError('PillarNestBBoxCoder.encode is not implemented.')

    def decode(self,
               heat: Tensor,
               rot_sine: Tensor,
               rot_cosine: Tensor,
               hei: Tensor,
               dim: Tensor,
               vel: Optional[Tensor],
               reg: Optional[Tensor] = None,
               iou_scores: Optional[Tensor] = None,
               task_id: int = -1) -> List[Dict[str, Tensor]]:
        """Decode bboxes from dense predictions."""
        self._dbg(
            f'decode(task_id={task_id}) heat={tuple(heat.shape)} '
            f'reg={None if reg is None else tuple(reg.shape)} '
            f'iou={None if iou_scores is None else tuple(iou_scores.shape)}'
        )

        batch, _, _, _ = heat.size()
        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds).view(batch, self.max_num, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        rot_sine = self._transpose_and_gather_feat(rot_sine, inds).view(batch, self.max_num, 1)
        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds).view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)
        if self.legacy_yaw_transform:
            rot = -rot - (np.pi / 2)

        hei = self._transpose_and_gather_feat(hei, inds).view(batch, self.max_num, 1)

        # Robust dim handling: supports both [B,C,H,W] and pre-gathered [B,K,3]
        if dim.dim() == 4:
            dim = self._transpose_and_gather_feat(dim, inds).view(batch, self.max_num, 3)
        elif dim.dim() == 3:
            if dim.shape[1] != self.max_num or dim.shape[2] != 3:
                raise RuntimeError(f'Unexpected gathered dim shape: {tuple(dim.shape)}')
        else:
            raise RuntimeError(f'Unexpected dim rank in decode: {dim.dim()}, shape={tuple(dim.shape)}')

        if self.legacy_dim_swap:
            dim = dim[..., [1, 0, 2]]


        '''
        # dim of the box dimension swap
        if dim.dim() == 4:
            dim = self._transpose_and_gather_feat(dim, inds).view(batch, self.max_num, 3)
        elif dim.dim() == 3:
            # already gathered [B, K, 3]
            if dim.shape[1] != self.max_num or dim.shape[2] != 3:
                raise RuntimeError(f'Unexpected gathered dim shape: {tuple(dim.shape)}')
        else:
            raise RuntimeError(f'Unexpected dim rank in decode: {dim.dim()}, shape={tuple(dim.shape)}')

        # apply w/l swap AFTER gather (always on [B, K, 3])
        dim = dim[..., [1, 0, 2]]
        '''


        final_scores = scores.view(batch, self.max_num)
        final_labels = clses.view(batch, self.max_num).long()

        final_iou_scores = None
        if iou_scores is not None:
            final_iou_scores = self._transpose_and_gather_feat(
                iou_scores, inds).view(batch, self.max_num)

        xs = xs * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

        if vel is None:
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:
            vel = self._transpose_and_gather_feat(vel, inds).view(batch, self.max_num, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        else:
            thresh_mask = torch.ones_like(final_scores, dtype=torch.bool)

        if self.post_center_range is None:
            raise NotImplementedError(
                'post_center_range must be set for PillarNestBBoxCoder.decode.')

        post_center_range = heat.new_tensor(self.post_center_range)
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)
        mask &= thresh_mask

        predictions_dicts = []
        for i in range(batch):
            cmask = mask[i]
            boxes3d = final_box_preds[i, cmask]
            scores_i = final_scores[i, cmask]
            labels_i = final_labels[i, cmask]

            pred = dict(
                bboxes=boxes3d,
                scores=scores_i,
                labels=labels_i
            )
            if final_iou_scores is not None:
                pred['iou_scores'] = final_iou_scores[i, cmask]

            predictions_dicts.append(pred)
            self._dbg(f'batch={i} kept_boxes={boxes3d.shape[0]}')

        return predictions_dicts
