# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for PillarNeSt from mmdet3d 0.18 to 1.x
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS
from .pillarnest_utils import PFNLayer, SEPFNLayer, SEPFNLayerV2, get_paddings_indicator


@MODELS.register_module()
class PillarNestFeatureNet(BaseModule):
    """Pillar Feature Net for PillarNeSt."""

    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: Tuple[int, ...] = (64,),
        with_distance: bool = False,
        with_cluster_center: bool = True,
        with_voxel_center: bool = True,
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 4),
        point_cloud_range: Tuple[float, ...] = (0, -40, -3, 70.4, 40, 1),
        norm_cfg: Dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode: str = 'max',
        legacy: bool = True,
        max_num_points: int = 20,
        debug: bool = False,
        debug_max_print: int = 50,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert len(feat_channels) > 0

        self.legacy = legacy
        self.max_num_points = max_num_points
        self.debug = debug
        self.debug_max_print = int(debug_max_print)
        self._debug_count = 0

        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center

        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels

        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            last_layer = i == len(feat_channels) - 2
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

        self._dbg(
            f'init: in_channels={self.in_channels}, feat_channels={feat_channels}, '
            f'with_cluster_center={self._with_cluster_center}, with_voxel_center={self._with_voxel_center}, '
            f'with_distance={self._with_distance}, legacy={self.legacy}')

    def _dbg(self, msg: str) -> None:
        if self.debug and self._debug_count < self.debug_max_print:
            print(f'[{self.__class__.__name__}] {msg}', flush=True)
            self._debug_count += 1

    def forward(
        self,
        features: Tensor,
        num_points: Tensor,
        coors: Tensor,
        img_feats: Optional[Sequence[Tensor]] = None,
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Tensor:
        self._dbg(
            f'forward input: features={tuple(features.shape)}, num_points={tuple(num_points.shape)}, '
            f'coors={tuple(coors.shape)}, dtype={features.dtype}, device={features.device}')
        if num_points.numel() > 0:
            self._dbg(
                f'num_points stats: min={int(num_points.min().item())}, '
                f'max={int(num_points.max().item())}, mean={float(num_points.float().mean().item()):.3f}')

        features_ls = [features]

        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)
            self._dbg(f'f_cluster={tuple(f_cluster.shape)}')

        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :2])
                f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
            else:
                f_center = features[:, :, :2].clone()
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx + self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy + self.y_offset)
            features_ls.append(f_center)
            self._dbg(f'f_center(xy)={tuple(f_center.shape)}')

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
            self._dbg(f'points_dist={tuple(points_dist.shape)}')

        features = torch.cat(features_ls, dim=-1)
        self._dbg(f'concat features={tuple(features.shape)}')

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features = features * mask
        self._dbg(f'mask={tuple(mask.shape)}, valid_ratio={float(mask.mean().item()):.4f}')

        for i, pfn in enumerate(self.pfn_layers):
            features = pfn(features, num_points)
            self._dbg(f'after pfn[{i}] -> {tuple(features.shape)}')

        out = features.squeeze(1)
        self._dbg(f'forward output={tuple(out.shape)}')
        return out


@MODELS.register_module()
class PillarNestSEFeatureNet(PillarNestFeatureNet):
    """Pillar Feature Net with SE PFN layers."""

    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: Tuple[int, ...] = (64,),
        with_distance: bool = False,
        with_cluster_center: bool = True,
        with_voxel_center: bool = True,
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 4),
        point_cloud_range: Tuple[float, ...] = (0, -40, -3, 70.4, 40, 1),
        norm_cfg: Dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode: str = 'max',
        legacy: bool = True,
        max_num_points: int = 20,
        debug: bool = False,
        debug_max_print: int = 50,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            with_distance=with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy,
            max_num_points=max_num_points,
            debug=debug,
            debug_max_print=debug_max_print,
            init_cfg=init_cfg)

        feat_channels = [self.in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            last_layer = i == len(feat_channels) - 2
            pfn_layers.append(
                SEPFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self._dbg('replaced PFN layers with SEPFNLayer')


@MODELS.register_module()
class PillarNestHeightFeatureNet(PillarNestFeatureNet):
    """Height-aware Pillar Feature Net with z-center offsets."""

    def __init__(
        self,
        in_channels: int = 5,
        feat_channels: Tuple[int, ...] = (64,),
        with_distance: bool = False,
        with_cluster_center: bool = True,
        with_voxel_center: bool = True,
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 4),
        point_cloud_range: Tuple[float, ...] = (0, -40, -3, 70.4, 40, 1),
        norm_cfg: Dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode: str = 'max',
        encoder_layer: str = 'PFNLayer',
        legacy: bool = True,
        max_num_points: int = 20,
        debug: bool = False,
        debug_max_print: int = 50,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            with_distance=with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy,
            max_num_points=max_num_points,
            debug=debug,
            debug_max_print=debug_max_print,
            init_cfg=init_cfg)

        in_channels_new = in_channels
        if with_cluster_center:
            in_channels_new += 3
        if with_voxel_center:
            in_channels_new += 3
        if with_distance:
            in_channels_new += 1
        feat_channels = [in_channels_new] + list(feat_channels)

        if encoder_layer == 'PFNLayer':
            layer_cls = PFNLayer
        elif encoder_layer == 'SEPFNLayer':
            layer_cls = SEPFNLayer
        elif encoder_layer == 'SEPFNLayerV2':
            layer_cls = partial(SEPFNLayerV2, max_num_points=self.max_num_points)
        else:
            raise ValueError(f'Unknown encoder_layer: {encoder_layer}')

        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            last_layer = i == len(feat_channels) - 2
            pfn_layers.append(
                layer_cls(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.vz = voxel_size[2]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

        self._dbg(
            f'height init: encoder_layer={encoder_layer}, in_channels_new={in_channels_new}, '
            f'z_offset={self.z_offset:.4f}')

    def forward(
        self,
        features: Tensor,
        num_points: Tensor,
        coors: Tensor,
        img_feats: Optional[Sequence[Tensor]] = None,
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Tensor:
        self._dbg(
            f'height forward input: features={tuple(features.shape)}, num_points={tuple(num_points.shape)}, '
            f'coors={tuple(coors.shape)}')

        features_ls = [features]

        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)
            self._dbg(f'f_cluster={tuple(f_cluster.shape)}')

        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - self.z_offset
            else:
                f_center = features[:, :, :3].clone()
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx + self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy + self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - self.z_offset
            features_ls.append(f_center)
            self._dbg(f'f_center(xyz)={tuple(f_center.shape)}')

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
            self._dbg(f'points_dist={tuple(points_dist.shape)}')

        features = torch.cat(features_ls, dim=-1)
        self._dbg(f'concat features={tuple(features.shape)}')

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features = features * mask
        self._dbg(f'mask={tuple(mask.shape)}, valid_ratio={float(mask.mean().item()):.4f}')

        for i, pfn in enumerate(self.pfn_layers):
            features = pfn(features, num_points)
            self._dbg(f'after pfn[{i}] -> {tuple(features.shape)}')

        out = features.squeeze(1)
        self._dbg(f'height forward output={tuple(out.shape)}')
        return out
