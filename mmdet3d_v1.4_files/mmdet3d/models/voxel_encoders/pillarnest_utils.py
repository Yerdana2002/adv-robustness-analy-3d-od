# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for PillarNeSt from mmdet3d 0.18 to 1.x
from typing import Dict, Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule


def get_paddings_indicator(actual_num: Tensor, 
                           max_num: int, 
                           axis: int = 0) -> Tensor:
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num: Actual number of points in each voxel.
        max_num: Maximum number of points in a voxel.
        axis: Axis to create mask along.

    Returns:
        Boolean mask tensor.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device)
    max_num = max_num.view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class PFNLayer(BaseModule):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers.
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_cfg: Dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
        last_layer: bool = False,
        mode: str = 'max'
    ) -> None:
        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = build_norm_layer(norm_cfg, self.units)[1]
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        assert mode in ['max', 'avg', 'maxavg']
        self.mode = mode

    def forward(
        self,
        inputs: Tensor,
        num_voxels: Optional[Tensor] = None,
        aligned_distance: Optional[Tensor] = None
    ) -> Tensor:
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)  # [N, M, C_in] --> [N, M, C_out]
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        # Pooling
        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]  # [N, 1, C]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(dim=1, keepdim=True) / num_voxels.type_as(inputs).view(-1, 1, 1)
        elif self.mode == 'maxavg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            x_avg = x.sum(dim=1, keepdim=True) / num_voxels.type_as(inputs).view(-1, 1, 1)
            x_max = (x_max + x_avg) / 2.

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class ChannelAttention(BaseModule):
    """Channel Attention module for SE block.
    
    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for FC layers.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(1, in_channels // reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, in_channels // reduction_ratio), in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.
        
        Args:
            x: Input tensor of shape (N, M, C).
            
        Returns:
            Scaled feature tensor.
        """
        N, _, channels = x.size()

        # Squeeze: global average pooling
        squeeze = self.avg_pool(x.permute(0, 2, 1)).view(N, channels)

        # Excitation: FC layers
        excitation = self.fc(squeeze).view(N, channels, 1)

        # Scale: element-wise multiplication
        scaled_feature = x * excitation.permute(0, 2, 1)

        return scaled_feature


class ChannelAttentionV2(BaseModule):
    """Channel Attention V2 module - attention along M dimension.
    
    Args:
        in_channels (int): Number of points per voxel (M dimension).
        reduction_ratio (int): Reduction ratio for FC layers.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(1, in_channels // reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, in_channels // reduction_ratio), in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.
        
        Args:
            x: Input tensor of shape (N, M, C).
            
        Returns:
            Scaled feature tensor.
        """
        N, M, channels = x.size()

        # Squeeze: global average pooling
        squeeze = self.avg_pool(x).view(N, M)

        # Excitation: FC layers
        excitation = self.fc(squeeze).view(N, M, 1)

        # Scale: element-wise multiplication
        scaled_feature = x * excitation

        return scaled_feature


class SEPFNLayer(BaseModule):
    """Pillar Feature Net Layer with SE Block.
    
    Channel attention for height information.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers.
        last_layer (bool): If last_layer, there is no concatenation.
        mode (str): Pooling mode ('max', 'avg', 'maxavg').
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_cfg: Dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
        last_layer: bool = False,
        mode: str = 'max'
    ) -> None:
        super().__init__()
        self.name = 'SEPFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = build_norm_layer(norm_cfg, self.units)[1]
        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.channel_attention = ChannelAttention(in_channels=self.units)

        assert mode in ['max', 'avg', 'maxavg']
        self.mode = mode

    def forward(
        self,
        inputs: Tensor,
        num_voxels: Optional[Tensor] = None,
        aligned_distance: Optional[Tensor] = None
    ) -> Tensor:
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
            num_voxels (torch.Tensor, optional): Number of points in each voxel.
            aligned_distance (torch.Tensor, optional): Distance to voxel center.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)  # [N, M, C_in] --> [N, M, C_out]
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        # Apply channel attention
        x = self.channel_attention(x)

        # Pooling
        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(dim=1, keepdim=True) / num_voxels.type_as(inputs).view(-1, 1, 1)
        elif self.mode == 'maxavg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            x_avg = x.sum(dim=1, keepdim=True) / num_voxels.type_as(inputs).view(-1, 1, 1)
            x_max = (x_max + x_avg) / 2.

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class SEPFNLayerV2(SEPFNLayer):
    """SE PFN Layer V2 - attention along M dimension instead of C."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_cfg: Dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
        last_layer: bool = False,
        mode: str = 'max',
        max_num_points: int = 20
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            last_layer=last_layer,
            mode=mode,
        )
        self.channel_attention = ChannelAttentionV2(in_channels=max_num_points)