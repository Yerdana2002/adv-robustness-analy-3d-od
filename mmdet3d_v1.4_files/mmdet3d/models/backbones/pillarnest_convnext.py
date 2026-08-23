# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for PillarNeSt from mmdet3d 0.18 to 1.x
from functools import partial
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule, ModuleList, Sequential

from mmdet3d.registry import MODELS


@MODELS.register_module(name='LN2d')
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images."""

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, (
            'LayerNorm2d only supports inputs with shape (N, C, H, W), '
            f'but got tensor with shape {x.shape}'
        )
        return F.layer_norm(
            x.permute(0, 2, 3, 1).contiguous(),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps).permute(0, 3, 1, 2).contiguous()


class GRN(nn.Module):
    """Global Response Normalization Module."""

    def __init__(self, in_channels: int, eps: float = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return (
            self.gamma.view(1, -1, 1, 1) * (x * nx)
            + self.beta.view(1, -1, 1, 1)
            + x
        )


def build_activation_layer(act_cfg: Dict) -> nn.Module:
    """Build activation layer from config."""
    act_type = act_cfg.get('type', 'GELU')
    if act_type == 'GELU':
        return nn.GELU()
    if act_type == 'ReLU':
        return nn.ReLU(inplace=True)
    if act_type == 'SiLU':
        return nn.SiLU(inplace=True)
    raise ValueError(f'Unknown activation type: {act_type}')


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block."""

    def __init__(self,
                 in_channels: int,
                 norm_cfg: Dict = dict(type='LN2d', eps=1e-6),
                 act_cfg: Dict = dict(type='GELU'),
                 mlp_ratio: float = 4.,
                 linear_pw_conv: bool = True,
                 drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6,
                 use_grn: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.mlp_ratio = mlp_ratio
        self.linear_pw_conv = linear_pw_conv
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.use_grn = use_grn

        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        pw_conv = nn.Linear if self.linear_pw_conv else partial(nn.Conv2d, kernel_size=1)

        # For PillarNeSt this is intentionally LN2d
        self.norm = LayerNorm2d(in_channels)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        self.grn = GRN(mid_channels) if use_grn else None
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(in_channels), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.depthwise_conv(x)
        x = self.norm(x)

        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1).contiguous()

        x = self.pointwise_conv1(x)
        x = self.act(x)

        if self.grn is not None:
            if self.linear_pw_conv:
                x = x.permute(0, 3, 1, 2).contiguous()
            x = self.grn(x)
            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1).contiguous()

        x = self.pointwise_conv2(x)

        if self.linear_pw_conv:
            x = x.permute(0, 3, 1, 2).contiguous()

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        return shortcut + self.drop_path(x)


class ConvNeXtBlockLarge(ConvNeXtBlock):
    """ConvNeXtBlock with larger kernel size for PillarNeSt."""

    def __init__(self,
                 in_channels: int,
                 norm_cfg: Dict = dict(type='LN2d', eps=1e-6),
                 act_cfg: Dict = dict(type='GELU'),
                 kernel_size: int = 9,
                 padding: int = 4,
                 mlp_ratio: float = 4.,
                 linear_pw_conv: bool = True,
                 drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6):
        super().__init__(
            in_channels=in_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            mlp_ratio=mlp_ratio,
            linear_pw_conv=linear_pw_conv,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value)
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels)


@MODELS.register_module()
class PillarNestConvNeXt(BaseModule):
    """ConvNeXt backbone adapted for PillarNeSt."""

    arch_settings = {
        'tiny': {'depths': [2, 2, 1, 1, 1], 'channels': [48, 96, 96, 96, 96]},
        'small': {'depths': [3, 3, 2, 1, 1], 'channels': [48, 192, 192, 192, 192]},
        'base': {'depths': [4, 4, 2, 2, 1], 'channels': [64, 192, 384, 384, 384]},
        'large': {'depths': [6, 6, 4, 2, 1], 'channels': [96, 192, 384, 384, 384]},
    }

    def __init__(self,
                 arch: Union[str, Dict] = 'tiny',
                 in_channels: int = 3,
                 stem_patch_size: int = 4,
                 norm_cfg: Dict = dict(type='LN2d', eps=1e-6),
                 act_cfg: Dict = dict(type='GELU'),
                 linear_pw_conv: bool = True,
                 drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6,
                 out_indices: Union[int, Sequence[int]] = -1,
                 frozen_stages: int = 0,
                 gap_before_final_norm: bool = True,
                 first_downsample: int = 1,
                 large_arch: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None,
                 debug: bool = False,
                 debug_max_print: int = 50):
        super().__init__(init_cfg=init_cfg)

        self.debug = debug
        self.debug_max_print = debug_max_print
        self._debug_count = 0

        self.first_downsample = first_downsample
        self.stem_patch_size = stem_patch_size  # kept for config compatibility

        if isinstance(arch, str):
            assert arch in self.arch_settings, (
                f'Unavailable arch: {arch}, choose from {set(self.arch_settings)} or pass a dict.'
            )
            arch = self.arch_settings[arch]
        else:
            assert 'depths' in arch and 'channels' in arch, (
                f'arch dict must contain "depths" and "channels", got keys={list(arch.keys())}'
            )

        self.depths = list(arch['depths'])
        self.channels = list(arch['channels'])
        assert len(self.depths) == len(self.channels), (
            f'depths and channels must have same length, got {len(self.depths)} vs {len(self.channels)}'
        )
        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        out_indices = list(out_indices)
        for i, idx in enumerate(out_indices):
            if idx < 0:
                out_indices[i] = self.num_stages + idx
        for idx in out_indices:
            assert 0 <= idx < self.num_stages, f'Invalid out index {idx}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        self._dbg(
            f'Init arch={arch}, in_channels={in_channels}, depths={self.depths}, '
            f'channels={self.channels}, out_indices={self.out_indices}, '
            f'first_downsample={self.first_downsample}, gap_before_final_norm={self.gap_before_final_norm}',
            force=True
        )

        # stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        block_idx = 0

        # Downsample layers
        if self.first_downsample == 0:
            self.downsample_layers = ModuleList()
        else:
            self.downsample_layers = ModuleList([None])

        # kept from your original implementation
        self.bias = nn.Parameter(torch.randn(3))

        self.stages = ModuleList()
        if large_arch is not None:
            large_stages = large_arch.get('stages', [])
            large_kernel_sizes = large_arch.get('large_kernel_sizes', [])
            large_kernel_paddings = [k // 2 for k in large_kernel_sizes]
        else:
            large_stages = []
            large_kernel_sizes = []
            large_kernel_paddings = []

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = in_channels if i == 0 else self.channels[i]

            if i >= self.first_downsample:
                if self.first_downsample == 0 and i == 0:
                    downsample_layer = nn.Sequential(
                        LayerNorm2d(in_channels),
                        nn.Conv2d(in_channels, channels, kernel_size=2, stride=2),
                    )
                else:
                    prev_channels = self.channels[i - 1]
                    downsample_layer = nn.Sequential(
                        LayerNorm2d(prev_channels),
                        nn.Conv2d(prev_channels, channels, kernel_size=2, stride=2),
                    )
                self.downsample_layers.append(downsample_layer)

            if large_arch is not None and i in large_stages:
                stage_idx = large_stages.index(i)
                ksize = large_kernel_sizes[stage_idx]
                kpad = large_kernel_paddings[stage_idx]
                stage = Sequential(*[
                    ConvNeXtBlockLarge(
                        in_channels=channels,
                        kernel_size=ksize,
                        padding=kpad,
                        drop_path_rate=dpr[block_idx + j],
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        linear_pw_conv=linear_pw_conv,
                        layer_scale_init_value=layer_scale_init_value)
                    for j in range(depth)
                ])
            else:
                stage = Sequential(*[
                    ConvNeXtBlock(
                        in_channels=channels,
                        drop_path_rate=dpr[block_idx + j],
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        linear_pw_conv=linear_pw_conv,
                        layer_scale_init_value=layer_scale_init_value)
                    for j in range(depth)
                ])

            block_idx += depth
            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)[1]
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def _dbg(self, msg: str, force: bool = False) -> None:
        if self.debug and (force or self._debug_count < self.debug_max_print):
            print(f'[PillarNestConvNeXt] {msg}', flush=True)
            self._debug_count += 1

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            x: [B, C, H, W]

        Returns:
            tuple: Features from out_indices.
        """
        self._dbg(f'forward input shape={tuple(x.shape)}')

        outs = []
        for i, stage in enumerate(self.stages):
            if i >= self.first_downsample:
                x = self.downsample_layers[i](x)
                self._dbg(f'stage{i} after downsample shape={tuple(x.shape)}')

            x = stage(x)
            self._dbg(f'stage{i} output shape={tuple(x.shape)}')

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    out = norm_layer(gap).flatten(1)
                    self._dbg(f'stage{i} out(norm+gap) shape={tuple(out.shape)}')
                    outs.append(out)
                else:
                    out = norm_layer(x).contiguous()
                    self._dbg(f'stage{i} out(norm) shape={tuple(out.shape)}')
                    outs.append(out)

        self._dbg(f'forward returned {len(outs)} tensors')
        return tuple(outs)

    def _freeze_stages(self):
        """Freeze stages for finetuning."""
        if self.frozen_stages <= 0:
            return

        for i in range(self.frozen_stages):
            if i < len(self.downsample_layers) and self.downsample_layers[i] is not None:
                self.downsample_layers[i].eval()
                for param in self.downsample_layers[i].parameters():
                    param.requires_grad = False

            self.stages[i].eval()
            for param in self.stages[i].parameters():
                param.requires_grad = False

        self._dbg(f'froze first {self.frozen_stages} stages', force=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        self._dbg(f'train(mode={mode}) called')
