# =============================================================================
# transformer_utils.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
#     from mmcv.cnn.bricks.registry       → still available in mmcv 2.x
#   - TRANSFORMER from mmdet.models.utils.builder → MODELS from mmdet3d.registry
#     (TRANSFORMER registry was merged into MODELS)
#   - BaseModule from mmcv.runner.base_module → mmengine.model
#   - xavier_init from mmcv.cnn            → mmengine.model
#   - build_conv_layer, build_norm_layer, build_activation_layer
#     from mmcv.cnn                        → still in mmcv.cnn (unchanged)
#   - to_2tuple from mmcv.utils            → mmcv.utils (still works) or
#     mmengine.utils if mmcv.utils removed
#   - BaseTransformerLayer, TransformerLayerSequence,
#     build_transformer_layer_sequence     → still in mmcv.cnn.bricks.transformer
#   - MultiScaleDeformableAttention        → mmcv.ops (unchanged)
# =============================================================================
import logging
import math
import warnings
from typing import Sequence
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

logger = logging.getLogger(__name__)
logger.info('[transformer_utils] Loading module...')

# --- mmcv.cnn (unchanged in mmcv 2.x) ---
try:
    from mmcv.cnn import (build_activation_layer, build_conv_layer,
                          build_norm_layer)
    logger.info('[transformer_utils] ✓ Imported build_*_layer from mmcv.cnn')
except ImportError as e:
    logger.error(f'[transformer_utils] ✗ Failed mmcv.cnn imports: {e}')
    raise

# --- xavier_init: moved from mmcv.cnn to mmengine.model in mmcv 2.x ---
try:
    from mmengine.model import xavier_init
    logger.info('[transformer_utils] ✓ Imported xavier_init '
                'from mmengine.model')
except ImportError:
    try:
        from mmcv.cnn import xavier_init
        logger.info('[transformer_utils] ✓ Imported xavier_init '
                    'from mmcv.cnn (fallback)')
    except ImportError as e:
        logger.error(f'[transformer_utils] ✗ Failed to import '
                     f'xavier_init: {e}')
        raise

# --- BaseModule: moved from mmcv.runner to mmengine.model ---
try:
    from mmengine.model import BaseModule
    logger.info('[transformer_utils] ✓ Imported BaseModule '
                'from mmengine.model')
except ImportError as e:
    logger.error(f'[transformer_utils] ✗ Failed to import BaseModule: {e}')
    logger.error('  → Moved from mmcv.runner.base_module to mmengine.model')
    raise

# --- Transformer layer registries (still in mmcv 2.x) ---
try:
    from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                          TRANSFORMER_LAYER_SEQUENCE)
    logger.info('[transformer_utils] ✓ Imported TRANSFORMER_LAYER, '
                'TRANSFORMER_LAYER_SEQUENCE from mmcv.cnn.bricks.registry')
except ImportError as e:
    logger.error(f'[transformer_utils] ✗ Failed transformer registry '
                 f'imports: {e}')
    logger.error('  → These should still exist in mmcv 2.x. '
                 'Check mmcv version.')
    raise

# --- Transformer base classes (still in mmcv 2.x) ---
try:
    from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                             TransformerLayerSequence,
                                             build_transformer_layer_sequence)
    logger.info('[transformer_utils] ✓ Imported BaseTransformerLayer, '
                'TransformerLayerSequence, '
                'build_transformer_layer_sequence')
except ImportError as e:
    logger.error(f'[transformer_utils] ✗ Failed transformer base '
                 f'class imports: {e}')
    raise

# --- to_2tuple ---
try:
    from mmcv.utils import to_2tuple
    logger.info('[transformer_utils] ✓ Imported to_2tuple from mmcv.utils')
except ImportError:
    # mmcv 2.x may have moved this
    try:
        from mmengine.utils import to_2tuple
        logger.info('[transformer_utils] ✓ Imported to_2tuple '
                    'from mmengine.utils (fallback)')
    except ImportError:
        # Last resort: define it ourselves
        def to_2tuple(x):
            if isinstance(x, (list, tuple)):
                return tuple(x)
            return (x, x)
        logger.warning('[transformer_utils] ⚠ to_2tuple not found, '
                       'using local fallback')

# --- TRANSFORMER registry → MODELS (unified in mmdet3d 1.1+) ---
# The old TRANSFORMER registry from mmdet.models.utils.builder was
# merged into MODELS. We import MODELS but alias it so that existing
# @TRANSFORMER.register_module() decorators elsewhere still work.
try:
    from mmdet3d.registry import MODELS
    # Alias for backward compat if other code references TRANSFORMER
    TRANSFORMER = MODELS
    logger.info('[transformer_utils] ✓ Imported MODELS from mmdet3d.registry '
                '(aliased as TRANSFORMER)')
except ImportError as e:
    logger.error(f'[transformer_utils] ✗ Failed to import MODELS: {e}')
    raise

# --- MultiScaleDeformableAttention ---
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
    logger.info('[transformer_utils] ✓ Imported '
                'MultiScaleDeformableAttention from mmcv.ops')
except ImportError:
    try:
        from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
        logger.warning('[transformer_utils] ⚠ Imported '
                       'MultiScaleDeformableAttention from '
                       'mmcv.cnn.bricks.transformer (deprecated path)')
    except ImportError as e:
        logger.error(f'[transformer_utils] ✗ Failed to import '
                     f'MultiScaleDeformableAttention: {e}')
        logger.error('  → Make sure mmcv is compiled with CUDA ops')
        raise


# ===================================================================
# Utility functions
# ===================================================================
def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor."""
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor."""
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


# ===================================================================
# AdaptivePadding
# ===================================================================
class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully
    covered by filter you specified."""

    def __init__(self, kernel_size=1, stride=1, dilation=1,
                 padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        logger.debug(f'[AdaptivePadding] kernel_size={kernel_size}, '
                     f'stride={stride}')

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2
                ])
        return x


# ===================================================================
# PatchEmbed
# ===================================================================
class PatchEmbed(BaseModule):
    """Image to Patch Embedding via conv layer."""

    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        conv_type='Conv2d',
        kernel_size=16,
        stride=16,
        padding='corner',
        dilation=1,
        bias=True,
        norm_cfg=None,
        input_size=None,
        init_cfg=None,
    ):
        super(PatchEmbed, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

        logger.info(f'[PatchEmbed] ✓ Initialized: in_ch={in_channels}, '
                    f'embed_dims={embed_dims}, kernel={kernel_size}, '
                    f'stride={stride}, '
                    f'init_out_size={self.init_out_size}')

    def forward(self, x):
        logger.debug(f'[PatchEmbed.forward] input={x.shape}')

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)

        logger.debug(f'[PatchEmbed.forward] output={x.shape}, '
                     f'out_size={out_size}')
        return x, out_size


# ===================================================================
# PatchMerging
# ===================================================================
class PatchMerging(BaseModule):
    """Merge patch feature map."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            padding = 0
        else:
            self.adap_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)
        logger.info(f'[PatchMerging] ✓ Initialized: '
                    f'in_ch={in_channels}→out_ch={out_channels}, '
                    f'kernel={kernel_size}, stride={stride}')

    def forward(self, x, input_size):
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), \
            f'Expect input_size is Sequence but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'
        logger.debug(f'[PatchMerging.forward] input: B={B}, '
                     f'L={L}, C={C}, H={H}, W={W}')

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)

        out_h = (H + 2 * self.sampler.padding[0] -
                 self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] -
                 self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)

        logger.debug(f'[PatchMerging.forward] output: {x.shape}, '
                     f'out_size={output_size}')
        return x, output_size


# ===================================================================
# inverse_sigmoid
# ===================================================================
def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# ===================================================================
# Checkpoint converters (pure logic, no API changes needed)
# ===================================================================
def pvt_convert(ckpt):
    """Convert PVT checkpoint keys."""
    logger.info(f'[pvt_convert] Converting checkpoint with '
                f'{len(ckpt)} keys')
    new_ckpt = OrderedDict()
    use_abs_pos_embed = False
    use_conv_ffn = False
    for k in ckpt.keys():
        if k.startswith('pos_embed'):
            use_abs_pos_embed = True
        if k.find('dwconv') >= 0:
            use_conv_ffn = True
    logger.debug(f'[pvt_convert] use_abs_pos_embed={use_abs_pos_embed}, '
                 f'use_conv_ffn={use_conv_ffn}')

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        if k.startswith('norm.'):
            continue
        if k.startswith('cls_token'):
            continue
        if k.startswith('pos_embed'):
            stage_i = int(k.replace('pos_embed', ''))
            new_k = k.replace(f'pos_embed{stage_i}',
                              f'layers.{stage_i - 1}.1.0.pos_embed')
            if stage_i == 4 and v.size(1) == 50:
                new_v = v[:, 1:, :]
            else:
                new_v = v
        elif k.startswith('patch_embed'):
            stage_i = int(k.split('.')[0].replace('patch_embed', ''))
            new_k = k.replace(f'patch_embed{stage_i}',
                              f'layers.{stage_i - 1}.0')
            new_v = v
            if 'proj.' in new_k:
                new_k = new_k.replace('proj.', 'projection.')
        elif k.startswith('block'):
            stage_i = int(k.split('.')[0].replace('block', ''))
            layer_i = int(k.split('.')[1])
            new_layer_i = layer_i + use_abs_pos_embed
            new_k = k.replace(f'block{stage_i}.{layer_i}',
                              f'layers.{stage_i - 1}.1.{new_layer_i}')
            new_v = v
            if 'attn.q.' in new_k:
                sub_item_k = k.replace('q.', 'kv.')
                new_k = new_k.replace('q.', 'attn.in_proj_')
                new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
            elif 'attn.kv.' in new_k:
                continue
            elif 'attn.proj.' in new_k:
                new_k = new_k.replace('proj.', 'attn.out_proj.')
            elif 'attn.sr.' in new_k:
                new_k = new_k.replace('sr.', 'sr.')
            elif 'mlp.' in new_k:
                new_k = new_k.replace('mlp.', 'ffn.layers.')
                if 'fc1.weight' in new_k or 'fc2.weight' in new_k:
                    new_v = v.reshape((*v.shape, 1, 1))
                new_k = new_k.replace('fc1.', '0.')
                new_k = new_k.replace('dwconv.dwconv.', '1.')
                if use_conv_ffn:
                    new_k = new_k.replace('fc2.', '4.')
                else:
                    new_k = new_k.replace('fc2.', '3.')
        elif k.startswith('norm'):
            stage_i = int(k[4])
            new_k = k.replace(f'norm{stage_i}',
                              f'layers.{stage_i - 1}.2')
            new_v = v
        else:
            new_k = k
            new_v = v
        new_ckpt[new_k] = new_v

    logger.info(f'[pvt_convert] Converted: {len(new_ckpt)} keys')
    return new_ckpt


def swin_converter(ckpt):
    """Convert Swin Transformer checkpoint keys."""
    logger.info(f'[swin_converter] Converting checkpoint with '
                f'{len(ckpt)} keys')
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(
            1, 2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt['backbone.' + new_k] = new_v

    logger.info(f'[swin_converter] Converted: {len(new_ckpt)} keys')
    return new_ckpt


logger.info('[transformer_utils] ✓ Module fully loaded')