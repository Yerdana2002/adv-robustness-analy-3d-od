# =============================================================================
# swin_transformer.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - BaseModule, ModuleList from mmcv.runner → mmengine.model
#   - _load_checkpoint from mmcv.runner      → load_checkpoint from mmengine.runner
#   - to_2tuple from mmcv.utils              → mmcv.utils (still works) or fallback
#   - trunc_normal_ from mmcv.cnn.utils.weight_init → mmengine.model (or mmcv fallback)
#   - constant_init, trunc_normal_init from mmcv.cnn → mmengine.model (or mmcv fallback)
#   - get_root_logger from mmdet.utils       → mmengine.logging.print_log / logging
#   - BACKBONES from mmdet.models.builder    → MODELS from mmdet3d.registry
#   - build_norm_layer from mmcv.cnn         → still in mmcv.cnn (unchanged)
#   - FFN, build_dropout from mmcv.cnn.bricks.transformer → still works
# =============================================================================
import logging
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

logger = logging.getLogger(__name__)
logger.info('[swin_transformer] Loading module...')

# --- mmcv.cnn (unchanged in mmcv 2.x) ---
try:
    from mmcv.cnn import build_norm_layer
    logger.info('[swin_transformer] ✓ Imported build_norm_layer from mmcv.cnn')
except ImportError as e:
    logger.error(f'[swin_transformer] ✗ build_norm_layer: {e}')
    raise

# --- constant_init, trunc_normal_init: try mmengine first, mmcv fallback ---
try:
    from mmengine.model import constant_init, trunc_normal_init
    logger.info('[swin_transformer] ✓ Imported constant_init, '
                'trunc_normal_init from mmengine.model')
except ImportError:
    from mmcv.cnn import constant_init, trunc_normal_init
    logger.info('[swin_transformer] ✓ Imported constant_init, '
                'trunc_normal_init from mmcv.cnn (fallback)')

# --- FFN, build_dropout (still in mmcv 2.x) ---
try:
    from mmcv.cnn.bricks.transformer import FFN, build_dropout
    logger.info('[swin_transformer] ✓ Imported FFN, build_dropout')
except ImportError as e:
    logger.error(f'[swin_transformer] ✗ FFN/build_dropout: {e}')
    raise

# --- trunc_normal_: try mmengine, then mmcv ---
try:
    from mmengine.model import trunc_normal_
    logger.info('[swin_transformer] ✓ Imported trunc_normal_ '
                'from mmengine.model')
except ImportError:
    try:
        from mmcv.cnn.utils.weight_init import trunc_normal_
        logger.info('[swin_transformer] ✓ Imported trunc_normal_ '
                    'from mmcv.cnn.utils.weight_init (fallback)')
    except ImportError as e:
        logger.error(f'[swin_transformer] ✗ trunc_normal_: {e}')
        raise

# --- BaseModule, ModuleList: mmcv.runner → mmengine.model ---
try:
    from mmengine.model import BaseModule, ModuleList
    logger.info('[swin_transformer] ✓ Imported BaseModule, ModuleList '
                'from mmengine.model')
except ImportError as e:
    logger.error(f'[swin_transformer] ✗ BaseModule/ModuleList: {e}')
    logger.error('  → Moved from mmcv.runner to mmengine.model')
    raise

# --- load_checkpoint: mmcv.runner._load_checkpoint → mmengine.runner ---
try:
    from mmengine.runner import load_checkpoint
    # Alias so existing code calling _load_checkpoint still works
    _load_checkpoint = load_checkpoint
    logger.info('[swin_transformer] ✓ Imported load_checkpoint '
                'from mmengine.runner')
except ImportError:
    try:
        from mmcv.runner import _load_checkpoint
        logger.info('[swin_transformer] ✓ Imported _load_checkpoint '
                    'from mmcv.runner (fallback)')
    except ImportError as e:
        logger.error(f'[swin_transformer] ✗ _load_checkpoint: {e}')
        raise

# --- to_2tuple ---
try:
    from mmcv.utils import to_2tuple
    logger.info('[swin_transformer] ✓ Imported to_2tuple from mmcv.utils')
except ImportError:
    try:
        from mmengine.utils import to_2tuple
        logger.info('[swin_transformer] ✓ Imported to_2tuple '
                    'from mmengine.utils (fallback)')
    except ImportError:
        def to_2tuple(x):
            if isinstance(x, (list, tuple)):
                return tuple(x)
            return (x, x)
        logger.warning('[swin_transformer] ⚠ to_2tuple not found, '
                       'using local fallback')

# --- Registry: BACKBONES → MODELS ---
try:
    from mmdet3d.registry import MODELS
    logger.info('[swin_transformer] ✓ Imported MODELS from mmdet3d.registry')
except ImportError as e:
    logger.error(f'[swin_transformer] ✗ MODELS: {e}')
    raise

# --- swin_utils (local import — already refactored) ---
try:
    from .swin_utils import swin_converter, PatchEmbed, PatchMerging
    logger.info('[swin_transformer] ✓ Imported swin_converter, PatchEmbed, '
                'PatchMerging from .swin_utils')
except ImportError as e:
    logger.error(f'[swin_transformer] ✗ swin_utils import: {e}')
    logger.error('  → Make sure swin_utils.py (transformer_utils.py) '
                 'is refactored and in the same package')
    raise


# ===================================================================
# WindowMSA
# ===================================================================
class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))

        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

        logger.debug(f'[WindowMSA] embed_dims={embed_dims}, '
                     f'num_heads={num_heads}, window_size={window_size}')

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        logger.debug(f'[WindowMSA.forward] input: B={B}, N={N}, C={C}, '
                     f'mask={"None" if mask is None else mask.shape}')

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


# ===================================================================
# ShiftWindowMSA
# ===================================================================
class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)
        logger.debug(f'[ShiftWindowMSA] window_size={window_size}, '
                     f'shift_size={shift_size}')

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            img_mask = torch.zeros((1, H_pad, W_pad, 1),
                                   device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(
                    attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        query_windows = self.window_partition(shifted_query)
        query_windows = query_windows.view(-1, self.window_size**2, C)

        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size,
                         window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size,
                    W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


# ===================================================================
# SwinBlock
# ===================================================================
class SwinBlock(BaseModule):
    """A single Swin Transformer block."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__()
        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):
        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


# ===================================================================
# SwinBlockSequence
# ===================================================================
class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate)
                               for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample
        logger.debug(f'[SwinBlockSequence] depth={depth}, '
                     f'embed_dims={embed_dims}, num_heads={num_heads}, '
                     f'downsample={"yes" if downsample else "no"}')

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


# ===================================================================
# SwinTransformer (main backbone)
# ===================================================================
@MODELS.register_module()
class SwinTransformer(BaseModule):
    """Swin Transformer backbone.

    A PyTorch implementation of `Swin Transformer: Hierarchical Vision
    Transformer using Shifted Windows`.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):

        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                (f'The size of image should have length 1 or 2, '
                 f'but got {len(pretrain_img_size)}')

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SwinTransformer, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        logger.info(f'[SwinTransformer] Building: embed_dims={embed_dims}, '
                    f'depths={depths}, num_heads={num_heads}, '
                    f'window_size={window_size}, '
                    f'frozen_stages={frozen_stages}')

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))
            logger.info(f'[SwinTransformer] Using abs_pos_embed: '
                        f'num_patches={num_patches}')

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i)
                             for i in range(num_layers)]
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        logger.info(f'[SwinTransformer] ✓ Built {num_layers} stages, '
                    f'num_features={self.num_features}')

    def train(self, mode=True):
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()
            logger.debug('[SwinTransformer._freeze_stages] '
                         'Froze patch_embed and pos_embed')

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            logger.debug(f'[SwinTransformer._freeze_stages] '
                         f'Froze stage {i-1}')

    def init_weights(self):
        if self.init_cfg is None:
            logger.warning(f'No pre-trained weights for '
                           f'{self.__class__.__name__}, '
                           f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, \
                (f'Only support specify `Pretrained` in `init_cfg` in '
                 f'{self.__class__.__name__}')

            logger.info(f'[SwinTransformer.init_weights] Loading from '
                        f'{self.init_cfg["checkpoint"]}')

            ckpt = _load_checkpoint(
                self.init_cfg['checkpoint'],
                logger=logger,
                map_location='cpu')

            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
                logger.debug('[SwinTransformer.init_weights] '
                             'Found "state_dict" key in checkpoint')
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
                logger.debug('[SwinTransformer.init_weights] '
                             'Found "model" key in checkpoint')
            else:
                _state_dict = ckpt
                logger.debug('[SwinTransformer.init_weights] '
                             'Using raw checkpoint dict')

            logger.debug(f'[SwinTransformer.init_weights] '
                         f'Raw state_dict has {len(_state_dict)} keys')

            if self.convert_weights:
                logger.info('[SwinTransformer.init_weights] '
                            'Converting weights with swin_converter')
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            if not state_dict:
                state_dict = _state_dict
                logger.debug('[SwinTransformer.init_weights] '
                             'No "backbone." prefix found, using as-is')

            # strip 'module.' prefix
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                logger.debug('[SwinTransformer.init_weights] '
                             'Stripped "module." prefix')

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('[SwinTransformer.init_weights] '
                                   'Error in loading absolute_pos_embed, '
                                   f'ckpt=({N1},{L},{C1}) vs '
                                   f'model=({N2},{C2},{H},{W}), skipping')
                else:
                    state_dict['absolute_pos_embed'] = \
                        absolute_pos_embed.view(
                            N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'[SwinTransformer.init_weights] '
                                   f'Error in loading {table_key}: '
                                   f'nH mismatch {nH1} vs {nH2}, skipping')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    logger.debug(f'[SwinTransformer.init_weights] '
                                 f'Interpolating {table_key}: '
                                 f'{S1}x{S1} → {S2}x{S2}')
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(
                            1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = \
                        table_pretrained_resized.view(
                            nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            missing, unexpected = self.load_state_dict(state_dict, False)
            if missing:
                logger.warning(f'[SwinTransformer.init_weights] '
                               f'Missing keys ({len(missing)}): '
                               f'{missing[:5]}...')
            if unexpected:
                logger.warning(f'[SwinTransformer.init_weights] '
                               f'Unexpected keys ({len(unexpected)}): '
                               f'{unexpected[:5]}...')
            logger.info('[SwinTransformer.init_weights] ✓ Weights loaded')

    def forward(self, x):
        logger.debug(f'[SwinTransformer.forward] input={x.shape}')

        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(
                                   0, 3, 1, 2).contiguous()
                outs.append(out)
                logger.debug(f'[SwinTransformer.forward] stage[{i}] '
                             f'output={out.shape}')

        logger.debug(f'[SwinTransformer.forward] '
                     f'Returning {len(outs)} feature maps')
        return outs


logger.info('[swin_transformer] ✓ Registered SwinTransformer to MODELS')
logger.info('[swin_transformer] ✓ Module fully loaded')