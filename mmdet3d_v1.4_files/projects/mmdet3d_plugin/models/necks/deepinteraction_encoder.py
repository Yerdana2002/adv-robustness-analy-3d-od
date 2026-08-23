# =============================================================================
# deep_interaction_encoder.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - NECKS from mmdet3d.models.builder  → MODELS from mmdet3d.registry
#   - build_conv_layer from mmcv.cnn     → unchanged (still in mmcv.cnn)
#   - import pdb                         → REMOVED
#   - Added debug logging throughout
# =============================================================================
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)
logger.info('[deep_interaction_encoder] Loading module...')

# --- mmcv.cnn (unchanged in mmcv 2.x) ---
try:
    from mmcv.cnn import build_conv_layer
    logger.info('[deep_interaction_encoder] ✓ Imported build_conv_layer '
                'from mmcv.cnn')
except ImportError as e:
    logger.error(f'[deep_interaction_encoder] ✗ mmcv.cnn imports: {e}')
    raise

# --- Registry: NECKS → MODELS ---
try:
    from mmdet3d.registry import MODELS
    logger.info('[deep_interaction_encoder] ✓ Imported MODELS '
                'from mmdet3d.registry')
except ImportError as e:
    logger.error(f'[deep_interaction_encoder] ✗ Registry imports: {e}')
    raise

# --- Local encoder utils (unchanged — these are project-internal) ---
try:
    from projects.mmdet3d_plugin.models.utils.encoder_utils import (
        MMRI_I2P, LocalContextAttentionBlock, ConvBNReLU, MMRI_P2I)
    logger.info('[deep_interaction_encoder] ✓ Imported encoder_utils '
                '(MMRI_I2P, LocalContextAttentionBlock, ConvBNReLU, MMRI_P2I)')
except ImportError as e:
    logger.error(f'[deep_interaction_encoder] ✗ encoder_utils: {e}')
    raise


# ===================================================================
# DeepInteractionEncoderLayer
# ===================================================================
class DeepInteractionEncoderLayer(nn.Module):
    """Single encoder layer for Deep Interaction multi-modal fusion.

    Performs bidirectional cross-modal attention:
      - Image-to-Point (I2P) + Point self-attention → fused point features
      - Point-to-Image (P2I) + Image self-attention → fused image features
    """

    def __init__(self, hidden_channel):
        super(DeepInteractionEncoderLayer, self).__init__()

        self.I2P_block = MMRI_I2P(hidden_channel, hidden_channel, 0.1)
        self.P_IML = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
        self.P_out_proj = ConvBNReLU(
            2 * hidden_channel, hidden_channel,
            kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.P_integration = ConvBNReLU(
            2 * hidden_channel, hidden_channel,
            kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=None)

        self.P2I_block = MMRI_P2I(hidden_channel, hidden_channel, 9)
        self.I_IML = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
        self.I_out_proj = ConvBNReLU(
            2 * hidden_channel, hidden_channel,
            kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.I_integration = ConvBNReLU(
            2 * hidden_channel, hidden_channel,
            kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=None)

        logger.debug(f'[DeepInteractionEncoderLayer] Built: '
                     f'hidden_channel={hidden_channel}')

    def forward(self, img_feat, lidar_feat, img_metas, pts_metas):
        """Forward pass for a single encoder layer.

        Args:
            img_feat (torch.Tensor): Image features [BN, C, H, W].
            lidar_feat (torch.Tensor): LiDAR BEV features [B, C, H', W'].
            img_metas (list[dict]): Image metadata.
            pts_metas (dict): Point cloud metadata.

        Returns:
            tuple: (new_img_feat, new_lidar_feat)
        """
        batch_size = lidar_feat.shape[0]
        BN, I_C, I_H, I_W = img_feat.shape

        # --- Image-to-Point path ---
        I2P_feat = self.I2P_block(
            lidar_feat,
            img_feat.view(batch_size, -1, I_C, I_H, I_W),
            img_metas, pts_metas)
        P2P_feat = self.P_IML(lidar_feat, lidar_feat)
        P_Aug_feat = self.P_out_proj(
            torch.cat((I2P_feat, P2P_feat), dim=1))
        new_lidar_feat = self.P_integration(
            torch.cat((P_Aug_feat, lidar_feat), dim=1))

        # --- Point-to-Image path ---
        P2I_feat = self.P2I_block(
            lidar_feat,
            img_feat.view(batch_size, -1, I_C, I_H, I_W),
            img_metas, pts_metas)
        I2I_feat = self.I_IML(img_feat, img_feat)
        I_Aug_feat = self.I_out_proj(
            torch.cat((P2I_feat.view(BN, -1, I_H, I_W), I2I_feat), dim=1))
        new_img_feat = self.I_integration(
            torch.cat((I_Aug_feat, img_feat), dim=1))

        logger.debug(f'[DeepInteractionEncoderLayer.forward] '
                     f'lidar_feat={lidar_feat.shape}, '
                     f'img_feat={img_feat.shape} → '
                     f'new_lidar={new_lidar_feat.shape}, '
                     f'new_img={new_img_feat.shape}')

        return new_img_feat, new_lidar_feat


# ===================================================================
# DeepInteractionEncoder (Neck)
# ===================================================================
@MODELS.register_module()
class DeepInteractionEncoder(nn.Module):
    """Deep Interaction Encoder neck for multi-modal fusion.

    Applies shared convolutions followed by N layers of bidirectional
    cross-modal attention between image and point cloud features.
    """

    def __init__(self,
                 num_layers=2,
                 in_channels_img=64,
                 in_channels_pts=128 * 3,
                 hidden_channel=128,
                 bn_momentum=0.1,
                 bias='auto',
                 ):
        super(DeepInteractionEncoder, self).__init__()

        logger.info(f'[DeepInteractionEncoder] Building: '
                    f'num_layers={num_layers}, '
                    f'in_channels_img={in_channels_img}, '
                    f'in_channels_pts={in_channels_pts}, '
                    f'hidden_channel={hidden_channel}')

        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        self.num_layers = num_layers
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.fusion_blocks.append(
                DeepInteractionEncoderLayer(hidden_channel))
            logger.debug(f'[DeepInteractionEncoder] Built fusion_block[{i}]')

        self.bn_momentum = bn_momentum
        self.init_weights()

        logger.info('[DeepInteractionEncoder] ✓ Built successfully')

    def init_weights(self):
        self.init_bn_momentum()
        logger.debug('[DeepInteractionEncoder] init_weights complete')

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, img_feats, pts_feats, img_metas, pts_metas):
        """Forward pass.

        Args:
            img_feats (torch.Tensor): Image features.
            pts_feats (torch.Tensor): Point cloud BEV features.
            img_metas (list[dict]): Image metadata.
            pts_metas (dict): Point cloud metadata.

        Returns:
            tuple: (new_img_feat, [pts_feat_conv, new_pts_feat])
        """
        new_img_feat = self.shared_conv_img(img_feats)
        new_pts_feat = self.shared_conv_pts(pts_feats)
        pts_feat_conv = new_pts_feat.clone()

        logger.debug(f'[DeepInteractionEncoder.forward] '
                     f'img_feats={img_feats.shape}, '
                     f'pts_feats={pts_feats.shape}, '
                     f'after shared_conv: img={new_img_feat.shape}, '
                     f'pts={new_pts_feat.shape}')

        for i in range(self.num_layers):
            new_img_feat, new_pts_feat = self.fusion_blocks[i](
                new_img_feat, new_pts_feat, img_metas, pts_metas)

        logger.debug(f'[DeepInteractionEncoder.forward] '
                     f'output: img={new_img_feat.shape}, '
                     f'pts=[{pts_feat_conv.shape}, {new_pts_feat.shape}]')

        return new_img_feat, [pts_feat_conv, new_pts_feat]


logger.info('[deep_interaction_encoder] ✓ Registered '
            'DeepInteractionEncoder to MODELS')
logger.info('[deep_interaction_encoder] ✓ Module fully loaded')