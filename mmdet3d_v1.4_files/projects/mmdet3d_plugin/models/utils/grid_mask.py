# =============================================================================
# grid_mask.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - auto_fp16, force_fp32 from mmcv.runner → REMOVED
#     (AmpOptimWrapper / torch.cuda.amp handles mixed precision in mmdet3d 1.1+)
#   - Added debug logging
# =============================================================================
import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)
logger.info('[grid_mask] Loading module...')


# ===================================================================
# Grid — callable augmentation (used standalone, e.g. in datasets)
# ===================================================================
class Grid(object):
    """Grid masking augmentation (callable version).

    Creates a grid pattern mask and applies it to the input image tensor.
    """

    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5,
                 mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

        logger.debug(f'[Grid] Built: use_h={use_h}, use_w={use_w}, '
                     f'rotate={rotate}, offset={offset}, '
                     f'ratio={ratio}, mode={mode}, prob={prob}')

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label

        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)

        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0

        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(
                2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img, label


# ===================================================================
# GridMask — nn.Module version (used in model forward)
# ===================================================================
class GridMask(nn.Module):
    """Grid masking augmentation as nn.Module.

    Applied during training to input image tensors. Creates a random
    grid pattern mask to regularize the model.
    """

    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5,
                 mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

        logger.debug(f'[GridMask] Built: use_h={use_h}, use_w={use_w}, '
                     f'rotate={rotate}, offset={offset}, '
                     f'ratio={ratio}, mode={mode}, prob={prob}')

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    # NOTE: @auto_fp16() REMOVED — mixed precision handled by
    # AmpOptimWrapper / torch.cuda.amp.autocast in mmdet3d 1.1+
    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x

        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0

        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).to(x.dtype).cuda()
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)

        if self.offset:
            offset = torch.from_numpy(
                2 * (np.random.rand(h, w) - 0.5)).to(x.dtype).cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        logger.debug(f'[GridMask.forward] applied: '
                     f'input=({n},{c},{h},{w}), d={d}, l={self.l}')

        return x.view(n, c, h, w)


logger.info('[grid_mask] ✓ Module fully loaded')