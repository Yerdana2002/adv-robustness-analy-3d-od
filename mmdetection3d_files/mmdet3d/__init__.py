# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

import mmdet
import mmseg
from .version import __version__, short_version
from .models.necks.focal_encoder import FocalEncoder #TODO: UNCOMMENT!!!
from .models.dense_heads.focal_decoder import FocalDecoder
from .models.detectors.focalformer3d import FocalFormer3D
from .core.bbox.assigners.hungarian_assigner import HungarianAssigner3D, HeuristicAssigner3D
from .core.bbox.coders.transfusion_bbox_coder import TransFusionBBoxCoder
from .core.hook.fading import Fading
from .datasets.pipelines import (
  PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
  NormalizeMultiviewImage, ScaleImageMultiViewImage)


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.3.8'
mmcv_maximum_version = '1.5.0'
mmcv_version = digit_version(mmcv.__version__)


assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

mmdet_minimum_version = '2.19.0'
mmdet_maximum_version = '3.0.0'
mmdet_version = digit_version(mmdet.__version__)
assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version <= digit_version(mmdet_maximum_version)), \
    f'MMDET=={mmdet.__version__} is used but incompatible. ' \
    f'Please install mmdet>={mmdet_minimum_version}, ' \
    f'<={mmdet_maximum_version}.'

mmseg_minimum_version = '0.20.0'
mmseg_maximum_version = '1.0.0'
mmseg_version = digit_version(mmseg.__version__)
assert (mmseg_version >= digit_version(mmseg_minimum_version)
        and mmseg_version <= digit_version(mmseg_maximum_version)), \
    f'MMSEG=={mmseg.__version__} is used but incompatible. ' \
    f'Please install mmseg>={mmseg_minimum_version}, ' \
    f'<={mmseg_maximum_version}.'

__all__ = ['__version__', 'short_version']
