# FocalFormer3D plugin for mmdetection3d 1.4 / mmcv 2.2.
#
# This is the working plugin from our fork with the BEVFormer and BEVFusion
# imports removed, so it stands alone for anyone who only wants FocalFormer3D.
# The module layout deliberately mirrors NVlabs/FocalFormer3D
# (projects/mmdet3d_plugin/...) so upstream files can be diffed against these
# one-to-one -- see docs/SETTING_UP_FOCALFORMER3D_MMDET14.md.

# FocalFormer3D components need CUDA, which login nodes do not have.
#
# Keep this try/except. On a CPU-only node the import raises and these become
# None, and the FAILURE MODE IS CONFUSING: MODELS.build later reports
# "FocalFormer3D is not in the mmdet3d::model registry", which reads like a
# config bug but only means there was no GPU. Run registry checks in a GPU job.
try:
    from .models.necks.focal_encoder import FocalEncoder
    from .models.dense_heads.focal_decoder import FocalDecoder
    from .models.detectors.focalformer3d import FocalFormer3D
except (AssertionError, ImportError):
    FocalEncoder = None
    FocalDecoder = None
    FocalFormer3D = None

from .core.bbox.assigners.hungarian_assigner import (HungarianAssigner3D,
                                                     HeuristicAssigner3D)
from .core.bbox.coders.transfusion_bbox_coder import TransFusionBBoxCoder
from .core.hook.fading import Fading
from .datasets.pipelines import (PhotoMetricDistortionMultiViewImage,
                                 PadMultiViewImage,
                                 NormalizeMultiviewImage,
                                 ScaleImageMultiViewImage)
from .datasets.nuscenes_dataset import CustomNuScenesDataset
from .hooks.attack_record_hook import AttackRecordHook

# The LiDAR+camera pipeline transforms (FFLoadMultiViewImage, FFImageAug3D) are
# NOT imported here on purpose. They are pulled in through the LC config's
# custom_imports instead, because importing them unconditionally collides with
# projects/BEVFusion's registry names when both are on the path.

__all__ = [
    'FocalEncoder', 'FocalDecoder', 'FocalFormer3D',
    'HungarianAssigner3D', 'HeuristicAssigner3D', 'TransFusionBBoxCoder',
    'Fading', 'PhotoMetricDistortionMultiViewImage', 'PadMultiViewImage',
    'NormalizeMultiviewImage', 'ScaleImageMultiViewImage',
    'CustomNuScenesDataset', 'AttackRecordHook',
]
