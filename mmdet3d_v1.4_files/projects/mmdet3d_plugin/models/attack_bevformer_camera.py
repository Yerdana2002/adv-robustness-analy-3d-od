#!/usr/bin/env python
"""Camera-corruption attacks for BEVFormer on the nuScenes val split.

Three corruptions, each with its own config under projects/configs/bevformer:

  mb  Motion Blur         zoom blur on CAM_FRONT/CAM_BACK, horizontal motion
                          blur on the four side cameras.       (severity 3)
  sa  Spatial Alignment   Gaussian noise on the lidar2img extrinsics, so
                          spatial cross-attention samples the wrong image
                          locations. Pixels are untouched.      (severity 2)
  sc  Scale               per-object TPS warp of the image patch, driven by
                          the projected GT 3D box corners.      (severity 2)

Severities are the group's agreed settings and live in the configs, not here.

Usage:
    python projects/mmdet3d_plugin/models/attack_bevformer_camera.py \
        projects/configs/bevformer/bevformer_base_mb.py CKPT.pth \
        --data-root /path/to/nuscenes --out-dir /path/to/results

Every run writes <out-dir>/bevformer_attack_<abbr>.pkl containing, per val
frame: GT boxes/labels, predicted boxes/scores/labels, lidar2img, can_bus,
scene/frame identifiers, and the measured image distortion. See
projects/mmdet3d_plugin/hooks/attack_record_hook.py.
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmcv.transforms import BaseTransform  # noqa: E402
from mmdet3d.registry import TRANSFORMS  # noqa: E402

from projects.mmdet3d_plugin.camera_corruptions import (  # noqa: E402
    ImageBBoxOperation,
    ImageMotionBlurFrontBack,
    ImageMotionBlurLeftRight,
    spatial_alignment_noise,
)

_ABBR_TO_TYPE = {'mb': 'MotionBlur', 'sa': 'SpatialAlignment', 'sc': 'Scale'}


@TRANSFORMS.register_module()
class CameraCorruption(BaseTransform):
    """Apply a camera corruption to the multi-view images.

    Subclasses BaseTransform deliberately: mmengine's Compose invokes each
    stage as ``t(data)``, so a plain class exposing only ``transform()`` is not
    callable and raises "object is not callable". BaseTransform.__call__
    forwards to transform().

    Sits after LoadAnnotations3D and before NormalizeMultiviewImage, i.e. GT is
    available and images are still raw uint8-valued BGR.

    Args:
        corruption_type (str): 'mb', 'sa' or 'sc'.
        severity (int): 1-5.
        measure_distortion (bool): Record how much the images actually changed.
            The group needs an image-side analogue of the point-cloud AI@x
            metric, so the perturbation magnitude is stored per frame rather
            than inferred afterwards.
    """

    def __init__(self, corruption_type='mb', severity=3,
                 measure_distortion=True):
        assert corruption_type in _ABBR_TO_TYPE, corruption_type
        self.corruption_type = corruption_type
        self.severity = int(severity)
        self.measure_distortion = measure_distortion

        if corruption_type == 'mb':
            self.frontback_blur = ImageMotionBlurFrontBack(self.severity)
            self.leftright_blur = ImageMotionBlurLeftRight(self.severity)
        elif corruption_type == 'sc':
            self.bbox_scale = ImageBBoxOperation(self.severity)

    # -- distortion bookkeeping -------------------------------------------
    @staticmethod
    def _distortion(before, after):
        """Per-frame perturbation magnitude, averaged over the 6 cameras."""
        rmse_sum, linf, changed, denom = 0.0, 0.0, 0.0, 0
        for a, b in zip(before, after):
            d = np.abs(b.astype(np.float32) - a.astype(np.float32))
            rmse_sum += float(np.sqrt(np.mean(d ** 2)))
            linf = max(linf, float(d.max()))
            changed += float((d > 1.0).mean())
            denom += 1
        denom = max(denom, 1)
        return dict(
            rmse=rmse_sum / denom,
            linf=linf,
            frac_pixels_changed=changed / denom,
        )

    def transform(self, results):
        imgs = results['img']  # list of 6 HxWx3 arrays, BGR

        if self.corruption_type == 'mb':
            before = [i.copy() for i in imgs] if self.measure_distortion else None
            results['img'] = self._motion_blur(imgs)
            if self.measure_distortion:
                results['img_distortion'] = self._distortion(
                    before, results['img'])

        elif self.corruption_type == 'sa':
            # Extrinsic-only corruption: images are bit-identical, so image
            # distortion is zero by construction. Record the extrinsic delta
            # instead, otherwise the perturbation is unrecoverable downstream.
            before = np.asarray(results['lidar2img'], dtype=np.float64).copy()
            results['lidar2img'] = self._spatial_alignment(results['lidar2img'])
            after = np.asarray(results['lidar2img'], dtype=np.float64)
            if self.measure_distortion:
                results['img_distortion'] = dict(
                    rmse=0.0, linf=0.0, frac_pixels_changed=0.0,
                    lidar2img_rmse=float(
                        np.sqrt(np.mean((after - before) ** 2))),
                    lidar2img_linf=float(np.abs(after - before).max()),
                )

        elif self.corruption_type == 'sc':
            gt = results.get('gt_bboxes_3d', None)
            if gt is None or len(gt) == 0:
                # No GT in this frame -> nothing to warp. Record a zero delta
                # so every sample still has the field.
                if self.measure_distortion:
                    results['img_distortion'] = dict(
                        rmse=0.0, linf=0.0, frac_pixels_changed=0.0,
                        num_objects_warped=0)
                return results
            before = [i.copy() for i in imgs] if self.measure_distortion else None
            results['img'] = self._scale(imgs, results['lidar2img'], gt)
            if self.measure_distortion:
                d = self._distortion(before, results['img'])
                d['num_objects_warped'] = int(len(gt))
                results['img_distortion'] = d

        return results

    # -- corruptions -------------------------------------------------------
    def _motion_blur(self, imgs):
        out = []
        for i, img_bgr in enumerate(imgs):
            img_rgb = img_bgr[:, :, [2, 1, 0]]
            # nuScenes camera order is CAM_FRONT, CAM_FRONT_RIGHT,
            # CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, so
            # indices 0 and 3 are the front/back pair.
            blur = self.frontback_blur if i % 3 == 0 else self.leftright_blur
            out.append(blur(image=img_rgb)[:, :, [2, 1, 0]])
        return out

    def _spatial_alignment(self, lidar2img_list):
        return [
            spatial_alignment_noise(
                np.asarray(m, dtype=np.float64).copy(), self.severity)
            for m in lidar2img_list
        ]

    def _scale(self, imgs, lidar2img_list, gt_bboxes_3d):
        c = [0.1, 0.2, 0.3, 0.4, 0.5][self.severity - 1]
        t = np.random.choice([-1, 1])
        a = b = d = 1.0 + c * t
        transform_matrix = torch.tensor(
            [[a, 0, 0], [0, b, 0], [0, 0, d]]).float()

        corners = gt_bboxes_3d.corners
        centers = gt_bboxes_3d.center
        if corners.numel() == 0:
            return imgs

        out = []
        for i, img_bgr in enumerate(imgs):
            img_rgb = img_bgr[:, :, [2, 1, 0]]
            warped = self.bbox_scale(
                image=img_rgb,
                lidar2img=lidar2img_list[i],
                transform_matrix=transform_matrix,
                bboxes_centers=centers,
                bboxes_corners=corners,
                is_nus=True,
            )
            out.append(warped[:, :, [2, 1, 0]])
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'corruption_type={self.corruption_type}, '
                f'severity={self.severity})')


# ==========================================================================
# Driver
# ==========================================================================

def run_attack(cfg_path, checkpoint_path, data_root, ann_file, out_dir,
               severity=None, seed=0, clean=False):
    logger = MMLogger.get_instance('attack_bevformer_camera', log_level='INFO')
    os.makedirs(out_dir, exist_ok=True)

    cfg = Config.fromfile(cfg_path)
    if hasattr(cfg, 'custom_imports'):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    assert 'corruption' in cfg, (
        f'{cfg_path} has no `corruption` block; use one of '
        'bevformer_base_{mb,sa,sc}.py')
    abbr = cfg.corruption['abbreviation']
    sev = int(severity) if severity is not None else int(
        cfg.corruption['severity'])

    pipeline = cfg.test_dataloader.dataset.pipeline

    # Refuse to run if augmentation crept back in -- it would confound the
    # measured corruption delta.
    banned = ('PhotoMetricDistortionMultiViewImage', 'RandomFlip3D',
              'GlobalRotScaleTrans', 'MultiScaleFlipAug3D')
    found = [s['type'] for s in pipeline if s['type'] in banned]
    if found:
        raise RuntimeError(
            f'Data augmentation present in the attack pipeline: {found}. '
            'The corruption must be the only perturbation.')

    # The Scale attack needs GT; that requires the train-style dataset path.
    if cfg.test_dataloader.dataset.get('test_mode', True):
        raise RuntimeError(
            'test_mode=True: Det3DDataset will not populate ann_info, so GT '
            'is unavailable. Use bevformer_base_attack.py as the base.')

    if not clean:
        insert_at = next(
            (i for i, s in enumerate(pipeline)
             if s['type'] == 'NormalizeMultiviewImage'), None)
        if insert_at is None:
            raise ValueError('NormalizeMultiviewImage not found in pipeline')
        pipeline.insert(insert_at, dict(
            type='CameraCorruption', corruption_type=abbr, severity=sev))
        logger.info(f'Injected CameraCorruption({abbr}, severity={sev}) '
                    f'at pipeline index {insert_at}')
    else:
        abbr = 'clean'
        logger.info('Clean run: no corruption injected (baseline reference)')

    logger.info('Pipeline: ' + ' -> '.join(s['type'] for s in pipeline))

    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.ann_file = ann_file
    cfg.test_evaluator.data_root = data_root
    cfg.test_evaluator.ann_file = os.path.join(data_root, ann_file)
    cfg.test_evaluator.jsonfile_prefix = os.path.join(
        out_dir, f'bevformer_{abbr}_results')

    rec_name = f'bevformer_attack_{abbr}.pkl'
    cfg.custom_hooks = cfg.get('custom_hooks', []) + [
        dict(type='AttackRecordHook', out_path=rec_name, tag=abbr)
    ]

    cfg.work_dir = out_dir
    cfg.load_from = checkpoint_path
    cfg.resume = False
    cfg.randomness = dict(seed=seed, deterministic=False)

    runner = Runner.from_cfg(cfg)
    logger.info(f'Running {abbr} (severity={sev}) over '
                f'{len(runner.test_dataloader.dataset)} val frames...')
    metrics = runner.test()

    logger.info(f'{abbr} metrics: {metrics}')
    logger.info(f'Records: {os.path.join(out_dir, rec_name)}')
    return metrics


def parse_args():
    p = argparse.ArgumentParser(
        description='Camera corruption attacks for BEVFormer (nuScenes val)')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--severity', type=int, default=None,
                   help='override the severity set in the config')
    p.add_argument('--data-root', default='data/nuscenes/')
    p.add_argument('--ann-file', default='nuscenes_infos_temporal_val.pkl')
    p.add_argument('--out-dir', default='./attack_results')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--clean', action='store_true',
                   help='skip the corruption; produces the paired clean '
                        'reference through the identical pipeline')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_attack(
        cfg_path=args.config,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        ann_file=args.ann_file,
        out_dir=args.out_dir,
        severity=args.severity,
        seed=args.seed,
        clean=args.clean,
    )
