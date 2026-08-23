# =============================================================================
# custom_pipelines.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - PIPELINES from mmdet.datasets.builder    → TRANSFORMS from mmdet3d.registry
#   - @PIPELINES.register_module()             → @TRANSFORMS.register_module()
#   - RandomFlip from mmdet.datasets.pipelines → mmdet.datasets.transforms
#   - mmcv.is_list_of still works in mmcv 2.x  (mmcv.utils → mmcv)
#   - mmcv image ops (impad, imnormalize, imresize, imread, etc.) still in
#     mmcv 2.x — no change needed
#   - Removed unused imports: VoxelGenerator, build_from_cfg, box_np_ops,
#     OBJECTSAMPLERS, noise_per_object_v3_, LoadAnnotations, LoadImageFromFile,
#     CameraInstance3DBoxes, DepthInstance3DBoxes, LiDARInstance3DBoxes,
#     is_tuple_of, warnings
#   - PhotoMetricDistortionMultiViewImage: added missing `import random`
# =============================================================================
import logging
import random

import mmcv
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)
logger.info('[custom_pipelines] Loading module...')

# --- Registry import ---
try:
    from mmdet3d.registry import TRANSFORMS
    logger.info('[custom_pipelines] ✓ Imported TRANSFORMS '
                'from mmdet3d.registry')
except ImportError as e:
    logger.error(f'[custom_pipelines] ✗ Failed to import TRANSFORMS: {e}')
    logger.error('  → This replaced PIPELINES from mmdet.datasets.builder')
    raise

# --- RandomFlip base class (for MyFlip3D) ---
try:
    from mmdet.datasets.transforms import RandomFlip
    logger.info('[custom_pipelines] ✓ Imported RandomFlip '
                'from mmdet.datasets.transforms')
except ImportError as e:
    logger.error(f'[custom_pipelines] ✗ Failed to import RandomFlip: {e}')
    logger.error('  → Moved from mmdet.datasets.pipelines to '
                 'mmdet.datasets.transforms in mmdet 3.x')
    raise


# ===================================================================
# ImageAug3D
# ===================================================================
@TRANSFORMS.register_module()
class ImageAug3D:
    def __init__(
        self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
        is_train,
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train
        logger.info(f'[ImageAug3D] ✓ Initialized: final_dim={final_dim}, '
                    f'resize_lim={resize_lim}, is_train={is_train}')

    def sample_augmentation(self, results):
        H, W, _, _ = results["img_shape"]
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (int((1 - np.random.uniform(*self.bot_pct_lim)) * newH)
                       - fH)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip,
        rotate
    ):
        img = Image.fromarray(img)
        if np.fabs(resize - 1.) > 1e-10:
            img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        img = np.asarray(img)
        return img, rotation, translation

    def __call__(self, data):
        imgs = data["img"]
        imgs = [i.astype(np.uint8) for i in imgs]
        logger.debug(f'[ImageAug3D] Processing {len(imgs)} images, '
                     f'is_train={self.is_train}')
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = \
                self.sample_augmentation(data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img, post_rot, post_tran,
                resize=resize, resize_dims=resize_dims, crop=crop,
                flip=flip, rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img)
            transforms.append(transform.numpy())

        assert (len(data['seg_fields']) == 0 and
                len(data['bbox_fields']) == 0 and
                len(data['mask_fields']) == 0), \
            'bbox2d should not be used in resize (MyResize)'

        data["img"] = [i.astype(np.float32) for i in new_imgs]
        data["img_aug_matrix"] = torch.as_tensor(transforms)
        data['img_shape'] = [img.shape for img in data['img']]
        logger.debug(f'[ImageAug3D] Output: {len(data["img"])} images, '
                     f'shapes={[i.shape for i in data["img"]]}')
        return data


logger.info('[custom_pipelines] ✓ Registered ImageAug3D')


# ===================================================================
# PadMultiViewImage
# ===================================================================
@TRANSFORMS.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image."""

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None
        logger.info(f'[PadMultiViewImage] ✓ Initialized: '
                    f'size={size}, size_divisor={size_divisor}')

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val)
                for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val)
                for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        logger.debug(f'[PadMultiViewImage] Input: '
                     f'{len(results["img"])} images')
        self._pad_img(results)
        logger.debug(f'[PadMultiViewImage] Output pad_shape='
                     f'{results["pad_shape"]}')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


logger.info('[custom_pipelines] ✓ Registered PadMultiViewImage')


# ===================================================================
# NormalizeMultiviewImage
# ===================================================================
@TRANSFORMS.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        logger.info(f'[NormalizeMultiviewImage] ✓ Initialized: '
                    f'mean={self.mean}, std={self.std}, to_rgb={to_rgb}')

    def __call__(self, results):
        logger.debug(f'[NormalizeMultiviewImage] Normalizing '
                     f'{len(results["img"])} images')
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb)
            for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(mean={self.mean}, std={self.std}, '
                     f'to_rgb={self.to_rgb})')
        return repr_str


logger.info('[custom_pipelines] ✓ Registered NormalizeMultiviewImage')


# ===================================================================
# ScaleImageMultiViewImage
# ===================================================================
@TRANSFORMS.register_module()
class ScaleImageMultiViewImage(object):
    """Scale the image."""

    def __init__(self, scales=(800, 448)):
        self.scales = np.array(scales)
        self.scales[0] = self.scales[0] + self.scales[1]
        self.scales[1] = self.scales[0] - self.scales[1]
        self.scales[0] = self.scales[0] - self.scales[1]
        logger.info(f'[ScaleImageMultiViewImage] ✓ Initialized: '
                    f'scales={self.scales}')

    def __call__(self, results):
        img_shape = results['img_shape']
        rand_scale = self.scales / np.array(img_shape[:2])
        y_size = int(img_shape[0] * rand_scale[0])
        x_size = int(img_shape[1] * rand_scale[1])
        logger.debug(f'[ScaleImageMultiViewImage] '
                     f'img_shape={img_shape}, scale→({x_size},{y_size})')
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale[1]
        scale_factor[1, 1] *= rand_scale[0]
        results['img'] = [mmcv.imresize(
            img, (x_size, y_size), return_scale=False)
            for img in results['img']]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


logger.info('[custom_pipelines] ✓ Registered ScaleImageMultiViewImage')


# ===================================================================
# PhotoMetricDistortionMultiViewImage
# ===================================================================
@TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially."""

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        logger.info(f'[PhotoMetricDistortionMV] ✓ Initialized: '
                    f'brightness_delta={brightness_delta}, '
                    f'hue_delta={hue_delta}')

    def __call__(self, results):
        imgs = results['img']
        logger.debug(f'[PhotoMetricDistortionMV] Processing '
                     f'{len(imgs)} images')
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                ('PhotoMetricDistortion needs input dtype np.float32, '
                 'please set "to_float32=True" in "LoadImageFromFile"')
            # random brightness
            if random.randint(0, 1):
                delta = random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(0, 1)
            if mode == 1:
                if random.randint(0, 1):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(0, 1):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                              self.saturation_upper)

            # random hue
            if random.randint(0, 1):
                img[..., 0] += random.uniform(-self.hue_delta,
                                              self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(0, 1):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(0, 1):
                img = img[..., random.sample(range(3), 3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


logger.info('[custom_pipelines] ✓ Registered '
            'PhotoMetricDistortionMultiViewImage')


# ===================================================================
# LoadMultiViewImageFromFilesWaymo
# ===================================================================
@TRANSFORMS.register_module()
class LoadMultiViewImageFromFilesWaymo(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, to_float32=False, img_scale=None,
                 color_type='unchanged'):
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type
        logger.info(f'[LoadMultiViewImageFromFilesWaymo] ✓ Initialized: '
                    f'to_float32={to_float32}, img_scale={img_scale}')

    def pad(self, img):
        if img.shape[0] != self.img_scale[0]:
            img = np.concatenate(
                [img, np.zeros_like(img[0:1280 - 886, :])], axis=0)
        return img

    def __call__(self, results):
        filename = results['img_filename']
        logger.debug(f'[LoadMultiViewImageFromFilesWaymo] Loading '
                     f'{len(filename)} images')
        if self.img_scale is None:
            img = np.stack(
                [mmcv.imread(name, self.color_type) for name in filename],
                axis=-1)
        else:
            img = np.stack(
                [self.pad(mmcv.imread(name, self.color_type))
                 for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        logger.debug(f'[LoadMultiViewImageFromFilesWaymo] Loaded: '
                     f'img_shape={img.shape}, '
                     f'num_views={len(results["img"])}')
        return results

    def __repr__(self):
        return ("{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type))


logger.info('[custom_pipelines] ✓ Registered '
            'LoadMultiViewImageFromFilesWaymo')


# ===================================================================
# MyResize
# ===================================================================
@TRANSFORMS.register_module()
class MyResize(object):
    """Resize images & bbox & mask.
    Multi-view aware: operates on results['img'] as a list.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            assert len(self.img_scale) == 1
        else:
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.override = override
        self.bbox_clip_border = bbox_clip_border
        logger.info(f'[MyResize] ✓ Initialized: img_scale={img_scale}, '
                    f'keep_ratio={keep_ratio}, '
                    f'multiscale_mode={multiscale_mode}')

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = (np.random.random_sample() * (max_ratio - min_ratio)
                 + min_ratio)
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError
        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        imgs = results['img']
        results['img'] = [imgs[i] for i in range(len(imgs))]
        for key in results.get('img_fields', ['img']):
            for idx in range(len(results['img'])):
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(
                        results[key][idx], results['scale'],
                        return_scale=True, backend=self.backend)
                    new_h, new_w = img.shape[:2]
                    h, w = results[key][idx].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(
                        results[key][idx], results['scale'],
                        return_scale=True, backend=self.backend)
                results[key][idx] = img

            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            results['img_shape'] = img.shape
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'],
                    interpolation='nearest', backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'],
                    interpolation='nearest', backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'][0].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, \
                    'scale and scale_factor cannot be both set.'
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        logger.debug(f'[MyResize] Resizing to scale={results["scale"]}')
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        logger.debug(f'[MyResize] Output img_shape={results["img_shape"]}')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


logger.info('[custom_pipelines] ✓ Registered MyResize')


# ===================================================================
# MyNormalize
# ===================================================================
@TRANSFORMS.register_module()
class MyNormalize(object):
    """Normalize the image. Multi-view aware."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        logger.info(f'[MyNormalize] ✓ Initialized: mean={self.mean}, '
                    f'std={self.std}, to_rgb={to_rgb}')

    def __call__(self, results):
        logger.debug(f'[MyNormalize] Normalizing {len(results["img"])} imgs')
        for key in results.get('img_fields', ['img']):
            for idx in range(len(results['img'])):
                results[key][idx] = mmcv.imnormalize(
                    results[key][idx], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(mean={self.mean}, std={self.std}, '
                     f'to_rgb={self.to_rgb})')
        return repr_str


logger.info('[custom_pipelines] ✓ Registered MyNormalize')


# ===================================================================
# MyPad
# ===================================================================
@TRANSFORMS.register_module()
class MyPad(object):
    """Pad the image & mask. Multi-view aware."""

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None
        logger.info(f'[MyPad] ✓ Initialized: size={size}, '
                    f'size_divisor={size_divisor}')

    def _pad_img(self, results):
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                for idx in range(len(results[key])):
                    padded_img = mmcv.impad_to_multiple(
                        results[key][idx], self.size_divisor,
                        pad_val=self.pad_val)
                    results[key][idx] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def _pad_seg(self, results):
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2])

    def __call__(self, results):
        logger.debug('[MyPad] Padding images')
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        logger.debug(f'[MyPad] pad_shape={results["pad_shape"]}')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


logger.info('[custom_pipelines] ✓ Registered MyPad')


# ===================================================================
# MyFlip3D
# ===================================================================
@TRANSFORMS.register_module()
class MyFlip3D(RandomFlip):
    """Flip the points & bbox.

    NOTE: In mmdet 3.x, RandomFlip's API changed. The parent __init__
    now takes `prob` instead of `flip_ratio`. If you encounter
    "unexpected keyword argument 'flip_ratio'", change the super().__init__
    to use `prob=flip_ratio_bev_horizontal` instead.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        # mmdet 3.x RandomFlip uses `prob` instead of `flip_ratio`
        # Try both for compatibility
        try:
            super(MyFlip3D, self).__init__(
                prob=flip_ratio_bev_horizontal, **kwargs)
            logger.info('[MyFlip3D] ✓ Initialized with prob= '
                        f'(mmdet 3.x style), '
                        f'h_flip={flip_ratio_bev_horizontal}, '
                        f'v_flip={flip_ratio_bev_vertical}')
        except TypeError:
            logger.warning('[MyFlip3D] ⚠ prob= kwarg failed, trying '
                           'flip_ratio= (mmdet 2.x style)')
            super(MyFlip3D, self).__init__(
                flip_ratio=flip_ratio_bev_horizontal, **kwargs)
            logger.info('[MyFlip3D] ✓ Initialized with flip_ratio= '
                        f'(mmdet 2.x style)')

        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(flip_ratio_bev_horizontal, (int, float)) \
                and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(flip_ratio_bev_vertical, (int, float)) \
                and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
            else:
                input_dict[key].flip(direction)
        if 'centers2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['ori_shape'][1]
            input_dict['centers2d'][..., 0] = \
                w - input_dict['centers2d'][..., 0]
            input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]

    def __call__(self, input_dict):
        # flip 2D image and its annotations
        super(MyFlip3D, self).__call__(input_dict)
        input_dict['img'] = [img for img in input_dict['img']]
        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])

        logger.debug(f'[MyFlip3D] h_flip={input_dict["pcd_horizontal_flip"]}, '
                     f'v_flip={input_dict["pcd_vertical_flip"]}')
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += (f' flip_ratio_bev_vertical='
                     f'{self.flip_ratio_bev_vertical})')
        return repr_str


logger.info('[custom_pipelines] ✓ Registered MyFlip3D')
logger.info('[custom_pipelines] ✓ Module fully loaded')