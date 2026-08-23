# datasets/nuscenes_dataset.py
"""NuScenes dataset variant that exposes what BEVFormer needs.

Stock ``NuScenesDataset`` cannot feed BEVFormer, for three reasons:

  * ``can_bus`` / ``scene_token`` / ``prev`` / ``next`` / ``frame_idx`` are not
    in the v2 info schema at all. Run
    ``tools/inject_bevformer_temporal_fields.py`` over the pkl first; this
    class only surfaces what that script wrote.
  * ``Det3DDataset.parse_data_info`` computes ``lidar2img`` for the single
    ``default_cam_key`` only, but ``BEVFormerEncoder.point_sampling`` indexes
    ``img_metas[i]['lidar2img']`` as one 4x4 per camera.
  * ``LoadMultiViewImageFromFiles`` reads ``results['img_filename']``, a list
    the stock parser never builds.

Note on training: BEVFormer's original dataset also assembles a temporal queue
of ``queue_length`` frames per training sample. That is not implemented here --
this class targets inference, where history is carried by the detector's
``prev_frame_info`` cache instead. ``loss()`` will need the queue added before
training works.
"""

import numpy as np

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    """NuScenes dataset with BEVFormer's temporal and multi-view fields."""

    def parse_data_info(self, info: dict) -> dict:
        data_info = super().parse_data_info(info)

        # --- temporal / CAN-bus fields -----------------------------------
        # Fail loudly rather than silently degrading: a missing can_bus means
        # the pkl was never passed through the injection script, and BEVFormer
        # would otherwise run with a zero ego-motion prior.
        if 'can_bus' not in info:
            raise KeyError(
                "'can_bus' missing from the info file. Run "
                'tools/inject_bevformer_temporal_fields.py over it first -- '
                'create_data.py cannot produce this field.')

        data_info['can_bus'] = np.array(info['can_bus'], dtype=np.float64)
        data_info['scene_token'] = info['scene_token']
        data_info['frame_idx'] = info['frame_idx']
        data_info['prev_idx'] = info['prev']
        data_info['next_idx'] = info['next']
        data_info['prev_bev_exists'] = info['prev'] != ''

        # --- multi-view image paths and lidar2img ------------------------
        if self.modality.get('use_camera', False):
            image_paths = []
            lidar2img_rts = []
            cam_intrinsics = []
            lidar2cam_rts = []

            for _, cam_info in data_info['images'].items():
                image_paths.append(cam_info['img_path'])

                lidar2cam = np.array(cam_info['lidar2cam'], dtype=np.float64)
                intrinsic = np.array(cam_info['cam2img'], dtype=np.float64)

                # cam2img is 3x3 in the v2 schema; pad to 4x4 so the product
                # with the 4x4 lidar2cam is well formed.
                viewpad = np.eye(4, dtype=np.float64)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                lidar2img_rts.append(viewpad @ lidar2cam)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam)

            data_info['img_filename'] = image_paths
            data_info['lidar2img'] = lidar2img_rts
            data_info['cam_intrinsic'] = cam_intrinsics
            data_info['lidar2cam'] = lidar2cam_rts

        return data_info
