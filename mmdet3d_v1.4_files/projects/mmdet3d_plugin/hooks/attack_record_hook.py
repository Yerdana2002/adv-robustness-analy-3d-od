# hooks/attack_record_hook.py
"""Dump per-sample GT + predictions + metadata from a Det3DDataSample run.

Attaches to Runner.test() and collects, for every val frame, everything the
dataloader and the model produced, then writes ONE file at the end.

Format: pickle, not JSON. Det3DDataSample carries torch tensors and
LiDARInstance3DBoxes; JSON would need every array converted to nested lists,
which roughly triples the size, loses dtype, and is far slower to re-read for
analysis. Pickle keeps native numpy. What is stored is plain numpy/str/float
only -- no mmdet3d classes -- so the file unpickles without mmdet3d installed.

One file per attack keeps the project filesystem's ~1000-file quota safe
(3 attacks -> 3 files, ~150 MB each).
"""

import os.path as osp
import pickle

import numpy as np
from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


def _boxes_to_numpy(instances, prefix):
    """Flatten an InstanceData of 3D boxes into plain numpy arrays."""
    out = {}
    if instances is None:
        return out
    boxes = getattr(instances, 'bboxes_3d', None)
    if boxes is not None:
        tensor = boxes.tensor if hasattr(boxes, 'tensor') else boxes
        out[f'{prefix}_bboxes_3d'] = \
            tensor.detach().cpu().numpy().astype(np.float32)
        out[f'{prefix}_box_type'] = type(boxes).__name__
    for attr, name in (('labels_3d', 'labels_3d'), ('scores_3d', 'scores_3d')):
        val = getattr(instances, attr, None)
        if val is not None:
            arr = val.detach().cpu().numpy()
            out[f'{prefix}_{name}'] = (
                arr.astype(np.float32) if name == 'scores_3d'
                else arr.astype(np.int64))
    return out


@HOOKS.register_module()
class AttackRecordHook(Hook):
    """Record GT, predictions and metadata for every test sample.

    Args:
        out_path (str): Destination .pkl. Relative paths resolve against
            ``runner.work_dir``.
        tag (str): Free-form label stored in the file header (e.g. 'mb').
        meta_fields (Sequence[str]): metainfo keys to copy verbatim.
        save_lidar2img (bool): Store the 6 lidar2img matrices. Needed for the
            spatial-alignment attack, where the perturbation lives entirely in
            the extrinsics and is otherwise unrecoverable from the output.
    """

    priority = 'LOW'

    def __init__(self,
                 out_path='attack_records.pkl',
                 tag='',
                 meta_fields=('token', 'sample_idx', 'scene_token',
                              'frame_idx', 'timestamp', 'can_bus',
                              'img_distortion', 'prev_bev_exists'),
                 save_lidar2img=True):
        self.out_path = out_path
        self.tag = tag
        self.meta_fields = tuple(meta_fields)
        self.save_lidar2img = save_lidar2img
        self.records = []

    def before_test(self, runner):
        self.records = []
        runner.logger.info(
            f'[AttackRecordHook] recording to {self._resolve(runner)}')

    def _resolve(self, runner):
        return self.out_path if osp.isabs(self.out_path) \
            else osp.join(runner.work_dir, self.out_path)

    def after_test_iter(self, runner, batch_idx, data_batch=None,
                        outputs=None):
        if outputs is None:
            return
        for sample in outputs:
            meta = sample.metainfo
            rec = {k: meta[k] for k in self.meta_fields if k in meta}

            if self.save_lidar2img and 'lidar2img' in meta:
                rec['lidar2img'] = np.asarray(
                    meta['lidar2img'], dtype=np.float32)

            rec.update(_boxes_to_numpy(
                getattr(sample, 'gt_instances_3d', None), 'gt'))
            rec.update(_boxes_to_numpy(
                getattr(sample, 'pred_instances_3d', None), 'pred'))

            rec['num_gt'] = int(len(rec.get('gt_labels_3d', [])))
            rec['num_pred'] = int(len(rec.get('pred_labels_3d', [])))
            self.records.append(rec)

    def after_test_epoch(self, runner, metrics=None):
        path = self._resolve(runner)
        payload = {
            'tag': self.tag,
            'num_samples': len(self.records),
            'metrics': dict(metrics) if metrics else {},
            'classes': runner.test_dataloader.dataset.metainfo.get('classes'),
            'records': self.records,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        n_gt = sum(r['num_gt'] for r in self.records)
        n_pred = sum(r['num_pred'] for r in self.records)
        runner.logger.info(
            f'[AttackRecordHook] wrote {len(self.records)} samples '
            f'({n_gt} GT boxes, {n_pred} predictions) -> {path} '
            f'({osp.getsize(path) / 1e6:.0f} MB)')
