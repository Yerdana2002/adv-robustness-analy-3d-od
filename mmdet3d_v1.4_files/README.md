# mmdet3d_v1.4_files

Working source for **FocalFormer3D** and **PillarNeSt** ported to
**mmdetection3d 1.4.0 / mmcv 2.2.0 / mmdet 3.3.0 / mmengine 0.10.5 / torch 2.5.1 /
Python 3.11 / CUDA 12.2**.

This is the counterpart to [`mmdetection3d_files/`](../mmdetection3d_files/), which holds
the same models for the older **mmdetection3d 1.0.0rc / mmcv-full 1.4.x** stack. Use this
tree if your GPUs are newer than that stack can be built for — see
[docs/VOXELIZATION_MMCV2.md §1.2](../docs/VOXELIZATION_MMCV2.md#12-why-we-cannot-stay-on-the-14x-stack-h100).

**Setup guides — read these first:**

* [docs/SETTING_UP_FOCALFORMER3D_MMDET14.md](../docs/SETTING_UP_FOCALFORMER3D_MMDET14.md)
* [docs/SETTING_UP_PILLARNEST_MMDET14.md](../docs/SETTING_UP_PILLARNEST_MMDET14.md)

Both models need a **checkpoint conversion** and a **coordinate-convention fix**. Without
either, they load without error and evaluate wrongly. Copying these files in and running
is not sufficient.

## Layout

```
projects/
  mmdet3d_plugin/          FocalFormer3D. Self-contained plugin; copy to
                           <mmdetection3d>/projects/ and reference via
                           custom_imports. Paths mirror NVlabs/FocalFormer3D
                           so upstream can be diffed against it file-by-file.
  configs/focalformer3d/   L, LC, Waymo; test, gradient-extraction, attack.

mmdet3d/                   PillarNeSt. NOT a plugin -- these patch mmdetection3d
  models/backbones/        core in place. Every filename is prefixed
  models/dense_heads/      `pillarnest_` so nothing shadows a stock module, and
  models/voxel_encoders/   four __init__.py files must be edited to register
  models/task_modules/     them. See the guide, section 3.1.
configs/pillarnest/        nuScenes / KITTI / Waymo; clean and adversarial.

tools/
  convert_focalformer_ckpt.py            mmcv 1.x attention names -> mmengine
  convert_pillarnest_ckpt_to_mmdet14.py  component prefixes -> pts_* names
```

## Scope

The plugin here is our working tree with the **BEVFormer, BEVFusion and
camera_corruptions** modules removed, so it stands alone for FocalFormer3D. Its
`__init__.py` was rewritten to match; the original imports those other models
unconditionally.

It also contains work that is **not** part of the upstream port and can be ignored if you
only want the detectors running: gradient-extraction hooks, adversarial attack entry
points (`models/attack_*.py`), and debug instrumentation. The setup guides mark which is
which.
