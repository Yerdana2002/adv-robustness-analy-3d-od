# Voxelization non-differentiability
The issue with the voxelization implemented in mmdetection3d is that the backward pass is not implemented. This breaks the gradient flow and does not allow our attacks to use the gradients of the models. To fix this we use the proposed solution from the [IoU-S Attack](https://github.com/haichen-ber/IoU-S-Attack/tree/main) repository.
## The solution
1. Go to the [paper repository](https://github.com/zzj403/BEV_Robust) they mentioned
2. Open [transfusion_changes.patch](https://github.com/zzj403/BEV_Robust/blob/main/transfusion_changes.patch)
3. Go to `mmdet3d/ops/voxel` and apply all changes to `src/voxelization_cuda.cu`, `src/voxelization_cpu.cpp`, `src/voxelization.h` and `voxelize.py`
4. Install mmdetection3d by using `pip -v -e .` in the `mmdetection3d/` directory

> **Applies to `mmcv-full` 1.4.x / `mmdetection3d` v1.0.0rc only.** On mmdetection3d ≥ 1.1
> the `mmdet3d/ops/` directory no longer exists, voxelization was moved into mmcv, so
> step 3 above has nothing to patch.

## On MMCV 2.x.x

If you are on **mmcv 2.x**, refer to
**[VOXELIZATION_MMCV2.md](VOXELIZATION_MMCV2.md)**, which documents the same fix
re-targeted at `mmcv/ops/` and verified against mmcv 2.2.0 / mmdetection3d 1.4.0 /
torch 2.5.1 / CUDA 12.2 / Python 3.11.

You will need this if your GPUs are newer than the 1.4.x stack supports. On NVIDIA H100
(sm_90) nodes, e.g. Alliance Canada, whose current software environment ships no CUDA
11.x toolkit at all, mmcv-full 1.4.2 cannot be built, so the 1.x patch above is
unusable regardless of how it is applied.

[VOXELIZATION_MMCV2_USAGE.md](VOXELIZATION_MMCV2_USAGE.md) then shows how to call the
patched op from an attack script, worked through on FocalFormer3D.
