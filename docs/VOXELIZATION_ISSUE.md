# Voxelization non-differentiability
The issue with the voxelization implemented in mmdetection3d is that the backward pass is not implemented. This breaks the gradient flow and does not allow our attacks to use the gradients of the models. To fix this we use the proposed solution from the [IoU-S Attack](https://github.com/haichen-ber/IoU-S-Attack/tree/main) repository.
## The solution
1. Go to the [paper repository](https://github.com/zzj403/BEV_Robust) they mentioned
2. Open [transfusion_changes.patch](https://github.com/zzj403/BEV_Robust/blob/main/transfusion_changes.patch)
3. Go to `mmdet3d/ops/voxel` and apply all changes to `src/voxelization_cuda.cu`, `src/voxelization_cpu.cpp`, `src/voxelization.h` and `voxelize.py`
4. Install mmdetection3d by using `pip -v -e .` in the `mmdetection3d/` directory