# Differentiable Voxelization on MMCV 2.x
## NOTE: AI WAS USED TO COMPILED THE DOCUMENTATION AND IT WAS READ BY A HUMAN BEFORE POSTING

Companion to [VOXELIZATION_ISSUE.md](VOXELIZATION_ISSUE.md). That document solves the
non-differentiable voxelization problem for the **mmcv-full 1.4.x / mmdetection3d
v0.18.1** stack that [INSTALL_MMD3D.md](INSTALL_MMD3D.md) pins. This document solves the
same problem for **mmcv 2.2.0 / mmdetection3d 1.4.0**, which is what you are forced onto
if your GPUs are newer than that stack.

---

## 1. Problem statement

### 1.1 The patch in VOXELIZATION_ISSUE.md has no target on mmdet3d ≥ 1.1

The existing fix says to edit `mmdet3d/ops/voxel/`: `voxelize.py`,
`src/voxelization_cuda.cu`, `src/voxelization_cpu.cpp`, `src/voxelization.h`.

On mmdetection3d 1.4.0 that directory does not exist:

OpenMMLab moved the CUDA ops out of mmdetection3d and into mmcv during the 1.x
refactor. Voxelization now lives in `mmcv/ops/voxelize.py` and
`mmcv/ops/csrc/pytorch/`, so the patch has to be re-targeted at mmcv itself.

### 1.2 Why we cannot stay on the 1.4.x stack: H100

Our compute is Alliance Canada's **rorqual** cluster. Its GPU nodes are NVIDIA H100
80 GB HBM3 (compute capability **sm_90**):


`INSTALL_MMD3D.md` pins CUDA 11.8, torch 2.0.1, Python 3.8 and mmcv-full 1.4.2. None of
that is reachable here:

* **rorqual exposes only `StdEnv/2023`.** `module load StdEnv/2020` is a silent no-op,
  it returns 0 and loads nothing. Under `StdEnv/2023`, `module avail cuda` returns
  `12.2`, `12.6`, `12.9`, `13.2`. **There is no CUDA 11.x toolkit on this cluster.**
* The Alliance wheelhouse for `StdEnv/2023` + `python/3.11` supplies **torch 2.5.1**
  built against **CUDA 12.2**. mmcv-full 1.4.2 was released in December 2021 and still
  includes `THC/THC.h`, removed in torch 1.11. `INSTALL_MMD3D.md` already documents
  patching those out, but the THC fix is necessary, not sufficient. 1.4.2 does not
  compile against torch 2.5 / CUDA 12.2, and mmdetection3d v0.18.1 is not importable on
  Python 3.11.

This is the general form of the problem, not specific to rorqual: **a large body of 3D
detection papers released code against mmcv 1.4.x, and that code cannot be built on any
Hopper or newer machine whose site software stack has dropped CUDA 11.** 

### 1.3 mmcv 2.2.0 is non-differentiable

Moving to mmcv 2.2.0 solves the build and reintroduces the original problem verbatim.
Stock `mmcv/ops/voxelize.py` wraps the CUDA op in an `autograd.Function` with **no
`backward` implemented**, and the op writes its results into a preallocated buffer
**in place**:

```python
voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1)))
ext_module.hard_voxelize_forward(points, ..., voxels, coors, ...)
return voxels[:vn], coors[:vn], num_points_per_voxel[:vn]
```

`voxels` is created by `new_zeros` (so `requires_grad=False`) and is only ever written to
by a C++ call that autograd cannot see. The returned tensor has `grad_fn=None`.

No exception or warnings are outputted and the you will notice that VFE, middle encoder,
backbone, neck and loss all will run normally; `loss.backward()` succeeds; and
`points.grad` is exactly zero. Therefore, an attack built on it produces adversarial point clouds
that are pure initialization noise while reporting plausible losses. We tested it by running a control with the adversarial term deleted (`loss_sign=0`) and
finding it reproduced the "attack" to within 0.001 mAP (paired test over 124,831 boxes,
p = 0.49). **Do not assume the gradient is alive because the attack ran**, see
[§5.2](#52-the-gradient-is-actually-alive) for the check.

---

## 2. Environment

Everything below is against these exact versions:

| Component | Version |
| :--- | :--- |
| Cluster | Alliance Canada `rorqual`, NVIDIA H100 80 GB HBM3 (sm_90) |
| Modules | `StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.10.0 python/3.11` |
| Python | 3.11.5 |
| PyTorch | 2.5.1 (CUDA 12.2) |
| **mmcv** | **2.2.0, built from source, editable** |
| mmengine | 0.10.5 |
| mmdet | 3.3.0 |
| mmdetection3d | 1.4.0 |
| chamferdist | 1.0.3 |

```bash
git clone https://github.com/open-mmlab/mmcv.git ~/mmcv
cd ~/mmcv
git checkout v2.2.0
```

Work on the `v2.2.0` tag, not `main`. Every line number and context below is relative to
that tag.

---

## 3. Changes required

```console
$ git diff --stat
 mmcv/ops/csrc/pytorch/cuda/voxelization_cuda.cu | 118 ++++++++++++++++
 mmcv/ops/csrc/pytorch/pybind.cpp                |  31 ++++-
 mmcv/ops/csrc/pytorch/voxelization.cpp          |  40 ++++++
 mmcv/ops/voxelize.py                            | 174 ++++++++++++++++++++++++
 4 files changed, 357 insertions(+), 6 deletions(-)
```

### 3.0 The design: adding a new function, and there will be no backward kernel
Two decisions differ from the 1.x patch
**The changes are additive.** `hard_voxelize_forward` is left behaviourally identical and
a *second* entry point `hard_voxelize_forward_v2` is added beside it. Ordinary training
and evaluation keep the stock code path; only attack code opts in. Nothing you have
already validated changes.

**There is no backward kernel, and none is needed.** Voxelization does two things:

1. **Assignment**: which voxel each point falls into. This is piecewise constant in the
   point coordinates: its gradient is zero almost everywhere. There is nothing to
   differentiate.
2. **Aggregation**: copying point features into their voxel slots. This is a pure
   gather/scatter and *is* differentiable.

So we do not write a backward pass. We run the existing (non-differentiable) CUDA kernel
purely to compute the **assignment**, have it hand back the mapping tensors it normally
throws away, and then redo the **aggregation** in Python with an out-of-place
`index_put`, which autograd records for free. Stock mmcv already computes those mappings
internally. It just never returns them. The entire patch is, in essence, *making three
existing tensors visible.*

### 3.1 `mmcv/ops/csrc/pytorch/cuda/voxelization_cuda.cu`: the v2 kernel launcher

Adds `HardVoxelizeForwardCUDAKernelLauncher_v2`. It runs the same five kernels as the
stock launcher, in the same order, with one difference: the mapping tensors are
**caller-provided** rather than allocated locally.

```cpp
int HardVoxelizeForwardCUDAKernelLauncher_v2(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel,
    at::Tensor &point_to_pointidx,      // <-- exposed
    at::Tensor &point_to_voxelidx,      // <-- exposed
    at::Tensor &coor_to_voxelidx,       // <-- exposed
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int max_points, const int max_voxels, const int NDim = 3) {

  // 1. dynamic_voxelize_kernel   : point -> voxel coordinate
  // 2. point_to_voxelidx_kernel  : point -> slot within its voxel
  // 3. determin_voxel_num        : voxel coordinate -> voxel index, and the count
  // 4. assign_point_to_voxel     : features into the voxel grid
  // 5. assign_voxel_coors        : voxel coordinates out
  ...
  return voxel_num_int;
}
```

Stock mmcv already computes all three. Its own comment calls them *"temporary
variables"*, allocates them with `-at::ones(...)`, and lets them fall out of scope when
the launcher returns:

```cpp
  // 2. map point to the idx of the corresponding voxel, find duplicate coor
  // create some temporary variables
  auto point_to_pointidx = -at::ones({num_points,}, points.options().dtype(at::kInt));
  auto point_to_voxelidx = -at::ones({num_points,}, points.options().dtype(at::kInt));
```

The v2 launcher takes them as arguments instead, so Python keeps them. That is the entire
substantive change:

| Tensor | Shape | Meaning |
| :--- | :--- | :--- |
| `point_to_pointidx` | `(N,)` | for duplicate-coordinate resolution; `-1` if unused |
| `point_to_voxelidx` | `(N,)` | which **slot** (`0 … max_points-1`) this point occupies |
| `coor_to_voxelidx` | `(N,)` | which **voxel row** this point's coordinate maps to; `-1` if dropped |

Together, `(coor_to_voxelidx[i], point_to_voxelidx[i])` is the exact `(row, slot)`
destination of point `i` in the voxel grid. That is all Python needs to rebuild the grid
differentiably.

Note the launcher returns `voxel_num` as an `int` rather than writing it to an output
tensor, and that points dropped by `max_voxels`/`max_points` clipping keep `-1` in the
mapping tensors, so the Python side must mask on that.

### 3.2 `mmcv/ops/csrc/pytorch/voxelization.cpp`: the dispatcher

Declares the launcher and adds the `hard_voxelize_forward_v2` dispatcher, which unpacks
`voxel_size`/`coors_range` into `std::vector<float>` and fills the `voxel_num` tensor from
the launcher's return value:

```cpp
void hard_voxelize_forward_v2(const at::Tensor &points, ..., at::Tensor &voxel_num,
                              at::Tensor &point_to_pointidx,
                              at::Tensor &point_to_voxelidx,
                              at::Tensor &coor_to_voxelidx,
                              const int max_points, const int max_voxels,
                              const int NDim = 3, const bool deterministic = true) {
  ...
  if (points.device().is_cuda()) {
    int voxel_num_int = HardVoxelizeForwardCUDAKernelLauncher_v2(...);
    voxel_num.fill_(voxel_num_int);
  } else {
    AT_ERROR("hard_voxelize_forward_v2 is only implemented for CUDA");
  }
}
```

**CUDA only.** There is deliberately no CPU fallback, because attacks run on GPU. `hard_voxelize_forward`
keeps its CPU implementation untouched.

### 3.3 `mmcv/ops/csrc/pytorch/pybind.cpp`: the binding

Declares `hard_voxelize_forward_v2` and binds it so it appears on `mmcv._ext`:

```cpp
m.def("hard_voxelize_forward_v2", &hard_voxelize_forward_v2,
      "hard voxelize forward v2 with gradient support", py::arg("points"),
      py::arg("voxel_size"), py::arg("coors_range"), py::arg("voxels"),
      py::arg("coors"), py::arg("num_points_per_voxel"), py::arg("voxel_num"),
      py::arg("point_to_pointidx"), py::arg("point_to_voxelidx"),
      py::arg("coor_to_voxelidx"), py::arg("max_points"),
      py::arg("max_voxels"), py::arg("NDim") = 3,
      py::arg("deterministic") = true);
```

`NDim` and `deterministic` also gain defaults on the existing `hard_voxelize_forward`
binding. That is the only edit to stock behaviour and it is source-compatible: every
existing call site passes both explicitly.

### 3.4 `mmcv/ops/voxelize.py`: the Python side

The original module body is kept inside a `'''...'''` block and a rewrite is
appended below it. The rewrite replaces the `_Voxelization(Function)` class with a plain
function thus no `autograd.Function`, so autograd tracks the tensor ops directly instead of
being told to stop at a `backward` that does not exist:

```python
# We remove 'Function' because we want Autograd to track ops automatically
def _voxelization_func(points, voxel_size, coors_range, max_points=35,
                       max_voxels=20000, deterministic=True,
                       differentiable=False):
```

With `differentiable=False` the function is the stock hard-voxelize path. With
`differentiable=True` it calls `_ext.hard_voxelize_forward_v2` and rebuilds the grid:

```python
vn = voxel_num.item()

# Differentiable scatter reconstruction
valid_mask = (point_to_voxelidx > -1) & (coor_to_voxelidx > -1)
points_valid = points[valid_mask]

voxels_new = torch.zeros((max_voxels, max_points, points.size(1)),
                         device=points.device, dtype=points.dtype)
voxels_new_flat = voxels_new.view(-1, points.size(1))
flat_indices = (coor_to_voxelidx[valid_mask] * max_points
                + point_to_voxelidx[valid_mask]).long()
voxels_new_flat[flat_indices] = points_valid       # scatter the ORIGINAL points
voxels_new = voxels_new_flat.view(max_voxels, max_points, points.size(1))

return voxels_new[:vn], coors[:vn], num_points_per_voxel[:vn]
```

The buffer the C++ op wrote into is discarded. What is returned is built by indexing the
caller's `points` tensor, so it carries a graph back to it.

> ### ⚠ Known limitation: `Voxelization.forward` does not expose the flag
>
> `Voxelization.forward` calls `voxelization(...)` with six positional arguments, so
> `differentiable` always falls back to `False`. Nothing in the codebase passes
> `differentiable=True`.
>
> This is known choice since a model built from a config gets stock behaviour, which is what
> you want for evaluation but it means **the `differentiable=True` branch is not the
> code path our attacks actually use.** Attack scripts call
> `_ext.hard_voxelize_forward_v2` directly and do the scatter themselves, which also lets
> them batch and control the coordinate ordering per model. See
> [VOXELIZATION_MMCV2_USAGE.md](VOXELIZATION_MMCV2_USAGE.md).
>
> If you want the flag to work end-to-end, thread it through `Voxelization.__init__` and
> `forward`. Note that `index_put` with `accumulate=False` (the default) is only
> deterministic when the indices are unique. That holds here, `point_to_voxelidx_kernel`
> gives each point within a voxel its own slot, so every `(row, slot)` pair is distinct
> but it is a property of this particular mapping, and not a general guarantee.

---

## 4. Build

Build on a **GPU node**, `nvcc` needs to be present and the install check exercises CUDA
ops. Submit it, do not run it on a login node. In my case, the environment was named centerpoint, so you may name it however you like. 

```bash
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.10.0 python/3.11
source ~/centerpoint/bin/activate

cd ~/mmcv

# a stale _ext will shadow the new one and hide the rebuild
pip uninstall -y mmcv mmcv-full
rm -rf build/ dist/ *.egg-info
find . -name "*.o" -delete
find . -name "*.so" -delete

export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="9.0"          # H100. Use "8.0;9.0" to also cover A100.
export MAX_JOBS=${SLURM_CPUS_PER_TASK:-4}

python setup.py clean --all
python setup.py build_ext --force --inplace
pip install -v -e .
```

Notes:

* `TORCH_CUDA_ARCH_LIST` must include your target arch or the kernels are JIT-recompiled
  at first use, or fail outright. `9.0` is H100; `8.0` is A100. Our runtime job scripts
  export `"8.0;9.0"` so the same tree runs on both.
* Compiling all of mmcv's ops takes roughly 30–45 min at `MAX_JOBS=8`. Request an hour.
* `numpy<2` is required if you load the `opencv/4.10.0` module, otherwise the import
  raises `AttributeError: _ARRAY_API not found`.

---

## 5. Verification

### 5.1 Both symbols present

```bash
python -c "
from mmcv import _ext
print('hard_voxelize_forward   :', hasattr(_ext, 'hard_voxelize_forward'))
print('hard_voxelize_forward_v2:', hasattr(_ext, 'hard_voxelize_forward_v2'))
"
```

Both must print `True`. If `_v2` is missing the rebuild did not take, check for a stale
`.so` in `site-packages` shadowing the editable install.

### 5.2 The gradient is actually alive

Symbol presence proves the build; it does not prove the gradient. This is the check that
matters, and it is the one whose absence cost us months:

```python
import torch
from mmcv import _ext

device = 'cuda'
voxel_size = [0.075, 0.075, 0.2]
coors_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
max_points, max_voxels, NDim = 10, 120000, 3

# Points inside the detector's range, packed into a small enough volume that
# voxels actually collect several points each. This matters: spread over the
# full 108 x 108 x 8 m range, 20k points land ~1 per voxel, every slot index is
# 0, and the multi-point path is never exercised at all.
#
# 1.5 x 1.5 x 2.0 m at this voxel size is 20 x 20 x 10 = 4000 voxels, so ~5
# points per voxel on average. Voxels that draw more than max_points=10 drop
# the surplus -- that is correct behaviour, and the `kept` mask below is what
# handles it.
N = 20000
xy = torch.rand(N, 2, device=device) * 1.5 - 0.75     # +-0.75 m
z = torch.rand(N, 1, device=device) * 2.0 - 1.0       # [-1, 1], inside [-5, 3]
feat = torch.rand(N, 2, device=device)                # intensity, time
points = torch.cat([xy, z, feat], dim=1).requires_grad_(True)

voxels = torch.zeros((max_voxels, max_points, 5), device=device)
coors = torch.zeros((max_voxels, NDim), dtype=torch.int32, device=device)
num_pts = torch.zeros((max_voxels,), dtype=torch.int32, device=device)
voxel_num = torch.zeros((1,), dtype=torch.int32, device=device)
p2p = -torch.ones((N,), dtype=torch.int32, device=device)
p2v = -torch.ones((N,), dtype=torch.int32, device=device)
c2v = -torch.ones((N,), dtype=torch.int32, device=device)

_ext.hard_voxelize_forward_v2(
    points.contiguous(),
    torch.tensor(voxel_size, dtype=torch.float32),      # CPU tensors
    torch.tensor(coors_range, dtype=torch.float32),
    voxels, coors, num_pts, voxel_num, p2p, p2v, c2v,
    max_points, max_voxels, NDim, True)

# the buffer the kernel wrote is NOT differentiable -- this is the bug, reproduced
assert voxels.grad_fn is None

# rebuild it with a graph
row, slot = c2v.long(), p2v.long()
kept = (row >= 0) & (slot >= 0) & (slot < max_points)
nvox = int(voxel_num.item())
rebuilt = torch.zeros((nvox, max_points, 5), device=device).index_put(
    (row[kept], slot[kept]), points[kept])

assert rebuilt.grad_fn is not None, "no graph -- the scatter did not attach"
rebuilt.sum().backward()

g = points.grad.norm().item()
print(f"voxels kept : {nvox}")
print(f"points kept : {int(kept.sum())} / {N}")
print(f"|d(sum)/d(points)| = {g:.6f}")
assert g > 0, "GRADIENT IS DEAD"
print("OK: gradient flows through voxelization")
```

A non-zero norm here is the property every attack in this repository depends on.

---

## 6. Next

[VOXELIZATION_MMCV2_USAGE.md](VOXELIZATION_MMCV2_USAGE.md), how `_ext.hard_voxelize_forward_v2`
is called from the attack scripts, worked through end to end on FocalFormer3D.
