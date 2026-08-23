# Gradient Extraction

How we extract `dL/df` which is the gradient of the detection loss with respect to an
intermediate feature map for every frame of a validation split. These gradients are the
input to the adversarial attacks in
[VOXELIZATION_MMCV2_USAGE.md](VOXELIZATION_MMCV2_USAGE.md).

As an example, we work through on **FocalFormer3D**; the mechanism is the same for BEVFusion, PillarNeSt,
CenterPoint and PointPillars, and [§7](#7-porting-to-another-model) covers what changes.

> **Important fact:** the saved tensors are **L2-normalised direction
> fields, not `dL/df` magnitudes** ([§5](#5-normalisation)).

---

## 1. The problem
This is an implementation of https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.70011 for our evaluation, their source code hasn;t been released, thus this code was compiled to the best of our abilities and may not represent their intentions or ideas fully. 
An attack needs to know which way to push a feature map to hurt the detector. That means a
**backward pass through a detection loss**, which means **ground-truth annotations**.

So gradient extraction runs mmengine's **train loop**, not the test loop, on the
validation split, with the weights frozen. Four things have to be arranged:

| need | mechanism |
| :--- | :--- |
| GT annotations for the val split | `train_dataloader` with `ann_file` pointing at the **val** infos |
| A loss, and a backward pass | the normal `EpochBasedTrainLoop`, one epoch |
| Weights that never change | [`NoOpOptimizer`](#3-freezing-the-weights) |
| The intermediate gradient, per frame | [`FocalFormerGradientHook`](#4-the-hook) |

### 1.1 Augmentation must be off

The train pipeline exists to augment. Here it must not: an attack is computed against a
specific point cloud, so the gradient must correspond to **the frame as the evaluator will
see it**. Flip, rotate, scale and GT-paste all have to go.

```python
custom_hooks = [
    dict(type='DisableObjectSampleHook', disable_after_epoch=0),   # kill GT-paste at once
    ...
]
```

and the train pipeline is written without `GlobalRotScaleTrans` / `RandomFlip3D`. **Check
this by reading the pipeline your config actually resolves to**, not the one you think you
inherited. This is the easiest thing in the whole setup to get silently wrong, because a
gradient extracted through an augmented frame does not look wrong, it just is.

---

## 2. Config

Full example: `projects/configs/focalformer3d/FocalFormer3D_L_grad_extract.py`.

```python
custom_imports = dict(
    imports=[
        'projects.mmdet3d_plugin',
        'projects.mmdet3d_plugin.hooks.focalformer_gradient_hook',
        'projects.mmdet3d_plugin.hooks.noop_optimizer',
    ],
    allow_failed_imports=False)

custom_hooks = [
    dict(type='DisableObjectSampleHook', disable_after_epoch=0),
    dict(
        type='FocalFormerGradientHook',
        target_layer='neck',              # <-- see section 6
        save_path=gradients_output_dir,
        normalize=True,                   # <-- see section 5
        save_interval=100),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='NoOpOptimizer', lr=0.0))

epoch_num = 1
param_scheduler = [dict(type='LinearLR', start_factor=1.0, begin=0, end=epoch_num)]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=999),   # never write one
    ...)
```

> **Do not inherit `param_scheduler` from a training config.** Real schedules anneal
> momentum as well as LR, and a momentum scheduler against `NoOpOptimizer` dies with
> `optimizer must support momentum when using momentum scheduler`. Declare the flat
> `LinearLR(start_factor=1.0)` above explicitly.

---

## 3. Freezing the weights

### Please note that adversarial attack and training refer to the dataset, in our case LiDAR point clouds, not the model weights. We only change point clouds using various methods, this one is using gradients of the feature map produced by the point cloud. Thus if we want deterministic gradient tensors, our model weights need to be frozen. 

`projects/mmdet3d_plugin/hooks/noop_optimizer.py` is the optimizer that "freezes" model weights:

```python
@OPTIMIZERS.register_module()
class NoOpOptimizer(Optimizer):
    """Optimizer that computes gradients but never updates weights."""

    def __init__(self, params, lr=0.0, **kwargs):
        super().__init__(params, dict(lr=lr))

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # No weight updates -- gradients remain for extraction
        return loss
```

This is the right mechanism in our opinion, since. the obvious
alternative which is `param.requires_grad = False` on everything **would break extraction**:
autograd would stop building the graph through the frozen parameters, and there would be
no gradient at the feature map either. We need the backward pass to happen in full and
only the *update* to be suppressed. `step()` returning without touching the parameters
does exactly that.

These conditions are all in the config:

1. `NoOpOptimizer.step()` is a no-op, weights cannot move.
2. `lr=0.0`, so even a mistakenly-substituted real optimizer would be a no-op.
3. `CheckpointHook(interval=999)` over a 1-epoch run therefore no model intermediate checkpoints are ever written to disk.
---

## 4. The hook

`projects/mmdet3d_plugin/hooks/focalformer_gradient_hook.py`

Three temporal points:

| method | does |
| :--- | :--- |
| `before_train` | resolve `target_layer` → module path, find the module, register hooks |
| `after_train_iter` | take the captured gradient, normalise, save as `{sample_id}_grad.pt` |
| `after_train` | remove the handles, report how many files landed |

### 4.1 Why it is not just a backward hook

The obvious implementation is `register_full_backward_hook` on the target module. That is
registered, but for the layer we actually use it is **not the mechanism that works**.

`pts_neck` is a `SECONDFPN`, and it returns a **list** of tensors, instead of just one tensor. Module
backward hooks do not reliably deliver gradients for list outputs and this I found out trhough trial and error. So the hook installs a
**forward** hook, traverses the outputs, and attaches a per-tensor gradient hook to each:

```python
def _fwd_hook(self, module, inputs, output):
    self._activation = None
    tensors = output if isinstance(output, (list, tuple)) else [output]
    for i, t in enumerate(tensors):
        if torch.is_tensor(t):
            ...
            if t.requires_grad:
                t.register_hook(lambda g, idx=i: self._save_tensor_grad(g, idx))

def _save_tensor_grad(self, g, idx):
    g = g.detach().clone()
    if self._capturer.gradient is None:
        self._capturer.gradient = g
    else:
        self._capturer.gradient = torch.cat([self._capturer.gradient, g], dim=1)
```

Tensor hooks fire per output, so the pieces are **concatenated along dim 1** which is the channel
axis. That is what turns `SECONDFPN`'s two `[B, 256, 180, 180]` outputs into the single
`[B, 512, 180, 180]` the attack expects, and why the attack asserts on that shape.

> **Caveat for non-`neck` targets.** `before_train` calls
> `register_full_backward_hook` unconditionally *and again* when
> `target_layer != 'neck'`, so for those targets the capturer is registered twice and the
> first handle is never removed. The module hook **assigns** `self.gradient` while the
> tensor hooks **concatenate** into it, so the two paths interact and the result depends on
> execution order. `target_layer='neck'` is the path we validated end-to-end (5,980 frames,
> shape `[1, 512, 180, 180]`); treat anything else as needing verification before you trust
> it.

### 4.2 One file per frame

```python
meta = data_samples[0].metainfo
sample_id = _extract_sample_id(meta, batch_idx)
torch.save(grad, os.path.join(self.save_path, f'{sample_id}_grad.pt'))
```

`_extract_sample_id` derives the name from the LiDAR filename with `.pcd.bin` / `.bin` /
`.npy` stripped, falling back to `token` / `sample_idx`. That is what lets the attack match
a gradient to a point cloud by name later, and it is why extraction must run at
**`batch_size=1`** thus the id comes from `data_samples[0]`, so a larger batch would label the
whole batch's gradient with the first frame's name.

---

## 5. Normalisation

```python
grad = self._capturer.gradient.detach().cpu().to(torch.float32)
if self.normalize:
    grad = F.normalize(grad, p=2.0, dim=1)
```

`dim=1` is the **channel** axis. Every spatial cell `(b, :, h, w)` is scaled to unit L2
norm independently.

**This is the most consequential line in the pipeline, and it is easy to miss.**

* The saved tensors are **directions, not magnitudes**. All per-cell within the feature map information about
  *how strongly* the loss responds is discarded, and only *which way* we want to suppress or hgihglight features survive.
* So you **cannot compare gradient magnitudes** across cells, frames or models. 
* And because the magnitude is gone, the **sign convention of the attack objective is an
  empirical question. `loss_sign` has to be resolved by
  running it, with a `loss_sign=0` control to prove the term contributes at all, see
  [VOXELIZATION_MMCV2_USAGE.md §7.2](VOXELIZATION_MMCV2_USAGE.md#72-the-loss_sign0-control).

`normalize=True` **is** channel
normalisation; Set `normalize=False` if you want raw `dL/df`, and expect
nothing downstream to be affected for it.

---

## 6. Choosing the layer

`target_layer` is a config string, resolved through a small alias map:

```python
self._module_map = {
    'backbone_block0': 'pts_backbone.blocks.0',
    'neck':            'pts_neck',
}

def _resolve_module_name(self):
    return self._module_map.get(self.target_layer, self.target_layer)
```

`.get(..., self.target_layer)` is the useful part: **an unrecognised string is passed
through as a literal module path**, matched against `model.named_modules()`. So any module
in the network is addressable without touching the hook:

```python
dict(type='FocalFormerGradientHook', target_layer='neck')                # alias
dict(type='FocalFormerGradientHook', target_layer='pts_middle_encoder')  # literal path
dict(type='FocalFormerGradientHook', target_layer='pts_backbone.blocks.1')
```

A name that matches nothing raises `RuntimeError: Module '<name>' not found` in
`before_train`, before any compute is spent, which is the one failure in this pipeline
that is loud.

**The layer must match what the attack expects.** The attack reconstructs the encoder up to
the same point and asserts the shapes agree:

| model | target | gradient shape (nuScenes) |
| :--- | :--- | :--- |
| FocalFormer3D-L / -LC | `neck` → `pts_neck` | `[B, 512, 180, 180]` |
| FocalFormer3D | `backbone_block0` → `pts_backbone.blocks.0` | `[B, 128, 180, 180]` |
| BEVFusion | `pts_neck` | `[B, 512, 180, 180]` |
| PillarNeSt | `pts_middle_encoder` | depends on `output_shape` |

Waymo differs from nuScenes on the same model, because range and voxel size differ:
`[-54, 54] / 0.075 → 1440 → /8 = 180` for nuScenes against
`[-75.2, 75.2] / 0.1 → 1504 → /8 = 188` for Waymo.

---

## 7. Porting to another model

The hook is not FocalFormer-specific, it resolves modules by name off `named_modules()`,
so it works on any mmdet3d detector. What you supply per model:

1. **A module path** that exists in that model (`target_layer`).
2. **A matching encoder reconstruction** in the attack script, up to the same layer.
3. **A train-mode config on the val split** with augmentation off and `NoOpOptimizer`.

PillarNeSt and CenterPoint hook `pts_middle_encoder` rather than the neck, because their
attack scripts rebuild the encoder to that point. The choice is because of where the attack can reconstruct a differentiable path back to the
points, so it is dependent on the architecture of each of these models.

---

## 8. Running it

Gradient extraction is a SLURM job. It writes **one file per frame**, about 5,980 of them for
nuScenes val which is why it must stage through node-local SSD and ship a single tar. 
The scripts and the reasoning are in
[`mmdet3d_v1.4_files/slurm/`](../mmdet3d_v1.4_files/slurm/README.md).

```bash
sbatch mmdet3d_v1.4_files/slurm/single_gpu/extract_grad_focalformer_l_nus.sh                 # quick
sbatch --export=ALL,MODE=full mmdet3d_v1.4_files/slurm/single_gpu/extract_grad_focalformer_l_nus.sh
```

**Run the quick mode first.** It extracts a spread of ~10 frames to test out without wasting resources.

### 8.1 Expected counts

For nuScenes val, 6,019 frames go in and **5,980 gradients** come out. The 39 missing are dropped by the pipeline's own filters:

| stage | dropped |
| :--- | ---: |
| frames with no instances | 3 |
| `use_valid_flag` | 7 |
| `ObjectNameFilter` | 25 |
| `ObjectRangeFilter` | 14 |
| **total** | **39** |

---

## 9. Things to look out for

| Symptom | Cause |
| :--- | :--- |
| `Module '<name>' not found` | Bad `target_layer`. |
| `optimizer must support momentum when using momentum scheduler` | Inherited `param_scheduler` from a training config. [§2](#2-config). |
| Gradient files written, attack has no effect | Normalisation is not the problem, check the attack's voxelization graph. [VOXELIZATION_MMCV2_USAGE.md §7](VOXELIZATION_MMCV2_USAGE.md#7-confirming-the-gradient-is-alive). |
| Shape mismatch in the attack | `target_layer` disagrees with the encoder the attack rebuilds, or nuScenes-vs-Waymo grid size. [§6](#6-choosing-the-layer). |
| Every gradient named after the same frame | `batch_size > 1`. [§4.2](#42-one-file-per-frame). |
| Far fewer than 5,980 files | Check augmentation and the ann_file. [§8.1](#81-expected-counts). |
| Job dies partway with the output lost | Gradients were not staged to `$SLURM_TMPDIR`, or the tar was never shipped. [`slurm/README.md`](../mmdet3d_v1.4_files/slurm/README.md). |

---

## See also

* [`mmdet3d_v1.4_files/slurm/README.md`](../mmdet3d_v1.4_files/slurm/README.md), the job scripts, single- and multi-GPU
* [VOXELIZATION_MMCV2_USAGE.md](VOXELIZATION_MMCV2_USAGE.md), what consumes these gradients
* [SETTING_UP_FOCALFORMER3D_MMDET14.md](SETTING_UP_FOCALFORMER3D_MMDET14.md), getting the model running first
