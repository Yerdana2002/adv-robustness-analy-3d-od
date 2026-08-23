# Using the Differentiable Voxelization in an Attack Script
## NOTE: AI WAS USED TO COMPILED THE DOCUMENTATION AND IT WAS READ BY A HUMAN BEFORE POSTING

Prerequisite: [VOXELIZATION_MMCV2.md](VOXELIZATION_MMCV2.md). That document explains why
`_ext.hard_voxelize_forward_v2` exists and how to build it. This one shows how to call it,
worked end to end on **FocalFormer3D** on nuScenes.

---

## 1. Which entry point to call

The patched `mmcv/ops/voxelize.py` exposes a `differentiable=True` flag, but
`Voxelization.forward` never passes it (see the limitation note in
[VOXELIZATION_MMCV2.md §3.4](VOXELIZATION_MMCV2.md#34-mmcvopsvoxelizepy--the-python-side)).
**Attack code calls the extension directly:**

```python
from mmcv import _ext
```

Calling `_ext` directly rather than going through `Voxelization`
allows three things the stock module cannot provide:

1. **Batching.** The op is single-sample. Attacks run 4 clouds at a time and need
   per-sample voxel grids concatenated with a batch index column.
2. **Coordinate ordering.** The op returns `coors` as `(z, y, x)`. Different detectors
   declare different `sparse_shape` conventions, so the caller has to decide whether to
   permute. Getting this wrong moves geometry..
3. **Control of the scatter.** The rebuild below is what carries the gradient, and it
   needs the mapping tensors, which `Voxelization` discards.

### 1.1 Where it is used

| File | Model |
| :--- | :--- |
| `projects/mmdet3d_plugin/models/attack_focalformer_nus.py` | FocalFormer3D-L / -LC, nuScenes |
| `projects/mmdet3d_plugin/models/attack_focalformer_waymo.py` | FocalFormer3D, Waymo |
| `projects/mmdet3d_plugin/models/attack_bevfusion_nus.py` | BEVFusion, nuScenes |
| `mmdet3d/models/centerpoint_nuscenes_batch.py` | CenterPoint, nuScenes |
| `mmdet3d/models/attack_pillarnest_nus_attach.py` | PillarNeSt, nuScenes |
| `mmdet3d/models/pointpillars_nuscenes_attack_batched.py` | PointPillars, nuScenes |

All follow the same three steps. FocalFormer3D is used as the example below because it
does its own voxelization inside the detector rather than delegating to
`data_preprocessor`, which makes the encoder path explicit.

---

## 2. The recipe

> **Step 1** — call `_ext.hard_voxelize_forward_v2` to get the voxel **assignment** plus
> the three mapping tensors.
> **Step 2** — throw away the `voxels` buffer it filled and rebuild the grid with
> `index_put` from the original `points` tensor. *This is the step that carries the
> gradient.*
> **Step 3** — run the rebuilt grid through VFE → middle encoder → backbone → neck,
> and backprop to `points`.

---

## 3. Step 1 — call the extension

`attack_focalformer_nus.py`, `hard_voxelize_v2()`:

```python
from mmcv import _ext

def hard_voxelize_v2(points, voxel_size, coors_range,
                     max_points=10, max_voxels=120000,
                     NDim=3, deterministic=True) -> dict:
    """Single sample voxelization using mmcv C++ extension."""
    device = points.device
    num_points, num_features = points.shape[0], points.shape[1]

    # NOTE: voxel_size and coors_range must be CPU float32 tensors.
    # The dispatcher reads them with .data_ptr<float>() on the host.
    voxel_size_tensor  = torch.tensor(voxel_size,  dtype=torch.float32, device='cpu')
    coors_range_tensor = torch.tensor(coors_range, dtype=torch.float32, device='cpu')

    # Outputs, preallocated on device -- the op writes into these in place.
    voxels    = torch.zeros((max_voxels, max_points, num_features),
                            dtype=points.dtype, device=device)
    coors     = torch.zeros((max_voxels, NDim), dtype=torch.int32, device=device)
    num_points_per_voxel = torch.zeros((max_voxels,), dtype=torch.int32, device=device)
    voxel_num = torch.zeros((1,), dtype=torch.int32, device=device)

    # The three mapping tensors. Must be initialised to -1: the kernels leave
    # dropped points untouched, and -1 is how you tell "dropped" from "row 0".
    point_to_pointidx = -torch.ones((num_points,), dtype=torch.int32, device=device)
    point_to_voxelidx = -torch.ones((num_points,), dtype=torch.int32, device=device)
    coor_to_voxelidx  = -torch.ones((num_points,), dtype=torch.int32, device=device)

    _ext.hard_voxelize_forward_v2(
        points.contiguous(),                    # .contiguous() is required
        voxel_size_tensor.contiguous(),
        coors_range_tensor.contiguous(),
        voxels, coors, num_points_per_voxel, voxel_num,
        point_to_pointidx, point_to_voxelidx, coor_to_voxelidx,
        max_points, max_voxels, NDim, deterministic
    )

    actual_voxel_num = voxel_num.item()
    return {
        'voxels': voxels[:actual_voxel_num],                 # NOT differentiable
        'coors': coors[:actual_voxel_num],
        'num_points_per_voxel': num_points_per_voxel[:actual_voxel_num],
        'voxel_num': actual_voxel_num,
        'point_to_pointidx': point_to_pointidx,
        'point_to_voxelidx': point_to_voxelidx,
        'coor_to_voxelidx': coor_to_voxelidx,
    }
```

The dict's `voxels` entry is returned for inspection only. **Do not feed it to the
model** — it is the in-place buffer with `grad_fn=None`, and using it is exactly the
silent failure this whole document exists to prevent.

Argument gotchas, each of which fails at the C++ boundary rather than gracefully:

* `voxel_size` / `coors_range` must be **CPU** `float32` tensors, not lists and not CUDA
  tensors.
* `points` must be `.contiguous()`.
* The three mapping tensors must be `int32` and pre-filled with `-1`.
* `max_voxels` is an int here. FocalFormer3D's config declares
  `max_voxels=(120000, 160000)` — a `(train, test)` pair — so the caller takes `[0]`.

---

## 4. Step 2 — rebuild the grid so it carries a gradient

This is the crux. `attack_focalformer_nus.py`, inside `batched_voxelize()`:

```python
for batch_idx, points in enumerate(points_list):
    voxel_result = hard_voxelize_v2(points=points, voxel_size=voxel_size,
                                    coors_range=coors_range,
                                    max_points=max_points, max_voxels=max_voxels)

    coors   = voxel_result['coors']
    num_pts = voxel_result['num_points_per_voxel']

    # Rebuild the voxel tensor so it carries a graph back to `points`.
    #
    # voxel_result['voxels'] is a slice of a torch.zeros buffer that the C++ op
    # wrote into in place. No autograd.Function wraps that call, so it has
    # requires_grad=False and grad_fn=None. Everything downstream (VFE reduce,
    # middle_encoder, backbone, neck, <grad, features>) therefore detaches, and
    # d(adv)/d(points) is EXACTLY zero.
    #
    # The assignment itself stays non-differentiable, which is correct: which
    # voxel a point falls into is piecewise constant and has zero gradient
    # almost everywhere. Only the aggregation needs a graph, and scattering the
    # ORIGINAL points into their slots supplies one.
    #
    # index_put is out-of-place, returning a new tensor with IndexPutBackward
    # attached rather than mutating a buffer that does not require grad.
    row  = voxel_result['coor_to_voxelidx'].long()
    slot = voxel_result['point_to_voxelidx'].long()
    kept = (row >= 0) & (slot >= 0) & (slot < max_points)
    nvox = int(voxel_result['voxel_num'])

    voxels = torch.zeros(
        (nvox, max_points, points.shape[1]),
        dtype=points.dtype, device=device).index_put(
            (row[kept], slot[kept]), points[kept])

    # batch index column: (N, 3) -> (N, 4) as [batch_idx, z, y, x]
    batch_idx_col = torch.full((coors.shape[0], 1), batch_idx,
                               dtype=coors.dtype, device=device)
    coors_with_batch = torch.cat([batch_idx_col, coors], dim=1)
```

Three things to hold onto:

* **`index_put`, not `voxels[row, slot] = points`.** The in-place form mutates a tensor
  that does not require grad and gives you back the same dead tensor. `index_put` is
  out-of-place and returns a new tensor with `IndexPutBackward` attached. This one
  method call is the difference between a working attack and noise.
* **`kept` is not optional.** Points dropped by `max_voxels`/`max_points` clipping keep
  `-1` in the mapping tensors. Without the mask, `-1` indexes the *last* row and corrupts
  it.
* **`coors` is passed through unpermuted here.** BEVFusion's version of this function
  additionally permutes `coors` to `(x, y, z)`, because `BEVFusionSparseEncoder` declares
  `sparse_shape [1440, 1440, 41]` while the op returns `(z, y, x)`. FocalFormer3D's
  `SparseEncoder` uses the other convention. **This is per-model — copying the permute
  across silently rotates the scene.**

---

## 5. Step 3 — forward to the gradient target

FocalFormer3D voxelizes inside the detector, so the attack rebuilds the encoder stack
explicitly from the loaded model's submodules:

```python
class FocalFormer3DEncoder(nn.Module):
    """
    pts_voxel_layer (Voxelization) -> voxels
    pts_voxel_encoder (HardSimpleVFE) -> voxel features
    pts_middle_encoder (SparseEncoder) -> 2D dense BEV
    pts_backbone (SECOND) -> multi-scale 2D features
    pts_neck (SECONDFPN) -> [B, 512, 180, 180]   <-- gradient target
    """

    def forward(self, points_list):
        batch_size = len(points_list)

        # 1. Voxelize -- our differentiable version, NOT self.pts_voxel_layer
        voxels, coors, num_points, voxel_infos = batched_voxelize(
            points_list=points_list,
            voxel_size=self.voxel_size, coors_range=self.point_cloud_range,
            max_points=self.max_num_points, max_voxels=self.max_voxels)

        # 2. VFE (HardSimpleVFE: mean of points per voxel)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)

        # 3. Middle encoder (SparseEncoder: 3D sparse -> 2D dense)
        bev_features = self.middle_encoder(voxel_features, coors, batch_size)

        # 4. Backbone (SECOND)
        backbone_features = self.backbone(bev_features)

        # 5. Neck (SECONDFPN) -> [[B,256,180,180], [B,256,180,180]]
        neck_features = self.neck(backbone_features)
        features = torch.cat(neck_features, dim=1)   # [B, 512, 180, 180]
        return features, voxel_infos
```

Built from the real checkpoint, frozen:

```python
full_model = MODELS.build(cfg.model).to(device)
load_checkpoint(full_model, checkpoint_path, map_location=device)
full_model.eval()

voxel_cfg = cfg.model.pts_voxel_layer
voxel_size        = list(voxel_cfg.voxel_size)          # [0.075, 0.075, 0.2]
point_cloud_range = list(voxel_cfg.point_cloud_range)   # [-54,-54,-5, 54,54,3]
max_num_points    = voxel_cfg.max_num_points            # 10
max_voxels        = voxel_cfg.max_voxels                # (120000, 160000)
if isinstance(max_voxels, (list, tuple)):
    max_voxels = max_voxels[0]

encoder = FocalFormer3DEncoder(
    voxel_encoder=full_model.pts_voxel_encoder,
    middle_encoder=full_model.pts_middle_encoder,
    backbone=full_model.pts_backbone,
    neck=full_model.pts_neck,
    voxel_size=voxel_size, point_cloud_range=point_cloud_range,
    max_num_points=max_num_points, max_voxels=max_voxels,
    target_layer='neck').to(device)
encoder.eval()

for param in encoder.parameters():       # only the POINTS are optimised
    param.requires_grad = False
```

Note the voxelization parameters are read **from the model config**, not hardcoded. The
grid the attack builds must match the grid the detector was trained on, or the sparse
encoder receives coordinates outside its declared `sparse_shape`.

---

## 6. The optimization loop

```python
# adversarial points are the leaves of the graph
adv_points_list = []
for orig_pts in original_points_list:
    adv_pts = orig_pts.clone().detach()
    if init_noise_std > 0:
        noise = torch.normal(0, init_noise_std,
                             size=(orig_pts.shape[0], 3), device=device)
        adv_pts[:, :3] = adv_pts[:, :3] + noise
    adv_pts.requires_grad_(True)
    adv_points_list.append(adv_pts)

optimizer = optim.Adam(adv_points_list, lr=learning_rate)

for it in range(num_iterations):
    optimizer.zero_grad()

    dist_loss = batched_chamfer_distance_bidirectional(
        adv_points_list, original_points_list, chamfer_dist)

    # <-- the differentiable voxelization is inside here
    features, voxel_infos = encoder(adv_points_list)

    # align the neck features with the pre-extracted gradient
    adv_loss   = loss_sign * torch.sum(gradient_batch * features)
    total_loss = adv_loss + dist_weight * dist_loss

    total_loss.backward()          # <-- reaches adv_pts ONLY via index_put
    optimizer.step()

    with torch.no_grad():          # keep points inside the detector's range
        for adv_pts in adv_points_list:
            adv_pts[:, 0].clamp_(point_cloud_range[0], point_cloud_range[3])
            adv_pts[:, 1].clamp_(point_cloud_range[1], point_cloud_range[4])
            adv_pts[:, 2].clamp_(point_cloud_range[2], point_cloud_range[5])
```

`gradient_batch` is `[B, 512, 180, 180]`, extracted in a prior pass by hooking `pts_neck`
during a backward over the detection loss on ground-truth annotations. The attack then
pushes the neck features along that direction while the Chamfer term holds the cloud near
the original geometry.

Invocation:

```bash
python projects/mmdet3d_plugin/models/attack_focalformer_nus.py \
    --cfg projects/configs/focalformer3d/FocalFormer3D_L_v14_grad.py \
    --grads   /path/to/gradients \
    --results /path/to/save/adversarial \
    --checkpoint /path/to/FocalFormer3D_L_ep6_converted.pth \
    --data_root  /path/to/nuscenes/ \
    --batch_size 4 --iterations 40 --lr 0.01 --dist_weight 1.0 \
    --init_noise_std 0.3 --loss_sign 1.0 --target_layer neck
```

---

## 7. Confirming the gradient is alive

**Do not skip this.** A dead voxelization gradient produces no error, the loop runs, the
losses look reasonable, and the saved clouds are pure initialization noise.

### 7.1 Watch `grad_norm` during the run

The loop prints it every 10 iterations:

```python
grad_norms = [p.grad.norm().item() if p.grad is not None else 0
              for p in adv_points_list]
print(f"    Iter {it:3d}: adv={adv_loss.item():.4f}, dist={dist_loss.item():.4f}, "
      f"total={total_loss.item():.4f}, grad_norm={sum(grad_norms)/len(grad_norms):.6f}")
```

`grad_norm` identically `0.000000` means the graph is broken. It will still be non-zero
if only the Chamfer term is connected, so this check is necessary but not sufficient —
which is why §7.2 exists.

### 7.2 The `loss_sign=0` control

The decisive test. `loss_sign=0` deletes the adversarial term and leaves everything else —
initialization noise, Chamfer, Adam, clamping — identical:

```bash
sbatch --export=ALL,LOSS_SIGN=0.0 attack_focalformer_l_nuscenes.sh
```

Evaluate both sets. If the control reproduces the "attack" you have measured your noise,
not your attack.

---

## 8. Porting to another detector

The three steps are model-independent. What changes:

| Problem | Solution |
| :--- | :--- |
| Encoder path | Read the detector's `extract_feat` / `extract_pts_feat` and mirror it. FocalFormer3D voxelizes in the detector; CenterPoint delegates to `data_preprocessor`. |
| `coors` axis order | Compare the op's `(z, y, x)` against the middle encoder's declared `sparse_shape`. Permute only if they disagree. |
| Voxel parameters | Read from `cfg.model.pts_voxel_layer` — never hardcode. |
| Gradient target | Must match the shape of the extracted gradient tensor. The script asserts this and aborts on mismatch. |
| `max_voxels` | Configs often give a `(train, test)` tuple; take one element. |

Whatever the model, run §7.2 to verify that differentiable backprop works as it should.
