"""
Batched Adversarial Attack for BEVFusion on nuScenes
=====================================================
Companion to attack_focalformer_nus.py. Same recipe: 40 iterations, Adam at
lr 0.01, unidirectional Chamfer regulariser, differentiable voxelization
through the custom mmcv `_ext`, final-iteration points written out.

Target layer is `pts_middle_encoder` -> [B, 256, 180, 180].

WHY middle_encoder AND NOT neck (as FocalFormer3D uses)
-------------------------------------------------------
The two models fuse at different depths:

    FocalFormer3D   middle_encoder -> backbone -> neck   all LiDAR-only,
                    fusion happens afterwards in imgpts_neck
    BEVFusion       fusion_layer sits BEFORE pts_backbone, so pts_backbone
                    and pts_neck already carry camera content on the L+C model

So `neck` is not the same quantity across the two models, nor between
BEVFusion L+C and L-only. `pts_middle_encoder` is the last purely-LiDAR tensor
in the graph and means the same thing in both BEVFusion variants, which is what
makes the L+C vs L-only comparison meaningful. It also halves the gradient
storage (33.2 vs 66.4 MB/sample).

The practical consequence for this file: the encoder below stops at the middle
encoder, so pts_backbone, pts_neck, fusion_layer and the entire Swin-T camera
branch are never built or run. That is roughly an order of magnitude less work
per iteration than the FocalFormer equivalent.

WHY THE MODEL'S OWN FORWARD CANNOT BE USED
-------------------------------------------
BEVFusion.voxelize is decorated @torch.no_grad() (bevfusion.py:175) and calls
BEVFusion's own voxel_layer.so, not mmcv's _ext. Gradients cannot reach the
points through the shipped path at all, so voxelization is re-implemented here
against `_ext.hard_voxelize_forward_v2` -- the same custom differentiable build
attack_focalformer_nus.py uses. Nothing in mmcv is modified.

Two details that must match the model exactly or the features will not
correspond to the extracted gradients:

  voxelize_reduce -- BEVFusion does NOT call pts_voxel_encoder in
    extract_pts_feat. It mean-reduces inline: feats.sum(1) / sizes
    (bevfusion.py:196-199). HardSimpleVFE computes the identical mean, but the
    inline form is replicated here so the correspondence is obvious.

  max_voxels -- voxelize_cfg gives [120000, 160000] = (train, test), and
    Voxelization.forward selects on self.training. The gradient hook forces
    model.eval(), so extraction ran with 160000; this script defaults to the
    same index. Change --max_voxels_idx only if extraction ran with
    eval_mode=False.

Usage:
  python attack_bevfusion_nus.py \
      --cfg projects/configs/bevfusion/bevfusion_lidar-cam_grad_extract.py \
      --grads /scratch/.../gradients_lidar-cam \
      --results /scratch/.../adv_points_lidar-cam \
      --checkpoint /path/to/bevfusion_lidar-cam_mmcv_spconv.pth \
      --batch_size 4 --iterations 40 --lr 0.01 --dist_weight 1.0
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.getcwd())

from chamferdist import ChamferDistance                         # noqa: E402
from mmengine.config import Config                              # noqa: E402
from mmengine.registry import init_default_scope                # noqa: E402
from mmengine.runner import Runner, load_checkpoint             # noqa: E402

from mmdet3d.registry import MODELS                             # noqa: E402

from mmcv import _ext                                           # noqa: E402
print(f'DEBUG: _ext loaded from: {_ext}', flush=True)

import faulthandler                                             # noqa: E402
faulthandler.enable()

init_default_scope('mmdet3d')


# =============================================================================
# Differentiable Voxelization (identical to the CenterPoint/FocalFormer path)
# =============================================================================

def hard_voxelize_v2(
    points: torch.Tensor,
    voxel_size: list,
    coors_range: list,
    max_points: int = 10,
    max_voxels: int = 160000,
    NDim: int = 3,
    deterministic: bool = True,
) -> dict:
    """Single-sample voxelization via the custom mmcv C++ extension."""
    device = points.device
    num_points = points.shape[0]
    num_features = points.shape[1]

    voxel_size_tensor = torch.tensor(
        voxel_size, dtype=torch.float32, device='cpu')
    coors_range_tensor = torch.tensor(
        coors_range, dtype=torch.float32, device='cpu')

    voxels = torch.zeros((max_voxels, max_points, num_features),
                         dtype=points.dtype, device=device)
    coors = torch.zeros((max_voxels, NDim), dtype=torch.int32, device=device)
    num_points_per_voxel = torch.zeros((max_voxels, ),
                                       dtype=torch.int32, device=device)
    voxel_num = torch.zeros((1, ), dtype=torch.int32, device=device)

    point_to_pointidx = -torch.ones((num_points, ),
                                    dtype=torch.int32, device=device)
    point_to_voxelidx = -torch.ones((num_points, ),
                                    dtype=torch.int32, device=device)
    coor_to_voxelidx = -torch.ones((num_points, ),
                                   dtype=torch.int32, device=device)

    _ext.hard_voxelize_forward_v2(
        points.contiguous(),
        voxel_size_tensor.contiguous(),
        coors_range_tensor.contiguous(),
        voxels, coors, num_points_per_voxel, voxel_num,
        point_to_pointidx, point_to_voxelidx, coor_to_voxelidx,
        max_points, max_voxels, NDim, deterministic)

    actual_voxel_num = voxel_num.item()

    return {
        'voxels': voxels[:actual_voxel_num],
        'coors': coors[:actual_voxel_num],
        'num_points_per_voxel': num_points_per_voxel[:actual_voxel_num],
        'voxel_num': actual_voxel_num,
        # These three used to be allocated, passed to the op, and thrown away.
        # They are the point -> voxel assignment, and batched_voxelize needs
        # them to rebuild the voxel tensor differentiably: the op writes into
        # the torch.zeros buffers above, so `voxels` carries no graph and any
        # loss computed from it has a gradient of exactly zero w.r.t. `points`.
        # Semantics verified on a hand-built cloud in job 18750221:
        #   coor_to_voxelidx[i]  voxel row for point i, -1 if dropped
        #   point_to_voxelidx[i] slot within that voxel, -1 if dropped
        #   point_to_pointidx[i] first point of that voxel, -1 if dropped
        # "dropped" covers both out-of-range points and those cut by the
        # max_points truncation; all three read -1 in either case.
        'point_to_voxelidx': point_to_voxelidx,
        'coor_to_voxelidx': coor_to_voxelidx,
        'point_to_pointidx': point_to_pointidx,
    }


def batched_voxelize(
    points_list: List[torch.Tensor],
    voxel_size: list,
    coors_range: list,
    max_points: int = 10,
    max_voxels: int = 160000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxelize a batch and prepend the batch index to the coordinates."""
    device = points_list[0].device
    all_voxels, all_coors, all_num_points = [], [], []

    for batch_idx, points in enumerate(points_list):
        res = hard_voxelize_v2(
            points=points,
            voxel_size=voxel_size,
            coors_range=coors_range,
            max_points=max_points,
            max_voxels=max_voxels)

        # hard_voxelize_v2 returns (z, y, x); BEVFusionSparseEncoder declares
        # sparse_shape = [1440, 1440, 41], i.e. (x, y, z). Feeding (z, y, x)
        # puts x (max 1439) into an axis sized 41. The SubMConv3d layers do not
        # notice -- submanifold convs reuse the input indices -- but the first
        # strided SparseConv3d has to DERIVE output indices, and out-of-grid
        # inputs yield none, which is the "N > 0 assert failed ... got N= 0"
        # crash on ~18% of scenes. The other ~82% did not crash; they silently
        # encoded scrambled geometry, which is worse.
        #
        # Verified in job 17520295: with this permutation the encoder output
        # matches BEVFusion's own extract_pts_feat bit for bit (max|diff| = 0),
        # confirming the two voxelizers agree and ONLY the axis order differed.
        coors = res['coors'][:, [2, 1, 0]].contiguous()
        batch_idx_col = torch.full((coors.shape[0], 1), batch_idx,
                                   dtype=coors.dtype, device=device)

        # Rebuild the voxel tensor so it carries a graph back to `points`.
        #
        # res['voxels'] is a slice of a torch.zeros buffer that the C++ op
        # wrote into in place. No autograd.Function wraps that call, so the
        # tensor has requires_grad=False and grad_fn=None -- confirmed
        # directly in job 18732092. Everything downstream (the mean reduce,
        # middle_encoder, <grad, features>) therefore detaches, and
        # total_loss.backward() succeeds only because the Chamfer term
        # supplies a valid graph. The adversarial term contributed EXACTLY
        # zero on every run before this fix, which is why loss_sign=0
        # reproduced the full attack to 0.0009 mAP (job 18699985).
        #
        # The assignment itself stays non-differentiable, which is correct --
        # which voxel a point falls into is piecewise constant, so it has zero
        # gradient almost everywhere. Only the aggregation needs a graph, and
        # scattering the ORIGINAL points into their slots supplies one.
        #
        # index_put (out-of-place) rather than an in-place write: it returns a
        # new tensor with IndexPutBackward attached, instead of relying on
        # in-place mutation of a buffer that does not require grad.
        #
        # Verified bit-exact against the op's own output (atol=0, rtol=0) in
        # job 18750221, including max_points truncation and out-of-range
        # points, so this changes the gradient and nothing else.
        row = res['coor_to_voxelidx'].long()
        slot = res['point_to_voxelidx'].long()
        kept = (row >= 0) & (slot >= 0) & (slot < max_points)
        nvox = int(res['voxel_num'])
        voxels_diff = torch.zeros(
            (nvox, max_points, points.shape[1]),
            dtype=points.dtype, device=device).index_put(
                (row[kept], slot[kept]), points[kept])

        all_voxels.append(voxels_diff)
        all_coors.append(torch.cat([batch_idx_col, coors], dim=1))
        all_num_points.append(res['num_points_per_voxel'])

    return (torch.cat(all_voxels, dim=0),
            torch.cat(all_coors, dim=0),
            torch.cat(all_num_points, dim=0))


# =============================================================================
# BEVFusion LiDAR encoder -- points through pts_middle_encoder
# =============================================================================

def _voxel_counts(encoder, points_list):
    """Per-sample voxel counts, for diagnosing a failed batched forward.

    The batch only reaches middle_encoder with zero active sites if the
    concatenation of every sample voxelised to nothing, so printing the
    per-sample counts says immediately whether one degenerate sample took the
    batch down or whether something else is going on.
    """
    out = []
    for p in points_list:
        try:
            r = hard_voxelize_v2(
                p.detach(), encoder.voxel_size, encoder.point_cloud_range,
                max_points=encoder.max_num_points,
                max_voxels=encoder.max_voxels)
            out.append(int(r['voxel_num']))
        except Exception as e:  # noqa: BLE001
            out.append(f'ERR:{type(e).__name__}')
    return out


class BEVFusionLidarEncoder(nn.Module):
    """points -> voxelize -> mean-reduce -> middle_encoder -> [B,256,180,180].

    Mirrors BEVFusion.extract_pts_feat, with voxelization swapped for the
    differentiable one. pts_backbone / pts_neck / fusion_layer / camera branch
    are deliberately absent: the gradient target is the middle encoder output,
    so nothing downstream contributes.
    """

    def __init__(self, middle_encoder, voxel_size, point_cloud_range,
                 max_num_points=10, max_voxels=160000, voxelize_reduce=True):
        super().__init__()
        self.middle_encoder = middle_encoder
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.voxelize_reduce = voxelize_reduce

    def forward(self, points_list: List[torch.Tensor]) -> torch.Tensor:
        batch_size = len(points_list)

        feats, coors, sizes = batched_voxelize(
            points_list=points_list,
            voxel_size=self.voxel_size,
            coors_range=self.point_cloud_range,
            max_points=self.max_num_points,
            max_voxels=self.max_voxels)

        if self.voxelize_reduce:
            # bevfusion.py:196-199, verbatim. Equivalent to HardSimpleVFE.
            feats = feats.sum(dim=1, keepdim=False) / \
                sizes.type_as(feats).view(-1, 1)
            feats = feats.contiguous()

        return self.middle_encoder(feats, coors, batch_size)


# =============================================================================
# Unidirectional Chamfer distance
# =============================================================================

def unidirectional_chamfer(adv_points_list, orig_points_list, chamfer_fn):
    """dist(adv -> orig), summed over the batch.

    One direction only, matching attack_focalformer_nus.py. Note that the
    function there is *named* batched_chamfer_distance_bidirectional but
    computes dist1 alone -- the name is wrong, the behaviour is what its
    docstring banner ("Chamfer: UniDIRECTIONAL (dist1)") advertises. This is
    the same behaviour under an accurate name.
    """
    total = 0.0
    for adv_pts, orig_pts in zip(adv_points_list, orig_points_list):
        total = total + chamfer_fn(
            adv_pts[:, :3].unsqueeze(0), orig_pts[:, :3].unsqueeze(0))
    return total


# =============================================================================
# Attack
# =============================================================================

def run_attack(cfg_path, gradient_folder, result_save_path, checkpoint_path,
               data_root=None, device='cuda:0', batch_size=4,
               num_iterations=40, learning_rate=0.01, dist_weight=1.0,
               max_batches=None, max_voxels_idx=1, loss_sign=1.0,
               init_noise_std=0.3, diag_grad_split=False,
               eps_ball=0.0, eps_step=0.0):

    # 2.5 * eps / T is the standard PGD step: enough total travel (2.5x) to
    # reach the far side of the ball and still manoeuvre once there, rather
    # than eps/T, which only just arrives if every step points the same way.
    if eps_ball > 0 and eps_step <= 0:
        eps_step = 2.5 * eps_ball / max(num_iterations, 1)

    print('=' * 60)
    print('Batched BEVFusion Adversarial Attack')
    print('  Target layer : pts_middle_encoder')
    print(f'  Batch size   : {batch_size}')
    print(f'  Iterations   : {num_iterations}   lr: {learning_rate}')
    if eps_ball > 0:
        print(f'  Budget       : PGD, per-point L2 ball eps={eps_ball} m')
        print(f'  Step         : {eps_step:.6f} m/iter (L2-normalised)')
        print('  Chamfer      : reporting only, NOT in the loss')
    else:
        print('  Chamfer      : UNIdirectional (adv -> orig), soft penalty')
        print(f'  dist_weight  : {dist_weight}')
    print(f'  loss_sign    : {loss_sign:+.0f}')
    print('  Save         : FINAL iteration')
    print('=' * 60)

    cfg = Config.fromfile(cfg_path)

    if data_root:
        cfg.train_dataloader.dataset.data_root = data_root
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = min(batch_size * 2, 16)

    # BEVFusion.__init__ pops 'voxelize_cfg' out of data_preprocessor, so read
    # it before MODELS.build mutates the dict.
    vcfg = cfg.model.data_preprocessor.voxelize_cfg
    voxel_size = list(vcfg.voxel_size)
    point_cloud_range = list(vcfg.point_cloud_range)
    max_num_points = vcfg.max_num_points
    voxelize_reduce = vcfg.get('voxelize_reduce', True)
    max_voxels = vcfg.max_voxels
    if isinstance(max_voxels, (list, tuple)):
        max_voxels = max_voxels[max_voxels_idx]

    print('\nVoxelization (from data_preprocessor.voxelize_cfg):')
    print(f'  voxel_size      : {voxel_size}')
    print(f'  pcd range       : {point_cloud_range}')
    print(f'  max_num_points  : {max_num_points}')
    print(f'  max_voxels      : {max_voxels} (index {max_voxels_idx} of '
          f'{vcfg.max_voxels})')
    print(f'  voxelize_reduce : {voxelize_reduce}')

    print('\nBuilding BEVFusion...')
    # deepcopy because BEVFusion.__init__ is destructive: it does
    # data_preprocessor.pop('voxelize_cfg') (bevfusion.py:38), mutating the cfg
    # in place. Anything that later reads cfg.model -- including a second
    # build -- then finds voxelize_cfg gone.
    full_model = MODELS.build(deepcopy(cfg.model)).to(device)
    load_checkpoint(full_model, checkpoint_path, map_location=device)
    full_model.eval()
    print('  model loaded')

    encoder = BEVFusionLidarEncoder(
        middle_encoder=full_model.pts_middle_encoder,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
        max_voxels=max_voxels,
        voxelize_reduce=voxelize_reduce).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    chamfer_dist = ChamferDistance().to(device)

    # Only the dataloader is needed, so build just that.
    #
    # Runner.from_cfg would construct a SECOND, complete BEVFusion -- Swin-T,
    # view transform, TransFusion head, all of it -- purely as a side effect of
    # wanting an iterator. On top of the wasted GPU memory and startup time,
    # that second build crashes: the first build already popped voxelize_cfg
    # out of the shared cfg, so BEVFusion.__init__ hits
    #     AttributeError: 'NoneType' object has no attribute 'pop'
    # The deepcopy above keeps cfg.model intact, but building the model twice
    # was never wanted in the first place. build_dataloader is a staticmethod
    # and touches none of that.
    #
    # It also sidesteps the grad-extract config's custom_hooks (the gradient
    # hook must not re-run here) and its NoOpOptimizer, since no Runner exists
    # to honour either.
    print('\nBuilding data loader...')
    dataloader = Runner.build_dataloader(cfg.train_dataloader)
    num_batches = len(dataloader)
    print(f'  {num_batches} batches, ~{num_batches * batch_size} samples')

    os.makedirs(result_save_path, exist_ok=True)

    processed = skipped = 0
    all_batch_losses = []
    failures = []

    for batch_idx, data_batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        data_samples = data_batch['data_samples']
        points_list_raw = data_batch['inputs']['points']

        gradient_tensors, lidar_filenames, original_points_list = [], [], []

        for i, data_sample in enumerate(data_samples):
            meta = data_sample.metainfo
            lidar_path = (meta.get('lidar_path', None)
                          or meta.get('pts_filename', None))
            if lidar_path is None:
                lp = meta.get('lidar_points', {})
                if isinstance(lp, dict):
                    lidar_path = lp.get('lidar_path', None)
            if lidar_path is None:
                skipped += 1
                continue

            lidar_filename = os.path.basename(lidar_path)
            sample_id = lidar_filename
            for ext in ['.pcd.bin', '.bin', '.npy']:
                if sample_id.endswith(ext):
                    sample_id = sample_id[:-len(ext)]
                    break

            gradient_path = os.path.join(gradient_folder,
                                         f'{sample_id}_grad.pt')
            if not os.path.exists(gradient_path):
                if batch_idx == 0 and i == 0:
                    print(f"  DEBUG: no gradient for '{sample_id}'")
                    try:
                        for f in sorted(os.listdir(gradient_folder))[:5]:
                            print(f'    {f}')
                    except OSError:
                        pass
                skipped += 1
                continue

            gradient_tensors.append(
                torch.load(gradient_path, map_location=device).float())
            lidar_filenames.append(lidar_filename)
            original_points_list.append(points_list_raw[i].to(device).float())

        if not gradient_tensors:
            continue

        print(f'\n[Batch {batch_idx}/{num_batches}] '
              f'{len(gradient_tensors)} valid samples')

        shapes = {tuple(g.shape) for g in gradient_tensors}
        if len(shapes) > 1:
            print(f'  WARNING: mixed gradient shapes {shapes}, '
                  'processing one at a time')
            for g, o, fn in zip(gradient_tensors, original_points_list,
                                lidar_filenames):
                st = _single(encoder, g, o, fn, chamfer_dist,
                             point_cloud_range, num_iterations, learning_rate,
                             dist_weight, result_save_path, device, loss_sign,
                             init_noise_std, eps_ball, eps_step)
                if st != 'ok':
                    failures.append({'file': fn, 'batch': int(batch_idx),
                                     'reason': 'forward_failed_mixed_shape',
                                     'wrote': 'clean'})
                processed += 1
            continue

        gradient_batch = torch.cat(gradient_tensors, dim=0)

        adv_points_list = []
        for orig_pts in original_points_list:
            adv_pts = orig_pts.clone().detach()
            noise = torch.normal(0, init_noise_std,
                                 size=(orig_pts.shape[0], 3), device=device)
            adv_pts[:, :3] = adv_pts[:, :3] + noise
            adv_pts.requires_grad_(True)
            adv_points_list.append(adv_pts)

        optimizer = optim.Adam(adv_points_list, lr=learning_rate)
        history = {'adv_loss': [], 'dist_loss': [], 'total_loss': []}
        best_loss, best_iteration = float('inf'), -1
        forward_error, failed_at = None, -1

        for it in range(num_iterations):
            optimizer.zero_grad()

            # In eps-ball mode Chamfer is REPORTING ONLY. The budget is
            # enforced by projection, so the penalty must contribute no
            # gradient -- computed under no_grad so that it structurally
            # cannot, rather than relying on it being left out of total_loss.
            # Still worth measuring: it is the direct comparison against the
            # ~8.5 mm the soft-penalty equilibrium produced.
            if eps_ball > 0:
                with torch.no_grad():
                    dist_loss = unidirectional_chamfer(
                        adv_points_list, original_points_list, chamfer_dist)
            else:
                dist_loss = unidirectional_chamfer(
                    adv_points_list, original_points_list, chamfer_dist)

            # loss_sign == 0 is the Chamfer-only ablation. It multiplies the
            # adversarial term to zero, so that term contributes exactly no
            # gradient and the encoder forward becomes pure waste -- and the
            # encoder dominates the ~36 s/batch. Skipping it leaves the
            # optimisation trajectory mathematically identical (0 * x has zero
            # derivative) while cutting the run from ~17 h to a few hours.
            # Guarded on == 0 so the normal loss_sign=1.0 path is untouched.
            if loss_sign == 0:
                adv_loss = torch.zeros((), device=device)
                total_loss = dist_weight * dist_loss
            else:
                try:
                    features = encoder(adv_points_list)
                except Exception as e:  # noqa: BLE001
                    forward_error, failed_at = e, it
                    break

                if features.shape != gradient_batch.shape:
                    print(f'  shape mismatch: features '
                          f'{tuple(features.shape)} vs gradient '
                          f'{tuple(gradient_batch.shape)}')
                    break

                adv_loss = loss_sign * torch.sum(gradient_batch * features)
                total_loss = (adv_loss if eps_ball > 0
                              else adv_loss + dist_weight * dist_loss)

            history['adv_loss'].append(adv_loss.item())
            history['dist_loss'].append(dist_loss.item())
            history['total_loss'].append(total_loss.item())
            if total_loss.item() < best_loss:
                best_loss, best_iteration = total_loss.item(), it

            # Is the adversarial term actually steering the points, or is
            # Chamfer doing all the work? The combined grad_norm printed below
            # cannot answer that -- while the encoder gradient was dead it
            # still looked healthy, because Chamfer alone produces a large
            # gradient and `adv` still drifted as the points moved underneath
            # it. Only the SPLIT distinguishes the two.
            #
            # Measured before backward() and with retain_graph, since backward
            # frees the graph. xyz columns only: 3-4 are zeroed just below, so
            # including them would compare a number the optimiser never sees.
            # eps_ball <= 0 guard: in eps mode dist_loss is built under no_grad
            # and carries no graph, so autograd.grad on it would raise. The
            # split is also meaningless there -- Chamfer contributes exactly
            # nothing by construction, so the adv share is trivially 100%.
            diag = None
            if (diag_grad_split and loss_sign != 0 and eps_ball <= 0
                    and (it % 10 == 0 or it == num_iterations - 1)):
                def _xyz_norm(grads):
                    vals = [float(g[:, :3].norm())
                            for g in grads if g is not None]
                    return sum(vals) / max(len(vals), 1)

                g_adv = torch.autograd.grad(
                    adv_loss, adv_points_list, retain_graph=True,
                    allow_unused=True)
                g_dist = torch.autograd.grad(
                    dist_weight * dist_loss, adv_points_list,
                    retain_graph=True, allow_unused=True)
                diag = (_xyz_norm(g_adv), _xyz_norm(g_dist))

            total_loss.backward()

            # Geometry only. adv_pts is the full 5-column tensor and Adam owns
            # all of it, but an attack may move x/y/z -- not a return's
            # intensity or its timestamp.
            #
            # This was harmless while the encoder gradient was dead: Chamfer
            # reads [:, :3] alone, so columns 3-4 never moved, which is exactly
            # why pc_change_stats.py measured them as bit-identical. With the
            # gradient live they are suddenly the LOUDEST signal -- on the
            # synthetic check in job 18750353 the xyz component was 113.6 of a
            # 26,962 total, so ~99.6% of the update would go into editing
            # intensity and time. That is physically meaningless as a LiDAR
            # perturbation and it would also destroy the time-channel
            # alignment that the pct_change measurement depends on.
            for adv_pts in adv_points_list:
                if adv_pts.grad is not None:
                    adv_pts.grad[:, 3:] = 0

            if it % 10 == 0 or it == num_iterations - 1:
                norms = [p.grad.norm().item() if p.grad is not None else 0.0
                         for p in adv_points_list]
                extra = ''
                if diag is not None:
                    ga, gd = diag
                    share = 100.0 * ga / max(ga + gd, 1e-12)
                    extra = (f' | xyz grad adv={ga:.4e} dist={gd:.4e} '
                             f'adv_share={share:.2f}%')
                if eps_ball > 0:
                    # What budget is actually being spent? The projection caps
                    # displacement at eps, but points can sit anywhere inside
                    # the ball -- if mean displacement stays far below eps the
                    # attack is not using the budget it was given, which is a
                    # different failure from using it and not helping.
                    with torch.no_grad():
                        d = torch.cat([
                            (a[:, :3] - o[:, :3]).norm(dim=1)
                            for a, o in zip(adv_points_list,
                                            original_points_list)])
                        extra = (f'{extra} | disp mean={d.mean():.4f} '
                                 f'max={d.max():.4f} '
                                 f'at_cap={100.0 * (d > eps_ball * 0.99).float().mean():.1f}%')
                print(f'    iter {it:3d}: adv={adv_loss.item():.4f} '
                      f'dist={dist_loss.item():.4f} '
                      f'total={total_loss.item():.4f} '
                      f'grad_norm={sum(norms) / len(norms):.6f}{extra}')

            if eps_ball > 0:
                _pgd_step(adv_points_list, original_points_list, eps_ball,
                          eps_step, point_cloud_range)
            else:
                optimizer.step()

                with torch.no_grad():
                    for adv_pts in adv_points_list:
                        adv_pts[:, 0].clamp_(point_cloud_range[0],
                                             point_cloud_range[3])
                        adv_pts[:, 1].clamp_(point_cloud_range[1],
                                             point_cloud_range[4])
                        adv_pts[:, 2].clamp_(point_cloud_range[2],
                                             point_cloud_range[5])

        # A batched forward that dies takes all B samples with it even when
        # only one of them is degenerate. Retry those samples individually so
        # the good ones still get attacked. Before this, the whole batch fell
        # through to the save below holding original + N(0, init_noise_std)
        # noise -- unoptimised random perturbation written out as if it were
        # adversarial, which then passed the downstream 6019-file completeness
        # check unnoticed.
        if forward_error is not None and failed_at == 0:
            print(f'  forward failed at iter 0: '
                  f'{type(forward_error).__name__}: {forward_error}')
            print(f'    points/sample : '
                  f'{[int(p.shape[0]) for p in adv_points_list]}')
            print(f'    voxels/sample : {_voxel_counts(encoder, adv_points_list)}')
            print('    retrying per sample')
            for g, o, fn in zip(gradient_tensors, original_points_list,
                                lidar_filenames):
                st = _single(encoder, g, o, fn, chamfer_dist,
                             point_cloud_range, num_iterations, learning_rate,
                             dist_weight, result_save_path, device, loss_sign,
                             init_noise_std, eps_ball, eps_step)
                if st != 'ok':
                    failures.append({'file': fn, 'batch': int(batch_idx),
                                     'reason': 'forward_failed_single',
                                     'wrote': 'clean'})
                processed += 1
            del gradient_batch, gradient_tensors, adv_points_list
            del original_points_list
            torch.cuda.empty_cache()
            continue

        if forward_error is not None:
            # Died mid-optimisation: the completed iterations are real work, so
            # keep them, but record that this sample got fewer than asked.
            print(f'  forward failed at iter {failed_at} '
                  f'({type(forward_error).__name__}); keeping the '
                  f'{failed_at} completed iterations')
            for fn in lidar_filenames:
                failures.append({'file': fn, 'batch': int(batch_idx),
                                 'reason': 'forward_failed_partial',
                                 'completed_iters': int(failed_at),
                                 'wrote': 'partial'})

        all_batch_losses.append({
            'batch_idx': batch_idx,
            'filenames': lidar_filenames,
            'loss_history': history,
            'best_loss': best_loss,
            'best_iteration': best_iteration,
            'final_loss': history['total_loss'][-1]
            if history['total_loss'] else None,
        })

        for adv_pts, filename in zip(adv_points_list, lidar_filenames):
            adv_pts.detach().cpu().numpy().astype(np.float32).tofile(
                os.path.join(result_save_path, filename))
            processed += 1

        del gradient_batch, gradient_tensors, adv_points_list
        del original_points_list
        torch.cuda.empty_cache()

    print('\n' + '=' * 60)
    print(f'Attack complete. processed={processed} skipped={skipped}')
    print(f'Results: {result_save_path}')
    if all_batch_losses:
        finals = [b['final_loss'] for b in all_batch_losses
                  if b['final_loss'] is not None]
        # A batch whose forward never ran keeps best_loss=inf, which would turn
        # the mean into inf and hide the real distribution.
        bests = [b['best_loss'] for b in all_batch_losses
                 if np.isfinite(b['best_loss'])]
        if finals:
            print(f'  final loss  mean={np.mean(finals):.4f} '
                  f'min={np.min(finals):.4f} max={np.max(finals):.4f}')
        if bests:
            print(f'  best  loss  mean={np.mean(bests):.4f} '
                  f'min={np.min(bests):.4f} max={np.max(bests):.4f}')
    torch.save(all_batch_losses,
               os.path.join(result_save_path, 'loss_history.pt'))

    # Manifest of every sample that did not get a full attack. Without this the
    # only record lives in the slurm log, and the tar of point clouds looks
    # complete either way -- the downstream 6019-file check cannot tell an
    # attacked cloud from a clean one.
    manifest = os.path.join(result_save_path, 'attack_failures.json')
    with open(manifest, 'w') as f:
        json.dump({'total_processed': processed,
                   'total_skipped': skipped,
                   'n_failures': len(failures),
                   'failures': failures}, f, indent=2)
    if failures:
        n_clean = sum(1 for x in failures if x.get('wrote') == 'clean')
        n_part = sum(1 for x in failures if x.get('wrote') == 'partial')
        print(f'  !! {len(failures)} sample(s) did not get a full attack: '
              f'{n_clean} written CLEAN, {n_part} partial')
        print(f'     manifest: {manifest}')
    else:
        print('  all samples attacked for the full iteration budget')
    print('=' * 60)


def _pgd_step(adv_points_list, original_points_list, eps_ball, step_size,
              point_cloud_range):
    """One projected-gradient step inside a per-point L2 ball of radius eps.

    Replaces Adam + the soft Chamfer penalty. Two things change, both of them
    the point of the exercise:

    1. The budget is CHOSEN rather than emergent. Under `adv + beta * chamfer`
       the trajectory settles wherever ||grad chamfer|| == ||grad adv||. At
       beta=1 that landed at ~8.5 mm (sweep 18798999) -- a number nobody picked
       and which moves if anything about the model or the gradient scale moves.
       Worse, at that point the two gradients are near parity (ratio 0.66) and
       oppose each other, so the net update is ~0 and the run stalls: over 429
       batches of job 18801696 the objective's direction was indistinguishable
       from a coin flip (p=0.385). AI@x needs x to be an input, not an output.

    2. Only the gradient DIRECTION survives. The step is L2-normalised per
       point, so scaling g by any constant changes nothing at all. That is what
       frees extraction to use normalize='global' -- saliency preserved,
       whole-tensor norm 1 -- instead of 'channel', which erases cross-cell
       saliency (measured ratio 1.000 vs 49.474) to buy a norm of 180 that only
       a soft penalty needs. Under projection that trade disappears.

    In-place on leaf tensors under no_grad, the standard PGD idiom.
    """
    with torch.no_grad():
        for adv_pts, orig_pts in zip(adv_points_list, original_points_list):
            if adv_pts.grad is None:
                continue
            g = adv_pts.grad[:, :3]
            # clamp_min guards zero-gradient points, which are common here: a
            # point landing in no retained voxel receives no gradient at all,
            # and 0/0 would put NaN into the cloud.
            adv_pts[:, :3] -= step_size * g / g.norm(
                dim=1, keepdim=True).clamp_min(1e-12)
            # Project onto the ball around the ORIGINAL point. Points inside
            # are untouched; points outside are pulled back onto its surface.
            delta = adv_pts[:, :3] - orig_pts[:, :3]
            scale = (eps_ball / delta.norm(dim=1, keepdim=True)
                     .clamp_min(1e-12)).clamp(max=1.0)
            adv_pts[:, :3] = orig_pts[:, :3] + delta * scale
            for k in range(3):
                adv_pts[:, k].clamp_(point_cloud_range[k],
                                     point_cloud_range[k + 3])


def _single(encoder, gradient, orig_pts, filename, chamfer_fn,
            point_cloud_range, num_iterations, learning_rate, dist_weight,
            result_save_path, device, loss_sign, init_noise_std,
            eps_ball=0.0, eps_step=0.0):
    """Attack one sample on its own.

    Used for shape-mismatched batches and as the fallback when a batched
    forward dies, since one degenerate sample takes its whole batch with it.

    Returns 'ok' if at least one optimisation step ran, else 'failed'.

    On failure the CLEAN cloud is written, never the noised one. An
    unoptimised adv_pts is original + N(0, init_noise_std) -- 30 cm of
    isotropic random noise by default -- which is an uncontrolled corruption,
    not an attack. Writing it would understate the model's robustness for
    reasons that have nothing to do with the gradient, so the honest fallback
    is a genuine no-op that the manifest records.
    """
    adv_pts = orig_pts.clone().detach()
    noise = torch.normal(0, init_noise_std, size=(orig_pts.shape[0], 3),
                         device=device)
    adv_pts[:, :3] = adv_pts[:, :3] + noise
    adv_pts.requires_grad_(True)

    optimizer = optim.Adam([adv_pts], lr=learning_rate)
    completed = 0

    for _ in range(num_iterations):
        optimizer.zero_grad()
        # eps-ball mode enforces the budget by projection, so Chamfer must not
        # contribute gradient. Skipped outright here rather than computed and
        # discarded -- this is the per-sample retry path and it runs one sample
        # at a time, so the saved forward is worth having.
        dist_loss = (torch.zeros((), device=adv_pts.device) if eps_ball > 0
                     else chamfer_fn(adv_pts[:, :3].unsqueeze(0),
                                     orig_pts[:, :3].unsqueeze(0)))
        try:
            features = encoder([adv_pts])
        except Exception as e:  # noqa: BLE001
            print(f'      [{filename}] failed after {completed} iters: '
                  f'{type(e).__name__}')
            break
        total_loss = loss_sign * torch.sum(gradient * features)
        if eps_ball <= 0:
            total_loss = total_loss + dist_weight * dist_loss
        total_loss.backward()
        # Geometry only -- same reason as the batched path above: with the
        # encoder gradient live, intensity and timestamp would otherwise soak
        # up ~99.6% of the update.
        if adv_pts.grad is not None:
            adv_pts.grad[:, 3:] = 0
        if eps_ball > 0:
            _pgd_step([adv_pts], [orig_pts], eps_ball, eps_step,
                      point_cloud_range)
        else:
            optimizer.step()
            with torch.no_grad():
                adv_pts[:, 0].clamp_(point_cloud_range[0],
                                     point_cloud_range[3])
                adv_pts[:, 1].clamp_(point_cloud_range[1],
                                     point_cloud_range[4])
                adv_pts[:, 2].clamp_(point_cloud_range[2],
                                     point_cloud_range[5])
        completed += 1

    out = adv_pts if completed > 0 else orig_pts
    out.detach().cpu().numpy().astype(np.float32).tofile(
        os.path.join(result_save_path, filename))
    return 'ok' if completed > 0 else 'failed'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='BEVFusion adversarial attack')
    p.add_argument('--cfg', required=True,
                   help='a *_grad_extract.py config (model + dataloader)')
    p.add_argument('--grads', required=True, help='dir of *_grad.pt')
    p.add_argument('--results', required=True, help='dir for adversarial .bin')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root', default=None)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--iterations', type=int, default=40)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--dist_weight', type=float, default=1.0)
    p.add_argument('--init_noise_std', type=float, default=0.3)
    p.add_argument(
        '--eps_ball', type=float, default=0.0,
        help='If > 0, replace the soft Chamfer penalty with PGD inside a '
             'per-point L2 ball of this radius, in METRES. The Chamfer term '
             'is then computed for reporting only and contributes no '
             'gradient. 0 (default) keeps the existing soft-penalty attack. '
             'Reference points: the voxel is 0.075 m, and the beta=1 soft '
             'penalty settles at ~0.0085 m on its own.')
    p.add_argument(
        '--eps_step', type=float, default=0.0,
        help='PGD step in metres. 0 (default) uses the standard '
             '2.5 * eps / iterations. Ignored unless --eps_ball > 0.')
    p.add_argument('--max_batches', type=int, default=None)
    p.add_argument('--max_voxels_idx', type=int, default=1, choices=[0, 1],
                   help='index into voxelize_cfg.max_voxels (train, test). '
                        '1 matches gradient extraction run with eval_mode=True')
    p.add_argument(
        '--loss_sign', type=float, default=1.0, choices=[1.0, -1.0, 0.0],
        help='sign on <grad, features>. +1 reproduces '
             'attack_focalformer_nus.py, which MINIMISES the dot product; -1 '
             'ascends the detection loss. 0 drops the adversarial term for the '
             'Chamfer-only ablation and skips the encoder forward entirely, '
             'since a zero-weighted term contributes no gradient. See the note '
             'printed at startup.')
    p.add_argument(
        '--diag_grad_split', action='store_true',
        help='at the logged iterations, report ||d(adv)/d(xyz)|| and '
             '||d(beta*dist)/d(xyz)|| separately. Costs one extra backward '
             'through the encoder at those iterations, so it is off by '
             'default for full runs; on for the lightweight test, where it is '
             'the check that the adversarial term is actually steering the '
             'points rather than Chamfer doing all the work.')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    os.makedirs(args.results, exist_ok=True)

    if args.eps_ball > 0:
        if args.loss_sign == 0:
            raise SystemExit(
                'X --loss_sign 0 with --eps_ball drops the adversarial term '
                'AND the Chamfer term, leaving nothing to optimise. The '
                'Chamfer-only ablation only means something under the soft '
                'penalty.')
        if args.init_noise_std > 0:
            # Not fatal -- a random start inside the ball is standard PGD --
            # but noise LARGER than the ball is immediately projected back
            # onto the surface, so every point starts at exactly eps in a
            # random direction. That is a very different initialisation from
            # what the flag name suggests.
            print(f'\nNOTE: --init_noise_std {args.init_noise_std} with '
                  f'--eps_ball {args.eps_ball}: the first projection clamps '
                  f'any displacement above eps, so points starting beyond the '
                  f'ball land on its surface in a random direction.\n',
                  flush=True)

    if args.loss_sign > 0:
        print('\nWARNING on --loss_sign +1:\n'
              '  Adam minimises total_loss, so this minimises <dL/df, f>,\n'
              '  which moves the feature OPPOSITE the loss gradient -- the\n'
              '  DESCENT direction. Ascending the detection loss, i.e. the\n'
              '  actual attack, is --loss_sign -1.\n'
              '  This used to be defended on the grounds that +1 degrades\n'
              '  detection anyway, because <g, f> is unbounded below and the\n'
              '  points get pushed hard enough to wreck the cloud regardless\n'
              '  of direction. That defence died with init_noise_std=0: the\n'
              '  perturbation is now Chamfer-bounded to ~8.5 mm, so there is\n'
              '  no cloud destruction left to hide behind and +1 does nothing\n'
              '  but steer features down the loss gradient inside a tiny\n'
              '  budget -- an anti-attack.\n'
              '  Parity with the existing FocalFormer3D numbers is NOT a\n'
              '  reason to keep +1: those runs predate the voxelization fix,\n'
              '  so d(adv)/d(points) was exactly zero and the sign of a zero\n'
              '  vector never mattered. There is no parity to preserve.\n',
              flush=True)

    run_attack(
        cfg_path=args.cfg,
        gradient_folder=args.grads,
        result_save_path=args.results,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        dist_weight=args.dist_weight,
        max_batches=args.max_batches,
        max_voxels_idx=args.max_voxels_idx,
        loss_sign=args.loss_sign,
        eps_ball=args.eps_ball,
        eps_step=args.eps_step,
        init_noise_std=args.init_noise_std,
        diag_grad_split=args.diag_grad_split)
