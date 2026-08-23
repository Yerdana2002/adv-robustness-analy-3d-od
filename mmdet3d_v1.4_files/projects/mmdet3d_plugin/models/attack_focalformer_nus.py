"""
Batched Adversarial Attack for FocalFormer3D on nuScenes
=========================================================
Adapted from CenterPoint attack script.

Key differences from CenterPoint version:
  - FocalFormer3D does its own voxelization (not via data_preprocessor)
  - Encoder path: voxelize -> voxel_encoder -> middle_encoder -> backbone -> neck
  - Gradient target is 'neck' (pts_neck output) [B, 512, 180, 180]
    vs CenterPoint's 'backbone.blocks.0' [B, 128, 180, 180]
  - FocalFormer3D uses 5 features: (x, y, z, intensity, time) from 10 sweeps
  - Model is in projects/mmdet3d_plugin, not a native mmdet3d model

Attack pipeline (no augmentation):
  1. Load clean val point cloud
  2. Initialize adversarial points = clean + small noise
  3. For each iteration:
     a. Voxelize adversarial points
     b. Forward through encoder up to neck
     c. Compute: loss = dot(gradient, features) + lambda * chamfer_dist
     d. Backprop and update point positions
  4. Save final adversarial point cloud

Usage:
  python attack_focalformer.py \
      --cfg projects/configs/focalformer3d/FocalFormer3D_L_v14_grad.py \
      --grads /path/to/gradients \
      --results /path/to/save/adversarial \
      --checkpoint /path/to/FocalFormer3D_L_ep6_converted.pth \
      --batch_size 4 --iterations 40 --lr 0.01 --dist_weight 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
from typing import Optional, List, Tuple, Dict
import argparse

# FocalFormer3D is in projects/, need to import it
sys.path.insert(0, os.getcwd())

from mmdet3d.registry import MODELS
from mmengine.runner import load_checkpoint, Runner
from mmengine.config import Config
from mmengine.registry import init_default_scope
from chamferdist import ChamferDistance

from mmcv import _ext
print(f"DEBUG: _ext loaded from: {_ext}", flush=True)

import faulthandler
faulthandler.enable()

init_default_scope('mmdet3d')


# =============================================================================
# Differentiable Voxelization (same as CenterPoint version)
# =============================================================================

def hard_voxelize_v2(
    points: torch.Tensor,
    voxel_size: list,
    coors_range: list,
    max_points: int = 10,
    max_voxels: int = 120000,
    NDim: int = 3,
    deterministic: bool = True
) -> dict:
    """Single sample voxelization using mmcv C++ extension."""
    device = points.device
    num_points = points.shape[0]
    num_features = points.shape[1]

    voxel_size_tensor = torch.tensor(voxel_size, dtype=torch.float32, device='cpu')
    coors_range_tensor = torch.tensor(coors_range, dtype=torch.float32, device='cpu')

    voxels = torch.zeros((max_voxels, max_points, num_features), dtype=points.dtype, device=device)
    coors = torch.zeros((max_voxels, NDim), dtype=torch.int32, device=device)
    num_points_per_voxel = torch.zeros((max_voxels,), dtype=torch.int32, device=device)
    voxel_num = torch.zeros((1,), dtype=torch.int32, device=device)

    point_to_pointidx = -torch.ones((num_points,), dtype=torch.int32, device=device)
    point_to_voxelidx = -torch.ones((num_points,), dtype=torch.int32, device=device)
    coor_to_voxelidx = -torch.ones((num_points,), dtype=torch.int32, device=device)

    _ext.hard_voxelize_forward_v2(
        points.contiguous(),
        voxel_size_tensor.contiguous(),
        coors_range_tensor.contiguous(),
        voxels, coors, num_points_per_voxel, voxel_num,
        point_to_pointidx, point_to_voxelidx, coor_to_voxelidx,
        max_points, max_voxels, NDim, deterministic
    )

    actual_voxel_num = voxel_num.item()

    return {
        'voxels': voxels[:actual_voxel_num],
        'coors': coors[:actual_voxel_num],
        'num_points_per_voxel': num_points_per_voxel[:actual_voxel_num],
        'voxel_num': actual_voxel_num,
        'point_to_pointidx': point_to_pointidx,
        'point_to_voxelidx': point_to_voxelidx,
        'coor_to_voxelidx': coor_to_voxelidx,
    }


def batched_voxelize(
    points_list: List[torch.Tensor],
    voxel_size: list,
    coors_range: list,
    max_points: int = 10,
    max_voxels: int = 120000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
    """Voxelize a batch of point clouds and combine results."""
    device = points_list[0].device

    all_voxels = []
    all_coors = []
    all_num_points = []
    voxel_infos = []

    for batch_idx, points in enumerate(points_list):
        voxel_result = hard_voxelize_v2(
            points=points,
            voxel_size=voxel_size,
            coors_range=coors_range,
            max_points=max_points,
            max_voxels=max_voxels
        )

        coors = voxel_result['coors']
        num_pts = voxel_result['num_points_per_voxel']

        # Rebuild the voxel tensor so it carries a graph back to `points`.
        #
        # voxel_result['voxels'] is a slice of a torch.zeros buffer that the
        # C++ op wrote into in place. No autograd.Function wraps that call, so
        # it has requires_grad=False and grad_fn=None. Everything downstream
        # (VFE reduce, middle_encoder, backbone, neck, <grad, features>)
        # therefore detaches, and d(adv)/d(points) is EXACTLY zero -- the same
        # defect found in attack_bevfusion_nus.py (job 18732092), where it made
        # loss_sign=0 reproduce the full attack to 0.0009 mAP (job 18699985).
        #
        # The assignment stays non-differentiable, which is correct: which
        # voxel a point falls into is piecewise constant and has zero gradient
        # almost everywhere. Only the aggregation needs a graph, and scattering
        # the ORIGINAL points into their slots supplies one.
        #
        # index_put is out-of-place, returning a new tensor with
        # IndexPutBackward attached rather than mutating a buffer that does not
        # require grad.
        #
        # NOTE: BEVFusion's version of this fix also permutes coors to
        # (x, y, z). That is NOT copied here. It exists because
        # BEVFusionSparseEncoder declares sparse_shape [1440, 1440, 41] while
        # hard_voxelize_v2 returns (z, y, x). FocalFormer3D's SparseEncoder is
        # a different module with its own convention, and this function already
        # passed coors through unpermuted, so changing the axis order would
        # alter geometry rather than fix a gradient. Only the graph is changed
        # here; coors, num_points_per_voxel and voxel_num are untouched.
        row = voxel_result['coor_to_voxelidx'].long()
        slot = voxel_result['point_to_voxelidx'].long()
        kept = (row >= 0) & (slot >= 0) & (slot < max_points)
        nvox = int(voxel_result['voxel_num'])
        voxels = torch.zeros(
            (nvox, max_points, points.shape[1]),
            dtype=points.dtype, device=device).index_put(
                (row[kept], slot[kept]), points[kept])

        batch_idx_col = torch.full(
            (coors.shape[0], 1), batch_idx,
            dtype=coors.dtype, device=device
        )
        coors_with_batch = torch.cat([batch_idx_col, coors], dim=1)

        all_voxels.append(voxels)
        all_coors.append(coors_with_batch)
        all_num_points.append(num_pts)
        voxel_infos.append(voxel_result)

    combined_voxels = torch.cat(all_voxels, dim=0)
    combined_coors = torch.cat(all_coors, dim=0)
    combined_num_points = torch.cat(all_num_points, dim=0)

    return combined_voxels, combined_coors, combined_num_points, voxel_infos


# =============================================================================
# FocalFormer3D Encoder — voxelize through neck
# =============================================================================

class FocalFormer3DEncoder(nn.Module):
    """
    Differentiable encoder for FocalFormer3D: voxelize -> neck output.

    FocalFormer3D architecture:
        pts_voxel_layer (Voxelization) -> voxels
        pts_voxel_encoder (HardSimpleVFE) -> voxel features
        pts_middle_encoder (SparseEncoder) -> 2D dense BEV
        pts_backbone (SECOND) -> multi-scale 2D features
        pts_neck (SECONDFPN) -> [B, 512, 180, 180]   <-- gradient target

    Unlike CenterPoint, FocalFormer3D does its own voxelization
    (not delegated to data_preprocessor).
    """

    def __init__(
        self,
        voxel_encoder: nn.Module,
        middle_encoder: nn.Module,
        backbone: nn.Module,
        neck: nn.Module,
        voxel_size: list,
        point_cloud_range: list,
        max_num_points: int = 10,
        max_voxels: int = 120000,
        target_layer: str = 'neck'
    ):
        super().__init__()
        self.voxel_encoder = voxel_encoder
        self.middle_encoder = middle_encoder
        self.backbone = backbone
        self.neck = neck
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.target_layer = target_layer

    def forward(self, points_list: List[torch.Tensor]) -> Tuple[torch.Tensor, List[dict]]:
        """
        Forward pass: point clouds -> feature maps at target layer.

        Args:
            points_list: List of [N_i, 5] tensors (x, y, z, intensity, time)

        Returns:
            features: Tensor at target layer
            voxel_infos: Per-sample voxelization metadata
        """
        batch_size = len(points_list)

        # 1. Voxelize
        voxels, coors, num_points, voxel_infos = batched_voxelize(
            points_list=points_list,
            voxel_size=self.voxel_size,
            coors_range=self.point_cloud_range,
            max_points=self.max_num_points,
            max_voxels=self.max_voxels
        )

        # 2. Voxel encoder (HardSimpleVFE: mean of points per voxel)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)

        # 3. Middle encoder (SparseEncoder: 3D sparse -> 2D dense)
        bev_features = self.middle_encoder(voxel_features, coors, batch_size)

        # 4. Backbone (SECOND: 2D feature extraction)
        backbone_features = self.backbone(bev_features)
        # backbone_features is a tuple: ([B, 128, 180, 180], [B, 256, 90, 90])

        if self.target_layer == 'backbone_block0':
            # Return first backbone block output only
            return backbone_features[0], voxel_infos

        # 5. Neck (SECONDFPN: upsample + concat)
        neck_features = self.neck(backbone_features)
        # neck_features is a list: [[B, 256, 180, 180], [B, 256, 180, 180]]

        if self.target_layer == 'neck':
            # Concatenate along channel dim to match gradient shape [B, 512, 180, 180]
            features = torch.cat(neck_features, dim=1)
            return features, voxel_infos

        # Default: return concatenated neck output
        features = torch.cat(neck_features, dim=1)
        return features, voxel_infos


# =============================================================================
# Unidirectional Chamfer Distance
# =============================================================================

def batched_chamfer_distance_bidirectional(
    adv_points_list: List[torch.Tensor],
    orig_points_list: List[torch.Tensor],
    chamfer_fn: ChamferDistance
) -> torch.Tensor:
    """
    Bidirectional Chamfer distance for a batch:
        dist_loss = dist(adv -> orig) + dist(orig -> adv)
    """
    total_dist = 0.0

    for adv_pts, orig_pts in zip(adv_points_list, orig_points_list):
        dist1 = chamfer_fn(
            adv_pts[:, :3].unsqueeze(0),
            orig_pts[:, :3].unsqueeze(0)
        )
        total_dist = total_dist + dist1

    return total_dist


# =============================================================================
# Main Attack Function
# =============================================================================

def run_batched_adversarial_attack(
    cfg_path: str,
    gradient_folder: str,
    result_save_path: str,
    checkpoint_path: str,
    data_root: str = None,
    device: str = 'cuda:0',
    batch_size: int = 4,
    num_iterations: int = 40,
    learning_rate: float = 0.01,
    dist_weight: float = 1.0,
    max_batches: int = None,
    target_layer: str = 'neck',
    init_noise_std: float = 0.3,
    loss_sign: float = 1.0,
    skip_existing: bool = False
):
    """
    Run batched adversarial attack on FocalFormer3D.

    Key behaviors:
    - Unidirectional Chamfer distance (dist1)
    - Saves FINAL iteration points (not best loss)
    - No data augmentation
    - skip_existing resumes an interrupted run (see the note at the skip site)
    """

    print("=" * 60)
    print(f"Batched FocalFormer3D Adversarial Attack")
    print(f"  Target layer: {target_layer}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Chamfer: UniDIRECTIONAL (dist1)")
    print(f"  Save: FINAL iteration (not best loss)")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Load config and import FocalFormer3D
    # -------------------------------------------------------------------------
    cfg = Config.fromfile(cfg_path)

    # Import custom modules (FocalFormer3D, hooks, etc.)
    if hasattr(cfg, 'custom_imports'):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)

    # Override dataloader settings for attack
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = min(batch_size * 2, 16)

    if data_root:
        cfg.train_dataloader.dataset.data_root = data_root

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    print("\nBuilding FocalFormer3D model...")
    full_model = MODELS.build(cfg.model).to(device)
    load_checkpoint(full_model, checkpoint_path, map_location=device)
    full_model.eval()
    print("✓ Model loaded")

    # -------------------------------------------------------------------------
    # Get voxelization config from model (FocalFormer3D does its own voxelization)
    # -------------------------------------------------------------------------
    voxel_cfg = cfg.model.pts_voxel_layer
    voxel_size = list(voxel_cfg.voxel_size)
    point_cloud_range = list(voxel_cfg.point_cloud_range)
    max_num_points = voxel_cfg.max_num_points
    max_voxels = voxel_cfg.max_voxels
    if isinstance(max_voxels, (list, tuple)):
        max_voxels = max_voxels[0]

    print(f"\nVoxelization config (from FocalFormer3D):")
    print(f"  Voxel size: {voxel_size}")
    print(f"  Point cloud range: {point_cloud_range}")
    print(f"  Max points per voxel: {max_num_points}")
    print(f"  Max voxels: {max_voxels}")

    # -------------------------------------------------------------------------
    # Create encoder (voxelize -> backbone -> neck)
    # -------------------------------------------------------------------------
    encoder = FocalFormer3DEncoder(
        voxel_encoder=full_model.pts_voxel_encoder,
        middle_encoder=full_model.pts_middle_encoder,
        backbone=full_model.pts_backbone,
        neck=full_model.pts_neck,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
        max_voxels=max_voxels,
        target_layer=target_layer
    ).to(device)
    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    # Distance function
    chamfer_dist = ChamferDistance().to(device)

    # -------------------------------------------------------------------------
    # Build dataloader via Runner
    # -------------------------------------------------------------------------
    cfg.work_dir = result_save_path
    cfg.default_scope = 'mmdet3d'
    cfg.load_from = checkpoint_path
    cfg.resume = False

    print("\nBuilding data loader...")
    runner = Runner.from_cfg(cfg)
    num_batches = len(runner.train_dataloader)
    print(f"✓ Data loader ready: {num_batches} batches, ~{num_batches * batch_size} samples")

    os.makedirs(result_save_path, exist_ok=True)

    print(f"\nAttack parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Distance weight: {dist_weight}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Target layer: {target_layer}")
    print("=" * 60)

    processed = 0
    skipped = 0
    resumed = 0
    all_batch_losses = []

    for batch_idx, data_batch in enumerate(runner.train_dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        # =================================================================
        # 1. Extract batch data
        # =================================================================
        data_samples = data_batch['data_samples']
        points_list_raw = data_batch['inputs']['points']
        actual_batch_size = len(data_samples)

        print(f"\n[Batch {batch_idx}/{num_batches}] Processing {actual_batch_size} samples...")

        # =================================================================
        # 2. Load gradients and filter valid samples
        # =================================================================
        valid_indices = []
        gradient_tensors = []
        lidar_filenames = []
        original_points_list = []

        for i, data_sample in enumerate(data_samples):
            meta = data_sample.metainfo

            # Get lidar path (handle multiple metadata formats)
            lidar_path = (
                meta.get('lidar_path', None)
                or meta.get('pts_filename', None)
            )
            if lidar_path is None:
                lp = meta.get('lidar_points', {})
                if isinstance(lp, dict):
                    lidar_path = lp.get('lidar_path', None)

            if lidar_path is None:
                skipped += 1
                continue

            lidar_filename = os.path.basename(lidar_path)

            # Resume support. FocalFormer runs ~52 batches/h, so a full pass is
            # ~29h against a 24h wall -- job 18882811 died at ~batch 1240 of
            # 1505 with every result on node-local $SLURM_TMPDIR. Re-staging
            # the rescued clouds and skipping them here turns a 24h restart
            # into an ~8h top-up.
            #
            # Safe because train_dataloader has shuffle=False and the attack is
            # deterministic given (points, gradient, seed): a frame redone in a
            # later job would land in the same place, so keeping the earlier
            # copy loses nothing. Cost is one wasted 11-sweep point-cloud load
            # per already-done frame, which is minutes, not hours.
            if skip_existing and os.path.exists(
                    os.path.join(result_save_path, lidar_filename)):
                resumed += 1
                continue

            # Derive sample_id the same way the gradient hook does
            sample_id = lidar_filename
            for ext in ['.pcd.bin', '.bin', '.npy']:
                if sample_id.endswith(ext):
                    sample_id = sample_id[:-len(ext)]
                    break

            # Try gradient filename patterns
            gradient_path = None
            for pattern in [
                f"{sample_id}_grad.pt",
                f"{sample_id}.pcd_grad.pt",
                f"{sample_id}grad.pt",
            ]:
                candidate = os.path.join(gradient_folder, pattern)
                if os.path.exists(candidate):
                    gradient_path = candidate
                    break

            if gradient_path is None:
                if batch_idx == 0 and i == 0:
                    # Debug: show what we're looking for
                    print(f"  DEBUG: Looking for gradient for sample_id='{sample_id}'")
                    print(f"  DEBUG: Gradient folder contents (first 5):")
                    try:
                        files = sorted(os.listdir(gradient_folder))[:5]
                        for f in files:
                            print(f"    {f}")
                    except Exception:
                        pass
                skipped += 1
                continue

            gradient_tensor = torch.load(gradient_path, map_location=device)
            points = points_list_raw[i].to(device).float()

            valid_indices.append(i)
            gradient_tensors.append(gradient_tensor)
            lidar_filenames.append(lidar_filename)
            original_points_list.append(points)

        if len(valid_indices) == 0:
            print(f"  No valid samples in batch, skipping...")
            continue

        current_batch_size = len(valid_indices)
        print(f"  Valid samples: {current_batch_size}/{actual_batch_size}")

        # =================================================================
        # 3. Stack gradients
        # =================================================================
        grad_shapes = [g.shape for g in gradient_tensors]
        if len(set(str(s) for s in grad_shapes)) > 1:
            print(f"  WARNING: Different gradient shapes: {grad_shapes}")
            print(f"  Processing sequentially...")

            for grad, orig_pts, filename in zip(gradient_tensors, original_points_list, lidar_filenames):
                _run_single_sample_attack(
                    encoder, grad, orig_pts, filename,
                    chamfer_dist, point_cloud_range,
                    num_iterations, learning_rate, dist_weight,
                    result_save_path, device, init_noise_std, loss_sign
                )
                processed += 1
            continue

        gradient_batch = torch.cat(gradient_tensors, dim=0)
        print(f"  Gradient batch shape: {gradient_batch.shape}")

        # =================================================================
        # 4. Initialize adversarial points
        # =================================================================
        adv_points_list = []
        for orig_pts in original_points_list:
            adv_pts = orig_pts.clone().detach()
            # sigma was hardcoded at 0.3. On BEVFusion that magnitude (~387 mm
            # of thrashing) dominated the perturbation so completely that it
            # masked a DEAD adversarial gradient for months -- sweep 18798999
            # showed sigma=0 is 100% attack-driven while 0.3 is not. Kept as
            # the default for parity with the existing tars; set 0 to measure
            # what the attack alone does.
            if init_noise_std > 0:
                noise = torch.normal(0, init_noise_std,
                                     size=(orig_pts.shape[0], 3), device=device)
                adv_pts[:, :3] = adv_pts[:, :3] + noise
            adv_pts.requires_grad_(True)
            adv_points_list.append(adv_pts)

        optimizer = optim.Adam(adv_points_list, lr=learning_rate)

        # =================================================================
        # 5. Optimization loop
        # =================================================================
        batch_loss_history = {
            'adv_loss': [], 'dist_loss': [], 'total_loss': []
        }
        best_loss = float('inf')
        best_iteration = -1

        for it in range(num_iterations):
            optimizer.zero_grad()

            # Bidirectional Chamfer distance
            dist_loss = batched_chamfer_distance_bidirectional(
                adv_points_list, original_points_list, chamfer_dist
            )

            # Forward through encoder
            try:
                features, voxel_infos = encoder(adv_points_list)
            except Exception as e:
                print(f"  Forward failed at iter {it}: {e}")
                break

            if features.shape != gradient_batch.shape:
                print(f"  Shape mismatch: features {features.shape} vs gradient {gradient_batch.shape}")
                break

            # Adversarial loss: maximize alignment with extracted gradient
            # loss_sign=0 drops the adversarial term entirely. That is the
            # control that PROVED BEVFusion's gradient was dead (job 18858947:
            # mAP 0.1840 vs 0.1830, paired p=0.49 over 124,831 boxes). Running
            # it here is how we show FocalFormer's term now DOES contribute.
            adv_loss = loss_sign * torch.sum(gradient_batch * features)
            total_loss = adv_loss + dist_weight * dist_loss

            # Track
            batch_loss_history['adv_loss'].append(adv_loss.item())
            batch_loss_history['dist_loss'].append(dist_loss.item())
            batch_loss_history['total_loss'].append(total_loss.item())

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_iteration = it

            # Backward
            total_loss.backward()

            if it % 10 == 0 or it == num_iterations - 1:
                grad_norms = [p.grad.norm().item() if p.grad is not None else 0 for p in adv_points_list]
                avg_grad_norm = sum(grad_norms) / len(grad_norms)
                print(f"    Iter {it:3d}: adv={adv_loss.item():.4f}, "
                      f"dist={dist_loss.item():.4f}, total={total_loss.item():.4f}, "
                      f"grad_norm={avg_grad_norm:.6f}")

            optimizer.step()

            # Clamp to point cloud range
            with torch.no_grad():
                for adv_pts in adv_points_list:
                    adv_pts[:, 0].clamp_(point_cloud_range[0], point_cloud_range[3])
                    adv_pts[:, 1].clamp_(point_cloud_range[1], point_cloud_range[4])
                    adv_pts[:, 2].clamp_(point_cloud_range[2], point_cloud_range[5])

        # Store loss history
        all_batch_losses.append({
            'batch_idx': batch_idx,
            'filenames': lidar_filenames,
            'loss_history': batch_loss_history,
            'best_loss': best_loss,
            'best_iteration': best_iteration,
            'final_loss': batch_loss_history['total_loss'][-1] if batch_loss_history['total_loss'] else None
        })

        # =================================================================
        # 6. Save FINAL iteration results
        # =================================================================
        print(f"  Saving FINAL iteration (iter {num_iterations - 1}), best was iter {best_iteration}")

        for adv_pts, filename in zip(adv_points_list, lidar_filenames):
            save_path = os.path.join(result_save_path, filename)
            adv_pts.detach().cpu().numpy().astype(np.float32).tofile(save_path)
            processed += 1

        print(f"  Saved {len(lidar_filenames)} files (final iteration)")

        # Cleanup
        del gradient_batch, gradient_tensors, adv_points_list, original_points_list
        torch.cuda.empty_cache()

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"Attack complete!")
    # Count from the filesystem, never from the counters. processed+resumed
    # printed 6019 on the 18931271 resume while the truth was 5980: `resumed`
    # over-reports by exactly the 39 frames that have no GT and therefore no
    # gradient, so a derived total contradicts the authoritative count and
    # invites someone to believe the wrong one.
    on_disk = len([f for f in os.listdir(result_save_path)
                   if f.endswith('.bin')])
    print(f"  Processed this run: {processed}")
    print(f"  Skipped (no gradient): {skipped}")
    print(f"  Already present (resume counter): {resumed}")
    print(f"  Total clouds on disk: {on_disk}")
    print(f"  Results: {result_save_path}")
    print("=" * 60)

    if all_batch_losses:
        final_losses = [b['final_loss'] for b in all_batch_losses if b['final_loss'] is not None]
        best_losses = [b['best_loss'] for b in all_batch_losses]

        print(f"\nLoss Statistics:")
        print(f"  Final losses - Mean: {np.mean(final_losses):.4f}, "
              f"Min: {np.min(final_losses):.4f}, Max: {np.max(final_losses):.4f}")
        print(f"  Best losses  - Mean: {np.mean(best_losses):.4f}, "
              f"Min: {np.min(best_losses):.4f}, Max: {np.max(best_losses):.4f}")

    loss_history_path = os.path.join(result_save_path, 'loss_history.pt')
    torch.save(all_batch_losses, loss_history_path)
    print(f"  Loss history saved to: {loss_history_path}")
    print("=" * 60)


def _run_single_sample_attack(
    encoder, gradient, orig_pts, filename,
    chamfer_fn, point_cloud_range,
    num_iterations, learning_rate, dist_weight,
    result_save_path, device, init_noise_std=0.3, loss_sign=1.0
):
    """Fallback: attack a single sample (for shape-mismatched batches)."""
    adv_pts = orig_pts.clone().detach()
    if init_noise_std > 0:
        noise = torch.normal(0, init_noise_std,
                             size=(orig_pts.shape[0], 3), device=device)
        adv_pts[:, :3] = adv_pts[:, :3] + noise
    adv_pts.requires_grad_(True)

    optimizer = optim.Adam([adv_pts], lr=learning_rate)

    for it in range(num_iterations):
        optimizer.zero_grad()

        # Unidirectional Chamfer
        dist1 = chamfer_fn(adv_pts[:, :3].unsqueeze(0), orig_pts[:, :3].unsqueeze(0))
        dist_loss = dist1

        try:
            features, _ = encoder([adv_pts])
        except Exception as e:
            print(f"    Forward failed: {e}")
            break

        adv_loss = loss_sign * torch.sum(gradient * features)
        total_loss = adv_loss + dist_weight * dist_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            adv_pts[:, 0].clamp_(point_cloud_range[0], point_cloud_range[3])
            adv_pts[:, 1].clamp_(point_cloud_range[1], point_cloud_range[4])
            adv_pts[:, 2].clamp_(point_cloud_range[2], point_cloud_range[5])

    save_path = os.path.join(result_save_path, filename)
    adv_pts.detach().cpu().numpy().astype(np.float32).tofile(save_path)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batched FocalFormer3D Adversarial Attack')
    parser.add_argument('--cfg', type=str, required=True,
                        help='FocalFormer3D gradient extraction config (has model + dataloader)')
    parser.add_argument('--grads', type=str, required=True,
                        help='Directory containing *_grad.pt files')
    parser.add_argument('--results', type=str, required=True,
                        help='Directory to save adversarial point clouds')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='FocalFormer3D checkpoint path')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data_root in config')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (FocalFormer3D uses more VRAM than CenterPoint)')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dist_weight', type=float, default=1.0)
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit number of batches (for debugging)')
    parser.add_argument('--target_layer', type=str, default='neck',
                        choices=['neck', 'backbone_block0'],
                        help='Which layer gradients were extracted from')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument(
        '--init_noise_std', type=float, default=0.3,
        help='sigma of the initial Gaussian jitter, metres. 0.3 matches the '
             'existing tars; 0 makes the perturbation purely attack-driven.')
    parser.add_argument(
        '--loss_sign', type=float, default=1.0,
        help='multiplies the adversarial term. 1.0 is the trained default; '
             '0 removes it entirely, which is the control for whether the '
             'gradient contributes at all.')
    parser.add_argument(
        '--skip_existing', action='store_true',
        help='skip any frame whose output .bin is already in --results. '
             'Stage a partial run into --results and this resumes it instead '
             'of redoing ~24h of work.')

    args = parser.parse_args()

    os.makedirs(args.results, exist_ok=True)

    run_batched_adversarial_attack(
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
        target_layer=args.target_layer,
        init_noise_std=args.init_noise_std,
        loss_sign=args.loss_sign,
        skip_existing=args.skip_existing
    )