"""
Batched Adversarial Attack for FocalFormer3D on Waymo
======================================================
Adapted from NuScenes FocalFormer3D attack script.

Key differences from NuScenes version:
  - Waymo uses single-sweep point clouds (no temporal accumulation)
  - Point features: (x, y, z, intensity, elongation) — 5 dims, single sweep
  - Waymo point cloud range: [-76.8, -76.8, -2, 76.8, 76.8, 4]
  - Voxel size: [0.1, 0.1, 0.15], sparse shape: [41, 1536, 1536]
  - HardVFE encoder (not HardSimpleVFE) with feat_channels=[64]
  - sample_idx used for gradient matching (not lidar filename)
  - Chamfer distance: UNIDIRECTIONAL (adv -> orig only)
  - Dataloader loads only samples in the provided infos pkl
    (e.g. waymo_infos_val_15.pkl = 1/5 subset ~7960 frames)

Attack pipeline:
  1. Load clean val point cloud via dataloader
  2. Load pre-extracted gradient for that sample (matched by sample_idx)
  3. Initialize adversarial points = clean + small noise
  4. For each iteration:
     a. Voxelize adversarial points (differentiable mapping)
     b. Forward through encoder up to neck
     c. loss = dot(gradient, features) + lambda * chamfer_dist(adv -> orig)
     d. Backprop and update point positions
  5. Save final adversarial point cloud

Usage:
  python attack_focalformer_waymo.py \
      --cfg projects/configs/focalformer3d/FocalFormer3D_Waymo_L.py \
      --grads /path/to/waymo_neck_gradients \
      --results /path/to/save/adversarial \
      --checkpoint /path/to/FocalFormer3d_Waymo_converted.pth \
      --batch_size 2 --iterations 40 --lr 0.01 --dist_weight 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
from typing import List, Tuple
import argparse

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
# Differentiable Voxelization
# =============================================================================

def hard_voxelize_v2(
    points: torch.Tensor,
    voxel_size: list,
    coors_range: list,
    max_points: int = 5,
    max_voxels: int = 150000,
    NDim: int = 3,
    deterministic: bool = True
) -> dict:
    """Single sample voxelization using mmcv C++ extension."""
    device = points.device
    num_points = points.shape[0]
    num_features = points.shape[1]

    voxel_size_tensor = torch.tensor(voxel_size, dtype=torch.float32, device='cpu')
    coors_range_tensor = torch.tensor(coors_range, dtype=torch.float32, device='cpu')

    voxels = torch.zeros((max_voxels, max_points, num_features),
                         dtype=points.dtype, device=device)
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
    max_points: int = 5,
    max_voxels: int = 150000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
    """Voxelize a batch of point clouds and combine with batch indices."""
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

        voxels = voxel_result['voxels']
        coors = voxel_result['coors']
        num_pts = voxel_result['num_points_per_voxel']

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
# FocalFormer3D Waymo Encoder — points through neck
# =============================================================================

class FocalFormer3DWaymoEncoder(nn.Module):
    """
    Differentiable encoder for FocalFormer3D on Waymo.

    Architecture:
        HardVFE (feat_channels=[64])
        -> SparseEncoder (sparse_shape=[41, 1536, 1536])
        -> SECOND backbone
        -> SECONDFPN neck  [B, 512, 192, 192]  <-- gradient target

    Note: Waymo sparse_shape is [41, 1536, 1536] vs NuScenes [41, 1024, 1024]
    due to the larger point cloud range [-76.8, 76.8] vs [-54, 54].
    """

    def __init__(
        self,
        voxel_encoder: nn.Module,
        middle_encoder: nn.Module,
        backbone: nn.Module,
        neck: nn.Module,
        voxel_size: list,
        point_cloud_range: list,
        max_num_points: int = 5,
        max_voxels: int = 150000,
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

    def forward(
        self,
        points_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[dict]]:
        batch_size = len(points_list)

        # 1. Voxelize
        voxels, coors, num_points, voxel_infos = batched_voxelize(
            points_list=points_list,
            voxel_size=self.voxel_size,
            coors_range=self.point_cloud_range,
            max_points=self.max_num_points,
            max_voxels=self.max_voxels
        )

        # 2. HardVFE: produces [N_voxels, 64]
        voxel_features = self.voxel_encoder(voxels, num_points, coors)

        # 3. SparseEncoder: 3D sparse -> 2D dense BEV [B, 128, 192, 192]
        bev_features = self.middle_encoder(voxel_features, coors, batch_size)

        # 4. SECOND backbone: -> ([B,128,192,192], [B,256,96,96])
        backbone_features = self.backbone(bev_features)

        if self.target_layer == 'backbone_block0':
            return backbone_features[0], voxel_infos

        # 5. SECONDFPN neck: upsample + concat -> [B,256,192,192], [B,256,192,192]
        neck_features = self.neck(backbone_features)

        # Concatenate along channels -> [B, 512, 192, 192]
        features = torch.cat(neck_features, dim=1)
        return features, voxel_infos


# =============================================================================
# Unidirectional Chamfer Distance (adv -> orig only)
# =============================================================================

def chamfer_unidirectional(
    adv_points_list: List[torch.Tensor],
    orig_points_list: List[torch.Tensor],
    chamfer_fn: ChamferDistance
) -> torch.Tensor:
    total_dist = torch.tensor(0.0, device=adv_points_list[0].device)

    for adv_pts, orig_pts in zip(adv_points_list, orig_points_list):
        out = chamfer_fn(
            adv_pts[:, :3].unsqueeze(0).float(),
            orig_pts[:, :3].unsqueeze(0).float(),
            bidirectional=False
        )
        dist1 = out[0] if isinstance(out, (tuple, list)) else out
        total_dist = total_dist + dist1.mean()

    return total_dist / len(adv_points_list)


# =============================================================================
# Gradient filename resolution for Waymo
# =============================================================================

def _get_lidar_stem(meta) -> str | None:
    lidar_path = (
        meta.get('lidar_path', None) or
        meta.get('pts_filename', None)
    )
    if lidar_path is None:
        lp = meta.get('lidar_points', {})
        if isinstance(lp, dict):
            lidar_path = lp.get('lidar_path', None)

    if lidar_path is None:
        return None

    stem = os.path.basename(lidar_path)
    for ext in ['.pcd.bin', '.bin', '.npy']:
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break
    return stem


def find_gradient_file(gradient_folder: str, data_sample) -> str | None:
    meta = data_sample.metainfo

    stem = _get_lidar_stem(meta)
    if stem is not None:
        path = os.path.join(gradient_folder, f"{stem}_grad.pt")
        if os.path.exists(path):
            return path

    sample_idx = meta.get('sample_idx', None)
    if sample_idx is not None:
        for pattern in [
            f"{sample_idx}_grad.pt",
            f"{int(sample_idx):07d}_grad.pt",
        ]:
            path = os.path.join(gradient_folder, pattern)
            if os.path.exists(path):
                return path

    return None


def get_save_filename(data_sample) -> str:
    meta = data_sample.metainfo

    stem = _get_lidar_stem(meta)
    if stem is not None:
        return f"{stem}.bin"

    sample_idx = meta.get('sample_idx', None)
    if sample_idx is not None:
        return f"{int(sample_idx):07d}.bin"

    lidar_path = (
        meta.get('lidar_path', None) or
        meta.get('pts_filename', None)
    )
    if lidar_path is not None:
        return os.path.basename(lidar_path)

    return f"unknown_{id(data_sample)}.bin"


# =============================================================================
# Single sample attack (fallback for shape-mismatched batches)
# =============================================================================

def _run_single_sample_attack(
    encoder: FocalFormer3DWaymoEncoder,
    gradient: torch.Tensor,
    orig_pts: torch.Tensor,
    save_filename: str,
    chamfer_fn: ChamferDistance,
    point_cloud_range: list,
    num_iterations: int,
    learning_rate: float,
    dist_weight: float,
    result_save_path: str,
    device: str
):
    adv_pts = orig_pts.clone().detach()
    noise = torch.normal(0, 0.3, size=(orig_pts.shape[0], 3), device=device)
    adv_pts[:, :3] = adv_pts[:, :3] + noise
    adv_pts.requires_grad_(True)

    optimizer = optim.Adam([adv_pts], lr=learning_rate)

    for it in range(num_iterations):
        optimizer.zero_grad()

        dist_loss = chamfer_unidirectional([adv_pts], [orig_pts], chamfer_fn)

        try:
            features, _ = encoder([adv_pts])
        except Exception as e:
            print(f"    Forward failed at iter {it}: {e}", flush=True)
            break

        adv_loss = torch.sum(gradient * features)
        total_loss = adv_loss + dist_weight * dist_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            adv_pts[:, 0].clamp_(point_cloud_range[0], point_cloud_range[3])
            adv_pts[:, 1].clamp_(point_cloud_range[1], point_cloud_range[4])
            adv_pts[:, 2].clamp_(point_cloud_range[2], point_cloud_range[5])

    save_path = os.path.join(result_save_path, save_filename)
    adv_pts.detach().cpu().numpy().astype(np.float32).tofile(save_path)
    print(f"    Saved: {save_filename}", flush=True)


# =============================================================================
# Main Attack
# =============================================================================

def run_waymo_adversarial_attack(
    cfg_path: str,
    gradient_folder: str,
    result_save_path: str,
    checkpoint_path: str,
    data_root: str = None,
    device: str = 'cuda:0',
    batch_size: int = 2,
    num_iterations: int = 40,
    learning_rate: float = 0.01,
    dist_weight: float = 1.0,
    max_batches: int = None,
    target_layer: str = 'neck',
    rank: int = 0,
    world_size: int = 1,
    skip_existing: bool = True,
):
    print("=" * 60, flush=True)
    print(f"FocalFormer3D Waymo Adversarial Attack  [rank {rank}/{world_size}]", flush=True)
    print(f"  Target layer : {target_layer}", flush=True)
    print(f"  Batch size   : {batch_size}", flush=True)
    print(f"  Iterations   : {num_iterations}", flush=True)
    print(f"  LR           : {learning_rate}", flush=True)
    print(f"  Dist weight  : {dist_weight}", flush=True)
    print(f"  Chamfer      : UNIDIRECTIONAL (adv -> orig)", flush=True)
    print(f"  Save         : FINAL iteration", flush=True)
    print(f"  Skip existing: {skip_existing}", flush=True)
    print(f"  Device       : {device}", flush=True)
    print("=" * 60, flush=True)

    # -------------------------------------------------------------------------
    # Load config and register custom modules
    # -------------------------------------------------------------------------
    cfg = Config.fromfile(cfg_path)

    if hasattr(cfg, 'custom_imports'):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)

    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = min(batch_size * 2, 8)

    if data_root:
        cfg.train_dataloader.dataset.data_root = data_root

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    print(f"\n[rank {rank}] Building FocalFormer3D model...", flush=True)
    full_model = MODELS.build(cfg.model).to(device)
    load_checkpoint(full_model, checkpoint_path, map_location=device)
    full_model.eval()
    print(f"[rank {rank}] ✓ Model loaded", flush=True)

    # -------------------------------------------------------------------------
    # Voxelization config (Waymo-specific)
    # -------------------------------------------------------------------------
    voxel_cfg = cfg.model.pts_voxel_layer
    voxel_size = list(voxel_cfg.voxel_size)
    point_cloud_range = list(voxel_cfg.point_cloud_range)
    max_num_points = voxel_cfg.max_num_points
    max_voxels = voxel_cfg.max_voxels
    if isinstance(max_voxels, (list, tuple)):
        max_voxels = max_voxels[0]

    print(f"\n[rank {rank}] Waymo voxelization config:", flush=True)
    print(f"  voxel_size        : {voxel_size}", flush=True)
    print(f"  point_cloud_range : {point_cloud_range}", flush=True)
    print(f"  max_num_points    : {max_num_points}", flush=True)
    print(f"  max_voxels        : {max_voxels}", flush=True)

    # -------------------------------------------------------------------------
    # Build encoder
    # -------------------------------------------------------------------------
    encoder = FocalFormer3DWaymoEncoder(
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

    chamfer_fn = ChamferDistance().to(device)

    # -------------------------------------------------------------------------
    # Build dataloader via Runner (same pipeline as single-GPU)
    # -------------------------------------------------------------------------
    cfg.work_dir = result_save_path
    cfg.default_scope = 'mmdet3d'
    cfg.load_from = checkpoint_path
    cfg.resume = False

    print(f"\n[rank {rank}] Building dataloader...", flush=True)
    runner = Runner.from_cfg(cfg)
    num_batches = len(runner.train_dataloader)
    total_samples = num_batches * batch_size
    print(f"[rank {rank}] ✓ Dataloader ready: {num_batches} batches (~{total_samples} samples)", flush=True)
    if world_size > 1:
        my_batches = (num_batches + world_size - 1) // world_size
        print(f"[rank {rank}] DDP sharding: ~{my_batches} batches for this rank", flush=True)

    # Debug: show first few gradient files to confirm naming
    try:
        grad_files = sorted(os.listdir(gradient_folder))
        print(f"\n[rank {rank}] Gradient folder: {gradient_folder}", flush=True)
        print(f"  Total gradient files: {len(grad_files)}", flush=True)
        print(f"  First 5: {grad_files[:5]}", flush=True)
    except Exception as e:
        print(f"  Could not list gradient folder: {e}", flush=True)

    os.makedirs(result_save_path, exist_ok=True)

    # -------------------------------------------------------------------------
    # Attack loop
    # -------------------------------------------------------------------------
    processed = 0
    skipped = 0
    already_done = 0
    all_batch_losses = []

    for batch_idx, data_batch in enumerate(runner.train_dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        data_samples = data_batch['data_samples']
        points_list_raw = data_batch['inputs']['points']
        actual_batch_size = len(data_samples)

        print(f"\n[rank {rank}][Batch {batch_idx+1}/{num_batches}] {actual_batch_size} samples...",
              flush=True)

        # ---------------------------------------------------------------------
        # Match samples to gradient files
        # ---------------------------------------------------------------------
        valid_indices = []
        gradient_tensors = []
        save_filenames = []
        original_points_list = []

        for i, data_sample in enumerate(data_samples):
            save_fn = get_save_filename(data_sample)

            # Skip if already exists on disk
            if skip_existing:
                out_path = os.path.join(result_save_path, save_fn)
                if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                    already_done += 1
                    continue

            gradient_path = find_gradient_file(gradient_folder, data_sample)

            if gradient_path is None:
                # Debug info for first missing sample
                if skipped == 0:
                    meta = data_sample.metainfo
                    print(f"  [DEBUG] First missing gradient:", flush=True)
                    print(f"    sample_idx   = {meta.get('sample_idx', 'N/A')}", flush=True)
                    print(f"    context_name = {meta.get('context_name', 'N/A')}", flush=True)
                    print(f"    timestamp    = {meta.get('timestamp', 'N/A')}", flush=True)
                skipped += 1
                continue

            gradient_tensor = torch.load(gradient_path, map_location=device)
            points = points_list_raw[i].to(device).float()

            valid_indices.append(i)
            gradient_tensors.append(gradient_tensor)
            save_filenames.append(save_fn)
            original_points_list.append(points)

        if len(valid_indices) == 0:
            print(f"  No valid samples (no matching gradients or all done), skipping batch",
                  flush=True)
            continue

        current_batch_size = len(valid_indices)
        print(f"  Valid: {current_batch_size}/{actual_batch_size} "
              f"(skipped {actual_batch_size - current_batch_size} missing/done)",
              flush=True)

        # ---------------------------------------------------------------------
        # Handle shape mismatches (different num points per sample)
        # ---------------------------------------------------------------------
        grad_shapes = [g.shape for g in gradient_tensors]
        all_same_shape = len(set(str(s) for s in grad_shapes)) == 1

        if not all_same_shape:
            print(f"  Gradient shapes differ: {grad_shapes} — processing sequentially",
                  flush=True)
            for grad, orig_pts, save_fn in zip(
                    gradient_tensors, original_points_list, save_filenames):
                _run_single_sample_attack(
                    encoder, grad, orig_pts, save_fn,
                    chamfer_fn, point_cloud_range,
                    num_iterations, learning_rate, dist_weight,
                    result_save_path, device
                )
                processed += 1
            continue

        gradient_batch = torch.cat(gradient_tensors, dim=0)
        print(f"  Gradient batch shape: {gradient_batch.shape}", flush=True)

        # ---------------------------------------------------------------------
        # Initialize adversarial points with small noise
        # ---------------------------------------------------------------------
        adv_points_list = []
        for orig_pts in original_points_list:
            adv_pts = orig_pts.clone().detach()
            noise = torch.randn(orig_pts.shape[0], 3, device=device) * 0.3
            adv_pts[:, :3] = adv_pts[:, :3] + noise
            adv_pts.requires_grad_(True)
            adv_points_list.append(adv_pts)

        optimizer = optim.Adam(adv_points_list, lr=learning_rate)

        # ---------------------------------------------------------------------
        # Optimization loop
        # ---------------------------------------------------------------------
        loss_history = {'adv_loss': [], 'dist_loss': [], 'total_loss': []}
        best_loss = float('inf')
        best_iteration = -1

        for it in range(num_iterations):
            optimizer.zero_grad()

            # Unidirectional Chamfer: adv -> orig
            dist_loss = chamfer_unidirectional(
                adv_points_list, original_points_list, chamfer_fn
            )

            # Forward through encoder
            try:
                features, _ = encoder(adv_points_list)
            except Exception as e:
                print(f"  Forward failed at iter {it}: {e}", flush=True)
                break

            # Shape check
            if features.shape != gradient_batch.shape:
                print(f"  Shape mismatch: features {features.shape} "
                      f"vs gradient {gradient_batch.shape}", flush=True)
                break

            # Adversarial loss: align features with extracted gradient direction
            adv_loss = torch.sum(gradient_batch * features)
            total_loss = adv_loss + dist_weight * dist_loss

            loss_history['adv_loss'].append(adv_loss.item())
            loss_history['dist_loss'].append(dist_loss.item())
            loss_history['total_loss'].append(total_loss.item())

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_iteration = it

            total_loss.backward()

            if it % 10 == 0 or it == num_iterations - 1:
                grad_norms = [
                    p.grad.norm().item() if p.grad is not None else 0.0
                    for p in adv_points_list
                ]
                print(f"    iter {it:3d}: adv={adv_loss.item():.4f}  "
                      f"dist={dist_loss.item():.6f}  "
                      f"total={total_loss.item():.4f}  "
                      f"grad_norm={np.mean(grad_norms):.6f}",
                      flush=True)

            optimizer.step()

            # Clamp xyz to valid Waymo range
            with torch.no_grad():
                for adv_pts in adv_points_list:
                    adv_pts[:, 0].clamp_(point_cloud_range[0], point_cloud_range[3])
                    adv_pts[:, 1].clamp_(point_cloud_range[1], point_cloud_range[4])
                    adv_pts[:, 2].clamp_(point_cloud_range[2], point_cloud_range[5])

        all_batch_losses.append({
            'batch_idx': batch_idx,
            'filenames': save_filenames,
            'loss_history': loss_history,
            'best_loss': best_loss,
            'best_iteration': best_iteration,
        })

        # ---------------------------------------------------------------------
        # Save final iteration adversarial point clouds
        # ---------------------------------------------------------------------
        print(f"  Saving final iter {num_iterations-1} "
              f"(best was iter {best_iteration})", flush=True)

        for adv_pts, save_fn in zip(adv_points_list, save_filenames):
            save_path = os.path.join(result_save_path, save_fn)
            adv_pts.detach().cpu().numpy().astype(np.float32).tofile(save_path)
            processed += 1
            print(f"    Saved: {save_fn}", flush=True)

        # Cleanup GPU memory
        del gradient_batch, gradient_tensors, adv_points_list, original_points_list
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print(f"[rank {rank}] Attack complete!", flush=True)
    print(f"  Processed    : {processed}", flush=True)
    print(f"  Skipped      : {skipped}", flush=True)
    print(f"  Already done : {already_done}", flush=True)
    print(f"  Results      : {result_save_path}", flush=True)

    if all_batch_losses:
        final_losses = [
            b['loss_history']['total_loss'][-1]
            for b in all_batch_losses
            if b['loss_history']['total_loss']
        ]
        best_losses = [b['best_loss'] for b in all_batch_losses]
        if final_losses:
            print(f"\n  Final loss — mean: {np.mean(final_losses):.4f}  "
                  f"min: {np.min(final_losses):.4f}  "
                  f"max: {np.max(final_losses):.4f}", flush=True)
        print(f"  Best loss  — mean: {np.mean(best_losses):.4f}  "
              f"min: {np.min(best_losses):.4f}  "
              f"max: {np.max(best_losses):.4f}", flush=True)

    loss_path = os.path.join(result_save_path, f'loss_history_rank{rank}.pt')
    torch.save(all_batch_losses, loss_path)
    print(f"\n  Loss history: {loss_path}", flush=True)
    print("=" * 60, flush=True)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FocalFormer3D Waymo Adversarial Attack')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Waymo FocalFormer3D config (with train_dataloader)')
    parser.add_argument('--grads', type=str, required=True,
                        help='Directory containing *_grad.pt gradient files')
    parser.add_argument('--results', type=str, required=True,
                        help='Directory to save adversarial point clouds')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='FocalFormer3D Waymo checkpoint path')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data_root in config')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (Waymo uses more VRAM: 1536x1536 BEV)')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dist_weight', type=float, default=1.0)
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit batches for debugging')
    parser.add_argument('--target_layer', type=str, default='neck',
                        choices=['neck', 'backbone_block0'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--no_skip_existing', action='store_false', dest='skip_existing')

    args = parser.parse_args()
    os.makedirs(args.results, exist_ok=True)

    run_waymo_adversarial_attack(
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
        rank=0,
        world_size=1,
        skip_existing=args.skip_existing,
    )   