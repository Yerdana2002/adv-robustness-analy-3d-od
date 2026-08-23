"""Utility functions for bounding box operations.

Adapted from DETR3D/PETR normalize_bbox for BEVFormer compatibility.
"""

from typing import List
import torch
from torch import Tensor


def normalize_bbox(bboxes: Tensor, pc_range: List = None) -> Tensor:
    """Normalize bboxes for BEVFormer head.

    Args:
        bboxes (Tensor): Boxes with unnormalized coordinates
            (cx, cy, cz, L, W, H, φ, v_x, v_y). Shape [num_gt, 9].
        pc_range (List): Perception range (unused, kept for API compat).

    Returns:
        Tensor: Normalized boxes (cx, cy, L, W, cz, H, sin(φ), cos(φ), v_x, v_y).
            Shape [num_query, 10].
    """
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    L = bboxes[..., 3:4].log()
    W = bboxes[..., 4:5].log()
    H = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, L, W, cz, H, rot.sin(), rot.cos(), vx, vy), dim=-1)
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, L, W, cz, H, rot.sin(), rot.cos()), dim=-1)
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range=None):
    """Denormalize bboxes from BEVFormer head output.

    Args:
        normalized_bboxes (Tensor): (cx, cy, L, W, cz, H, sin(φ), cos(φ), v_x, v_y).
        pc_range (List): Perception range (unused, kept for API compat).

    Returns:
        Tensor: Denormalized boxes.
    """
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the BEV
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    L = normalized_bboxes[..., 2:3].exp()
    W = normalized_bboxes[..., 3:4].exp()
    H = normalized_bboxes[..., 5:6].exp()

    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        denormalized_bboxes = torch.cat(
            (cx, cy, cz, L, W, H, rot, vx, vy), dim=-1)
    else:
        denormalized_bboxes = torch.cat(
            (cx, cy, cz, L, W, H, rot), dim=-1)

    return denormalized_bboxes