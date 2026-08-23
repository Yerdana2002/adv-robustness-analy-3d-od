# =============================================================================
# vis_utils.py — refactored for mmdet3d >= 1.1 / v1.4.x
# =============================================================================
# Changes from old version:
#   - No mmdet3d/mmcv imports to migrate (pure Open3D/NumPy utility)
#   - print() → logger.info()
#   - Added debug logging
# =============================================================================
import logging

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def save_point_cloud(xyz, filename='pc.ply', color=None):
    """Save a point cloud to PLY file.

    Args:
        xyz (np.ndarray): Point coordinates [N, 3].
        filename (str): Output file path.
        color (np.ndarray, optional): Per-point RGB colors [N, 3].
            Defaults to all white.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(xyz) if color is None else color)
    o3d.io.write_point_cloud(filename, pcd)

    logger.info(f'[vis_utils] Point cloud saved to {filename} '
                f'({len(xyz)} points)')


def save_box_corners(boxes_corners_points, filename='box.ply',
                     color=(1., 0., 0.)):
    """Save 3D bounding box corners as a line set to PLY file.

    Args:
        boxes_corners_points (np.ndarray): Box corners [num_boxes, 8, 3].
        filename (str): Output file path.
        color (tuple): RGB color for box edges.

    Returns:
        o3d.geometry.LineSet: The constructed line set.
    """
    box_lines = np.array([
        [2, 3], [0, 3], [4, 5], [4, 7], [5, 6], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [0, 1],  # front down edge
        [1, 2],  # right down edge
    ])

    points = boxes_corners_points.reshape(-1, 3)

    lines = []
    for i, b in enumerate(boxes_corners_points):
        lines.append(box_lines + i * 8)
    lines = np.concatenate(lines)

    colors = np.array([color for _ in range(len(lines))])

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_line_set(filename, lineset)

    logger.info(f'[vis_utils] Line set saved to {filename} '
                f'({len(boxes_corners_points)} boxes, {len(lines)} edges)')
    return lineset