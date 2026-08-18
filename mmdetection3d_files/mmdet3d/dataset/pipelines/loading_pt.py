import torch
import numpy as np

from mmdet3d.registry import PIPELINES
from mmdet3d.structures.points import get_points_type

@PIPELINES.register_module()
class LoadPointsFromPT:

    def __init__(
        self,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4
    ):
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim

    def transform(self, results):

        pts_file = results['lidar_points']['lidar_path']

        points = torch.load(pts_file)

        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()

        points = points.reshape(-1, self.load_dim)
        points = points[:, :self.use_dim]

        points_class = get_points_type(self.coord_type)

        points = points_class(
            points,
            points_dim=points.shape[-1]
        )

        results['points'] = points

        return results