from mmdet3d.datasets import WaymoDataset
from mmdet.datasets import DATASETS
import os

@DATASETS.register_module()
class WaymoReducedDataset(WaymoDataset):
    """
    Smaller version of the Waymo dataset. It only contains a specified amount of random samples from each scene.
    This allows us to speed up experiments, while still experimenting on every scene!
    """

    def __init__(self, samples_per_scene=5, seed=42, **kwargs):
        self.samples_per_scene = samples_per_scene
        self.seed = seed
        super().__init__(**kwargs)

        self._reduce_by_equal_timestamp_groups()

    def _reduce_by_equal_timestamp_groups(self):
        """
        Looking at the timestamp data, it looks like there are 199 samples with the same timestamp. 
        I assume that the same timestamp equals being in the same scene!
        """
        import random
        random.seed(self.seed)

        frames = self.data_infos
        scenes = []
        current_scene = [frames[0]]
        last_ts = frames[0]["timestamp"]

        for info in frames[1:]:
            ts = info["timestamp"]
            if ts == last_ts:
                # same scene
                current_scene.append(info)
            else:
                # new scene begins
                scenes.append(current_scene)
                current_scene = [info]
                last_ts = ts

        scenes.append(current_scene)

        # sample N frames per scene
        reduced = []
        for scene in scenes:
            if len(scene) > self.samples_per_scene:
                scene = random.sample(scene, self.samples_per_scene)
            reduced.extend(scene)

        # print(f"[WaymoReducedDataset] Original frames: {len(frames)}")
        # print(f"[WaymoReducedDataset] Scenes found:    {len(scenes)}")
        # print(f"[WaymoReducedDataset] Reduced frames: {len(reduced)}")
        self.data_infos = reduced
