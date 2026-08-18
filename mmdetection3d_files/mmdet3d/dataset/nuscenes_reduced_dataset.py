from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class NuScenesReducedDataset(NuScenesDataset):
    """
    Smaller version of the NuScenes dataset. It only contains a specified amount of random samples from each scene.
    This allows us to speed up experiments, while still experimenting on every scene!
    """

    def __init__(self, samples_per_scene=5, seed=None, **kwargs):
        self.samples_per_scene = samples_per_scene
        self.seed = seed if seed else 42 #TODO: make seed random if not given
        super().__init__(**kwargs)
        # Overwrite the dataset infos
        self._reduce_by_scene()

    def _reduce_by_scene(self):
        import random
        from nuscenes import NuScenes

        random.seed(self.seed)

        data_infos = self.data_infos
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)

        # Group by scene
        scenes = {}
        for info in data_infos:
            scene_token = nusc.get('sample', info['token'])['scene_token']
            scenes.setdefault(scene_token, []).append(info)

        # Sample from each scene 
        reduced = []
        for scene_token, frames in scenes.items():
            if len(frames) > self.samples_per_scene:
                frames = random.sample(frames, self.samples_per_scene)
            reduced.extend(frames)

        # print(f"[NuScenesReducedDataset] Reduced {len(data_infos)} → {len(reduced)} samples.")

        # Replace internal info list
        self.data_infos = reduced
