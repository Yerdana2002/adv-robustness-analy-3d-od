from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS
import sqlite3

@DATASETS.register_module()
class NuScenesAdversarialDataset(NuScenesDataset):
    """
    Adversarial version of the NuScenes dataset. Created using IoU-S Detachment on FocalFormer3D
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Overwrite the dataset infos
        self._replace_with_adv_paths()

    def _replace_with_adv_paths(self):
        conn = sqlite3.connect("/beegfs/krink/Projects/adversarial-attacks/visualizations/Centerpoint/NuScenes/iou_detachment/run_2026-01-06_11-40-39/results.db") # CenterPoint attacked by IoU-S Detachment

        query = """
        SELECT t.sample_id, ar.adv_pc_path
        FROM tasks t
        JOIN attack_results ar
        ON t.id = ar.task_id
        """

        rows = conn.execute(query).fetchall()

        adv_map = {
            sample_id: adv_path
            for sample_id, adv_path in rows
        }

        replaced = 0

        for info in self.data_infos:
            token = info['token']
            if token in adv_map:
                info['lidar_path'] = adv_map[token]
                replaced += 1

        print(f"Replaced {replaced} lidar paths.")
