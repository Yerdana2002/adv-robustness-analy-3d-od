# Setting up PointPillars
PointPillars is already included in mmdetection3d, the setup is quite easy.
## Steps
* For Kitti it is necessary to change line 128 in pillar_encoder.py to `f_center = features[:, :, :3].clone()` (just adding the .clone()) to allows gradients to flow)
* Move the Config from `config/pillarnest` and place it into the mmdetection3d `config/pillarnest` folder.
* Model Weights for NuScenes and Kitti are in `weights/pointpillars`

## Important Changes:
* HardVFE was removed from the config!!!!