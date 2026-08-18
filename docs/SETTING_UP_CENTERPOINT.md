# Setting up Centerpoint
Centerpoint is included in mmdetection3d as a base model, that makes it easy for us to use!
## Steps

* Move the Config from `config/centerpoint_attacks` and place it into the mmdetection3d `config` folder.
* Model Weights for NuScenes and Kitti are in `weights/centerpoint`, adapt the path in the code or provide it as an argument (if used in argument, do not use a preset)!

## Important Changes:
* For Kitti and Waymo: Uses the fixed Centerpoint-Head from this [Github Pull Request](https://github.com/open-mmlab/mmdetection3d/pull/924). We expect the model to perform similar to the original.
