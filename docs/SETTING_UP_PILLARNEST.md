# Setting up PillarNeSt
Setting up PillarNeSt is a bit more complicated than CenterPoint and PointPillars, because it is not included in mmdetection3d by default. The first three steps can be ignored if `mmdetection3d_patch.patch` has been applied. 
## Steps
* Copy the files from their [github](https://github.com/WayneMao/PillarNeSt) into their respective folder.
* Add the necessary functions to the __init__.py files
* It might be necessary to rename a couple functions (e.g. core/bbox/coder/centerpoint_bbox_coder.py to core/bbox/coder/centerpoint_bbox_coder_pn.py )
* Move the Config from `config/pillarnest` and place it into the mmdetection3d `config` folder.
* Model Weights for NuScenes and Kitti are in `weights/pillarnest`

## Important Changes:
* For Kitti: Does not use the "Centerpoint plus"-Head, because it is not compatable with Kitti. Instead the standart Centerpoint head ([fixed](https://github.com/open-mmlab/mmdetection3d/pull/924) for Kitti) is used. We expect the model to perform worse with the fixed centerpoint head