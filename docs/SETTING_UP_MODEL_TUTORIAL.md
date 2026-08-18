# Setting up a custom model
This tutorial will cover the basics of adding a new model to the pipeline.

## Steps
Creating a new model can be divided into three stages: 
### 1. Creating a config
Creating the config is quite simple. You just need to follow these two steps. Some examples for adversarial configs can be found in `config/` that can be used as reference.
* Create a standard mmdetection3d config (or use the config that your model is trained on if you use a pretrained model)
* In train_cfg: Add `dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),` and append the keys of Collect3d `dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])`. This allows the pipeline to access the ground truth, even when using test mode!
* Also make sure to include `adv_mode=True,` in the data dict (e.g. `data = dict(train=dict(...),val=dict(..., adv_mode=True),test=dict(...,adv_mode=True))`)

### 2. Training the model
When using a pretrained model, this step is not necessary. The changes in the config should not change the model functionality.
* Train the model like a normal mmdetection3d model

### 3. Creating the model wrapper
The idea behind the wrapper is that we can access the models internal procedures without needing to change the code in mmdetection3d. That can be used to removr entire parts of the model (nms head from PointPillars), that would otherwise disturb gradient flow. Additionally, Model wrappers allow us to use the same code for different models, even if the output format of the models is slightly different.
When implementing the model wrapper keep these things in mind:
* Models that are based on the same detector implementations should not need a new seperate model wrapper (e.g. Centerpoint and PillarNeSt)
* Your model wrapper should inherit `model_wrapper.py`
* `def predict(...):` should be the unaltered model inference. If you want to change how inference works during the adversarial attack (e.g. to get access to the gradients) use `def forward(...):` instead.
* Don't forget to implement `def grad():`, most attacks need access to the gradient
* Don't forget to add your wrapper in `adversarial_attack_pipeline.py/load_model_and_dataset()` 

### 4. Other important steps
It is important that the voxelization allows gradients and that the model's precision float is 32! [Github issue](https://github.com/haichen-ber/IoU-S-Attack/issues/3#issuecomment-3555609103).
* check that the `voxelization()` function in the detector and the `forward()` function in the middle encoder have `@force_fp32()` and that `@torch.no_grad()` is removed or commented!

Feel free to look at already implemented model wrappers for reference.