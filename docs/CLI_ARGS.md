# Command Line Arguments
## Adversarial Attack Pipeline 
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--preset-model` | `str` | `"custom"` | Model preset providing predefined config and checkpoint paths. Choices: `centerpoint`, `pillarnest`, `pointpillars`, `focalformer3d`, their KITTI/Waymo variants, and `custom`. | 
| `--config` | `str` | `None` | Path to the model configuration file. | 
| `--reduced` | `bool` (flag) | `False` | Use the reduced dataset instead of the full dataset. | 
| `--lc-fusion` | `bool` (flag) | `False` | Use fusion model weights. | 
| `--model` | `str` | `None` | Path to model checkpoint file. | 
| `--checkpoint` | `str` | `None` | Path to a checkpoint or previous run directory. | 
| `--num-samples` | `int` | `None` | Number of samples to attack. | 
| `--attack` | `str` | `"iou_detachment"` | Attack method to execute. Choices: `iou_detachment`, `iou_attachment`, `iou_perturbation`, `fgsm`, `pgd`, `lidattack`. | 
| `--save-dir` | `str` | `None` | Directory for saving visualizations and outputs. | 
| `--no-visual` | `bool` (flag) | `False` | Skip visualization generation. | 
| `--launcher` | `str` | `None` | Launcher backend for distributed or multi-GPU execution. | 
| `--sub_loss` | `str` | `"iou"` | Objective used by IoU-based attacks. Choices: `iou`, `score`, `all`. | 
| `--num-drop` | `int` | `1024` | Total number of points removed during IoU Detachment attacks. | 
| `--k-drop-round` | `int` | `16` | Number of points removed per iteration during IoU Detachment attacks. | 
| `--attack_lr` | `float` | `0.01` | Learning rate used by perturbation and attachment attacks. | 
| `--steps` | `int` | `500` | Number of optimization steps for attachment and perturbation attacks. | 
| `--num_add` | `int` | `1024` | Number of points added during IoU Attachment attacks. | 
| `--gen_iterations` | `int` | `100` | Maximum number of genetic algorithm iterations for LidAttack. | 
| `--population` | `int` | `20` | Population size for LidAttack's genetic algorithm. | 
| `--epsilon` | `float` | `0.3` | Perturbation magnitude for FGSM and PGD attacks. | 
| `--iterations` | `int` | `1000` | Number of PGD optimization iterations. | 
| `--step-size` | `float` | `None` | PGD step size per iteration. | 
| `--debug` | `bool` | `False` | Enable debug mode with predefined parameters. | 
| `--prefix` | `str` | `""` | Prefix added to generated output file names. | 
| `--base-rank` | `int` | `0` | Base rank used for global worker ID assignment in distributed execution. |

## Evaluation

| Argument | Type | Default | Description | Example |
|----------|------|---------|-------------|---------|
| `--data_path` | `str` | `""` | Path to adversarial attack database | `--data_path path/to/db` |
| `--suffix` | `str` | `""` | Optional suffix added to output file names. Converted to lowercase. | `--suffix exp1` |
| `--innout_thresh` | `float` | `0.8` | In/Out threshold value. | `--innout_thresh 0.9` |
| `--save` | `str` | `None` | Path where outputs/results are saved. | `--save results/` |
| `--input_suffix` | `str` | `""` | Suffix used when the input file name differs from the standard naming convention. | `--input_suffix v2` |
| `--full` | `bool` (flag) | `False` | Also compute prediction-box tables for mAP computation. | `--full` |

## Visualization

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--db-path` | `str` | `""` | Path to adversarial attack database | `--data_path path/to/db` |
| `--output` | `str` | `None` | Directory to save generated visuals. |
| `--samples` | `list[str]` | Required | Sample identifiers to visualize. Accepts one or more values. |
| `--adv` | `bool` (flag) | `False` | Compare original and adversarial point clouds. |
| `--raw` | `bool` (flag) | `False` | Plot the raw point cloud. |
| `--objects` | `bool` (flag) | `False` | Visualize all objects with inner and outer points. |
| `--reduced` | `bool` (flag) | `False` | Use the reduced dataset configuration. |
| `--show-score-thr` | `float` | `0.1` | Score threshold for visualizing predictions. |
| `--color-mode` | `str` | `"depth"` | Point coloring mode. Choices: `depth`, `height`, `intensity`, `density`. |
| `--points-keep-ratio` | `float` | `1.0` | Ratio of points to display. |
| `--point-size` | `float` | `0.5` | Size of points in the visualization. |
| `--no-3d` | `bool` (flag) | `False` | Skip generation of 3D visualization output. |
| `--no-bev` | `bool` (flag) | `False` | Skip generation of Bird’s Eye View (BEV) output. |
| `--bev-dpi` | `int` | `300` | DPI used for BEV output. |
| `--bev-figsize` | `float float` | `[24, 20]` | Figure size for BEV output in inches (`width height`). |
| `--input_suffix` | `str` | `""` | Suffix used when the input file name differs from the standard naming convention. |
| `--no-legend` | `bool` (flag) | `False` | Hide the visualization legend. |
