# MMDetection3D v1.0.0rc Setup Guide (for Adversarial Attack Pipeline)

This guide walks you through installing the specific MMDetection3D version and compatible dependencies required to run the adversarial attack pipeline described in this repository.

---

## 1. Create and Activate a Conda Environment

```bash
mamba create -n NAME python=3.8 -y
mamba activate NAME
```

---

## 2. Install CUDA Toolkit

Ensure you're using CUDA 11.8:

```bash
mamba install cudatoolkit=11.8
mamba install conda-forge::cudatoolkit-dev
```

---

## 3. Clone and Checkout MMDetection3D

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.18.1
```
If you plan to use all models from this repository, I recommend replacing/adding all mmdetection3d files provided in this repository! You still need to install every dependancy, but you dont need to change any files in mmdetection3d (And all models and configs are included). 

---

## 4. Install PyTorch

Use the official PyTorch wheels with CUDA 11.8:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

---

## 5. Install OpenMIM and Build MMCV

```bash
pip install -U openmim
```

Manually build `mmcv-full==1.4.2` from source:

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.4.2
```

### Important Fixes:

BEFORE building mmcv-full it is required to change some files to keep compatability with our torch version. Look at this [github comment](https://github.com/open-mmlab/mmdetection3d/issues/1332#issuecomment-2594408887) (https://github.com/open-mmlab/mmdetection3d/issues/1332#issuecomment-2594408887)  to find out what to replace (MMCV already fixed most of it, but the THC imports were left in):

* **Remove and replace all `THC` references** from:

  * `ops/csrc/pytorch/cuda/ms_deform_attn_cuda.cu`
  * `ops/csrc/pytorch/cuda/psamask_cuda.cu`


Then build MMCV ([for Guide click here](https://mmcv.readthedocs.io/en/latest/get_started/build.html)):

```bash
pip install -e .
```

---

## 6. Install MMDet3D and MMDetection3D
Similar to building MMCV, it is necessary to replace the references to THC. The [github comment](https://github.com/open-mmlab/mmdetection3d/issues/1332#issuecomment-2594408887) mentioned before lists the files that need to be changed and the necessary replacements (not necessary if patch is applied).

**IMPORTANT**:Before building mmdetection3d it is necessary to apply the [non-differentiable voxelization fix from docs/VOXELIZATION_ISSUE.md](VOXELIZATION_ISSUE.md)!

### 6.1. Adding files for the adversarial attack pipeline:
All changes made to my mmdetection3d installation can be found in the included mmdetection3d folder. Every file that is included in this folder has been changed to some degree from the original mmdetection3d installation. Below, there is a list of the minimum required files to add for running the adversarial attack pipeline.

**Minimum required files:**
```
mmdetection3d/
└── mmdet3d/
    └── datasets/nuscenes_dataset.py
    └── datasets/waymo_dataset.py
    └── ops/voxel/
        └── voxelize.py
        └── src/voxelization_cuda.cu
        └── src/voxelization.h
```
**Warning:** I did not test the pipeline with only these files, you might need to change/add more. Check the mmdetection3d folder

### 6.2. Installing mmdet3d and mmdetection3d

Inside the `mmdetection3d/` root directory:

```bash
pip install -v -e .
```

---

## 7. Install Required OpenMMLab Libraries

```bash
pip install 'mmdet==2.24.0'
pip install 'mmsegmentation==0.20.2'
pip install 'mmcls==0.22.1'
```

---

## 8. Additional Python Tools

```bash
pip install yapf==0.40.1
```

---

You should now have a fully working environment for running the adversarial attack pipeline built on MMDetection3D v0.18.1.

Make sure to validate your setup by running a basic MMDetection3D config script or training step before integrating the attack logic.

---

## FAQ
* I get an error when building mmcv-full or mmdetection3d!
    * A common mistake is that not all THC references have been fully replaced. Make sure that you did not forget to remove ```extern THCState *state;```
* I get a dataset key error when trying to run some test code.
    * Keep in mind that we are using an older version of mmdetection3d. The structure of the datasets has most likely changed and the info files need to be recomputed. Make sure you **backup** your old info files before overwriting them!
