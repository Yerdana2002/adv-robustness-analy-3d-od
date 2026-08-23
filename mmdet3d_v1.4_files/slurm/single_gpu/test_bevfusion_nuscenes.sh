#!/bin/bash
#SBATCH --account=def-instructor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=bevfusion_test_nuscenes
#SBATCH --tmp=800G
# =============================================================================
# BEVFusion (lidar-cam) inference on nuScenes val.
#
# Uses mmdet3d 1.4's OFFICIAL projects/BEVFusion, not projects/mmdet3d_plugin/
# bevfusion. The plugin copy is the original MIT-HAN-LAB model class dropped
# into the tree: it subclasses BaseModule (not Base3DDetector) and its
# forward() takes raw tensors (img, points, camera2ego, lidar2ego, ...), so it
# has no loss/predict/_forward, no data_preprocessor and no test_step --
# Runner.test() cannot drive it at all. The official implementation is already
# mmengine-native and has published weights, so it is the PillarNeSt pattern:
# use upstream rather than hand-port.
#
# The official BEVFusion builds its OWN bev_pool_ext and voxel_layer CUDA
# extensions via projects/BEVFusion/setup.py. They are separate from mmcv's
# _ext, so the custom differentiable-voxelization mmcv build is NOT touched.
#
# Expected (official model zoo): NDS 71.4 / mAP 68.6
# =============================================================================

set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit=$?)" >&2' ERR

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11
source ~/centerpoint/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export BLIS_JC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# 9.0 = H100 (the GPU nodes here are gpu:h100). 8.0 kept so the .so also runs
# on A100 if the scheduler ever lands the job elsewhere. If this is unset or
# ignored, PyTorch falls back to 7.0;7.5;8.0;8.6 -- no sm_90 -- and the run
# dies with "no kernel image is available for execution on the device".
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export CUDA_HOME=$CUDA_PATH

PROJECT_ROOT=/home/yerdana/links/projects/def-zhengliu/yerdana
MMDET_ROOT=$PROJECT_ROOT/mmdetection3d
export PYTHONPATH="$MMDET_ROOT:${PYTHONPATH:-}"

CONFIG_PATH=$MMDET_ROOT/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
# Sparse-conv layout-converted checkpoint. The stock download loads 582/582
# keys but 21 pts_middle_encoder tensors are transposed: it was saved under
# spconv 2.x ([out,kD,kH,kW,in]) while this env has no spconv, so mmdet3d falls
# back to mmcv's sparse ops ([kD,kH,kW,in,out]). Installing spconv would fix
# BEVFusion but flip the layout for FocalFormer3D / PillarNeSt / CenterPoint,
# which already load correctly -- so the checkpoint is converted instead.
# See convert_bevfusion_ckpt_spconv_layout.py.
CHECKPOINT_PATH=$PROJECT_ROOT/checkpoint/bevfusion_lidar-cam_mmcv_spconv.pth
NUSCENES_FULL_TAR=$PROJECT_ROOT/data/nuscenes_bevformer.tar.zst
VAL_PKL_TAR=$PROJECT_ROOT/data/temporal_pkls/nuscenes_infos_val_bevfusion.tar
VAL_PKL=nuscenes_infos_val_bevfusion.pkl
WORK_DIR=$PROJECT_ROOT/work_dirs/bevfusion_nuscenes

mkdir -p "$WORK_DIR"
[ -f "$CHECKPOINT_PATH" ] || { echo "X checkpoint missing: $CHECKPOINT_PATH"; exit 1; }
[ -f "$VAL_PKL_TAR" ]     || { echo "X val pkl tar missing: $VAL_PKL_TAR"; exit 1; }

# ============================================================
# Step 1 - Build the BEVFusion CUDA ops (self-contained)
# ============================================================
echo "============================================================"
echo "Step 1: Building BEVFusion ops (bev_pool_ext, voxel_layer)"
echo "============================================================"
cd "$MMDET_ROOT"
# build_ext --inplace, not `develop`: the latter shells out to
# `pip install -e . --use-pep517`, which fails here. Compiling in place is all
# that is needed since PYTHONPATH already contains MMDET_ROOT.
#
# Always build from clean, and always on the GPU node. setup.py branches on
# torch.cuda.is_available(): on a login node that is False, so it silently
# falls back to CppExtension WITHOUT -DWITH_CUDA -- the .so then builds but the
# CUDA paths are compiled out. FORCE_CUDA=1 makes that branch deterministic.
export FORCE_CUDA=1
rm -rf build/temp.linux-x86_64-cpython-311/projects/BEVFusion \
       build/lib.linux-x86_64-cpython-311/projects/BEVFusion
rm -f projects/BEVFusion/bevfusion/ops/bev_pool/*.so \
      projects/BEVFusion/bevfusion/ops/voxel/*.so
python projects/BEVFusion/setup.py build_ext --inplace 2>&1 | tail -5

# Verify the .so files actually contain kernels for this GPU. A stale object
# built without TORCH_CUDA_ARCH_LIST carries only sm_70..sm_86 and fails ~20
# minutes later, deep in depth_lss bev_pool, with an opaque CUDA error.
echo ""
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true
for SO in projects/BEVFusion/bevfusion/ops/bev_pool/*.so \
          projects/BEVFusion/bevfusion/ops/voxel/*.so; do
    ARCHS=$(cuobjdump "$SO" 2>/dev/null | grep -E '^arch =' | sort -u | tr '\n' ' ')
    echo "  $(basename "$SO"): ${ARCHS:-<none>}"
    case "$ARCHS" in
        *sm_90*) ;;
        *) echo "X $(basename "$SO") has no sm_90 kernel. Rebuild with:"
           echo "  rm -rf build/*/projects/BEVFusion \\"
           echo "     projects/BEVFusion/bevfusion/ops/*/*.so"
           echo "  TORCH_CUDA_ARCH_LIST='8.0;9.0' python projects/BEVFusion/setup.py build_ext --inplace"
           exit 1 ;;
    esac
done
python -c "
import projects.BEVFusion.bevfusion  # noqa
print('OK BEVFusion package imports')
from mmdet3d.registry import MODELS
print('OK BEVFusion registered:', 'BEVFusion' in MODELS.module_dict)
"

# ============================================================
# Step 2 - Extract full nuScenes
# ============================================================
echo ""
echo "============================================================"
echo "Step 2: Extracting full nuScenes (zstd)"
echo "============================================================"
FULL_TMP="$SLURM_TMPDIR/full"
mkdir -p "$FULL_TMP"
df -h "$SLURM_TMPDIR"
time tar -I "zstd -d" -xf "$NUSCENES_FULL_TAR" -C "$FULL_TMP"

FULL_PATH=""
for candidate in nuscenes nuscenes_processed; do
    [ -d "$FULL_TMP/$candidate" ] && FULL_PATH="$FULL_TMP/$candidate" && break
done
[ -z "$FULL_PATH" ] && { echo "X nuScenes dir not found"; ls -la "$FULL_TMP"; exit 1; }
echo "OK FULL_PATH=$FULL_PATH"

# ============================================================
# Step 3 - Install the val pkl that carries lidar_sweeps
# ============================================================
echo ""
echo "============================================================"
echo "Step 3: Installing val pkl (with lidar_sweeps)"
echo "============================================================"
# The project's stock val pkl has NO lidar_sweeps: update_infos_to_v2's
# clear_data_info_unused_keys drops empty lists. LoadPointsFromMultiSweeps
# then silently pads with the keyframe (loading.py:416) instead of crashing,
# which would under-report NDS badly. tools/inject_lidar_sweeps.py rebuilt the
# field from devkit metadata: mean 9.75 sweeps/sample, 0 only on the 150
# scene-start frames, dt 0.050 s.
tar xf "$VAL_PKL_TAR" -C "$FULL_PATH"
ls -lh "$FULL_PATH/$VAL_PKL"

echo ""
echo "--- verifying sweeps are present AND their files exist on disk ---"
python - "$FULL_PATH" "$VAL_PKL" <<'PYCHECK'
import os, pickle, sys
import numpy as np
root, name = sys.argv[1], sys.argv[2]
d = pickle.load(open(os.path.join(root, name), 'rb'))
dl = d['data_list']
counts = np.array([len(e.get('lidar_sweeps', [])) for e in dl])
print(f'  samples            : {len(dl)}')
print(f'  sweeps/sample      : mean={counts.mean():.2f} min={counts.min()} max={counts.max()}')
print(f'  samples with 0     : {(counts == 0).sum()}  (expect 150 scene starts)')
assert counts.mean() > 5, 'lidar_sweeps missing or too sparse'

# The pkl only stores paths; confirm the actual sweep files were in the tarball.
missing = 0
checked = 0
for e in dl[:200]:
    for s in e.get('lidar_sweeps', [])[:2]:
        p = os.path.join(root, s['lidar_points']['lidar_path'])
        checked += 1
        if not os.path.exists(p):
            missing += 1
            if missing <= 3:
                print(f'  MISSING: {p}')
print(f'  sweep files checked: {checked}, missing: {missing}')
if missing:
    raise SystemExit(
        'FATAL: sweeps/LIDAR_TOP point files are absent from the tarball. '
        'BEVFusion cannot run with real sweeps; aborting rather than '
        'reporting a silently degraded NDS.')
print('  OK sweep point files present')

# Resolve keyframe lidar + camera paths exactly as Det3DDataset does, i.e.
# data_prefix joined onto the stored path. This catches double-prefixing: the
# CAN-bus pkl BEVFormer uses stores 'samples/CAM_FRONT/x.jpg' because its
# config sets data_prefix=dict(img=''), whereas the official BEVFusion config
# supplies 'samples/CAM_FRONT' itself and therefore needs bare filenames.
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
bad = 0
for e in dl[:50]:
    lp = os.path.join(root, 'samples/LIDAR_TOP',
                      e['lidar_points']['lidar_path'])
    if not os.path.exists(lp):
        bad += 1
        if bad <= 2:
            print(f'  MISSING lidar: {lp}')
    for cam in CAMS:
        ip = os.path.join(root, 'samples', cam, e['images'][cam]['img_path'])
        if not os.path.exists(ip):
            bad += 1
            if bad <= 4:
                print(f'  MISSING img: {ip}')
if bad:
    raise SystemExit(
        'FATAL: keyframe lidar/camera paths do not resolve. If the printed '
        'path repeats "samples/<SENSOR>" twice, the pkl stores prefixed paths '
        'but the config also applies data_prefix -- strip to bare filenames.')
print('  OK keyframe lidar + 6-camera paths resolve')
PYCHECK

# ============================================================
# Step 4 - Data symlink + import check
# ============================================================
echo ""
echo "============================================================"
echo "Step 4: Data symlink and import check"
echo "============================================================"
cd "$MMDET_ROOT"
mkdir -p data && rm -f data/nuscenes && ln -s "$FULL_PATH" data/nuscenes
echo "OK data/nuscenes -> $FULL_PATH"

python -c "
import mmdet3d, mmcv, mmdet, mmengine
print('mmdet3d', mmdet3d.__version__, '| mmcv', mmcv.__version__,
      '| mmdet', mmdet.__version__, '| mmengine', mmengine.__version__)
"

# ============================================================
# Step 5 - Inference
# ============================================================
echo ""
echo "============================================================"
echo "Step 5: BEVFusion lidar-cam inference on nuScenes val"
echo "============================================================"
echo "  Config    : $CONFIG_PATH"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Val pkl   : $VAL_PKL"
echo "  Expected  : NDS 71.4 / mAP 68.6 (official model zoo)"
echo ""

python tools/test.py \
    "$CONFIG_PATH" \
    "$CHECKPOINT_PATH" \
    --work-dir "$WORK_DIR" \
    --cfg-options \
        "test_dataloader.dataset.data_root='${FULL_PATH}/'" \
        "test_dataloader.dataset.ann_file='${VAL_PKL}'" \
        "test_dataloader.num_workers=4" \
        "test_evaluator.data_root='${FULL_PATH}/'" \
        "test_evaluator.ann_file='${FULL_PATH}/${VAL_PKL}'" \
        "test_evaluator.jsonfile_prefix=${WORK_DIR}/bevfusion_results"

echo ""
echo "Done."
