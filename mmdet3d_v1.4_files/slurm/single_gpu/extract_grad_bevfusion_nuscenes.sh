#!/bin/bash
#SBATCH --account=def-instructor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=grad_bevfusion
#SBATCH --tmp=800G
# =============================================================================
# BEVFusion gradient extraction on nuScenes val -- dL/df at pts_middle_encoder.
#
#   MODALITY=lidar-cam | lidar     which config/checkpoint
#   MODE=quick | full              quick does a 10-frame spread into work_dirs
#                                  and checks shape/norm/size/saliency;
#                                  full does all 6019
#
#   sbatch --export=ALL,MODALITY=lidar-cam,MODE=quick extract_grad_bevfusion_nuscenes.sh
#   sbatch --export=ALL,MODALITY=lidar-cam,MODE=full  extract_grad_bevfusion_nuscenes.sh
#
# Steps 1-4 are lifted from test_bevfusion_nuscenes.sh, which reproduced
# NDS 0.7116. Step 5 swaps tools/test.py for tools/train.py: extraction runs
# one "training" epoch with NoOpOptimizer so backward fires and the hook can
# capture, while weights stay frozen.
#
# Output lands on /scratch (GRAD_ROOT), not /project. 33.18 MB/sample fp32
# x 6019 = 200 GB per modality, and /project (def-instructor) hit 950 GB of its
# 1000 GB with two such sets on it -- job 18871825 moved them off. Writing here
# again would refill it. Scratch has 18 TB free but IS PURGED (~60 days
# untouched), so tar anything that must survive to /nearline.
# =============================================================================

set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit=$?)" >&2' ERR

MODALITY=${MODALITY:-lidar-cam}
MODE=${MODE:-quick}
# 'channel' unit-norms every BEV cell independently (what FocalFormerGradient
# Hook does); 'global' unit-norms the whole sample. The two produce gradient
# tensors whose total norms differ by sqrt(180*180) = 180, which changes how
# <g, f> weighs against beta*chamfer in the attack objective. Output dirs are
# suffixed so both variants coexist rather than silently overwriting.
NORMALIZE=${NORMALIZE:-channel}
case "$MODALITY" in lidar-cam|lidar) ;; *) echo "X MODALITY must be lidar-cam or lidar"; exit 1 ;; esac
case "$MODE" in quick|full) ;; *) echo "X MODE must be quick or full"; exit 1 ;; esac
case "$NORMALIZE" in channel|global|none) ;; *) echo "X NORMALIZE must be channel|global|none"; exit 1 ;; esac

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
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export CUDA_HOME=$CUDA_PATH

PROJECT_ROOT=/home/yerdana/links/projects/def-instructor/yerdana
MMDET_ROOT=$PROJECT_ROOT/mmdetection3d
export PYTHONPATH="$MMDET_ROOT:${PYTHONPATH:-}"

CONFIG_PATH=$MMDET_ROOT/projects/configs/bevfusion/bevfusion_${MODALITY}_grad_extract.py
if [ "$MODALITY" = "lidar-cam" ]; then
    CHECKPOINT_PATH=$PROJECT_ROOT/checkpoint/bevfusion_lidar-cam_mmcv_spconv.pth
else
    CHECKPOINT_PATH=$PROJECT_ROOT/checkpoint/bevfusion_lidar_mmcv_spconv.pth
fi

NUSCENES_FULL_TAR=$PROJECT_ROOT/data/nuscenes_bevformer.tar.zst
VAL_PKL_TAR=$PROJECT_ROOT/data/temporal_pkls/nuscenes_infos_val_bevfusion.tar
VAL_PKL=nuscenes_infos_val_bevfusion.pkl

WORK_DIR=$PROJECT_ROOT/work_dirs/grad_bevfusion_${MODALITY}_${SLURM_JOB_ID}
if [ "$MODE" = "full" ]; then
    GRAD_DIR=${GRAD_ROOT:-/home/yerdana/links/scratch/yerdana/gradients}/gradients_bevfusion_${MODALITY}_${NORMALIZE}
else
    GRAD_DIR=$WORK_DIR/gradients_quick
fi
mkdir -p "$WORK_DIR" "$GRAD_DIR"

[ -f "$CONFIG_PATH" ]     || { echo "X config missing: $CONFIG_PATH"; exit 1; }
[ -f "$CHECKPOINT_PATH" ] || { echo "X checkpoint missing: $CHECKPOINT_PATH"; exit 1; }
[ -f "$VAL_PKL_TAR" ]     || { echo "X val pkl tar missing: $VAL_PKL_TAR"; exit 1; }

echo "============================================================"
echo "BEVFusion gradient extraction"
echo "  modality   : $MODALITY"
echo "  mode       : $MODE"
echo "  config     : $CONFIG_PATH"
echo "  checkpoint : $CHECKPOINT_PATH"
echo "  normalize  : $NORMALIZE"
echo "  gradients  : $GRAD_DIR"
echo "============================================================"

# Refuse to start a full run that cannot finish. 6019 x 33.18 MB = 200 GB.
if [ "$MODE" = "full" ]; then
    NEED_GB=210
    # Check the filesystem GRAD_DIR actually lives on. Checking $PROJECT_ROOT
    # while writing to scratch would pass or fail for the wrong reason.
    AVAIL_GB=$(df -BG --output=avail "$(dirname "$GRAD_DIR")" | tail -1 | tr -dc '0-9')
    echo "  $(dirname "$GRAD_DIR") avail: ${AVAIL_GB} GB, need ~${NEED_GB} GB"
    [ "$AVAIL_GB" -gt "$NEED_GB" ] || {
        echo "X not enough room for a full gradient set at $GRAD_DIR."
        echo "  Delete a previous gradients_bevfusion_* set first."
        exit 1; }
    EXISTING=$(find "$GRAD_DIR" -name '*_grad.pt' 2>/dev/null | wc -l)
    [ "$EXISTING" -eq 0 ] || {
        echo "X $GRAD_DIR already holds $EXISTING *_grad.pt files."
        echo "  Remove them or pick another dir; refusing to mix two runs."
        exit 1; }
fi

# ============================================================
# Step 1 - Build the BEVFusion CUDA ops
# ============================================================
echo ""
echo "=== Step 1: Building BEVFusion ops ==="
cd "$MMDET_ROOT"
export FORCE_CUDA=1

# Only rebuild when the objects are missing or lack an sm_90 kernel.
#
# The rm below targets the SHARED tree under $MMDET_ROOT, nothing job-local.
# Rebuilding unconditionally means two of these jobs overlapping -- a queued
# quick run starting while a full run is in Step 1, say -- can have one delete
# .so the other is about to import, giving an intermittent "cannot open shared
# object file" or a torn, half-written object. Skipping the rebuild when the
# objects are already correct removes that window in the normal case.
#
# The cuobjdump verification below still runs unconditionally. That is the real
# guard, and it is what caught the sm_70..sm_86 build that produced "no kernel
# image is available for execution on the device" 20 minutes into inference.
NEED_BUILD=0
for SO in projects/BEVFusion/bevfusion/ops/bev_pool/bev_pool_ext*.so \
          projects/BEVFusion/bevfusion/ops/voxel/voxel_layer*.so; do
    if [ ! -f "$SO" ] || ! cuobjdump "$SO" 2>/dev/null | grep -q 'arch = sm_90'; then
        NEED_BUILD=1
    fi
done

if [ "$NEED_BUILD" -eq 1 ]; then
    echo "  ops missing or without sm_90 -- rebuilding"
    rm -rf build/temp.linux-x86_64-cpython-311/projects/BEVFusion \
           build/lib.linux-x86_64-cpython-311/projects/BEVFusion
    rm -f projects/BEVFusion/bevfusion/ops/bev_pool/*.so \
          projects/BEVFusion/bevfusion/ops/voxel/*.so
    python projects/BEVFusion/setup.py build_ext --inplace 2>&1 | tail -3
else
    echo "  existing ops already carry sm_90 -- skipping rebuild"
fi

nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true
for SO in projects/BEVFusion/bevfusion/ops/bev_pool/*.so \
          projects/BEVFusion/bevfusion/ops/voxel/*.so; do
    ARCHS=$(cuobjdump "$SO" 2>/dev/null | grep -E '^arch =' | sort -u | tr '\n' ' ')
    echo "  $(basename "$SO"): ${ARCHS:-<none>}"
    case "$ARCHS" in
        *sm_90*) ;;
        *) echo "X $(basename "$SO") has no sm_90 kernel"; exit 1 ;;
    esac
done
python -c "
import projects.BEVFusion.bevfusion, projects.BEVFusion.attack  # noqa
from mmdet3d.registry import MODELS, HOOKS, OPTIMIZERS
print('OK BEVFusion       :', 'BEVFusion' in MODELS.module_dict)
print('OK gradient hook   :', 'BEVFusionGradientHook' in HOOKS.module_dict)
print('OK NoOpOptimizer   :', 'NoOpOptimizer' in OPTIMIZERS.module_dict)
"

# ============================================================
# Step 2 - Extract full nuScenes
# ============================================================
echo ""
echo "=== Step 2: Extracting full nuScenes (zstd) ==="
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
echo "=== Step 3: Installing val pkl (with lidar_sweeps) ==="
tar xf "$VAL_PKL_TAR" -C "$FULL_PATH"
python - "$FULL_PATH" "$VAL_PKL" <<'PYCHECK'
import os, pickle, sys
import numpy as np
root, name = sys.argv[1], sys.argv[2]
dl = pickle.load(open(os.path.join(root, name), 'rb'))['data_list']
counts = np.array([len(e.get('lidar_sweeps', [])) for e in dl])
print(f'  samples       : {len(dl)}')
print(f'  sweeps/sample : mean={counts.mean():.2f} min={counts.min()} max={counts.max()}')
assert counts.mean() > 5, 'lidar_sweeps missing or too sparse'
missing = sum(
    not os.path.exists(os.path.join(root, s['lidar_points']['lidar_path']))
    for e in dl[:200] for s in e.get('lidar_sweeps', [])[:2])
print(f'  missing sweeps: {missing}')
assert missing == 0, 'sweep point files absent from the tarball'
print('  OK')
PYCHECK

# ============================================================
# Step 4 - Data symlink
# ============================================================
echo ""
echo "=== Step 4: Data symlink ==="
mkdir -p "$MMDET_ROOT/data"
ln -sfn "$FULL_PATH" "$MMDET_ROOT/data/nuscenes"
echo "OK data/nuscenes -> $FULL_PATH"
python -c "import mmdet3d, mmcv, mmdet, mmengine; print(f'mmdet3d {mmdet3d.__version__} | mmcv {mmcv.__version__} | mmdet {mmdet.__version__} | mmengine {mmengine.__version__}')"

# ============================================================
# Step 5 - Extraction
# ============================================================
echo ""
echo "=== Step 5: Gradient extraction ==="

# gradients_output_dir is read at config-exec time to build custom_hooks[0]
# .save_path, so overriding only the former after exec would silently do
# nothing -- the hook keeps whatever save_path the file baked in. Set both.
CFG_OPTS=(
    "load_from=$CHECKPOINT_PATH"
    "gradients_output_dir=$GRAD_DIR"
    "custom_hooks.0.save_path=$GRAD_DIR"
    "custom_hooks.0.normalize=$NORMALIZE"
    "train_dataloader.dataset.ann_file=$VAL_PKL"
)
if [ "$MODE" = "quick" ]; then
    # An explicit spread, NOT indices=24. A contiguous head slice was the
    # original quick check and it passed cleanly, then the full run died at
    # sample 41 -- the first val frame with no in-class GT. 25 of the 6019
    # frames are like that, and none of them is in 0..23, so the quick check
    # could not have caught it.
    #
    # This list is 0-2 (ordinary frames), then 41 / 67 / 4591 / 4593 / 5476 /
    # 5477 (empty-GT frames, from the three clusters where they occur), then
    # 6018 (last frame, catches off-by-one at the tail).
    CFG_OPTS+=("train_dataloader.dataset.indices=[0,1,2,41,67,4591,4593,5476,5477,6018]")
fi

time python tools/train.py "$CONFIG_PATH" \
    --work-dir "$WORK_DIR" \
    --cfg-options "${CFG_OPTS[@]}"

# ============================================================
# Step 6 - Validate what was written
# ============================================================
echo ""
echo "=== Step 6: Validating gradients ==="
python - "$GRAD_DIR" "$MODE" "$NORMALIZE" <<'PYVAL'
import os, sys
import torch
d, mode, mode_norm = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(f for f in os.listdir(d) if f.endswith('_grad.pt'))
print(f'  files written : {len(files)}')
assert files, 'no gradient files produced'

expected = 10 if mode == 'quick' else 6019
if len(files) != expected:
    print(f'  !! expected {expected}, got {len(files)}')

total = sum(os.path.getsize(os.path.join(d, f)) for f in files)
print(f'  total size    : {total/1e9:.2f} GB '
      f'({total/len(files)/1e6:.2f} MB/file)')

norms = []
for f in files[:min(8, len(files))]:
    t = torch.load(os.path.join(d, f), map_location='cpu')
    norms.append(t.float().flatten().norm().item())
    if f == files[0]:
        print(f'  shape         : {tuple(t.shape)}  dtype={t.dtype}')
        assert tuple(t.shape) == (1, 256, 180, 180), \
            f'unexpected shape {tuple(t.shape)} -- is target_layer middle_encoder?'

import statistics
print(f'  L2 norm       : mean={statistics.mean(norms):.6f} '
      f'min={min(norms):.6f} max={max(norms):.6f}')
# normalize='global' divides by the whole-sample L2 norm, so every file should
# come back at 1.0. A norm of ~sqrt(32400)=180 would mean 'channel' slipped
# through; anything else means no normalization ran.
# Expected whole-tensor norm depends on the mode: 'global' gives 1.0, while
# 'channel' unit-norms each of the 180x180 cells so the total is sqrt(32400)=180.
expect = {'global': 1.0, 'channel': 180.0}.get(mode_norm)
if expect is not None:
    ok = all(abs(n - expect) / expect < 1e-3 for n in norms)
    print(f"  expected norm : {expect} for normalize='{mode_norm}' -> "
          f"{'OK' if ok else 'MISMATCH'}")
    assert ok, f"norms {norms[:3]} do not match normalize='{mode_norm}'" 

nz = t.float().abs()
print(f'  sparsity      : {100*(nz < 1e-9).float().mean():.1f}% near-zero')
cell = t.float()[0].pow(2).sum(0).sqrt()
ratio = (cell.max() / cell.median()).item()
print(f'  BEV cell norm : max/median = {ratio:.1f}x')
print("                  (1.0x is expected and correct for 'channel'; "
      "large for 'global')")
PYVAL

echo ""
echo "============================================================"
echo "Done: $MODALITY / $MODE"
echo "  gradients: $GRAD_DIR"
du -sh "$GRAD_DIR"
df -h "$PROJECT_ROOT" | tail -1
echo "============================================================"
