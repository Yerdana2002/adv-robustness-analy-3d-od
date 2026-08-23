#!/bin/bash
#SBATCH --account=def-instructor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=ff_lc_grad
#SBATCH --tmp=800G
# =============================================================================
# FocalFormer3D LiDAR-only gradient extraction on nuScenes val -- dL/df at the
# neck. nuScenes ONLY.
#
#   sbatch extract_grad_focalformer_l_nus.sh                      # MODE=quick
#   sbatch --export=ALL,MODE=full extract_grad_focalformer_l_nus.sh
#
# Replaces extract_grad_focalformer_nus.sh, which is unusable as-is: it points
# at PROJECT_ROOT=~/links/projects/rrg-zhengliu, an allocation that is OVER
# QUOTA (28K files against a 2048 limit), and at a nuscenes_processed.tar that
# predates the current dataset.
#
# Three deliberate departures from that script
# --------------------------------------------
# 1. ann_file is nuscenes_infos_val_bevfusion.pkl, NOT nuscenes_infos_val.pkl.
#    The latter does not exist in this project. Both configs use a plain
#    NuScenesDataset over mmdet3d 1.4 infos, so the BEVFusion val pkl loads
#    unchanged -- and it puts FocalFormer and BEVFusion on the IDENTICAL 6019
#    frames with identical GT, which any cross-model AI@x comparison needs
#    anyway.
#
# 2. Gradients are written to node-local $SLURM_TMPDIR and shipped as ONE tar,
#    never as 6019 loose files on shared storage. /project is at 951 GB of
#    1000 GB, and /nearline caps at 5000 files -- one gradient set alone would
#    exceed that cap.
#
# 3. MODE=quick (default) does a 10-frame spread first. The BEVFusion
#    extraction died at sample 41 on the first val frame with no in-class GT
#    after a contiguous head slice passed cleanly, so the spread deliberately
#    includes empty-GT frames rather than the first N.
#
# The hook hardcodes F.normalize(dim=1), i.e. 'channel' -- the same
# normalization the BEVFusion gradients carry. There is no knob and nothing to
# match up; normalize=True IS channel.
#
# NOTE: gradients are at the NECK here, versus pts_middle_encoder for
# BEVFusion. Different tensor, different shape. Fine within a model; not
# comparable across the two.
# =============================================================================
set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit=$?)" >&2' ERR

MODE=${MODE:-quick}
TARGET_LAYER=${TARGET_LAYER:-neck}
case "$MODE" in quick|full) ;; *) echo "X MODE must be quick or full"; exit 1 ;; esac

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11
source ~/centerpoint/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export CUDA_HOME=$CUDA_PATH

PROJECT_ROOT=/home/yerdana/links/projects/def-zhengliu/yerdana
MMDET_ROOT=$PROJECT_ROOT/mmdetection3d
export PYTHONPATH="$MMDET_ROOT:${PYTHONPATH:-}"

CONFIG_PATH=$MMDET_ROOT/projects/configs/focalformer3d/FocalFormer3D_LC_grad_extract.py
CHECKPOINT=$PROJECT_ROOT/checkpoint/FocalFormer3D_LC_ep6_converted.pth
NUSCENES_FULL_TAR=$PROJECT_ROOT/data/nuscenes_bevformer.tar.zst
VAL_PKL_TAR=$PROJECT_ROOT/data/temporal_pkls/nuscenes_infos_val_bevfusion.tar
VAL_PKL=nuscenes_infos_val_bevfusion.pkl

SCRATCH_OUT=/home/yerdana/links/scratch/yerdana/gradients
NEARLINE_OUT=/home/yerdana/links/nearlines/def-zhengliu/yerdana/gradients
WORK_DIR=$PROJECT_ROOT/work_dirs/ff_lc_grad_${TARGET_LAYER}_${MODE}_${SLURM_JOB_ID}
LOCAL_GRAD_DIR=$SLURM_TMPDIR/gradients
TAR_NAME=gradients_focalformer_lc_${TARGET_LAYER}_channel.tar

mkdir -p "$WORK_DIR" "$LOCAL_GRAD_DIR" "$SCRATCH_OUT"

for f in "$CONFIG_PATH" "$CHECKPOINT" "$NUSCENES_FULL_TAR" "$VAL_PKL_TAR"; do
    [ -f "$f" ] || { echo "X missing: $f"; exit 1; }
done

# Step 2 repoints the SHARED symlink $MMDET_ROOT/data/nuscenes. Job 18801696
# died 4h55m in when another job moved it underneath. mv_grads only copies
# files and never touches the symlink, so it is not a conflict.
if [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
    OTHERS=$(squeue -h -u "$USER" -t RUNNING -o '%i %j' 2>/dev/null \
        | awk -v me="${SLURM_JOB_ID:-0}" \
              '$1 != me && $2 != "mv_grads" {printf "%s(%s) ", $1, $2}') || OTHERS=""
    if [ -n "$OTHERS" ]; then
        echo "X another job is RUNNING that may move the shared symlink: $OTHERS"
        echo "  override: sbatch --export=ALL,ALLOW_CONCURRENT=1 $0"
        exit 1
    fi
fi

echo "============================================================"
echo "FocalFormer3D-LC gradient extraction (nuScenes val)"
echo "  mode       : $MODE"
echo "  layer      : $TARGET_LAYER"
echo "  checkpoint : $(basename "$CHECKPOINT")"
echo "  ann_file   : $VAL_PKL  (shared with BEVFusion -- identical split)"
echo "  normalize  : hook hardcodes F.normalize(dim=1) == channel"
echo "============================================================"

echo ""
echo "=== Step 1: Extracting nuScenes ==="
FULL_TMP="$SLURM_TMPDIR/full"; mkdir -p "$FULL_TMP"
df -h "$SLURM_TMPDIR" | tail -1
time tar -I "zstd -d" -xf "$NUSCENES_FULL_TAR" -C "$FULL_TMP"
FULL_PATH=""
for c in nuscenes nuscenes_processed; do
    [ -d "$FULL_TMP/$c" ] && FULL_PATH="$FULL_TMP/$c" && break
done
[ -z "$FULL_PATH" ] && { echo "X nuScenes dir not found"; exit 1; }
echo "OK FULL_PATH=$FULL_PATH"

echo ""
echo "=== Step 2: val pkl + symlink ==="
tar xf "$VAL_PKL_TAR" -C "$FULL_PATH"
[ -f "$FULL_PATH/$VAL_PKL" ] || { echo "X val pkl missing after untar"; exit 1; }
mkdir -p "$MMDET_ROOT/data"
ln -sfn "$FULL_PATH" "$MMDET_ROOT/data/nuscenes"
cd "$MMDET_ROOT"

echo ""
echo "=== Step 3: Extraction ($MODE) ==="
CFG_OPTS=(
    "load_from=${CHECKPOINT}"
    "custom_hooks.1.target_layer=${TARGET_LAYER}"
    "custom_hooks.1.save_path=${LOCAL_GRAD_DIR}"
    "custom_hooks.1.normalize=True"
    "train_dataloader.dataset.ann_file=${VAL_PKL}"
    "train_cfg.val_interval=999"
)
if [ "$MODE" = "quick" ]; then
    # Same spread the BEVFusion extraction settled on: ordinary frames, then
    # empty-GT frames from the three clusters where they occur, then the tail.
    # A contiguous head slice passed there and the full run still died at 41.
    CFG_OPTS+=("train_dataloader.dataset.indices=[0,1,2,41,67,4591,4593,5476,5477,6018]")
fi
time python tools/train.py "$CONFIG_PATH" --work-dir "$WORK_DIR" \
    --cfg-options "${CFG_OPTS[@]}"

echo ""
echo "=== Step 4: Validate ==="
GRAD_COUNT=$(find "$LOCAL_GRAD_DIR" -name '*_grad.pt' | wc -l)
# 5980, NOT 6019. 39 of the 6019 val frames end the train pipeline with no GT
# at all, so there is no loss, backward never reaches the neck, and
# FocalFormerGradientHook returns on `_capturer.gradient is None` without
# writing a file or logging anything.
#
# This constant read 5994 until the full run (job 18875807) produced 5980 and
# tripped the check below. 5994 was wrong, not the run: it counted only the
# ObjectNameFilter casualties. Counted directly off
# nuscenes_infos_val_bevfusion.pkl through the config's actual filter chain:
#
#     3   frames have no instances at all
#     7   after use_valid_flag=True          (bbox_3d_isvalid)
#    25   after ObjectNameFilter             (bbox_label_3d == -1)
#    39   after ObjectRangeFilter            <-- what the loss sees
#
# The last step is the one that was missed: 14 further frames hold GT whose
# only boxes sit outside x,y in (-54, 54). 6019 - 39 = 5980 exactly.
#
# The quick spread still expects 4: of its ten indices, 41/67/4591/4593/5476/
# 5477 are all in the empty-39, so 10 - 6 = 4 either way.
#
# This differs from BEVFusion, whose sets hold all 6019 -- its loss still has
# background terms on empty frames. Do not assume 1:1 coverage across models.
#
# Consequence downstream: the attack skips those 39 (skipped += 1; continue),
# so the adversarial set is 5980 clouds and 39 val frames stay CLEAN. That is
# 0.65% of the split and must be stated wherever the metrics are quoted.
EXPECTED=$([ "$MODE" = "full" ] && echo 5980 || echo 4)
echo "  gradient files : $GRAD_COUNT (expected $EXPECTED)"
[ "$GRAD_COUNT" -gt 0 ] || { echo "X no gradients produced"; exit 1; }
if [ "$GRAD_COUNT" -ne "$EXPECTED" ]; then
    echo "  !! count differs from the verified expectation -- investigate"
    echo "     before using this set; a silent shortfall means frames were"
    echo "     dropped for a reason other than empty GT."
fi
python - "$LOCAL_GRAD_DIR" <<'PYVAL'
import os, sys, torch
d = sys.argv[1]
fs = sorted(f for f in os.listdir(d) if f.endswith('_grad.pt'))
t = torch.load(os.path.join(d, fs[0]), map_location='cpu')
print(f'  shape          : {tuple(t.shape)}  dtype={t.dtype}')
print(f'  total L2       : {t.float().norm().item():.4f}')
print(f'  per-cell norms : min={t.float().flatten(0,-2).norm(dim=-1).min().item():.4f} '
      f'max={t.float().flatten(0,-2).norm(dim=-1).max().item():.4f}')
print(f'  finite         : {bool(torch.isfinite(t).all())}')
tot = sum(os.path.getsize(os.path.join(d, f)) for f in fs)
print(f'  total size     : {tot/1e9:.2f} GB over {len(fs)} files')
PYVAL

if [ "$MODE" != "full" ]; then
    echo ""
    echo "quick check done -- rerun with MODE=full for all 6019."
    exit 0
fi

echo ""
echo "=== Step 5: Tar + ship ==="
# One tar, never loose files: /nearline caps at 5000 files and 6019 alone
# would exceed it.
cd "$LOCAL_GRAD_DIR"
time tar -cf "$SLURM_TMPDIR/$TAR_NAME" .
ls -lh "$SLURM_TMPDIR/$TAR_NAME"

echo "  -> scratch (working copy, fast to stage into a job)"
time cp "$SLURM_TMPDIR/$TAR_NAME" "$SCRATCH_OUT/"
echo "  -> nearline (durable; scratch purge already destroyed one FocalFormer"
echo "     gradient set, focalformer3d_neck_6957533, on 2026-07-19)"
if mkdir -p "$NEARLINE_OUT" 2>/dev/null; then
    time cp "$SLURM_TMPDIR/$TAR_NAME" "$NEARLINE_OUT/" || \
        echo "  !! nearline copy failed; scratch copy stands"
else
    echo "  !! could not create $NEARLINE_OUT; scratch copy stands"
fi

echo ""
echo "Wrote:"
ls -lh "$SCRATCH_OUT/$TAR_NAME" 2>/dev/null | sed 's/^/  /'
ls -lh "$NEARLINE_OUT/$TAR_NAME" 2>/dev/null | sed 's/^/  /'
echo ""
echo "Done."
