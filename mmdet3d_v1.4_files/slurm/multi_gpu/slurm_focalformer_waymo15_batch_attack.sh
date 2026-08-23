#!/bin/bash
#SBATCH --account=rrg-instructor
#SBATCH --time=32:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=focal_waymo_adv_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --tmp=2400G


module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11
source ~/centerpoint/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=$CUDA_PATH

PROJECT_ROOT=~/links/projects/rrg-instructor/yerdana
MMDET_ROOT=$PROJECT_ROOT/mmdetection3d

CONFIG_PATH=$MMDET_ROOT/projects/configs/focalformer3d/FocalFormer3D_Waymo_L_gradient.py
CHECKPOINT_PATH=$PROJECT_ROOT/checkpoint/FocalFormer3d_Waymo_converted.pth
ATTACK_SCRIPT=$MMDET_ROOT/projects/mmdet3d_plugin/models/attack_focalformer_waymo_distributed.py

WAYMO_ARCHIVE=~/links/scratch/yerdana/waymo_kitti_format_v14_7235271.tar.zst
WAYMO_GT=~/links/scratch/yerdana/waymo_pkl_backup_7235271/gt.bin
WAYMO_VAL_15SPLIT=~/links/scratch/yerdana/waymo_pkl_backup_7235271/waymo_infos_val_15split.pkl
GRADIENT_SRC=~/links/scratch/yerdana/focalformer_waymo_neck_gradients_7256547.tar.zst

TARGET_LAYER="${1:-neck}"
GPUS=4

LOCAL_DATA=$SLURM_TMPDIR/data
LOCAL_GRADS=$SLURM_TMPDIR/gradients
LOCAL_RESULTS=$SLURM_TMPDIR/adv_results
DEST_DIR=~/links/scratch/yerdana/focalformer/waymo_pc
OUT_TAR=focalformer_waymo_adv_${TARGET_LAYER}_ddp_${SLURM_JOB_ID}.tar

mkdir -p "$LOCAL_DATA" "$LOCAL_GRADS" "$LOCAL_RESULTS" "$DEST_DIR"

echo "============================================================"
echo "FocalFormer Waymo Adversarial Attack (${GPUS}-GPU DDP)"
echo "Job ID:        $SLURM_JOB_ID"
echo "Target layer:  $TARGET_LAYER"
echo "Config:        $CONFIG_PATH"
echo "Checkpoint:    $CHECKPOINT_PATH"
echo "============================================================"



#################
set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit=$?)" >&2' ERR

THREADS=${SLURM_CPUS_PER_TASK:-32}
ZSTD_CMD="zstd -d -T${THREADS}"

# [0/5] Stage archives to local scratch first (faster than streaming from network FS)
echo ""
echo "[0/5] Staging archives to local scratch..."
LOCAL_WAYMO_SRC="$SLURM_TMPDIR/$(basename "$WAYMO_ARCHIVE")"
LOCAL_GRAD_SRC="$SLURM_TMPDIR/$(basename "$GRADIENT_SRC")"

time cp -f "$WAYMO_ARCHIVE" "$LOCAL_WAYMO_SRC"

case "$GRADIENT_SRC" in
    *.tar.zst|*.tzst|*.tar)
        time cp -f "$GRADIENT_SRC" "$LOCAL_GRAD_SRC"
        ;;
    *)
        # if gradient source is already a directory, skip archive staging
        LOCAL_GRAD_SRC="$GRADIENT_SRC"
        ;;
esac

# [1/5] Extract Waymo dataset
echo ""
echo "[1/5] Extracting Waymo dataset..."
case "$LOCAL_WAYMO_SRC" in
    *.tar.zst|*.tzst) time tar -I "$ZSTD_CMD" -xf "$LOCAL_WAYMO_SRC" -C "$LOCAL_DATA" --no-same-owner --no-same-permissions ;;
    *.tar)            time tar -xf "$LOCAL_WAYMO_SRC" -C "$LOCAL_DATA" --no-same-owner --no-same-permissions ;;
    *)                echo "✗ Unsupported Waymo archive format: $LOCAL_WAYMO_SRC"; exit 1 ;;
esac

WAYMO_KITTI_PATH="$(find "$LOCAL_DATA" -maxdepth 6 -type d -name kitti_format | head -n 1 || true)"
[ -n "$WAYMO_KITTI_PATH" ] || { echo "✗ kitti_format not found"; exit 1; }
WAYMO_ROOT="$(dirname "$WAYMO_KITTI_PATH")"
WAYMO_FORMAT_PATH="$WAYMO_ROOT/waymo_format"
mkdir -p "$WAYMO_FORMAT_PATH"
echo "  ✓ kitti_format: $WAYMO_KITTI_PATH"

# Install gt.bin and val split
[ -f "$WAYMO_GT" ] && cp -f "$WAYMO_GT" "$WAYMO_FORMAT_PATH/gt.bin" && echo "  ✓ gt.bin"
if [ -f "$WAYMO_VAL_15SPLIT" ]; then
    cp -f "$WAYMO_VAL_15SPLIT" "$WAYMO_KITTI_PATH/waymo_infos_val.pkl"
    echo "  ✓ 1/5 val split installed"
fi

# Verify
for f in waymo_infos_val.pkl waymo_dbinfos_train.pkl; do
    [ -f "$WAYMO_KITTI_PATH/$f" ] || { echo "✗ Missing: $f"; exit 1; }
done
[ -d "$WAYMO_KITTI_PATH/training/velodyne" ] || { echo "✗ Missing velodyne"; exit 1; }

# Symlinks
cd "$MMDET_ROOT"
mkdir -p data/waymo
rm -f data/waymo/kitti_format data/waymo/waymo_format
ln -s "$WAYMO_KITTI_PATH"  data/waymo/kitti_format
ln -s "$WAYMO_FORMAT_PATH" data/waymo/waymo_format
echo "  ✓ symlinks created"

df -h "$SLURM_TMPDIR"

# [2/5] Extract gradients
echo ""
echo "[2/5] Extracting gradients..."
case "$LOCAL_GRAD_SRC" in
    *.tar.zst|*.tzst)
        # only unpack gradient tensors if archive has extra files
        time tar -I "$ZSTD_CMD" -xf "$LOCAL_GRAD_SRC" -C "$LOCAL_GRADS" --wildcards --no-anchored '*_grad.pt'
        ;;
    *.tar)
        time tar -xf "$LOCAL_GRAD_SRC" -C "$LOCAL_GRADS" --wildcards --no-anchored '*_grad.pt'
        ;;
    *)
        time rsync -a "$LOCAL_GRAD_SRC"/ "$LOCAL_GRADS"/
        ;;
esac

# Flatten if nested
GRAD_COUNT=$(find "$LOCAL_GRADS" -type f -name "*_grad.pt" | wc -l)
if [ "$GRAD_COUNT" -eq 0 ]; then
    find "$LOCAL_GRADS" -mindepth 2 -type f -name "*_grad.pt" -exec mv -n {} "$LOCAL_GRADS"/ \;
    GRAD_COUNT=$(find "$LOCAL_GRADS" -maxdepth 1 -type f -name "*_grad.pt" | wc -l)
fi
[ "$GRAD_COUNT" -gt 0 ] || { echo "✗ No gradient files found"; exit 1; }
echo "  ✓ Gradient files: $GRAD_COUNT"

df -h "$SLURM_TMPDIR"
################



# [3] Restore previous results for resume (if any)
echo ""
echo "[3/5] Checking for previous results to resume..."

PREV_ADV=~/links/scratch/yerdana/focalformer/waymo_pc/focalformer_waymo_adv_neck_ddp_7489967.tar

if [ -f "$PREV_ADV" ]; then
    echo "  Restoring previous adversarial files from: $PREV_ADV"
    case "$PREV_ADV" in
        *.tar.zst|*.tzst) tar -I "$ZSTD_CMD" -xf "$PREV_ADV" -C "$LOCAL_RESULTS" ;;
        *.tar)            tar -xf "$PREV_ADV" -C "$LOCAL_RESULTS" ;;
        *)                echo "  ✗ Unsupported archive format: $PREV_ADV"; exit 1 ;;
    esac
fi

EXISTING_COUNT=$(find "$LOCAL_RESULTS" -maxdepth 1 -type f -name '*.bin' | wc -l)
echo "  Existing adversarial files: $EXISTING_COUNT"


# [4] Run distributed attack
echo ""
echo "[4/5] Running distributed attack ($GPUS GPUs)..."
export PYTHONPATH="$MMDET_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"

VISIBLE_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
[ "$VISIBLE_GPUS" -ge "$GPUS" ] || { echo "Need $GPUS GPUs, found $VISIBLE_GPUS"; exit 1; }

time torchrun --standalone --nproc_per_node="$GPUS" "$ATTACK_SCRIPT" \
    --cfg          "$CONFIG_PATH" \
    --grads        "$LOCAL_GRADS" \
    --results      "$LOCAL_RESULTS" \
    --checkpoint   "$CHECKPOINT_PATH" \
    --data_root    "$WAYMO_KITTI_PATH" \
    --batch_size   2 \
    --iterations   40 \
    --lr           0.01 \
    --dist_weight  1.0 \
    --target_layer "$TARGET_LAYER" \
    --skip_existing

# [5] Package results
echo ""
echo "[5/5] Packaging results..."
ADV_COUNT=$(find "$LOCAL_RESULTS" -maxdepth 1 -type f -name "*.bin" | wc -l)
echo "  Adversarial .bin files: $ADV_COUNT"

# Copy loss histories
find "$LOCAL_RESULTS" -name "loss_history*.pt" -exec cp {} "$DEST_DIR/" \;

if [ "$ADV_COUNT" -gt 0 ]; then
    cd "$LOCAL_RESULTS"
    tar -cf "$SLURM_TMPDIR/$OUT_TAR" ./*
    cp -f "$SLURM_TMPDIR/$OUT_TAR" "$DEST_DIR/"
    echo "  ✓ Saved: $DEST_DIR/$OUT_TAR"
else
    echo "  ⚠ No adversarial files produced!"
fi

# Cleanup symlinks
rm -f "$MMDET_ROOT/data/waymo/kitti_format" "$MMDET_ROOT/data/waymo/waymo_format" 2>/dev/null || true

echo ""
echo "============================================================"
echo "✓ Attack complete"
echo "  Adv .bin count: $ADV_COUNT"
echo "  Gradient files: $GRAD_COUNT"
echo "  Output:         $DEST_DIR/$OUT_TAR"
echo "  Job ID:         $SLURM_JOB_ID"
echo "============================================================"

