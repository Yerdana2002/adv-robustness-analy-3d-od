#!/bin/bash
#SBATCH --account=def-instructor
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=pillarnest_adv_resume_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --tmp=1200G

set -euo pipefail

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11
source ~/centerpoint/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

PROJECT_ROOT=~/links/projects/def-instructor/yerdana
MMDET_ROOT=$PROJECT_ROOT/mmdetection3d

CONFIG_PATH=$MMDET_ROOT/configs/pillarnest/pillarnest_large_mininus.py
CHECKPOINT_PATH=$PROJECT_ROOT/checkpoint/pillarnest_large.pth
ATTACK_SCRIPT=$MMDET_ROOT/mmdet3d/models/attack_pillarnest_nus_attach_ddp.py

NUSCENES_TAR=~/links/scratch/yerdana/nuscenes_processed.tar
GRADIENT_TAR=~/links/scratch/yerdana/pillarnest/gradients/pillarnest_full_nus_val_middle_encoder_grads_7059518.tar

PREV_ADV_1=/home/yerdana/links/scratch/yerdana/pillarnest/adversarial/pillarnest_adv_full_ddp_7068171.tar
PREV_ADV_2=/home/yerdana/links/scratch/yerdana/pillarnest/adversarial/pillarnest_adv_full_ddp_7087958.tar

LOCAL_DATA=$SLURM_TMPDIR
LOCAL_GRADS=$SLURM_TMPDIR/gradients
LOCAL_RESULTS=$SLURM_TMPDIR/adv_results
DEST_DIR=~/links/scratch/yerdana/pillarnest/adversarial
OUT_TAR=pillarnest_adv_resume_ddp_${SLURM_JOB_ID}.tar

mkdir -p "$LOCAL_GRADS" "$LOCAL_RESULTS" "$DEST_DIR"

echo "Extracting nuScenes..."
tar -xf "$NUSCENES_TAR" -C "$LOCAL_DATA"
if [ -d "$LOCAL_DATA/nuscenes" ]; then
  NUSCENES_PATH="$LOCAL_DATA/nuscenes"
elif [ -d "$LOCAL_DATA/nuscenes_processed" ]; then
  NUSCENES_PATH="$LOCAL_DATA/nuscenes_processed"
else
  echo "✗ nuScenes extraction failed"
  ls -la "$LOCAL_DATA"
  exit 1
fi

echo "Extracting gradients..."
tar -xf "$GRADIENT_TAR" -C "$LOCAL_GRADS"

echo "Restoring previous adversarial outputs for resume..."
if [ -f "$PREV_ADV_1" ]; then
  tar -xf "$PREV_ADV_1" -C "$LOCAL_RESULTS"
fi
if [ -f "$PREV_ADV_2" ]; then
  tar -xf "$PREV_ADV_2" -C "$LOCAL_RESULTS"
fi
echo "Existing adversarial files before run: $(find "$LOCAL_RESULTS" -type f -name '*.bin' | wc -l)"

cd "$MMDET_ROOT"
export PYTHONPATH="$MMDET_ROOT:${PYTHONPATH:-}"

# Fixed to 2 as requested; avoids weird SLURM var parsing
GPUS=2

torchrun --standalone --nproc_per_node="$GPUS" "$ATTACK_SCRIPT" \
  --cfg "$CONFIG_PATH" \
  --grads "$LOCAL_GRADS" \
  --results "$LOCAL_RESULTS" \
  --checkpoint "$CHECKPOINT_PATH" \
  --data_root "$NUSCENES_PATH" \
  --ann_file "nuscenes_infos_val.pkl" \
  --batch_size 8 \
  --iterations 40 \
  --lr 0.01 \
  --dist_weight 1.0 \
  --target_layer "pts_middle_encoder" \
  --skip_existing

echo "Packing results..."
cd "$LOCAL_RESULTS"
tar -cf "$SLURM_TMPDIR/$OUT_TAR" ./*
cp -f "$SLURM_TMPDIR/$OUT_TAR" "$DEST_DIR/"

echo "✓ Done: $DEST_DIR/$OUT_TAR"

