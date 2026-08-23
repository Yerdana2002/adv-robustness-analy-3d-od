#!/bin/bash
#SBATCH --account=def-instructor
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=pillarnest_attach_train20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
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

CONFIG_PATH=$MMDET_ROOT/configs/pillarnest/pillarnest_large_adv.py
CHECKPOINT_PATH=$PROJECT_ROOT/checkpoint/pillarnest_large.pth
ATTACK_SCRIPT=$MMDET_ROOT/mmdet3d/models/attack_pillarnest_nus_attach_ddp.py

# Prefer the zstd archive; fall back to plain tar if not present
# 2026-07-27: data/nuscenes_processed.tar.zst was moved off /project
# (373 GB, sha256 2cce584c...2636) and now lives at
#   $HOME/links/scratch/yerdana/nuscenes_processed_may11.tar.zst
# It is byte-identical to what this script previously read.
# data/nuscenes_bevformer.tar.zst is a verified superset of it
# (0 paths missing, +831,314 paths: temporal PKLs, temporal
# gt_database, can_bus) if you would rather use the /project copy.
NUSCENES_TAR_ZSTD=$HOME/links/scratch/yerdana/nuscenes_processed_may11.tar.zst
#/home/yerdana/links/projects/def-instructor/yerdana/data/nuscenes_processed.tar.zst
NUSCENES_TAR_PLAIN=~/links/scratch/yerdana/nuscenes_processed.tar

LOCAL_DATA=$SLURM_TMPDIR
LOCAL_RESULTS=$SLURM_TMPDIR/adv_results

DEST_DIR=$PROJECT_ROOT/data
ADV_OUT_TAR=pillarnest_attach_train20_${SLURM_JOB_ID}.tar.zst

mkdir -p "$LOCAL_RESULTS" "$DEST_DIR"

echo "============================================================"
echo "PillarNeSt Attachment Attack — first 20% of nuScenes train"
echo "Job ID:       $SLURM_JOB_ID"
echo "Config:       $CONFIG_PATH"
echo "Checkpoint:   $CHECKPOINT_PATH"
echo "Output:       $DEST_DIR/$ADV_OUT_TAR"
echo "============================================================"

[ -f "$CONFIG_PATH" ]     || { echo "✗ Missing config"; exit 1; }
[ -f "$CHECKPOINT_PATH" ] || { echo "✗ Missing checkpoint"; exit 1; }
[ -f "$ATTACK_SCRIPT" ]   || { echo "✗ Missing attack script"; exit 1; }

echo "[1/3] Extracting nuScenes..."
if [ -f "$NUSCENES_TAR_ZSTD" ]; then
    echo "Using zstd archive: $NUSCENES_TAR_ZSTD"
    time tar -I "zstd -d -T${SLURM_CPUS_PER_TASK}" -xf "$NUSCENES_TAR_ZSTD" -C "$LOCAL_DATA"
elif [ -f "$NUSCENES_TAR_PLAIN" ]; then
    echo "Using plain tar: $NUSCENES_TAR_PLAIN"
    time tar -xf "$NUSCENES_TAR_PLAIN" -C "$LOCAL_DATA"
else
    echo "✗ No nuScenes archive found"
    exit 1
fi

if   [ -d "$LOCAL_DATA/nuscenes" ];          then NUSCENES_PATH="$LOCAL_DATA/nuscenes"
elif [ -d "$LOCAL_DATA/nuscenes_processed" ]; then NUSCENES_PATH="$LOCAL_DATA/nuscenes_processed"
else echo "✗ extraction failed"; ls -la "$LOCAL_DATA"; exit 1
fi
echo "✓ nuScenes at: $NUSCENES_PATH"

echo "[2/3] Launching DDP attachment attack on 4 GPUs..."
cd "$MMDET_ROOT"
export PYTHONPATH="$MMDET_ROOT:${PYTHONPATH:-}"

GPUS=4
torchrun --standalone --nproc_per_node="$GPUS" "$ATTACK_SCRIPT" \
  --cfg "$CONFIG_PATH" \
  --results "$LOCAL_RESULTS" \
  --checkpoint "$CHECKPOINT_PATH" \
  --data_root "$NUSCENES_PATH" \
  --ann_file "nuscenes_infos_train.pkl" \
  --batch_size 16 \
  --iterations 200 \
  --lr 0.1 \
  --num_add 1024 \
  --sub_loss all \
  --subsample_fraction 0.2 \
  --skip_existing

NUM_BIN=$(find "$LOCAL_RESULTS" -type f -name '*.bin' | wc -l)
echo "✓ Generated $NUM_BIN adversarial point cloud files"
[ "$NUM_BIN" -gt 0 ] || { echo "✗ No adversarial bins produced"; exit 1; }

echo "[3/3] Packing adversarial outputs with zstd..."
cd "$LOCAL_RESULTS"
time tar -I "zstd -T${SLURM_CPUS_PER_TASK} -3" -cf "$SLURM_TMPDIR/$ADV_OUT_TAR" ./*
cp -f "$SLURM_TMPDIR/$ADV_OUT_TAR" "$DEST_DIR/"

# This used to double as a one-time repack: if the zstd archive was absent it
# built one from whatever had been extracted and cached it for later runs.
#
# DISABLED 2026-07-27. NUSCENES_TAR_ZSTD now points at
#   scratch/nuscenes_processed_may11.tar.zst
# which is a sha256-verified copy of the archive deleted from /project. The
# repack would have sourced from NUSCENES_TAR_PLAIN (the Jan 29 tar, believed
# broken) and written over that verified copy under a name implying it was the
# good one -- silent corruption that would only surface as degraded metrics
# much later. Failing loudly is the safer default.
if [ ! -f "$NUSCENES_TAR_ZSTD" ]; then
    echo "✗ Missing $NUSCENES_TAR_ZSTD"
    echo "  Not regenerating it: the only plain-tar source here is"
    echo "  $NUSCENES_TAR_PLAIN (2026-01-29, predates the March re-download)."
    echo "  Use data/nuscenes_bevformer.tar.zst instead -- verified superset,"
    echo "  and the copy that reproduced BEVFormer 0.5174 / BEVFusion 0.7116."
    exit 1
fi

echo "============================================================"
echo "✓ Done"
echo "  Adversarial samples: $NUM_BIN"
echo "  Adversarial archive: $DEST_DIR/$ADV_OUT_TAR"
[ -f "$NUSCENES_TAR_ZSTD" ] && echo "  nuScenes (zstd):     $NUSCENES_TAR_ZSTD"
echo "============================================================"