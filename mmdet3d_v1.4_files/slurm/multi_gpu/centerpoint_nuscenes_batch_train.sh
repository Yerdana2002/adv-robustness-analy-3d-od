#!/bin/bash
#SBATCH --account=rrg-instructor
#SBATCH --time=36:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=nuscenes_adv_ddp4_final
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --tmp=3000G

set -euo pipefail

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11
source ~/centerpoint/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

PROJECT_ROOT=~/links/projects/rrg-instructor/yerdana
MMDET_ROOT=$PROJECT_ROOT/mmdetection3d

CONFIG_PATH=$MMDET_ROOT/configs/centerpoint/centerpoint_voxel0075_dcn_nus_adv.py
CHECKPOINT_PATH=$PROJECT_ROOT/checkpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth
ATTACK_SCRIPT=$MMDET_ROOT/mmdet3d/models/centerpoint_nuscenes_batch.py

NUSCENES_TAR=~/links/scratch/yerdana/nuscenes_processed.tar
GRADIENT_TAR=$PROJECT_ROOT/data/results/nuscenes_centerpoint_gradients.tar

LOCAL_DATA=$SLURM_TMPDIR
LOCAL_GRADS=$SLURM_TMPDIR/gradients
LOCAL_RESULTS=$SLURM_TMPDIR/adversarial_results

DEST_DIR=~/links/scratch/yerdana/centerpoint_nus_pc
OUT_TAR=nuscenes_adv_ddp4_final_${SLURM_JOB_ID}.tar

mkdir -p "$LOCAL_GRADS" "$LOCAL_RESULTS" "$DEST_DIR"

echo "Extracting nuScenes..."
tar -xf "$NUSCENES_TAR" -C "$LOCAL_DATA"
if [ -d "$LOCAL_DATA/nuscenes" ]; then
  NUSCENES_PATH="$LOCAL_DATA/nuscenes"
elif [ -d "$LOCAL_DATA/nuscenes_processed" ]; then
  NUSCENES_PATH="$LOCAL_DATA/nuscenes_processed"
else
  echo "ERROR: nuScenes extraction failed"
  ls -la "$LOCAL_DATA"
  exit 1
fi
[ -f "$NUSCENES_PATH/nuscenes_infos_val.pkl" ] || { echo "Missing nuscenes_infos_val.pkl"; exit 1; }

ls -l "$NUSCENES_PATH/nuscenes_dbinfos_train.pkl"
ls -l "$MMDET_ROOT/data/nuscenes/nuscenes_dbinfos_train.pkl" || true
readlink -f "$MMDET_ROOT/data/nuscenes" || true


echo "Extracting gradients..."
tar -xf "$GRADIENT_TAR" -C "$LOCAL_GRADS"
GRAD_COUNT=$(ls -1 "$LOCAL_GRADS"/*.pt 2>/dev/null | wc -l || true)
echo "Gradient files: $GRAD_COUNT"

cd "$MMDET_ROOT"
export PYTHONPATH="$MMDET_ROOT:${PYTHONPATH:-}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
python - <<'PY'
import torch, os
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY


# ---- GPU health probe (strict 4-GPU requirement) ----
REQUIRE_GPUS=2
VISIBLE_CUDA="${CUDA_VISIBLE_DEVICES:-0,1}"

IFS=',' read -r -a CANDIDATE_IDS <<< "$VISIBLE_CUDA"
GOOD_IDS=()

echo "Probing CUDA devices: ${CANDIDATE_IDS[*]}"
for gid in "${CANDIDATE_IDS[@]}"; do
  if CUDA_VISIBLE_DEVICES="$gid" python - <<'PY' >/dev/null 2>&1
import torch
torch.cuda.init()
torch.cuda.get_device_properties(0)
print("OK")
PY
  then
    GOOD_IDS+=("$gid")
    echo "GPU $gid: OK"
  else
    echo "GPU $gid: BAD"
  fi
done

if [ "${#GOOD_IDS[@]}" -lt "$REQUIRE_GPUS" ]; then
  echo "ERROR: Need $REQUIRE_GPUS healthy GPUs, found ${#GOOD_IDS[@]} (${GOOD_IDS[*]})."
  echo "Requeue on another node or lower REQUIRE_GPUS."
  exit 2
fi

USE_IDS=("${GOOD_IDS[@]:0:$REQUIRE_GPUS}")
USE_VISIBLE=$(IFS=,; echo "${USE_IDS[*]}")
export CUDA_VISIBLE_DEVICES="$USE_VISIBLE"

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

torchrun --standalone --nproc_per_node="$REQUIRE_GPUS" /home/yerdana/links/projects/rrg-instructor/yerdana/mmdetection3d/mmdet3d/models/centerpoint_nuscenes_batch.py \
  --cfg "$CONFIG_PATH" \
  --grads "$LOCAL_GRADS" \
  --results "$LOCAL_RESULTS" \
  --checkpoint "$CHECKPOINT_PATH" \
  --data_root "$NUSCENES_PATH" \
  --batch_size 16 \
  --iterations 40 \
  --lr 0.01 \
  --dist_weight 1.0 \
  --target_layer blocks.0

echo "Packaging results..."
ADV_COUNT=$(ls -1 "$LOCAL_RESULTS"/*.bin 2>/dev/null | wc -l || true)
echo "Generated adversarial point clouds: $ADV_COUNT"

cd "$LOCAL_RESULTS"
tar -cf "$SLURM_TMPDIR/$OUT_TAR" ./*.bin loss_history_rank*.pt
cp -f "$SLURM_TMPDIR/$OUT_TAR" "$DEST_DIR/"

echo "Done: $DEST_DIR/$OUT_TAR"
