#!/bin/bash
#SBATCH --account=def-instructor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=attack_ff_lc
#SBATCH --tmp=800G
#SBATCH --signal=B:USR1@1800
# =============================================================================
# FocalFormer3D-LC adversarial attack on nuScenes val. nuScenes ONLY.
#
#   sbatch attack_focalformer_l_nuscenes.sh                              # sigma=0.3
#   sbatch --export=ALL,INIT_NOISE_STD=0.0 attack_focalformer_l_nuscenes.sh
#   sbatch --export=ALL,LOSS_SIGN=0.0 attack_focalformer_l_nuscenes.sh   # control
#
# Replaces focalformer3d_nus_batched_attack.sh, which cannot run as written:
# it points at rrg-instructor (OVER QUOTA -- 28K files against a 2048 limit), at
# a nuscenes_processed.tar that predates the current dataset, and its
# GRADIENT_OUTPUT_DIR line still carries the note "I need to put an actual
# directory".
#
# Only meaningful because of the index_put port
# ---------------------------------------------
# Before it, d(adv)/d(points) was exactly zero and this attack optimised
# nothing -- proven for BEVFusion in job 18858947, where a run WITH the
# adversarial term and one with loss_sign=0 agreed to 0.0010 mAP and disagreed
# per-box at p=0.49 over 124,831 GT boxes. Job 18873745 verified the port
# restores a live gradient while leaving the forward bit-identical.
#
# LOSS_SIGN=0 runs that same falsification test here. If FocalFormer's numbers
# now DIVERGE from the loss_sign=0 control, the adversarial term is
# contributing -- which is the positive result BEVFusion never produced.
#
# 5980, not 6019
# --------------
# 39 val frames end the train pipeline with no GT, so no loss exists, no
# gradient was written for them (job 18875807 produced exactly 5980), and the
# attack skips them. The adversarial set is 5980 clouds; 39 frames stay CLEAN.
# State that wherever these metrics are quoted, and note the eval must not
# assert 6019 overlaid files.
#
# This said 5994 until the full extraction landed. That number counted only
# ObjectNameFilter; ObjectRangeFilter drops 14 more frames whose only GT sits
# outside x,y in (-54, 54). See extract_grad_focalformer_l_nus.sh Step 4 for
# the counts at each filter stage.
#
# INIT_NOISE_STD=0.3 is kept as the default for parity with the existing
# BEVFusion tars, but it is the magnitude that masked the dead gradient for
# months (~387 mm of thrashing vs the ~8.5 mm Chamfer equilibrium). A sigma=0
# run is the one that measures the attack rather than the noise.
#
# 24h is not enough -- ask for more
# --------------------------------
# FocalFormer runs ~52 batches/h, so 1505 batches need ~29h. Job 18882811 was
# submitted at the 24h default and was killed at ~batch 1240 with every cloud
# still on node-local $SLURM_TMPDIR, because Step 5 tars only after python
# returns. The GPU partitions go to 7 days (gpubase_bygpu_b4/b5), so:
#
#     sbatch --time=48:00:00 attack_focalformer_l_nuscenes.sh
#
# The USR1 trap below is the backstop for when that is still not enough:
# --signal=B:USR1@1800 has SLURM raise USR1 half an hour before the wall, and
# the handler tars whatever exists into a _partial<jobid> archive that
# attack_focalformer_lc_resume.sh can stage back in. rescue_running_attack.sh
# does the same thing from outside, for a job that is already running without
# the trap.
# =============================================================================
set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit=$?)" >&2' ERR

INIT_NOISE_STD=${INIT_NOISE_STD:-0.3}
LOSS_SIGN=${LOSS_SIGN:-1.0}
ITERATIONS=${ITERATIONS:-40}
LR=${LR:-0.01}
DIST_WEIGHT=${DIST_WEIGHT:-1.0}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_BATCHES=${MAX_BATCHES:-}

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11
source ~/centerpoint/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export CUDA_HOME=$CUDA_PATH

PROJECT_ROOT=/home/yerdana/links/projects/def-instructor/yerdana
MMDET_ROOT=$PROJECT_ROOT/mmdetection3d
export PYTHONPATH="$MMDET_ROOT:${PYTHONPATH:-}"

CONFIG_PATH=$MMDET_ROOT/projects/configs/focalformer3d/FocalFormer3D_LC_grad_extract.py
CHECKPOINT=$PROJECT_ROOT/checkpoint/FocalFormer3D_LC_ep6_converted.pth
GRAD_TAR=${GRAD_TAR:-/home/yerdana/links/scratch/yerdana/gradients/gradients_focalformer_lc_neck_channel.tar}
NUSCENES_FULL_TAR=$PROJECT_ROOT/data/nuscenes_bevformer.tar.zst
VAL_PKL_TAR=$PROJECT_ROOT/data/temporal_pkls/nuscenes_infos_val_bevfusion.tar
VAL_PKL=nuscenes_infos_val_bevfusion.pkl

RUN_TAG=""
[ "$INIT_NOISE_STD" != "0.3" ] && RUN_TAG="${RUN_TAG}_noise${INIT_NOISE_STD}"
[ "$LOSS_SIGN" != "1.0" ]      && RUN_TAG="${RUN_TAG}_lossign${LOSS_SIGN}"
PERSIST_TAR=$PROJECT_ROOT/data/adv_points_focalformer_lc_channel${RUN_TAG}.tar

WORK_DIR=$PROJECT_ROOT/work_dirs/attack_ff_lc_${SLURM_JOB_ID}
mkdir -p "$WORK_DIR"

for f in "$CONFIG_PATH" "$CHECKPOINT" "$GRAD_TAR" "$NUSCENES_FULL_TAR" "$VAL_PKL_TAR"; do
    [ -f "$f" ] || { echo "X missing: $f"; exit 1; }
done

# 30 GB for the adversarial tar. /project sat at 950 of 1000 GB before job
# 18871825 moved the gradient sets off; do not let it silently refill.
AVAIL_GB=$(df -BG --output=avail "$PROJECT_ROOT" | tail -1 | tr -dc '0-9')
echo "  /project avail: ${AVAIL_GB} GB (need ~30)"
[ "$AVAIL_GB" -gt 35 ] || { echo "X not enough room for the adversarial tar"; exit 1; }

# Step 2 repoints the SHARED symlink. Job 18801696 died 4h55m in when another
# job moved it underneath.
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
echo "FocalFormer3D-LC adversarial attack (nuScenes val)"
echo "  checkpoint     : $(basename "$CHECKPOINT")"
echo "  gradients      : $(basename "$GRAD_TAR")"
echo "  init_noise_std : $INIT_NOISE_STD"
echo "  loss_sign      : $LOSS_SIGN$([ "$LOSS_SIGN" = "0.0" ] && echo '   <-- CONTROL: adversarial term removed')"
echo "  iterations     : $ITERATIONS   lr=$LR   dist_weight=$DIST_WEIGHT"
echo "  output tar     : $PERSIST_TAR"
echo "============================================================"

echo ""
echo "=== Step 1: Extracting nuScenes ==="
FULL_TMP="$SLURM_TMPDIR/full"; mkdir -p "$FULL_TMP"
time tar -I "zstd -d" -xf "$NUSCENES_FULL_TAR" -C "$FULL_TMP"
FULL_PATH=""
for c in nuscenes nuscenes_processed; do
    [ -d "$FULL_TMP/$c" ] && FULL_PATH="$FULL_TMP/$c" && break
done
[ -z "$FULL_PATH" ] && { echo "X nuScenes dir not found"; exit 1; }

echo ""
echo "=== Step 2: val pkl + symlink ==="
tar xf "$VAL_PKL_TAR" -C "$FULL_PATH"
mkdir -p "$MMDET_ROOT/data"
ln -sfn "$FULL_PATH" "$MMDET_ROOT/data/nuscenes"

echo ""
echo "=== Step 3: Staging gradients ==="
# Node-local, not read across the network 40x per epoch.
GRAD_DIR=$SLURM_TMPDIR/gradients; mkdir -p "$GRAD_DIR"
time tar -xf "$GRAD_TAR" -C "$GRAD_DIR"
NGRAD=$(find "$GRAD_DIR" -name '*_grad.pt' | wc -l)
echo "  staged $NGRAD gradient tensors (expect 5980)"
[ "$NGRAD" -gt 0 ] || { echo "X no gradients staged"; exit 1; }
[ "$NGRAD" -eq 5980 ] || echo "  !! not the 5980 of job 18875807 -- check the tar before trusting the run"

echo ""
echo "=== Step 4: Attack ==="
RESULT_DIR=$SLURM_TMPDIR/adv_points; mkdir -p "$RESULT_DIR"

# Wall backstop. Everything the attack produces lives on node-local
# $SLURM_TMPDIR until Step 5, and SLURM purges that the instant the job ends.
# Job 18882811 lost ~21 GPU-hours that way. USR1 arrives 1800s early; tar what
# exists so attack_focalformer_lc_resume.sh can pick it up.
persist () {
    local dest=$1 n
    n=$(find "$RESULT_DIR" -name '*.bin' | wc -l)
    echo ""
    echo "=== Persisting $n clouds -> $(basename "$dest") ==="
    [ "$n" -gt 0 ] || { echo "  nothing to persist"; return 0; }
    # Write then rename: an interrupted tar must not land on a good archive.
    tar -cf "${dest}.partial" -C "$RESULT_DIR" . && mv -f "${dest}.partial" "$dest"
    ls -lh "$dest"
}
on_wall () {
    echo ""
    echo "!! USR1: 30 min to the wall, persisting before \$SLURM_TMPDIR is purged"
    persist "$PROJECT_ROOT/data/adv_points_focalformer_lc_channel_partial${SLURM_JOB_ID}.tar"
    echo "!! resume: sbatch --export=ALL,RESUME_JOB=${SLURM_JOB_ID}$([ "$LOSS_SIGN" != "1.0" ] && echo ",LOSS_SIGN=$LOSS_SIGN") attack_focalformer_lc_resume.sh"
    exit 1
}
trap on_wall USR1

cd "$MMDET_ROOT"
ARGS=(
    --cfg "$CONFIG_PATH"
    --grads "$GRAD_DIR"
    --results "$RESULT_DIR/"
    --checkpoint "$CHECKPOINT"
    --data_root "$FULL_PATH/"
    --batch_size "$BATCH_SIZE"
    --iterations "$ITERATIONS"
    --lr "$LR"
    --dist_weight "$DIST_WEIGHT"
    --init_noise_std "$INIT_NOISE_STD"
    --loss_sign "$LOSS_SIGN"
    --target_layer neck
)
[ -n "$MAX_BATCHES" ] && ARGS+=(--max_batches "$MAX_BATCHES")
# Backgrounded + wait: bash defers a trap until the running foreground command
# returns, so a foreground python would swallow USR1 until the attack finished
# -- which is exactly the case the trap exists for.
time python projects/mmdet3d_plugin/models/attack_focalformer_nus.py "${ARGS[@]}" &
wait $!

echo ""
echo "=== Step 5: Persist ==="
NADV=$(find "$RESULT_DIR" -name '*.bin' | wc -l)
echo "  adversarial clouds: $NADV (expect 5980; 39 frames have no gradient)"
[ "$NADV" -gt 0 ] || { echo "X no adversarial clouds produced"; exit 1; }
[ "$NADV" -eq 5980 ] || echo "  !! short of 5980 -- resume before evaluating"
time persist "$PERSIST_TAR"
echo ""
echo "Done. Eval with the FocalFormer variant -- it must NOT assert 6019"
echo "overlaid files; $NADV is correct and the remainder stay clean."
