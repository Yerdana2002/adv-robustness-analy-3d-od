#!/bin/bash
#SBATCH --account=def-zhengliu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=ff_attack_resume
#SBATCH --tmp=800G
#SBATCH --signal=B:USR1@1800
# =============================================================================
# Resume an interrupted FocalFormer3D-L attack from rescued partial tars.
#
#   sbatch attack_focalformer_l_resume.sh
#   sbatch --export=ALL,LOSS_SIGN=0.0 attack_focalformer_l_resume.sh
#
# Why this is separate from attack_focalformer_l_nuscenes.sh
# ----------------------------------------------------------
# Resuming needs staging that a fresh run must not do: extracting partial tars
# over $RESULT_DIR before the attack starts. Folding that into the main script
# behind a flag would put the recovery path in the same file as the thing it
# recovers from, and the main script has to stay runnable from scratch for the
# control arm.
#
# (Editing the main script mid-run would have been safe -- SLURM copies the
# batch script to the node's spool at launch, so a running job never re-reads
# the original path. That is why the USR1 trap below could also be added to
# attack_focalformer_l_nuscenes.sh while 18882811 was still executing it.)
#
# What went wrong, and the two fixes
# ----------------------------------
# FocalFormer runs ~52 batches/h. A full 1505-batch pass needs ~29h, and
# attack_focalformer_l_nuscenes.sh asks for 24h, so 18882811 was killed at
# ~batch 1240 with every cloud still on node-local $SLURM_TMPDIR -- Step 5
# tars only after python returns. rescue_running_attack.sh pulled 4309+ of
# them out through `srun --overlap` before the wall.
#
#   1. --signal=B:USR1@1800 plus the trap below. SLURM raises USR1 half an
#      hour before the wall; the trap tars whatever exists and ships it. The
#      job can no longer die holding the only copy.
#   2. The GPU partitions go to 7 days (gpubase_bygpu_b4/b5); 24h was never
#      required. The control runs at 48h and needs no resume at all.
#
# Staging order matters: base tar first, then _inc2, _inc3, ... so later
# rescues win on any frame captured twice. --skip_existing then makes the
# attack step over every frame already on disk.
# =============================================================================
set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit=$?)" >&2' ERR

INIT_NOISE_STD=${INIT_NOISE_STD:-0.3}
LOSS_SIGN=${LOSS_SIGN:-1.0}
ITERATIONS=${ITERATIONS:-40}
LR=${LR:-0.01}
DIST_WEIGHT=${DIST_WEIGHT:-1.0}
BATCH_SIZE=${BATCH_SIZE:-4}
RESUME_JOB=${RESUME_JOB:-18882811}

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

CONFIG_PATH=$MMDET_ROOT/projects/configs/focalformer3d/FocalFormer3D_L_grad_extract.py
CHECKPOINT=$PROJECT_ROOT/checkpoint/FocalFormer3D_L_ep6_converted.pth
GRAD_TAR=${GRAD_TAR:-/home/yerdana/links/scratch/yerdana/gradients/gradients_focalformer_l_neck_channel.tar}
NUSCENES_FULL_TAR=$PROJECT_ROOT/data/nuscenes_bevformer.tar.zst
VAL_PKL_TAR=$PROJECT_ROOT/data/temporal_pkls/nuscenes_infos_val_bevfusion.tar
VAL_PKL=nuscenes_infos_val_bevfusion.pkl

RUN_TAG=""
[ "$INIT_NOISE_STD" != "0.3" ] && RUN_TAG="${RUN_TAG}_noise${INIT_NOISE_STD}"
[ "$LOSS_SIGN" != "1.0" ]      && RUN_TAG="${RUN_TAG}_lossign${LOSS_SIGN}"
PERSIST_TAR=$PROJECT_ROOT/data/adv_points_focalformer_l_channel${RUN_TAG}.tar
PARTIAL_GLOB="$PROJECT_ROOT/data/adv_points_focalformer_l_channel_partial${RESUME_JOB}"

WORK_DIR=$PROJECT_ROOT/work_dirs/attack_ff_resume_${SLURM_JOB_ID}
mkdir -p "$WORK_DIR"

for f in "$CONFIG_PATH" "$CHECKPOINT" "$GRAD_TAR" "$NUSCENES_FULL_TAR" "$VAL_PKL_TAR"; do
    [ -f "$f" ] || { echo "X missing: $f"; exit 1; }
done
[ -f "${PARTIAL_GLOB}.tar" ] || { echo "X no base partial tar ${PARTIAL_GLOB}.tar"; exit 1; }

AVAIL_GB=$(df -BG --output=avail "$PROJECT_ROOT" | tail -1 | tr -dc '0-9')
echo "  /project avail: ${AVAIL_GB} GB (need ~35)"
[ "$AVAIL_GB" -gt 40 ] || { echo "X not enough room for the adversarial tar"; exit 1; }

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

RESULT_DIR=$SLURM_TMPDIR/adv_points
mkdir -p "$RESULT_DIR"

# --- the thing whose absence cost 21 GPU-hours -------------------------------
# SLURM raises USR1 1800s before the wall. Persist and exit rather than let
# $SLURM_TMPDIR be purged with the only copy of the run in it.
persist () {
    local dest=$1 n
    n=$(find "$RESULT_DIR" -name '*.bin' | wc -l)
    echo ""
    echo "=== Persisting $n clouds -> $(basename "$dest") ==="
    [ "$n" -gt 0 ] || { echo "  nothing to persist"; return 0; }
    # Write beside the target then rename: a tar interrupted mid-write must
    # never land on top of a good archive under the real name.
    tar -cf "${dest}.partial" -C "$RESULT_DIR" . && mv -f "${dest}.partial" "$dest"
    ls -lh "$dest"
}
on_wall () {
    echo ""
    echo "!! USR1: 30 min to the wall, persisting before SLURM purges \$SLURM_TMPDIR"
    persist "${PARTIAL_GLOB}_resume${SLURM_JOB_ID}.tar"
    echo "!! resume again with: sbatch --export=ALL,RESUME_JOB=${RESUME_JOB} $0"
    exit 1
}
trap on_wall USR1

echo "============================================================"
echo "FocalFormer3D-L attack RESUME (nuScenes val)"
echo "  resuming      : job $RESUME_JOB"
echo "  loss_sign     : $LOSS_SIGN$([ "$LOSS_SIGN" = "0.0" ] && echo '   <-- CONTROL')"
echo "  iterations    : $ITERATIONS   lr=$LR   dist_weight=$DIST_WEIGHT"
echo "  output tar    : $PERSIST_TAR"
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
GRAD_DIR=$SLURM_TMPDIR/gradients; mkdir -p "$GRAD_DIR"
# Foreground here and in Step 1: bash defers a trap until the running
# foreground command returns, but both finish long before the wall is in play.
# Only the attack itself is backgrounded, where a deferred USR1 would be fatal.
time tar -xf "$GRAD_TAR" -C "$GRAD_DIR"
NGRAD=$(find "$GRAD_DIR" -name '*_grad.pt' | wc -l)
echo "  staged $NGRAD gradient tensors (expect 5980)"
[ "$NGRAD" -eq 5980 ] || echo "  !! not the 5980 of job 18875807 -- check the tar"

echo ""
echo "=== Step 4: Staging rescued partials ==="
# Base first, then _inc2, _inc3, ... in numeric order so the newest capture of
# any duplicated frame is the one left on disk.
STAGED=()
[ -f "${PARTIAL_GLOB}.tar" ] && STAGED+=("${PARTIAL_GLOB}.tar")
while IFS= read -r t; do STAGED+=("$t"); done < <(
    ls -1 "${PARTIAL_GLOB}"_inc*.tar "${PARTIAL_GLOB}"_resume*.tar 2>/dev/null \
    | sed 's/.*_inc\([0-9]*\)\.tar/\1 &/; t; s/.*/9999 &/' | sort -n | cut -d' ' -f2-)
for t in "${STAGED[@]}"; do
    echo "  <- $(basename "$t")"
    tar -xf "$t" -C "$RESULT_DIR"
done
NSTAGED=$(find "$RESULT_DIR" -name '*.bin' | wc -l)
echo "  staged $NSTAGED already-attacked clouds"
[ "$NSTAGED" -gt 0 ] || { echo "X nothing staged -- this is not a resume"; exit 1; }
echo "  remaining to attack: ~$((5980 - NSTAGED))"

echo ""
echo "=== Step 5: Attack (resume) ==="
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
    --skip_existing
)
# Backgrounded + wait so the USR1 trap fires promptly instead of queueing
# behind a foreground python that bash will not interrupt.
time python projects/mmdet3d_plugin/models/attack_focalformer_nus.py "${ARGS[@]}" &
wait $!

echo ""
echo "=== Step 6: Persist ==="
NADV=$(find "$RESULT_DIR" -name '*.bin' | wc -l)
echo "  adversarial clouds: $NADV (expect 5980; 39 frames have no gradient)"
[ "$NADV" -eq 5980 ] || echo "  !! short of 5980 -- resume again before evaluating"
persist "$PERSIST_TAR"

echo ""
echo "Done. Eval must NOT assert 6019 overlaid files; $NADV is correct."
