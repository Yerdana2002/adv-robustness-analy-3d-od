#!/bin/bash
#SBATCH --account=def-instructor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=ff_lc_clean
#SBATCH --tmp=800G
# =============================================================================
# THE GATE. Clean nuScenes val eval of FocalFormer3D-LC, nothing else.
#
#   sbatch eval_focalformer_lc_clean.sh
#
# FocalFormer3D_LC_test.py is a config that did not exist until now -- there
# was no LC config in this project, only commented-out fragments. Every value
# in it was recovered from those fragments or read off the checkpoint, and
# diag_focalformer_lc_keys.sh already proved every tensor lines up by name and
# shape. That is necessary but NOT sufficient: correct weights in a model whose
# images are normalised wrong, resized wrong, or projected with the wrong
# lidar2img still evaluate, still produce numbers, and are still wrong.
#
# So this runs before the ~1.5h gradient extraction and the ~30h attack, and
# the whole point is the comparison at the bottom:
#
#     reference   mAP 0.7050   NDS 0.7310     (from the checkpoint's own name)
#
# For calibration, the LiDAR-only config reproduces its reference to within 0.6
# of a point (0.6578 vs 0.664, 0.7042 vs 0.709), so treat a gap much beyond
# ~1 point as a failure and fix the config rather than proceeding.
#
# The most likely ways to land close-but-wrong, in rough order:
#   - bgr_to_rgb. The original img_norm_cfg says to_rgb=True and
#     BEVLoadMultiViewImageFromFiles returns BGR, so the preprocessor must
#     swap. Getting this backwards costs a few points, not all of them.
#   - lidar2img vs img_aug_matrix. FocalEncoder inverts lidar2img for the
#     camera rotations and LiftSplatShoot separately applies img_aug_matrix to
#     undo the resize/crop. Lose either and camera features splat to the wrong
#     BEV cells.
#   - resize_lim. 0.5 on 1600x900 gives 800x450, cropped to 448.
# =============================================================================
set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit=$?)" >&2' ERR

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

NUSCENES_FULL_TAR=$PROJECT_ROOT/data/nuscenes_bevformer.tar.zst
VAL_PKL_TAR=$PROJECT_ROOT/data/temporal_pkls/nuscenes_infos_val_bevfusion.tar
VAL_PKL=nuscenes_infos_val_bevfusion.pkl
CFG=$MMDET_ROOT/projects/configs/focalformer3d/FocalFormer3D_LC_test.py
CKPT=$PROJECT_ROOT/checkpoint/FocalFormer3D_LC_ep6_converted.pth

OUT_DIR=$PROJECT_ROOT/work_dirs/ff_lc_clean_${SLURM_JOB_ID}
mkdir -p "$OUT_DIR"

for f in "$NUSCENES_FULL_TAR" "$VAL_PKL_TAR" "$CFG" "$CKPT"; do
    [ -f "$f" ] || { echo "X missing: $f"; exit 1; }
done

if [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
    OTHERS=$(squeue -h -u "$USER" -t RUNNING -o '%i %j' 2>/dev/null \
        | awk -v me="${SLURM_JOB_ID:-0}" '$1 != me {printf "%s(%s) ", $1, $2}') || OTHERS=""
    if [ -n "$OTHERS" ]; then
        echo "X another job is RUNNING and Step 2 would move the shared symlink: $OTHERS"
        echo "  override: sbatch --export=ALL,ALLOW_CONCURRENT=1 $0"
        exit 1
    fi
fi

echo "============================================================"
echo "FocalFormer3D-LC clean eval (nuScenes val) -- THE GATE"
echo "  config     : $(basename "$CFG")"
echo "  checkpoint : $(basename "$CKPT")"
echo "  reference  : mAP 0.7050  NDS 0.7310"
echo "============================================================"

echo ""
echo "=== Step 1: Extracting full nuScenes (zstd) ==="
FULL_TMP="$SLURM_TMPDIR/full"; mkdir -p "$FULL_TMP"
df -h "$SLURM_TMPDIR" | tail -1
time tar -I "zstd -d" -xf "$NUSCENES_FULL_TAR" -C "$FULL_TMP"
FULL_PATH=""
for c in nuscenes nuscenes_processed; do
    [ -d "$FULL_TMP/$c" ] && FULL_PATH="$FULL_TMP/$c" && break
done
[ -z "$FULL_PATH" ] && { echo "X nuScenes dir not found"; exit 1; }
echo "OK FULL_PATH=$FULL_PATH"

# Every FocalFormer/BEVFusion run so far has been LiDAR-only, so this tar has
# never had to contain camera data. Check before spending an hour finding out.
echo ""
echo "=== Step 1b: Camera data present? ==="
MISSING_CAM=0
for c in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT; do
    # -print -quit, NOT `| head -1`. Under `set -Eeuo pipefail` head closes the
    # pipe, find dies of SIGPIPE, the pipeline returns 141 and the whole job
    # aborts -- which is exactly how job 19032727 threw away a 16-minute
    # extraction at the hands of its own safety check.
    first=$(find "$FULL_PATH/samples/$c" -name '*.jpg' -print -quit 2>/dev/null || true)
    if [ -z "$first" ]; then echo "  X samples/$c EMPTY or absent"; MISSING_CAM=1
    else echo "  ok samples/$c"; fi
done
[ "$MISSING_CAM" -eq 0 ] || {
    echo "X this nuScenes tar has no camera images; an LC eval is impossible"
    echo "  from it. A camera-bearing extract is needed before anything else."
    exit 1; }

echo ""
echo "=== Step 2: val pkl + symlink ==="
tar xf "$VAL_PKL_TAR" -C "$FULL_PATH"
mkdir -p "$MMDET_ROOT/data"
ln -sfn "$FULL_PATH" "$MMDET_ROOT/data/nuscenes"
cd "$MMDET_ROOT"

echo ""
echo "=== Step 3: Clean eval ==="
WD=$OUT_DIR/eval_clean; mkdir -p "$WD"
time python tools/test.py "$CFG" "$CKPT" --work-dir "$WD" --cfg-options \
    "test_dataloader.dataset.data_root='${FULL_PATH}/'" \
    "test_dataloader.dataset.ann_file='${VAL_PKL}'" \
    "test_dataloader.num_workers=4" \
    "test_evaluator.data_root='${FULL_PATH}/'" \
    "test_evaluator.ann_file='${FULL_PATH}/${VAL_PKL}'" \
    "test_evaluator.jsonfile_prefix=${WD}/results" 2>&1 | tail -40

echo ""
echo "============================================================"
echo "GATE RESULT"
echo "============================================================"
LG=$(find "$WD" -name '*.log' -type f 2>/dev/null | sort | tail -1)
GOT=$(grep -oE 'NuScenes/(NDS|mAP): [0-9.]+' "$LG" 2>/dev/null | tail -2 | tr '\n' ' ')
echo "  measured  : $GOT"
echo "  reference : NuScenes/NDS: 0.7310 NuScenes/mAP: 0.7050"
python - <<PYGATE
import re
got = """$GOT"""
d = dict(re.findall(r'NuScenes/(NDS|mAP): ([0-9.]+)', got))
if not d:
    print('  !! could not parse metrics from the log'); raise SystemExit(1)
ref = dict(NDS=0.7310, mAP=0.7050)
bad = False
for k, r in ref.items():
    v = float(d.get(k, 'nan'))
    gap = v - r
    print(f'  {k:<4}: {v:.4f} vs {r:.4f}   gap {gap:+.4f}')
    if abs(gap) > 0.01:
        bad = True
print()
print('VERDICT:', 'FAIL -- config is wrong somewhere, do NOT extract gradients'
      if bad else 'PASS -- LC config reproduces the checkpoint; proceed')
PYGATE
echo ""
echo "Done."
