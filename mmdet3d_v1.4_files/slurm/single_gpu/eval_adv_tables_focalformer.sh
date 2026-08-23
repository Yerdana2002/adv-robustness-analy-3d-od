#!/bin/bash
#SBATCH --account=def-instructor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=ff_adv_tables
#SBATCH --tmp=800G
# =============================================================================
# nuScenes ONLY. Metrics + AI@x tables for the FocalFormer3D-L adversarial set,
# in the same format as the BEVFusion tables so the four can be sliced together
# by metric_scripts/adv_breakdown.py.
#
#   sbatch eval_adv_tables_focalformer.sh
#
# Produces:
#   - NDS / mAP for clean and adversarial (nuScenes devkit, 6019-sample split)
#   - results_nusc.json per eval
#   - metric_scripts/tables_focalformer-l_lidar.pkl   boxes_df / samples_df
#     carrying map location, BEV radius, ego distance and lidar-point count
#     per GT box, matching the columns build_tables.py writes for BEVFusion.
#
# 5980 overlaid, not 6019
# -----------------------
# 39 val frames end the train pipeline with no GT (3 have no instances, 7 after
# use_valid_flag, 25 after ObjectNameFilter, 39 after ObjectRangeFilter), so no
# loss existed for them, no gradient was extracted, and the attack skipped
# them. Those 39 keyframes stay CLEAN under the overlay. All 6019 samples are
# still SCORED -- NuScenesMetric asserts full coverage -- so the adversarial
# metrics below are over a split that is 0.65% pristine. State that wherever
# they are quoted. This is unlike the BEVFusion sets, which cover all 6019.
#
# The loss_sign=0 control is deliberately NOT here
# ------------------------------------------------
# Job 18931272 produces adv_points_focalformer_l_channel_lossign0.0.tar, and it
# is excluded from these metric files by request. It is still the falsification
# test -- whether FocalFormer's adversarial term does anything, or whether the
# dead gradient proven for BEVFusion in 18858947 survives the architecture
# change -- so it can be evaluated separately by adding its tar to SETS. The
# tables here are the attack arm only.
# =============================================================================
set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit=$?)" >&2' ERR

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11
source ~/centerpoint/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export CUDA_HOME=$CUDA_PATH

PROJECT_ROOT=/home/yerdana/links/projects/def-instructor/yerdana
MMDET_ROOT=$PROJECT_ROOT/mmdetection3d
export PYTHONPATH="$MMDET_ROOT:${PYTHONPATH:-}"

NUSCENES_FULL_TAR=$PROJECT_ROOT/data/nuscenes_bevformer.tar.zst
VAL_PKL_TAR=$PROJECT_ROOT/data/temporal_pkls/nuscenes_infos_val_bevfusion.tar
VAL_PKL=nuscenes_infos_val_bevfusion.pkl
# The same checkpoint the gradients and the attack were produced with. The
# sibling FocalFormer3D_L_ep6_mAP664_NDS709.pth names the reference figures:
# the clean eval below should land near mAP 0.664 / NDS 0.709.
CKPT=$PROJECT_ROOT/checkpoint/FocalFormer3D_L_ep6_converted.pth
CFG=$MMDET_ROOT/projects/configs/focalformer3d/FocalFormer3d_L_test.py

NFRAMES=5980

# name | tar | model tag | attack tag
SETS=(
  "lidar|$PROJECT_ROOT/data/adv_points_focalformer_l_channel.tar|focalformer-l|advlidar"
)

OUT_DIR=$PROJECT_ROOT/work_dirs/ff_adv_tables_${SLURM_JOB_ID}
TABLE_DIR=$PROJECT_ROOT/metric_scripts
mkdir -p "$OUT_DIR"

for f in "$NUSCENES_FULL_TAR" "$VAL_PKL_TAR" "$CKPT" "$CFG" \
         "$TABLE_DIR/build_tables.py" "$TABLE_DIR/adv_breakdown.py"; do
    [ -f "$f" ] || { echo "X missing: $f"; exit 1; }
done
for s in "${SETS[@]}"; do
    t=$(echo "$s" | cut -d'|' -f2)
    [ -f "$t" ] || { echo "X missing adv tar: $t"; exit 1; }
done

# Step 2 repoints the SHARED symlink $MMDET_ROOT/data/nuscenes. Job 18801696
# died 4h55m in when a second job did that underneath it.
if [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
    OTHERS=$(squeue -h -u "$USER" -t RUNNING -o '%i %j' 2>/dev/null \
        | awk -v me="${SLURM_JOB_ID:-0}" '$1 != me {printf "%s(%s) ", $1, $2}') || OTHERS=""
    if [ -n "$OTHERS" ]; then
        echo "X another job is RUNNING and Step 2 would move the shared"
        echo "  symlink under it: $OTHERS"
        echo "  override: sbatch --export=ALL,ALLOW_CONCURRENT=1 $0"
        exit 1
    fi
fi

echo "============================================================"
echo "FocalFormer3D-L adversarial metrics + AI@x tables (nuScenes)"
echo "  checkpoint : $(basename "$CKPT")"
echo "  config     : $(basename "$CFG")"
echo "  overlaid   : $NFRAMES of 6019 (39 frames have no gradient, stay clean)"
echo "  out        : $OUT_DIR"
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

echo ""
echo "=== Step 2: val pkl + symlink ==="
tar xf "$VAL_PKL_TAR" -C "$FULL_PATH"
mkdir -p "$MMDET_ROOT/data"
ln -sfn "$FULL_PATH" "$MMDET_ROOT/data/nuscenes"
NUSC_META=$FULL_PATH/v1.0-trainval
if [ -d "$NUSC_META" ]; then
    echo "OK metadata for map join: $NUSC_META"
    META_ARG=(--nusc-meta "$NUSC_META")
else
    echo "!! $NUSC_META absent; location column will be 'unknown'"
    META_ARG=()
fi
cd "$MMDET_ROOT"

TARGET_DIR=$FULL_PATH/samples/LIDAR_TOP

echo ""
echo "=== Step 3: Stashing the $NFRAMES keyframes the overlay will touch ==="
# Only the frames the tar actually covers are stashed, because only those get
# overwritten. The other 39 are never touched and need no restore.
MANIFEST=$SLURM_TMPDIR/manifest.txt
tar -tf "$(echo "${SETS[0]}" | cut -d'|' -f2)" \
    | sed 's#^\./##' | grep '\.bin$' | sort > "$MANIFEST"
NMAN=$(wc -l < "$MANIFEST")
echo "  manifest: $NMAN files"
[ "$NMAN" -eq "$NFRAMES" ] || { echo "X expected $NFRAMES in manifest, got $NMAN"; exit 1; }

STASH=$SLURM_TMPDIR/clean_lidar_top; mkdir -p "$STASH"
# Per-file existence check here: a val frame absent from the extracted tree
# would otherwise be silently skipped and later "restored" from a stash that
# never held it, leaving the perturbation in place.
while read -r fn; do
    [ -f "$TARGET_DIR/$fn" ] || { echo "X clean file missing: $fn"; exit 1; }
    cp -f "$TARGET_DIR/$fn" "$STASH/$fn"
done < "$MANIFEST"
NSTASH=$(find "$STASH" -name '*.bin' | wc -l)
echo "  stashed $NSTASH clean keyframes"
[ "$NSTASH" -eq "$NFRAMES" ] || { echo "X stashed $NSTASH, expected $NFRAMES"; exit 1; }
df -h "$SLURM_TMPDIR" | tail -1

# ============================================================
# helpers
# ============================================================
run_eval () {   # tag
    local tag=$1
    local wd=$OUT_DIR/eval_$tag
    mkdir -p "$wd"
    echo ""
    echo "------------------------------------------------------------"
    echo "Eval [$tag]"
    echo "------------------------------------------------------------"
    # ann_file must be overridden: FocalFormer3d_L_test.py ships pointing at
    # nuscenes_infos_test.pkl, which has no GT annotations at all.
    local opts=(
        "test_dataloader.dataset.data_root='${FULL_PATH}/'"
        "test_dataloader.dataset.ann_file='${VAL_PKL}'"
        "test_dataloader.num_workers=4"
        "test_evaluator.data_root='${FULL_PATH}/'"
        "test_evaluator.ann_file='${FULL_PATH}/${VAL_PKL}'"
        "test_evaluator.jsonfile_prefix=${wd}/results"
    )
    time python tools/test.py "$CFG" "$CKPT" \
        --work-dir "$wd" --cfg-options "${opts[@]}" 2>&1 | tail -60
}

restore_clean () {
    echo "  restoring clean keyframes ..."
    cp -f "$STASH"/*.bin "$TARGET_DIR"/
}

overlay () {    # unpacked_dir
    local src=$1 n
    n=$(find "$src" -name '*.bin' | wc -l)
    [ "$n" -eq "$NFRAMES" ] || { echo "X unpacked $n .bin files, expected $NFRAMES"; exit 1; }
    cp -f "$src"/*.bin "$TARGET_DIR"/
    echo "  overlaid $n frames ($((6019 - n)) stay clean)"
}

metric_of () {  # tag -> "mAP: x NDS: y"
    local lg
    lg=$(find "$OUT_DIR/eval_$1" -name '*.log' -type f 2>/dev/null | sort | tail -1)
    [ -n "$lg" ] || { echo "<no log>"; return; }
    grep -oE 'NuScenes/(NDS|mAP): [0-9.]+' "$lg" 2>/dev/null | tail -2 | tr '\n' ' '
}

# ============================================================
# Step 4 - CLEAN baseline on the pristine tree
# ============================================================
echo ""
echo "=== Step 4: Clean baseline (pristine tree) ==="
run_eval clean

# ============================================================
# Step 5 - The adversarial set
# ============================================================
for entry in "${SETS[@]}"; do
    NAME=$(echo "$entry"  | cut -d'|' -f1)
    TAR=$(echo "$entry"   | cut -d'|' -f2)
    MODEL=$(echo "$entry" | cut -d'|' -f3)
    ATK=$(echo "$entry"   | cut -d'|' -f4)

    echo ""
    echo "============================================================"
    echo "SET $NAME   model=$MODEL  attack=$ATK"
    echo "  tar: $(basename "$TAR")  ($(date -r "$TAR" +%Y-%m-%d))"
    echo "============================================================"

    restore_clean
    ADV=$SLURM_TMPDIR/adv_$NAME; mkdir -p "$ADV"
    echo "  unpacking ..."
    time tar -xf "$TAR" -C "$ADV"
    overlay "$ADV"
    rm -rf "$ADV"

    run_eval "adv_$NAME"

    echo ""
    echo "--- build_tables: $MODEL / $ATK ---"
    CLEAN_JSON=$OUT_DIR/eval_clean/results/pred_instances_3d/results_nusc.json
    ADV_JSON=$OUT_DIR/eval_adv_$NAME/results/pred_instances_3d/results_nusc.json
    if [ -f "$CLEAN_JSON" ] && [ -f "$ADV_JSON" ]; then
        time python "$TABLE_DIR/build_tables.py" \
            --val-pkl "$FULL_PATH/$VAL_PKL" \
            --clean "$CLEAN_JSON" --adv "$ADV_JSON" \
            --model "$MODEL" --attack "$ATK" "${META_ARG[@]}" \
            --out "$TABLE_DIR/tables_${MODEL}_${NAME}.pkl"
    else
        echo "X missing results json; skipping tables for $NAME"
        [ -f "$CLEAN_JSON" ] || echo "    absent: $CLEAN_JSON"
        [ -f "$ADV_JSON" ]   || echo "    absent: $ADV_JSON"
    fi
done

# ============================================================
# Step 6 - Detailed cross-set breakdown
# ============================================================
# Tables are listed explicitly rather than globbed: metric_scripts also holds
# the Aug-8 prebaked/stock tables, and sweeping those in would silently mix
# three different perturbations into one comparison.
echo ""
echo "=== Step 6: Detailed breakdown across all four sets ==="
BD=(
  "$TABLE_DIR/tables_focalformer-l_lidar.pkl"
  "$TABLE_DIR/tables_bevfusion-l_lidar.pkl"
  "$TABLE_DIR/tables_bevfusion-l_lidar_lossign0.pkl"
  "$TABLE_DIR/tables_bevfusion-lc_lidar-cam.pkl"
)
PRESENT=()
for t in "${BD[@]}"; do
    [ -f "$t" ] && PRESENT+=("$t") || echo "  (absent, skipping: $(basename "$t"))"
done
if [ "${#PRESENT[@]}" -gt 0 ]; then
    python "$TABLE_DIR/adv_breakdown.py" "${PRESENT[@]}" 2>&1 | tee "$OUT_DIR/breakdown.txt"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "SUMMARY -- nuScenes val, 6019 samples scored"
echo "============================================================"
printf '%-24s %s\n' "clean focalformer-l" "$(metric_of clean)"
for entry in "${SETS[@]}"; do
    NAME=$(echo "$entry" | cut -d'|' -f1)
    printf '%-24s %s\n' "adv $NAME" "$(metric_of "adv_$NAME")"
done
echo ""
echo "Reference for the clean row: mAP 0.664 / NDS 0.709."
echo "The adversarial row is over a split where 39 of 6019 frames are pristine."
echo ""
echo "tables in $TABLE_DIR:"
ls -la "$TABLE_DIR"/tables_*.pkl 2>/dev/null | sed 's/^/  /'
echo ""
echo "Done."
