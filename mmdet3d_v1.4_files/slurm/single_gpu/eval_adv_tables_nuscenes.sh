#!/bin/bash
#SBATCH --account=def-instructor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=adv_tables
#SBATCH --tmp=800G
# =============================================================================
# nuScenes ONLY. Metrics + AI Score input tables for all three persisted
# adversarial point-cloud sets, in ONE job on ONE extraction.
#
#   sbatch eval_adv_tables_nuscenes.sh
#
# What it produces, per set:
#   - NDS / mAP  (nuScenes devkit, full 6019-sample val split)
#   - results_nusc.json          predictions in nuScenes global frame
#   - tables_<model>_<tag>.pkl   boxes_df / samples_df for the AI Score
#     notebook, now carrying map location, BEV radius, ego distance and
#     lidar-point count per GT box, so degradation can be sliced by geometry
#     and by city rather than only by class.
#
# Why all three in one job
# ------------------------
# The three tars are NOT three attacks. Two of them predate the voxelization
# fix (jobs 18750353/18755486, Aug 10), so d(adv)/d(points) was exactly zero
# when they were produced and their perturbation is init noise sigma=0.3 plus
# Chamfer denoising. The third had loss_sign=0, i.e. no adversarial term at
# all, by construction.
#
# That makes lossign0 a built-in falsification test rather than a spare run: if
# lidar and lidar-cam land on the same metrics as the set that provably had no
# adversarial term, the dead gradient is demonstrated from outputs instead of
# from reading the code. For that comparison to mean anything the three have to
# share a clean baseline, a matcher, a score threshold and a GT filter -- hence
# one job, one extraction, one code path. Splitting them across jobs would
# reintroduce exactly the confound the comparison exists to remove.
#
# The clean baseline is measured on the PRISTINE tree before any overlay, and
# the 6019 original keyframes are stashed so each set starts from clean rather
# than from the previous set's perturbation.
#
# Runtime ~4.5 h of the 12 h limit: ~30 min extract, ~5 min stash, 2 clean
# evals, 3 adversarial evals, 3 table builds.
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
CKPT_L=$PROJECT_ROOT/checkpoint/bevfusion_lidar_mmcv_spconv.pth
CKPT_LC=$PROJECT_ROOT/checkpoint/bevfusion_lidar-cam_mmcv_spconv.pth
CFG_L=$MMDET_ROOT/projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
CFG_LC=$MMDET_ROOT/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py

# name | tar | modality | model tag | attack tag
SETS=(
  "lidar|$PROJECT_ROOT/data/adv_points_bevfusion_lidar_channel.tar|lidar|bevfusion-l|advlidar"
  "lidar-cam|$PROJECT_ROOT/data/adv_points_bevfusion_lidar-cam_channel.tar|lidar-cam|bevfusion-lc|advlidar"
  "lidar_lossign0|$PROJECT_ROOT/data/adv_points_bevfusion_lidar_channel_lossign0.tar|lidar|bevfusion-l|chamferonly"
)

OUT_DIR=$PROJECT_ROOT/work_dirs/adv_tables_${SLURM_JOB_ID}
TABLE_DIR=$PROJECT_ROOT/metric_scripts
mkdir -p "$OUT_DIR"

for f in "$NUSCENES_FULL_TAR" "$VAL_PKL_TAR" "$CKPT_L" "$CKPT_LC" \
         "$CFG_L" "$CFG_LC" "$TABLE_DIR/build_tables.py"; do
    [ -f "$f" ] || { echo "X missing: $f"; exit 1; }
done
for s in "${SETS[@]}"; do
    t=$(echo "$s" | cut -d'|' -f2)
    [ -f "$t" ] || { echo "X missing adv tar: $t"; exit 1; }
done

# Step 2 repoints the SHARED symlink $MMDET_ROOT/data/nuscenes at this job's
# node-local scratch. Job 18801696 died 4h55m in when a second job did exactly
# that underneath it. Refuse to start rather than repeat that.
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
echo "BEVFusion adversarial metrics + AI Score tables (nuScenes)"
echo "  sets    : lidar, lidar-cam, lidar_lossign0"
echo "  out     : $OUT_DIR"
echo "  tables  : $TABLE_DIR"
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

# ============================================================
# Step 3 - Stash the 6019 clean keyframes
# ============================================================
# Every set overwrites the same files. Without a stash, set 2 would be scored
# on set 1's perturbation and the comparison the job exists for would be
# meaningless. Uses the first tar's manifest, then asserts the other two match
# it, so a mismatched split fails loudly instead of silently mixing frames.
echo ""
echo "=== Step 3: Stashing clean keyframes ==="
MANIFEST=$SLURM_TMPDIR/manifest.txt
tar -tf "$(echo "${SETS[0]}" | cut -d'|' -f2)" \
    | sed 's#^\./##' | grep '\.bin$' | sort > "$MANIFEST"
NMAN=$(wc -l < "$MANIFEST")
echo "  manifest: $NMAN files"
[ "$NMAN" -eq 6019 ] || { echo "X expected 6019 in manifest, got $NMAN"; exit 1; }

for s in "${SETS[@]:1}"; do
    t=$(echo "$s" | cut -d'|' -f2)
    tar -tf "$t" | sed 's#^\./##' | grep '\.bin$' | sort > "$SLURM_TMPDIR/m2.txt"
    if ! cmp -s "$MANIFEST" "$SLURM_TMPDIR/m2.txt"; then
        echo "X $(basename "$t") covers a different frame set than set 1;"
        echo "  scoring them against one clean baseline would be invalid."
        exit 1
    fi
done
echo "  all three tars cover the identical 6019 frames"

STASH=$SLURM_TMPDIR/clean_lidar_top; mkdir -p "$STASH"
# Per-file here (unlike restore/overlay) because this one DOES need the
# existence check: a val frame absent from the extracted tree would otherwise
# be silently skipped and later "restored" from a stash that never had it,
# leaving one set's perturbation in place for every subsequent set.
while read -r fn; do
    [ -f "$TARGET_DIR/$fn" ] || { echo "X clean file missing: $fn"; exit 1; }
    cp -f "$TARGET_DIR/$fn" "$STASH/$fn"
done < "$MANIFEST"
NSTASH=$(find "$STASH" -name '*.bin' | wc -l)
echo "  stashed $NSTASH clean keyframes"
[ "$NSTASH" -eq 6019 ] || { echo "X stashed $NSTASH, expected 6019"; exit 1; }
df -h "$SLURM_TMPDIR" | tail -1

# ============================================================
# helpers
# ============================================================
run_eval () {   # tag cfg ckpt
    local tag=$1 cfg=$2 ckpt=$3
    local wd=$OUT_DIR/eval_$tag
    mkdir -p "$wd"
    echo ""
    echo "------------------------------------------------------------"
    echo "Eval [$tag]  cfg=$(basename "$cfg")"
    echo "------------------------------------------------------------"
    # No dataset.indices subsetting: NuScenesMetric asserts predictions cover
    # the full 6019-sample split.
    local opts=(
        "test_dataloader.dataset.data_root='${FULL_PATH}/'"
        "test_dataloader.dataset.ann_file='${VAL_PKL}'"
        "test_dataloader.num_workers=4"
        "test_evaluator.data_root='${FULL_PATH}/'"
        "test_evaluator.ann_file='${FULL_PATH}/${VAL_PKL}'"
        "test_evaluator.jsonfile_prefix=${wd}/results"
    )
    time python tools/test.py "$cfg" "$ckpt" \
        --work-dir "$wd" --cfg-options "${opts[@]}" 2>&1 | tail -60
}

# One cp process for all 6019, not one per file. A per-file loop across 3 sets
# is ~42k process spawns and costs ~10 min of wall clock for nothing; the
# manifest was already verified against every tar in Step 3, so the per-file
# existence check it bought is redundant.
restore_clean () {
    echo "  restoring clean keyframes ..."
    cp -f "$STASH"/*.bin "$TARGET_DIR"/
}

overlay () {    # unpacked_dir
    local src=$1 n
    n=$(find "$src" -name '*.bin' | wc -l)
    [ "$n" -eq 6019 ] || { echo "X unpacked $n .bin files, expected 6019"; exit 1; }
    cp -f "$src"/*.bin "$TARGET_DIR"/
    echo "  overlaid $n frames"
}

metric_of () {  # tag -> "mAP: x NDS: y"
    local lg
    lg=$(find "$OUT_DIR/eval_$1" -name '*.log' -type f 2>/dev/null | sort | tail -1)
    [ -n "$lg" ] || { echo "<no log>"; return; }
    grep -oE 'NuScenes/(NDS|mAP): [0-9.]+' "$lg" 2>/dev/null | tail -2 | tr '\n' ' '
}

# ============================================================
# Step 4 - CLEAN baselines on the pristine tree
# ============================================================
echo ""
echo "=== Step 4: Clean baselines (pristine tree) ==="
run_eval clean_lidar     "$CFG_L"  "$CKPT_L"
run_eval clean_lidar-cam "$CFG_LC" "$CKPT_LC"

# ============================================================
# Step 5 - Each adversarial set
# ============================================================
for entry in "${SETS[@]}"; do
    NAME=$(echo "$entry" | cut -d'|' -f1)
    TAR=$(echo "$entry"  | cut -d'|' -f2)
    MOD=$(echo "$entry"  | cut -d'|' -f3)
    MODEL=$(echo "$entry" | cut -d'|' -f4)
    ATK=$(echo "$entry"  | cut -d'|' -f5)
    if [ "$MOD" = "lidar-cam" ]; then CFG=$CFG_LC; CKPT=$CKPT_LC; CLEAN_TAG=clean_lidar-cam
    else CFG=$CFG_L; CKPT=$CKPT_L; CLEAN_TAG=clean_lidar; fi

    echo ""
    echo "============================================================"
    echo "SET $NAME   modality=$MOD  model=$MODEL  attack=$ATK"
    echo "  tar: $(basename "$TAR")  ($(date -r "$TAR" +%Y-%m-%d))"
    echo "============================================================"

    restore_clean
    ADV=$SLURM_TMPDIR/adv_$NAME; mkdir -p "$ADV"
    echo "  unpacking ..."
    time tar -xf "$TAR" -C "$ADV"
    overlay "$ADV"
    rm -rf "$ADV"

    run_eval "adv_$NAME" "$CFG" "$CKPT"

    echo ""
    echo "--- build_tables: $MODEL / $ATK ---"
    CLEAN_JSON=$OUT_DIR/eval_$CLEAN_TAG/results/pred_instances_3d/results_nusc.json
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
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "SUMMARY -- nuScenes val, 6019 samples"
echo "============================================================"
printf '%-22s %s\n' "clean lidar"       "$(metric_of clean_lidar)"
printf '%-22s %s\n' "clean lidar-cam"   "$(metric_of clean_lidar-cam)"
echo ""
for entry in "${SETS[@]}"; do
    NAME=$(echo "$entry" | cut -d'|' -f1)
    printf '%-22s %s\n' "adv $NAME" "$(metric_of "adv_$NAME")"
done
echo ""
echo "Read: lidar_lossign0 had NO adversarial term by construction. If the"
echo "other two match it, their adversarial term contributed nothing either,"
echo "which is the dead gradient shown from outputs rather than from code."
echo ""
echo "tables written to $TABLE_DIR:"
ls -la "$TABLE_DIR"/tables_*.pkl 2>/dev/null | sed 's/^/  /'
echo ""
echo "Done."
