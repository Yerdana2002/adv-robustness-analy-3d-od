#!/bin/bash
#SBATCH --account=rrg-zhengliu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=waymo_self_eval
#SBATCH --output=%x-%j.out

set -euo pipefail

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11
source ~/centerpoint/bin/activate

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=$CUDA_PATH

PROJECT_ROOT="$HOME/links/projects/rrg-zhengliu/yerdana"
MMDET_ROOT="$PROJECT_ROOT/mmdetection3d"
export PYTHONPATH="$MMDET_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"

EXPORT_SCRIPT="$PROJECT_ROOT/export_waymo_self_eval_adv_with_point_metrics.py"
RUN_ANALYSIS="$PROJECT_ROOT/run_analysis_waymo.py"
WAYMO_TAR="$HOME/links/projects/rrg-zhengliu/yerdana/waymo_kitti_format_v14_7679710.tar.zst"

WAYMO_GT="$HOME/links/scratch/yerdana/waymo_pkl_backup_7235271/gt.bin"
WAYMO_VAL_15="$HOME/links/scratch/yerdana/waymo_pkl_backup_7235271/waymo_infos_val_15split.pkl"

FF_ADV_1="$HOME/links/projects/rrg-zhengliu/yerdana/tars/focalformer_waymo_adv_neck_ddp_7489967.tar"
FF_ADV_2="$HOME/links/projects/rrg-zhengliu/yerdana/tars/focalformer_waymo_adv_neck_ddp_7524731.tar"
PN_ADV="$HOME/links/projects/rrg-zhengliu/yerdana/tars/pillarnest_waymo_adv_pts_middle_encoder_ddp_7554443.tar"

OUT_ROOT="$HOME/links/projects/rrg-zhengliu/yerdana/waymo_self_eval_${SLURM_JOB_ID}"
LOG_DIR="$OUT_ROOT/logs"
DB_DIR="$OUT_ROOT/db"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$DB_DIR"

LOCAL_DATA="$SLURM_TMPDIR/data"
mkdir -p "$LOCAL_DATA"

echo "============================================================"
echo "Waymo Self-Eval (FF + PN)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

# ---- 1. Extract adversarial tars first (small, fast) ----
echo ""
echo "[1/4] Extracting adversarial point clouds..."
ADV_FF="$SLURM_TMPDIR/adv_ff"
ADV_PN="$SLURM_TMPDIR/adv_pn"
mkdir -p "$ADV_FF" "$ADV_PN"

time tar -xf "$FF_ADV_1" -C "$ADV_FF"
time tar -xf "$FF_ADV_2" -C "$ADV_FF"
time tar -xf "$PN_ADV"   -C "$ADV_PN"

echo "  FF adv bins: $(find "$ADV_FF" -type f -name '*.bin' | wc -l)"
echo "  PN adv bins: $(find "$ADV_PN" -type f -name '*.bin' | wc -l)"



echo "[2/4] Extracting Waymo dataset..."
set +e
time tar -I "zstd -d" -xf "$WAYMO_TAR" -C "$LOCAL_DATA" --no-same-owner --no-same-permissions
TAR_EXIT=$?
set -e

if [ $TAR_EXIT -ne 0 ]; then
    echo "WARNING: tar exited with code $TAR_EXIT (truncated archive). Continuing with partial extraction."
fi




WAYMO_KITTI_PATH="$(find "$LOCAL_DATA" -maxdepth 6 -type d -name kitti_format | head -n 1 || true)"
[ -n "$WAYMO_KITTI_PATH" ] || { echo "ERROR: kitti_format not found"; exit 1; }
echo "Extracted velodyne files: $(find "$WAYMO_KITTI_PATH/training/velodyne" -name '*.bin' | wc -l)"
echo "  kitti_format: $WAYMO_KITTI_PATH"

WAYMO_ROOT="$(dirname "$WAYMO_KITTI_PATH")"
WAYMO_FORMAT_PATH="$WAYMO_ROOT/waymo_format"
mkdir -p "$WAYMO_FORMAT_PATH"

[ -f "$WAYMO_GT" ] && cp -f "$WAYMO_GT" "$WAYMO_FORMAT_PATH/gt.bin" && echo "  ✓ gt.bin"
[ -f "$WAYMO_VAL_15" ] && cp -f "$WAYMO_VAL_15" "$WAYMO_KITTI_PATH/waymo_infos_val.pkl" && echo "  ✓ 1/5 val split"
[ -f "$WAYMO_KITTI_PATH/waymo_infos_val.pkl" ] || { echo "Missing waymo_infos_val.pkl"; exit 1; }
[ -d "$WAYMO_KITTI_PATH/training/velodyne" ] || { echo "Missing velodyne dir"; exit 1; }

df -h "$SLURM_TMPDIR"


# Filter val pkl to only samples with extracted velodyne files
python -c "
import pickle, os
ann = '$WAYMO_KITTI_PATH/waymo_infos_val.pkl'
data = pickle.load(open(ann, 'rb'))
velo_dir = '$WAYMO_KITTI_PATH/training/velodyne'
if isinstance(data, dict) and 'data_list' in data:
    before = len(data['data_list'])
    data['data_list'] = [d for d in data['data_list']
        if os.path.exists(os.path.join(velo_dir,
            (d.get('lidar_points',{}).get('lidar_path','') or '').split('/')[-1]))]
    print(f'Filtered: {before} -> {len(data[\"data_list\"])} samples')
    pickle.dump(data, open(ann, 'wb'))
"


# ---- 3. Run export (2 models in parallel on 2 GPUs) ----
echo ""
echo "[3/4] Running export (FF on GPU0, PN on GPU1)..."

MODEL_IDS=("ff" "pn")
CFGS=(
    "$MMDET_ROOT/projects/configs/focalformer3d/FocalFormer3D_Waymo_L_gradient.py"
    "$MMDET_ROOT/configs/pillarnest/pillarnest_waymo_adv.py"
)
CKPTS=(
    "$PROJECT_ROOT/checkpoint/FocalFormer3d_Waymo_converted.pth"
    "$PROJECT_ROOT/checkpoint/pillarnest_base_waymo_v14.pth"
)
ADV_DIRS=("$ADV_FF" "$ADV_PN")

COMBO_FILE="$OUT_ROOT/combos.txt"
FAIL_FILE="$OUT_ROOT/failures.txt"
: > "$COMBO_FILE"
: > "$FAIL_FILE"

for i in "${!MODEL_IDS[@]}"; do
    MID="${MODEL_IDS[$i]}"
    CFG="${CFGS[$i]}"
    CKPT="${CKPTS[$i]}"
    ADIR="${ADV_DIRS[$i]}"
    OUT_PKL="$OUT_ROOT/${MID}_self_eval.pkl"
    LOG_FILE="$LOG_DIR/${MID}_self_eval.log"
    echo "${MID}|${CFG}|${CKPT}|${ADIR}|${OUT_PKL}|${LOG_FILE}" >> "$COMBO_FILE"
done

run_worker() {
    local worker_id="$1"
    local gpu_id="$2"
    local idx=0

    while IFS='|' read -r MID CFG CKPT ADIR OUT_PKL LOG_FILE; do
        if (( idx % 2 != worker_id )); then
            idx=$((idx + 1)); continue
        fi

        echo "[worker $worker_id] GPU=$gpu_id model=$MID"
        if CUDA_VISIBLE_DEVICES="$gpu_id" python "$EXPORT_SCRIPT" \
            --model-id "${MID}_self" \
            --cfg "$CFG" \
            --ckpt "$CKPT" \
            --clean-data-root "$WAYMO_KITTI_PATH" \
            --clean-ann-file "$WAYMO_KITTI_PATH/waymo_infos_val.pkl" \
            --adv-dir "$ADIR" \
            --out-pkl "$OUT_PKL" \
            --loader train \
            --batch-size 1 \
            --num-workers 4 \
            --device cuda:0 \
            --omit_points_in_pkl \
            --compute-point-metrics \
            --innout-thresh 0.8 \
            --sample-change-thresh 0.1 \
            --obj-change-thresh 0.05 \
            > "$LOG_FILE" 2>&1; then
            echo "[worker $worker_id] OK: model=$MID"
        else
            echo "[worker $worker_id] FAIL: model=$MID log=$LOG_FILE" | tee -a "$FAIL_FILE"
        fi

        idx=$((idx + 1))
    done < "$COMBO_FILE"
}

run_worker 0 0 & PID0=$!
run_worker 1 1 & PID1=$!

set +e
wait "$PID0"; S0=$?
wait "$PID1"; S1=$?
set -e
echo "worker0_exit=$S0 worker1_exit=$S1"

# Check for failures before proceeding
if [ -s "$FAIL_FILE" ]; then
    echo "WARNING: Some workers failed:"
    cat "$FAIL_FILE"
fi

# ---- 4. Build analysis DBs ----
echo ""
echo "[4/4] Building analysis databases..."

for MID in "${MODEL_IDS[@]}"; do
    PKL="$OUT_ROOT/${MID}_self_eval.pkl"
    DB="$DB_DIR/${MID}_waymo_self.db"

    if [ ! -s "$PKL" ]; then
        echo "  SKIP $MID: pkl empty or missing"
        continue
    fi

    echo "  Building DB for $MID..."
    python "$RUN_ANALYSIS" \
        --input-pkl "$PKL" \
        --output-db "$DB" \
        --class-filter waymo
done

# ---- Summary ----
echo ""
echo "============================================================"
echo "✓ Done"
echo "  Output:    $OUT_ROOT"
echo "  Logs:      $LOG_DIR"
echo "  DBs:       $DB_DIR"
echo "  Failures:  $FAIL_FILE"
echo "============================================================"

