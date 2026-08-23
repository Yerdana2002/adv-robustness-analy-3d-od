#!/bin/bash
#SBATCH --account=def-instructor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=attack_bevfusion
#SBATCH --tmp=800G
# =============================================================================
# BEVFusion adversarial attack + evaluation, all in ONE job.
#
#   MODALITY=lidar-cam | lidar
#   MODE=quick | full
#
#   sbatch --export=ALL,MODALITY=lidar-cam,MODE=quick attack_bevfusion_nuscenes.sh
#   sbatch --export=ALL,MODALITY=lidar-cam,MODE=full  attack_bevfusion_nuscenes.sh
#
# Flow:
#   1-4  build ops, extract nuScenes, install val pkl, symlink   (as before)
#   5    attack -> writes adversarial .bin into $SLURM_TMPDIR    (fast local)
#   6    validate them
#   7    PERSIST to /project as a tar, BEFORE any evaluation
#   8    overlay onto the extracted tree in $SLURM_TMPDIR
#   9    eval A: stock pipeline
#   10   eval B: pre-baked pipeline
#
# Step 7 comes before 9/10 on purpose: the attack is the expensive,
# irreproducible-without-gradients part, so it reaches persistent storage
# before anything that could hit the wall clock. Evaluation can always be rerun
# from the tar.
#
# Steps 8-10 reuse the nuScenes tree already in $SLURM_TMPDIR, so the 380 GB
# tarball is unpacked once for attack AND both evaluations.
#
# TWO EVALUATIONS, deliberately
# -----------------------------
# The attack perturbs the 10-sweep ACCUMULATED cloud (~250k points), and the
# project convention (test_centerpoint_nuscenes_adv.sh:71-84) writes that over
# the keyframe file samples/LIDAR_TOP/<name>.pcd.bin. Evaluating that with the
# stock pipeline does two unintended things:
#   - LoadPointsFromMultiSweeps appends 9 more CLEAN sweeps on top (~470k
#     points), diluting the attack;
#   - loading.py:414 runs points.tensor[:, 4] = 0 unconditionally, relabelling
#     all 250k adversarial points t=0 even though they span ~0.45 s.
# So:
#   eval A (stock)    comparable with this project's earlier CenterPoint /
#                     FocalFormer3D / PillarNeSt adversarial numbers
#   eval B (prebaked) drops LoadPointsFromMultiSweeps, so the model sees
#                     exactly the tensor the attack optimised against
# Clean reference for lidar-cam is NDS 0.7116 / mAP 0.6839.
# =============================================================================

set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit=$?)" >&2' ERR

MODALITY=${MODALITY:-lidar-cam}
MODE=${MODE:-quick}
case "$MODALITY" in lidar-cam|lidar) ;; *) echo "X MODALITY must be lidar-cam or lidar"; exit 1 ;; esac
case "$MODE" in quick|full) ;; *) echo "X MODE must be quick or full"; exit 1 ;; esac

# THIS SCRIPT IS SINGLE-INSTANCE. It mutates state shared across jobs:
# $MMDET_ROOT/data/nuscenes is one symlink pointing into the RUNNING job's
# node-local $SLURM_TMPDIR, and Step 1 rebuilds .so files in the shared tree.
# Two jobs on two nodes therefore corrupt each other. That is not theoretical:
# on 2026-08-10 job 18812271 repointed the symlink at 23:29, and job 18801696
# -- 4h37m into its attack on a different node -- died at 23:40 with
# FileNotFoundError on a path that only ever existed on 18812271's node.
# Chain runs with  sbatch --dependency=afterany:<jobid>  instead.
if [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
    OTHERS=$(squeue -h -u "$USER" -t RUNNING -o '%i %j' 2>/dev/null \
        | awk -v me="${SLURM_JOB_ID:-0}" \
              '$2 ~ /attack_bevfusion/ && $1 != me {printf "%s ", $1}') || OTHERS=""
    if [ -n "$OTHERS" ]; then
        echo "X another attack_bevfusion job is RUNNING: $OTHERS"
        echo "  Concurrent runs clobber data/nuscenes and the shared .so build."
        echo "  Use --dependency=afterany:<jobid>, or ALLOW_CONCURRENT=1 if you"
        echo "  have verified this run touches no shared path."
        exit 1
    fi
fi

BATCH_SIZE=${BATCH_SIZE:-4}
ITERATIONS=${ITERATIONS:-40}
LR=${LR:-0.01}
DIST_WEIGHT=${DIST_WEIGHT:-1.0}
LOSS_SIGN=${LOSS_SIGN:-1.0}
# 0.0, not the historical 0.3. The noise existed to work around the dead
# encoder gradient: with d(adv)/d(points) exactly zero AND no initial noise,
# the total gradient was zero in every coordinate and the attack was a no-op,
# so noise was the only thing that made points move. That motion was Chamfer
# denoising, not an attack. Sweep 18798999, final mean displacement:
#   0.000  ->   8.479 mm   100% attack-driven (dist gradient starts at zero)
#   0.005  ->   8.565 mm   washes out to the same equilibrium
#   0.010  ->  11.843 mm   ~8 mm of surviving noise Chamfer cannot see
#   0.300  -> 372.535 mm   only ~2% attack  (reproduced production: the
#                          measured ratio 0.0062 matched the 0.0063 predicted
#                          from job 18755486, and 372.5 mm matched the 387.2 mm
#                          measured over all 6019 samples in job 18690528)
# Above 0.005 the leftover noise lands near OTHER original points, so
# unidirectional Chamfer scores it as free and the loss gives no sign of the
# contamination -- only displacement against the original cloud reveals it.
INIT_NOISE_STD=${INIT_NOISE_STD:-0.0}
# EPS_BALL > 0 swaps the soft Chamfer penalty for PGD inside a per-point L2
# ball of that radius, in metres. Chamfer then becomes reporting-only.
# Why it exists: under the penalty the budget is not a parameter. The
# trajectory settles where ||grad chamfer|| == ||grad adv||, which at beta=1 is
# ~8.5 mm, and there the two gradients are at near parity (0.66) and cancel --
# job 18801696 measured the objective's direction as a coin flip over 429
# batches (p=0.385). AI@x needs x to be an input, not an output. A projection
# also makes the gradient's SCALE irrelevant (the step is normalised), which is
# what would let extraction move to normalize='global' and recover the
# cross-cell saliency that 'channel' erases to buy its norm of 180.
# beta is NOT touched by any of this; with EPS_BALL set there is no beta.
EPS_BALL=${EPS_BALL:-0.0}
EPS_STEP=${EPS_STEP:-0.0}   # 0 => 2.5*eps/iterations, the standard PGD step
# must match the NORMALIZE used at extraction -- it selects the gradient dir
NORMALIZE=${NORMALIZE:-channel}

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11
source ~/centerpoint/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export BLIS_JC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export CUDA_HOME=$CUDA_PATH

PROJECT_ROOT=/home/yerdana/links/projects/def-instructor/yerdana
MMDET_ROOT=$PROJECT_ROOT/mmdetection3d
export PYTHONPATH="$MMDET_ROOT:${PYTHONPATH:-}"

ATTACK_CFG=$MMDET_ROOT/projects/configs/bevfusion/bevfusion_${MODALITY}_grad_extract.py
STOCK_CFG=$MMDET_ROOT/projects/BEVFusion/configs/bevfusion_${MODALITY}_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
PREBAKED_CFG=$MMDET_ROOT/projects/configs/bevfusion/bevfusion_${MODALITY}_adv_eval.py

if [ "$MODALITY" = "lidar-cam" ]; then
    CHECKPOINT_PATH=$PROJECT_ROOT/checkpoint/bevfusion_lidar-cam_mmcv_spconv.pth
else
    CHECKPOINT_PATH=$PROJECT_ROOT/checkpoint/bevfusion_lidar_mmcv_spconv.pth
fi

# Gradients live on /scratch since job 18871825: /project (def-instructor)
# hit 950 GB of 1000 GB and these two sets were 399 GB of it. Override
# with GRAD_ROOT=... if they are staged somewhere else.
GRAD_ROOT=${GRAD_ROOT:-/home/yerdana/links/scratch/yerdana/gradients}
GRAD_DIR=$GRAD_ROOT/gradients_bevfusion_${MODALITY}_${NORMALIZE}
NUSCENES_FULL_TAR=$PROJECT_ROOT/data/nuscenes_bevformer.tar.zst
VAL_PKL_TAR=$PROJECT_ROOT/data/temporal_pkls/nuscenes_infos_val_bevfusion.tar
VAL_PKL=nuscenes_infos_val_bevfusion.pkl

WORK_DIR=$PROJECT_ROOT/work_dirs/attack_bevfusion_${MODALITY}_${SLURM_JOB_ID}
mkdir -p "$WORK_DIR"

# Adversarial clouds are written to LOCAL disk, not /project: ~6019 files x
# ~6 MB of small writes is far cheaper on /localscratch, and Step 7 persists
# them as a single tar afterwards.
ADV_DIR=$SLURM_TMPDIR/adv_points
mkdir -p "$ADV_DIR"
# The tar name must encode LOSS_SIGN. Without this the Chamfer-only ablation
# (LOSS_SIGN=0) writes to adv_points_bevfusion_lidar_channel.tar -- the exact
# path holding the real 17 h attack output -- and Step 7 would silently destroy
# it. MODALITY and NORMALIZE alone do not distinguish the two runs.
RUN_TAG=""
[ "$LOSS_SIGN" != "1.0" ] && RUN_TAG="${RUN_TAG}_lossign${LOSS_SIGN}"
# Every tar currently on disk was produced at init_noise_std=0.3, so anything
# else must get its own name. Without this the sigma=0 re-run would overwrite
# the historical output it is meant to be compared against.
[ "$INIT_NOISE_STD" != "0.3" ] && RUN_TAG="${RUN_TAG}_noise${INIT_NOISE_STD}"
# A PGD run and a soft-penalty run are different experiments, not different
# settings of one, so they must never share a filename.
[ "$EPS_BALL" != "0.0" ] && RUN_TAG="${RUN_TAG}_eps${EPS_BALL}"
# /project is a 1000 GB quota sitting at ~950 GB, so it holds ONE more 28.5 GB
# tar, not two. PERSIST_DIR lets a concurrent run park its output on /scratch
# (19 TB free) instead of forcing a result to be deleted to make room.
# /scratch is purged on Compute Canada, so anything kept long-term has to be
# moved back to /project once space is reclaimed.
PERSIST_DIR=${PERSIST_DIR:-$PROJECT_ROOT/data}
mkdir -p "$PERSIST_DIR"
PERSIST_TAR=$PERSIST_DIR/adv_points_bevfusion_${MODALITY}_${NORMALIZE}${RUN_TAG}.tar
[ "$MODE" = "quick" ] && PERSIST_TAR=$WORK_DIR/adv_points_quick.tar
if [ -e "$PERSIST_TAR" ] && [ "$MODE" = "full" ]; then
    echo "X $PERSIST_TAR already exists; refusing to overwrite a completed run."
    echo "  Move or delete it first if this rerun is intended."
    exit 1
fi

# Step 7 persists ~28.5 GB, but it runs 15 h in. Ordering it before the evals
# protects the attack from the WALL CLOCK; it does nothing about the
# filesystem, and a quota failure there destroys the same 15 h the ordering
# exists to save. So check up front, when the run is still free to abort.
# Advisory only: another job can consume the space while this one attacks, and
# df on Lustre reports the project quota rather than the device.
if [ "$MODE" = "full" ]; then
    AVAIL_KB=$(df -Pk "$PERSIST_DIR" 2>/dev/null | tail -1 | tr -s ' ' | cut -d' ' -f4)
    NEED_KB=$((34 * 1024 * 1024))
    if [ -n "${AVAIL_KB:-}" ] && [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
        echo "X only $((AVAIL_KB / 1024 / 1024)) GB free on $PERSIST_DIR;"
        echo "  Step 7 needs ~28.5 GB for the tar plus room for eval JSONs."
        echo "  Either free space or re-submit with PERSIST_DIR=/scratch/$USER/..."
        exit 1
    fi
fi

for f in "$ATTACK_CFG" "$STOCK_CFG" "$PREBAKED_CFG" "$CHECKPOINT_PATH" "$VAL_PKL_TAR"; do
    [ -f "$f" ] || { echo "X missing: $f"; exit 1; }
done

NGRAD=$(find "$GRAD_DIR" -name '*_grad.pt' 2>/dev/null | wc -l)
echo "============================================================"
echo "BEVFusion adversarial attack + evaluation"
echo "  modality    : $MODALITY"
echo "  mode        : $MODE"
echo "  normalize   : $NORMALIZE   (expected per-sample L2: channel=180, global=1)"
echo "  gradients   : $GRAD_DIR  ($NGRAD files)"
echo "  adv (local) : $ADV_DIR"
echo "  persisted   : $PERSIST_TAR"
[ -n "${AVAIL_KB:-}" ] && echo "  free there  : $((AVAIL_KB / 1024 / 1024)) GB   (Step 7 needs ~28.5 GB)"
echo "  batch/iters : $BATCH_SIZE / $ITERATIONS   lr $LR   beta $DIST_WEIGHT   sign $LOSS_SIGN"
echo "  init noise  : $INIT_NOISE_STD   (0.0 => perturbation is 100% attack-driven)"
echo "============================================================"
[ "$NGRAD" -gt 0 ] || { echo "X no gradients in $GRAD_DIR"; exit 1; }
if [ "$MODE" = "full" ] && [ "$NGRAD" -ne 6019 ]; then
    echo "X gradient set incomplete ($NGRAD/6019); a partial set would silently"
    echo "  skip samples and bias the metrics."
    exit 1
fi

# ============================================================
# Step 1 - ops + pre-flight
# ============================================================
echo ""
echo "=== Step 1: BEVFusion ops ==="
cd "$MMDET_ROOT"
export FORCE_CUDA=1
NEED_BUILD=0
for SO in projects/BEVFusion/bevfusion/ops/bev_pool/bev_pool_ext*.so \
          projects/BEVFusion/bevfusion/ops/voxel/voxel_layer*.so; do
    if [ ! -f "$SO" ] || ! cuobjdump "$SO" 2>/dev/null | grep -q 'arch = sm_90'; then
        NEED_BUILD=1
    fi
done
if [ "$NEED_BUILD" -eq 1 ]; then
    echo "  rebuilding"
    rm -rf build/temp.linux-x86_64-cpython-311/projects/BEVFusion \
           build/lib.linux-x86_64-cpython-311/projects/BEVFusion
    rm -f projects/BEVFusion/bevfusion/ops/bev_pool/*.so \
          projects/BEVFusion/bevfusion/ops/voxel/*.so
    python projects/BEVFusion/setup.py build_ext --inplace 2>&1 | tail -3
else
    echo "  existing ops already carry sm_90 -- skipping rebuild"
fi
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true
for SO in projects/BEVFusion/bevfusion/ops/bev_pool/*.so \
          projects/BEVFusion/bevfusion/ops/voxel/*.so; do
    ARCHS=$(cuobjdump "$SO" 2>/dev/null | grep -E '^arch =' | sort -u | tr '\n' ' ')
    echo "  $(basename "$SO"): ${ARCHS:-<none>}"
    case "$ARCHS" in *sm_90*) ;; *) echo "X no sm_90"; exit 1 ;; esac
done

# The attack needs the CUSTOM differentiable mmcv voxelization -- a different
# extension from the BEVFusion ops above.
python -c "
from mmcv import _ext
assert hasattr(_ext, 'hard_voxelize_forward_v2'), 'custom mmcv build missing'
print('OK mmcv._ext  :', _ext.__file__)
from chamferdist import ChamferDistance  # noqa
print('OK chamferdist')
"

# Construct-only pre-flight, before the 380 GB extraction: an earlier attempt
# burned 17 minutes unpacking only to die 44 s later on model construction.
echo ""
echo "  --- pre-flight: model + eval configs ---"
python - "$ATTACK_CFG" "$STOCK_CFG" "$PREBAKED_CFG" <<'PYPRE'
import sys
from copy import deepcopy
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope('mmdet3d')
from mmdet3d.registry import MODELS

attack_cfg, stock_cfg, prebaked_cfg = sys.argv[1:4]
cfg = Config.fromfile(attack_cfg)
model = MODELS.build(deepcopy(cfg.model))
print(f'  OK model constructs: {type(model).__name__}')
# BEVFusion.__init__ pops voxelize_cfg in place; assert nothing mutated cfg.
assert 'voxelize_cfg' in cfg.model.data_preprocessor, \
    'the build mutated cfg -- a deepcopy is missing'
print('  OK cfg survived the build')

for tag, p in (('stock', stock_cfg), ('prebaked', prebaked_cfg)):
    c = Config.fromfile(p)
    names = [t['type'] for t in c.test_dataloader.dataset.pipeline]
    ms = 'LoadPointsFromMultiSweeps' in names
    print(f'  OK {tag:9s} pipeline, multi-sweep={ms}')
    if tag == 'prebaked':
        assert not ms, 'prebaked config must NOT re-accumulate sweeps'
    else:
        assert ms, 'stock config should retain multi-sweep'

# --- equivalence: the attack's encoder MUST match BEVFusion's own path -------
# This is the check whose absence let a coordinate-order bug (mmcv returns
# (z,y,x), sparse_shape is (x,y,z)) reach a 5-hour production run. It crashed
# on ~18% of scenes and, far worse, silently encoded scrambled geometry on the
# other ~82%. Comparing against BEVFusion.voxelize + pts_middle_encoder on a
# known-hard real sample catches any such divergence in seconds.
# Weights are irrelevant here -- both paths share them -- so no checkpoint load.
import os
import torch
sample_p = ('/home/yerdana/links/projects/def-instructor/yerdana/'
            'data/diag_samples/failing_sample.pt')
if not torch.cuda.is_available():
    print('  -- no GPU, skipping encoder equivalence check')
elif not os.path.exists(sample_p):
    print(f'  !! {sample_p} missing -- encoder equivalence NOT checked')
else:
    sys.path.insert(0, 'projects/mmdet3d_plugin/models')
    from attack_bevfusion_nus import BEVFusionLidarEncoder
    dev = 'cuda:0'
    m = model.to(dev).eval()
    pts = torch.load(sample_p, map_location=dev).float()
    v = cfg.model.data_preprocessor.voxelize_cfg
    mv = v.max_voxels[1] if isinstance(v.max_voxels, (list, tuple)) \
        else v.max_voxels
    with torch.no_grad():
        f_ref, c_ref, _ = m.voxelize([pts])
        ref = m.pts_middle_encoder(f_ref, c_ref, 1)
        enc = BEVFusionLidarEncoder(
            m.pts_middle_encoder, v.voxel_size, v.point_cloud_range,
            max_num_points=v.max_num_points, max_voxels=mv,
            voxelize_reduce=v.get('voxelize_reduce', True)).to(dev).eval()
        got = enc([pts])
    d = (got - ref).abs().max().item()
    rel = d / max(ref.abs().max().item(), 1e-12)
    print(f'  OK encoder equivalence: max|diff|={d:.3e} rel={rel:.3e}')
    assert rel < 1e-3, (
        f'attack encoder disagrees with BEVFusion.extract_pts_feat '
        f'(rel={rel:.3e}) -- check the coors axis order')
PYPRE

# ============================================================
# Step 2 - Extract full nuScenes (ONCE, reused by attack + both evals)
# ============================================================
echo ""
echo "=== Step 2: Extracting full nuScenes (zstd) ==="
FULL_TMP="$SLURM_TMPDIR/full"
mkdir -p "$FULL_TMP"
df -h "$SLURM_TMPDIR"
time tar -I "zstd -d" -xf "$NUSCENES_FULL_TAR" -C "$FULL_TMP"
FULL_PATH=""
for candidate in nuscenes nuscenes_processed; do
    [ -d "$FULL_TMP/$candidate" ] && FULL_PATH="$FULL_TMP/$candidate" && break
done
[ -z "$FULL_PATH" ] && { echo "X nuScenes dir not found"; exit 1; }
echo "OK FULL_PATH=$FULL_PATH"

# ============================================================
# Step 3 / 4 - val pkl + symlink
# ============================================================
echo ""
echo "=== Step 3: Installing val pkl ==="
tar xf "$VAL_PKL_TAR" -C "$FULL_PATH"
ls -lh "$FULL_PATH/$VAL_PKL"
echo ""
echo "=== Step 4: Data symlink ==="
mkdir -p "$MMDET_ROOT/data"
ln -sfn "$FULL_PATH" "$MMDET_ROOT/data/nuscenes"
echo "OK data/nuscenes -> $FULL_PATH"

# Keep a pristine copy of the keyframes the overlay will overwrite, so eval A
# and eval B both start from a known state and a rerun is possible in-job.
KEYFRAME_BACKUP=$SLURM_TMPDIR/keyframe_backup
mkdir -p "$KEYFRAME_BACKUP"

# ============================================================
# Step 5 - Attack
# ============================================================
echo ""
echo "=== Step 5: Attack ==="
ARGS=(
    --cfg "$ATTACK_CFG"
    --grads "$GRAD_DIR"
    --results "$ADV_DIR"
    --checkpoint "$CHECKPOINT_PATH"
    # Explicit, so the attack resolves its data through this job's own
    # $SLURM_TMPDIR rather than through the shared data/nuscenes symlink that
    # a concurrent job can repoint mid-run. The evals already override
    # data_root the same way; this closes the last consumer of the symlink.
    --data_root "$FULL_PATH/"
    --batch_size "$BATCH_SIZE"
    --iterations "$ITERATIONS"
    --lr "$LR"
    --dist_weight "$DIST_WEIGHT"
    --loss_sign "$LOSS_SIGN"
    --init_noise_std "$INIT_NOISE_STD"
    --eps_ball "$EPS_BALL"
    --eps_step "$EPS_STEP"
    --max_voxels_idx 1
)
# 8, not 3. The N=0 spconv failure first appears at batch 3, so a 3-batch check
# (batches 0-2) structurally could not see it and the bug reached a 5-hour full
# run. 8 covers batches 3 and 4, which both failed in job 17504460, so the
# per-sample fallback path is exercised before committing the GPU.
# --diag_grad_split is quick-mode only: it reports ||d(adv)/d(xyz)|| against
# ||d(beta*dist)/d(xyz)||, which is the whole point of this run. It costs an
# extra encoder backward at the 4 logged iterations, so a full run leaves it
# off.
[ "$MODE" = "quick" ] && ARGS+=(--max_batches 8 --diag_grad_split)
time python projects/mmdet3d_plugin/models/attack_bevfusion_nus.py "${ARGS[@]}"

# ============================================================
# Step 6 - Validate adversarial clouds
# ============================================================
echo ""
echo "=== Step 6: Validating adversarial point clouds ==="
python - "$ADV_DIR" "$MODE" <<'PYVAL'
import os, sys
import numpy as np
res, mode = sys.argv[1], sys.argv[2]
files = sorted(f for f in os.listdir(res) if f.endswith('.bin'))
print(f'  adversarial files : {len(files)}')
assert files, 'no adversarial point clouds produced'
if mode == 'full' and len(files) != 6019:
    print(f'  !! expected 6019, got {len(files)}')
tot = sum(os.path.getsize(os.path.join(res, f)) for f in files)
print(f'  total size        : {tot/1e9:.2f} GB ({tot/len(files)/1e6:.2f} MB/file)')
bad = 0
npts, tmax = [], []
for f in files[:20]:
    a = np.fromfile(os.path.join(res, f), dtype=np.float32)
    assert a.size % 5 == 0, f'{f}: {a.size} floats, not a multiple of 5'
    p = a.reshape(-1, 5)
    npts.append(p.shape[0])
    tmax.append(float(p[:, 4].max()))
    if not np.isfinite(p).all():
        bad += 1
        print(f'  !! {f} has non-finite values')
print(f'  points/cloud      : {min(npts)} .. {max(npts)}  (10 accumulated sweeps)')
print(f'  max time channel  : {max(tmax):.3f} s  '
      '(non-zero => per-sweep offsets survived the attack)')
assert bad == 0, 'NaN/Inf in adversarial clouds'
assert max(tmax) > 0, 'time channel is all zero -- sweep structure was lost'

# The file count alone cannot distinguish an attacked cloud from a clean one,
# so a run where the encoder died on N samples looks identical to a clean run
# here. Surface the manifest the attack writes.
import json
man = os.path.join(res, 'attack_failures.json')
if os.path.exists(man):
    with open(man) as fh:
        m = json.load(fh)
    n = m.get('n_failures', 0)
    if n:
        frac = 100.0 * n / max(len(files), 1)
        print(f'  !! NOT FULLY ATTACKED : {n} sample(s) ({frac:.1f}% of set)')
        by = {}
        for x in m.get('failures', []):
            by[x.get('reason', '?')] = by.get(x.get('reason', '?'), 0) + 1
        for k, v in sorted(by.items()):
            print(f'       {k}: {v}')
        print('     metrics below are a MIX of attacked and un-attacked frames')
    else:
        print('  every sample attacked for the full iteration budget')
else:
    print('  !! no attack_failures.json -- attack script predates the manifest')
print('  OK')
PYVAL

# ============================================================
# Step 7 - PERSIST before any evaluation
# ============================================================
echo ""
echo "=== Step 7: Persisting adversarial clouds to /project ==="
echo "  (done before evaluation so a wall-clock kill cannot lose the attack)"
# -C rather than a shell glob: 6019 filenames on the command line is ~540 KB of
# argv, close enough to ARG_MAX to be worth not relying on.
time tar -cf "$SLURM_TMPDIR/adv.tar" -C "$ADV_DIR" .
mv -f "$SLURM_TMPDIR/adv.tar" "$PERSIST_TAR"
ls -lh "$PERSIST_TAR"
echo "  tar entries: $(tar -tf "$PERSIST_TAR" | wc -l)"
df -h "$PROJECT_ROOT" | tail -1
cd "$MMDET_ROOT"

run_eval () {
    local tag=$1 cfg=$2
    local wd=$WORK_DIR/eval_$tag
    mkdir -p "$wd"
    echo ""
    echo "============================================================"
    echo "Eval [$tag]  config: $(basename "$cfg")"
    echo "============================================================"
    local opts=(
        "test_dataloader.dataset.data_root='${FULL_PATH}/'"
        "test_dataloader.dataset.ann_file='${VAL_PKL}'"
        "test_dataloader.num_workers=4"
        "test_evaluator.data_root='${FULL_PATH}/'"
        "test_evaluator.ann_file='${FULL_PATH}/${VAL_PKL}'"
        "test_evaluator.jsonfile_prefix=${wd}/results"
    )
    # No dataset.indices subsetting, even for a quick run. NuScenesMetric
    # asserts "Samples in split doesn't match samples in predictions" -- it
    # scores against the whole 6019-sample split and cannot take a subset.
    # Restricting the dataloader killed the quick run at the stock eval (job
    # 17512883) and left the eval wiring untested, which is exactly what a
    # quick run exists to cover. It therefore pays ~20 min per eval; the
    # metrics land near clean because only its handful of frames were
    # overlaid, and that is itself the sanity signal (expect NDS ~0.7116).
    # tail -80, not -40: the nuScenes summary is mAP/mATE/../NDS followed by a
    # 10-row per-class table and mmengine's own summary block. 40 lines cuts
    # into it, and re-running a multi-hour eval to recover a number we already
    # computed is not a trade worth making.
    time python tools/test.py "$cfg" "$CHECKPOINT_PATH" \
        --work-dir "$wd" --cfg-options "${opts[@]}" 2>&1 | tail -80
}

# ============================================================
# Step 7b - CLEAN baseline, measured BEFORE the overlay
# ============================================================
# The tree in $SLURM_TMPDIR is still pristine at this point, so this yields an
# exactly matched denominator: same data, same pipeline, same checkpoint, same
# job. Needed because no clean lidar-only number exists -- 0.7116 / 0.6839 are
# lidar-cam (job 17486442), and quoting a cross-modality baseline would
# misstate the attack delta. ~20 min against a 24 h limit.
if [ "${RUN_CLEAN_EVAL:-1}" = "1" ] && [ "$MODE" = "full" ]; then
    run_eval clean "$STOCK_CFG"
else
    echo ""
    echo "  (clean baseline skipped: RUN_CLEAN_EVAL=${RUN_CLEAN_EVAL:-1} MODE=$MODE)"
fi

# ============================================================
# Step 8 - Overlay onto the extracted tree
# ============================================================
echo ""
echo "=== Step 8: Overlaying adversarial clouds ==="
TARGET_DIR=$FULL_PATH/samples/LIDAR_TOP
REPLACED=0
for adv in "$ADV_DIR"/*.bin; do
    fn=$(basename "$adv")
    if [ -f "$TARGET_DIR/$fn" ]; then
        cp -f "$TARGET_DIR/$fn" "$KEYFRAME_BACKUP/$fn"
        cp -f "$adv" "$TARGET_DIR/$fn"
        REPLACED=$((REPLACED+1))
    else
        echo "  !! no clean counterpart for $fn"
    fi
done
echo "  replaced $REPLACED keyframe files (backed up to $KEYFRAME_BACKUP)"
if [ "$MODE" = "full" ] && [ "$REPLACED" -ne 6019 ]; then
    echo "X expected to replace 6019, replaced $REPLACED"; exit 1
fi

# ============================================================
# Steps 9 / 10 - Evaluate
# ============================================================
# (run_eval is defined above Step 8, so the clean baseline can use it
#  while the extracted tree is still pristine.)

run_eval stock    "$STOCK_CFG"
run_eval prebaked "$PREBAKED_CFG"

# The prebaked config removes LoadPointsFromMultiSweeps for the WHOLE dataset,
# so any frame the overlay did not replace is served as a bare ~34k-point
# keyframe instead of a ~250k-point accumulation. BEVFusion is trained on the
# accumulated input and collapses without it. In a quick run only a handful of
# frames are replaced, so the prebaked number reflects 5987 under-populated
# clean frames, not the attack -- job 17520372 reported NDS 0.2030 this way.
# In a full run every frame is replaced and the number is meaningful.
if [ "$MODE" = "quick" ]; then
    echo ""
    echo "  NOTE: the prebaked eval above is MEANINGLESS in quick mode."
    echo "        It drops multi-sweep loading for all 6019 frames while only"
    echo "        a handful carry accumulated adversarial clouds, so the rest"
    echo "        are evaluated single-sweep. Only the full run's prebaked"
    echo "        number is interpretable."
fi

echo ""
echo "============================================================"
echo "Done: $MODALITY / $MODE"
echo "  adversarial clouds : $PERSIST_TAR"
echo "  eval work dirs     : $WORK_DIR/eval_{stock,prebaked}"
echo ""
if [ "$MODALITY" = "lidar-cam" ]; then
    echo "  published clean (lidar-cam, job 17486442): NDS 0.7116 / mAP 0.6839"
else
    echo "  no published clean for $MODALITY -- compare against the measured clean row"
fi
# The bare "NDS: x" / "mAP: y" lines are printed by the nuScenes devkit to
# STDOUT, so they land in the slurm .out -- not in mmengine's --work-dir .log,
# which only carries the "NuScenes metric/.../NDS: x" form on the Epoch(test)
# line. Grepping '^NDS:' against *.log therefore matches nothing. Pull the
# metric-key form out of each eval's log instead.
for tag in clean stock prebaked; do
    # find|sort|tail, not ls with two globs: ls exits 2 when either glob fails
    # to match, and under set -e + pipefail that aborts the run AFTER all the
    # expensive work is done (job 17520372 died exactly here).
    # mmengine timestamps its log dirs, so lexicographic sort puts newest last.
    #
    # The -d guard is the second half of that lesson. find returns 0 for "no
    # matches" but returns 1 for "directory does not exist", and pipefail
    # propagates that through sort|tail into the assignment, so set -e kills
    # the script. RUN_CLEAN_EVAL=0 leaves no eval_clean dir, which is exactly
    # how job 18699985 died here after a 9-hour run with every result already
    # on disk. || true covers any other find failure for the same reason.
    lg=""
    if [ -d "$WORK_DIR/eval_$tag" ]; then
        lg=$(find "$WORK_DIR/eval_$tag" -name '*.log' -type f 2>/dev/null \
             | sort | tail -1) || true
    fi
    if [ -n "$lg" ]; then
        line=$(grep -oE 'NuScenes/(NDS|mAP): [0-9.]+' "$lg" 2>/dev/null | tail -2 | tr '\n' ' ')
        echo "  ${tag}: ${line:-<no metric line found in $lg>}"
    else
        echo "  ${tag}: <no log found>"
    fi
done
df -h "$PROJECT_ROOT" | tail -1
echo "============================================================"
