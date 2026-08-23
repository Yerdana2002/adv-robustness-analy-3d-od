# SLURM job scripts


### Before reading this guide, if you found it extra wordy then my apologies. It is targeted for beginner users thus it introduces some simplified versions of concepts needed. Operating Systems was a class I thoroughly enjoyed, and I hope that the guide can be beneficial to someone else. 

Working scripts for the four stages of an adversarial robustness run, in single-GPU and
multi-GPU form. Written for **Alliance Canada** (`rorqual`, H100), but the constraints
they encode apply to any shared HPC filesystem.

| stage | single GPU | multi GPU (torchrun / DDP) |
| :--- | :--- | :--- |
| **Clean test** | `single_gpu/eval_focalformer_lc_clean.sh`<br>`single_gpu/test_bevfusion_nuscenes.sh` | `multi_gpu/slurm_waymo_eval.sh` |
| **Gradient extraction** | `single_gpu/extract_grad_focalformer_l_nus.sh`<br>`single_gpu/extract_grad_focalformer_lc_nus.sh`<br>`single_gpu/extract_grad_bevfusion_nuscenes.sh` | none *(must stay batch_size=1; see below)* |
| **Adversarial attack** | `single_gpu/attack_focalformer_l_nuscenes.sh`<br>`single_gpu/attack_focalformer_lc_nuscenes.sh`<br>`single_gpu/attack_bevfusion_nuscenes.sh`<br>*(+ `*_resume.sh`)* | `multi_gpu/train_pillarnest_nus_distributed.sh`<br>`multi_gpu/train_pillarnest_nus_attach.sh`<br>`multi_gpu/slurm_focalformer_waymo15_batch_attack.sh`<br>`multi_gpu/centerpoint_nuscenes_batch_train.sh` |
| **Adversarial test** | `single_gpu/eval_adv_tables_focalformer.sh`<br>`single_gpu/eval_adv_tables_focalformer_lc.sh`<br>`single_gpu/eval_adv_tables_nuscenes.sh` | none |

Gradient extraction has no multi-GPU form on purpose: the hook names each output file from
`data_samples[0]`, so it must run at `batch_size=1`
([GRADIENT_EXTRACTION.md §4.2](../../docs/GRADIENT_EXTRACTION.md#42-one-file-per-frame)).

---

## The filesystem constraint

**This is the part that is not optional, and the part that is easy to get wrong in a way
that takes the cluster down with you. Don't make other users on the cluster or your quota holder MAD xd**

Shared HPC filesystems are quota'd on **inode count**, not only bytes. Our allocations:

```
                            Description                Space         # of files
                   /home (user yerdana)          17GB/  50GB         316K/ 500K
                /scratch (user yerdana)        2033GB/  20TB          14K/1000K
        /project (project def-instructor)         687GB/1000GB         104K/ 500K
       /nearline (project def-instructor)         772GB/1000GB           7 /5000
```


An unpacked nuScenes contains one file per LiDAR sweep and per camera image
across the full trainval split. We have never unpacked
it onto shared storage. It is stored as a single compressed tar and only ever expanded on node-local disk.

Exceeding an inode quota does not fail cleanly. Writes start failing across **every** job
using that allocation, including other people's. So on a personal note, please don't rely on /scratch too much, especially when a deadline for paper submission is around the clock. 

### The rule

> **Never untar a dataset onto shared storage. Stage it into `$SLURM_TMPDIR` which is the
> node-local SSD and copy results back as a single tar.**

`$SLURM_TMPDIR` is per-job, node-local, fast, and **purged when the job ends**. The
`mmdetection3d/data/nuscenes` symlink in our tree currently is at
`/localscratch/yerdana.19129161.0/full/nuscenes` for exactly that reason. Request it
explicitly:

```bash
#SBATCH --tmp=800G          # node-local SSD; 1200G for Waymo
```

Extract there:

```bash
FULL_TMP="$SLURM_TMPDIR/full"; mkdir -p "$FULL_TMP"
df -h "$SLURM_TMPDIR" | tail -1
time tar -I "zstd -d" -xf "$NUSCENES_FULL_TAR" -C "$FULL_TMP"
```

Write outputs there too, then **tar once** on the way out:

```bash
LOCAL_GRAD_DIR=$SLURM_TMPDIR/gradients          # 5,980 files land HERE
...
tar -cf "$SCRATCH_OUT/$TAR_NAME" -C "$LOCAL_GRAD_DIR" .   # ONE file leaves
```

Same rule applies for the ~6,000 adversarial
point clouds an attack produces.

### Reading a tar you cannot unpack

To inspect individual files inside a large tar, do **not** unpack it on a login node.
Submit a job that stages it into `$SLURM_TMPDIR`, reads what it needs, and prints to the
log:

```bash
sbatch inspect.sh          # then read inspect-<jobid>.out
```

---

## Anatomy of a job

Every script here follows the same shape:

```bash
#!/bin/bash
#SBATCH --account=def-instructor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --tmp=800G                  # node-local SSD -- see above
#SBATCH --output=%x-%j.out

set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit=$?)" >&2' ERR

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11
source ~/centerpoint/bin/activate
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export CUDA_HOME=$CUDA_PATH

# 1. stage dataset into $SLURM_TMPDIR
# 2. run
# 3. tar results out
```

`module --force purge` rather than `module purge`: the plain form leaves the sticky
`StdEnv` loaded and you end up with a mixed toolchain.

### Threading: `OMP_NUM_THREADS` and friends

Every script here exports a block of thread-count variables:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export BLIS_JC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
export NUMEXPR_NUM_THREADS=1
```

**On rorqual, only two of those have any effect.** The rest are harmless, but knowing which
one to change matters when a job produces an error that may be hard to comprehend for beginner users.

#### First, what is OpenMP?

OpenMP is a standard for running one program across several CPU cores at once. A library
built with it (PyTorch's CPU maths, or a BLAS library) can take a single operation, split
it into chunks, and hand each chunk to a separate thread.

The number of threads it uses is decided at runtime, and the usual way to control it is the
environment variable `OMP_NUM_THREADS`. If you do not set it, OpenMP picks a default by
asking the operating system how many CPU cores exist. That question is the source of the
whole problem described below, because on a shared cluster the number of cores that exist
is not the number of cores you were given.

A **thread** here is just an independent stream of execution inside one process. Threads
are useful when there is real work to divide. When there are more threads than cores, they
take turns instead, and the switching between them costs time that could have been spent
computing.

**BLAS** (Basic Linear Algebra Subprograms) is the standard interface for matrix and vector
maths. It is what numpy and scipy call underneath when you multiply matrices. Several
different libraries implement that same interface, they differ in speed, and most of them
are threaded with OpenMP. Which one you actually get is what the next section is about.

#### What rorqual actually links against

Four pieces, and it helps to know what each one is:

| name | what it is |
| :--- | :--- |
| **`StdEnv/2023`** | Alliance Canada's standard environment. One module that loads a consistent set of compilers and core libraries so everything on the cluster is built against the same toolchain. It is loaded by default when you log in. |
| **`flexiblas/3.3.1`** | FlexiBLAS, a **dispatcher** for BLAS. |
| **`aocl-blas/5.1`** | AMD Optimizing CPU Libraries, BLAS component. This is AMD's build of **BLIS**, a high-performance BLAS implementation. |
| **`aocl-lapack/5.1`** | The LAPACK component of the same suite. LAPACK builds on BLAS and provides higher-level routines such as solvers, factorisations and eigenvalue problems. |

**A dispatcher** means FlexiBLAS does not do any maths itself. It presents the standard BLAS
interface, so numpy and scipy link against it and never need to know what is behind it. At
run time it forwards every call to whichever real BLAS library the `$FLEXIBLAS` environment
variable names. The practical benefit is that you can swap the BLAS implementation without
rebuilding numpy, and the practical consequence is that **the library whose name appears in
`numpy.show_config()` is not the library actually doing the work.**

```console
$ python -c "import numpy; print(numpy.show_config(mode='dicts')['Build Dependencies']['blas'])"
{'name': 'flexiblas', 'version': '3.3.1', ...}

$ echo $FLEXIBLAS
aocl

$ flexiblas list
 BLIS      library = libflexiblas_blis.so
 NETLIB    library = libflexiblas_netlib.so
 AOCL      library = libflexiblas_aocl_mt.so      <-- the default
 IMKL      library = libflexiblas_imkl.so
 OPENBLAS  library = libflexiblas_openblas.so
```

So numpy reports FlexiBLAS, `$FLEXIBLAS` says `aocl`, and AOCL turns out to be AMD's build
of BLIS:

```console
$ ls $EBROOTAOCLMINBLAS/lib | grep blis
libblis-mt.so.5.1.0
```

PyTorch is a separate stack from numpy and does not go through FlexiBLAS at all. It is built
**without MKL**, and its CPU parallelism is plain OpenMP:

```console
$ python -c "import torch; print(torch.__config__.parallel_info())"
ATen/Parallel:
        at::get_num_threads() : 192            # login node, 192 cores
        omp_get_max_threads() : 192
MKL not found                          <-- note
ATen parallel backend: OpenMP          <-- note
```

(**MKL** is Intel's Math Kernel Library, another BLAS implementation. **ATen** is PyTorch's
internal tensor library, the C++ layer that actually runs operations.)

So the chain on this cluster is:

```
numpy / scipy  ->  FlexiBLAS  ->  FLEXIBLAS=aocl  ->  libblis-mt.so   (AMD BLIS)
PyTorch        ->  ATen       ->  OpenMP                              (no MKL)
```

Both ends are threaded with OpenMP, which is why one variable can control both.

#### Which variable does what

| variable | on rorqual | why |
| :--- | :--- | :--- |
| **`OMP_NUM_THREADS`** | **affects your job** | ATen's parallel backend is OpenMP, and BLIS threads through OpenMP too. This one variable controls both stacks. |
| **`BLIS_NUM_THREADS`**, `BLIS_{JC,IC,JR,IR}_NT` | **affects your job** | the active BLAS backend is BLIS. The `_NT` set are BLIS's per-loop controls, only needed if `BLIS_NUM_THREADS` alone is not respected. |
| `MKL_NUM_THREADS` | does not affect your job | PyTorch reports `MKL not found`, nothing in the venv links MKL, and the FlexiBLAS default is not `IMKL`. |
| `OPENBLAS_NUM_THREADS` | does not affect your job by default | the backend is AOCL, not OPENBLAS. It starts to matter only if you set `FLEXIBLAS=openblas`. |
| `NUMEXPR_NUM_THREADS` | does not affect your job | numexpr is not installed in the venv. |
| `VECLIB_MAXIMUM_THREADS` | does not affect your job | that is macOS Accelerate. Nothing on Linux reads it. |

Keeping the ones that do nothing here is still defensible. They cost nothing, and they
travel with the script to clusters configured differently, where `imkl` or OpenBLAS may well
be the backend. Just do not conclude that a threading problem is fixed because you set
`MKL_NUM_THREADS` on rorqual.

#### The failure mode

**OpenMP sizes its thread pool from the machine, not from your allocation.** Measured on a
rorqual GPU node with `--cpus-per-task=8`

```console
cpus-per-task   : 8
nproc           : 8                    <-- respects the affinity mask
cpuset cores    : 0-63                 <-- the cgroup cpuset is the whole socket

########## A: NOTHING SET ##########
  torch.get_num_threads() : 64
  hardware_concurrency    : 64
  omp_get_max_threads     : 64
  MKL present             : False

########## B: OMP_NUM_THREADS=1 ##########
  torch.get_num_threads() : 1
```

##### What is an affinity mask?

A **CPU affinity mask** is the list of CPU cores that a particular process is allowed to run
on. Every process on a Linux machine has one. The kernel's scheduler will only ever place
that process's threads on cores that appear in its mask, and never anywhere else.

A rorqual GPU node has 64 cores. When you ask for `--cpus-per-task=8`, SLURM does not give
you a smaller machine, it gives you the same machine with a narrower mask: it picks 8 of
those 64 cores and writes them into your task's affinity mask.

In other words, the node is a room with 64 chairs. SLURM does not move
you to a smaller room, it stamps your ticket with 8 chair numbers. You may sit in any of
those 8 chairs, and the doorman will not let you sit in any of the other 56, even when they
are empty.

`sched_getaffinity` is the system call that asks the kernel "which cores are on my affinity mask?"
The `nproc` command calls it, which is why `nproc` correctly reports **8**.

The trouble is that `hardware_concurrency()` never asks that question. It asks "how many
cores are in this node?", gets **64**, and OpenMP sizes its thread pool to fill them. You
then have 64 threads trying to get executed in 8 cores. Nothing crashes and no error is printed. The
threads simply take turns, and the time they spend swapping places (context switch time if you have taken Operating Systems) is time not spent
computing.

# One example which highlights how this can become an issue is:

* `nproc` says **8**, because it calls `sched_getaffinity`, and SLURM pinned the task to 8 CPUs.
* The cgroup cpuset says **0-63**, because the container is scoped to the socket. The 8-CPU limit
  is enforced by the CPU **quota**, not by narrowing the cpuset.
* `hardware_concurrency()` says **64**, because libstdc++ reports *online* CPUs and **ignores the
  affinity mask entirely**. ATen and OpenMP size their pools from that.

So with nothing set, this job runs **64 threads against an 8-CPU budget**.
Setting `OMP_NUM_THREADS=1` brings it to 1.

The consequences, in rough order of how often we have seen them:

* **Wall time far worse than expected**, with GPU utilisation low. Threads contend and
  context-switch instead of computing. This is the common case, and it shows itself as "the model
  is slow" rather than as a misconfiguration.
* **Memory growth**, since each OpenMP thread carries its own stack, and BLIS its own packing
  buffers.
* **Thread-creation failures** (`pthread_create`, `Resource temporarily unavailable`) when
  several ranks each build a full-socket pool inside one cgroup.



#### What to set

For these jobs, **`OMP_NUM_THREADS=1` is correct**. The work is GPU-bound;
CPU threads exist to feed the dataloader, and mmengine already runs that in
`num_workers` separate processes. Letting each of those spawn its own OpenMP pool
multiplies threads for no gain.

```bash
export OMP_NUM_THREADS=1        # PyTorch ATen + BLIS
export BLIS_NUM_THREADS=1       # belt and braces on the active BLAS
```

If you have a genuinely CPU-bound stage, scale to the allocation rather than the machine and importantly
never leave it unset:

```bash
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
```

Under `torchrun` with N ranks on one node, that budget is **per rank**, so divide:

```bash
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-1} / GPUS ))
```

To confirm what a job actually sees, print it. `torch.__config__.parallel_info()` reports
the resolved values and the backend:

```bash
python -c "import torch; print(torch.__config__.parallel_info())"
```

### Walltime and the partition tiers

**Walltime** is real elapsed time, such as a clock on the wall would measure it, therefore it is not CPU time. It
is what you request with `--time`, and it is a hard limit: when it runs out SLURM kills the
job wherever it has got to.

A **partition** is a named queue of nodes with its own rules, the most important being the
maximum walltime it accepts. Alliance GPU partitions are tiered by that maximum:

| partition | max walltime |
| :--- | :--- |
| `gpubase_bygpu_b1` | 3 h |
| `gpubase_bygpu_b2` | 12 h |
| `gpubase_bygpu_b3` | 24 h |
| `gpubase_bygpu_b4` | 3 d |
| `gpubase_bygpu_b5` | 7 d |

`gpubase_bynode_b1..b5` mirror the same tiers for whole-node allocations, and
`gpubase_interac` caps at 8 h.

**You do not name a partition.** None of the scripts here set `--partition`, and you should
not either. SLURM reads your `--time` and routes the job to the matching tier by itself. You
can see which one it chose after the fact:

```console
$ sacct -j <jobid> --format=JobID,Partition,Timelimit
19438005     gpubackfi+   00:10:00
19433136     gpubase_b+   00:15:00
```

Note the first one landed in `gpubackfill`. **Backfill** is how a scheduler uses gaps: when a
large job is waiting for nodes to free up, the scheduler will slot in a short job that is
guaranteed to finish before those nodes are needed. Short jobs therefore often start almost
immediately, while long ones queue.

That is the concrete reason **an over-generous `--time` costs you queue position**. It is not
a penalty, it is that a job asking for 7 days can never be backfilled into a 2-hour gap, so
it waits for a full slot. Ask for what you need plus a margin, not for the maximum.

The opposite mistake is worse, because SLURM does not warn you. A FocalFormer attack over
nuScenes val runs about 29 h at roughly 52 batches/h. Job 18882811 requested
`--time=24:00:00` and was killed at exactly the limit:

```console
$ grep -oE "Batch [0-9]+/[0-9]+" attack_focalformer-18882811.out | tail -1
Batch 1239/1505
$ sacct -j 18882811 --format=Elapsed,Timelimit,State
1-00:00:22 1-00:00:00    TIMEOUT
```

Batch 1239 of 1505, with every result still on node-local disk that was then purged. Which
is what the next section is about.

### Some helpful tips against accidentally losing everything after a purge

Two mechanisms, both are in the attack scripts:
Here we are essentially saving what we can 30 minutes before the allocated time ends if by that time our job hasn't naturally finished yet.

```bash
#SBATCH --signal=B:USR1@1800        # SIGUSR1 30 min before the wall

persist () {
    local dest=$1 n
    n=$(find "$RESULT_DIR" -name '*.bin' | wc -l)
    [ "$n" -gt 0 ] || { echo "  nothing to persist"; return 0; }
    tar -cf "${dest}.partial" -C "$RESULT_DIR" . && mv -f "${dest}.partial" "$dest"
}
on_wall () { persist "$PROJECT_ROOT/data/..._partial${SLURM_JOB_ID}.tar"; exit 1; }
trap on_wall USR1

python attack.py "${ARGS[@]}" &      # BACKGROUNDED, then `wait`
wait $!
```

Two details that are not optional:

* **`tar` to `.partial`, then `mv`.** `mv` within a filesystem is atomic, so a tar
  interrupted mid-write never replaces a good (meaning non-corrupted or partial) tar.
* **Background the python and `wait`.** This one needs explaining, see below.

##### Foreground and background, and why it matters here

When you run a command normally, the shell starts it and then **waits**, doing nothing else
until it finishes. That is running in the **foreground**. Putting `&` at the end instead
tells the shell to start the command and carry on immediately, which is running in the
**background**. The shell gets its prompt back (or moves to the next line of the script)
while the command keeps running. `wait $!` then says "now pause until that background job
finishes", where `$!` is the process ID of the most recent background command.

At first glance `python attack.py & wait $!` looks like a pointless detour: start it in the
background, then immediately wait for it, which is what the foreground would have done
anyway. The difference is what happens when a **signal** arrives.

A signal is a message the operating system delivers to a process. SLURM sends `SIGUSR1`
because we asked it to with `--signal=B:USR1@1800`, and `trap on_wall USR1` says "when that
arrives, run `on_wall`".

The catch is that **bash will not run a trap handler while a foreground command is still
running.** It notes the signal and defers the handler until the foreground command returns.
So with a plain foreground `python attack.py`:

```
19:30  SIGUSR1 arrives, 30 min before the wall
19:30  bash notes it, but python is in the foreground, so the handler waits
20:00  SLURM kills the job at the wall
       on_wall never ran, $SLURM_TMPDIR is purged, ~29 h of work is gone
```

Backgrounding python means bash itself is sitting in `wait` rather than inside the command.
`wait` is interruptible, so the signal is handled immediately:

```
19:30  SIGUSR1 arrives
19:30  wait is interrupted, on_wall runs, results are tarred to /project
20:00  SLURM kills the job, but everything worth keeping is already safe
```

The trap exists precisely for the case where the job is still running at the wall, which is
exactly the case a foreground `python` would prevent it from handling.

Resume with `*_resume.sh`, which stages the partial tars back in and passes
`--skip_existing`:

```bash
sbatch --export=ALL,RESUME_JOB=18882811 attack_focalformer_l_resume.sh
```

### In case you want to resume an attack after N batches were processed: 

> `RESUME_JOB` is deliberately **required** (`${RESUME_JOB:?...}`) with no default. An
> earlier version defaulted to a hardcoded job id, which would happily stage a LiDAR-only
> run's partials into a LiDAR+camera resume and produce a corrupt mixed set with no error.

---

## Multi-GPU

Single node, `torchrun`, one task with N GPUs, and **not** `--ntasks-per-node=N`. Let
`torchrun` own the process spawning:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --tmp=1200G

GPUS=2
torchrun --standalone --nproc_per_node="$GPUS" "$ATTACK_SCRIPT" \
  --cfg "$CONFIG_PATH" --grads "$LOCAL_GRADS" --results "$LOCAL_RESULTS" \
  --checkpoint "$CHECKPOINT_PATH" --data_root "$NUSCENES_PATH" \
  --batch_size 8 --iterations 40 --lr 0.01 --dist_weight 1.0 \
  --target_layer "pts_middle_encoder" --skip_existing
```

`--standalone` handles rendezvous on a single node, so no `MASTER_ADDR`/`MASTER_PORT`
plumbing. The DDP entrypoints are the `*_ddp.py` / `*_distributed.py` variants under
`mmdet3d/models/` and `projects/mmdet3d_plugin/models/`.

Three things to keep in mind:

* **`--skip_existing` is what makes DDP restartable.** Each rank writes its own output
  files; on resume, work already on disk is skipped rather than recomputed.
* **`$SLURM_TMPDIR` is per-node, and every rank on the node shares it.** Fine on one node.
  Across nodes each has its own copy, so the dataset must be staged per node and results
  gathered per node.
* **Scale `--cpus-per-task` with GPUs** (we use ~12 per GPU). Dataloader workers are the
  bottleneck in these jobs far more often than the GPU is.

---

## Order of operations

```
1. clean test          confirm the port reproduces the checkpoint's reference numbers
                       -> if this fails, nothing downstream means anything
2. gradient extraction MODE=quick first, then MODE=full     -> gradients_*.tar
3. adversarial attack  consumes the gradient tar            -> adv_points_*.tar
4. adversarial test    clean + adversarial eval, tables     -> tables_*.pkl
```

Step 1 is a gate for the following steps. A mis-wired config still loads, still evaluates, and
still produces plausible numbers but you may lose a lot of time. See
[SETTING_UP_FOCALFORMER3D_MMDET14.md §6](../../docs/SETTING_UP_FOCALFORMER3D_MMDET14.md#6-the-coordinate-convention-lwh-and-yaw).

---

## Before you submit

* Paths are hardcoded to `/home/yerdana/links/projects/def-instructor/yerdana`. Change
  `PROJECT_ROOT` and `--account`.
* Several scripts refuse to start if another job of yours is RUNNING, because Step 2
  repoints the shared `mmdetection3d/data/nuscenes` symlink and job 18801696 died 4h55m in
  when another job moved it underneath. Override with
  `sbatch --export=ALL,ALLOW_CONCURRENT=1 ...` only if you are certain.
* Check `--tmp` against your dataset: 800G for nuScenes, 1200G for Waymo.
* Check `--time` against the tier table above **and** against measured throughput. Not
  against optimism.
* Keep the `OMP_NUM_THREADS` block. Unset, PyTorch sizes its thread pool from the node's
  core count, not your allocation. See
  [Threading](#threading-omp_num_threads-and-friends).

---

## See also

* [GRADIENT_EXTRACTION.md](../../docs/GRADIENT_EXTRACTION.md), what the extraction jobs do
* [VOXELIZATION_MMCV2_USAGE.md](../../docs/VOXELIZATION_MMCV2_USAGE.md), what the attack jobs do
* [SETTING_UP_FOCALFORMER3D_MMDET14.md](../../docs/SETTING_UP_FOCALFORMER3D_MMDET14.md) · [SETTING_UP_PILLARNEST_MMDET14.md](../../docs/SETTING_UP_PILLARNEST_MMDET14.md)
