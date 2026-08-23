# attack_pillarnest_nus_batch_ddp.py
"""
Distributed launcher wrapper for attack_pillarnest_nus_batch.py
Shards batches by batch_idx % world_size.
Run with torchrun --nproc_per_node=4 ...
"""

import argparse
import datetime
import os
import torch
import torch.distributed as dist

from attack_pillarnest_nus_batch import run_batched_adversarial_attack


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, False
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    n_cuda = torch.cuda.device_count()
    if n_cuda <= 0:
        raise RuntimeError("No CUDA devices visible.")
    if local_rank >= n_cuda:
        raise RuntimeError(f"LOCAL_RANK={local_rank} but visible CUDA devices={n_cuda}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=12))
    return rank, world_size, local_rank, True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed PillarNeSt adversarial attack")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--grads", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--ann_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dist_weight", type=float, default=1.0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")

    parser.add_argument(
    "--target_layer",
    type=str,
    default="pts_middle_encoder",
    help="Attack target feature. Use pts_middle_encoder for PillarNeSt middle encoder attack.",
)
    args = parser.parse_args()

    rank, world_size, local_rank, is_dist = init_dist()
    device = f"cuda:{local_rank}"

    os.makedirs(args.results, exist_ok=True)
    if is_dist:
        dist.barrier()

    run_batched_adversarial_attack(
        cfg_path=args.cfg,
        gradient_folder=args.grads,
        result_save_path=args.results,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        ann_file=args.ann_file,
        device=device,
        batch_size=args.batch_size,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        dist_weight=args.dist_weight,
        max_batches=args.max_batches,
        target_layer=args.target_layer,
        rank=rank,
        world_size=world_size,
        skip_existing=args.skip_existing,
    )

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
