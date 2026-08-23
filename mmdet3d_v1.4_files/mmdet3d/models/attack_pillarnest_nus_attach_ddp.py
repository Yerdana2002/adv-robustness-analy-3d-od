# attack_pillarnest_nus_attach_ddp.py
"""Distributed launcher for the attachment attack. Shards by batch_idx % world_size."""

import argparse
import datetime
import os

import torch
import torch.distributed as dist

from attack_pillarnest_nus_attach import run_attachment_attack


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
    parser = argparse.ArgumentParser(description="Distributed PillarNeSt Attachment Attack")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--ann_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_add", type=int, default=1024)
    parser.add_argument("--sub_loss", type=str, default="all", choices=["iou", "score", "all"])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--subsample_fraction", type=float, default=None,
                    help="Use only the first N fraction of the dataset (e.g. 0.2 = first 20%).")
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    args = parser.parse_args()

    rank, world_size, local_rank, is_dist = init_dist()
    device = f"cuda:{local_rank}"

    os.makedirs(args.results, exist_ok=True)
    if is_dist:
        dist.barrier()

    run_attachment_attack(
        cfg_path=args.cfg,
        result_save_path=args.results,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        ann_file=args.ann_file,
        device=device,
        batch_size=args.batch_size,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        num_add=args.num_add,
        sub_loss=args.sub_loss,
        max_batches=args.max_batches,
        rank=rank,
        world_size=world_size,
        skip_existing=args.skip_existing,
        subsample_fraction=args.subsample_fraction,
    )

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()