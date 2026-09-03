#!/usr/bin/env python3
"""Reproduce RCCL CE's synchronous-null-stream crash on imported VMM pointers.

The async-stream control uses RCCL's hipMemcpyBatchAsync CE path.  The
sync-null-stream case makes RCCL fall back to per-peer hipMemcpyAsync, which
queries imported peer VMM pointer metadata through ROCr VMemoryPtrInfo.
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import _coalescing_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("async", "synchronous", "sequence"),
        default="sequence",
        help="sequence runs the passing async control before the crashing synchronous case",
    )
    parser.add_argument(
        "--elements-per-rank",
        type=int,
        default=1 << 20,
        help="BF16 elements contributed by each rank",
    )
    return parser.parse_args()


def log(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def make_zero_cta_group() -> dist.ProcessGroup:
    options = dist.ProcessGroupNCCL.Options()
    options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    options.config.split_share = 0
    return dist.new_group(
        ranks=list(range(dist.get_world_size())),
        backend="nccl",
        pg_options=options,
        group_desc="RCCL_CE_NULL_STREAM_VMM_REPRO",
    )


def run_gather(
    *,
    group: dist.ProcessGroup,
    output: torch.Tensor,
    local_input: torch.Tensor,
    stream: torch.cuda.Stream,
    async_op: bool,
    label: str,
    rank: int,
) -> None:
    log(rank, f"{label}: current stream={stream.cuda_stream:#x}; entering AllGather")
    with torch.cuda.stream(stream):
        with _coalescing_manager(group, async_ops=async_op) as work:
            dist.all_gather_into_tensor(
                output,
                local_input,
                group=group,
                async_op=async_op,
            )
        if async_op:
            work.wait()
    stream.synchronize()
    log(rank, f"{label}: PASS")


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=120),
    )
    group = make_zero_cta_group()

    symm_mem.set_backend("NCCL")
    symm_mem.enable_symm_mem_for_group(group.group_name)
    dist.barrier(group=group)
    pool = symm_mem.get_mem_pool(torch.device("cuda", local_rank))

    total_elements = args.elements_per_rank * dist.get_world_size(group)
    with torch.cuda.use_mem_pool(pool):
        output = torch.empty(total_elements, dtype=torch.bfloat16, device="cuda")
    symmetric_memory = symm_mem.rendezvous(output, group=group.group_name)

    local_input = output.view(dist.get_world_size(group), -1)[rank]
    local_input.fill_(rank + 1)
    torch.cuda.synchronize()
    log(
        rank,
        f"rendezvoused bytes={output.nbytes} base={output.data_ptr():#x} "
        f"local={local_input.data_ptr():#x}",
    )

    if args.mode in ("async", "sequence"):
        run_gather(
            group=group,
            output=output,
            local_input=local_input,
            stream=torch.cuda.Stream(device=local_rank),
            async_op=True,
            label="async-stream",
            rank=rank,
        )

    if args.mode in ("synchronous", "sequence"):
        # Megatron's final force_sync=True gather uses async_ops=False. In this
        # path ProcessGroupNCCL passes RCCL a null stream, so CE falls back to
        # per-peer hipMemcpyAsync and queries imported peer VMM pointers.
        run_gather(
            group=group,
            output=output,
            local_input=local_input,
            stream=torch.cuda.default_stream(local_rank),
            async_op=False,
            label="synchronous-null-stream",
            rank=rank,
        )

    # The current PyTorch symmetric-memory pool has a separate interpreter-
    # teardown ordering issue with ProcessGroupNCCL. Bypass Python finalizers
    # after all ranks finish so it cannot obscure the operation under test.
    del symmetric_memory
    dist.barrier(group=group)
    os._exit(0)


if __name__ == "__main__":
    main()
