#!/usr/bin/env python3
"""Manual multi-GPU correctness test for RCCL-SDMA ReduceScatter."""

from __future__ import annotations

import argparse
import gc
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from primus.backends.megatron.core.distributed import (
    rccl_sdma_param_gather,
    rccl_sdma_reduce_scatter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--subgroup-size", type=int, default=0)
    return parser.parse_args()


def upstream_reference(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    op: dist.ReduceOp,
    group: dist.ProcessGroup,
) -> None:
    world_size = group.size()
    received = torch.empty_like(input_tensor)
    dist.all_to_all_single(received, input_tensor, group=group)
    reduced = received.view(world_size, -1).sum(dim=0, dtype=torch.float32)
    if op == dist.ReduceOp.AVG:
        reduced.mul_(1.0 / world_size)
    output.copy_(reduced)


def main() -> None:
    args = parse_args()
    os.environ["MEGATRON_RCCL_SDMA_RS_STAGING_BYTES"] = str(
        args.staging_bytes
    )
    os.environ.setdefault("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"])

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    global_world_size = dist.get_world_size()
    original_group = dist.group.WORLD
    created_groups = []
    if args.subgroup_size:
        if global_world_size % args.subgroup_size:
            raise ValueError("subgroup size must divide world size")
        for start in range(0, global_world_size, args.subgroup_size):
            group = dist.new_group(
                ranks=list(range(start, start + args.subgroup_size)),
                backend="nccl",
            )
            created_groups.append(group)
            if start <= rank < start + args.subgroup_size:
                original_group = group
    world_size = original_group.size()
    group_rank = original_group.rank()

    for dtype in (torch.bfloat16, torch.float16):
        for op in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
            for async_op in (False, True):
                for output_numel in (1003, 131071):
                    generator = torch.Generator(device=device)
                    generator.manual_seed(1234 + rank)
                    input_tensor = torch.randn(
                        world_size * output_numel,
                        dtype=dtype,
                        device=device,
                        generator=generator,
                    )
                    reference_input = input_tensor.clone()
                    reference = torch.empty(
                        output_numel,
                        dtype=dtype,
                        device=device,
                    )
                    upstream_reference(
                        reference,
                        reference_input,
                        op,
                        original_group,
                    )

                    # Match Megatron: output is this rank's grad_data shard.
                    output = input_tensor.view(
                        world_size,
                        output_numel,
                    )[group_rank]
                    work = rccl_sdma_reduce_scatter.reduce_scatter(
                        output,
                        input_tensor,
                        op,
                        original_group,
                        async_op=async_op,
                    )
                    if async_op:
                        work.wait()
                    else:
                        assert work is None
                    torch.cuda.synchronize(device)
                    torch.testing.assert_close(
                        output,
                        reference,
                        rtol=0,
                        atol=0,
                    )

    sdma_group = rccl_sdma_reduce_scatter.get_sdma_process_group(
        original_group
    )
    dist.barrier(group=sdma_group)
    rccl_sdma_reduce_scatter.reset_runtime_state_for_tests()
    symm_mem._symm_mem_pools.clear()
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier(group=sdma_group)
    dist.destroy_process_group(sdma_group)
    rccl_sdma_param_gather.reset_runtime_state_for_tests()
    for group in created_groups:
        if group is not dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(group)
    dist.destroy_process_group()
    if rank == 0:
        print(
            "RCCL-SDMA ReduceScatter correctness PASS "
            f"world_size={world_size}",
            flush=True,
        )


if __name__ == "__main__":
    main()
