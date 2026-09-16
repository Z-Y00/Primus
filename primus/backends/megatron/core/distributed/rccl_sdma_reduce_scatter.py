###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ReduceScatter through chunked RCCL-SDMA AllToAll and FP32 accumulation."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

DEFAULT_STAGING_BYTES = 512 * 1024 * 1024
DEFAULT_WORKSPACE_DEPTH = 2
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

_SUBGROUPS: dict[tuple[int, ...], dist.ProcessGroup] = {}
_NATIVE_TAIL_GROUPS: dict[tuple[int, ...], dist.ProcessGroup] = {}
_BUCKET_IDS_BY_PTR: dict[int, int] = {}
_POOLS: dict[tuple[str, int], torch.cuda.MemPool] = {}
_EAGER_DIRECT_RECV: dict[
    tuple[str, int, torch.dtype, int],
    tuple[torch.Tensor, object],
] = {}
_WORKSPACES: dict[
    tuple[str, int, torch.dtype, bool],
    "ReduceScatterWorkspacePool",
] = {}


def _cta_options() -> dist.ProcessGroupNCCL.Options:
    options = dist.ProcessGroupNCCL.Options()
    cta_policy = int(os.getenv("MEGATRON_RCCL_SDMA_CTA_POLICY", "2"))
    if cta_policy == 2:
        options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    elif cta_policy == 0:
        options.config.cta_policy = getattr(
            dist.ProcessGroupNCCL,
            "NCCL_CTA_POLICY_DEFAULT",
            0,
        )
    else:
        raise ValueError(
            f"MEGATRON_RCCL_SDMA_CTA_POLICY must be 0 or 2, got {cta_policy}"
        )
    options.config.split_share = 0
    return options


def _require_single_node() -> None:
    world_size = dist.get_world_size()
    local_world_size_value = os.getenv("LOCAL_WORLD_SIZE")
    if local_world_size_value is None:
        raise RuntimeError(
            "RCCL-SDMA ReduceScatter requires LOCAL_WORLD_SIZE to verify "
            "single-node topology"
        )
    local_world_size = int(local_world_size_value)
    if local_world_size != world_size:
        raise RuntimeError(
            "RCCL-SDMA ReduceScatter currently requires the entire job to be "
            f"single-node, got world_size={world_size} "
            f"LOCAL_WORLD_SIZE={local_world_size}"
        )


def get_sdma_process_group(
    original_group: dist.ProcessGroup,
) -> dist.ProcessGroup:
    """Return a zero-CTA communicator with the original group's rank set."""
    _require_single_node()
    ranks = tuple(dist.get_process_group_ranks(original_group))
    if ranks == tuple(range(dist.get_world_size())):
        from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
            get_sdma_process_group as get_full_world_sdma_group,
        )

        return get_full_world_sdma_group(original_group)

    group = _SUBGROUPS.get(ranks)
    if group is None:
        group = dist.new_group(
            ranks=list(ranks),
            backend="nccl",
            pg_options=_cta_options(),
            timeout=timedelta(
                minutes=int(
                    os.getenv("MEGATRON_RCCL_SDMA_TIMEOUT_MINUTES", "10")
                )
            ),
            use_local_synchronization=True,
            group_desc="MEGATRON_RCCL_SDMA_REDUCE_SCATTER",
        )
        _SUBGROUPS[ranks] = group
        if dist.get_rank() == min(ranks):
            print(
                "[RCCL-SDMA:Megatron] created ReduceScatter zero-CTA group "
                f"ranks={list(ranks)} name={group.group_name}",
                flush=True,
            )
    return group


def get_native_tail_process_group(
    original_group: dist.ProcessGroup,
) -> dist.ProcessGroup:
    """Return a default-CTA group isolated from Megatron's coalescing state."""
    ranks = tuple(dist.get_process_group_ranks(original_group))
    group = _NATIVE_TAIL_GROUPS.get(ranks)
    if group is None:
        group = dist.new_group(
            ranks=list(ranks),
            backend="nccl",
            timeout=timedelta(
                minutes=int(
                    os.getenv("MEGATRON_RCCL_SDMA_TIMEOUT_MINUTES", "10")
                )
            ),
            use_local_synchronization=True,
            group_desc="MEGATRON_RCCL_NATIVE_REDUCE_SCATTER_TAIL",
        )
        _NATIVE_TAIL_GROUPS[ranks] = group
    return group


def _get_pool(
    group: dist.ProcessGroup,
    device: torch.device,
) -> torch.cuda.MemPool:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (group.group_name, device_index)
    pool = _POOLS.get(key)
    if pool is None:
        symm_mem.set_backend("NCCL")
        dist.barrier(group=group)
        pool = symm_mem.get_mem_pool(device)
        _POOLS[key] = pool
    return pool


def _staging_numel(
    staging_bytes: int,
    element_size: int,
    world_size: int,
) -> int:
    if staging_bytes <= 0:
        raise ValueError("RCCL-SDMA ReduceScatter staging bytes must be positive")
    numel = staging_bytes // element_size
    numel -= numel % world_size
    if numel == 0:
        raise ValueError(
            "RCCL-SDMA ReduceScatter staging buffer cannot hold one element per rank"
        )
    return numel


def reserve_direct_input_workspace(
    original_group: dist.ProcessGroup,
    device: torch.device,
    dtype: torch.dtype,
    staging_bytes: int,
) -> None:
    """Reserve the direct-input receive buffer before model construction."""
    group = get_sdma_process_group(original_group)
    if os.getenv("MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL", "0") == "1":
        get_native_tail_process_group(original_group)
    staging_numel = _staging_numel(
        staging_bytes,
        torch.empty((), dtype=dtype).element_size(),
        group.size(),
    )
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (group.group_name, device_index, dtype, staging_numel)
    if key in _EAGER_DIRECT_RECV:
        return
    pool = _get_pool(group, device)
    with torch.cuda.use_mem_pool(pool):
        recv = torch.empty(staging_numel, dtype=dtype, device=device)
    symmetric_handle = symm_mem.rendezvous(recv, group=group.group_name)
    _EAGER_DIRECT_RECV[key] = (recv, symmetric_handle)
    if group.rank() == 0:
        print(
            "[RCCL-SDMA:Megatron] eagerly reserved direct-input receive "
            f"bytes={recv.nbytes}",
            flush=True,
        )


@triton.jit
def _fp32_reduce_store_kernel(
    input_ptr,
    output_ptr,
    numel,
    WORLD_SIZE: tl.constexpr,
    AVERAGE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for rank in tl.static_range(0, WORLD_SIZE):
        values = tl.load(
            input_ptr + rank * numel + offsets,
            mask=mask,
            other=0.0,
        )
        accumulator += values.to(tl.float32)
    if AVERAGE:
        accumulator *= 1.0 / WORLD_SIZE
    tl.store(output_ptr + offsets, accumulator, mask=mask)


def _fp32_reduce_store(
    received: torch.Tensor,
    output: torch.Tensor,
    world_size: int,
    average: bool,
) -> None:
    """Accumulate in FP32 registers and store directly to the output dtype."""
    if received.device.type != "cuda":
        reduced = received.view(world_size, -1).sum(
            dim=0,
            dtype=torch.float32,
        )
        if average:
            reduced.mul_(1.0 / world_size)
        output.copy_(reduced)
        return

    block_size = 1024
    _fp32_reduce_store_kernel[(triton.cdiv(output.numel(), block_size),)](
        received,
        output,
        output.numel(),
        WORLD_SIZE=world_size,
        AVERAGE=average,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )


class ReduceScatterWorkspace:
    """Persistent buffers for eager look-ahead pipelining."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        dtype: torch.dtype,
        staging_numel: int,
        direct_input: bool,
    ) -> None:
        self.group = group
        self.device = device
        self.dtype = dtype
        self.world_size = group.size()
        self.staging_numel = staging_numel
        self.chunk_numel = staging_numel // self.world_size
        self.direct_input = direct_input
        num_slots = 1 if direct_input else 2
        self.send = (
            []
            if direct_input
            else [
                torch.empty(staging_numel, dtype=dtype, device=device)
                for _ in range(num_slots)
            ]
        )
        pool = _get_pool(group, device)
        self.recv = []
        self.symmetric_handles = []
        direct_key = (
            group.group_name,
            device.index
            if device.index is not None
            else torch.cuda.current_device(),
            dtype,
            staging_numel,
        )
        eager_direct = (
            _EAGER_DIRECT_RECV.pop(direct_key, None)
            if direct_input
            else None
        )
        if eager_direct is not None:
            recv, symmetric_handle = eager_direct
            self.recv.append(recv)
            self.symmetric_handles.append(symmetric_handle)
        for _ in range(num_slots - len(self.recv)):
            with torch.cuda.use_mem_pool(pool):
                recv = torch.empty(staging_numel, dtype=dtype, device=device)
            self.recv.append(recv)
            self.symmetric_handles.append(
                symm_mem.rendezvous(recv, group=group.group_name)
            )
        self.stream: torch.cuda.Stream | None = None
        self.active_work: ReduceScatterWork | None = None
        self.workspace_bytes = sum(
            tensor.nbytes
            for tensor in self.send + self.recv
        )


class ReduceScatterWorkspacePool:
    """Bounded ring of workspaces shared across gradient buckets."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        dtype: torch.dtype,
        staging_numel: int,
        depth: int,
        direct_input: bool,
    ) -> None:
        self.group = group
        self.device = device
        self.dtype = dtype
        self.staging_numel = staging_numel
        self.depth = depth
        self.direct_input = direct_input
        self.workspaces = [
            ReduceScatterWorkspace(
                group,
                device,
                dtype,
                staging_numel,
                direct_input,
            )
            for _ in range(depth)
        ]
        self.next_index = 0
        self.workspace_bytes = sum(
            workspace.workspace_bytes for workspace in self.workspaces
        )

    def acquire(
        self,
        caller_stream: torch.cuda.Stream | None,
    ) -> ReduceScatterWorkspace:
        """Return the next slot with a stream distinct from the caller."""
        workspace = self.workspaces[self.next_index]
        caller_handle = (
            int(caller_stream.cuda_stream)
            if caller_stream is not None
            else None
        )
        if caller_stream is not None and workspace.stream is None:
            forbidden_handles = {caller_handle}
            forbidden_handles.update(
                int(other.stream.cuda_stream)
                for other in self.workspaces
                if other.stream is not None
            )
            rejected_streams = []
            for _ in range(64):
                candidate = torch.cuda.Stream(device=self.device)
                if int(candidate.cuda_stream) not in forbidden_handles:
                    workspace.stream = candidate
                    break
                rejected_streams.append(candidate)
            else:
                raise RuntimeError(
                    "RCCL-SDMA ReduceScatter could not allocate a distinct "
                    "reduction stream"
                )
        elif (
            caller_stream is not None
            and int(workspace.stream.cuda_stream) == caller_handle
        ):
            raise RuntimeError(
                "RCCL-SDMA ReduceScatter reduction stream aliases the caller"
            )
        self.next_index = (self.next_index + 1) % self.depth
        return workspace


def prepare_workspace(
    original_group: dist.ProcessGroup,
    input_tensor: torch.Tensor,
) -> ReduceScatterWorkspacePool:
    """Allocate or return the shared bounded workspace ring."""
    if input_tensor.dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            "RCCL-SDMA ReduceScatter supports BF16 and FP16 gradients, "
            f"got {input_tensor.dtype}"
        )
    group = get_sdma_process_group(original_group)
    staging_bytes = int(
        os.getenv(
            "MEGATRON_RCCL_SDMA_RS_STAGING_BYTES",
            str(DEFAULT_STAGING_BYTES),
        )
    )
    direct_input = (
        os.getenv("MEGATRON_RCCL_SDMA_RS_DIRECT_INPUT", "0") == "1"
    )
    if not direct_input:
        staging_bytes = min(staging_bytes, input_tensor.nbytes)
    staging_numel = _staging_numel(
        staging_bytes,
        input_tensor.element_size(),
        group.size(),
    )
    depth = int(
        os.getenv(
            "MEGATRON_RCCL_SDMA_RS_WORKSPACE_DEPTH",
            str(DEFAULT_WORKSPACE_DEPTH),
        )
    )
    if depth <= 0:
        raise ValueError(
            "RCCL-SDMA ReduceScatter workspace depth must be positive"
        )
    if direct_input and depth != 1:
        raise ValueError(
            "RCCL-SDMA direct-input ReduceScatter requires workspace depth 1"
        )
    device_index = input_tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (
        group.group_name,
        device_index,
        input_tensor.dtype,
        direct_input,
    )
    workspace_pool = _WORKSPACES.get(key)
    if (
        workspace_pool is None
        or workspace_pool.staging_numel < staging_numel
        or workspace_pool.depth != depth
    ):
        if workspace_pool is not None and any(
            workspace.active_work is not None
            and not workspace.active_work.is_completed()
            for workspace in workspace_pool.workspaces
        ):
            raise RuntimeError(
                "RCCL-SDMA ReduceScatter cannot resize an active workspace ring"
            )
        workspace_pool = ReduceScatterWorkspacePool(
            group,
            input_tensor.device,
            input_tensor.dtype,
            staging_numel,
            depth,
            direct_input,
        )
        _WORKSPACES[key] = workspace_pool
        if (
            group.rank() == 0
            and os.getenv("MEGATRON_RCCL_SDMA_LOG", "0") == "1"
        ):
            print(
                "[RCCL-SDMA:Megatron] initialized ReduceScatter workspace "
                f"dtype={input_tensor.dtype} staging_bytes="
                f"{staging_numel * input_tensor.element_size()} "
                f"depth={depth} "
                f"direct_input={direct_input} "
                f"workspace_bytes={workspace_pool.workspace_bytes}",
                flush=True,
            )
    return workspace_pool


def _use_native_tail(
    workspace_pool: ReduceScatterWorkspacePool,
    input_tensor: torch.Tensor,
) -> bool:
    """Use native ReduceScatter for Megatron's exposed tail buckets."""
    bucket_id = _BUCKET_IDS_BY_PTR.get(input_tensor.data_ptr())
    native_tail_buckets = int(
        os.getenv("MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL_BUCKETS", "2")
    )
    if native_tail_buckets <= 0:
        raise ValueError(
            "MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL_BUCKETS must be positive"
        )
    return (
        os.getenv("MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL", "0") == "1"
        and workspace_pool.direct_input
        and bucket_id is not None
        and bucket_id
        > max(_BUCKET_IDS_BY_PTR.values()) - native_tail_buckets
    )


def register_gradient_bucket(bucket) -> None:
    """Record persistent gradient-buffer identity for tail selection."""
    _BUCKET_IDS_BY_PTR[bucket.grad_data.data_ptr()] = bucket.bucket_id


class ReduceScatterWork:
    """Event-backed handle for an eagerly queued ReduceScatter pipeline."""

    def __init__(
        self,
        workspace: ReduceScatterWorkspace,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: dist.ReduceOp,
        caller_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.workspace = workspace
        self.output_tensor = output_tensor
        self.input_tensor = input_tensor
        self.op = op
        self.completed = False
        self.done_event = None

        if workspace.device.type == "cuda":
            if caller_stream is None:
                caller_stream = torch.cuda.current_stream(workspace.device)
            if workspace.stream is None:
                raise RuntimeError(
                    "RCCL-SDMA ReduceScatter workspace has no reduction stream"
                )
            workspace.stream.wait_stream(caller_stream)
            with torch.cuda.stream(workspace.stream):
                self._schedule()
                self.done_event = torch.cuda.Event()
                self.done_event.record()
        else:
            # CPU is used only by isolated unit tests.
            self._schedule()
            self.completed = True

    def _launch(self, offset: int, slot: int):
        workspace = self.workspace
        output_numel = self.output_tensor.numel()
        count = min(
            workspace.chunk_numel,
            output_numel - offset,
        )
        if workspace.direct_input:
            if offset != 0 or count != output_numel:
                raise RuntimeError(
                    "RCCL-SDMA direct-input mode requires the complete bucket "
                    "to fit in staging"
                )
            if not self.input_tensor.is_contiguous():
                raise RuntimeError(
                    "RCCL-SDMA direct-input mode requires contiguous gradients"
                )
            send = self.input_tensor
        else:
            send = workspace.send[slot][: workspace.world_size * count]
            send.view(workspace.world_size, count).copy_(
                self.input_tensor.view(workspace.world_size, output_numel)[
                    :, offset : offset + count
                ]
            )
        recv = workspace.recv[slot][: workspace.world_size * count]
        work = dist.all_to_all_single(
            recv,
            send,
            group=workspace.group,
            async_op=True,
        )
        return offset, count, slot, work

    def _schedule(self) -> None:
        pipeline_enabled = (
            os.getenv("MEGATRON_RCCL_SDMA_RS_PIPELINE", "1") != "0"
        )
        current = self._launch(offset=0, slot=0)
        while current is not None:
            offset, count, slot, work = current
            next_offset = offset + count
            next_work = None
            if pipeline_enabled and next_offset < self.output_tensor.numel():
                # Queue the next pack and AllToAll before waiting on the
                # current chunk. ProcessGroupNCCL's own stream can then
                # overlap that transfer with this stream's FP32 sum.
                next_work = self._launch(
                    offset=next_offset,
                    slot=1 - slot,
                )

            work.wait()
            workspace = self.workspace
            recv = workspace.recv[slot][: workspace.world_size * count]
            _fp32_reduce_store(
                recv,
                self.output_tensor[offset : offset + count],
                workspace.world_size,
                self.op == dist.ReduceOp.AVG,
            )
            if (
                not pipeline_enabled
                and next_offset < self.output_tensor.numel()
            ):
                next_work = self._launch(
                    offset=next_offset,
                    slot=1 - slot,
                )
            current = next_work

    def wait(self) -> bool:
        if self.done_event is not None:
            torch.cuda.current_stream(self.workspace.device).wait_event(
                self.done_event
            )
        self.completed = True
        return True

    def is_completed(self) -> bool:
        if self.done_event is None:
            return self.completed
        return self.done_event.query()


@torch.compiler.disable
def reduce_scatter(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    op: dist.ReduceOp,
    group: dist.ProcessGroup,
    async_op: bool,
):
    """Megatron-compatible ReduceScatter with FP32 local accumulation."""
    if group is None:
        group = dist.group.WORLD
    if input_tensor.device != output_tensor.device:
        raise RuntimeError("RCCL-SDMA ReduceScatter tensor device mismatch")
    if input_tensor.dtype != output_tensor.dtype:
        raise RuntimeError("RCCL-SDMA ReduceScatter tensor dtype mismatch")
    world_size = group.size()
    if input_tensor.numel() != output_tensor.numel() * world_size:
        raise RuntimeError(
            "RCCL-SDMA ReduceScatter input must contain one output shard per rank"
        )
    if op not in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
        raise RuntimeError(
            "RCCL-SDMA ReduceScatter supports only SUM and AVG reductions"
        )

    workspace_pool = prepare_workspace(group, input_tensor)
    if _use_native_tail(workspace_pool, input_tensor):
        return dist.reduce_scatter_tensor(
            output_tensor,
            input_tensor,
            op=op,
            group=get_native_tail_process_group(group),
            async_op=async_op,
        )
    caller_stream = (
        torch.cuda.current_stream(input_tensor.device)
        if input_tensor.device.type == "cuda"
        else None
    )
    workspace = workspace_pool.acquire(caller_stream)
    work = ReduceScatterWork(
        workspace,
        output_tensor,
        input_tensor,
        op,
        caller_stream,
    )
    workspace.active_work = work
    if (
        workspace.group.rank() == 0
        and os.getenv("MEGATRON_RCCL_SDMA_LOG", "0") == "1"
    ):
        chunks = (
            output_tensor.numel() + workspace.chunk_numel - 1
        ) // workspace.chunk_numel
        print(
            "[RCCL-SDMA:Megatron] ReduceScatter "
            f"input_bytes={input_tensor.nbytes} "
            f"output_bytes={output_tensor.nbytes} "
            f"chunks={chunks}",
            flush=True,
        )
    if async_op:
        return work
    work.wait()
    return None


def reset_runtime_state_for_tests() -> None:
    """Clear process-global caches used by isolated tests."""
    _EAGER_DIRECT_RECV.clear()
    _WORKSPACES.clear()
    _POOLS.clear()
    _SUBGROUPS.clear()
    _NATIVE_TAIL_GROUPS.clear()
    _BUCKET_IDS_BY_PTR.clear()
