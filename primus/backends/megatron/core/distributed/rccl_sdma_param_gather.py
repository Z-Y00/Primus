###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RCCL copy-engine parameter AllGather for Megatron's distributed optimizer.

The output buffers owned by Megatron are regular allocations, while RCCL's
zero-CTA copy-engine path requires symmetric memory.  This adapter therefore
uses one reusable symmetric scratch allocation, gathers bounded chunks into
that allocation on a dedicated stream, and copies each completed chunk into
Megatron's parameter buffer.

Only parameter AllGather uses the dedicated zero-CTA process group.  Gradient
ReduceScatter and all other collectives continue to use Megatron's original
process groups.
"""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import threading
import time
from typing import Iterable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


DEFAULT_SCRATCH_BYTES = 512 * 1024 * 1024

_RUNTIMES: dict[tuple[str, int], "_RcclSdmaRuntime"] = {}
_SDMA_GROUP: dist.ProcessGroup | None = None


def max_chunk_numel(
    capacity_bytes: int,
    element_size: int,
    world_size: int,
) -> int:
    """Return the largest per-rank chunk that fits in symmetric scratch."""
    if capacity_bytes <= 0:
        raise ValueError("capacity_bytes must be positive")
    if element_size <= 0:
        raise ValueError("element_size must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return capacity_bytes // element_size // world_size


def chunk_count(numel: int, chunk_numel: int) -> int:
    """Return the number of bounded chunks needed for ``numel`` elements."""
    if numel < 0:
        raise ValueError("numel must be non-negative")
    if chunk_numel <= 0:
        raise ValueError("chunk_numel must be positive")
    return (numel + chunk_numel - 1) // chunk_numel


class EventWork:
    """Work-like object whose wait inserts a device dependency, not a host sync."""

    def __init__(self, event: torch.cuda.Event, device: torch.device) -> None:
        self.event = event
        self.device = device
        self.waited = False

    def wait(self) -> bool:
        if not self.waited:
            if os.getenv("MEGATRON_RCCL_SDMA_HOST_WAIT", "0") == "1":
                self.event.synchronize()
            else:
                torch.cuda.current_stream(self.device).wait_event(self.event)
            self.waited = True
        return True

    def is_completed(self) -> bool:
        return self.waited


class _RcclSdmaRuntime:
    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        capacity_bytes: int,
    ) -> None:
        self.group = group
        self.device = device
        self.capacity_bytes = capacity_bytes
        self.rank = group.rank()
        self.world_size = group.size()
        self.stream = torch.cuda.Stream(device=device)
        self._logged_layouts: set[tuple[int, int]] = set()
        self._collective_sequence = 0
        self._enqueue_sequence = 0
        self._trace_file_lock = threading.Lock()
        self._pending_device_events: deque[
            tuple[int, int, int, int, torch.cuda.Event]
        ] = deque()
        self._pending_device_events_lock = threading.Lock()
        self._trace_device = (
            os.getenv("MEGATRON_RCCL_SDMA_TRACE_DEVICE", "0") == "1"
        )

        symm_mem.set_backend("NCCL")
        symm_mem.enable_symm_mem_for_group(group.group_name)
        dist.barrier(group=group)
        pool = symm_mem.get_mem_pool(device)
        with torch.cuda.use_mem_pool(pool):
            self.scratch = torch.empty(
                capacity_bytes,
                dtype=torch.uint8,
                device=device,
            )
        self.symmetric_memory = symm_mem.rendezvous(
            self.scratch,
            group=group.group_name,
        )

        if self.rank == 0:
            print(
                "[RCCL-SDMA:Megatron] rendezvoused symmetric scratch "
                f"bytes={capacity_bytes} group={group.group_name}",
                flush=True,
            )
        if self._trace_device:
            threading.Thread(
                target=self._device_trace_loop,
                name=f"rccl-sdma-device-trace-{self.rank}",
                daemon=True,
            ).start()

    def _log_layout(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        chunks: int,
    ) -> None:
        if self.rank != 0 or os.getenv("MEGATRON_RCCL_SDMA_LOG", "0") != "1":
            return
        layout = (output_tensor.numel(), input_tensor.element_size())
        if layout in self._logged_layouts:
            return
        self._logged_layouts.add(layout)
        print(
            "[RCCL-SDMA:Megatron] gather-layout "
            f"input_bytes={input_tensor.nbytes} "
            f"output_bytes={output_tensor.nbytes} chunks={chunks}",
            flush=True,
        )

    def _trace_sequence(self, message: str) -> None:
        trace_dir = os.getenv("MEGATRON_RCCL_SDMA_TRACE_DIR")
        if not trace_dir:
            return
        path = Path(trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        with self._trace_file_lock:
            with (path / f"rank{self.rank}.log").open("a") as stream:
                stream.write(f"{message}\n")
                stream.flush()

    def _record_device_trace(
        self,
        sequence: int,
        output_numel: int,
        offset: int,
        chunk_numel: int,
    ) -> None:
        if not self._trace_device:
            return
        event = torch.cuda.Event()
        event.record(self.stream)
        with self._pending_device_events_lock:
            self._pending_device_events.append(
                (sequence, output_numel, offset, chunk_numel, event)
            )

    def _device_trace_loop(self) -> None:
        torch.cuda.set_device(self.device)
        while True:
            with self._pending_device_events_lock:
                pending = (
                    self._pending_device_events[0]
                    if self._pending_device_events
                    else None
                )
            if pending is None:
                time.sleep(0.001)
                continue
            sequence, output_numel, offset, chunk_numel, event = pending
            if not event.query():
                time.sleep(0.001)
                continue
            with self._pending_device_events_lock:
                completed = self._pending_device_events.popleft()
            if completed is not pending:
                raise RuntimeError("RCCL-SDMA device trace queue reordered")
            self._trace_sequence(
                f"seq={sequence} phase=device_collective_complete "
                f"output_numel={output_numel} "
                f"offset={offset} chunk_numel={chunk_numel}"
            )

    @torch.compiler.disable
    def enqueue(
        self,
        jobs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> EventWork:
        jobs = list(jobs)
        self._enqueue_sequence += 1
        enqueue_sequence = self._enqueue_sequence
        self._trace_sequence(
            f"enqueue={enqueue_sequence} phase=start "
            f"outputs={[output.numel() for output, _input in jobs]}"
        )
        caller_stream = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(caller_stream)
            for output_tensor, input_tensor in jobs:
                if output_tensor.device != self.device or input_tensor.device != self.device:
                    raise RuntimeError("RCCL-SDMA bucket device changed")
                if output_tensor.dtype != input_tensor.dtype:
                    raise RuntimeError("RCCL-SDMA bucket dtype mismatch")
                if output_tensor.numel() != input_tensor.numel() * self.world_size:
                    raise RuntimeError("RCCL-SDMA bucket is not evenly sharded")

                per_rank_chunk_numel = max_chunk_numel(
                    self.capacity_bytes,
                    output_tensor.element_size(),
                    self.world_size,
                )
                if per_rank_chunk_numel < 1:
                    raise RuntimeError("RCCL-SDMA scratch cannot hold one element per rank")

                chunks = chunk_count(input_tensor.numel(), per_rank_chunk_numel)
                self._log_layout(output_tensor, input_tensor, chunks)
                output_by_rank = output_tensor.view(
                    self.world_size,
                    input_tensor.numel(),
                )
                for offset in range(0, input_tensor.numel(), per_rank_chunk_numel):
                    current_chunk_numel = min(
                        per_rank_chunk_numel,
                        input_tensor.numel() - offset,
                    )
                    scratch_numel = current_chunk_numel * self.world_size
                    scratch_bytes = scratch_numel * output_tensor.element_size()
                    scratch = self.scratch[:scratch_bytes].view(output_tensor.dtype)
                    local_scratch = scratch.narrow(
                        0,
                        self.rank * current_chunk_numel,
                        current_chunk_numel,
                    )
                    local_scratch.copy_(
                        input_tensor.narrow(0, offset, current_chunk_numel),
                        non_blocking=True,
                    )
                    self._collective_sequence += 1
                    sequence = self._collective_sequence
                    if os.getenv("MEGATRON_RCCL_SDMA_TRACE_CHUNKS", "0") == "1":
                        self._trace_sequence(
                            f"seq={sequence} phase=call "
                            f"output_numel={output_tensor.numel()} "
                            f"offset={offset} chunk_numel={current_chunk_numel}"
                        )
                    dist.all_gather_into_tensor(
                        scratch,
                        local_scratch,
                        group=self.group,
                        async_op=False,
                    )
                    if os.getenv("MEGATRON_RCCL_SDMA_TRACE_CHUNKS", "0") == "1":
                        self._trace_sequence(
                            f"seq={sequence} phase=host_return "
                            f"output_numel={output_tensor.numel()} "
                            f"offset={offset} chunk_numel={current_chunk_numel}"
                        )
                    self._record_device_trace(
                        sequence,
                        output_tensor.numel(),
                        offset,
                        current_chunk_numel,
                    )
                    output_by_rank[:, offset : offset + current_chunk_numel].copy_(
                        scratch.view(self.world_size, current_chunk_numel),
                        non_blocking=True,
                    )
                    if os.getenv("MEGATRON_RCCL_SDMA_SYNC_EACH_CHUNK", "0") == "1":
                        self.stream.synchronize()
                        self._trace_sequence(
                            f"seq={sequence} phase=device_complete "
                            f"output_numel={output_tensor.numel()} "
                            f"offset={offset} chunk_numel={current_chunk_numel}"
                        )

            done = torch.cuda.Event()
            done.record(self.stream)
        self._trace_sequence(
            f"enqueue={enqueue_sequence} phase=host_return "
            f"collective_sequence={self._collective_sequence}"
        )
        return EventWork(done, self.device)


def get_runtime(
    group: dist.ProcessGroup,
    device: torch.device,
    capacity_bytes: int,
) -> _RcclSdmaRuntime:
    """Return the per-process-group/device reusable runtime."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (group.group_name, device_index)
    runtime = _RUNTIMES.get(key)
    if runtime is None:
        runtime = _RcclSdmaRuntime(group, device, capacity_bytes)
        _RUNTIMES[key] = runtime
    return runtime


def get_sdma_process_group(
    original_group: dist.ProcessGroup,
) -> dist.ProcessGroup:
    """Create one full-rank communicator used only for zero-CTA AllGather."""
    global _SDMA_GROUP

    if original_group.size() != dist.get_world_size():
        raise RuntimeError(
            "RCCL-SDMA currently requires the distributed-optimizer group "
            "to contain every rank"
        )
    if _SDMA_GROUP is None:
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
                "MEGATRON_RCCL_SDMA_CTA_POLICY must be 0 or 2, "
                f"got {cta_policy}"
            )
        options.config.split_share = 0
        _SDMA_GROUP = dist.new_group(
            ranks=list(range(dist.get_world_size())),
            backend="nccl",
            pg_options=options,
            group_desc=f"MEGATRON_RCCL_SDMA_PARAM_GATHER_POLICY_{cta_policy}",
        )
        if dist.get_rank() == 0:
            print(
                "[RCCL-SDMA:Megatron] created dedicated zero-CTA group "
                f"name={_SDMA_GROUP.group_name}",
                flush=True,
            )
    return _SDMA_GROUP


def reset_runtime_state_for_tests() -> None:
    """Clear process-global caches used by isolated unit tests."""
    global _SDMA_GROUP
    _RUNTIMES.clear()
    _SDMA_GROUP = None
