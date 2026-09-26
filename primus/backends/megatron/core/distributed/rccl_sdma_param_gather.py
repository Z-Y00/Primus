###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Direct RCCL copy-engine parameter AllGather support for Megatron.

Megatron's parameter buffer is allocated from PyTorch NCCL symmetric memory
and gathered in place through a dedicated zero-CTA ProcessGroupNCCL
communicator. When gradient ReduceScatter is also routed through the direct
buffer (see ``rccl_sdma_param_all_gather_patches.patch_rccl_sdma_grad_reduce_scatter``),
the gradient buffer shares this same pool and dedicated group. All other
collectives retain their original process groups.
"""

from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

LARGE_SEGMENT_BYTES = 2 * 1024 * 1024

_DIRECT_POOLS: dict[tuple[str, int], torch.cuda.MemPool] = {}
# Keyed by (role, group_name, device_index, size_bytes). ``role`` ("param" or
# "grad") keeps same-sized param and gradient eager reservations from
# colliding -- e.g. bf16 param_data and bf16 grad_data are frequently the same
# size, and without a role dimension the second reservation would either
# no-op against the first key or be silently consumed by the first buffer's
# `torch.zeros` interception, leaving the other role with nothing to take.
_DIRECT_EAGER_BUFFERS: dict[tuple[str, str, int, int], torch.Tensor] = {}
_SDMA_GROUP: dist.ProcessGroup | None = None
DIRECT_BUFFER_ATTR = "_primus_rccl_sdma_direct_buffer"


def _eager_bytes_env_var(role: str) -> str:
    """Return the env var that sizes the eager reservation for ``role``."""
    return (
        "MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES" if role == "param" else "MEGATRON_RCCL_SDMA_EAGER_GRAD_BYTES"
    )


def recommended_eager_param_bytes(size_bytes: int) -> int:
    """Round a parameter buffer size up to the symmetric allocator granule."""
    if size_bytes <= 0:
        raise ValueError("parameter buffer size must be positive")
    return (size_bytes + LARGE_SEGMENT_BYTES - 1) // LARGE_SEGMENT_BYTES * LARGE_SEGMENT_BYTES


def get_sdma_process_group(
    original_group: dist.ProcessGroup,
) -> dist.ProcessGroup:
    """Create one full-rank communicator used only for zero-CTA AllGather."""
    global _SDMA_GROUP

    if original_group.size() != dist.get_world_size():
        raise RuntimeError(
            "RCCL-SDMA direct parameter gather requires the distributed-optimizer "
            "group to contain every rank"
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
            raise ValueError(f"MEGATRON_RCCL_SDMA_CTA_POLICY must be 0 or 2, got {cta_policy}")
        options.config.split_share = 0
        _SDMA_GROUP = dist.new_group(
            ranks=list(range(dist.get_world_size())),
            backend="nccl",
            pg_options=options,
            group_desc=f"MEGATRON_RCCL_SDMA_PARAM_GATHER_POLICY_{cta_policy}",
        )
        if dist.get_rank() == 0:
            print(
                "[RCCL-SDMA:Megatron] created dedicated zero-CTA group " f"name={_SDMA_GROUP.group_name}",
                flush=True,
            )
    return _SDMA_GROUP


def prepare_direct_param_buffer_pool(
    original_group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[dist.ProcessGroup, torch.cuda.MemPool]:
    """Enable and return the symmetric pool used by direct parameter buffers."""
    group = get_sdma_process_group(original_group)
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (group.group_name, device_index)
    pool = _DIRECT_POOLS.get(key)
    if pool is None:
        symm_mem.set_backend("NCCL")
        symm_mem.enable_symm_mem_for_group(group.group_name)
        dist.barrier(group=group)
        pool = symm_mem.get_mem_pool(device)
        _DIRECT_POOLS[key] = pool
    return group, pool


def reserve_direct_param_buffer(
    group: dist.ProcessGroup | None,
    pool: torch.cuda.MemPool,
    device: torch.device,
    size_bytes: int,
    role: str = "param",
) -> None:
    """Allocate a direct buffer before model allocations fragment HBM.

    ``role`` distinguishes param-data from grad-data reservations so two
    same-sized eager reservations (a common case: bf16 param_data and bf16
    grad_data are often identically sized) don't collide on the same cache key.
    """
    if size_bytes <= 0:
        raise ValueError("eager direct buffer size must be positive")
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    group_name = group.group_name if group is not None else ""
    key = (role, group_name, device_index, size_bytes)
    if key in _DIRECT_EAGER_BUFFERS:
        return
    with torch.cuda.use_mem_pool(pool):
        storage = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    _DIRECT_EAGER_BUFFERS[key] = storage
    rank = group.rank() if group is not None else int(os.getenv("RANK", "0"))
    if rank == 0:
        print(
            f"[RCCL-SDMA:Megatron] eagerly reserved direct {role} buffer "
            f"bytes={size_bytes} group={group_name or '<pending>'}",
            flush=True,
        )


def take_direct_param_buffer(
    group: dist.ProcessGroup,
    device: torch.device,
    shape,
    dtype: torch.dtype,
    role: str = "param",
) -> torch.Tensor | None:
    """Transfer a size-compatible eager reservation to Megatron's param/grad data."""
    numel = int(shape) if isinstance(shape, int) else math.prod(shape)
    size_bytes = numel * torch.empty((), dtype=dtype).element_size()
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    # Any reservation at least as large as the request is usable -- the buffer
    # is sliced to size_bytes below. Requiring a near-exact size match would
    # force callers to predict the buffer size to within the allocator granule;
    # an over-sized reservation would silently go unmatched and fall back to a
    # second pool allocation on top of the reservation it should have reused,
    # which exhausts the pool instead of consuming it.
    matching_keys = [
        key
        for key in _DIRECT_EAGER_BUFFERS
        if key[0] == role
        and key[1] in ("", group.group_name)
        and key[2] == device_index
        and key[3] >= size_bytes
    ]
    if not matching_keys:
        if group.rank() == 0:
            recommended_bytes = recommended_eager_param_bytes(size_bytes)
            env_var = _eager_bytes_env_var(role)
            print(
                f"[RCCL-SDMA:Megatron] no eager direct {role} buffer match "
                f"requested_bytes={size_bytes} "
                f"recommended_eager_bytes={recommended_bytes} "
                f"reserved={list(_DIRECT_EAGER_BUFFERS)}; "
                "if direct allocation fails, rerun with "
                f"{env_var}={recommended_bytes}",
                flush=True,
            )
        return None
    key = min(matching_keys, key=lambda candidate: candidate[3])
    storage = _DIRECT_EAGER_BUFFERS.pop(key)
    tensor = storage[:size_bytes].view(dtype).view(shape)
    tensor.zero_()
    if group.rank() == 0:
        slack_bytes = storage.nbytes - size_bytes
        message = (
            f"[RCCL-SDMA:Megatron] consumed eager direct {role} buffer "
            f"reserved_bytes={storage.nbytes} requested_bytes={size_bytes}"
        )
        if slack_bytes >= LARGE_SEGMENT_BYTES:
            env_var = _eager_bytes_env_var(role)
            message += (
                f"; {slack_bytes} bytes of the reservation are unused, "
                f"set {env_var}={recommended_eager_param_bytes(size_bytes)} to reclaim them"
            )
        print(message, flush=True)
    return tensor


def rendezvous_direct_param_buffer(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> object:
    """Register a pool-backed Megatron parameter buffer for direct CE gather."""
    symmetric_memory = symm_mem.rendezvous(tensor, group=group.group_name)
    setattr(tensor, DIRECT_BUFFER_ATTR, True)
    if group.rank() == 0 and os.getenv("MEGATRON_RCCL_SDMA_LOG", "0") == "1":
        print(
            "[RCCL-SDMA:Megatron] rendezvoused direct parameter buffer "
            f"bytes={tensor.nbytes} group={group.group_name}",
            flush=True,
        )
    return symmetric_memory


def mark_direct_param_buffer(tensor: torch.Tensor) -> None:
    """Mark a view whose storage was rendezvoused for direct gather."""
    setattr(tensor, DIRECT_BUFFER_ATTR, True)


def is_direct_param_buffer(tensor: torch.Tensor) -> bool:
    """Return whether a tensor view belongs to a direct symmetric buffer."""
    return bool(getattr(tensor, DIRECT_BUFFER_ATTR, False))


def reset_runtime_state_for_tests() -> None:
    """Clear process-global caches used by isolated unit tests."""
    global _SDMA_GROUP
    _DIRECT_POOLS.clear()
    _DIRECT_EAGER_BUFFERS.clear()
    _SDMA_GROUP = None
