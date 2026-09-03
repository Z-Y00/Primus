###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Route Megatron distributed-optimizer parameter AllGather through RCCL CE.

Activation is deliberately separate from Primus's direct-HIP SDMA backend:

    MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma
    MEGATRON_RCCL_SDMA_DIRECT=1  # optional direct symmetric-buffer path

The replacement touches only ``_ParamAndGradBucketGroup.start_param_sync``.
Gradient ReduceScatter, gradient-norm AllReduce, and other collectives retain
their original process groups and algorithms.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
from unittest import mock

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

BACKEND_ENV = "MEGATRON_PARAM_GATHER_BACKEND"
RCCL_SDMA_BACKEND = "rccl_sdma"
DIRECT_GATHER_ENV = "MEGATRON_RCCL_SDMA_DIRECT"
_EAGER_RUNTIME_INITIALIZED = False


def rccl_sdma_param_gather_enabled(_ctx: PatchContext | None = None) -> bool:
    return os.getenv(BACKEND_ENV, "").strip().lower() == RCCL_SDMA_BACKEND


def direct_param_gather_enabled() -> bool:
    return rccl_sdma_param_gather_enabled() and os.getenv(DIRECT_GATHER_ENV, "0") == "1"


def validate_global_cta_policy() -> None:
    """Reject a process-wide policy that would override per-group settings."""
    global_policy = os.getenv("NCCL_CTA_POLICY")
    if global_policy:
        raise RuntimeError(
            "MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma requires "
            "NCCL_CTA_POLICY to be unset. A process-wide CTA policy is applied "
            "while every RCCL communicator is initialized and cannot be "
            "safely undone during the before_train patch phase."
        )


def make_start_param_sync(original):
    """Build the RCCL CE replacement for Megatron's parameter gather."""
    from megatron.core.distributed.param_and_grad_buffer import shard_buffer

    from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
        DEFAULT_SCRATCH_BYTES,
        get_runtime,
        get_sdma_process_group,
        is_direct_param_buffer,
    )
    from torch.distributed.distributed_c10d import _coalescing_manager

    def start_param_sync(self, force_sync: bool = False):
        if not self.ddp_config.use_distributed_optimizer:
            return original(self, force_sync=force_sync)
        if force_sync and self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
            return
        if not force_sync:
            assert self.param_gather_handle is None

        async_op = self.ddp_config.overlap_param_gather and not force_sync
        jobs = []
        for index, bucket in enumerate(self.buckets):
            if self.cached_param_buffer_shard_list[index] is None:
                self.cached_param_buffer_shard_list[index] = shard_buffer(
                    bucket.param_data,
                    self.intra_distributed_optimizer_instance_size,
                )
            local_data = self.cached_param_buffer_shard_list[index][
                self.intra_distributed_optimizer_instance_rank
            ]
            jobs.append((bucket.param_data, local_data))

        if jobs:
            group = get_sdma_process_group(self.intra_distributed_optimizer_instance_group)
            if direct_param_gather_enabled() and all(
                is_direct_param_buffer(output) for output, _input in jobs
            ):
                # Match Megatron's native parameter-gather stream semantics:
                # submit from the caller's current stream, let ProcessGroupNCCL
                # manage its internal stream/events, and retain its coalesced
                # work handle for finish_param_sync().  The dedicated process
                # group still selects RCCL's zero-CTA copy-engine path.
                with _coalescing_manager(group, async_ops=async_op) as cm:
                    for output, local_input in jobs:
                        torch.distributed.all_gather_into_tensor(
                            output,
                            local_input,
                            group=group,
                            async_op=async_op,
                        )
                self.param_gather_handle = cm if async_op else None
            else:
                capacity_bytes = int(
                    os.getenv(
                        "MEGATRON_RCCL_SDMA_SCRATCH_BYTES",
                        str(DEFAULT_SCRATCH_BYTES),
                    )
                )
                runtime = get_runtime(
                    group,
                    jobs[0][0].device,
                    capacity_bytes,
                )
                work = runtime.enqueue(jobs)
                if async_op:
                    self.param_gather_handle = work
                else:
                    work.wait()
                    self.param_gather_handle = None
        else:
            self.param_gather_handle = None
        self.param_gather_dispatched = True

    return start_param_sync


def make_param_and_grad_buffer_init(original):
    """Allocate eligible Megatron parameter buffers from the symmetric pool."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        ddp_config = bound.arguments["ddp_config"]
        params = bound.arguments["params"]
        original_group = bound.arguments["data_parallel_group"]
        nccl_ub = bound.arguments["nccl_ub"]

        # nccl_ub owns a separate registered pool. MXFP8 may alias parameter
        # and gradient storage, so both retain the compatibility scratch path.
        from megatron.core.distributed.param_and_grad_buffer import is_mxfp8tensor

        eligible = (
            ddp_config.use_distributed_optimizer
            and not nccl_ub
            and not any(is_mxfp8tensor(param) for param in params)
        )
        if not eligible:
            return original(self, *args, **kwargs)

        from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
            mark_direct_param_buffer,
            prepare_direct_param_buffer_pool,
            rendezvous_direct_param_buffer,
            take_direct_param_buffer,
        )

        device = params[0].device
        group, pool = prepare_direct_param_buffer_pool(original_group, device)
        original_zeros = torch.zeros
        allocation_thread = threading.get_ident()
        param_data_allocated = False

        def allocate_param_data(*zeros_args, **zeros_kwargs):
            nonlocal param_data_allocated
            if threading.get_ident() == allocation_thread and not param_data_allocated:
                param_data_allocated = True
                dtype = zeros_kwargs.get("dtype")
                if dtype is None:
                    raise RuntimeError("RCCL-SDMA direct param_data allocation requires an explicit dtype")
                eager_tensor = take_direct_param_buffer(
                    group,
                    device,
                    zeros_args[0],
                    dtype,
                )
                if eager_tensor is not None:
                    return eager_tensor
                with torch.cuda.use_mem_pool(pool):
                    return original_zeros(*zeros_args, **zeros_kwargs)
            return original_zeros(*zeros_args, **zeros_kwargs)

        # Megatron allocates param_data first and grad_data second in this
        # eligible non-MXFP8 constructor path. Only param_data participates in
        # direct AllGather, so keep grad_data on the normal allocator.
        with mock.patch.object(torch, "zeros", allocate_param_data):
            result = original(self, *args, **kwargs)

        if not param_data_allocated or self.param_data is None:
            raise RuntimeError("RCCL-SDMA direct gather did not allocate param_data")
        symmetric_memory = rendezvous_direct_param_buffer(self.param_data, group)
        # Keep the pool and rendezvous handle alive for the buffer lifetime.
        self._primus_rccl_sdma_pool = pool
        self._primus_rccl_sdma_symmetric_memory = symmetric_memory
        mark_direct_param_buffer(self.param_data)
        for bucket in self.buckets:
            if bucket.param_data is not None:
                mark_direct_param_buffer(bucket.param_data)
        return result

    return wrapped


def eager_initialize_runtime() -> bool:
    """Reserve WORLD and CE cuMem pools before DDP/optimizer allocations."""
    global _EAGER_RUNTIME_INITIALIZED
    if _EAGER_RUNTIME_INITIALIZED:
        return True

    import torch
    import torch.distributed as dist

    from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
        DEFAULT_SCRATCH_BYTES,
        get_runtime,
        get_sdma_process_group,
        prepare_direct_param_buffer_pool,
    )

    if not dist.is_initialized():
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    world_probe = torch.zeros(1, dtype=torch.int32, device=device)
    dist.broadcast(world_probe, src=0)
    if direct_param_gather_enabled():
        prepare_direct_param_buffer_pool(dist.group.WORLD, device)
    else:
        capacity_bytes = int(
            os.getenv(
                "MEGATRON_RCCL_SDMA_SCRATCH_BYTES",
                str(DEFAULT_SCRATCH_BYTES),
            )
        )
        get_runtime(
            get_sdma_process_group(dist.group.WORLD),
            device,
            capacity_bytes,
        )
    _EAGER_RUNTIME_INITIALIZED = True
    return True


@register_patch(
    "megatron.distributed.rccl_sdma_param_all_gather",
    backend="megatron",
    phase="before_train",
    description=(
        "Route distributed-optimizer parameter AllGather through a dedicated "
        "zero-CTA RCCL copy-engine process group."
    ),
    condition=rccl_sdma_param_gather_enabled,
)
def patch_rccl_sdma_param_all_gather(ctx: PatchContext) -> None:
    del ctx

    if os.getenv("ENABLE_SDMA_ALLGATHER", "0") == "1":
        warning_rank_0(
            "[Patch:megatron.distributed.rccl_sdma_param_all_gather] "
            "disabling ENABLE_SDMA_ALLGATHER to avoid the direct-HIP SDMA path"
        )
        os.environ["ENABLE_SDMA_ALLGATHER"] = "0"

    # The dedicated process group selects zero CTA through pg_options. A global
    # policy may already have affected WORLD/DP communicators by this phase, so
    # fail instead of silently removing it too late.
    validate_global_cta_policy()
    os.environ["NCCL_CUMEM_ENABLE"] = "1"
    os.environ["NCCL_LOCAL_REGISTER"] = "0"
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "true"

    eager_param_bytes = int(os.getenv("MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES", "0"))
    if int(os.getenv("RANK", "0")) == 0:
        print(
            "[RCCL-SDMA:Megatron] eager direct configuration "
            f"enabled={direct_param_gather_enabled()} bytes={eager_param_bytes}",
            flush=True,
        )
    if direct_param_gather_enabled() and eager_param_bytes:
        import torch.distributed._symmetric_memory as symm_mem

        from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
            reserve_direct_param_buffer,
        )

        device = torch.device("cuda", int(os.getenv("LOCAL_RANK", torch.cuda.current_device())))
        torch.cuda.set_device(device)
        symm_mem.set_backend("NCCL")
        pool = symm_mem.get_mem_pool(device)
        reserve_direct_param_buffer(
            None,
            pool,
            device,
            eager_param_bytes,
        )

    try:
        import megatron.core.distributed.param_and_grad_buffer as pgb
    except ImportError as exc:
        warning_rank_0(
            "[Patch:megatron.distributed.rccl_sdma_param_all_gather] "
            f"Megatron distributed modules are unavailable; skipping: {exc}"
        )
        return

    bucket_group = getattr(pgb, "_ParamAndGradBucketGroup", None)
    if bucket_group is None:
        warning_rank_0(
            "[Patch:megatron.distributed.rccl_sdma_param_all_gather] "
            "_ParamAndGradBucketGroup is unavailable; skipping"
        )
        return

    marker = "_primus_rccl_sdma_param_gather_patched"
    if not getattr(bucket_group, marker, False):
        bucket_group.start_param_sync = make_start_param_sync(bucket_group.start_param_sync)
        setattr(bucket_group, marker, True)

    if direct_param_gather_enabled():
        param_and_grad_buffer = getattr(pgb, "_ParamAndGradBuffer", None)
        if param_and_grad_buffer is None:
            raise RuntimeError("RCCL-SDMA direct gather requires _ParamAndGradBuffer")
        direct_marker = "_primus_rccl_sdma_direct_allocation_patched"
        if not getattr(param_and_grad_buffer, direct_marker, False):
            param_and_grad_buffer.__init__ = make_param_and_grad_buffer_init(param_and_grad_buffer.__init__)
            setattr(param_and_grad_buffer, direct_marker, True)

    if os.getenv("MEGATRON_RCCL_SDMA_EAGER_INIT", "0") == "1":
        if not eager_initialize_runtime():
            import megatron.training.training as training_module

            eager_marker = "_primus_rccl_sdma_eager_init_patched"
            if not getattr(training_module, eager_marker, False):
                original_initialize_megatron = training_module.initialize_megatron

                @functools.wraps(original_initialize_megatron)
                def wrapped_initialize_megatron(*args, **kwargs):
                    result = original_initialize_megatron(*args, **kwargs)
                    if callable(result):
                        deferred_initialize = result

                        @functools.wraps(deferred_initialize)
                        def wrapped_deferred_initialize(*deferred_args, **deferred_kwargs):
                            deferred_result = deferred_initialize(
                                *deferred_args,
                                **deferred_kwargs,
                            )
                            if not eager_initialize_runtime():
                                raise RuntimeError(
                                    "torch.distributed is not initialized after "
                                    "deferred Megatron initialization"
                                )
                            return deferred_result

                        return wrapped_deferred_initialize
                    if not eager_initialize_runtime():
                        raise RuntimeError(
                            "torch.distributed is not initialized after "
                            "Megatron initialization"
                        )
                    return result

                training_module.initialize_megatron = wrapped_initialize_megatron
                setattr(training_module, eager_marker, True)

    log_rank_0("[Patch:megatron.distributed.rccl_sdma_param_all_gather] installed")
