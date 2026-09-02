###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Route Megatron distributed-optimizer parameter AllGather through RCCL CE.

Activation is deliberately separate from Primus's direct-HIP SDMA backend:

    MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma

The replacement touches only ``_ParamAndGradBucketGroup.start_param_sync``.
Gradient ReduceScatter, gradient-norm AllReduce, and other collectives retain
their original process groups and algorithms.
"""

from __future__ import annotations

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


BACKEND_ENV = "MEGATRON_PARAM_GATHER_BACKEND"
RCCL_SDMA_BACKEND = "rccl_sdma"
_EAGER_RUNTIME_INITIALIZED = False


def rccl_sdma_param_gather_enabled(_ctx: PatchContext | None = None) -> bool:
    return os.getenv(BACKEND_ENV, "").strip().lower() == RCCL_SDMA_BACKEND


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


def validate_global_cumem_enable() -> None:
    """Reject process-wide cuMem, which can stall non-CE Megatron work."""
    global_cumem = os.getenv("NCCL_CUMEM_ENABLE", "").strip().lower()
    if global_cumem not in ("", "0"):
        raise RuntimeError(
            "MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma requires "
            "NCCL_CUMEM_ENABLE to be unset or 0. The dedicated parameter-"
            "AllGather group manages its symmetric scratch allocation without "
            "enabling cuMem process-wide."
        )


def make_start_param_sync(original):
    """Build the RCCL CE replacement for Megatron's parameter gather."""
    from megatron.core.distributed.param_and_grad_buffer import shard_buffer

    from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
        DEFAULT_SCRATCH_BYTES,
        get_runtime,
        get_sdma_process_group,
    )

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
            capacity_bytes = int(
                os.getenv(
                    "MEGATRON_RCCL_SDMA_SCRATCH_BYTES",
                    str(DEFAULT_SCRATCH_BYTES),
                )
            )
            runtime = get_runtime(
                get_sdma_process_group(
                    self.intra_distributed_optimizer_instance_group
                ),
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
    )

    if not dist.is_initialized():
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    world_probe = torch.zeros(1, dtype=torch.int32, device=device)
    dist.broadcast(world_probe, src=0)
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
    validate_global_cumem_enable()
    os.environ["NCCL_LOCAL_REGISTER"] = "0"
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "true"

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
        bucket_group.start_param_sync = make_start_param_sync(
            bucket_group.start_param_sync
        )
        setattr(bucket_group, marker, True)

    if os.getenv("MEGATRON_RCCL_SDMA_EAGER_INIT", "0") == "1":
        if not eager_initialize_runtime():
            import megatron.training.initialize as init_module

            eager_marker = "_primus_rccl_sdma_eager_init_patched"
            if not getattr(init_module, eager_marker, False):
                original_initialize_distributed = init_module._initialize_distributed

                def wrapped_initialize_distributed(*args, **kwargs):
                    result = original_initialize_distributed(*args, **kwargs)
                    if not eager_initialize_runtime():
                        raise RuntimeError(
                            "torch.distributed is not initialized after "
                            "Megatron distributed initialization"
                        )
                    return result

                init_module._initialize_distributed = wrapped_initialize_distributed
                setattr(init_module, eager_marker, True)

    log_rank_0(
        "[Patch:megatron.distributed.rccl_sdma_param_all_gather] installed"
    )
