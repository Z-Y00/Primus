###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Route Megatron's FP32-accumulation ReduceScatter through RCCL SDMA."""

from __future__ import annotations

import functools
import inspect
import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

BACKEND_ENV = "MEGATRON_GRAD_REDUCE_BACKEND"
RCCL_SDMA_A2A_BACKEND = "rccl_sdma_a2a"


def rccl_sdma_reduce_scatter_enabled(
    _ctx: PatchContext | None = None,
) -> bool:
    return (
        os.getenv(BACKEND_ENV, "").strip().lower()
        == RCCL_SDMA_A2A_BACKEND
    )


def make_bucket_group_init(original):
    """Enable Megatron's custom FP32-accumulation work-handle path."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        ddp_config = bound.arguments["ddp_config"]
        collective_group = bound.arguments["collective_group"]

        if not ddp_config.use_distributed_optimizer:
            raise RuntimeError(
                "RCCL-SDMA ReduceScatter requires the distributed optimizer"
            )
        if ddp_config.num_distributed_optimizer_instances != 1:
            raise RuntimeError(
                "RCCL-SDMA ReduceScatter does not support multiple "
                "distributed-optimizer instances"
            )

        ddp_config.reduce_scatter_with_fp32_accumulation = True
        result = original(self, *args, **kwargs)
        if len(self.buckets) != 1:
            raise RuntimeError(
                "RCCL-SDMA ReduceScatter currently requires one bucket per "
                f"bucket group, got {len(self.buckets)}"
            )

        from primus.backends.megatron.core.distributed.rccl_sdma_reduce_scatter import (
            prepare_workspace,
            register_gradient_bucket,
        )

        bucket = self.buckets[0]
        register_gradient_bucket(bucket)
        prepare_workspace(
            collective_group,
            bucket.grad_data,
        )
        return result

    return wrapped


@register_patch(
    "megatron.distributed.rccl_sdma_reduce_scatter",
    backend="megatron",
    phase="before_train",
    description=(
        "Route distributed-optimizer gradient ReduceScatter through chunked "
        "zero-CTA RCCL AllToAll with local FP32 accumulation."
    ),
    condition=rccl_sdma_reduce_scatter_enabled,
)
def patch_rccl_sdma_reduce_scatter(ctx: PatchContext) -> None:
    del ctx

    from primus.backends.megatron.patches.parallelism.rccl_sdma_param_all_gather_patches import (
        validate_global_cta_policy,
    )

    validate_global_cta_policy()
    os.environ["NCCL_CUMEM_ENABLE"] = "1"
    os.environ["NCCL_LOCAL_REGISTER"] = "0"
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "true"

    try:
        import megatron.core.distributed.param_and_grad_buffer as pgb
        import megatron.core.distributed.reduce_scatter_with_fp32_accumulation as rs_module
    except ImportError as exc:
        warning_rank_0(
            "[Patch:megatron.distributed.rccl_sdma_reduce_scatter] "
            f"Megatron distributed modules are unavailable; skipping: {exc}"
        )
        return

    from primus.backends.megatron.core.distributed.rccl_sdma_reduce_scatter import (
        reduce_scatter,
    )

    # Megatron imports this function into param_and_grad_buffer.py, then copies
    # that binding into its module-global dist_reduce_scatter_func when each
    # bucket group is initialized. Patch both bindings before wrapping __init__.
    rs_module.reduce_scatter_with_fp32_accumulation = reduce_scatter
    pgb.reduce_scatter_with_fp32_accumulation = reduce_scatter

    bucket_group = getattr(pgb, "_ParamAndGradBucketGroup", None)
    if bucket_group is None:
        raise RuntimeError(
            "RCCL-SDMA ReduceScatter requires _ParamAndGradBucketGroup"
        )
    marker = "_primus_rccl_sdma_reduce_scatter_patched"
    if not getattr(bucket_group, marker, False):
        bucket_group.__init__ = make_bucket_group_init(bucket_group.__init__)
        setattr(bucket_group, marker, True)

    if os.getenv("MEGATRON_RCCL_SDMA_RS_DIRECT_INPUT", "0") == "1":
        import torch
        import megatron.training.training as training
        from megatron.core import parallel_state
        from megatron.training import get_args

        get_model_marker = "_primus_rccl_sdma_rs_eager_workspace_patched"
        if not getattr(training, get_model_marker, False):
            original_get_model = training.get_model

            @functools.wraps(original_get_model)
            def wrapped_get_model(*args, **kwargs):
                megatron_args = get_args()
                if getattr(megatron_args, "bf16", False):
                    dtype = torch.bfloat16
                elif getattr(megatron_args, "fp16", False):
                    dtype = torch.float16
                else:
                    raise RuntimeError(
                        "RCCL-SDMA direct-input ReduceScatter requires "
                        "BF16 or FP16 training"
                    )
                from primus.backends.megatron.core.distributed.rccl_sdma_reduce_scatter import (
                    reserve_direct_input_workspace,
                )

                reserve_direct_input_workspace(
                    parallel_state.get_data_parallel_group(
                        with_context_parallel=True
                    ),
                    torch.device("cuda", torch.cuda.current_device()),
                    dtype,
                    int(
                        os.getenv(
                            "MEGATRON_RCCL_SDMA_RS_STAGING_BYTES",
                            "536870912",
                        )
                    ),
                )
                return original_get_model(*args, **kwargs)

            training.get_model = wrapped_get_model
            setattr(training, get_model_marker, True)

    log_rank_0(
        "[Patch:megatron.distributed.rccl_sdma_reduce_scatter] installed"
    )
