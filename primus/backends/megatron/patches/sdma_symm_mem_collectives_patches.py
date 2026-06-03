###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron (use_torch_fsdp2) SDMA-eligible All-Gather Patch

What this patches:
    Megatron's ``TorchFullyShardedDataParallel`` (enabled by
    ``use_torch_fsdp2``) wraps submodules with PyTorch FSDP2
    ``torch.distributed.fsdp.fully_shard`` -- the same primitive the
    TorchTitan backend uses. This patch wraps the ``fully_shard``
    symbol bound inside
    ``megatron.core.distributed.torch_fully_sharded_data_parallel`` so
    that every fully-sharded module also gets
    ``set_custom_all_gather(SymmMemAllGather(...))`` attached, routing
    FSDP's all-gather through a cuMem-backed symmetric buffer that
    NCCL/RCCL dispatches on the copy-engine (SDMA) path.

    This is the Megatron counterpart of
    ``primus/backends/torchtitan/patches/sdma_symm_mem_collectives.py``
    and reuses the same FSDP2 attach mechanism. Reduce-scatter is left
    on its FSDP default (out of scope).

Scope:
    Only applies to the ``use_torch_fsdp2`` path (which Primus's
    Megatron dense pretrain uses on MI300X/MI355X). It does NOT touch:
      - Megatron-FSDP (``use_megatron_fsdp``), which has its own native
        NCCL user-buffer registration (``--use-nccl-ub``);
      - the classic distributed optimizer param all-gather;
      - LoRA/PEFT (megatron_bridge), which is tensor-parallel and has no
        FSDP all-gather.

Activation:
    Export ``SDMA_ALL_GATHER=1`` (the same trigger as the TorchTitan
    patch) AND run with ``use_torch_fsdp2: true``. No-op otherwise. The
    companion hook ``runner/helpers/hooks/06_enable_sdma_all_gather.sh``
    exports the zero-CTA env (``NCCL_CTA_POLICY=2``, ...) and the
    LD_PRELOAD interposer so no YAML changes are required to opt in.

Requirements:
    - PyTorch >= 2.12 (introduces ``SymmMemAllGather``).
    - On ROCm, the RCCL transport must be able to take the copy-engine
      path; FSDP already requests zero-CTA via ``pg_options``.
"""

import functools
import os

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


def _sdma_all_gather_enabled(ctx: PatchContext) -> bool:
    """Gate on SDMA_ALL_GATHER=1 AND the use_torch_fsdp2 path."""
    if os.environ.get("SDMA_ALL_GATHER", "0") != "1":
        return False
    return getattr(get_args(ctx), "use_torch_fsdp2", False)


@register_patch(
    "megatron.fsdp.sdma_symm_mem_collectives",
    backend="megatron",
    phase="before_train",
    description=(
        "Auto-attach SymmMemAllGather to every FSDP2 fully_shard'd "
        "module in Megatron's TorchFullyShardedDataParallel so the "
        "all-gather uses the SDMA (copy-engine) dispatch path. Gated on "
        "SDMA_ALL_GATHER=1 and use_torch_fsdp2."
    ),
    condition=_sdma_all_gather_enabled,
)
def patch_megatron_fsdp_sdma_symm_mem(ctx: PatchContext) -> None:
    """
    Wrap ``megatron.core.distributed.torch_fully_sharded_data_parallel.fully_shard``
    so post-construction we attach ``SymmMemAllGather`` on the FSDP
    module. Reduce-scatter is left on its FSDP default.
    """
    import megatron.core.distributed.torch_fully_sharded_data_parallel as _mfsdp_mod

    try:
        from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
            SymmMemAllGather,
        )
    except ImportError as e:
        warning_rank_0(
            f"[Patch:megatron.fsdp.sdma_symm_mem_collectives] SymmMemAllGather "
            f"not available (needs PyTorch >= 2.12); skipping: {e}"
        )
        return

    if not getattr(_mfsdp_mod, "HAVE_FSDP", False) or not hasattr(_mfsdp_mod, "fully_shard"):
        warning_rank_0(
            "[Patch:megatron.fsdp.sdma_symm_mem_collectives] "
            "torch_fully_sharded_data_parallel.fully_shard not found; skipping."
        )
        return

    backend = "NCCL"
    orig_fully_shard = _mfsdp_mod.fully_shard

    def _attach_symm_mem_all_gather(fsdp_module) -> None:
        """Switch a fully_shard'd module's AG comm to the SymmMem flavor."""
        try:
            state = fsdp_module._get_fsdp_state()
        except Exception:
            return

        groups = getattr(state, "_fsdp_param_groups", None) or []
        if len(groups) != 1:
            # set_custom_all_gather rejects multi-group modules; leave them.
            return

        pg = groups[0]._all_gather_process_group
        try:
            fsdp_module.set_custom_all_gather(SymmMemAllGather(pg, backend))
        except (AttributeError, ValueError, AssertionError) as e:
            warning_rank_0(
                f"[Patch:megatron.fsdp.sdma_symm_mem_collectives] WARN: failed to "
                f"attach SymmMemAllGather to {type(fsdp_module).__name__}: {e}"
            )

    @functools.wraps(orig_fully_shard)
    def wrapped_fully_shard(module, *args, **kwargs):
        result = orig_fully_shard(module, *args, **kwargs)
        # fully_shard mutates `module` in place and returns it; fall back
        # to `module` if a torch version returns None.
        _attach_symm_mem_all_gather(result if result is not None else module)
        return result

    _mfsdp_mod.fully_shard = wrapped_fully_shard

    log_rank_0(
        "[Patch:megatron.fsdp.sdma_symm_mem_collectives] installed: "
        "every FSDP2 fully_shard()-d module in Megatron's "
        f"TorchFullyShardedDataParallel will use SymmMemAllGather (backend={backend})."
    )
