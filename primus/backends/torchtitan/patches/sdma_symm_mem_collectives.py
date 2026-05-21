###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan / FSDP2 SDMA-eligible Collectives Patch

What this patches:
    Wraps ``torch.distributed.fsdp.fully_shard`` so that every fully-sharded
    module also gets ``set_custom_all_gather(SymmMemAllGather(...))`` and
    ``set_custom_reduce_scatter(SymmMemReduceScatter(...))`` attached
    immediately after construction.

Why:
    By default FSDP2's ``DefaultAllGather.allocate(...)`` is just
    ``torch.empty(...)``, which goes through the PyTorch caching allocator
    and ends up as a ``hipMalloc``-backed VA. With a non-cuMem buffer, RCCL
    cannot take the SDMA fast path (``__amd_rocclr_batchMemOp.kd`` /
    ``hsa_amd_memory_async_batch_copy``) and falls back to
    ``ncclDevKernel_Generic_2`` running on CUs.

    PyTorch >= 2.12 ships ``SymmMemAllGather`` /
    ``SymmMemReduceScatter`` in
    ``torch.distributed.fsdp._fully_shard._fsdp_collectives`` -- their
    ``allocate(...)`` uses ``symm_mem.get_mem_pool(device)``, so the AG
    output buffer is cuMem-backed and RCCL lands on the SDMA dispatch
    path. PyTorch's own source comment on ``SymmMemAllGather`` notes:

        # Calling regular all-gather would already cause libraries like NCCL to
        # use its optimized all-gather implementation for symmetric memory:
        #   - Copy Engine All-Gather (when zero-CTA policy is enabled)
        #   - Symmetric Kernel All-Gather (when zero-CTA policy is not enabled)

    This patch wires those classes into every TorchTitan fully_shard'd
    module automatically, with no torchtitan-side changes required.

Verified end-to-end with the FSDP-like probe under rocprof
(see ``debug/fsdp_like_ag_probe.py --mode fsdp_forward_symm``):
``hsa_amd_memory_async_batch_copy`` count goes from 0 (default
``fsdp_forward``) to 192 across 8 ranks (3 iters * 8 layers * 8 ranks)
with the symm-mem wiring in place.

Activation:
    Set ``primus_sdma.enable_symm_mem_collectives: true`` in the pretrain
    module config (or pass it on the CLI). The patch is otherwise a no-op.

Caveats observed in the probe:
    1. On process teardown, ``destroy_process_group`` may emit
       ``c10::DistBackendError: NCCL communicator was aborted`` on each
       rank. Training/forward/backward/step run correctly; the symm_mem
       mempool is likely not drained before PG destroy. Cosmetic.
    2. Requires the underlying RCCL transport to actually pick
       ``P2P/CUMEMCUMEM`` channels with ``cta_policy=NCCL_CTA_POLICY_ZERO``
       (set automatically by FSDP via ``pg_options``) -- i.e. the standard
       CE env: ``NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1``. ``NCCL_LOCAL_REGISTER``
       can be anything; with this RCCL/HIP build, leave it at 0 to avoid
       the unrelated ``hipMemRetainAllocationHandle`` SIGSEGV (see the
       parent repo README section ``(1b)``).
"""

from primus.core.patches import PatchContext, get_param, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "torchtitan.fsdp.sdma_symm_mem_collectives",
    backend="torchtitan",
    phase="setup",
    description=(
        "Auto-attach SymmMemAllGather/SymmMemReduceScatter to every "
        "fully_shard'd module so FSDP collectives use the SDMA dispatch path."
    ),
    condition=lambda ctx: get_param(
        ctx, "primus_sdma.enable_symm_mem_collectives", False
    ),
)
def patch_torchtitan_fsdp_sdma_symm_mem(ctx: PatchContext) -> None:
    """
    Wrap ``torch.distributed.fsdp.fully_shard`` so post-construction we
    attach ``SymmMemAllGather`` / ``SymmMemReduceScatter`` on the FSDP
    module, routing every all-gather / reduce-scatter through a
    cuMem-backed symm_mem buffer that RCCL recognizes as eligible for
    its SDMA dispatch path (``__amd_rocclr_batchMemOp.kd``).
    """
    import functools

    import torch.distributed as dist
    import torch.distributed.fsdp as _fsdp_pkg
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _ffsc
    from torch.distributed.fsdp._fully_shard import _fully_shard as _ffs_mod
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
        SymmMemAllGather,
        SymmMemReduceScatter,
    )

    # Optional config knobs (with safe defaults)
    backend = get_param(ctx, "primus_sdma.symm_mem_backend", "NCCL")
    log_all = get_param(ctx, "primus_sdma.log_per_module_attach", False)
    # NOTE: We deliberately use the upstream SymmMemAllGather /
    # SymmMemReduceScatter classes verbatim here. Earlier iterations
    # (v2-v6) experimented with Python-side caching of symm_mem.rendezvous
    # and per-FSDP-unit buffer pools, but on this PyTorch 2.12 / RCCL
    # build they either (a) saved no measurable time (rendezvous is
    # already C++ side cached), (b) corrupted training (class-level
    # buffer-cache aliased FSDP layers' AG outputs), or (c) raced the
    # collective ordering (pre-rendezvous in allocate() SIGSEGV'd inside
    # symm_mem C++ when allocate ordering jittered across ranks). The
    # safest known-correct configuration is the upstream classes.

    orig_fully_shard = _fsdp_pkg.fully_shard

    def _attach_symm_mem_comms(fsdp_module) -> None:
        """Switch a fully_shard'd module's AG/RS comms to the SymmMem flavor."""
        try:
            state = fsdp_module._get_fsdp_state()
        except Exception as e:
            if log_all:
                log_rank_0(
                    f"[Patch:sdma_symm_mem] skip (no _fsdp_state): "
                    f"{type(fsdp_module).__name__}: {e}"
                )
            return

        groups = getattr(state, "_fsdp_param_groups", None) or []
        if len(groups) != 1:
            # set_custom_all_gather rejects modules with multiple param groups
            # (e.g. per-param mesh via shard_placement_fn). Leave those alone;
            # they will use whatever comm FSDP chose.
            if log_all:
                log_rank_0(
                    f"[Patch:sdma_symm_mem] skip multi-group module "
                    f"({len(groups)} groups): {type(fsdp_module).__name__}"
                )
            return

        pg = groups[0]._all_gather_process_group
        try:
            fsdp_module.set_custom_all_gather(SymmMemAllGather(pg, backend))
            fsdp_module.set_custom_reduce_scatter(SymmMemReduceScatter(pg, backend))
        except (AttributeError, ValueError, AssertionError) as e:
            log_rank_0(
                f"[Patch:sdma_symm_mem] WARN: failed to attach symm_mem comms "
                f"to {type(fsdp_module).__name__}: {e}"
            )
            return
        if log_all:
            log_rank_0(
                f"[Patch:sdma_symm_mem] attached SymmMemAllGather/ReduceScatter "
                f"to {type(fsdp_module).__name__} (group={pg.group_name})"
            )

    @functools.wraps(orig_fully_shard)
    def wrapped_fully_shard(module, *args, **kwargs):
        result = orig_fully_shard(module, *args, **kwargs)
        # fully_shard mutates `module` in place and returns it; this is the
        # FSDP-augmented module.
        _attach_symm_mem_comms(result)
        return result

    # Copy across any non-dunder attributes from the original function to the
    # wrapper. In particular, FSDP attaches `fully_shard.state` (a method)
    # that nested fully_shard()ing reads as `fully_shard.state(modules[0])`
    # in `_fully_shard.py`. If we don't propagate it, the first inner call
    # crashes with `AttributeError: 'function' object has no attribute 'state'`.
    for _attr in dir(orig_fully_shard):
        if _attr.startswith("__"):
            continue
        try:
            setattr(wrapped_fully_shard, _attr, getattr(orig_fully_shard, _attr))
        except (AttributeError, TypeError):
            pass

    # Patch every alias of fully_shard that torchtitan (or anyone else) may
    # have already pulled in via `from ... import fully_shard`. We patch the
    # package re-export AND the inner module's binding; both are the same
    # function object at this point, and torchtitan imports happen later (in
    # TorchTitanPretrainTrainer.__init__, after run_patches('setup') returns).
    _fsdp_pkg.fully_shard = wrapped_fully_shard
    if hasattr(_ffs_mod, "fully_shard"):
        _ffs_mod.fully_shard = wrapped_fully_shard

    log_rank_0(
        "[Patch:torchtitan.fsdp.sdma_symm_mem_collectives] "
        f"installed: every fully_shard()-d module will use "
        f"SymmMemAllGather/SymmMemReduceScatter (backend={backend}). "
        "Expect rocclr's hsa_amd_memory_async_batch_copy / "
        "__amd_rocclr_batchMemOp.kd to dispatch the AG/RS payload "
        "on the copy engines (SDMA) instead of ncclDevKernel_Generic_2 on CUs."
    )

    # Convenience: stash references for tests / debugging.
    _ffsc._primus_sdma_orig_fully_shard = orig_fully_shard
    _ffsc._primus_sdma_attach = _attach_symm_mem_comms
    _ = dist  # silence unused-import warnings when nothing else uses dist here
