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
    that each fully-sharded **transformer layer** gets
    ``set_custom_all_gather(CopyOutSymmAllGather(...))`` attached, routing
    its all-gather through a cuMem-backed symmetric buffer that NCCL/RCCL
    dispatches on the copy-engine (SDMA) path. All other units are left on
    FSDP's default all-gather (regular RCCL).

    This is the Megatron counterpart of
    ``primus/backends/torchtitan/patches/sdma_symm_mem_collectives.py``
    and reuses the same FSDP2 attach mechanism. Reduce-scatter is left
    on its FSDP default (out of scope).

Scope:
    Only applies to the ``use_torch_fsdp2`` path (which Primus's
    Megatron dense pretrain uses on MI300X/MI355X).

Activation:
    Export ``SDMA_ALL_GATHER=1`` AND run with ``use_torch_fsdp2: true``.
    No-op otherwise. The companion hook
    ``runner/helpers/hooks/06_enable_sdma_all_gather.sh`` exports the
    zero-CTA env (``NCCL_CTA_POLICY=2``, ...) and the LD_PRELOAD interposer
    so no YAML changes are required to opt in.

Forward-prefetch overlap:
    To hide the forward all-gather behind compute, the uniform
    transformer-layer units use ``CopyOutSymmAllGather`` (all-gather into a
    single reused, copy-engine-registered symmetric scratch, then copy out
    to FSDP's regular buffer). Returning a regular buffer restores FSDP2's
    *implicit* forward-prefetch double-buffering, which the SDMA
    single-window allocator otherwise defeats -- so no explicit
    ``set_modules_to_forward_prefetch`` is used (it adds no overlap and
    costs large amounts of memory). The other units (embedding / lm_head /
    rotary) stay on FSDP's default all-gather (regular RCCL).

Requirements:
    - PyTorch >= 2.12 (introduces ``SymmMemAllGather``).
    - On ROCm, the RCCL transport must be able to take the copy-engine
      path; FSDP already requests zero-CTA via ``pg_options``.
"""

import functools
import os

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0

# FSDP2 symmetric-memory collectives (PyTorch >= 2.12). Imported defensively so
# this module still loads on older PyTorch; the patch no-ops in that case.
try:
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import AllGather as _AllGather
except Exception:  # pragma: no cover - exercised only on older PyTorch
    torch = None
    dist = None
    symm_mem = None
    _AllGather = object


# --- copy-out all-gather for forward-prefetch overlap ----------------------
# See the "Forward-prefetch overlap" section of the module docstring. Used for
# the uniform transformer-layer units.
_BACKEND_SET = False
# Process-wide symmetric scratch, reused across all (uniform-size) layer
# gathers: allocated from the symmetric mempool on first use, then copied out
# of and reused, so one copy-engine-registered window serves every layer.
_COPYOUT_BUF = None


def _ensure_backend(backend: str = "NCCL") -> None:
    global _BACKEND_SET
    if _BACKEND_SET:
        return
    try:
        symm_mem.set_backend(backend)
    except RuntimeError:
        # Backend already set / already in use -- nothing to do.
        pass
    _BACKEND_SET = True


class CopyOutSymmAllGather(_AllGather):
    """All-gather into a reused symmetric scratch, then copy out to FSDP's
    regular buffer, so the symmetric window is freed for the next gather.

    The stock ``SymmMemAllGather`` all-gathers directly into the buffer FSDP
    holds for compute, so that single symmetric window stays live for the whole
    layer forward and FSDP cannot prefetch the next layer (a second concurrent
    symmetric window is not allowed by the symmetric-memory mempool). This comm
    decouples the two roles:

      * ``allocate()`` returns a *regular* buffer FSDP owns (no symmetric-window
        limit, so it may hold two across a prefetch boundary).
      * ``__call__`` all-gathers into a single, process-wide symmetric scratch
        and copies the result out into FSDP's regular buffer, freeing the
        scratch for the next (prefetched) layer's gather.
    """

    def __init__(self, group, backend: str = "NCCL"):
        self._group = group
        _ensure_backend(backend)

    def allocate(self, size, *, dtype, device):
        # Regular (non-symmetric) buffer; FSDP owns it and reads it for compute.
        return torch.empty(*size, dtype=dtype, device=device)

    def __call__(self, output_tensor, input_tensor, group, async_op: bool = False):
        global _COPYOUT_BUF
        n = output_tensor.numel()
        device = output_tensor.device
        # Allocate the CE-registered scratch once and reuse it for every layer
        # (all layers share one all-gather size); re-allocate only if a
        # different size/dtype ever appears.
        if (
            _COPYOUT_BUF is None
            or _COPYOUT_BUF.numel() != n
            or _COPYOUT_BUF.dtype != output_tensor.dtype
        ):
            mempool = symm_mem.get_mem_pool(device)
            with torch.cuda.use_mem_pool(mempool):
                _COPYOUT_BUF = torch.empty(n, dtype=output_tensor.dtype, device=device)
        buf = _COPYOUT_BUF
        # Rendezvous every call (cached after the first) so all ranks stay in
        # lockstep -- a rendezvous-once scheme desyncs ranks under prefetch.
        symm_mem.rendezvous(buf, group=group.group_name)
        world_size, rank = group.size(), group.rank()
        chunk = n // world_size
        # In-place all-gather on the scratch: the per-rank input slot is a view
        # of the registered window, so the copy-engine path is valid.
        buf_in = buf.narrow(0, rank * chunk, chunk)
        buf_in.copy_(input_tensor.view(-1))
        dist.all_gather_into_tensor(buf, buf_in, group=group, async_op=False)
        output_tensor.copy_(buf)
        return None


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
        "Route FSDP2 transformer-layer all-gathers through the SDMA "
        "(copy-engine) path. Gated on SDMA_ALL_GATHER=1 and use_torch_fsdp2. "
        "Transformer layers use CopyOutSymmAllGather (restores implicit "
        "forward prefetch to hide the all-gather); other units stay on "
        "FSDP's default all-gather (regular RCCL)."
    ),
    condition=_sdma_all_gather_enabled,
)
def patch_megatron_fsdp_sdma_symm_mem(ctx: PatchContext) -> None:
    """
    Wrap ``megatron.core.distributed.torch_fully_sharded_data_parallel.fully_shard``
    so post-construction we attach ``CopyOutSymmAllGather`` to the
    transformer-layer units (which restores FSDP2's implicit forward
    prefetch on the SDMA path). All other units stay on FSDP's default
    all-gather (regular RCCL). Reduce-scatter is left on its FSDP default.
    """
    import megatron.core.distributed.torch_fully_sharded_data_parallel as _mfsdp_mod

    if symm_mem is None or _AllGather is object:
        warning_rank_0(
            "[Patch:megatron.fsdp.sdma_symm_mem_collectives] FSDP2 symmetric-memory "
            "collectives not available (needs PyTorch >= 2.12); skipping."
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

    # Only the uniform transformer-layer units get the copy-out comm (which
    # restores implicit forward prefetch on the SDMA path). Everything else is
    # left on FSDP's default all-gather (regular RCCL). If the layer type can't
    # be resolved we attach nothing and the patch is a no-op.
    try:
        from megatron.core.transformer.transformer_layer import TransformerLayer
    except Exception as e:
        warning_rank_0(
            "[Patch:megatron.fsdp.sdma_symm_mem_collectives] could not import "
            f"TransformerLayer; leaving all units on regular RCCL all-gather: {e}"
        )
        TransformerLayer = ()

    def _attach_symm_mem_all_gather(fsdp_module) -> None:
        """Attach the copy-out SDMA all-gather to transformer-layer units."""
        # Only transformer layers are routed onto the SDMA copy-out path;
        # other units (embedding/lm_head/rotary) keep FSDP's default all-gather
        # (regular RCCL), which already overlaps via implicit prefetch.
        if not (bool(TransformerLayer) and isinstance(fsdp_module, TransformerLayer)):
            return
        try:
            state = fsdp_module._get_fsdp_state()
        except Exception:
            return

        groups = getattr(state, "_fsdp_param_groups", None) or []
        if len(groups) != 1:
            # set_custom_all_gather rejects multi-group modules; leave them.
            return

        pg = groups[0]._all_gather_process_group
        # CopyOutSymmAllGather all-gathers into one reused symmetric window and
        # copies out to a regular buffer, which restores FSDP2's *implicit*
        # forward-prefetch double-buffering (the SDMA single-window allocator
        # otherwise defeats it). No explicit set_modules_to_forward_prefetch is
        # needed -- it adds no overlap and costs large amounts of memory by
        # prefetching many layers ahead.
        try:
            fsdp_module.set_custom_all_gather(CopyOutSymmAllGather(pg, backend))
        except (AttributeError, ValueError, AssertionError) as e:
            warning_rank_0(
                f"[Patch:megatron.fsdp.sdma_symm_mem_collectives] WARN: failed to "
                f"attach CopyOutSymmAllGather to {type(fsdp_module).__name__}: {e}"
            )
            return

    @functools.wraps(orig_fully_shard)
    def wrapped_fully_shard(module, *args, **kwargs):
        result = orig_fully_shard(module, *args, **kwargs)
        # fully_shard mutates `module` in place and returns it; fall back
        # to `module` if a torch version returns None.
        _attach_symm_mem_all_gather(result if result is not None else module)
        return result

    _mfsdp_mod.fully_shard = wrapped_fully_shard

    mode = (
        "CopyOutSymmAllGather on transformer layers (regular RCCL elsewhere)"
        if TransformerLayer
        else "regular RCCL on every unit (TransformerLayer unavailable)"
    )
    log_rank_0(
        "[Patch:megatron.fsdp.sdma_symm_mem_collectives] installed: "
        f"FSDP2 fully_shard()-d modules will use {mode} (backend={backend})."
    )
