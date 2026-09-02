###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.distributed import rccl_sdma_param_gather
from primus.backends.megatron.patches.parallelism import (
    rccl_sdma_param_all_gather_patches,
    sdma_param_all_gather_patches,
)


def test_chunk_size_accounts_for_every_rank():
    chunk_numel = rccl_sdma_param_gather.max_chunk_numel(
        capacity_bytes=512 * 1024 * 1024,
        element_size=2,
        world_size=8,
    )

    assert chunk_numel == 32 * 1024 * 1024
    assert rccl_sdma_param_gather.chunk_count(chunk_numel, chunk_numel) == 1
    assert rccl_sdma_param_gather.chunk_count(chunk_numel + 1, chunk_numel) == 2


@pytest.mark.parametrize(
    ("capacity_bytes", "element_size", "world_size"),
    [(0, 2, 8), (1024, 0, 8), (1024, 2, 0)],
)
def test_chunk_size_rejects_invalid_dimensions(
    capacity_bytes,
    element_size,
    world_size,
):
    with pytest.raises(ValueError):
        rccl_sdma_param_gather.max_chunk_numel(
            capacity_bytes,
            element_size,
            world_size,
        )


def test_event_work_wait_is_device_side_and_idempotent(monkeypatch):
    waited_events = []

    class ConsumerStream:
        def wait_event(self, event):
            waited_events.append(event)

    event = SimpleNamespace()
    monkeypatch.setattr(
        rccl_sdma_param_gather.torch.cuda,
        "current_stream",
        lambda _device: ConsumerStream(),
    )
    work = rccl_sdma_param_gather.EventWork(
        event,
        torch.device("cuda", 0),
    )

    assert work.wait()
    assert work.wait()
    assert work.is_completed()
    assert waited_events == [event]


def test_rccl_backend_disables_direct_hip_patch(monkeypatch):
    monkeypatch.setenv("ENABLE_SDMA_ALLGATHER", "1")
    monkeypatch.setenv("MEGATRON_PARAM_GATHER_BACKEND", "rccl_sdma")

    assert rccl_sdma_param_all_gather_patches.rccl_sdma_param_gather_enabled()
    assert not sdma_param_all_gather_patches._sdma_allgather_enabled(None)


def test_direct_hip_patch_remains_available_without_rccl_backend(monkeypatch):
    monkeypatch.setenv("ENABLE_SDMA_ALLGATHER", "1")
    monkeypatch.delenv("MEGATRON_PARAM_GATHER_BACKEND", raising=False)

    assert not rccl_sdma_param_all_gather_patches.rccl_sdma_param_gather_enabled()
    assert sdma_param_all_gather_patches._sdma_allgather_enabled(None)


def test_direct_rccl_gather_is_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("MEGATRON_PARAM_GATHER_BACKEND", "rccl_sdma")
    monkeypatch.setenv("MEGATRON_RCCL_SDMA_DIRECT", "1")

    assert rccl_sdma_param_all_gather_patches.direct_param_gather_enabled()


def test_direct_buffer_marker_applies_to_views():
    tensor = SimpleNamespace()

    rccl_sdma_param_gather.mark_direct_param_buffer(tensor)

    assert rccl_sdma_param_gather.is_direct_param_buffer(tensor)


def test_param_buffer_wrapper_rendezvouses_and_marks_buckets(monkeypatch):
    import megatron.core.distributed.param_and_grad_buffer as pgb

    group = SimpleNamespace(group_name="ce", rank=lambda: 1)
    pool = SimpleNamespace()
    handle = SimpleNamespace()
    param_data = SimpleNamespace()
    bucket_data = SimpleNamespace()

    monkeypatch.setattr(pgb, "is_mxfp8tensor", lambda _param: False)
    monkeypatch.setattr(
        rccl_sdma_param_gather,
        "prepare_direct_param_buffer_pool",
        lambda _group, _device: (group, pool),
    )
    monkeypatch.setattr(
        rccl_sdma_param_gather,
        "rendezvous_direct_param_buffer",
        lambda tensor, _group: (rccl_sdma_param_gather.mark_direct_param_buffer(tensor) or handle),
    )
    monkeypatch.setattr(torch.cuda, "use_mem_pool", lambda _pool: nullcontext())

    def original(
        self,
        ddp_config,
        param_dtype,
        grad_dtype,
        params,
        data_parallel_group,
        bucket_size,
        param_to_name,
        gradient_scaling_factor,
        param_indices,
        nccl_ub,
        pg_collection=None,
    ):
        del (
            ddp_config,
            param_dtype,
            grad_dtype,
            params,
            data_parallel_group,
            bucket_size,
            param_to_name,
            gradient_scaling_factor,
            param_indices,
            nccl_ub,
            pg_collection,
        )
        self.param_data = param_data
        self.buckets = [SimpleNamespace(param_data=bucket_data)]

    wrapped = rccl_sdma_param_all_gather_patches.make_param_and_grad_buffer_init(original)
    buffer = SimpleNamespace()
    wrapped(
        buffer,
        SimpleNamespace(use_distributed_optimizer=True),
        torch.bfloat16,
        torch.float32,
        [SimpleNamespace(device=torch.device("cuda", 0))],
        SimpleNamespace(),
        1024,
        {},
        1.0,
        [0],
        False,
    )

    assert buffer._primus_rccl_sdma_pool is pool
    assert buffer._primus_rccl_sdma_symmetric_memory is handle
    assert rccl_sdma_param_gather.is_direct_param_buffer(param_data)
    assert rccl_sdma_param_gather.is_direct_param_buffer(bucket_data)


def test_global_cta_policy_is_rejected_for_megatron_backend(monkeypatch):
    monkeypatch.setenv("NCCL_CTA_POLICY", "2")

    with pytest.raises(RuntimeError, match="requires NCCL_CTA_POLICY to be unset"):
        rccl_sdma_param_all_gather_patches.validate_global_cta_policy()


def test_dedicated_group_gets_zero_cta_without_mutating_original(monkeypatch):
    class OriginalGroup:
        def __init__(self):
            self.policy = "unchanged"

        def size(self):
            return 8

    class FakeOptions:
        def __init__(self):
            self.config = SimpleNamespace(cta_policy=None, split_share=None)

    class FakeProcessGroupNCCL:
        Options = FakeOptions
        NCCL_CTA_POLICY_ZERO = 2
        NCCL_CTA_POLICY_DEFAULT = 0

    captured = {}
    dedicated_group = SimpleNamespace(group_name="dedicated_ce_group")

    def fake_new_group(**kwargs):
        captured.update(kwargs)
        return dedicated_group

    original_group = OriginalGroup()
    monkeypatch.setattr(
        rccl_sdma_param_gather.dist,
        "ProcessGroupNCCL",
        FakeProcessGroupNCCL,
    )
    monkeypatch.setattr(rccl_sdma_param_gather.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(rccl_sdma_param_gather.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(rccl_sdma_param_gather.dist, "new_group", fake_new_group)
    monkeypatch.delenv("MEGATRON_RCCL_SDMA_CTA_POLICY", raising=False)
    rccl_sdma_param_gather.reset_runtime_state_for_tests()

    try:
        result = rccl_sdma_param_gather.get_sdma_process_group(original_group)
    finally:
        rccl_sdma_param_gather.reset_runtime_state_for_tests()

    assert result is dedicated_group
    assert captured["ranks"] == list(range(8))
    assert captured["backend"] == "nccl"
    assert captured["pg_options"].config.cta_policy == 2
    assert captured["pg_options"].config.split_share == 0
    assert "PARAM_GATHER_POLICY_2" in captured["group_desc"]
    assert original_group.policy == "unchanged"


def _run_sdma_hook(extra_env):
    hook = Path(__file__).resolve().parents[3] / "runner/helpers/hooks/06_enable_sdma_all_gather.sh"
    env = os.environ.copy()
    for name in (
        "FSDP_ALL_GATHER_BACKEND",
        "MEGATRON_PARAM_GATHER_BACKEND",
        "NCCL_CTA_POLICY",
        "NCCL_CUMEM_ENABLE",
    ):
        env.pop(name, None)
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(hook)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_megatron_hook_enables_cumem_without_global_cta_policy():
    result = _run_sdma_hook(
        {"MEGATRON_PARAM_GATHER_BACKEND": "rccl_sdma"},
    )

    assert result.returncode == 0
    assert "env.MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma" in result.stdout
    assert "env.NCCL_CTA_POLICY" not in result.stdout
    assert "env.NCCL_CUMEM_ENABLE=1" in result.stdout


def test_fsdp_hook_keeps_global_cumem_and_cta_policy():
    result = _run_sdma_hook(
        {"FSDP_ALL_GATHER_BACKEND": "rccl_sdma"},
    )

    assert result.returncode == 0
    assert "env.NCCL_CUMEM_ENABLE=1" in result.stdout
    assert "env.NCCL_CTA_POLICY=2" in result.stdout


def test_hook_rejects_combined_fsdp_and_megatron_backends():
    result = _run_sdma_hook(
        {
            "FSDP_ALL_GATHER_BACKEND": "rccl_sdma",
            "MEGATRON_PARAM_GATHER_BACKEND": "rccl_sdma",
        },
    )

    assert result.returncode == 2
    assert "cannot be enabled together" in result.stderr
