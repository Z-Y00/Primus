###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.distributed import (
    rccl_sdma_reduce_scatter,
)
from primus.backends.megatron.patches.parallelism import (
    rccl_sdma_reduce_scatter_patches,
)


def test_backend_selector(monkeypatch):
    monkeypatch.setenv(
        "MEGATRON_GRAD_REDUCE_BACKEND",
        "rccl_sdma_a2a",
    )
    assert rccl_sdma_reduce_scatter_patches.rccl_sdma_reduce_scatter_enabled()


@pytest.mark.parametrize(
    ("staging_bytes", "element_size", "world_size", "expected"),
    [
        (128, 2, 8, 64),
        (130, 2, 8, 64),
        (256, 4, 4, 64),
    ],
)
def test_staging_numel_rounds_to_rank_multiple(
    staging_bytes,
    element_size,
    world_size,
    expected,
):
    assert (
        rccl_sdma_reduce_scatter._staging_numel(
            staging_bytes,
            element_size,
            world_size,
        )
        == expected
    )


def test_full_world_group_reuses_param_sdma_group(monkeypatch):
    original_group = SimpleNamespace()
    dedicated_group = SimpleNamespace()
    monkeypatch.setattr(
        rccl_sdma_reduce_scatter.dist,
        "get_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        rccl_sdma_reduce_scatter.dist,
        "get_process_group_ranks",
        lambda _group: [0, 1, 2, 3],
    )
    monkeypatch.setattr(
        "primus.backends.megatron.core.distributed.rccl_sdma_param_gather.get_sdma_process_group",
        lambda group: dedicated_group if group is original_group else None,
    )
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    result = rccl_sdma_reduce_scatter.get_sdma_process_group(original_group)

    assert result is dedicated_group


def test_single_node_validation_requires_local_world_size(monkeypatch):
    monkeypatch.setattr(
        rccl_sdma_reduce_scatter.dist,
        "get_world_size",
        lambda: 4,
    )
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)

    with pytest.raises(RuntimeError, match="requires LOCAL_WORLD_SIZE"):
        rccl_sdma_reduce_scatter._require_single_node()


def test_workspace_rejects_unsupported_gradient_dtype():
    with pytest.raises(RuntimeError, match="supports BF16 and FP16"):
        rccl_sdma_reduce_scatter.prepare_workspace(
            SimpleNamespace(),
            torch.zeros(8, dtype=torch.float32),
        )


def test_prepare_workspace_reuses_storage_across_gradient_buckets(monkeypatch):
    rccl_sdma_reduce_scatter.reset_runtime_state_for_tests()
    group = SimpleNamespace(
        group_name="test-group",
        size=lambda: 2,
        rank=lambda: 0,
    )
    created = []

    def fake_workspace(group, device, dtype, staging_numel, direct_input):
        workspace = SimpleNamespace(
            group=group,
            device=device,
            dtype=dtype,
            staging_numel=staging_numel,
            direct_input=direct_input,
            busy=False,
            active_work=None,
            workspace_bytes=0,
        )
        created.append(workspace)
        return workspace

    monkeypatch.setenv("MEGATRON_RCCL_SDMA_RS_STAGING_BYTES", "128")
    monkeypatch.setenv("MEGATRON_RCCL_SDMA_RS_WORKSPACE_DEPTH", "2")
    monkeypatch.setattr(
        rccl_sdma_reduce_scatter,
        "get_sdma_process_group",
        lambda _group: group,
    )
    monkeypatch.setattr(
        rccl_sdma_reduce_scatter.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        rccl_sdma_reduce_scatter,
        "ReduceScatterWorkspace",
        fake_workspace,
    )
    first = torch.zeros(64, dtype=torch.bfloat16)
    second = torch.zeros(64, dtype=torch.bfloat16)

    first_pool = rccl_sdma_reduce_scatter.prepare_workspace(group, first)
    second_pool = rccl_sdma_reduce_scatter.prepare_workspace(group, second)

    assert first_pool is second_pool
    assert first_pool.depth == 2
    assert len(created) == 2


def test_workspace_ring_excludes_caller_and_sibling_streams(monkeypatch):
    class FakeStream:
        def __init__(self, handle):
            self.cuda_stream = handle

    candidates = iter(
        [
            FakeStream(1),
            FakeStream(2),
            FakeStream(2),
            FakeStream(3),
        ]
    )
    monkeypatch.setattr(
        rccl_sdma_reduce_scatter.torch.cuda,
        "Stream",
        lambda device: next(candidates),
    )
    pool = rccl_sdma_reduce_scatter.ReduceScatterWorkspacePool.__new__(
        rccl_sdma_reduce_scatter.ReduceScatterWorkspacePool
    )
    pool.device = torch.device("cuda", 0)
    pool.depth = 2
    pool.next_index = 0
    pool.workspaces = [
        SimpleNamespace(stream=None),
        SimpleNamespace(stream=None),
    ]
    caller = FakeStream(1)

    first = pool.acquire(caller)
    second = pool.acquire(caller)

    assert first.stream.cuda_stream == 2
    assert second.stream.cuda_stream == 3


def _make_cpu_workspace(monkeypatch):
    activity = []

    class FakeAllToAllWork:
        def wait(self):
            activity.append("wait")
            return True

    def fake_all_to_all(output, input, group, async_op):
        del group
        assert async_op
        output.copy_(input)
        activity.append("launch")
        return FakeAllToAllWork()

    monkeypatch.setattr(
        rccl_sdma_reduce_scatter.dist,
        "all_to_all_single",
        fake_all_to_all,
    )

    workspace = rccl_sdma_reduce_scatter.ReduceScatterWorkspace.__new__(
        rccl_sdma_reduce_scatter.ReduceScatterWorkspace
    )
    workspace.group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    workspace.device = torch.device("cpu")
    workspace.dtype = torch.bfloat16
    workspace.world_size = 2
    workspace.staging_numel = 8
    workspace.chunk_numel = 4
    workspace.direct_input = False
    workspace.send = [
        torch.empty(8, dtype=torch.bfloat16),
        torch.empty(8, dtype=torch.bfloat16),
    ]
    workspace.recv = [
        torch.empty(8, dtype=torch.bfloat16),
        torch.empty(8, dtype=torch.bfloat16),
    ]
    workspace.reduced = [
        torch.empty(4, dtype=torch.float32),
        torch.empty(4, dtype=torch.float32),
    ]
    workspace.busy = False
    workspace.active_work = None
    return workspace, activity


@pytest.mark.parametrize(
    ("op", "divisor"),
    [
        (torch.distributed.ReduceOp.SUM, 1.0),
        (torch.distributed.ReduceOp.AVG, 2.0),
    ],
)
def test_chunked_fp32_reduce_writes_original_output(
    monkeypatch,
    op,
    divisor,
):
    workspace, activity = _make_cpu_workspace(monkeypatch)
    input_tensor = torch.tensor(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
        ],
        dtype=torch.bfloat16,
    )
    expected = (
        input_tensor.view(2, 6).float().sum(dim=0) / divisor
    ).bfloat16()
    output = input_tensor[:6]

    work = rccl_sdma_reduce_scatter.ReduceScatterWork(
        workspace,
        output,
        input_tensor,
        op,
    )
    work.wait()

    assert torch.equal(output, expected)
    assert work.is_completed()
    assert activity == ["launch", "launch", "wait", "wait"]


def test_non_pipelined_mode_reduces_before_next_launch(monkeypatch):
    monkeypatch.setenv("MEGATRON_RCCL_SDMA_RS_PIPELINE", "0")
    workspace, activity = _make_cpu_workspace(monkeypatch)
    input_tensor = torch.arange(12, dtype=torch.bfloat16)
    output = torch.empty(6, dtype=torch.bfloat16)

    work = rccl_sdma_reduce_scatter.ReduceScatterWork(
        workspace,
        output,
        input_tensor,
        torch.distributed.ReduceOp.SUM,
    )
    work.wait()

    assert work.is_completed()
    assert activity == ["launch", "wait", "launch", "wait"]


def test_direct_input_mode_skips_send_staging(monkeypatch):
    workspace, activity = _make_cpu_workspace(monkeypatch)
    workspace.direct_input = True
    workspace.staging_numel = 12
    workspace.chunk_numel = 6
    workspace.send = []
    workspace.recv = [torch.empty(12, dtype=torch.bfloat16)]
    workspace.reduced = [torch.empty(6, dtype=torch.float32)]
    input_tensor = torch.arange(12, dtype=torch.bfloat16)
    output = torch.empty(6, dtype=torch.bfloat16)
    expected = input_tensor.view(2, 6).float().sum(dim=0).bfloat16()

    work = rccl_sdma_reduce_scatter.ReduceScatterWork(
        workspace,
        output,
        input_tensor,
        torch.distributed.ReduceOp.SUM,
    )
    work.wait()

    assert torch.equal(output, expected)
    assert activity == ["launch", "wait"]


def test_final_direct_input_bucket_uses_native_reduce_scatter(monkeypatch):
    rccl_sdma_reduce_scatter.reset_runtime_state_for_tests()
    monkeypatch.setenv("MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL", "1")
    workspace_pool = SimpleNamespace(
        direct_input=True,
        staging_numel=12,
    )
    monkeypatch.setattr(
        rccl_sdma_reduce_scatter,
        "prepare_workspace",
        lambda _group, _input: workspace_pool,
    )
    native_group = SimpleNamespace()
    monkeypatch.setattr(
        rccl_sdma_reduce_scatter,
        "get_native_tail_process_group",
        lambda _group: native_group,
    )
    calls = []
    expected_work = object()

    def fake_reduce_scatter(
        output,
        input,
        op,
        group,
        async_op,
    ):
        calls.append((output, input, op, group, async_op))
        return expected_work

    monkeypatch.setattr(
        rccl_sdma_reduce_scatter.dist,
        "reduce_scatter_tensor",
        fake_reduce_scatter,
    )
    group = SimpleNamespace(size=lambda: 2)
    input_tensor = torch.zeros(8, dtype=torch.bfloat16)
    output_tensor = torch.zeros(4, dtype=torch.bfloat16)
    rccl_sdma_reduce_scatter.register_gradient_bucket(
        SimpleNamespace(grad_data=input_tensor, bucket_id=25)
    )

    work = rccl_sdma_reduce_scatter.reduce_scatter(
        output_tensor,
        input_tensor,
        torch.distributed.ReduceOp.SUM,
        group,
        async_op=True,
    )

    assert work is expected_work
    assert calls == [
        (
            output_tensor,
            input_tensor,
            torch.distributed.ReduceOp.SUM,
            native_group,
            True,
        )
    ]


def test_native_tail_selects_last_two_bucket_ids(monkeypatch):
    rccl_sdma_reduce_scatter.reset_runtime_state_for_tests()
    monkeypatch.setenv("MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL", "1")
    workspace_pool = SimpleNamespace(direct_input=True)
    inputs = [
        torch.zeros(8, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    for bucket_id, input_tensor in zip((23, 24, 25), inputs):
        rccl_sdma_reduce_scatter.register_gradient_bucket(
            SimpleNamespace(
                grad_data=input_tensor,
                bucket_id=bucket_id,
            )
        )

    assert not rccl_sdma_reduce_scatter._use_native_tail(
        workspace_pool,
        inputs[0],
    )
    assert rccl_sdma_reduce_scatter._use_native_tail(
        workspace_pool,
        inputs[1],
    )
    assert rccl_sdma_reduce_scatter._use_native_tail(
        workspace_pool,
        inputs[2],
    )


def test_shared_workspace_eagerly_schedules_next_bucket(monkeypatch):
    workspace, activity = _make_cpu_workspace(monkeypatch)
    workspace_pool = rccl_sdma_reduce_scatter.ReduceScatterWorkspacePool.__new__(
        rccl_sdma_reduce_scatter.ReduceScatterWorkspacePool
    )
    workspace_pool.depth = 1
    workspace_pool.workspaces = [workspace]
    workspace_pool.next_index = 0
    group = SimpleNamespace(size=lambda: 2)
    first_input = torch.arange(12, dtype=torch.bfloat16)
    first_output = torch.empty(6, dtype=torch.bfloat16)
    first_work = rccl_sdma_reduce_scatter.ReduceScatterWork(
        workspace,
        first_output,
        first_input,
        torch.distributed.ReduceOp.SUM,
    )
    workspace.active_work = first_work
    monkeypatch.setattr(
        rccl_sdma_reduce_scatter,
        "prepare_workspace",
        lambda _group, _input: workspace_pool,
    )

    second_input = torch.arange(12, dtype=torch.bfloat16)
    second_output = torch.empty(6, dtype=torch.bfloat16)
    second_work = rccl_sdma_reduce_scatter.reduce_scatter(
        second_output,
        second_input,
        torch.distributed.ReduceOp.SUM,
        group,
        async_op=True,
    )

    assert first_work.is_completed()
    assert workspace.active_work is second_work
    assert activity == [
        "launch",
        "launch",
        "wait",
        "wait",
        "launch",
        "launch",
        "wait",
        "wait",
    ]


def test_bucket_group_init_enables_existing_megatron_path(monkeypatch):
    workspace_calls = []
    monkeypatch.setattr(
        rccl_sdma_reduce_scatter,
        "prepare_workspace",
        lambda *args: workspace_calls.append(args),
    )
    group = SimpleNamespace()
    grad_data = torch.zeros(8, dtype=torch.bfloat16)

    def original(self, buckets, ddp_config, collective_group, collective_group_size):
        del collective_group_size
        self.buckets = buckets
        self.ddp_config = ddp_config
        self.collective_group = collective_group

    wrapped = rccl_sdma_reduce_scatter_patches.make_bucket_group_init(original)
    bucket_group = SimpleNamespace()
    ddp_config = SimpleNamespace(
        use_distributed_optimizer=True,
        num_distributed_optimizer_instances=1,
        reduce_scatter_with_fp32_accumulation=False,
    )
    wrapped(
        bucket_group,
        [SimpleNamespace(grad_data=grad_data, bucket_id=0)],
        ddp_config,
        group,
        2,
    )

    assert ddp_config.reduce_scatter_with_fp32_accumulation
    assert workspace_calls == [(group, grad_data)]


def test_bucket_group_init_rejects_multi_instance_optimizer():
    def original(
        self,
        buckets,
        ddp_config,
        collective_group,
        collective_group_size,
    ):
        del self, buckets, ddp_config, collective_group, collective_group_size
        pytest.fail("unsupported configuration reached Megatron")

    wrapped = rccl_sdma_reduce_scatter_patches.make_bucket_group_init(original)
    with pytest.raises(RuntimeError, match="multiple distributed-optimizer"):
        wrapped(
            SimpleNamespace(),
            [SimpleNamespace(grad_data=torch.zeros(8, dtype=torch.bfloat16))],
            SimpleNamespace(
                use_distributed_optimizer=True,
                num_distributed_optimizer_instances=2,
            ),
            SimpleNamespace(),
            2,
        )
