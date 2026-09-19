###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
from torch.distributed.fsdp._fully_shard import _fsdp_collectives

from primus.backends.diffusion.trainers.fsdp2 import FSDP2Trainer


class FakeSymmMemAllGather:
    def __init__(self, group, backend):
        self.group = group
        self.backend = backend


def make_trainer():
    """Minimal stand-in for the trainer, carrying the real gate method."""
    trainer = SimpleNamespace(rank=0)
    trainer._sdma_all_gather_enabled = lambda: FSDP2Trainer._sdma_all_gather_enabled(trainer)
    return trainer


def make_module(attached, *, groups=None):
    """A stand-in for a fully_shard'd module exposing just what the attach reads."""
    if groups is None:
        groups = [SimpleNamespace(_all_gather_process_group=SimpleNamespace(group_name="test-group"))]

    module = SimpleNamespace()
    module._get_fsdp_state = lambda: SimpleNamespace(_fsdp_param_groups=groups)
    module.set_custom_all_gather = attached.append
    return module


@pytest.fixture(autouse=True)
def fake_symm_mem(monkeypatch):
    # torch < 2.12 has the module but not the symbol; the trainer imports it
    # lazily so the attach path is testable on either version.
    monkeypatch.setattr(
        _fsdp_collectives,
        "SymmMemAllGather",
        FakeSymmMemAllGather,
        raising=False,
    )


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("", False), ("1", False), ("rccl_sdma", True)],
)
def test_gate_requires_fsdp_backend_selector(monkeypatch, value, expected):
    monkeypatch.delenv("FSDP_ALL_GATHER_BACKEND", raising=False)
    if value is not None:
        monkeypatch.setenv("FSDP_ALL_GATHER_BACKEND", value)
    assert FSDP2Trainer._sdma_all_gather_enabled(SimpleNamespace()) is expected


def test_attaches_symm_mem_all_gather_when_selected(monkeypatch):
    monkeypatch.setenv("FSDP_ALL_GATHER_BACKEND", "rccl_sdma")
    attached = []

    FSDP2Trainer._maybe_attach_sdma_all_gather(make_trainer(), make_module(attached))

    assert len(attached) == 1
    assert attached[0].group.group_name == "test-group"
    assert attached[0].backend == "NCCL"


def test_no_attach_when_not_selected(monkeypatch):
    monkeypatch.delenv("FSDP_ALL_GATHER_BACKEND", raising=False)
    attached = []

    FSDP2Trainer._maybe_attach_sdma_all_gather(make_trainer(), make_module(attached))

    assert attached == []


def test_skips_multi_param_group_module(monkeypatch):
    monkeypatch.setenv("FSDP_ALL_GATHER_BACKEND", "rccl_sdma")
    attached = []
    groups = [SimpleNamespace(), SimpleNamespace()]

    FSDP2Trainer._maybe_attach_sdma_all_gather(make_trainer(), make_module(attached, groups=groups))

    assert attached == []
