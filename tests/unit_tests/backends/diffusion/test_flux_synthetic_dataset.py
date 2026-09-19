###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus.backends.diffusion.data.flux_precomputed import (
    FluxPrecomputedDataset,
    FluxSyntheticPrecomputedDataset,
)


def test_emits_the_fields_the_real_dataset_promises():
    ds = FluxSyntheticPrecomputedDataset(num_samples=4)
    sample = ds[0]
    assert set(FluxPrecomputedDataset.required_fields) <= set(sample)
    assert all(isinstance(v, torch.Tensor) for v in sample.values())


def test_shapes_follow_img_size():
    ds = FluxSyntheticPrecomputedDataset(num_samples=2, img_size=512)
    sample = ds[0]
    assert sample["t5_encodings"].shape == (256, 4096)
    assert sample["clip_encodings"].shape == (768,)
    # 512 / vae_scale_factor 8 = 64
    assert sample["mean"].shape == (16, 64, 64)
    assert sample["logvar"].shape == sample["mean"].shape


def test_samples_are_deterministic_and_index_dependent():
    a = FluxSyntheticPrecomputedDataset(num_samples=8)
    b = FluxSyntheticPrecomputedDataset(num_samples=8)
    # Same index must agree across instances, so every rank sees the same data.
    assert torch.equal(a[3]["mean"], b[3]["mean"])
    assert not torch.equal(a[3]["mean"], a[4]["mean"])


def test_eval_role_gets_a_valid_mlperf_timestep():
    ds = FluxSyntheticPrecomputedDataset(num_samples=4, require_timestep=True)
    timestep = ds[0]["timestep"]
    assert timestep.dtype == torch.int64
    assert 0 <= int(timestep.item()) <= 7


def test_train_role_has_no_timestep():
    assert "timestep" not in FluxSyntheticPrecomputedDataset(num_samples=4)[0]


def test_len_and_bounds():
    ds = FluxSyntheticPrecomputedDataset(num_samples=5)
    assert len(ds) == 5
    with pytest.raises(IndexError):
        ds[5]


@pytest.mark.parametrize("kwargs", [{"num_samples": 0}, {"num_samples": 4, "img_size": 100}])
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        FluxSyntheticPrecomputedDataset(**kwargs)
