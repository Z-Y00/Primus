###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from primus.backends.diffusion.data.flux_precomputed import (
    FluxPrecomputedDataset,
    FluxPrecomputedProcessor,
    FluxRawImageTextDataset,
    FluxRawImageTextProcessor,
    FluxSyntheticPrecomputedDataset,
)
from primus.backends.diffusion.utils.log import logger

_SYNTHETIC_BANNER = (
    "SYNTHETIC FLUX DATA (dataset_type=synthetic): encodings are random tensors, "
    "not real preprocessed samples. Throughput, memory and collective behaviour "
    "are valid; loss and convergence are MEANINGLESS and must never be reported "
    "as a training result."
)


def _build_flux_dataset_from_config(dataset_config: dict, *, role: str):
    processor_config = dataset_config.get("processor_config", {}) or {}
    dataset_type = str(dataset_config.get("dataset_type", "precomputed")).lower()
    if dataset_type == "raw":
        processor = FluxRawImageTextProcessor(processor_config)
        processor.build()
        dataset = FluxRawImageTextDataset(
            dataset_path=dataset_config.get("dataset_path"),
            dataset_format=dataset_config.get("dataset_format", "webdataset"),
            dataset_name=dataset_config.get("dataset"),
            data_folder=dataset_config.get("data_folder"),
        )
        logger.info(f"Built FLUX {role} raw image-text dataset with {len(dataset)} samples")
    elif dataset_type == "precomputed":
        processor = FluxPrecomputedProcessor(processor_config)
        processor.build()
        dataset = FluxPrecomputedDataset(
            dataset_config["dataset_path"],
            require_timestep=role == "eval",
        )
        logger.info(f"Built FLUX {role} precomputed dataset with {len(dataset)} samples")
    elif dataset_type == "synthetic":
        # Prompt dropout would need a real empty-encodings file on disk, which
        # defeats the point of a no-dataset run; random prompts are already
        # meaningless, so drop it rather than demand the file.
        synthetic_processor_config = dict(processor_config)
        synthetic_processor_config["prompt_dropout_prob"] = 0.0
        processor = FluxPrecomputedProcessor(synthetic_processor_config)
        processor.build()
        dataset = FluxSyntheticPrecomputedDataset(
            num_samples=int(dataset_config.get("num_samples") or 256),
            img_size=int(synthetic_processor_config.get("img_size") or 256),
            require_timestep=role == "eval",
        )
        # Loud and unconditional: a synthetic run must never be mistaken for a
        # real one when someone reads the log later.
        logger.warning("=" * 100)
        logger.warning(_SYNTHETIC_BANNER)
        logger.warning(
            f"Built FLUX {role} SYNTHETIC dataset with {len(dataset)} samples "
            f"(t5={dataset.t5_shape}, clip={dataset.clip_shape}, latent={dataset.latent_shape})"
        )
        logger.warning("=" * 100)
    else:
        raise ValueError("FLUX dataset_type must be one of: 'precomputed', 'raw', 'synthetic'")
    return dataset, processor


def build_flux_dataset(dataset_config: dict):
    dataset, processor = _build_flux_dataset_from_config(dataset_config, role="train")
    eval_dataset_path = dataset_config.get("eval_dataset_path")
    if not eval_dataset_path:
        return dataset, processor, None, None

    eval_config = dict(dataset_config)
    eval_config["dataset_path"] = eval_dataset_path
    eval_processor_config = dict(eval_config.get("processor_config", {}) or {})
    eval_processor_config["prompt_dropout_prob"] = 0.0
    eval_config["processor_config"] = eval_processor_config
    eval_dataset, eval_processor = _build_flux_dataset_from_config(eval_config, role="eval")
    return dataset, processor, eval_dataset, eval_processor
