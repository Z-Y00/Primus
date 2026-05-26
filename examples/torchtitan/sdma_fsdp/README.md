# SDMA-dispatched FSDP all-gather on 8x MI300X

This directory contains the self-contained reproduction for routing
PyTorch FSDP's all-gather (and reduce-scatter) through the AMD **SDMA
copy engines** instead of the default CU-based RCCL kernel
(`ncclDevKernel_Generic_2`), using a small patch wired into Primus'
TorchTitan backend.

## Reproduce in two commands

Prereqs: an 8x MI300X node with Docker + `/dev/kfd` + `/dev/dri` and
internet to pull the public image and the public
`unsloth/Meta-Llama-3.1-70B-Instruct` tokenizer mirror (no `HF_TOKEN`
needed).

```bash
git clone -b feature/sdma-symm-mem-fsdp https://github.com/Z-Y00/Primus.git
cd Primus && ./examples/torchtitan/sdma_fsdp/run.sh
```

That single runner:

1. Pulls `lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1`
   (PyTorch 2.12 + ROCm 7.14).
2. Builds the `libhip_attr_drain.so` LD_PRELOAD interposer in-container
   (workaround for an unrelated `cuDeviceGetAttribute` TLS-leak bug;
   full root-cause writeup in `hip_attr_drain_preload.c`).
3. Stages the public Llama-3.1 70B tokenizer + config to
   `./.tokenizer_cache/70b/`.
4. `pip install`s Primus' minimal trainer deps + inits the TorchTitan
   submodule.
5. Runs `primus-cli direct -- train pretrain --config
   examples/torchtitan/configs/MI300X/llama3.1_70B-BF16-SDMA-pretrain.yaml`
   with the `sdma_symm_mem_collectives` patch active and
   `HSA_SDMA_LINEAR_B2B=0` so the SDMA path runs at full xGMI bandwidth.

Outputs (Primus + torchtitan logs, chrome trace at iteration 5) land
in `./outputs_70b/`.

## Knobs

All optional. Defaults give the 70B SDMA-on run.

| env | default | meaning |
|---|---|---|
| `SCALE` | `70b` | `70b` (Llama-3.1 70B BF16 FSDP) or `8b` (8B smoke, mock data, no HF) |
| `SDMA_MODE` | `on` | `on` enables the patch (SDMA dispatch); `off` runs the same Primus stack with the patch disabled — drop-in A/B baseline |
| `STEPS` | `5` | number of train iters (profile trace fires at step 5 by default) |
| `NPROC` | `8` | GPUs / ranks |
| `HSA_SDMA_LINEAR_B2B` | `0` | `0` forces the SDMA fan-out path (full bandwidth on this build); `1` forces the throttled back-to-back path for A/B |
| `ROCM_BUG_TEST_IMAGE` | `lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1` | container image to run inside |
| `OUTPUTS_HOST` | `./outputs_${SCALE}` | where the runner copies trace + logs back to |
| `HF_TOKEN` | _(unset)_ | only needed if you switch `TOKENIZER_REPO` to the gated `meta-llama/*` repo; the default unsloth mirror is public |

Examples:

```bash
SCALE=8b      ./examples/torchtitan/sdma_fsdp/run.sh   # 8B smoke
SDMA_MODE=off ./examples/torchtitan/sdma_fsdp/run.sh   # 70B baseline (patch off)
STEPS=20      ./examples/torchtitan/sdma_fsdp/run.sh   # longer 70B run
```

## What's actually plugged in

- **Patch:** [`primus/backends/torchtitan/patches/sdma_symm_mem_collectives.py`](../../../primus/backends/torchtitan/patches/sdma_symm_mem_collectives.py)
  monkey-patches `torch.distributed.fsdp.fully_shard` so every sharded
  module's all-gather / reduce-scatter buffers come from
  `symm_mem.empty()` + `rendezvous` (cuMem) instead of the caching
  allocator. With cuMem buffers RCCL switches to its
  `__amd_rocclr_batchMemOp.kd` dispatcher which submits
  `hsa_amd_memory_async_batch_copy` packets to the SDMA copy engines.
- **SDMA YAMLs:** [`examples/torchtitan/configs/MI300X/llama3.1_70B-BF16-SDMA-pretrain.yaml`](../configs/MI300X/llama3.1_70B-BF16-SDMA-pretrain.yaml)
  and `llama3.1_8B-BF16-SDMA-pretrain.yaml` enable the patch via
  `model.patches.sdma_symm_mem_collectives.enable: true` (the matching
  `*-CE-baseline-*.yaml` flips it off for the SDMA_MODE=off A/B).
- **LD_PRELOAD interposer:** [`hip_attr_drain_preload.c`](./hip_attr_drain_preload.c)
  drains the per-thread `hipErrorInvalidValue` left behind by RCCL's
  `cuDeviceGetAttribute(HANDLE_TYPE_FABRIC_SUPPORTED)` probe on ROCm.
  Built in-container by the runner, then forced into the trainer
  process with `LD_PRELOAD`. No RCCL rebuild required.

## Why `HSA_SDMA_LINEAR_B2B=0`

`HSA_SDMA_LINEAR_B2B` is an undocumented ROCm runtime knob in
[`ROCm/rocm-systems`](https://github.com/ROCm/rocm-systems) at
[`projects/rocr-runtime/runtime/hsa-runtime/core/util/flag.h`](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocr-runtime/runtime/hsa-runtime/core/util/flag.h):

```cpp
// HSA_SDMA_LINEAR_B2B: 1=force B2B, 0=force broadcast, unset=auto (size threshold)
sdma_linear_b2b_ = (var == "0") ? SDMA_DISABLE
                  : (var == "1") ? SDMA_ENABLE
                  : SDMA_DEFAULT;
```

with selection logic in
[`projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp`](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp):

```cpp
// linearB2BCopy for per-copy sizes in [16KB, 256KB].
// Above 256KB the fan-out path parallelises across engines.
constexpr size_t kLinearB2BMinSize = 16 * 1024;
constexpr size_t kBroadcastMaxSize = 256 * 1024;
```

Both were added in
[commit `a484ae43`](https://github.com/ROCm/rocm-systems/commit/a484ae43c59b53d45d8149b22e7ef98f39820173)
(2026-05-20). The ROCm 7.14.0-1384 runtime in the test container ships
the **single-dst B2B precursor** of that PR — the env var works but
the 256 KiB upper cap on the auto path isn't there yet — so unset/auto
routes any size ≥ 16 KiB through the throttled B2B engine, including
our 26 MiB FSDP shards. `=0` forces the fan-out path and recovers the
full ~324 GB/s xGMI busbw.

Measured on this build for a Llama-3 70B-shaped 209 MiB AG (median of
30 timed iters, max-reduced across ranks):

| `HSA_SDMA_LINEAR_B2B` | path | busbw |
|---|---|---|
| `1` (force B2B) | SDMA | **48.4 GB/s** |
| `0` (force fan-out — our default) | SDMA | **323.7 GB/s** |
| (control) | CU-based RCCL | 310.9 GB/s |

So with `=0`, the SDMA path is **~4% faster** than the CU path at this
payload *and* keeps the CUs idle for compute.

## Related repo

The deeper investigation, all-gather op-level bandwidth benchmark, and
the original LD_PRELOAD fix bring-up live in
<https://github.com/Z-Y00/ROCM-sdma-rccl-test>, which uses this Primus
fork as a submodule. The runner here is functionally identical to the
wrapper repo's `torchtitan/run_primus_sdma.sh`.
