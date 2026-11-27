# Muse Spark on MI455X (gfx1250): 4-GPU EP 4 proxy

BF16 pretraining proxy of Muse-Spark-Open-816B-A42B on one MI455X node (4 GPUs, mock data): the first 4 layers
(sliding, full / NoPE, sliding, sliding) with all 256 experts, top-8, EP 4 (64 experts per GPU), DP 4, sequence
4096. `run_muse_spark_4gpu_ep4_mi455x.sh` runs
`examples/megatron/configs/MI455X/muse_spark_816B_A42B-BF16-pretrain-4gpu-ep4-proxy.yaml` with one of three
routed-expert paths.

## Run

From the host, the script starts `DOCKER_IMAGE` and re-runs itself inside it:

```bash
MOE=alltoall MBS=1 bash examples/megatron/models/muse-spark/run_muse_spark_4gpu_ep4_mi455x.sh
MOE=deepep   MBS=2 bash examples/megatron/models/muse-spark/run_muse_spark_4gpu_ep4_mi455x.sh
MOE=megamoe  MBS=1 bash examples/megatron/models/muse-spark/run_muse_spark_4gpu_ep4_mi455x.sh
```

Inside an already running container, add `PRIMUS_LAUNCHER=direct`. The log is written to
`output/<PRIMUS_TEAM>/<PRIMUS_USER>/<PRIMUS_EXP_NAME>/log_node0.txt`.

| variable | default | meaning |
|---|---|---|
| `MOE` | `deepep` | `alltoall` (Megatron alltoall dispatcher), `deepep` (Primus-Turbo DeepEP) or `megamoe` (Primus-Turbo Mega MoE, post-norm stage2) |
| `MBS`, `GBS` | `1`, `8 x MBS` | micro / global batch size (two micro-batches per GPU) |
| `TRAIN_ITERS` | `40` | the yaml's cosine LR decay runs over `train_iters`: compare losses only between runs of the same length |
| `DOCKER_IMAGE` | `amdprimus/amdprimus:tas-20261109-triton-83c9228329` | image used by the container launcher |
| `PRIMUS_LAUNCHER` | `container` | `container` (from the host) or `direct` (inside the container) |
| `EXTRA_ARGS` | | extra Megatron overrides |

The three paths share everything else in the yaml: Turbo FlyDSL grouped GEMM experts with the wgrads added into
the fp32 `main_grad`, the expert gradient scaling inside the fused Adam (`PRIMUS_FUSED_ADAM_EXPERT_GRAD_SCALE`),
`ddp_average_in_collective`, compiled norms and Turbo FlyDSL sliding-window attention.

## Results (2026-11-26)

Step time is the median of iterations 5-40 of one 40-iteration run (ms per iteration); tokens/s/GPU is
`MBS x 2 micro-batches x 4096 / step time`.

| routed-expert path | MBS 1 (GBS 8) | MBS 2 (GBS 16) | tokens/s/GPU (MBS 1 / 2) | yaml settings of the run |
|---|---|---|---|---|
| `alltoall` (no DeepEP, no all-to-all overlap) | 261.4 ms | 416.6 ms | 31.3k / 39.3k | before `deferred_cross_entropy` and `ddp_num_buckets: 24` |
| `deepep` (64 CUs) | **250.6 ms** | **388.6 ms** | 32.7k / 42.2k | the committed yaml |
| `megamoe` (post-norm stage2) | 264.2 ms | 403.9 ms | 31.0k / 40.6k | the committed yaml |

- Every run trained normally: lm loss 12.26 at iteration 1 and 6.61 (MBS 1) / 6.55 (MBS 2) at iteration 40, the
  same to within 0.0005 at every iteration across the three paths.
- `megamoe`: the median is over the iterations without a kernel compile. Two iterations per run took ~5 s:
  the FlyDSL grouped GEMMs on the Mega MoE pool compile one kernel per new pool size, until the on-disk
  FlyDSL cache covers the range of pool sizes routing produces. It needs a Primus-Turbo with
  `fused_mega_moe_stage2_post_norm` (Muse Spark applies the routing weight after a per-expert RMSNorm, which
  Mega MoE's own stage2 cannot express); measured with Primus-Turbo `dev/lorri/mi455x-stack` d3da7515.
- `alltoall` was measured before `deferred_cross_entropy` and `ddp_num_buckets: 24` were added to the yaml;
  with DeepEP those two saved 4.0 ms (MBS 1) and 10.6 ms (MBS 2).
- `overlap_moe_expert_parallel_comm` (1F1B all-to-all overlap, with DeepEP) is off in the yaml: with two
  micro-batches per GPU only the middle of the three phases overlaps. Measured 250.4 ms at MBS 1 (with
  `ddp_num_buckets: 24`) and 400.2 ms at MBS 2 (without it), against 250.6 / 388.6 ms without the overlap.
- Image `amdprimus/amdprimus:tas-20261109-triton-83c9228329` (Triton 3.9.0+git83c92283), with
  `TRITON_HIP_USE_EXPERT_SCHEDULING` unset and `PRIMUS_SKIP_PIP=1`.
