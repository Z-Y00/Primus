# GPT-OSS-20B MLPerf Pretraining

MLPerf-compliant GPT-OSS 20B pretraining on one MI355X node (8 GPUs, GBS=32)
using Primus. The layout matches `examples/mlperf/llama3.1_8b`.

## Setup

### Configuration

- **Model**: GPT-OSS 20B (2880 hidden, 24 layers, 64 heads, 32 experts)
- **Training**: 1.2M iteration ceiling, GBS=32, MBS=4, LR=8e-4
- **Default precision**: FP8 + Turbo attention (`gpt_oss_20B-FP8-turbo-attn-mlperf-pretrain.yaml`)
- **Optional precision**: MXFP4 grouped GEMM, QKVO BF16, weight de-oscillation
- **Data**: C4 dataset (tokenized, same as Llama 3.1 8B)

### Data

Download the preprocessed C4 dataset and tokenizer:

```bash
mkdir -p /data/gpt_oss_20b
cd /data/gpt_oss_20b

# data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# model
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
```

Training uses the `c4-train.en_6_text_document` prefix and validation uses
`c4-validation-91205-samples.en_text_document`.

## Run with Docker (recommended)

Run the launcher from the host. It starts the container, mounts the Primus
checkout and data directories, loads the selected system configuration, and
runs the requested number of experiments.

```bash
cd /path/to/Primus

export DATADIR=/data/gpt_oss_20b/data
export MODELDIR=/data/gpt_oss_20b/model
export LOGDIR=/data/gpt_oss_20b/results

# Optional; these are the defaults.
export CONT=rocm/primus:v26.5
export DGXSYSTEM=MI355X_1x8x1
export NEXP=1

# Optional host runtime tunables before each trial (cpupower, THP, drop_caches; see runtime_tunables.sh):
# export RUN_RUNTIME_TUNABLES=1

bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
```

`DATADIR` must contain the preprocessed C4 dataset. If `MODELDIR` exists and
is nonempty, it is mounted at `/model` and used as the local tokenizer. If a
local tokenizer is unavailable, omit `MODELDIR` (or point it to an empty
directory) and export a Hugging Face token:

```bash
export HF_TOKEN=<your_huggingface_token>
bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
```

### RCCL SDMA parameter gather

Enable direct RCCL copy-engine parameter AllGather from the host:

```bash
export MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma
bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
```

The first run attempts to allocate the symmetric parameter buffer after
Megatron calculates its exact size. If that allocation fails, the error reports
the rounded value to reserve before model construction. Rerun with the reported
value, for example:

```bash
export MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma
export MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES=<reported-recommended-eager-bytes>
bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
```

The launcher raises Docker's `nofile` limit for RCCL's dedicated parameter
AllGather communicator.

### Experimental RCCL SDMA gradient ReduceScatter

Enable Megatron's AllToAll-plus-FP32-accumulation ReduceScatter through bounded
RCCL copy-engine staging:

```bash
export MEGATRON_GRAD_REDUCE_BACKEND=rccl_sdma_a2a
export PRIMUS_DDP_NUM_BUCKETS=50
export MEGATRON_RCCL_SDMA_RS_STAGING_BYTES=536870912  # 512 MiB
export MEGATRON_RCCL_SDMA_RS_WORKSPACE_DEPTH=2
# Optional diagnostic: serialize A2A and local reduction (slower in GPT-OSS A/B).
# export MEGATRON_RCCL_SDMA_RS_PIPELINE=0
# Optional full-bucket path: use grad_data directly and one receive buffer.
# export MEGATRON_RCCL_SDMA_RS_DIRECT_INPUT=1
# export MEGATRON_RCCL_SDMA_RS_WORKSPACE_DEPTH=1
# export MEGATRON_RCCL_SDMA_RS_STAGING_BYTES=1645805312
bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
```

The normal gradient input and optimizer output shard are unchanged. The
`PRIMUS_DDP_NUM_BUCKETS=50` keeps gradient communication granular enough for
backward overlap. The staging value is the AllToAll receive capacity per slot
and rank. Each communicator/device/dtype owns a bounded ring of two workspaces
by default. A workspace has two normal send buffers, two symmetric receive
buffers, and a reduction stream. A fused Triton kernel accumulates in FP32
registers and stores directly to Megatron's output shard. Streams are allocated
at first use and checked against the caller and sibling workspace streams. The
full chunk pipeline is queued eagerly at gradient-sync start; its work handle
synchronizes the caller with a completion event. Within a bucket, the second
AllToAll is queued before reducing the first chunk, allowing ProcessGroupNCCL's
internal stream to overlap transport with reduction. Direct-input mode requires
depth 1 and enough staging capacity for the largest complete bucket.

The MI355X launcher defaults to
`PYTORCH_ALLOC_CONF=expandable_segments:True` to avoid the fragmentation stall
seen at MBS4; an explicit host value still overrides the default. This path
supports BF16/FP16 transport, one
distributed-optimizer instance, and one bucket per bucket group on a
single-node job.

Parameter and gradient SDMA can be enabled together by exporting both backend
selectors. Do not set global `NCCL_CTA_POLICY` for either Megatron backend.

### Final screened SDMA configuration

The best profiled configuration on one MI355X node combines direct SDMA
parameter AllGather with direct-input SDMA gradient ReduceScatter. The last two
gradient buckets use native RCCL because their communication is exposed at the
end of backward:

```bash
export GPU_MAX_HW_QUEUES=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PRIMUS_DDP_NUM_BUCKETS=50

export MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma
export MEGATRON_RCCL_SDMA_EAGER_INIT=1
export MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES=41819308032
# Allow rank-skew while a newly provisioned node compiles kernels.
export MEGATRON_RCCL_SDMA_TIMEOUT_MINUTES=60

export MEGATRON_GRAD_REDUCE_BACKEND=rccl_sdma_a2a
export MEGATRON_RCCL_SDMA_RS_DIRECT_INPUT=1
export MEGATRON_RCCL_SDMA_RS_STAGING_BYTES=1645805312
export MEGATRON_RCCL_SDMA_RS_WORKSPACE_DEPTH=1
export MEGATRON_RCCL_SDMA_RS_PIPELINE=1
export MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL=1
export MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL_BUCKETS=2

# Requires Z-Y00/Primus-Turbo commit af66bdff.
export PRIMUS_TURBO_ATTN_SINGLE_STREAM=1

bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
```

Matched profiler Step 3 screening results:

- Direct input, Q2, two native tail buckets: **843.88 ms** (best).
- Direct input, Q2, all SDMA buckets: 858.35 ms.
- SDMA AllGather with native ReduceScatter and original Turbo: 866.23 ms.
- 512 MiB two-slot pipelined staging, Q2: 891.82 ms.
- Direct input, Q3, two native tail buckets: 893.05 ms.

The winning trace contains 26 SDMA parameter AllGathers, 24 SDMA gradient
AllToAlls, and two native tail ReduceScatters per step. Run at least three
matched 50-step repetitions for this candidate and the native-RS/original-Turbo
control after the MI355X test node is recovered; its ROCm management process
entered uninterruptible kernel sleep after screening.

### MXFP4 recipe

Override `EXP` to switch from the default FP8 Turbo-attention yaml:

```bash
export EXP=/workspace/Primus/examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-MXFP4-deosc-mlperf-pretrain.yaml
export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR=mxfp4
# Optional scale rounding for Turbo MXFP4 quant: 0=RTE, 1=RZ, 2=stochastic
# export PRIMUS_TURBO_MXFP4_SCALE_ROUNDING=0
bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
```

## Run inside an existing container

### Start Docker Image

```bash
docker run -it     --device /dev/dri     --device /dev/kfd     --device /dev/infiniband     --network host --ipc host     --group-add video     --cap-add SYS_PTRACE     --security-opt seccomp=unconfined     --privileged     -v $HOME:$HOME   --shm-size 128G     --name primus_training_env rocm/primus:v26.5

cd /workspace/Primus
```

### Key Files

- `configs/MI355/gpt_oss_20B-FP8-turbo-attn-mlperf-pretrain.yaml` — default FP8 + Turbo attention
- `configs/MI355/gpt_oss_20B-MXFP4-deosc-mlperf-pretrain.yaml` — MXFP4 grouped GEMM, QKVO BF16, de-oscillation
- `configs/MI355/gpt_oss_20B-MXFP4-qkvo-bf16.yaml` — TE precision matcher (not a runnable experiment)
- `config_MI355X_1x8x1.sh` — system config and env vars
- `run_and_time.sh` — timed training entry
- `tune_gemm_results-v26.5.txt` — optional hipBLASLt replay for TE QKV GEMMs

```bash
export HF_TOKEN=<your_huggingface_token>
source config_MI355X_1x8x1.sh
# optional MXFP4:
# export EXP=${PRIMUS_PATH}/examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-MXFP4-deosc-mlperf-pretrain.yaml
bash run_and_time.sh
```

## Notes

- `log_interval: 999999` suppresses regular Primus logs
- Grouped GEMM backend is set in the yaml (`turbo_grouped_gemm_backend: fp4:flydsl,other:hipblaslt`), not in the config shell
- `RUN_RUNTIME_TUNABLES` defaults to `0`; set `RUN_RUNTIME_TUNABLES=1` to run `runtime_tunables.sh` on the host before each trial (some steps require `sudo`)
