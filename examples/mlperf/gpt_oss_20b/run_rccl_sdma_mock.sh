#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-sdma}"
IMAGE="${IMAGE:-primus:gptoss-rccl-sdma-flydsl-v26.5}"
STEPS="${STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
SCRATCH_BYTES="${SCRATCH_BYTES:-536870912}"
DDP_NUM_BUCKETS="${DDP_NUM_BUCKETS:-24}"
EAGER_PARAM_BYTES="${EAGER_PARAM_BYTES:-41011904512}"
PROFILE="${PROFILE:-0}"
PROFILE_START="${PROFILE_START:-40}"
PROFILE_END="${PROFILE_END:-42}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/gptoss-rccl-sdma-results}"
CACHE_DIR="${CACHE_DIR:-$PWD/gptoss-runtime-cache}"

case "${MODE}" in
baseline|sdma|direct) ;;
*)
    echo "MODE must be baseline, sdma, or direct" >&2
    exit 2
    ;;
esac

mkdir -p "${RESULTS_DIR}" "${CACHE_DIR}"
RESULTS_DIR="$(realpath "${RESULTS_DIR}")"
CACHE_DIR="$(realpath "${CACHE_DIR}")"

docker run --rm -i \
    --name "gptoss-mlperf-${MODE}" \
    --network host \
    --ipc host \
    --privileged \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 \
    --ulimit nofile=1048576:1048576 \
    --shm-size 128g \
    --volume "${RESULTS_DIR}:/results" \
    --volume "${CACHE_DIR}:/root/.cache" \
    "${IMAGE}" \
    bash -s -- "${MODE}" "${STEPS}" "${WARMUP_STEPS}" "${SCRATCH_BYTES}" "${DDP_NUM_BUCKETS}" \
        "${PROFILE}" "${PROFILE_START}" "${PROFILE_END}" "${EAGER_PARAM_BYTES}" <<'CONTAINER'
set -euo pipefail

MODE="$1"
STEPS="$2"
WARMUP_STEPS="$3"
SCRATCH_BYTES="$4"
export DDP_NUM_BUCKETS="$5"
PROFILE="$6"
PROFILE_START="$7"
PROFILE_END="$8"
EAGER_PARAM_BYTES="$9"

cd /workspace/Primus
export PYTHONPATH="${PYTHONPATH:-}"
source examples/mlperf/gpt_oss_20b/config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh

python3 - <<'PY'
import os
import re
from pathlib import Path

source = Path("/workspace/Primus/examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-FP8-mlperf-pretrain.yaml")
text = source.read_text()
text = text.replace("tokenizer_type: Llama3Tokenizer", "tokenizer_type: NullTokenizer")
text = re.sub(r"(?m)^(\s*)tokenizer_model:.*$", r"\1tokenizer_model: null", text)
text = text.replace("mock_data: false", "mock_data: true")
text = re.sub(r'(?m)^(\s*)(train|valid|test)_data_path:.*$', r"\1\2_data_path: null", text)
text = re.sub(
    r"(?m)^(\s*)ddp_num_buckets:\s*\d+\s*$",
    rf"\g<1>ddp_num_buckets: {os.environ['DDP_NUM_BUCKETS']}",
    text,
)
Path("/tmp/gptoss-mlperf-mock.yaml").write_text(text)
PY

export EXP=/tmp/gptoss-mlperf-mock.yaml
export PRIMUS_WORKSPACE="/results/workspace-${MODE}"
export PRIMUS_TRAIN_ITERS="${STEPS}"
export PRIMUS_PROFILE="$([[ "${PROFILE}" == "1" ]] && echo True || echo False)"
export PRIMUS_PROFILE_STEP_START="${PROFILE_START}"
export PRIMUS_PROFILE_STEP_END="${PROFILE_END}"
export EVAL_ITERS=0
export PRIMUS_EVAL_INTERVAL=1000000
export SYNTH_WARMUP_STEPS="${WARMUP_STEPS}"
export SYNTH_WARMUP_VALID_STEPS=0
export PRIMUS_MLPERF_PERF_MOCK=1
export DATA_PATH=/tmp/mock_data
export MLPERF_VERBOSE_LOGS=1
export LOG_INTERVAL=1
export ENABLE_SDMA_ALLGATHER=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export RCCL_MSCCL_ENABLE=0
export RCCL_DDA_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"

if [[ "${MODE}" == "sdma" || "${MODE}" == "direct" ]]; then
    export MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma
    export MEGATRON_RCCL_SDMA_EAGER_INIT=1
    export MEGATRON_RCCL_SDMA_CTA_POLICY=2
    export MEGATRON_RCCL_SDMA_SCRATCH_BYTES="${SCRATCH_BYTES}"
    if [[ "${MODE}" == "direct" ]]; then
        export MEGATRON_RCCL_SDMA_DIRECT=1
        export MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES="${EAGER_PARAM_BYTES}"
    else
        export MEGATRON_RCCL_SDMA_DIRECT=0
    fi
else
    unset MEGATRON_PARAM_GATHER_BACKEND
    unset MEGATRON_RCCL_SDMA_DIRECT
    unset MEGATRON_RCCL_SDMA_EAGER_INIT
    export NCCL_CUMEM_ENABLE=0
fi

mkdir -p /results /tmp/mock_data
if [[ -f /opt/mlperf-gpt-oss-20b/prewarm_attention.py ]]; then
    python3 /opt/mlperf-gpt-oss-20b/prewarm_attention.py
fi
./primus-cli direct -- train pretrain --config "${EXP}" \
    2>&1 | tee "/results/gptoss-${MODE}.log"
CONTAINER

echo "Completed ${MODE}; results are in ${RESULTS_DIR}"
