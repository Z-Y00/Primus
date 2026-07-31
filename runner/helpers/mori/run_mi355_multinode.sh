#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

MODE="${1:-collective}"
case "${MODE}" in
    collective|torchtitan|megatron) ;;
    *)
        echo "Usage: $0 {collective|torchtitan|megatron}" >&2
        exit 2
        ;;
esac

MASTER_NODE="${MASTER_NODE:-smci355-ccs-aus-n04-33}"
WORKER_NODE="${WORKER_NODE:-smci355-ccs-aus-n05-21}"
MASTER_ADDR="${MASTER_ADDR:-10.235.192.139}"
MASTER_PORT="${MASTER_PORT:-29620}"
SOCKET_IFNAME="${SOCKET_IFNAME:-fenic}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
IMAGE="${IMAGE:-primus-mori:ionic}"
PRIMUS_ROOT="${PRIMUS_ROOT:-/apps/tas/lorrirao/sdma_rccl_pytorch/primus}"
SHARED_DATA="${SHARED_DATA:-/apps/tas/lorrirao/sdma_rccl_pytorch/mori_multinode_data}"
HF_TOKEN_PATH="${HF_TOKEN_PATH:-/apps/tas/lorrirao/.cache/huggingface/token}"
LOG_DIR="${LOG_DIR:-/tmp/primus-mori-${MODE}-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${LOG_DIR}"

quote_cmd() {
    printf "%q " "$@"
}

remote() {
    local node="$1"
    shift
    ssh -o BatchMode=yes "${node}" "$(quote_cmd "$@")"
}

container_name() {
    local rank="$1"
    echo "primus_mori_${MODE}_${USER}_${rank}"
}

cleanup() {
    remote "${MASTER_NODE}" docker rm -f "$(container_name 0)" >/dev/null 2>&1 || true
    remote "${WORKER_NODE}" docker rm -f "$(container_name 1)" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for node in "${MASTER_NODE}" "${WORKER_NODE}"; do
    remote "${node}" docker image inspect "${IMAGE}" >/dev/null
done

COMMON_DOCKER_ARGS=(
    docker run --rm
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    --privileged
    --ipc=host
    --network=host
    -e MORI_ALL_GATHER=1
    -e MORI_SOCKET_IFNAME="${SOCKET_IFNAME}"
    -e MORI_HIER_CUDA_GRAPH=0
    -e MORI_SHMEM_HEAP_SIZE=8G
    -e NCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
    -e GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"
    -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}"
)

launch_collective() {
    local node="$1"
    local rank="$2"
    local name
    name="$(container_name "${rank}")"
    remote "${node}" \
        "${COMMON_DOCKER_ARGS[@]}" \
        --name "${name}" \
        -e PYTHONPATH=/src/primus \
        -v "${PRIMUS_ROOT}:/src/primus:ro" \
        "${IMAGE}" \
        torchrun \
        --nnodes=2 \
        --nproc_per_node=8 \
        --node_rank="${rank}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        /src/primus/runner/helpers/mori/multinode_allgather_smoke.py
}

training_command() {
    local backend="$1"
    local rank="$2"

    if [[ "${backend}" == "torchtitan" ]]; then
        cat <<EOF
export HF_TOKEN="\$(< /run/hf_token)"
cd /workspace/Primus
bash runner/primus-cli direct \
  --log_file /workspace/torchtitan_mori_node${rank}.log \
  --env NNODES=2 \
  --env NODE_RANK=${rank} \
  --env GPUS_PER_NODE=8 \
  --env MASTER_ADDR=${MASTER_ADDR} \
  --env MASTER_PORT=${MASTER_PORT} \
  --env NCCL_SOCKET_IFNAME=${SOCKET_IFNAME} \
  --env GLOO_SOCKET_IFNAME=${SOCKET_IFNAME} \
  --env NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
  --env MORI_ALL_GATHER=1 \
  --env MORI_SOCKET_IFNAME=${SOCKET_IFNAME} \
  --env MORI_HIER_CUDA_GRAPH=0 \
  --env MORI_SHMEM_HEAP_SIZE=8G \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/qwen3_0.6B-pretrain.yaml \
  --model.n_layers 2 \
  --model.converters= \
  --training.steps 3 \
  --training.mock_data True \
  --training.seq_len 128 \
  --training.local_batch_size 1 \
  --metrics.log_freq 1 \
  --compile.enable False \
  --job.dump_folder /workspace/torchtitan_mori_outputs
EOF
    else
        cat <<EOF
cd /workspace/Primus
bash runner/primus-cli direct \
  --log_file /workspace/megatron_mori_node${rank}.log \
  --env NNODES=2 \
  --env NODE_RANK=${rank} \
  --env GPUS_PER_NODE=8 \
  --env MASTER_ADDR=${MASTER_ADDR} \
  --env MASTER_PORT=${MASTER_PORT} \
  --env NCCL_SOCKET_IFNAME=${SOCKET_IFNAME} \
  --env GLOO_SOCKET_IFNAME=${SOCKET_IFNAME} \
  --env NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
  --env MORI_ALL_GATHER=1 \
  --env MORI_SOCKET_IFNAME=${SOCKET_IFNAME} \
  --env MORI_HIER_CUDA_GRAPH=0 \
  --env MORI_SHMEM_HEAP_SIZE=8G \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.2_1B-BF16-pretrain.yaml \
  --use_torch_fsdp2 true \
  --use_distributed_optimizer false \
  --overlap_param_gather false \
  --overlap_grad_reduce false \
  --num_layers 2 \
  --train_iters 3 \
  --seq_length 128 \
  --max_position_embeddings 128 \
  --micro_batch_size 1 \
  --global_batch_size 16 \
  --enable_primus_turbo false \
  --use_turbo_attention false \
  --use_turbo_grouped_gemm false \
  --ckpt_format torch_dist
EOF
    fi
}

launch_training() {
    local node="$1"
    local rank="$2"
    local backend="$3"
    local name
    name="$(container_name "${rank}")"

    local args=(
        "${COMMON_DOCKER_ARGS[@]}"
        --name "${name}"
        -v "${PRIMUS_ROOT}/primus:/workspace/Primus/primus:ro"
        -v "${PRIMUS_ROOT}/runner:/workspace/Primus/runner:ro"
        -v "${PRIMUS_ROOT}/examples:/workspace/Primus/examples:ro"
    )

    if [[ "${backend}" == "torchtitan" ]]; then
        args+=(
            -v "${HF_TOKEN_PATH}:/run/hf_token:ro"
            -v "${SHARED_DATA}:/workspace/Primus/data"
        )
    fi

    remote "${node}" \
        "${args[@]}" \
        "${IMAGE}" \
        bash -lc "$(training_command "${backend}" "${rank}")"
}

echo "Mode          : ${MODE}"
echo "Master        : ${MASTER_NODE} (${MASTER_ADDR})"
echo "Worker        : ${WORKER_NODE}"
echo "Image         : ${IMAGE}"
echo "Logs          : ${LOG_DIR}"

if [[ "${MODE}" == "collective" ]]; then
    launch_collective "${MASTER_NODE}" 0 >"${LOG_DIR}/master.log" 2>&1 &
    master_pid=$!
    sleep 1
    launch_collective "${WORKER_NODE}" 1 >"${LOG_DIR}/worker.log" 2>&1 &
    worker_pid=$!
else
    launch_training "${MASTER_NODE}" 0 "${MODE}" >"${LOG_DIR}/master.log" 2>&1 &
    master_pid=$!
    sleep 1
    launch_training "${WORKER_NODE}" 1 "${MODE}" >"${LOG_DIR}/worker.log" 2>&1 &
    worker_pid=$!
fi

status=0
wait "${master_pid}" || status=$?
wait "${worker_pid}" || status=$?

if [[ "${status}" -ne 0 ]]; then
    echo "Multi-node ${MODE} failed. Logs: ${LOG_DIR}" >&2
    exit "${status}"
fi

echo "Multi-node ${MODE} passed. Logs: ${LOG_DIR}"
