#!/bin/bash
# Self-contained reproduction for SDMA-dispatched FSDP all-gather on 8x MI300X.
#
# Stack:
#   - PyTorch 2.12 + ROCm 7.14 container (lorrisync/therock-main:gfx94X_...).
#   - Primus (this repo) -> TorchTitan -> Llama-3 70B BF16 FSDP.
#   - primus/backends/torchtitan/patches/sdma_symm_mem_collectives.py wraps
#     every fully_shard() call so its all-gather / reduce-scatter buffers come
#     from symm_mem (cuMem), which makes RCCL dispatch the collective on the
#     SDMA path (__amd_rocclr_batchMemOp / hsa_amd_memory_async_batch_copy)
#     instead of the CU-based ncclDevKernel_Generic_2.
#
# What this runner does on the host:
#   1. Build the libhip_attr_drain.so LD_PRELOAD interposer (workaround for
#      the cuDeviceGetAttribute TLS-leak bug; see the .c file for the full
#      root-cause writeup).
#   2. Snapshot-download the public unsloth/Meta-Llama-3.1-{70B,8B}
#      tokenizer / config files to a host dir. The unsloth mirror is
#      public so no HF_TOKEN is needed.
#
# What this runner does inside the container:
#   3. pip install Primus' minimal trainer deps (the image already has the
#      PyTorch + ROCm stack; Primus is pure-Python and tiny).
#   4. Set the CE-collective env we verified works on this build:
#        NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1
#        NCCL_LOCAL_REGISTER=0
#        TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
#        HSA_SDMA_LINEAR_B2B=0   (critical: unthrottles SDMA bandwidth)
#        LD_PRELOAD=libhip_attr_drain.so
#   5. Run primus-cli direct -- train pretrain --config <SDMA yaml>.
#
# Usage (from anywhere in the Primus checkout):
#   ./examples/torchtitan/sdma_fsdp/run.sh                     # 70B BF16 SDMA, 5 steps (default)
#   SCALE=8b      ./examples/torchtitan/sdma_fsdp/run.sh       # 8B smoke (mock data, no HF)
#   SDMA_MODE=off ./examples/torchtitan/sdma_fsdp/run.sh       # 70B BF16 baseline (patch disabled)
#   STEPS=20      ./examples/torchtitan/sdma_fsdp/run.sh       # longer 70B run
#   ROCM_BUG_TEST_IMAGE=lorrisync/...:other ./examples/torchtitan/sdma_fsdp/run.sh
#
# Outputs (Primus + torchtitan logs, chrome trace at iteration 5) are copied
# from the container to ${OUTPUTS_HOST} on the host.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ ! -f "${PRIMUS_DIR}/primus-cli" ] || [ ! -d "${PRIMUS_DIR}/primus/backends/torchtitan/patches" ]; then
    echo "FATAL: ${PRIMUS_DIR} does not look like a Primus checkout (missing primus-cli or backends/)." >&2
    echo "       Run this script from inside a Z-Y00/Primus clone." >&2
    exit 2
fi

IMAGE="${ROCM_BUG_TEST_IMAGE:-lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"
NPROC="${NPROC:-8}"
SCALE="${SCALE:-70b}"          # 70b or 8b
STEPS="${STEPS:-5}"
TOKENIZER_REPO="${TOKENIZER_REPO:-unsloth/Meta-Llama-3.1-70B-Instruct}"   # public mirror; no HF_TOKEN
HF_TOKEN="${HF_TOKEN:-}"
OUTPUTS_HOST="${OUTPUTS_HOST:-${SCRIPT_DIR}/outputs_${SCALE}}"
mkdir -p "${OUTPUTS_HOST}"

# Stage tokenizer to a host dir we mount into the container so it persists
# across runs.
TOKENIZER_HOST_DIR="${SCRIPT_DIR}/.tokenizer_cache/${SCALE}"
mkdir -p "${TOKENIZER_HOST_DIR}"

# SDMA_MODE=on (default) selects the SDMA-enabled yaml; SDMA_MODE=off selects
# the CE-baseline yaml that runs through the same Primus stack but with our
# sdma_symm_mem_collectives patch *disabled* -- useful for perf A/B.
SDMA_MODE="${SDMA_MODE:-on}"

case "${SCALE}/${SDMA_MODE}" in
    70b/on)
        CONFIG="examples/torchtitan/configs/MI300X/llama3.1_70B-BF16-SDMA-pretrain.yaml"
        ASSETS_IN_CTR="/workspace/llama3_70b_assets"
        ;;
    70b/off)
        CONFIG="examples/torchtitan/configs/MI300X/llama3.1_70B-BF16-CE-baseline-pretrain.yaml"
        ASSETS_IN_CTR="/workspace/llama3_70b_assets"
        ;;
    8b/on)
        CONFIG="examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-SDMA-pretrain.yaml"
        ASSETS_IN_CTR="/workspace/llama3_8b_assets"
        ;;
    *)
        echo "Unknown SCALE=${SCALE} / SDMA_MODE=${SDMA_MODE}" >&2
        echo "Valid combos: 70b/on, 70b/off, 8b/on" >&2
        exit 2
        ;;
esac

CNAME="primus_sdma_fsdp_${SCALE}_$$"
docker rm -f "${CNAME}" >/dev/null 2>&1 || true

echo "=== Image             : ${IMAGE}"
echo "=== Host kernel       : $(uname -r)"
echo "=== Scale / steps     : ${SCALE} / ${STEPS}"
echo "=== Config            : ${CONFIG}"
echo "=== Tokenizer (host)  : ${TOKENIZER_HOST_DIR} (from public mirror ${TOKENIZER_REPO})"
echo "=== SDMA_MODE         : ${SDMA_MODE}"
echo "=== Outputs (host)    : ${OUTPUTS_HOST}"
echo "=== Primus repo (host): ${PRIMUS_DIR}"

# Base64 the interposer source so the container can build it without a
# fragile bind mount (snap-docker can't reliably bind /apps on every host).
INTERPOSER_B64=$(base64 -w0 "${SCRIPT_DIR}/hip_attr_drain_preload.c")

docker run --name "${CNAME}" \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --ipc=host --shm-size=64g \
    --network=host \
    -v "${PRIMUS_DIR}:/workspace/primus" \
    -v "${TOKENIZER_HOST_DIR}:${ASSETS_IN_CTR}" \
    -e INTERPOSER_B64="${INTERPOSER_B64}" \
    -e NPROC="${NPROC}" \
    -e STEPS="${STEPS}" \
    -e SCALE="${SCALE}" \
    -e CONFIG_REL="${CONFIG}" \
    -e ASSETS_IN_CTR="${ASSETS_IN_CTR}" \
    -e TOKENIZER_REPO="${TOKENIZER_REPO}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HSA_SDMA_LINEAR_B2B="${HSA_SDMA_LINEAR_B2B:-0}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        export PATH=/opt/rocm/bin:${PATH}

        echo ""
        echo "############################################################"
        echo "  [1/4] Build LD_PRELOAD interposer"
        echo "############################################################"
        echo "${INTERPOSER_B64}" | base64 -d > /tmp/hip_attr_drain_preload.c
        gcc -O2 -fPIC -shared /tmp/hip_attr_drain_preload.c \
            -o /tmp/libhip_attr_drain.so -ldl
        ls -l /tmp/libhip_attr_drain.so

        echo ""
        echo "############################################################"
        echo "  [2/4] Install Primus deps + init torchtitan submodule"
        echo "############################################################"
        # The Primus repo is bind-mounted from the host; modern git refuses
        # to operate on it because the host UID != container root. Whitelist
        # all bind-mounted paths as safe so submodule init works.
        git config --global --add safe.directory '*'
        cd /workspace/primus
        if [ ! -f third_party/torchtitan/torchtitan/train.py ]; then
            git submodule update --init --depth 1 third_party/torchtitan
        fi
        # Only the trainer deps Primus imports unconditionally at startup; we
        # skip the megatron/maxtext/jax extras to keep bring-up fast.
        pip install --no-cache-dir -q \
            loguru tyro "tomli>=2.0" pyyaml typing_extensions \
            "datasets>=3.6.0" "torchdata>=0.8.0" \
            blobfile sentencepiece tiktoken huggingface_hub \
            pyrsmi plotext expecttest
        export PYTHONPATH="/workspace/primus:/workspace/primus/third_party/torchtitan:${PYTHONPATH:-}"

        echo ""
        echo "############################################################"
        echo "  [3/4] Stage public tokenizer assets -> ${ASSETS_IN_CTR}"
        echo "############################################################"
        if [ -z "$(ls -A ${ASSETS_IN_CTR} 2>/dev/null)" ]; then
            python3 - <<EOF
import os
from huggingface_hub import snapshot_download
repo  = os.environ["TOKENIZER_REPO"]
dest  = os.environ["ASSETS_IN_CTR"]
token = os.environ.get("HF_TOKEN") or None
snapshot_download(
    repo,
    allow_patterns=[
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "tokenizer.model",
        "original/tokenizer.model",
        "config.json", "generation_config.json",
    ],
    local_dir=dest, token=token,
)
print(f"[tokenizer] staged {repo} -> {dest}")
EOF
        else
            echo "[tokenizer] reusing cached assets in ${ASSETS_IN_CTR}"
        fi
        ls -la "${ASSETS_IN_CTR}" | head -15

        echo ""
        echo "############################################################"
        echo "  [4/4] Launch Primus -> torchtitan with SDMA patch"
        echo "############################################################"
        # CE env. FSDP forces cta_policy=ZERO via PG opts anyway, but we set
        # the env for completeness. NCCL_LOCAL_REGISTER=0 sidesteps the
        # unrelated hipMemRetainAllocationHandle SIGSEGV on this build.
        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        # Critical for SDMA bandwidth on this build: with the runtime
        # default the SDMA copy is throttled to ~48 GB/s busbw on a 209 MiB
        # AG -- 6.7x slower than the CU-driven path. Setting =0 forces the
        # fan-out path and unlocks the full ~324 GB/s xGMI ceiling.
        # See README in this dir for the upstream rocm-systems source pin.
        export HSA_SDMA_LINEAR_B2B="${HSA_SDMA_LINEAR_B2B:-0}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export OMP_NUM_THREADS=8
        export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=29500
        export PYTORCH_ROCM_ARCH=gfx942
        export NCCL_CTA_POLICY=2
        export NCCL_CUMEM_ENABLE=1
        export NCCL_LOCAL_REGISTER=0
        export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        # Force lo: Primus auto-detects an interface from `hostname -I`,
        # which can pick something not present inside the container.
        export NCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        export PRIMUS_HF_ASSETS_PATH="${ASSETS_IN_CTR}"
        env | grep -E "^(NCCL_|TORCH_NCCL_|HSA_|LD_PRELOAD|PRIMUS_)" | sort

        mkdir -p /workspace/outputs
        cd /workspace/primus
        # primus-cli direct == run in current shell (we are already inside
        # the container). It auto-detects MI300X and torchruns the CLI main.
        bash runner/primus-cli direct \
             --env NCCL_CTA_POLICY=2 \
             --env NCCL_CUMEM_ENABLE=1 \
             --env NCCL_LOCAL_REGISTER=0 \
             --env TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true \
             --env NCCL_SOCKET_IFNAME=lo \
             --env GLOO_SOCKET_IFNAME=lo \
             --env LD_PRELOAD=/tmp/libhip_attr_drain.so \
             -- train pretrain --config "${CONFIG_REL}" \
            2>&1 | tee /workspace/outputs/train.log
    '
RC=$?

echo ""
echo "=== Extracting outputs with docker cp ==="
docker cp "${CNAME}:/workspace/outputs/." "${OUTPUTS_HOST}/" 2>/dev/null || true
docker cp "${CNAME}:/workspace/primus/output/." "${OUTPUTS_HOST}/primus_output/" 2>/dev/null || true
docker cp "${CNAME}:/workspace/primus/outputs/." "${OUTPUTS_HOST}/torchtitan_outputs/" 2>/dev/null || true
docker rm -f "${CNAME}" >/dev/null 2>&1 || true

if [ ${RC} -ne 0 ]; then
    echo "Training exited non-zero (${RC})." >&2
    exit ${RC}
fi

echo ""
echo "Done. Outputs in: ${OUTPUTS_HOST}"
ls -la "${OUTPUTS_HOST}"
