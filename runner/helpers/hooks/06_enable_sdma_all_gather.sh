#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: opt into RCCL copy-engine collective paths.
#
# Backend selectors:
#
#   export FSDP_ALL_GATHER_BACKEND=rccl_sdma
#   export MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma  # direct symmetric buffers
#   export MEGATRON_GRAD_REDUCE_BACKEND=rccl_sdma_a2a
#   primus-cli direct -- train pretrain --config <any existing yaml>
#
# The FSDP selector uses global NCCL_CTA_POLICY=2 because FSDP's custom
# collective owns the relevant communicator. The Megatron distributed-
# optimizer selectors create dedicated zero-CTA process groups in Python, so
# this hook deliberately does not change the global policy.
#
# For any selector, this hook:
#   1. Exports cuMem and allocator prerequisites used by RCCL's copy-engine path.
#   2. Propagates the selector into torchrun children.
#   3. Rebuilds the bundled LD_PRELOAD interposer
#      (hooks/sdma/hip_attr_drain_preload.c) into /tmp and exports
#      LD_PRELOAD. The interposer is a workaround for the ROCm
#      cuDeviceGetAttribute hipErrorInvalidValue TLS-leak hit by RCCL's
#      cuMem code path on builds that don't have the upstream fix.
#      Recompiled every time so it never goes stale relative to the
#      source.
#      Lorri: Check JIRA ticket ROCM-24832 for more details.
#
# Anything else (HSA_SDMA_LINEAR_B2B, NCCL_DEBUG, ...) can be set
# independently via the normal `export` / `primus-cli --env` flow.

set -euo pipefail

fsdp_backend="${FSDP_ALL_GATHER_BACKEND:-}"
megatron_param_backend="${MEGATRON_PARAM_GATHER_BACKEND:-}"
megatron_grad_backend="${MEGATRON_GRAD_REDUCE_BACKEND:-}"

if [[ "${fsdp_backend}" != "rccl_sdma" \
      && "${megatron_param_backend}" != "rccl_sdma" \
      && "${megatron_grad_backend}" != "rccl_sdma_a2a" ]]; then
    exit 0
fi

if [[ "${fsdp_backend}" == "rccl_sdma" \
      && ( "${megatron_param_backend}" == "rccl_sdma" \
           || "${megatron_grad_backend}" == "rccl_sdma_a2a" ) ]]; then
    echo "[ERROR] FSDP and Megatron RCCL-SDMA cannot be enabled together: " \
         "the FSDP path requires global NCCL_CTA_POLICY=2, while the Megatron " \
         "paths require zero CTA only on dedicated process groups." >&2
    exit 2
fi

# RCCL's cuMem path requires Linux 6.8 or newer.
kernel_release="$(uname -r)"
if [[ ! "${kernel_release}" =~ ^([0-9]+)\.([0-9]+) ]]; then
    echo "[ERROR] RCCL-SDMA could not parse Linux kernel version: ${kernel_release}" >&2
    exit 2
fi
kernel_major="${BASH_REMATCH[1]}"
kernel_minor="${BASH_REMATCH[2]}"
if (( kernel_major < 6 || (kernel_major == 6 && kernel_minor < 8) )); then
    echo "[ERROR] RCCL-SDMA requires Linux kernel 6.8 or newer for cuMem; " \
         "found ${kernel_release}." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) Common cuMem and allocator prerequisites for the RCCL copy-engine path.
echo "env.NCCL_CUMEM_ENABLE=1"
echo "env.NCCL_LOCAL_REGISTER=0"
if [[ "${TORCH_NCCL_ALLOCATOR_HOOK_OVERRIDE:-}" != "unset" ]]; then
    echo "env.TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=${TORCH_NCCL_ALLOCATOR_HOOK_OVERRIDE:-true}"
fi

# 2) Make the active selector visible to torchrun children. primus-cli direct
#    does not otherwise inherit arbitrary host environment variables.
if [[ "${fsdp_backend}" == "rccl_sdma" ]]; then
    echo "env.NCCL_CTA_POLICY=2"
    echo "env.FSDP_ALL_GATHER_BACKEND=rccl_sdma"
    if [[ -n "${PRIMUS_TURBO_GROUPED_GEMM_BACKEND:-}" ]]; then
        echo "env.PRIMUS_TURBO_GROUPED_GEMM_BACKEND=${PRIMUS_TURBO_GROUPED_GEMM_BACKEND}"
    fi
fi
if [[ "${megatron_param_backend}" == "rccl_sdma" \
      || "${megatron_grad_backend}" == "rccl_sdma_a2a" ]]; then
    echo "env.MEGATRON_RCCL_SDMA_CTA_POLICY=${MEGATRON_RCCL_SDMA_CTA_POLICY:-2}"
    echo "env.MEGATRON_RCCL_SDMA_TIMEOUT_MINUTES=${MEGATRON_RCCL_SDMA_TIMEOUT_MINUTES:-10}"
    if [[ -n "${MEGATRON_RCCL_SDMA_LOG:-}" ]]; then
        echo "env.MEGATRON_RCCL_SDMA_LOG=${MEGATRON_RCCL_SDMA_LOG}"
    fi
fi
if [[ "${megatron_param_backend}" == "rccl_sdma" ]]; then
    echo "env.MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma"
    echo "env.ENABLE_SDMA_ALLGATHER=0"
    echo "env.MEGATRON_RCCL_SDMA_EAGER_INIT=${MEGATRON_RCCL_SDMA_EAGER_INIT:-1}"
    if [[ -n "${MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES:-}" ]]; then
        echo "env.MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES=${MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES}"
    fi
fi
if [[ "${megatron_grad_backend}" == "rccl_sdma_a2a" ]]; then
    echo "env.MEGATRON_GRAD_REDUCE_BACKEND=rccl_sdma_a2a"
    echo "env.MEGATRON_RCCL_SDMA_RS_STAGING_BYTES=${MEGATRON_RCCL_SDMA_RS_STAGING_BYTES:-536870912}"
    echo "env.MEGATRON_RCCL_SDMA_RS_WORKSPACE_DEPTH=${MEGATRON_RCCL_SDMA_RS_WORKSPACE_DEPTH:-2}"
    echo "env.MEGATRON_RCCL_SDMA_RS_PIPELINE=${MEGATRON_RCCL_SDMA_RS_PIPELINE:-1}"
    echo "env.MEGATRON_RCCL_SDMA_RS_DIRECT_INPUT=${MEGATRON_RCCL_SDMA_RS_DIRECT_INPUT:-0}"
    if [[ -n "${MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL:-}" ]]; then
        echo "env.MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL=${MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL}"
    fi
    if [[ -n "${MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL_BUCKETS:-}" ]]; then
        echo "env.MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL_BUCKETS=${MEGATRON_RCCL_SDMA_RS_NATIVE_TAIL_BUCKETS}"
    fi
    if [[ -n "${PRIMUS_TURBO_ATTN_SINGLE_STREAM:-}" ]]; then
        echo "env.PRIMUS_TURBO_ATTN_SINGLE_STREAM=${PRIMUS_TURBO_ATTN_SINGLE_STREAM}"
    fi
fi
# The native Primus RCCL build used by Megatron has the cuMem probe fix and
# does not need the legacy attribute-drain interposer.
if [[ "${fsdp_backend}" != "rccl_sdma" ]]; then
    exit 0
fi

# 3) Always (re)build the interposer. The source is tiny and gcc is
#    typically <1s; we don't bother with a staleness check so the .so
#    can never lag behind the source.
SRC="${SCRIPT_DIR}/sdma/hip_attr_drain_preload.c"
SO=/tmp/libhip_attr_drain.so
if [[ ! -f "${SRC}" ]]; then
    echo "[ERROR] [Hooks/sdma] interposer source not found: ${SRC}" >&2
    exit 1
fi
gcc -O2 -fPIC -shared "${SRC}" -o "${SO}" -ldl
echo "env.LD_PRELOAD=${SO}"
