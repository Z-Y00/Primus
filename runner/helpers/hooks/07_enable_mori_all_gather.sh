#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: opt into MORI FSDP all-gather.
#
# Trigger:
#
#   export MORI_ALL_GATHER=1
#   primus-cli direct -- train pretrain --config <existing yaml>
#
# When enabled, this hook propagates MORI_* env vars into torchrun
# children. The Python patches then attach MoriAllGather to FSDP2 modules.

set -euo pipefail

case "${MORI_ALL_GATHER:-0}" in
    1|true|True|yes|on) ;;
    *) exit 0 ;;
esac

if [[ "${SDMA_ALL_GATHER:-0}" == "1" ]]; then
    echo "[ERROR] MORI_ALL_GATHER and SDMA_ALL_GATHER are mutually exclusive." >&2
    exit 1
fi

# MORI all-gather uses SDMA for the intra-node leg. Allow an explicit caller
# value to win, but default it on for this feature.
export MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-1}"
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-8G}"

# MORI's single-node eager path is the correctness-safe default on the ROCm
# versions used by Primus v26.4. Explicit user settings still win.
export MORI_HIER_CUDA_GRAPH="${MORI_HIER_CUDA_GRAPH:-0}"

if [[ -z "${MORI_SOCKET_IFNAME:-}" && -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    export MORI_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME#=}"
fi

# primus-cli direct does not implicitly propagate host env into torchrun
# children. Emit every MORI_* variable as env.* so user-selected MORI tuning
# knobs (host-proxy, RDMA devices, async, graph/debug flags, etc.) survive.
for name in "${!MORI_@}"; do
    echo "env.${name}=${!name}"
done
