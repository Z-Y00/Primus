#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_TAG="${IMAGE_TAG:-primus:gptoss-rccl-sdma-flydsl-v26.5}"
BASE_IMAGE="${BASE_IMAGE:-rocm/primus:v26.5@sha256:3040bf42974d791dd42de2e36b3c919a00869a5754cfc57a06b96d004c55eed1}"
PRIMUS_REPO="${PRIMUS_REPO:-https://github.com/Z-Y00/Primus.git}"
PRIMUS_REF="${PRIMUS_REF:-feature/gptoss-rccl-sdma-flydsl}"
PRIMUS_TURBO_REPO="${PRIMUS_TURBO_REPO:-https://github.com/AMD-AGI/Primus-Turbo.git}"
PRIMUS_TURBO_REF="${PRIMUS_TURBO_REF:-9c58dc318ed1780e55f9743cb410a1108b50ff29}"

docker build \
    --network host \
    --file "${SCRIPT_DIR}/Dockerfile.runtime-v26.5" \
    --tag "${IMAGE_TAG}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "PRIMUS_REPO=${PRIMUS_REPO}" \
    --build-arg "PRIMUS_REF=${PRIMUS_REF}" \
    --build-arg "PRIMUS_TURBO_REPO=${PRIMUS_TURBO_REPO}" \
    --build-arg "PRIMUS_TURBO_REF=${PRIMUS_TURBO_REF}" \
    "${SCRIPT_DIR}"

echo "Built ${IMAGE_TAG}"
