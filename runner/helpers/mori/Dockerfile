ARG BASE_IMAGE=unifiedtrainingdockers.azurecr.io/utd/nightly:primus_the_rock_rocm7.15_20260728
FROM ${BASE_IMAGE}

ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_REF=12d1bc32d0c93dcd5062e74f4e0f772e36e1aac4
ARG MORI_DEVICE_NIC=bnxt

ENV DEBIAN_FRONTEND=noninteractive

# MORI's upstream docker/Dockerfile.dev is the source of truth for these
# runtime/build dependencies. MPI and UMBP dependencies are intentionally
# omitted because FSDP all-gather does not use them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ibverbs-utils \
        libibverbs-dev \
        libnuma-dev \
        libpci-dev \
    && rm -rf /var/lib/apt/lists/*

# PyTorch in rocm/primus:v26.4 requires setuptools<82. Version 81 accepts
# MORI's SPDX license metadata while staying compatible with that constraint.
RUN python3 -m pip install --no-cache-dir \
        "packaging<26" \
        "setuptools==81.0.0" \
        setuptools_scm \
        Cython \
        pybind11 \
        ninja

RUN git clone --filter=blob:none "${MORI_REPO}" /opt/mori \
    && cd /opt/mori \
    && git checkout "${MORI_REF}" \
    && git submodule update --init --depth 1 \
        3rdparty/msgpack-c \
        3rdparty/spdlog

# MORI's generated link dependencies expect the RHEL-style libc linker-script
# location. Provide the equivalent Ubuntu path inside this derived image.
RUN mkdir -p /usr/lib64 \
    && ln -s /usr/lib/x86_64-linux-gnu/libc.so /usr/lib64/libc.so

ENV CMAKE_PREFIX_PATH="${ROCM_PATH}:${ROCM_PATH}/lib/rocm_sysdeps"
ENV LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"
ENV MORI_DEVICE_NIC="${MORI_DEVICE_NIC}"

RUN cd /opt/mori \
    && MAX_JOBS=32 python3 -m pip install --no-build-isolation . \
    && python3 -c "import mori; from mori.ccl import HierAllGather; print(mori.__version__)"
