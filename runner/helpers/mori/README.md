# MORI for Primus FSDP

Primus uses MORI's FSDP2 `MoriAllGather` adapter when:

```bash
export MORI_ALL_GATHER=1
```

## Runtime preflight and build

MORI must inspect the live NIC driver, firmware, vendor library, and runtime
capabilities such as ionic CCQE. A static `docker build` cannot reliably see
those host devices. Run MORI mode through the unified Primus preflight command
on every target node:

```bash
cd /apps/tas/lorrirao/sdma_rccl_pytorch/primus

runner/primus-cli direct -- preflight --mori
```

The CLI invokes `primus/tools/preflight/mori_preflight.py`, which runs
`mori_preflight.sh` on every selected node. The shell worker:

1. Prints host identity, GPU, IP, RDMA links, valid GIDs, NIC
   driver/firmware, vendor-library hash, and required DV symbols.
2. Starts a privileged temporary container from the ROCm 7.15 Primus nightly.
3. Mounts the detected host vendor library into that container.
4. Calls `runner/helpers/mori/install_mori.sh` to install dependencies, clone
   the pinned source/submodules, and build MORI with live RDMA visibility.
5. Runs an 8-GPU bit-exact all-gather smoke.
6. When `--mori-nodes` is set, keeps the temporary containers for this same
   information/build/local smoke on
   every node, verifies matching node fingerprints, then launches one
   all-gather over all `8 × N` ranks before removing them.

The source build is required for now:

- `amd_mori==1.2.2` installs but does not expose
  `mori.ccl.HierAllGather`.
- The advertised `amd-mori-nightly` package currently has no wheel compatible
  with the image's Python environment.
- The merged MORI source contains `HierAllGather` and the FSDP adapter API.

The preflight always pulls `BASE_IMAGE`; Docker reuses cached layers when it is
already current. No image is saved. Logs and phase timing are written under
`/tmp/primus-mori-preflight-<node>-<timestamp>/`.

### Vendor library names

The library names used by preflight come directly from MORI:

| NIC | MORI runtime loader names |
|---|---|
| Ionic / AINIC | `libionic.so` |
| Broadcom BNXT | `libbnxt_re.so`, then `libbnxt_re-rdmav59.so`, then `libbnxt_re-rdmav34.so` |
| Mellanox mlx5 | `libmlx5.so` |

MORI's
[`dv_loader.hpp`](https://github.com/ROCm/mori/blob/main/include/mori/application/transport/rdma/providers/dv_loader.hpp)
uses these exact `dlopen()` names. Its
[`MoriDetectDevice.cmake`](https://github.com/ROCm/mori/blob/main/cmake/MoriDetectDevice.cmake)
uses the same names for build-time `find_library()` detection. Preflight mounts
the host's detected vendor library under these aliases so build-time detection
and runtime loading use the same library.

### Install MORI for real training

Run the same installer as root inside the actual training container where the
GPU, RDMA sysfs, and vendor userspace library are visible:

```bash
cd /workspace/Primus
runner/helpers/mori/install_mori.sh
```

The installer modifies the current training environment: it installs system
and Python build dependencies, checks out pinned MORI under `/opt/mori`, builds
it, and installs the resulting package into the active Python environment.

Useful overrides are `MORI_REPO`, `MORI_REF`, `MORI_SOURCE_DIR`, `MAX_JOBS`,
and `ROCM_PATH`. The installer clears `MORI_DEVICE_NIC` so MORI detects the
live NIC and mounted vendor library.

## Run

Use only one custom FSDP all-gather backend:

```bash
MORI_ALL_GATHER=1 \
  bash runner/primus-cli direct -- train pretrain --config <config>
```

Do not combine `MORI_ALL_GATHER=1` with `SDMA_ALL_GATHER=1`.

The launcher hook supplies correctness-safe SDMA, shared-memory, and graph-mode
defaults. Explicit user values override them.

## Reproduce the validated MI355X two-node run


### 1. Run general multi-node preflight

```bash
cd /apps/tas/lorrirao/sdma_rccl_pytorch/primus

runner/primus-cli direct -- preflight --mori \
  --mori-nodes smci355-ccs-aus-n04-33,smci355-ccs-aus-n05-21 \
  --mori-socket-ifname fenic \
  --mori-gid-index 1
```

The Python orchestrator runs `mori_preflight.sh` concurrently on every node.
Every node prints the same host/GPU/IP/RDMA/GID/driver/firmware/vendor-library
information, builds and tests its own runtime-detected image, and emits a
fingerprint. A mismatch fails before the N-node collective starts.

### 2. Prepare the shared TorchTitan data directory

TorchTitan node rank 0 downloads tokenizer assets while other node ranks wait
for the same file. Both containers must mount the same directory:

```bash
mkdir -p /apps/tas/lorrirao/sdma_rccl_pytorch/mori_multinode_data
```

The default tokenizer token path is:

```text
/apps/tas/lorrirao/.cache/huggingface/token
```

Mount this directory at `/workspace/Primus/data` and the token file at
`/run/hf_token` when creating each training container.

### 3. Prepare both training containers

`preflight --mori --mori-nodes ...` already runs both the per-node 8-GPU
checks and the general N-node collective. It does not save an image.

In the actual training container on each node, install MORI and export the
validated two-node settings. The hostname expression assigns rank 0 to the
validated master and rank 1 to the validated worker:

```bash
cd /workspace/Primus
runner/helpers/mori/install_mori.sh

export NNODES=2
export GPUS_PER_NODE=8
export NODE_RANK="$(
  [[ "$(hostname -s)" == "smci355-ccs-aus-n04-33" ]] && echo 0 || echo 1
)"
export MASTER_ADDR=10.235.192.139
export MASTER_PORT=29620
export NCCL_SOCKET_IFNAME=fenic
export GLOO_SOCKET_IFNAME=fenic
export NCCL_IB_GID_INDEX=1
export MORI_ALL_GATHER=1
unset MORI_DEVICE_NIC
```

For another node set, change `NODE_RANK`, `MASTER_ADDR`,
`NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and `NCCL_IB_GID_INDEX` to match
its preflight result.

### 4. Run TorchTitan on both nodes

Mount the shared data directory at `/workspace/Primus/data` and the Hugging
Face token at `/run/hf_token`, then copy this command into both containers:
Start it on both nodes without waiting for rank 0 to return.

```bash
export HF_TOKEN="$(< /run/hf_token)"

bash runner/primus-cli direct \
  --log_file "/workspace/torchtitan_mori_node${NODE_RANK}.log" \
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
  --job.dump_folder "/workspace/torchtitan_mori_outputs_node${NODE_RANK}"
```

### 5. Run Megatron on both nodes

Alternatively, start this command in both prepared containers:

```bash
bash runner/primus-cli direct \
  --log_file "/workspace/megatron_mori_node${NODE_RANK}.log" \
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
  --global_batch_size "$((GPUS_PER_NODE * NNODES))" \
  --enable_primus_turbo false \
  --use_turbo_attention false \
  --use_turbo_grouped_gemm false \
  --ckpt_format torch_dist
```

### Expected success markers

Collective:

```text
MORI_MULTINODE_ALLGATHER_PASS world=16 ranks_per_node=8 per_rank_mb=128.0
```

Training:

```text
[MORI:FSDP] initialized MORI SHMEM from torch process group
Ionic _ccqe_enabled: True
Training completed
torchrun finished successfully (code 0)
```

The Megatron smoke should report three iterations with
`number of nan iterations: 0`.

### Troubleshooting

- `ccqe=True` on one node and `ccqe=False` on the other: choose nodes with
  matching ionic stacks.
- `local GID N/A`: inspect `/sys/class/infiniband/ionic_*/ports/1/gids/`;
  this pair uses `NCCL_IB_GID_INDEX=1`, not `3`.
- Non-master node waits forever for tokenizer: ensure the same host data
  directory is mounted at `/workspace/Primus/data` on every node.
- BNXT `231.x`: unsupported for MORI IBGDA; use supported firmware/userspace or
  a validated mlx5/ionic pair.
