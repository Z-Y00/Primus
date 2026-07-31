# MORI for Primus FSDP

Primus uses MORI's FSDP2 `MoriAllGather` adapter when:

```bash
export MORI_ALL_GATHER=1
```

## Build the tested image

From the Primus repository root:

```bash
docker build \
  -f runner/helpers/mori/Dockerfile \
  --build-arg MORI_DEVICE_NIC=bnxt \
  -t primus-mori:bnxt .
```

The Dockerfile follows MORI's upstream `docker/Dockerfile.dev` dependency list
and pins the MORI source revision used for Primus validation. Its default base
is the tested ROCm 7.15 Primus nightly. The source build is required for now:

- `amd_mori==1.2.2` installs but does not expose
  `mori.ccl.HierAllGather`.
- The advertised `amd-mori-nightly` package currently has no wheel compatible
  with the image's Python environment.
- The merged MORI source contains `HierAllGather` and the FSDP adapter API.

`MORI_DEVICE_NIC` must match the target cluster (`bnxt`, `mlx5`, or `ionic`).
Docker builds cannot reliably auto-detect host RDMA devices, so the Dockerfile
defaults to `bnxt` for the validated MI300X cluster.

## Run

Use only one custom FSDP all-gather backend:

```bash
MORI_ALL_GATHER=1 \
  bash runner/primus-cli direct -- train pretrain --config <config>
```

Do not combine `MORI_ALL_GATHER=1` with `SDMA_ALL_GATHER=1`.

The launcher hook supplies correctness-safe single-node defaults:

```text
MORI_ENABLE_SDMA=1
MORI_SHMEM_HEAP_SIZE=8G
MORI_HIER_CUDA_GRAPH=0
```

Explicit user values override these defaults.

## Reproduce the validated MI355X two-node run

The validated pair used identical AINIC stacks:

| Setting | Value |
|---|---|
| Master | `smci355-ccs-aus-n04-33` |
| Worker | `smci355-ccs-aus-n05-21` |
| Driver | `26.03.3.001` |
| Firmware | `1.117.5-a-77` |
| `libionic` hash prefix | `1ab5ac7f6dda` |
| CCQE | enabled on both nodes |
| Bootstrap interface | `fenic` |
| RCCL GID index | `1` |

Do not mix nodes with different driver, firmware, or `libionic` builds. MORI
detects CCQE capability independently on each node; a `ccqe=True/False`
mismatch causes incompatible kernels and a collective hang.

### 1. Build the ionic image

```bash
cd /apps/tas/lorrirao/sdma_rccl_pytorch/primus

docker build \
  -f runner/helpers/mori/Dockerfile \
  --build-arg MORI_DEVICE_NIC=ionic \
  -t primus-mori:ionic .
```

### 2. Make the image available on both nodes

If a shared registry is unavailable, stream the image directly:

```bash
for node in smci355-ccs-aus-n04-33 smci355-ccs-aus-n05-21; do
  docker save primus-mori:ionic |
    zstd -1 -T0 |
    ssh "${node}" 'zstd -d -T0 | docker load' &
done
wait
```

### 3. Prepare the shared TorchTitan data directory

TorchTitan node rank 0 downloads tokenizer assets while other node ranks wait
for the same file. Both containers must mount the same directory:

```bash
mkdir -p /apps/tas/lorrirao/sdma_rccl_pytorch/mori_multinode_data
```

The default tokenizer token path is:

```text
/apps/tas/lorrirao/.cache/huggingface/token
```

Override `SHARED_DATA` or `HF_TOKEN_PATH` when launching if needed.

### 4. Run the validations

```bash
# Bit-exact 16-rank MORI all-gather, 8 MiB per rank.
IMAGE=primus-mori:ionic \
  runner/helpers/mori/run_mi355_multinode.sh collective

# TorchTitan Qwen3 0.6B, FSDP, 2 layers, 3 steps.
IMAGE=primus-mori:ionic \
  runner/helpers/mori/run_mi355_multinode.sh torchtitan

# Megatron Llama 3.2 1B, FSDP2, 2 layers, 3 iterations.
IMAGE=primus-mori:ionic \
  runner/helpers/mori/run_mi355_multinode.sh megatron
```

Useful overrides:

```bash
MASTER_NODE=<host> \
WORKER_NODE=<host> \
MASTER_ADDR=<master-fenic-ip> \
NCCL_IB_GID_INDEX=<valid-ipv4-rocev2-index> \
IMAGE=<image> \
LOG_DIR=<local-log-dir> \
runner/helpers/mori/run_mi355_multinode.sh collective
```

### Expected success markers

Collective:

```text
MORI_MULTINODE_ALLGATHER_PASS world=16 ranks_per_node=8 per_rank_mb=8.0
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
- Non-master node waits forever for tokenizer: ensure `SHARED_DATA` is mounted
  into `/workspace/Primus/data` on both nodes.
- BNXT `231.x`: unsupported for MORI IBGDA; use supported firmware/userspace or
  a validated mlx5/ionic pair.
