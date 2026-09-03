# RCCL CE synchronous AllGather on imported VMM: debug report

## Status

**Reproduced and root-caused, not fixed.**

On eight MI355X GPUs, a synchronous RCCL copy-engine (CE) AllGather into
PyTorch NCCL symmetric memory crashes every rank in ROCr. The equivalent
asynchronous AllGather on a non-null stream succeeds.

The immediate trigger is RCCL's legacy/null-stream fallback from
`hipMemcpyBatchAsync` to per-peer `hipMemcpyAsync`. The latter queries pointer
metadata for an imported peer VMM mapping. ROCr finds the imported mapping, but
then dereferences its intentionally null owning `MemoryRegion`.

This is not an unmatched collective and is not caused by Megatron's use of a
synchronous collective. Megatron's `force_sync=True` usage is valid.

## Affected environment

The failure was captured with:

- 8 x AMD Instinct MI355X
- PyTorch 2.12
- ROCm/HIP 7.15.0 development build from 2026-07-20
- RCCL 2.30.4, `rocm-systems` commit `8ceab8e`
- Megatron-LM commit
  [`d3528a2`](https://github.com/NVIDIA/Megatron-LM/tree/d3528a21301db2d12e92912b3ec025dc8a2ed4d6)
- `NCCL_CUMEM_ENABLE=1`
- dedicated ProcessGroupNCCL communicator with `NCCL_CTA_POLICY_ZERO`
- PyTorch NCCL symmetric-memory allocation and rendezvous

The same standalone reproducer uses only PyTorch distributed and symmetric
memory. It does not import Primus training code or construct a model.

## User-visible symptom

The original GPT-OSS FlyDSL run appeared to stall before producing iteration
logs. Live Python stacks showed that this long pause was FlyDSL compilation,
not RCCL:

- one rank was compiling `flash_attn_bwd`;
- the other ranks were sleeping on FlyDSL's cross-process compile lock;
- the main thread was waiting in synthetic-warmup backward.

After compilation completed, training progressed. At the end of the configured
training interval, all ranks received `SIGSEGV` at the same instruction:

```text
rocr::core::Runtime::VMemoryPtrInfo(...) + 163
fault address: 0x18
```

The native stack was:

```text
ProcessGroupNCCL::allgather_into_tensor_coalesced
  -> ncclGroupEnd
  -> groupLaunch
  -> ncclLaunchCeColl
  -> ncclCeAllGather
  -> ncclCeLaunchBatchOps
  -> hipMemcpyAsync
  -> hsa_amd_pointer_info
  -> rocr::core::Runtime::PtrInfo
  -> rocr::core::Runtime::VMemoryPtrInfo
  -> SIGSEGV
```

The pointer passed to `VMemoryPtrInfo` was an imported peer VMM address, not
the rank's locally owned parameter-buffer address.

## Why the failure appeared at the third iteration

The failure is tied to final synchronous parameter synchronization, not to
iteration number three.

Megatron performs the following shutdown sequence:

1. [`train()` disables the forward pre-hooks after its loop](https://github.com/NVIDIA/Megatron-LM/blob/d3528a21301db2d12e92912b3ec025dc8a2ed4d6/megatron/training/training.py#L3098-L3107).
2. [`DDP.disable_forward_pre_hook()` calls `start_param_sync(force_sync=True)`](https://github.com/NVIDIA/Megatron-LM/blob/d3528a21301db2d12e92912b3ec025dc8a2ed4d6/megatron/core/distributed/distributed_data_parallel.py#L371-L386).
3. [`DDP.start_param_sync()` forwards `force_sync` to every bucket group](https://github.com/NVIDIA/Megatron-LM/blob/d3528a21301db2d12e92912b3ec025dc8a2ed4d6/megatron/core/distributed/distributed_data_parallel.py#L467-L487).
4. [The bucket group computes `async_op = overlap_param_gather and not force_sync`](https://github.com/NVIDIA/Megatron-LM/blob/d3528a21301db2d12e92912b3ec025dc8a2ed4d6/megatron/core/distributed/param_and_grad_buffer.py#L292-L316).
5. [It submits the synchronous AllGather through `_coalescing_manager`](https://github.com/NVIDIA/Megatron-LM/blob/d3528a21301db2d12e92912b3ec025dc8a2ed4d6/megatron/core/distributed/param_and_grad_buffer.py#L398-L424).

The three-step log showed these parameter-gather operation counts:

```text
0x01 through 0x11: 17 asynchronous bucket gathers
0x12 through 0x22: 17 asynchronous bucket gathers
0x23 through 0x33: 17 asynchronous bucket gathers
0x34:              final force_sync gather, then SIGSEGV
```

Operations `0x01` through `0x33` used a non-null RCCL stream and selected the
`hipMemcpyBatchAsync` CE path. Operation `0x34` used `stream=(nil)`, selected
the per-peer `hipMemcpyAsync` fallback, and crashed. A 50-step run would expose
the same final-sync failure after step 50 rather than specifically at step 3.

## Root cause

### RCCL trigger

RCCL explicitly treats the legacy null stream as unsupported by
`hipMemcpyBatchAsync` and falls back to one `hipMemcpyAsync` call per peer:

[`projects/rccl/src/ce_coll.cc`](https://github.com/ROCm/rocm-systems/blob/c19fc7df/projects/rccl/src/ce_coll.cc#L620-L640)

```cpp
// cudaMemcpyBatchAsync does not accept the legacy null stream.
bool isLegacyStream;
ncclCudaStreamIsLegacyNull(stream, &isLegacyStream);

if (capturing || isLegacyStream) {
  for (int i = 0; i < params->numOps; i++) {
    cudaMemcpyAsync(params->dsts[i], params->srcs[i], params->sizes[i],
                    cudaMemcpyDeviceToDevice, stream);
  }
}
```

Calling `hipMemcpyAsync` on the imported VMM peer pointer causes HIP to query
the pointer's owning agent through `hsa_amd_pointer_info`.

### ROCr null dereference

ROCr represents imported VMM handles with `MemoryHandle::region == nullptr`.
That is intentional because the backing allocation is owned by another
process/GPU.

`Runtime::VMemoryPtrInfo`, however, handles a matching VMM range as if it were
locally owned:

[`projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp`](https://github.com/ROCm/rocm-systems/blob/c19fc7df/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp#L1025-L1047)

```cpp
info->agentOwner =
    mappedHandleIt->second.mem_handle->agentOwner()->public_handle();

const AMD::MemoryRegion* memRegion =
    mappedHandleIt->second.mem_handle->region;
assert(memRegion && "MappedHandle has a MemoryHandle with NULL region");
```

`agentOwner()` calls `region->owner()`. Because `region` is null for an
imported handle, the first statement faults at address `0x18`; the assertion
below it is never reached.

## Standalone reproducer

The reproducer is:

```text
examples/mlperf/gpt_oss_20b/rccl_ce_null_stream_vmm_repro.py
```

From the Primus repository root:

```bash
IMAGE=rocm/primus:v26.5
MODE=async  # change to synchronous for the failing case

docker run --rm \
  --name rccl-ce-null-stream-repro \
  --network host \
  --ipc host \
  --privileged \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  -v "$PWD/examples/mlperf/gpt_oss_20b/rccl_ce_null_stream_vmm_repro.py:/repro.py:ro" \
  -e NCCL_CUMEM_ENABLE=1 \
  -e NCCL_LOCAL_REGISTER=0 \
  -e TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true \
  -e RCCL_MSCCL_ENABLE=0 \
  -e RCCL_MSCCLPP_ENABLE=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=lo \
  -e GLOO_SOCKET_IFNAME=lo \
  -e RCCL_CE_BATCH_ASYNC_ENABLE=1 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=INIT,P2P,ALLOC,COLL,ENV \
  "$IMAGE" \
  torchrun --standalone --nproc-per-node=8 /repro.py --mode "$MODE"
```

Expected control result:

```text
[rank N] async-stream: PASS
```

Expected failing result:

```text
[rank N] synchronous-null-stream: current stream=0x0; entering AllGather
...
exitcode: -11 (SIGSEGV)
```

The reproducer uses `os._exit(0)` after a passing test to avoid an independent
PyTorch symmetric-memory/ProcessGroupNCCL interpreter-finalization ordering
issue from obscuring the result.

## SVM observations

SVM means **Shared Virtual Memory**. KFD's `svm_range_restore_work` validates
and restores GPU page-table mappings after a CPU mapping invalidation,
migration, or eviction. It restores address mappings, not tensor contents.

SVM migration/restore activity was present during both successful and failing
direct-memory runs. A successful 40-step run also logged the workqueue warning,
so that warning is not the root cause of this crash. The debugger identifies a
userspace null dereference in ROCr pointer metadata handling.

## Ownership and recommended fixes

Megatron's synchronous final parameter gather is valid API usage. A framework
workaround could avoid the final gather when it is provably redundant, but
correctness must not depend on that.

RCCL should own the immediate integration fix:

- do not route null-stream CE operations on imported VMM pointers through the
  unsafe per-peer `hipMemcpyAsync` path;
- bridge the operation through an internal non-null stream with correct
  event ordering, or use another imported-VMM-safe copy primitive;
- add a regression test combining `ncclMemAlloc`, symmetric window
  registration, zero-CTA AllGather, and a synchronous/null-stream launch.

ROCr should receive a companion bug:

- make `VMemoryPtrInfo` handle imported `MemoryHandle` objects without calling
  `agentOwner()` through a null `region`;
- resolve ownership through imported-handle/per-agent metadata, or return an
  error instead of dereferencing null;
- add an imported VMM `hsa_amd_pointer_info` test.

## Workarounds

Known operational workarounds are:

- use an explicit non-null communication stream so RCCL selects
  `hipMemcpyBatchAsync`;
- avoid direct symmetric-buffer CE mode and use the bounded symmetric-scratch
  path;
- avoid the final synchronous gather only when the framework can prove all
  required parameters are already synchronized.

The explicit-stream workaround was validated by the standalone control. It is
not equivalent to fixing synchronous/null-stream CE support.

## Later-build status

As of 2026-09-03, the relevant code remained unfixed in
`rocm-systems/develop`:

- RCCL still used the legacy/null-stream `hipMemcpyAsync` fallback.
- ROCr still dereferenced `mem_handle->agentOwner()` before checking the
  imported handle's null region.

A newer `c19fc7d` TheRock image was also tested, but both modes failed earlier
while initializing the dedicated communicator:

```text
HIP failure 'invalid argument' at transport/p2p_tmp.cc:358
```

That P2P/CUMEM initialization failure occurred at the first group barrier,
before symmetric allocation or either AllGather mode. It is a separate issue
and does not establish whether the imported-VMM null dereference is fixed in
that binary.

