+++
title = "Internals"
weight = 50

[extra]
mermaid = true
+++

## Checkpoint order

{% mermaid() %}
flowchart LR
  subgraph PREP["vLLM, before"]
    V1["/checkpoint/prepare<br/>drain, drop KV cache and weights,<br/>tear down communicators"]
  end
  subgraph CK["checkpoint"]
    direction LR
    C1["cuInterpose<br/>unmap GPU memory shared<br/>between processes"]
    C2["cuda-checkpoint<br/>lock every process, copy<br/>GPU state to host memory"]
    C3["CRIU<br/>dump the process tree"]
    C1 --> C2 --> C3
  end
  subgraph RS["restore"]
    direction LR
    S1["CRIU<br/>recreate the tree in the<br/>new pod's namespaces"]
    S2["cuda-checkpoint<br/>GPU state back onto the<br/>new pod's GPUs, unlock"]
    S3["cuInterpose<br/>re-import shared memory"]
    S1 --> S2 --> S3
  end
  subgraph POST["vLLM, after"]
    V2["/checkpoint/restore<br/>communicators, weights,<br/>KV cache, CUDA graphs"]
  end
  V1 --> C1
  C3 --> IMG[("image")]
  IMG --> S1
  S3 --> V2
{% end %}

- **cuInterpose** is needed only while processes share GPU memory at checkpoint time; the driver
  cannot checkpoint a mapping another process holds. `snapshotd` inspects each process over its
  control socket and runs the coordinator only if memory is shared.
- **cuda-checkpoint** must finish before CRIU, which cannot dump device memory. Restore reverses
  the order.
- **CRIU** does the rest, and is where the image size and most of the dump time go.

## Shared GPU memory at TP>1

Experiments on Qwen3-8B TP=2
([`experiments/in-place`](https://github.com/wseaton/snapshotd/tree/main/experiments/in-place),
[`experiments/criu`](https://github.com/wseaton/snapshotd/tree/main/experiments/criu)) found:

- The TP>1 restore failure on vLLM 0.27.1 is the 580 user-mode driver, not the CUDA runtime.
  Multicast and peer VMM calls after a restore fail with `CUDA_ERROR_INVALID_DEVICE` on 580 and
  succeed on the 13.1 compat `libcuda`. cuInterpose at TP>1 therefore needs the 13.1 `libcuda`.
- After upstream prepare, workers still share NCCL's cuMem buffers and custom all-reduce IPC
  buffers.
- On other GPUs: `ncclCommResume` leaves the communicator faulting, FlashInfer workspaces fail to
  rebuild (`CUDA_ERROR_NOT_INITIALIZED`), legacy IPC buffers survive `ncclCommSuspend`, and a new
  cuMem communicator fails its first all-reduce.

What works with no cuInterpose, on the same or other GPUs, is `patches/vllm/0003` plus:

```text
--disable-custom-all-reduce
NCCL_CUMEM_ENABLE=0
VLLM_ALLREDUCE_USE_FLASHINFER=0
--compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
```

0003 destroys the NCCL communicator at prepare and creates a new one at restore over legacy IPC,
releasing and recapturing CUDA graphs (~7 s at TP=2). The vLLM API server opens an `io_uring` that
CRIU cannot dump, so `io_uring_setup` is blocked by seccomp.

Without 0003, shared memory restores only onto GPUs with the same device minors
([Limits](@/limits.md#tensor-parallel-below-a-whole-node)).

## Cross-node restore requirements

- **Files.** Everything open at checkpoint, at the same path, mode and size: `/dev/shm` files, the
  server's log, JIT caches. A file outside shared storage (e.g. `~/.humming/cache`) must be on a
  volume both nodes see.
- **PIDs.** The harness raises `ns_last_pid` first. Nothing else may start a process in the pod
  during the restore; a `kubectl exec` took a needed PID. A failed restore leaves processes
  holding PIDs until killed.
- **GPU positions.** The driver places source device i on target position i (device-minor order),
  among the GPUs the container can open. The pod must own those positions.
- **Sockets.** Workers talk over loopback. CRIU keeps them with `--tcp-established`, plus Dynamo's
  `snapshot_inet_remap.so` when the pod IP changes. `--tcp-close` breaks vLLM's restore.
- **Headroom.** Without cuInterpose, 0.9 GPU memory utilization left no room to re-commit the KV
  cache after restore. 0.8 worked.

## Image transfer

[`daemon/src/transfer.rs`](https://github.com/wseaton/snapshotd/blob/main/daemon/src/transfer.rs),
`--features nixl`. Each side maps one 2 MiB-aligned `MADV_HUGEPAGE` region, registers it with NIXL
once (~0.1 s for 4 GiB) and cuts it into slots. A slot is filled by the sender, written into the
receiver's slot over RDMA, persisted, acknowledged and reused.

- `transfer send` reads files from disk.
- `transfer stream` takes `criu dump --stream`'s pipes directly; a slow receiver backs up into
  CRIU.
- `transfer receive --direct` writes with `O_DIRECT` (~3.4 GB/s, disk-bound). `--mmap` copies into
  RAM-backed files in parallel (~4.5 GB/s, bound by CRIU's serial dump).

Slower alternatives tried: plain `write` into tmpfs (one file lock serializes slots, 33 to 37 s),
and serving `criu restore --stream` from the receiver's arena (CRIU reads a stream range by range:
20.2 s restore).

## Huge-page capture

`snapshotd image hold` speaks CRIU's stream protocol directly (two protobuf messages and one passed
descriptor per file). It reads every pipe into anonymous 2 MiB huge-page segments.

- **Pipe size.** 1 MiB pipes cap the dump at ~400 MB/s. 64 MiB via `F_SETPIPE_SZ` needs
  `CAP_SYS_RESOURCE`.
- **Compaction.** Faulting huge pages during the dump caused ~75,000 compaction stalls. The pool
  must be faulted in before capture.
- **Throughput.** Rewriting faulted memory runs at 30 GB/s; first touch at 4.6 GB/s (huge pages)
  or 2.3 GB/s (4 KiB).

## What is in the image

DeepSeek-V4.1-Flash TP=8 after prepare, from
[`mem-breakdown.py`](https://github.com/wseaton/snapshotd/blob/main/experiments/bench/mem-breakdown.py)
(predicts 70.5 GB; the dump was 69.9 GB):

| | Each worker | All 8 |
| --- | --- | --- |
| GPU memory still allocated | 4.93 GB | 39.4 GB |
| Host heap | 2.05 GB | 17.1 GB |
| Host anonymous mappings | 1.6 GB | 12.6 GB |
| API server and engine core | | 1.8 GB |

| Still on one GPU | Size |
| --- | --- |
| ~500 blocks of 2 to 18 MB in the default pool, mostly sets of 40 (one per layer) | 2.7 GB |
| One block in a private pool | 1.5 GB |
| Other private pools (CUDA graphs) | 0.4 GB |
| CUDA context and other libraries | 0.3 GB |

The torch allocator still reports the KV cache's 61.8 GB after prepare (the address range stays;
the memory behind it is freed). `malloc_trim` reclaims 0.05 GB of 3.4 GB per worker.

## Privilege model

Checkpointing is privileged wherever it runs. The options:

| | Node agent (`snapshotd`, Dynamo) | Wrapper in every pod | Kubelet checkpoint API |
| --- | --- | --- | --- |
| Privileged | one DaemonSet per GPU node | every vLLM container (`CAP_CHECKPOINT_RESTORE`, `CAP_SYS_PTRACE`, often `CAP_SYS_ADMIN`) | the runtime, already root |
| Workload pod | unprivileged, non-root, `io_uring` blocked | privileged next to model code | unprivileged |
| Pod Security | one `privileged` namespace | an exception in every serving namespace | none |
| GPU ordering and remap | the agent | the wrapper | not possible |

The kubelet API is generic (no vLLM quiesce, no weight release, no GPU remap) and unsupported on
OpenShift. [snapshot-orchestrator](https://github.com/neuralmagic/snapshot-orchestrator) built the
in-pod wrapper first and moved to a node agent to get rid of privileged workload pods.

The agent is root on every GPU node and can read any process's memory. Bounds: run it only on
serving nodes, act only on labelled pods, and treat the checkpoint store like model data, since
images contain process memory.
