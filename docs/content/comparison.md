+++
title = "Comparison with Dynamo Snapshot"
weight = 40

[extra]
nav = "Comparison"
+++

`snapshotd` started from [NVIDIA Dynamo Snapshot](https://github.com/ai-dynamo/snapshot) and
replaces its node agent. "Published" numbers are from NVIDIA's
[2026-11-10 post](https://developer.nvidia.com/blog/start-and-restore-gpu-workloads-in-seconds-with-nvidia-snapshot)
(B200, VAST NFS). "Measured" numbers are Dynamo Snapshot `bc9d216` on our H200s.

## Summary

DeepSeek-V4.1-Flash, vLLM, TP=8 + EP:

| | Dynamo Snapshot, published | `snapshotd` |
| --- | --- | --- |
| In the checkpoint | process state and weights, 647 GB | process state, 74 GB |
| Checkpoint goes to | a shared volume | the target's RAM over InfiniBand, during the dump |
| Checkpoint time | 274.8 s | 25.5 s, image already on the second node |
| Weights at restore | from the checkpoint | from a serving replica over InfiniBand |
| Restore | 31.0 s | 20.5 s |
| Restore, nothing serving | 31.0 s | 101.5 s from a cold volume |

Different GPUs and storage, one run on our side, and NVIDIA does not say whether its restore was on
another node. Cold boot is 1110.7 s published and 104.7 to 166.1 s here, so compare seconds, not
speedups. Both exclude image pull and container setup.

## Shared

- CRIU 4.2.1 from Dynamo's agent image. CRIU restore: 5.0 s here, 4.2 to 10.8 s published.
- GPU state through the driver's `cuCheckpointProcess*` calls.
- Dynamo's `libcuinterpose.so` and coordinator, unchanged, for TP>1 on stock vLLM.
- vLLM's `/checkpoint/prepare` and `/checkpoint/restore`.

## Differences

| Change | Where | Before | After |
| --- | --- | --- | --- |
| Release weights at prepare | `patches/vllm/0003` | 647 GB image (published) | 74 GB |
| Release CPU KV offload at prepare | vLLM patch, not yet ported | 139.4 GB image | 11.4 GB |
| Weights from a serving peer | ModelExpress | 88.4 s reload, cold volume | 7.0 s |
| Driver calls for all PIDs in parallel | `daemon/src/cuda.rs` | 14.4 s CUDA checkpoint | 3.8 s |
| Stream CRIU into an RDMA sender | `daemon/src/transfer.rs` | 98.0 s dump then copy | 33.0 s |
| Restore from the target's RAM | `daemon/src/image.rs` | 26.3 s restore from disk | 17.2 to 18.6 s |
| 64 MiB CRIU pipes | `daemon/src/image.rs` | 30.1 s dump | 6.6 s |
| Huge-page pool faulted before the dump | `daemon/src/image.rs` | 692.9 s dump | 16.9 s |
| Direct I/O to disk, mapped writes to tmpfs | `daemon/src/transfer.rs` | 1.5 to 2.8 GB/s | 3.5 GB/s disk, 4.5 GB/s RAM |

Notes on the rows:

- **Weights.** Dynamo's PageBroker path (its default) needs a newer driver than 580, so every
  measured Dynamo number uses the path where CRIU dumps GPU memory. With our vLLM patches its
  checkpoint is 73 GB and CUDA restore 11.3 s, against 7.3 s here. gpt-oss-120b cannot release
  its weights yet, so its checkpoints are 100 to 108 GB.
- **Peer weights.** Scale-out only: with nothing serving there is no peer.
- **Streaming.** Needs [nixl#2672](https://github.com/ai-dynamo/nixl/pull/2672), unmerged. CRIU's
  serial dump (~4.2 GB/s) is the limit, not the network. Off InfiniBand, dump then copy: one
  streamed run over TCP on an H100 cluster restored but failed its first request.
- **Parallel driver calls.** Compared against Dynamo's per-PID helper. PageBroker makes the calls
  from one process and was not measured.
- **Shape.** Dynamo Snapshot is a Go operator and DaemonSet with a C++ sidecar, generic over vLLM,
  SGLang and TensorRT-LLM. `snapshotd` is a Rust DaemonSet, vLLM only. Neither uses the kubelet
  checkpoint API.

## What Dynamo Snapshot does that this does not

- Scale from zero at 31 s. Ours is 101.5 s from a cold volume, 42.9 s from page cache.
- Run on a shared volume alone. Our times need RDMA between nodes and, with weights released, a
  serving peer. Over TCP the copy ran at 2.3 Gbit/s.
- Support three engines, with published results on 22 configurations.
- `snapshotd node` releases the weights and reloads them from the model volume. Peer weights and
  streamed capture have run only from bench pods.
- In our peer run the source stops at the dump and a second server is cold-booted as the peer. A
  source that keeps serving through capture is not exercised.
