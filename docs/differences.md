# How this differs from NVIDIA Dynamo Snapshot

Both systems restore an initialized vLLM server from a CRIU checkpoint instead of cold-booting
it. This page says what is shared, where the two part ways, what each optimization here is worth,
and where Dynamo Snapshot is ahead.

Numbers for Dynamo Snapshot come from two places and are marked: **published** is NVIDIA's
[2026-11-10 post](https://developer.nvidia.com/blog/start-and-restore-gpu-workloads-in-seconds-with-nvidia-snapshot)
(B200, VAST NFS); **measured** is Dynamo Snapshot run on the same H200 cluster as everything else
here ([BENCHMARKS.md](benchmarks.md)).

## The short version

DeepSeek-V4.1-Flash, vLLM, 8 GPUs with tensor and expert parallel:

| | Dynamo Snapshot, published | This repository |
| --- | --- | --- |
| What is in the checkpoint | process state and weights, 647 GB | process state only, 74 GB |
| Where the checkpoint goes | a shared volume | the target node's RAM, over InfiniBand, during the dump |
| Checkpoint time | 274.8 s | 25.5 s, image already on the second node |
| Where restore gets the weights | the checkpoint | a serving replica, over InfiniBand |
| Restore | 31.0 s | 20.5 s |
| Restore with no replica serving | 31.0 s | 101.5 s from a cold shared volume |

Different GPUs and storage, one run on our side, and NVIDIA does not say whether its restore was
on another node. Details in [experiments/p2p-transfer.md](cross-node.md).

## What is the same

- **CRIU.** We run the CRIU 4.2.1 build that ships in Dynamo's snapshot agent image, with its
  restore optimizations. CRIU restore time is a tie: 5.0 s here, 4.2 to 10.8 s published.
- **The driver's checkpoint API.** GPU state is saved and restored by the NVIDIA driver
  (`cuCheckpointProcess{Lock,Checkpoint,Restore,Unlock}`), the same calls `cuda-checkpoint` makes.
- **cuInterpose.** Tensor parallel above 1 on stock vLLM uses Dynamo's `libcuinterpose.so` and its
  coordinator unchanged.
- **vLLM's checkpoint lifecycle** (`/checkpoint/prepare`, `/checkpoint/restore`), which releases
  the KV cache before the dump in both systems.

## Where we differ

### 1. Weights are not in the checkpoint

Dynamo Snapshot checkpoints the weights with the process: they are written into the CRIU image at
capture and put back on the GPUs by the driver at restore. NVIDIA's
[2026-05-27 post](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/)
describes a Global Memory Service that keeps the weights in a separate artifact and restores them
alongside the process (gpt-oss-120b: 129 GiB checkpoint down to 6.7 GiB plus 74 GiB of weights),
not shipped then because it needed a driver patch. The
[2026-11-10 post](https://developer.nvidia.com/blog/start-and-restore-gpu-workloads-in-seconds-with-nvidia-snapshot)
does not mention it, and its checkpoints are larger than the weights (647 GB for 510 GB of
DeepSeek-V4.1-Flash weights).

Here vLLM frees the weights at prepare (`--checkpoint-release-weights`,
[patches/vllm](../patches/vllm)) and loads them again after the driver restore.

| Effect | Dynamo Snapshot | Here |
| --- | --- | --- |
| Checkpoint size, DeepSeek TP=8 | 647 GB published, 73 GB measured with our vLLM patches | 74 GB |
| CUDA restore, 8 GPUs | 18 to 29 s published | 7.3 s |
| Cost moved to restore | none | the weight reload: 7.0 s from a peer, 88.4 s from a cold volume |

The same change is applied to CPU KV offload memory ([patch 0005](../patches/vllm)): with a 128 GiB
offload region the checkpoint is 139.4 GB kept and 11.4 GB released.

### 2. Weights come from a serving replica

A restored worker pulls its weights from a replica that is already serving, GPU to GPU over
InfiniBand through ModelExpress: 64 GB to each of 8 workers at 290 Gbps, 1.8 s on the wire, 7.0 s
for vLLM's whole restore step. From the shared volume the same step is 30.1 s with the files in
page cache and 88.4 s without.

This is why scale-out is fast and scale-from-zero is not: with nothing serving there is no peer.

### 3. The checkpoint is streamed to the target, not written to a volume

Dynamo Snapshot's storage backend is a volume claim. Measured here, dumping DeepSeek TP=8 to the
shared NFS volume took 3m52s at about 315 MB/s.

`snapshotd` is the other end of `criu dump --stream`. CRIU's pipes fill a registered memory pool
that is sent over RDMA (NIXL) while the dump runs, and the target keeps the image in RAM. The
source node writes nothing. The dump and the copy together take the same 16 to 18 s as the dump
alone: CRIU's serial dump, about 4.2 GB/s, is the limit and not the network.

### 4. The CUDA checkpoint runs every process at once

Dynamo's agent starts one helper process for each PID and each action, in sequence.
`snapshotd` loads the driver library once and calls it for every PID in parallel. Measured on
DeepSeek TP=8 (10 CUDA processes): 14.4 s with the helper, 3.8 s with `snapshotd`.

### 5. One node agent in Rust, application-specific

Dynamo Snapshot is a Go DaemonSet agent, driven by its operator, generic over vLLM, SGLang and
TensorRT-LLM. `snapshotd` is a Rust daemon for vLLM under KServe only. The reasons, the privilege
model and what KServe has to lifecycle are in [DESIGN.md](design.md) and
[ARCHITECTURE.md](architecture.md). The kubelet checkpoint API is not used by either.

## Our optimizations, and what each is worth

| Optimization | Where | Before | After |
| --- | --- | --- | --- |
| Release weights at prepare | vLLM patch | 647 GB image (published) | 74 GB |
| Release CPU KV offload memory at prepare | vLLM patch 0005 | 139.4 GB image | 11.4 GB |
| Weights from a serving peer over InfiniBand | ModelExpress | 88.4 s reload, cold volume | 7.0 s |
| Driver calls for all PIDs in parallel | `daemon/src/cuda.rs` | 14.4 s CUDA checkpoint | 3.8 s |
| Stream CRIU into an RDMA sender during the dump | `daemon/src/transfer.rs` | 98.0 s dump then copy (gpt-oss-120b TP=4) | 33.0 s |
| Image held in the target's RAM, restore from it | `daemon/src/image.rs`, RAM-backed directory | 26.3 s restore from disk (gpt-oss-120b TP=4) | 17.2 to 18.6 s |
| 64 MiB pipes between CRIU and the receiver | `daemon/src/image.rs` | 30.1 s dump at the 1 MiB default | 6.6 s |
| Huge-page pool faulted in before the dump | `daemon/src/image.rs` | 692.9 s dump, pages faulted during it | 16.9 s |
| One registered huge-page pool cut into slots | `daemon/src/transfer.rs` | | 0.1 s to register 4 GiB, once |
| Direct I/O to disk, mapped writes to tmpfs | `daemon/src/transfer.rs` | 1.5 to 2.8 GB/s buffered | 3.5 GB/s to disk, 4.5 GB/s to RAM |
| Compile caches on a shared volume | bench pod | | cold boot 104.7 s; the target maps the same files |

What a cross-node restore needed beyond speed, all in
[experiments/p2p-transfer.md](cross-node.md#what-a-cross-node-restore-needs):

- **GPU placement by position.** The driver restores each device to the same enumeration position
  it had on the source, so the pod must own those positions on the target and the device map must
  be in source enumeration order.
- **Established TCP connections.** Workers talk over loopback sockets; CRIU keeps them with
  `--tcp-established` and Dynamo's address-remap plugin when the pod IP changes.
- **PIDs.** Nothing may be started in the target pod while CRIU restores, and its PID counter is
  moved out of the way first.
- **Files mapped from outside shared storage.** A JIT cache in the container filesystem has to be
  on a volume both nodes see.

## Where Dynamo Snapshot is ahead

- **Scale from zero.** With no replica serving, its restore is still 31 s. Ours is 101.5 s from a
  cold shared volume, 42.9 s if the weights happen to be in page cache.
- **It needs only a volume.** We need InfiniBand between the nodes and a live peer.
- **It is a product.** Operator, DaemonSet, restore into a new pod, three engines, published
  results on more than 20 configurations. `snapshotd`'s CUDA, cuInterpose, image and transfer parts are
  built and measured from a privileged bench pod; the daemon that ties them into KServe is design.
- **Evidence.** Our DeepSeek two-node number is one run. The harness's exact-match output check
  cannot pass for this model even without a checkpoint, so correctness there rests on coherent
  output.

## Not like for like

- B200 and VAST NFS against H200, InfiniBand and RAM.
- NVIDIA's cold start for this model is 1110.7 s; ours is 104.7 to 166.1 s. Speedup ratios are
  not comparable, so this page compares seconds.
- Both sets of times exclude image pull and container setup.
- In our peer run the source process stops at the dump and a second server is cold-booted on the
  source node to act as the peer. A source that keeps serving through the capture is not
  exercised.
