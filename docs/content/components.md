+++
title = "Components"
description = "The projects snapshotd builds on, and what it uses each one for."
weight = 45
+++

## vLLM checkpoint lifecycle

[vLLM](https://github.com/vllm-project/vllm) gained `/checkpoint/prepare`, `/checkpoint/status`
and `/checkpoint/restore` in [vllm#62570](https://github.com/vllm-project/vllm/pull/62570). Prepare
drains requests and frees the KV cache so the process is small and quiet before it is dumped.
Restore rebuilds what prepare tore down.

The patches in [`patches/vllm`](https://github.com/wseaton/snapshotd/tree/main/patches/vllm)
extend it:

- give KV connectors checkpoint hooks, and checkpoint NIXL prefill and decode instances (`0001`, `0002`)
- release the weights at prepare and reload them at restore (`0003`)
- destroy and recreate the NCCL communicator, so no GPU memory is shared at checkpoint time (`0004`)
- serve the lifecycle over the Rust frontend's gRPC `Control` service, which the daemon calls (`0005`, `0006`)

Freeing the CPU KV offload region at prepare was measured on the earlier base and is not ported
yet ([Open](@/open.md)).

## CUDA checkpoint driver API

The NVIDIA driver can move a process's GPU state into that process's host memory and back:
`cuCheckpointProcessLock`, `Checkpoint`, `Restore`, `Unlock` and `GetState`. After checkpoint the
process holds no GPU resources, so CRIU can dump it like a CPU process. Restore takes a list of
`CUcheckpointGpuPair`s to land the state on different GPUs.

NVIDIA's [`cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint) CLI wraps the same calls.
`snapshotd` loads `libcuda.so.1` at run time and calls them directly, for every PID of a pod at
once. On driver 580 the GPU map must cover every GPU visible to the process.

## CRIU

[CRIU](https://criu.org) (Checkpoint/Restore In Userspace) dumps a process tree to image files
(memory, file descriptors, sockets, namespaces) and recreates it later, on the same or another
machine.

What `snapshotd` uses:

- `criu dump --stream` / `criu restore --stream`, which send image files over pipes instead of
  writing them. `snapshotd` is the other end of that protocol, so the image goes straight to RDMA
  or to huge-page memory.
- `--tcp-established` to keep the workers' loopback connections.
- Ghost files, for unlinked files still open (the CPU KV offload region).
- Dynamo's `snapshot_inet_remap.so` plugin to rewrite socket addresses when the pod IP changes.

The build is the one in Dynamo's agent image (criu-dev `1a935857`), plus
[`patches/criu`](https://github.com/wseaton/snapshotd/tree/main/patches/criu) for SELinux labels.

## cuInterpose

Part of [Dynamo Snapshot](https://github.com/ai-dynamo/snapshot): an `LD_PRELOAD` library and a
coordinator. The driver cannot checkpoint GPU memory that another process also maps, which is the
case at TP>1 when vLLM uses custom all-reduce or NCCL over cuMem. cuInterpose records those
shared allocations, unmaps them before the checkpoint, and re-exports and re-imports them after
the restore.

`snapshotd` asks each process over cuInterpose's control socket whether it shares memory, and runs
the coordinator only when it does. With `patches/vllm/0004` nothing is shared and cuInterpose is
not needed. At TP>1 on driver 580 it needs the CUDA 13.1 compat `libcuda`.

## NIXL and UCX

[NIXL](https://github.com/ai-dynamo/nixl) (NVIDIA Inference Xfer Library) moves data between
registered memory regions on different hosts. It runs over [UCX](https://openucx.org), which picks
the transport (InfiniBand RDMA, TCP, CUDA copies) from `UCX_TLS`.

`snapshotd` registers one huge-page pool per side with NIXL and moves the CRIU image through it in
fixed-size slots. Registering a caller's buffer needs
[nixl#2672](https://github.com/ai-dynamo/nixl/pull/2672), not yet merged. The `nixl` Cargo feature
gates all of it. vLLM's NIXL connector, used for prefill and decode, is a separate use of the same
library.

## ModelExpress

[ModelExpress](https://github.com/ai-dynamo/modelexpress) serves model weights from replicas that
are already running. A small server keeps metadata about who has what; a new vLLM worker pulls its
shard GPU to GPU over NIXL instead of reading files.

On restore, vLLM's weight reload goes through ModelExpress, so a restored replica gets its weights
from a serving one: 64 GB per worker in 1.8 s for DeepSeek-V4.1-Flash. The client is patched to
read its pod identity from a downward API volume, because a restored process keeps the environment
of the pod it was checkpointed in.

## Kubernetes interfaces

- **CRI**, the container runtime's gRPC API (containerd or CRI-O), gives the daemon a pod's
  container, its PID and its OCI spec.
- **Kubelet PodResources**, a node-local gRPC socket, lists which GPUs each pod was allocated.
- **[kube-rbac-proxy](https://github.com/brancz/kube-rbac-proxy)** sits in front of each daemon's
  peer API. It checks a caller's ServiceAccount token with `TokenReview` and its permissions with
  `SubjectAccessReview`, so daemons authenticate each other with plain Kubernetes RBAC.
- **KServe `LLMInferenceServiceConfig`** layers are merged into a service's pod template. The
  `snapshotd` layer is how a KServe service opts in without a webhook.

## Dynamo Snapshot

[NVIDIA Dynamo Snapshot](https://github.com/ai-dynamo/snapshot) is where this started. It is a Go
operator, node agent and sidecar that checkpoints inference servers into a shared volume.
`snapshotd` replaces its node agent and keeps its CRIU build, cuInterpose and network-remap plugin.
[Comparison](@/comparison.md) has the differences.
