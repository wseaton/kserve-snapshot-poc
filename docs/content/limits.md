+++
title = "Limits"
weight = 60
+++

## Tensor parallel below a whole node

A server whose workers share GPU memory (TP>1 with vLLM's defaults, under cuInterpose) restores
only onto GPUs with the same device minors. On other GPUs the daemon logs why and lets the pod cold
boot. A process can re-import shared memory only on a device it had initialized at checkpoint.
Qwen3-8B TP=2 and TP=4, driver 580.126.09:

| Restore lands on | Container sees | Result |
| --- | --- | --- |
| the same GPUs | its own | works |
| the same GPUs, other order | its own | works |
| another node, same minors | every GPU | works |
| other minors | its own | `CUDA_ERROR_INVALID_DEVICE` |
| other minors | every GPU | `CUDA_ERROR_NOT_INITIALIZED` |

Same with cuInterpose from Dynamo `main` of 2026-11-18. Pairing by PCI order, aliasing
`/dev/nvidiaN` and widening the restore map change nothing.

A whole-node pod (TP=8 on 8 GPUs) always gets the same minors. A partial-node pod restores only
where the kubelet happens to hand it the same minors. The fix is to stop sharing memory
([Internals](@/internals.md#shared-gpu-memory-at-tp-1)), not yet run on this path.

## What an orchestrator still has to add

`snapshotd node` makes every decision a control plane would make by itself, with a fixed rule.
Each gap below is the current behavior, then what is missing.

**Safety**

- Opting in takes only a pod label, so it needs admission policy: the label makes a privileged
  daemon enter the pod.
- Every daemon runs as one ServiceAccount and may pull any revision. Missing: per-node identity,
  NetworkPolicy on the proxy and NIXL ports.
- Checkpoints are unencrypted process memory (credentials, request data) on the node's disk.
- Every labelled pod is tried. Missing: refusing multi-node workers, runtime LoRA and cache-aware
  routing before paying for a failed checkpoint.

**Correctness**

- The revision key is `pod-template-hash`. It misses model files changed in place, images by tag
  and a recreated Deployment.
- GPU model, driver, kernel and CPU features are not recorded or checked before a restore.
- A failed restore is remembered for that pod only; the next pod tries the same checkpoint.
- The shim, daemon, CRIU and cuInterpose versions are not pinned to the checkpoint.

**Placement**

- The scheduler does not know which nodes hold a checkpoint.
- Every node captures the first pod of each revision it sees, with no cluster-wide claim and no
  per-node limit.
- Rollouts capture again on every node.

**Lifecycle**

- Store classes remove revisions by age. Missing: deletion with the workload, cluster-wide quota,
  eviction under pressure, an API to list and delete checkpoints.
- A dying daemon strands pods mid-operation; a pod with no daemon cold boots after 30 s.
- Connections other clients hold to the server are dead after restore.
- Only one container per pod (`--container`) is handled.
- Weights stay in the checkpoint; peer reload is not wired in.
- The checkpoint RPCs exist only in `patches/vllm/0006`, on the Rust frontend.

Not needed, unlike the Dynamo Snapshot integration: `SnapshotJob`/`PodSnapshot`, donor pods, pods
held unscheduled during capture, a volume every node mounts, and privileged workload pods.
