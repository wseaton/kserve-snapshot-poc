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
- A checkpoint records an id drawn when it is taken, the node's NVIDIA driver version, GPU
  models and compute capabilities, the first CPU's flags from `/proc/cpuinfo`, SHA-256 digests of
  CRIU and the cuInterpose libraries, the kernel release and the daemon version. Before a restore
  the driver, GPU models and tool digests must match the node's, and every CPU flag of the
  checkpoint must be one the node has. If not, the daemon removes the checkpoint, logs why, and
  the pod captures a new one. A checkpoint from before the environment or the id was recorded is
  replaced the same way; one from before CPU flags were recorded passes that check. A peer is not asked for a checkpoint whose environment does not match the asking node.
- A different kernel release is logged and the restore goes ahead. The shim version is not
  recorded. The daemon version is recorded but not compared.
- A failed restore cold boots its pod, which does not try a checkpoint again. Each restore of
  the node's checkpoint is recorded in the revision's directory against its id: the pod, and for
  a failure the step and the error. When `--condemn-after` (default 2) different pods fail it in
  a row in CRIU restore, CUDA restore, cuInterpose restore or the copy of shared memory, with no
  restore of it working in between, the daemon condemns it: it moves the image aside, deletes
  it, and keeps those records, its manifest and the last 4 KiB of CRIU's restore log under
  `condemned/<id>/`. Failures in vLLM restore (KV cache, weights) or in a pull do not count. The
  next pod asks peers first, skipping any that holds a condemned id, and captures only if none
  has another. A pulled checkpoint is checked again before it is published: one with a
  condemned id or another environment is dropped. A restore that works cuts the record down to
  itself, and a condemn moves its image's lines out with the evidence.
- A revision is recaptured at most once per node environment, whether its checkpoint went stale
  or was condemned. A stale checkpoint after that cold boots its pods; a second condemn stops
  captures of the revision on the node and its pods cold boot. Another environment, such as new
  tool digests after a daemon upgrade, starts over.
- A stale checkpoint is replaced by a capture on this node. A matching checkpoint on a peer is
  not pulled in its place.

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
- Weights are released and reloaded from the model volume. ModelExpress peer reload is not wired
  in.
- The checkpoint RPCs exist only in `patches/vllm/0006`, on the Rust frontend.

Not needed, unlike the Dynamo Snapshot integration: `SnapshotJob`/`PodSnapshot`, donor pods, pods
held unscheduled during capture, a volume every node mounts, and privileged workload pods.
