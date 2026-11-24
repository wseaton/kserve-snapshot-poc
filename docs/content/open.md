+++
title = "Open problems"
weight = 70

[extra]
nav = "Open"
+++

Checked against the code and upstream on 2026-11-24.

## Daemon

- **Peer weights.** `snapshotd node` reloads weights from the model volume. ModelExpress reload
  from a serving replica, the fastest path measured, runs only from bench pods.
- **Streamed capture.** The daemon dumps to its node store; streaming the dump to another node's RAM
  runs only from bench pods.
- **Partial-node TP>1** restores only onto the same device minors
  ([Limits](@/limits.md#tensor-parallel-below-a-whole-node)). The no-sharing configuration that
  would lift this has not been run under the daemon.

### Peer selection

- Choose the transport by distance; `UCX_TLS` is one setting for every pair.
- Remember delivered bandwidth and failures per peer.
- Weigh load against distance; nearness always wins today.
- Let a control plane name the source and place checkpoints ahead of scale-out.

## Restore

- **gpt-oss-120b does not survive `--checkpoint-release-weights`.** After reload every worker logs
  `YaRNScalingRotaryEmbedding: Failed to load weights` and output is incoherent, on one node too
  and with graphs off. Its checkpoints keep the weights (~100 GB).
- **Released weights at 0.9 utilization** captured in 10.4 s (31 GB) and got through CRIU, CUDA
  and cuInterpose restore, then ran out of memory re-committing the KV cache.
- **CUDA graph recapture** is most of the vLLM restore without cuInterpose (~7 s at TP=2, ~25 s
  for gpt-oss-120b TP=4). Capturing the batch sizes traffic needs first would hide most of it.
- **Streaming over TCP** restored but failed its first request once on an H100 cluster. Dump then
  copy works there. Cause unknown.

## Capture

- **CRIU dumps serially**, one process at a time in 4 MiB chunks: ~4.2 to 4.5 GB/s is the ceiling.
  Past it means dumping less or dumping processes in parallel.
- **~5 GB per GPU stays allocated after prepare** (2.7 GB of per-layer blocks, 1.5 GB in a private
  pool). Releasing it would take ~33 GB off a DeepSeek image. The owning vLLM objects are not
  identified.
- **Restore from a stream is slower than from files** (9.8 s against 5.1 s at TP=8; 20.2 s with
  `transfer receive --hold`): CRIU reads a stream range by range. This is why the RAM path writes
  files to tmpfs instead. `vmsplice` was not tried.
- **Target RAM.** On the bench RAM path the image holds its size in the target's memory until
  restore, charged to the pod.

## Platform

- OpenShift: restore into a pod with different MCS categories, and TP>1.
- Multi-node TP needs IB verbs released at prepare; NCCL's IB plugin leaves a `uverbs` mapping CRIU
  cannot dump.
- Whether `NCCL_CUMEM_ENABLE=0` costs anything in decode at TP=8.

## Upstream changes

| Change | Where | State |
| --- | --- | --- |
| Checkpoint lifecycle | [vllm#62570](https://github.com/vllm-project/vllm/pull/62570) | merged, in our base |
| Model loader sleep, wake and reload hooks | [vllm#57011](https://github.com/vllm-project/vllm/pull/57011), `patches/vllm/0001` | merged 2026-11-23, after our base |
| Release weights at prepare | `patches/vllm/0002` | carried |
| Recreate the NCCL communicator around a checkpoint | `patches/vllm/0003` | carried |
| Connector checkpoint hooks, NIXL prefill/decode | [vllm#64371](https://github.com/vllm-project/vllm/pull/64371), `0004`, `0007` | open |
| Free and refill the CPU KV offload region | `patches/vllm/0005` | carried, stacked on #64371 |
| Checkpoint lifecycle on the Rust frontend's gRPC service | `patches/vllm/0006` | carried, no upstream PR yet |
| Refill the offload region after serving resumes | vLLM | not started |
| Release custom all-reduce IPC buffers at prepare | vLLM | not started |
| Rebuild FlashInfer all-reduce workspaces after a device remap | FlashInfer, vLLM | not started |
| A new cuMem NCCL communicator after a GPU remap | NCCL or the driver | not reported |
| cuInterpose coordinator as a library | `patches/dynamo-snapshot/0002` | carried |
| Caller-provided NIXL buffers | [nixl#2672](https://github.com/ai-dynamo/nixl/pull/2672) | open |
| SELinux decided from the process's own label | `patches/criu/0001` | carried |
