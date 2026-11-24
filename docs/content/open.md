+++
title = "Open problems"
weight = 70

[extra]
nav = "Open"
+++

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
- **Restore from a stream is slower than from files** (9.8 s against 5.1 s at TP=8): CRIU cannot
  seek a stream. `vmsplice` was not tried.
- **The RDMA leg from `image hold` memory** is not written; `transfer` copies through its own pool.
- **Host memory.** The image needs its size in RAM on the target until restore, charged to the pod.
- **cuInterpose state and the manifest** are still files beside the stream.

## Peer selection

- Choose the transport by distance; `UCX_TLS` is one setting for every pair.
- Remember delivered bandwidth and failures per peer.
- Weigh load against distance; nearness always wins today.
- Let a control plane name the source and place checkpoints ahead of scale-out.

## Platform

- OpenShift: restore into a pod with different MCS categories, and TP>1.
- Multi-node TP needs IB verbs released at prepare; NCCL's IB plugin leaves a `uverbs` mapping CRIU
  cannot dump.
- Whether `NCCL_CUMEM_ENABLE=0` costs anything in decode at TP=8.

## Upstream changes

| Change | Where | State |
| --- | --- | --- |
| Recreate the NCCL communicator around a checkpoint, recapturing graphs | vLLM, `patches/vllm/0003` | tested at TP=2 onto other GPUs |
| Free and refill the CPU KV offload region, on [vllm#64371](https://github.com/vllm-project/vllm/pull/64371) | vLLM, `patches/vllm/0005` | tested at 16 and 128 GiB |
| Checkpoint lifecycle on the Rust frontend's gRPC `Control` service | vLLM, `patches/vllm/0006` | in use |
| NIXL prefill and decode checkpointing, second commit of #64371 | vLLM, `patches/vllm/0007` | in use |
| Refill the offload region after serving resumes | vLLM | not started |
| Release custom all-reduce IPC buffers at prepare | vLLM | not started; removes `--disable-custom-all-reduce` |
| Rebuild FlashInfer all-reduce workspaces after a device remap | FlashInfer, vLLM | not started |
| A new cuMem NCCL communicator after a GPU remap | NCCL or the driver | not reported |
| cuInterpose coordinator as a library | ai-dynamo/snapshot, `patches/dynamo-snapshot/0002` | proposed |
| Caller-provided NIXL buffers | [nixl#2672](https://github.com/ai-dynamo/nixl/pull/2672) | open |
| SELinux decided from the process's own label | CRIU, `patches/criu/0001` | in use |
