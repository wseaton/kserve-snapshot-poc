+++
title = "Validated configurations"
description = "Every configuration a checkpoint and restore has passed on, and the ones that failed."
weight = 35

[extra]
nav = "Validated"
+++

A row is a configuration that ran end to end: cold boot, checkpoint, restore, then the restored
server's greedy completion compared with the source's (MATCH), or requests answered correctly
where exact matching is not possible. Add a row for every run, failures included.

Stacks:

- **old**: vLLM nightly `a2bcb565` with
  [`checkpoint-patchset-with-pd`](https://github.com/wseaton/vllm/tree/checkpoint-patchset-with-pd),
  CRIU and cuInterpose from Dynamo Snapshot `bc9d216`.
- **main**: vLLM nightly `f2f9edca` with
  [`patches/vllm`](https://github.com/wseaton/snapshotd/tree/main/patches/vllm) 0001 to 0007,
  image `vllm-checkpoint:sha-f81db2f`.

Paths: **bench**, a privileged bench pod driven by `restore-bench.sh` or `lifecycle.sh`;
**node**, `snapshotd node` restoring into unprivileged pods; **kind**, the e2e suites with a stand-in
server.

## GPU runs

| Date | Stack | Path | Model | Parallelism | Weights | Also | Restored onto | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-25 | main | bench | Qwen3-8B | TP=2 | released, volume | CPU offload 16 GiB, released; KV loaded back after restore | rotated GPUs | MATCH |
| 2026-11-25 | main | bench | Qwen3-8B | TP=2 | released, volume | CPU offload 16 GiB under async_init; KV loaded back after restore | rotated GPUs | MATCH |
| 2026-11-25 | main | bench | Qwen3-8B | TP=2 | released, volume | Qwen3-0.6B draft model, 3 speculative tokens | rotated GPUs | MATCH |
| 2026-11-25 | main | bench | Qwen2.5-VL-3B-Instruct | TP=2 | released, volume | encoder CUDA graphs; text prompt only | rotated GPUs | MATCH |
| 2026-11-25 | main | bench | gpt-oss-120b | TP=4 | released, volume | | rotated GPUs | MATCH |
| 2026-11-24 | main | node | Qwen3-8B | TP=1, P/D, DisaggregatedSet | released, volume | NIXL over TCP, roles keyed apart | same node | 60/60 |
| 2026-11-24 | main | node | Qwen3-8B | TP=1, P/D, DisaggregatedSet | released, volume | second pod of each role, all four pairings | same node, and other node pulled | 160/160 |
| 2026-11-24 | main | node | Qwen3-8B | TP=1, P/D, DisaggregatedSet | released, volume | decode-only template change, both roles recaptured | same node | 40/40 |
| 2026-11-24 | main | bench | Qwen3-30B-A3B | DP=2 TP=1 EP | released, volume | CUDA graphs | rotated GPUs | MATCH |
| 2026-11-24 | main | bench | Qwen3-30B-A3B | TP=2 EP | released, volume | CUDA graphs | rotated GPUs | MATCH |
| 2026-11-18 | old | node | Qwen3-8B | TP=1 | released, volume | Deployment, KEDA | other node, pulled | answered |
| 2026-11-18 | old | node | Qwen3-8B | TP=1 | released, volume | KServe | other node, pulled | answered |
| 2026-11-18 | old | node | Qwen3-8B | TP=1, P/D | released, volume | NIXL over TCP | other node, pulled | 260/260 |
| 2026-11-18 | old | node | Qwen3-8B | TP=1, P/D | released, volume | KServe, EPP | other node, pulled | 280/280 |
| 2026-11-18 | old | node | Qwen3-0.6B | TP=1 | released, volume | OpenShift, SELinux, H100 | other GPUs | answered |
| 2026-11-14 | old | bench | DeepSeek-V4.1-Flash | TP=8 EP | released, peer | cuInterpose | other node | coherent |
| 2026-11-14 | old | bench | DeepSeek-V4.1-Flash | TP=8 EP | released, volume | cuInterpose | other node | coherent |
| 2026-11-13 | old | bench | gpt-oss-120b | TP=4 | kept | cuInterpose | other node | MATCH |
| 2026-11-13 | old | bench | gpt-oss-120b | TP=4 | kept | no cuInterpose | other node | MATCH |
| 2026-11-13 | old | bench | gpt-oss-120b | TP=4 | released | | same and other node | **FAIL**: incoherent after reload |
| 2026-11-13 | old | bench | Qwen3-8B | TP=4 | kept, released, peer | cuInterpose | other node | MATCH |
| 2026-11-09 to 11 | old | bench | Qwen3-8B | TP=2, TP=4 | released, volume | cuInterpose | rotated GPUs | MATCH |
| 2026-11-09 to 11 | old | bench | Qwen3-8B | TP=2, TP=4 | released, volume | no cuInterpose | rotated GPUs | MATCH |
| 2026-11-09 to 11 | old | bench | Qwen3-8B | TP=2 | released, volume | CPU offload 16, 128 GiB, kept and released | rotated GPUs | MATCH |

"coherent": the DeepSeek server gives different greedy completions to repeated requests even
without a checkpoint, so its restores were checked for coherent output instead.

## Without GPUs

| Suite | Covers | Runs |
| --- | --- | --- |
| `daemon/e2e/run.sh` | capture, restores, new revisions, servers that start prepared, store classes, metrics | CI, every daemon change |
| `daemon/e2e/peers.sh` | pulls between three nodes, nearest holder, kube-rbac-proxy, concurrent pulls | CI |
| `daemon/e2e/disaggregated.sh` | DisaggregatedSet roles under one workload, revision changes, multi-pod groups cold boot | CI |

## Not validated yet on main

- Prefill and decode through the llm-d router and its rollout screener, and with KV over RDMA.
- DeepSeek-V4.1-Flash TP=8 across two nodes, streamed into RAM, with weights from a serving peer.
- OpenShift, and KServe with the daemon.
- An image request to a restored multimodal server.
- Multi-pod data and expert parallel groups, which the daemon does not capture yet.
