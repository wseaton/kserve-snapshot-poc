# kserve-snapshot-poc

Scale out KServe `LLMInferenceService` replicas by restoring vLLM from a
[Dynamo Snapshot](https://github.com/ai-dynamo/snapshot) CRIU checkpoint instead of cold-booting it,
with the weights left out of the checkpoint and refilled on restore from a running replica over
RDMA through [ModelExpress](https://github.com/ai-dynamo/modelexpress).

This repository holds every change the proof of concept makes to upstream projects as patch files
against pinned commits, so each change reads as a diff, plus the KServe integration as a submodule.

## Layout

| Path | What it is |
| --- | --- |
| [`kserve-dynamo-snapshot/`](https://github.com/wseaton/kserve-dynamo-snapshot) | Submodule. The KServe integration: an `LLMInferenceServiceConfig` layer, the snapshot entrypoint that wraps `vllm serve`, a companion admission webhook and controller that capture a revision on its first scale-up and restore later replicas from it, and the end-to-end smoke tests. |
| [`patches/vllm/`](patches/vllm) | vLLM changes on `a2bcb565` (a nightly that includes the checkpoint lifecycle from [vllm-project/vllm#62570](https://github.com/vllm-project/vllm/pull/62570)). |
| [`patches/modelexpress/`](patches/modelexpress) | ModelExpress changes on `main` (`8b2e8787`), which includes in-place reload from [ai-dynamo/modelexpress#854](https://github.com/ai-dynamo/modelexpress/pull/854). |
| [`patches/dynamo-snapshot/`](patches/dynamo-snapshot) | Dynamo Snapshot agent changes on `bc9d216` for CoreWeave's GPU device plugin. |
| [`upstream.tsv`](upstream.tsv) | Each project's upstream repository, base commit and fork branch. |
| [`checkout.sh`](checkout.sh) | `./checkout.sh vllm` clones a project at its base commit into `src/` and applies its patches. |

## The patches

**vLLM**
- `0001`: [vllm-project/vllm#57011](https://github.com/vllm-project/vllm/pull/57011) (open), carried
  unchanged. Sleep, wake and in-place reload hooks on `BaseModelLoader`, which ModelExpress uses
  to stop serving before its memory is freed and to refill weights from a peer.
- `0002`: `--checkpoint-release-weights`. `/checkpoint/prepare` also discards the weights, so the
  checkpoint does not contain them, and `/checkpoint/restore` wakes them and reloads them through
  the model loader before the KV cache. Upstream #62570 keeps the weights in GPU memory, and so in
  the checkpoint.

**ModelExpress**
- `0001`: read `POD_NAME`, `POD_NAMESPACE` and `POD_UID` from a downward API volume named by
  `MX_POD_INFO_DIR`, and resolve the worker host from the network. A process restored into another
  pod keeps the environment of the pod it was checkpointed in, so without this it would publish
  itself to the ModelExpress server as the donor.

**Dynamo Snapshot**
- `0001`: accept CoreWeave's volume-mounts `NVIDIA_VISIBLE_DEVICES`, skip the per-GPU device-list
  marker mounts at dump, and log every CRIU error line on a failed restore.

## How a scale-out works

1. A new revision's first replica is held unscheduled with a scheduling gate. A `SnapshotJob`
   donor runs on the capacity it would use, loads the model, warms up and calls
   `/checkpoint/prepare`, which drains, frees the KV cache and the weights and tears down the
   communicators. The agent checkpoints it.
2. The held replica, and every replica added later, restores from that checkpoint and calls
   `/checkpoint/restore`. vLLM rebuilds the communicators and reloads the weights through
   ModelExpress, which pulls them from a running replica over RDMA, or from disk when there is
   none, and publishes the replica as a source for the next one.

## Status

The restore path built on these patches runs end to end (2026-11-05, Qwen3-8B TP=1 on 8x H200
with ConnectX-7 InfiniBand, scaled out one replica at a time to 4):

| Replica | Weights | Ready |
| --- | --- | --- |
| 1, held for capture | from disk through ModelExpress, no peer yet | 156 s after creation, including the donor's cold boot and the capture |
| 2 to 4 | 16.4 GB from a running replica over RDMA in 0.64 to 1.73 s (76 to 205 Gbps), inside `/checkpoint/restore` | 37 to 41 s after scale-up |
| cold boot | ModelExpress's normal load from a peer, 0.73 s | 89 s |

Every completion was token-identical to the first replica's, and each restored replica published
itself to ModelExpress under its own pod identity.

An earlier version of the restore path (a vLLM worker extension driven over vLLM's dev
endpoints) also ran at TP=2 and TP=8 with replicas on two nodes: each rank pulled its shard from
its peer rank on the other node (8.2 GB in 0.24 s at TP=2, 2.06 GB per rank in under 0.2 s at
TP=8). Those have not been rerun on the patches above yet. Details are in the submodule's README.
