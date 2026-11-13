# kserve-snapshot-poc

Scale out KServe `LLMInferenceService` replicas by restoring vLLM from a
[Dynamo Snapshot](https://github.com/ai-dynamo/snapshot) CRIU checkpoint instead of cold-booting it,
with the weights left out of the checkpoint and refilled on restore from a running replica over
RDMA through [ModelExpress](https://github.com/ai-dynamo/modelexpress).

This repository holds every change the proof of concept makes to upstream projects as patch files
against pinned commits, so each change reads as a diff, plus the KServe integration as a submodule.

## Where things are

- **The Rust daemon, `snapshotd`**: source in [`daemon/`](daemon) ([README](daemon/README.md)),
  design in [DESIGN.md](DESIGN.md). It replaces Dynamo Snapshot's node agent; the CUDA checkpoint
  and cuInterpose parts are built, the rest is design.
- **Numbers**: [BENCHMARKS.md](BENCHMARKS.md), summarized under [Results](#results).
- **What KServe has to manage, and where another snapshot engine plugs in**:
  [ARCHITECTURE.md](ARCHITECTURE.md).
- **The KServe integration**: [`kserve-dynamo-snapshot/`](https://github.com/wseaton/kserve-dynamo-snapshot).
- **Upstream changes**: [`patches/`](patches), described under [The patches](#the-patches).

## Layout

| Path | What it is |
| --- | --- |
| [`kserve-dynamo-snapshot/`](https://github.com/wseaton/kserve-dynamo-snapshot) | Submodule. The KServe integration: an `LLMInferenceServiceConfig` layer, the snapshot entrypoint that wraps `vllm serve`, a companion admission webhook and controller that capture a revision on its first scale-up and restore later replicas from it, and the end-to-end smoke tests. |
| [`patches/vllm/`](patches/vllm) | vLLM changes on `a2bcb565` (a nightly that includes the checkpoint lifecycle from [vllm-project/vllm#62570](https://github.com/vllm-project/vllm/pull/62570)). |
| [`patches/modelexpress/`](patches/modelexpress) | ModelExpress changes on `main` (`8b2e8787`), which includes in-place reload from [ai-dynamo/modelexpress#854](https://github.com/ai-dynamo/modelexpress/pull/854). |
| [`patches/dynamo-snapshot/`](patches/dynamo-snapshot) | Dynamo Snapshot agent changes on `bc9d216` for CoreWeave's GPU device plugin. |
| [`DESIGN.md`](DESIGN.md) | A Rust node daemon in place of Dynamo Snapshot, and the TP=2 experiments behind it. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | What KServe has to lifecycle for snapshot scale-out, with diagrams, and the boundary a different snapshot engine would plug into. |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Cold boot against restore at TP=2 and TP=4, `snapshotd` against Dynamo's CUDA path, and CPU KV offload kept or deferred. |
| [`daemon/`](daemon) | `snapshotd`, the Rust node path so far: CUDA checkpoint through the driver API with GPU remap, and the cuInterpose integration. |
| [`experiments/`](experiments) | Scripts and pod specs for the experiments in DESIGN.md. |
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
- `0003`: `/checkpoint/prepare` destroys the NCCL communicator and `/checkpoint/restore` creates a
  new one, releasing CUDA graphs before and capturing them again after. At TP>1 with custom
  all-reduce off, no CUDA memory is shared between workers when the process is checkpointed, and
  cuda-checkpoint alone restores them, onto other GPUs too.
- `0004`: the connector hooks of [vllm-project/vllm#64371](https://github.com/vllm-project/vllm/pull/64371)
  (open), its first commit backported: KV connectors get `checkpoint_prepare` and
  `checkpoint_restore`, and the engine refuses to prepare with a connector that does not declare
  `supports_checkpoint`. Its NIXL prefill/decode commit is not carried.
- `0005`: stacked on those hooks. With the CPU KV offload connector (`--kv-offloading-size`),
  `/checkpoint/prepare` frees the offload region and `/checkpoint/restore` refills it, so the
  checkpoint does not grow by the region size. No flag: stock vLLM keeps the region.

**ModelExpress**
- `0001`: read `POD_NAME`, `POD_NAMESPACE` and `POD_UID` from a downward API volume named by
  `MX_POD_INFO_DIR`, and resolve the worker host from the network. A process restored into another
  pod keeps the environment of the pod it was checkpointed in, so without this it would publish
  itself to the ModelExpress server as the donor.

**Dynamo Snapshot**
- `0001`: accept CoreWeave's volume-mounts `NVIDIA_VISIBLE_DEVICES`, skip the per-GPU device-list
  marker mounts at dump, and log every CRIU error line on a failed restore.
- `0002`: expose cuInterpose's coordinator as a library, so a node agent can run its phases
  in-process. Proposed for upstream; see [DESIGN.md](DESIGN.md).

## How a scale-out works

1. A new revision's first replica is held unscheduled with a scheduling gate. A `SnapshotJob`
   donor runs on the capacity it would use, loads the model, warms up and calls
   `/checkpoint/prepare`, which drains, frees the KV cache and the weights and tears down the
   communicators. The agent checkpoints it.
2. The held replica, and every replica added later, restores from that checkpoint and calls
   `/checkpoint/restore`. vLLM rebuilds the communicators and reloads the weights through
   ModelExpress, which pulls them from a running replica over RDMA, or from disk when there is
   none, and publishes the replica as a source for the next one.

## Results

Qwen3-8B on 8x H200 nodes, weights left out of the checkpoint and reloaded from a network volume,
the restored server checked against the source with a greedy completion. Full tables, setup and
caveats are in [BENCHMARKS.md](BENCHMARKS.md).

Node path only, one pod, restored onto other GPUs (mean of two runs):

| TP | Configuration | CUDA tool | CUDA checkpoint | Image | Restore | Cold boot |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | cuInterpose, CUDA 13.1 `libcuda` | Dynamo helper | 2.0 s | 11.0 GB | 7.4 s | 46.3 s |
| 2 | cuInterpose, CUDA 13.1 `libcuda` | `snapshotd` | 1.5 s | 11.0 GB | 7.1 s | 46.3 s |
| 2 | no cuInterpose (`patches/vllm/0003`) | `snapshotd` | 1.6 s | 11.7 GB | 13.6 s | 45.3 s |
| 4 | cuInterpose, CUDA 13.1 `libcuda` | Dynamo helper | 4.1 s | 20.5 GB | 8.8 s | 49.3 s |
| 4 | cuInterpose, CUDA 13.1 `libcuda` | `snapshotd` | 1.9 s | 20.4 GB | 8.0 s | 48.8 s |
| 4 | no cuInterpose (`patches/vllm/0003`) | `snapshotd` | 2.1 s | 22.1 GB | 15.1 s | 46.3 s |

Restore is 3 to 6 times faster than a cold boot. [`snapshotd`](daemon) keeps the CUDA checkpoint near 2 s
where Dynamo's helper grows with the worker count. Dropping cuInterpose runs on the host's 580
driver and costs about 7 s of CUDA graph recapture on each restore.

CPU KV offload (`--kv-offloading-size`) at TP=2:

| Offload | Region | Image | Restore | Cold boot |
| --- | --- | --- | --- | --- |
| none | | 11.3 GB | 14.7 s | 43.3 s |
| 128 GiB | kept in the checkpoint (stock vLLM) | 139.4 GB | 82.8 s | 75.5 s |
| 128 GiB | released (`patches/vllm/0005`) | 11.4 GB | 43.3 s | 74.5 s |

End to end through KServe with Dynamo Snapshot's agent, pod creation to Ready
([details](https://github.com/wseaton/kserve-dynamo-snapshot#results)):

| Run | Restored replica Ready | Cold boot Ready |
| --- | --- | --- |
| TP=1, weights from a peer over InfiniBand | 37 to 41 s | 89 s |
| TP=2, restored on another node | 62 s | 177 s |
| TP=8, restored on another node | 97 s | 391 s |
| DeepSeek-V4.1-Flash, TP=8 + expert parallel, same node | 60 s | 148 s |

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
