# The patches

Every change the proof of concept makes to an upstream project, as patch files against a pinned
commit. [`upstream.tsv`](../upstream.tsv) has each project's repository, base commit and fork
branch; `./checkout.sh vllm` clones a project at its base commit into `src/` and applies its
patches.

| Directory | Base |
| --- | --- |
| [`vllm/`](vllm) | `a2bcb565`, a nightly that includes the checkpoint lifecycle from [vllm-project/vllm#62570](https://github.com/vllm-project/vllm/pull/62570) |
| [`modelexpress/`](modelexpress) | `main` at `8b2e8787`, which includes in-place reload from [ai-dynamo/modelexpress#854](https://github.com/ai-dynamo/modelexpress/pull/854) |
| [`dynamo-snapshot/`](dynamo-snapshot) | `bc9d216` |

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
  in-process. Proposed for upstream; see [docs/design.md](../docs/design.md).
