# The patches

Every change the proof of concept makes to an upstream project, as patch files against a pinned
commit. [`upstream.tsv`](../upstream.tsv) has each project's repository, base commit and fork
branch; `./checkout.sh vllm` clones a project at its base commit into `src/` and applies its
patches.

| Directory | Base |
| --- | --- |
| [`vllm/`](vllm) | `main` at `f2f9edca`, the commit of a published nightly image, which includes the checkpoint lifecycle from [vllm-project/vllm#62570](https://github.com/vllm-project/vllm/pull/62570) and the loader hooks from [vllm-project/vllm#57011](https://github.com/vllm-project/vllm/pull/57011) |
| [`modelexpress/`](modelexpress) | `main` at `8b2e8787`, which includes in-place reload from [ai-dynamo/modelexpress#854](https://github.com/ai-dynamo/modelexpress/pull/854) |
| [`dynamo-snapshot/`](dynamo-snapshot) | `bc9d216` |
| [`criu/`](criu) | `criu-dev` at `1a935857`, the commit the snapshot-agent image builds |

**vLLM**
- `0001`, `0002`: [vllm-project/vllm#64371](https://github.com/vllm-project/vllm/pull/64371)
  (open), both commits, with their tests moved to main's engine-core test helper. KV connectors get
  `checkpoint_prepare` and `checkpoint_restore`, the engine refuses to prepare with a connector
  that does not declare `supports_checkpoint`, and NIXL prefill and decode instances are
  checkpointed with `--checkpoint-prepare-on-startup` and restored independently, each restore
  taking a new KV engine id and the restored pod's own side-channel address.
- `0003`: `--checkpoint-release-weights`. `/checkpoint/prepare` also discards the weights, so the
  checkpoint does not contain them, and `/checkpoint/restore` wakes them and reloads them through
  the model loader before the KV cache. Releasing the weights suspends only the `weights`
  allocations, through a `tags` filter on the sleep backend and the allocators: parameters a reload
  rewrites are dropped, buffers and preserved parameters go to host memory.
- `0004`: `/checkpoint/prepare` destroys the NCCL communicator and `/checkpoint/restore` creates a
  new one, releasing CUDA graphs before and capturing them again after. At TP>1 with custom
  all-reduce off, no CUDA memory is shared between workers when the process is checkpointed, and
  cuda-checkpoint alone restores them, onto other GPUs too. Graphs are captured again into a fresh
  graph pool, since blocks captured into the old one (Engram's pinned host buffers, for one) can
  outlive the released graphs.
- `0005`, `0006`: the checkpoint lifecycle on the Rust frontend's gRPC `Control` service
  (`CheckpointPrepare`, `CheckpointRestore`, `GetCheckpointState`), sharing one state machine with
  the `/checkpoint` HTTP routes. Upstream serves it only over HTTP, on the inference port.

- `0007`: with the CPU KV offload connector (`--kv-offloading-size`), `/checkpoint/prepare` stops
  any background setup, unregisters the offload region and frees its pages, and
  `/checkpoint/restore` sets it up again, in the background under `async_init`. The checkpoint does
  not grow by the region size. Workers sharing the replicated layout's one region meet at a
  barrier before freeing it; per-rank regions do not. The tiering spec refuses to prepare.

**ModelExpress**
- `0001`: read `POD_NAME`, `POD_NAMESPACE` and `POD_UID` from a downward API volume named by
  `MX_POD_INFO_DIR`, and resolve the worker host from the network. A process restored into another
  pod keeps the environment of the pod it was checkpointed in, so without this it would publish
  itself to the ModelExpress server as the donor.

**Dynamo Snapshot**
- `0001`: accept CoreWeave's volume-mounts `NVIDIA_VISIBLE_DEVICES`, skip the per-GPU device-list
  marker mounts at dump, and log every CRIU error line on a failed restore.
- `0002`: expose cuInterpose's coordinator as a library, so a node agent can run its phases
  in-process. Proposed for upstream; see [Open](https://wseaton.github.io/snapshotd/open/#upstream-changes).
- `0003`: build the image's CRIU with libselinux and with the patches in an `agent/criu/`
  directory, which the image workflow fills from [`criu/`](criu).

**CRIU**
- `0001`: decide whether SELinux is in force from the process's own label instead of
  `is_selinux_enabled()`. That call needs selinuxfs and `/etc/selinux/config`, and the daemon runs
  CRIU in the pod's mount namespace, which has neither. Without it a restored server keeps the
  daemon's context, `spc_t`, instead of its pod's.
