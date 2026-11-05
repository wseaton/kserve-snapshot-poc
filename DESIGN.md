# A Rust node daemon in place of Dynamo Snapshot

This replaces the [Dynamo Snapshot](https://github.com/ai-dynamo/snapshot) node agent in the
scale-out PoC with a node daemon written in Rust, built on
[snapshot-orchestrator](https://github.com/neuralmagic/snapshot-orchestrator)'s agent. The KServe
companion stays in Go and talks to a snapshot backend interface, so Dynamo and the new daemon can
run side by side and be compared on the same `LLMInferenceService`.

The reasons are memory safety and governance in the root-privileged component. The node agent runs
as root with `CAP_SYS_ADMIN`, enters workload namespaces, and drives CRIU and the CUDA checkpoint
API on behalf of arbitrary pods. Dynamo's agent is Go plus C helpers (`ns-bind-mount`,
`cuda-checkpoint-helper`) and is maintained by NVIDIA alone.

## What the experiments changed

### The CUDA runtime is not why TP>1 restore failed on vLLM 0.27.1

`frontend.c` resolves `cuda*` symbols it cannot find through `dlopen("libcudart.so.13")`. That would
miss a CUDA 12 runtime, but none of the images ship one:

| Image | vLLM | torch | libcudart | libcuda used at runtime |
| --- | --- | --- | --- | --- |
| `vllm/vllm-openai:v0.27.1-ubuntu2404@sha256:dafea057…` | 0.27.1 | 2.13.0+cu130 | `libcudart.so.13` (13.0.96) | host, 580.126.09 (CUDA 13.0) |
| `quay.io/wseaton/vllm-openai@sha256:20fd867e…` (compat) | 0.31.0 | 2.13.0+cu130 | `libcudart.so.13` (13.0.96) | `/opt/cuda-compat`, 590.48.01 (CUDA 13.1) |
| `quay.io/wseaton/vllm-checkpoint@sha256:e2659cc1…` (nightly) | 0.33.0rc1.dev806 | 2.14.0+cu130 | `libcudart.so.13` (13.0.96) | host, 580.126.09 (CUDA 13.0) |

`libtorch_cuda.so` and `libc10_cuda.so` import `cudaGetDriverEntryPointByVersion@libcudart.so.13`
in all three. The images differ in the user-mode driver, not the runtime: only the compat image puts
the CUDA 13.1 `libcuda` on `LD_LIBRARY_PATH`. Dynamo's cuInterpose GPU CI asserts
`cuDriverGetVersion() >= 13010` for the same reason.

The nightly image runs on the host's 580 `libcuda`, like 0.27.1.

### After vLLM's `/checkpoint/prepare`, TP=2 shares memory only through custom all-reduce

TP=2 Qwen3-8B on one H200 node, the nightly image, the host 580.126.09 driver, `NCCL_CUMEM_ENABLE=1`,
`NCCL_NVLS_ENABLE=0`, `NCCL_IB_DISABLE=1`, symmetric memory off, `--enable-sleep-mode
--checkpoint-release-weights --enable-nccl-comm-suspend`. cuInterpose was preloaded only to inspect:
its control socket returns every VMM allocation a process holds and whether another process maps
it. Per TP worker, after `/checkpoint/prepare` reported `prepared`:

| Configuration | Shared allocations | Shared bytes | Multicast objects |
| --- | ---: | ---: | ---: |
| upstream prepare, custom all-reduce on | 102 | 864 MB | 1 |
| prepare also suspends NCCL (`patches/vllm/0003`), custom all-reduce on | 6 | 260 MB | 1 |
| prepare also suspends NCCL, `--disable-custom-all-reduce` | 0 | 0 | 0 |

Upstream `checkpoint_prepare` releases the FlashInfer all-reduce workspaces and the all2all manager,
but leaves the NCCL communicator mapped even with `--enable-nccl-comm-suspend`, because that flag
only acts on `sleep`. `patches/vllm/0003` suspends it in `checkpoint_prepare` and resumes it in
`checkpoint_restore`. What remains with custom all-reduce on is its IPC buffers.

### Plain cuda-checkpoint restores TP=2 once nothing is shared

Same setup, with no cuInterpose: after prepare, lock, checkpoint, restore and unlock every CUDA
process (API server, engine core, two workers) through the driver API in that order, in place, then
`/checkpoint/restore` and compare a greedy completion with the one before the checkpoint.

| Run | libcuda | Graphs | All-reduce | NCCL at prepare | Result |
| --- | --- | --- | --- | --- | --- |
| eager, patched | 580 | off | NCCL | suspended | token-identical |
| eager, unpatched | 580 | off | NCCL | mapped | `cuCheckpointProcessRestore`: `CUDA_ERROR_NOT_SUPPORTED` on both workers |
| graphs, fused FlashInfer `mnnvl` | 580 | on | NCCL + FlashInfer | suspended | checkpoint and restore pass; FlashInfer's workspace rebuild fails in `cuMulticastAddDevice` with `CUDA_ERROR_INVALID_DEVICE` |
| graphs, fused FlashInfer `trtllm` | 580 | on | NCCL + FlashInfer | suspended | token-identical |
| graphs, fused FlashInfer `mnnvl` | 590 (13.1 compat) | on | NCCL + FlashInfer | suspended | token-identical |
| graphs, custom all-reduce on | 590 (13.1 compat) | on | custom + NCCL | suspended | `cuCheckpointProcessCheckpoint`: `CUDA_ERROR_OPERATING_SYSTEM` on both workers |

The driver-580 CUDA graph checkpoint hang recorded in snapshot-orchestrator's
`docs/cuda-checkpoint-tp2-graphs-rca.md` did not reproduce with custom all-reduce off.

The `mnnvl` rows are the likely cause of the 0.27.1 TP=2 failure. On the 580 user-mode driver, a
multicast or peer VMM call made after a cuda-checkpoint restore fails with
`CUDA_ERROR_INVALID_DEVICE`; on the 13.1 compat driver, on the same node and kernel module, the
same call succeeds. cuInterpose's `RestoreUnicast` re-imports and remaps peer memory with the same
class of calls (`cuMemImportFromShareableHandle`, `cuMemMap`, `cuMemSetAccess`).

The failure also reproduces with cuInterpose itself. With custom all-reduce on, `trtllm` FlashInfer,
NCCL suspended and cuInterpose preloaded, the run was: `/checkpoint/prepare`, then
`cuinterpose-coordinator --prepare`, then lock, checkpoint, restore and unlock in place, then
`cuinterpose-coordinator --restore`.

| libcuda | Coordinator restore | Result |
| --- | --- | --- |
| 580 | both workers abort: `cuinterpose: unrecoverable state change: CUDA error CUDA_ERROR_INVALID_DEVICE`, coordinator fails in `RestoreMulticastDevices` | server down |
| 590 (13.1 compat) | succeeds | token-identical |

So the 0.27.1 failure is the 580 user-mode driver, not the CUDA runtime or cuInterpose's runtime
interception. That run failed one phase earlier, in `RestoreUnicast`, and failed with custom
all-reduce off too: nothing suspended NCCL at checkpoint then, so NCCL's cuMem peer buffers were the
shared unicast memory cuInterpose had to re-import.

The scripts and pod spec for these runs are in [`experiments/tp2-shared-memory`](experiments/tp2-shared-memory).

### Through CRIU, and onto other GPUs

The in-place runs leave the processes alive. These dump the whole vLLM tree with CRIU (Dynamo's
bundled CRIU 4.2.1, its CUDA plugin disabled), restore it into a fresh network namespace in the same
privileged pod, then run the CUDA restore, either onto the same two GPUs or remapped onto two other
GPUs through `CUcheckpointGpuPair`, then `/checkpoint/restore`. TP=2, the nightly image, patch
0003, custom all-reduce off, no cuInterpose. The dump is about 6 s and 9 to 13 GB of host memory,
and the CRIU restore about 1.5 s.

| GPUs after restore | NCCL transport | Graphs | FlashInfer all-reduce | Result |
| --- | --- | --- | --- | --- |
| same | cuMem P2P, suspended | off | `trtllm` | token-identical |
| other | cuMem P2P, suspended | off | `trtllm` | workspace rebuild fails in `cuMemCreate`: `CUDA_ERROR_NOT_INITIALIZED` |
| other | cuMem P2P, suspended | off | off | restores and reloads weights; first forward `CUDA_ERROR_ILLEGAL_ADDRESS` |
| other, 13.1 driver | cuMem P2P, suspended | off | off | same illegal address |
| other | legacy IPC P2P (`NCCL_CUMEM_ENABLE=0`) | off | off | `cuCheckpointProcessRestore`: `CUDA_ERROR_INVALID_VALUE` on both workers; suspend does not release legacy IPC buffers |
| other | host shared memory (`NCCL_P2P_DISABLE=1`, cuMem off) | off | off | token-identical |
| other | host shared memory | on, fusion off | off | token-identical |
| other | host shared memory | on, allreduce-RMS fusion on | fused workspace | workspace rebuild fails: `CUDA_ERROR_NOT_INITIALIZED` |

Three things broke on the move to other GPUs, and none of them is cuInterpose's business:

- `ncclCommResume` re-maps cuMem peer buffers but leaves the communicator pointing at memory that
  faults once a kernel uses it, on both drivers. snapshot-orchestrator saw the same class of failure
  with fresh cuMem communicators after migration, and avoided it by destroying NCCL at checkpoint and
  creating a new communicator over legacy IPC after restore.
- FlashInfer's all-reduce workspaces are rebuilt through `cuda-python`, which returns
  `CUDA_ERROR_NOT_INITIALIZED` after the device remap. This is the standalone FlashInfer
  all-reduce and the workspace the allreduce-RMS compile fusion creates.
- Legacy IPC buffers survive `ncclCommSuspend`.

The PIDs of the dumped tree must be free at restore, and the source's loopback sockets must be gone,
which a new pod provides. The vLLM API server opens an `io_uring` through uvloop's libuv even with
`UV_USE_IO_URING=0`, and CRIU cannot dump it, so the workload needs `io_uring_setup` blocked, as
Dynamo's `block-iouring` seccomp profile does. The experiment installs the same filter before
exec'ing vLLM.

### What this means for the design

- With vLLM releasing its communicators, TP>1 on one node needs no cuInterpose when the restore
  lands on the source's GPUs: prepare, then the ordered lock-all, checkpoint-all, restore-all,
  unlock-all sequence. The pod contract requires `--disable-custom-all-reduce`, and either the 13.1+
  user-mode driver or `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` when the allreduce-RMS fusion is on.
- A scale-out restore almost never lands on the source's GPUs. Without cuInterpose that works today
  only with NCCL over host shared memory and FlashInfer all-reduce off, which costs all-reduce
  bandwidth (not measured yet). The fix belongs in vLLM: tear the NCCL communicator down at prepare
  and create a new one at restore, as snapshot-orchestrator's worker extension did, and rebuild the
  FlashInfer workspaces through the device remap. Until then, cuInterpose with the 13.1 driver is the
  way to keep NVLink P2P across a GPU move; the earlier Dynamo TP=2 and TP=8 cross-node runs did that.
- cuInterpose stays as a fallback for workloads that still share memory at checkpoint time: custom
  all-reduce, symmetric memory, NVLS, or engines without a checkpoint lifecycle. The daemon decides
  at checkpoint time by inspecting, not by configuration.
- Every TP>1 cuInterpose restore needs the CUDA 13.1+ user-mode driver until host drivers are
  upgraded. The daemon records `cuDriverGetVersion` of the workload's `libcuda` in the manifest and
  refuses a restore below 13010 when the checkpoint holds shared memory.

## Components

```text
                      Kubernetes API
             ┌──────────────┴───────────────┐
             │                              │
   kserve-dynamo-snapshot companion   snapshot daemon (Rust, privileged DaemonSet)
   Go, unprivileged                   ┌──────────────────────────────────────────┐
   ├ webhook: hold / shape pods       │ control loop (tokio, kube-rs)            │
   ├ capture + restore controllers    │  ├ CRI: pod/container → host PID, rootfs │
   └ SnapshotBackend ──┬─ dynamo      │  ├ vLLM HTTP: /checkpoint/*              │
                       └─ daemon ────►│  ├ readiness condition on the pod        │
                                      │  └ jobs, one per pod ──► worker process  │
                                      └──────────────────────────┬───────────────┘
                                                                 │ fork, setns in pre_exec,
                                                                 │ exec self (single-threaded)
                                      ┌──────────────────────────▼───────────────┐
                                      │ in-namespace worker                      │
                                      │  ├ CRIU swrk over RPC (prost)            │
                                      │  ├ cuInterpose coordinator (library)     │
                                      │  └ mount fixups, rootfs diff apply       │
                                      └──────────────────────────────────────────┘
                 host side: CUDA checkpoint driver API via cudarc, host libcuda
```

### Companion: a backend interface

The companion keeps deciding which pods restore and when to capture. What it calls changes:

```go
type Artifact struct {
    Name string            // PodSnapshot name, or the daemon's checkpoint ID
    Node string            // where the bytes are, for local storage
    Key  snapshotkey.Key   // revision compatibility key
}

type SnapshotBackend interface {
    // Checkpoint captures pod, which must be serving, and leaves it dead.
    Checkpoint(ctx context.Context, pod *corev1.Pod) (Artifact, error)
    // Restore restores artifact into pod, a placeholder already admitted
    // by the webhook, and returns once the pod is serving or failed.
    Restore(ctx context.Context, artifact Artifact, pod *corev1.Pod) error
    // ShapeRestorePod mutates a pod at admission so the backend can restore into it.
    ShapeRestorePod(pod *corev1.Pod, artifact Artifact) error
}
```

`dynamo` wraps today's `SnapshotJob`/`PodSnapshot` objects and the `nvidia.com/Restored` condition.
`daemon` writes a `SnapshotCheckpoint` (snapshot-orchestrator's existing CRD) and a request
annotation on the pod, and watches the pod's readiness condition. A per-revision annotation picks
the backend, so one cluster can A/B both on the same `LLMInferenceService`.

### The pod contract shrinks

The daemon drives vLLM's routes itself, so the snapshot entrypoint wrapper and its sentinel files
(`ready-for-snapshot`, `restore-complete`, `vllm-restore-*`) go away.

| | Dynamo today | Daemon |
| --- | --- | --- |
| Source command | entrypoint wrapper around `vllm serve` | `vllm serve`, prefixed with `cuinterpose-launch` only when cuInterpose is wanted |
| Capture trigger | wrapper writes `ready-for-snapshot`; readiness probe reads it | daemon polls `/health`, warms up, `POST /checkpoint/prepare`, polls `/checkpoint/status` |
| Restore placeholder | `snapshot-restore-proxy` as PID 1 | snapshot-orchestrator's `snapshot-shim` as PID 1 |
| After CRIU | wrapper waits for `restore-complete`, then calls vLLM | daemon calls `POST /checkpoint/restore` inside the pod's network namespace |
| Ready | wrapper's readiness file | daemon sets the `snapshot.llm-d.ai/Serving` readiness gate condition |
| Progress and errors | files in the control dir, read by the companion | pod condition message and events |

vLLM's `/checkpoint/*` routes are not behind `VLLM_SERVER_DEV_MODE`, so the dev endpoints stay off.
The daemon reaches them on the pod IP from the host network, or on loopback from the worker
process in the pod's network namespace.

## The privileged path

### Execution model

The control loop is a multi-threaded tokio process and cannot `setns(CLONE_NEWNS)`. Work that needs
the pod's mount or PID namespace runs in a short-lived worker: the daemon forks, calls `setns` for
each namespace in `pre_exec` (the child is single-threaded there), and execs its own static musl
binary with a subcommand. That replaces snapshot-orchestrator's `nsenter` dependency and Dynamo's
`nsenter`, `ns-bind-mount` and `nsrestore` binaries. Mount injection uses `open_tree` and
`move_mount` from the worker, through `rustix`.

Two things never need the pod's namespaces:

- **CUDA checkpoint calls.** `cuCheckpointProcess{Lock,Checkpoint,Restore,Unlock,GetState}` take a
  host PID and run in the daemon through `cudarc`'s driver bindings and the host `libcuda`. This
  replaces both the `cuda-checkpoint` CLI that snapshot-orchestrator copies into each pod and
  Dynamo's `cuda-checkpoint-helper`. Each phase runs for all processes in parallel, and phases run
  in order (lock all, checkpoint all; restore all with `CUcheckpointGpuPair` remaps, unlock all),
  which is the IPC-safe order and keeps the parallel speedup snapshot-orchestrator measured.
- **cuInterpose control sockets.** They are path sockets under the workload's `/tmp`, and nothing in
  the protocol checks the caller's namespace or credentials, so the daemon can reach them through
  `/proc/<host pid>/root/tmp/cuinterpose-<ns pid>.sock`, and write `cuinterpose.state` straight
  into the artifact. With the coordinator as a library (`patches/dynamo-snapshot/0002`), the daemon
  runs the phases in-process; no coordinator binary needs to be executed inside the container.
  Until that lands, the worker executes NVIDIA's musl coordinator from the bundle mount, as Dynamo
  does.

### Checkpoint

1. Resolve the pod's container through CRI: containerd (`k8s.io` namespace, task PID and OCI spec)
   or CRI-O (`ContainerStatus` verbose info). Rootfs is `/proc/<pid>/root`, cgroup v2 from the
   `0::` line, upper dir from the mountinfo overlay options. Process tree by BFS over
   `/proc/<pid>/task/*/children`; vLLM forks workers from non-main threads.
2. `POST /checkpoint/prepare`, poll `/checkpoint/status` to `prepared`.
3. CUDA processes: every PID in the tree for which `cuCheckpointProcessGetRestoreThreadId` succeeds.
4. If any CUDA process preloaded cuInterpose, inspect it. If it holds shared memory: record library
   hashes, check that each participant still maps the bundle files (device and inode from
   `/proc/<pid>/maps`), write the manifest, run `prepare`. From here the source cannot resume.
5. CUDA lock all, checkpoint all.
6. CRIU dump over swrk RPC: external net namespace, `ExtMnt` from the mount policy, `TcpClose`, file
   locks, external unix sockets, link remap, `ManageCgroups=soft`, `FreezeCgroup`, and `ExtMasters`
   and `mntnsCompatMode` recorded in the manifest because CRIU needs them at restore.
7. Rootfs diff: tar of the upper dir minus exclusions and bind destinations, plus deleted entries.
   Dynamo looks for `.wh.*` names, but overlayfs whiteouts are 0:0 character devices; record those.
8. Publish the artifact by rename (local or RWX storage), then kill the source.

### Restore

1. Admission: the webhook (Go) sets the placeholder command (`snapshot-shim`), the hostPath bundle
   mount, the readiness gate, and the Localhost seccomp profile the capture used.
2. Read and validate the manifest: compatibility key, GPU count, cuInterpose library hashes and the
   minimum user-mode driver when the checkpoint holds shared memory.
3. In the worker: apply the rootfs diff, unmount the placeholder's `/dev/shm` so CRIU recreates it,
   bind target GPU device nodes over the source paths, add the source pod IP to the pod's network
   namespace so restored listeners can bind it, then CRIU restore with `NetworkLock=SKIP`, the
   placeholder's net namespace as `extNetNs`, inherited stdio, and `RstSibling`.
4. CUDA restore all with the GPU pair map, unlock all.
5. cuInterpose `restore` if the manifest has participants, with the checkpoint's namespace PIDs.
6. `POST /checkpoint/restore`; ModelExpress refills the weights. Set the readiness condition.

### What gets ported from Dynamo's Go agent

| Area | Dynamo source (bc9d216, `agent/`) | Lines | Destination |
| --- | --- | ---: | --- |
| CRI resolution | `internal/runtime/oci_containerd.go`, `oci_crio.go`, `process.go`, `namespaces.go` | ~800 | daemon, `cri` module over tonic CRI client; containerd task API only if CRI lacks the PID |
| Mount policy | `internal/runtime/mounts.go` `ClassifyMounts`, `BuildMountPolicy` | ~140 | worker: skip non-OCI `/proc`, `/sys`, `/run`; dump `/dev/shm`; externalize the rest; masked paths to `/dev/null` |
| CoreWeave device plugin | `patches/dynamo-snapshot/0001` | ~40 | volume-mounts `NVIDIA_VISIBLE_DEVICES` (a path) means "use PodResources"; skip `/run/nvidia-container-devices/*` markers |
| GPU discovery and remap | `internal/cuda/cuda.go` `DiscoverGPUs`, `BuildDeviceMap`; `criu/restore.go` `GPUMountAliases` | ~350 | daemon: UUIDs from kubelet PodResources, `/proc/driver/nvidia/gpus/*/information` for minors; on R580 the pair map must cover every GPU visible to the process |
| Device nodes after CRIU | `executor/restore.go` `RestoreNativePaths` | ~90 | worker: `mknodat` and `move_mount` into the restored mount namespace |
| CRIU options | `internal/criu/dump.go`, `restore.go`, `util.go` | ~870 | worker, through swrk RPC; `mntnsCompatMode` because CoreWeave's `/dev` is shared |
| Network | `internal/criu/inet_remap.go`, `restore_images.go` | ~560 | not ported: Dynamo rewrites `files.img` and needs a C CRIU plugin; vLLM's only connections after prepare are loopback Gloo/NCCL, and snapshot-orchestrator restores the source IP as a second address instead |
| Rootfs diff | `internal/runtime/overlay.go` | ~340 | worker, with the `tar` crate; whiteout handling fixed |
| cuInterpose | `internal/cuda/cuinterpose.go`, `cuinterpose_identity.go`, `internal/nsmount/injector.go` | ~490 | daemon `cuinterpose` module (prototype below); bundle via hostPath instead of `open_tree` injection |
| Job file | `internal/cuda/job.go` | ~160 | not ported: only for `cuda-checkpoint --launch-job` on driver 610+ |
| Manifest | `pkg/types/manifest.go` | ~350 | Rust struct, YAML |

### Manifest

```yaml
artifact: {id, revisionKey, containerName, createdAt}
source: {node, podName, podNamespace, podIP, image, imageID, cpuLimit, memoryLimit}
host: {kernelVersion, arch, driverVersion, cudaDriverApi}   # cudaDriverApi: workload libcuda, e.g. 13010
criu: {settings recorded at dump: extMasters, mntnsCompatMode, tcpClose, manageCgroupsMode, ..., extMnt, skipMnt}
cuda: {pids: [ns pids], gpus: [{uuid, product, devicePath}], nvidiaVisibleDevices}
cuinterpose: {libraries: {libcuinterpose.so: sha256, libcuinterpose_core.so: sha256}, pids: [ns pids]}  # only when memory is shared
vllm: {version, tensorParallelSize, checkpointReleaseWeights: true}
```

### Supply

| Component | Language | Where it comes from |
| --- | --- | --- |
| daemon, worker, shim | Rust | snapshot-orchestrator |
| cuInterpose core, coordinator, launcher, protocol | Rust | `ai-dynamo/snapshot` at a pinned commit, as a git dependency and a musl build |
| cuInterpose frontend | C (~480 lines) | same; stays C because the `dlsym` interposer needs a guaranteed tail call (`musttail`) |
| CRIU | C | upstream release; driven over swrk RPC, the `rust-criu` fork moved to prost |
| CUDA checkpoint | driver API | host `libcuda` through `cudarc` |

## Prototype

Branch `cuinterpose-node-path` of snapshot-orchestrator, on `main` (`f392b75`), local only:

- `crates/agent/src/cuinterpose.rs`: library hashes and verification, `/proc/<pid>/maps` check that
  the running processes still map the hashed files, manifest, namespace PID resolution, an
  inspection client over `cuinterpose-protocol`, the shared-memory summary that decides whether the
  coordinator runs at all, and the coordinator phases (binary for now).
- `lifecycle.rs`: `CuinterposePrepare` runs before the CUDA suspend, `CuinterposeRestore` after the
  CUDA resume; a prepared source refuses a nondestructive dump and skips the roll-back to serving;
  restore verifies library hashes before CRIU.
- `snapshot-shim`: `--cuinterpose-bundle-dir` starts vLLM under `cuinterpose-launch`.
- `snapshot-agent`: `--cuinterpose-bundle-dir` and `--cuinterpose-socket-dir`; TP>1 is accepted
  when the bundle is configured.

Not done: driving the vLLM `/checkpoint/*` routes instead of the old dev endpoints, the cudarc CUDA
path, the in-process coordinator, CRI resolution and the Dynamo mount handling. The cuInterpose
paths were exercised on waldorf with NVIDIA's binaries and the Python equivalents of the
inspection client; the Rust module has unit tests only.

## Upstream changes

| Change | Where | State |
| --- | --- | --- |
| Suspend NCCL in `checkpoint_prepare`, resume in `checkpoint_restore` | vLLM, `patches/vllm/0003` | tested at TP=2, in place and through CRIU onto the same GPUs |
| Release custom all-reduce IPC buffers in `checkpoint_prepare` and re-register them in `checkpoint_restore` | vLLM | not started; removes `--disable-custom-all-reduce` |
| Destroy the NCCL communicator at prepare and create a new one at restore, instead of suspending it | vLLM | not started; needed for P2P across a GPU move |
| Rebuild FlashInfer all-reduce workspaces after a device remap | FlashInfer, vLLM | not started; `cuda-python` returns `CUDA_ERROR_NOT_INITIALIZED` |
| Expose the cuInterpose coordinator as a library | ai-dynamo/snapshot, `patches/dynamo-snapshot/0002` | builds, clippy clean, existing tests pass |
| Accept CoreWeave's volume-mounts device strategy | ai-dynamo/snapshot, `patches/dynamo-snapshot/0001` | in use |

## Open questions

- The CRIU runs restored into a fresh network namespace in the same pod, not into a new pod; the
  mount, cgroup and device-node handling of a real cross-pod restore is the daemon port above.
- NCCL over host shared memory versus NVLink P2P at TP=2 and TP=8: the decode latency cost decides
  whether the shared-memory path is acceptable until vLLM recreates the communicator.
- Multi-node TP needs IB verbs released at prepare; NCCL's IB plugin leaves an `uverbs` mapping
  that CRIU cannot dump. Out of scope here, as in snapshot-orchestrator.
- Whether NVIDIA would take the coordinator split, and whether cuInterpose's control socket should
  check `SO_PEERCRED` once callers outside the container are expected.
