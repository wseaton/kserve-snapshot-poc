# Experiments

The runs behind [docs/design.md](../docs/design.md): TP=2 Qwen3-8B on one 8x H200 waldorf node, driver
580.126.09, the nightly image `quay.io/wseaton/vllm-checkpoint@sha256:e2659cc1…`. Both pods mount
Dynamo's agent image for the cuInterpose libraries, coordinator and CRIU, and the compat image for
its CUDA 13.1 `libcuda`. Put Qwen3-8B at `/models/Qwen3-8B` before running anything.

## `in-place/`: what stays shared after prepare, and plain cuda-checkpoint

The processes stay alive: lock, checkpoint, restore and unlock them in place.

- `pod.yaml`: 2 GPUs; configmap `in-place-tp2` from the scripts here plus a
  `cuda_communicator.py` to copy over the image's.
- `exp.sh <tag> <inspect|plain|cuinterpose> <0|1> [vllm args]`: start vLLM, take a greedy
  completion, `/checkpoint/prepare`, then inspect shared memory (`inspect`), run the CUDA cycle
  (`plain`), or run it between `cuinterpose-coordinator --prepare` and `--restore`
  (`cuinterpose`); `/checkpoint/restore` and compare. With the third argument 1, the
  `cuda_communicator.py` from the configmap replaces the image's. These runs used an earlier
  version of `patches/vllm/0003` that suspended NCCL instead of recreating it.
- `inspect.py`: asks each cuInterpose-preloaded process for its records.
- `cudackpt.py`: the CUDA cycle through the driver API, in place.
- `series3.sh`, `series4.sh`: the graph, FlashInfer backend, 13.1 driver and cuInterpose runs.

## `criu/`: CRIU dump and restore, onto the same or other GPUs

- `pod.yaml`: privileged, 4 GPUs, a reaping PID 1 because the restore needs the dumped PIDs free.
  Configmap `criu-tp2` from `criuexp.sh`, `noiouring.py` and `overlay.tgz`, a tarball of the
  vLLM files `patches/vllm/0003` changes, taken from `./checkout.sh vllm`. Copy a `snapshotd`
  built from [`daemon/`](../daemon) for x86_64 glibc to `/work/snapshotd`.
- `criuexp.sh <tag> <remap 0|1> [vllm args]`: vLLM on the first two GPUs under `noiouring.py`,
  prepare, `snapshotd cuda checkpoint`, `criu dump`, `criu restore` into a new network namespace,
  `snapshotd cuda restore` (onto the third and fourth GPUs with remap 1), `/checkpoint/restore`,
  compare. NCCL, FlashInfer and graph settings come from the environment and the vLLM arguments,
  for example:

  ```bash
  VLLM_ALLREDUCE_USE_FLASHINFER=0 NCCL_CUMEM_ENABLE=0 /exp/criuexp.sh remap-graphs 1 \
    --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
  ```
- `noiouring.py`: installs a seccomp filter that fails `io_uring_setup` with ENOSYS, then execs.

## `bench/`: timed cold boot, checkpoint and restore of one server

- The pod and its volumes are kustomize in [`deploy/snapshotd`](../deploy/snapshotd);
  [deploy/README.md](../deploy/README.md) walks through them. The pod is privileged and takes a
  whole node's 8 GPUs. An init container copies `snapshotd` to `/models/bin` and these scripts to
  `/exp` out of the `snapshotd` image. An `overlay.tgz` of the vLLM files the patches change is
  copied in by hand: `kubectl cp overlay.tgz restore-bench:/exp/overlay.tgz`.
  A privileged pod sees every GPU on the node, so the script takes its GPUs from the kubelet's
  device checkpoint, mounted from the host.
- `restore-bench.sh <tag> [vllm args]`: cold boot, a greedy completion, prepare,
  `snapshotd cuda checkpoint`, `criu dump`, `criu restore` into a new network namespace,
  `snapshotd cuda restore`, `/checkpoint/restore`, the completion again. Every phase is timed and
  one `RESULT` line of JSON comes out. `MODE=cuinterpose` preloads cuInterpose and runs its prepare
  and restore phases; `REMAP=rotate` restores each rank onto the next rank's GPU.
- `offload-check.sh`: as `POST_CHECK`, counts the bytes CPU KV offload loads back after the
  restore.

The CPU KV offload rows in [design.md](../docs/design.md):

```bash
export MODEL=/models/Qwen3-8B TP=2 NCCL_CUMEM_ENABLE=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  POST_CHECK=/exp/offload-check.sh REMAP=rotate
A=(--disable-custom-all-reduce --max-model-len 4096 --kv-cache-memory-bytes 1G
   --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}')
/exp/restore-bench.sh released128 "${A[@]}" --kv-offloading-size 128
```

The kept rows are the same command with `GHOST_LIMIT=160G` on an overlay built without
`patches/vllm/0004` and `0005`.

The comparison rows in [benchmarks.md](../docs/benchmarks.md), with `TP=2` and `TP=4`:

```bash
export MODEL=/models/Qwen3-8B TP=4 REMAP=rotate
B=(--max-model-len 4096 --gpu-memory-utilization 0.6)
MODE=cuinterpose COMPAT=/usr/local/cuda-13.1/compat CUDA_TOOL=helper \
  /exp/restore-bench.sh dynamo "${B[@]}"
OVERLAY=/exp/overlay.tgz NCCL_CUMEM_ENABLE=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  /exp/restore-bench.sh ours "${B[@]}" --disable-custom-all-reduce \
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
```

Run the cuInterpose rows before any row that sets `OVERLAY`, which patches vLLM in the pod.

## Two nodes

- `bench/lifecycle.sh <tag> [vllm args]`: capture on one bench pod, copy to another over RDMA
  through the `xfer` container, restore there, and merge everything into one Perfetto trace.
  `OVERLAP=1` streams the dump as it is taken; `IMG_DIR=/mem/<tag>` keeps the image in RAM on the
  target. `restore-bench.sh` does the two halves with `PHASE=capture` and `PHASE=restore`.
- `MX=1 NET=pod PEER=1` on `lifecycle.sh` loads weights through ModelExpress, restores into the
  pod's network, and cold boots a second server on the source as the peer to pull from. The
  workload container needs `rdma/ib` and the pod-info volume, which the `modelexpress-weights`
  component in `deploy/snapshotd` adds.
- `bench/trace-merge.py`: merges the transfer's sender and receiver traces with the harness
  phases.
- `bench/trace-render.py <output dir> [--png FILE]`: draws a run as a standalone page that reads on
  a phone: capture, transfer, restore and the cold boot beside them. `lifecycle.sh` writes it as
  `lifecycle.html`.
- Notes and results: [cross-node.md](../docs/cross-node.md), [memory-capture.md](../docs/memory-capture.md).

## `timeslice/`: llm-d-rl-time-slicing's orchestrator, with snapshotd as its node agent

Results and how the calls map are in
[docs/orchestration.md](../docs/orchestration.md#time-slicing).

- `run.sh up|jobs|verl|down`: installs the project's TimeSlice Orchestrator, points it at the
  daemon on one node, and runs jobs that take turns on one of that node's GPUs. The daemon has to
  serve the agent API (`SNAPSHOTD_TIMESLICE_LISTEN`).
- `orchestrator.yaml`: the project's orchestrator chart, rendered at commit `8757acb`.
- `job.py`, `job.yaml`: a stand-in job. Each burst adds to a tensor that fills most of the GPU
  and checks that every earlier burst's additions are still there.
- `run.sh verl` fetches the project's own example, two verl RL jobs whose trainers share the GPU,
  and changes only how the trainer pods get at it.

## `weights-file/`: weights kept in files beside the checkpoint

vLLM's sleep-mode allocator already owns the weights' device memory and can copy it to host
memory and back. These runs point that copy at files, so the weights are in neither the process
nor the CRIU image, and compare it with reloading them through the model loader. Qwen3-8B
(15.9 GiB of weights) at TP=1 on one H200, `restore-bench.sh` in one pod:

| Weights at restore | Weights back on the GPU | Restore, total | CRIU image |
| --- | --- | --- | --- |
| Released at prepare, reloaded by vLLM (`RELEASE_WEIGHTS=1`) | 2.80 s | 5.57 s | 4.6 GB |
| Kept on the GPU, so in the process image (`RELEASE_WEIGHTS=0`) | in the image | 10.3 s | 20.5 GB |
| In files, read back through a pinned buffer by 8 threads | 1.21 s | 3.83 s | 4.6 GB |

- `vllm-overlay.diff`: the change to vLLM, against the files in the `vllm-checkpoint` image. Pack
  the two patched files as `OVERLAY` and set `VLLM_CHECKPOINT_WEIGHTS_DIR`.
- `pin.py <file> <GiB>`: what a copy between a GPU and a shared file mapping costs, pageable and
  pinned. 16 GiB from a pinned tmpfs file reached the GPU in 0.31 s, after 0.94 s to pin it; a
  mapping of a file on disk cannot be pinned and took 2.8 s.
- `gpucr-micro.sh`, `hold.py`: the same question put to GPU-CR, the engine llm-d-rl-time-slicing
  carries: 16 GiB back on the GPU in 0.43 s from tmpfs. `MODE=gpucr` in `restore-bench.sh` runs
  GPU-CR around the CUDA checkpoint; its dump covers memory from `cudaMalloc` only, which is
  0.5 GiB of this server's 17.9.

