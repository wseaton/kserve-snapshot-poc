# Experiments

The runs behind [DESIGN.md](../DESIGN.md): TP=2 Qwen3-8B on one 8x H200 waldorf node, driver
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

- `pod.yaml`: privileged, with the node and GPU count filled in, a models volume and a compile
  cache volume. Configmap `restore-bench` from `restore-bench.sh`, `../criu/noiouring.py` and an
  `overlay.tgz` of the vLLM files the patches change. `snapshotd` goes to `/models/bin/snapshotd`.
  A privileged pod sees every GPU on the node, so the script takes its GPUs from the kubelet's
  device checkpoint, mounted from the host.
- `restore-bench.sh <tag> [vllm args]`: cold boot, a greedy completion, prepare,
  `snapshotd cuda checkpoint`, `criu dump`, `criu restore` into a new network namespace,
  `snapshotd cuda restore`, `/checkpoint/restore`, the completion again. Every phase is timed and
  one `RESULT` line of JSON comes out. `MODE=cuinterpose` preloads cuInterpose and runs its prepare
  and restore phases; `REMAP=rotate` restores each rank onto the next rank's GPU.
- `offload-check.sh`: as `POST_CHECK`, counts the bytes CPU KV offload loads back after the
  restore.

The CPU KV offload rows in DESIGN.md:

```bash
export MODEL=/models/Qwen3-8B TP=2 NCCL_CUMEM_ENABLE=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  POST_CHECK=/exp/offload-check.sh REMAP=rotate
A=(--disable-custom-all-reduce --max-model-len 4096 --kv-cache-memory-bytes 1G
   --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}')
/exp/restore-bench.sh defer128 "${A[@]}" --kv-offloading-size 128 --checkpoint-release-kv-offload
GHOST_LIMIT=160G /exp/restore-bench.sh keep128 "${A[@]}" --kv-offloading-size 128
```

The comparison rows in BENCHMARKS.md, with `TP=2` and `TP=4`:

```bash
export MODEL=/models/Qwen3-8B TP=4 REMAP=rotate
B=(--max-model-len 4096 --gpu-memory-utilization 0.6)
MODE=cuinterpose COMPAT=/compat/usr/local/cuda-13.1/compat CUDA_TOOL=helper \
  /exp/restore-bench.sh dynamo "${B[@]}"
OVERLAY=/exp/overlay.tgz NCCL_CUMEM_ENABLE=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  /exp/restore-bench.sh ours "${B[@]}" --disable-custom-all-reduce \
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
```

Run the cuInterpose rows before any row that sets `OVERLAY`, which patches vLLM in the pod.
