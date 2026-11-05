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
