# TP=2 shared CUDA memory after `/checkpoint/prepare`

The runs behind DESIGN.md's tables, on one 8x H200 waldorf node with driver 580.126.09.

- `pod.yaml`: a 2-GPU pod with the nightly vLLM image, Dynamo's agent image mounted for the
  cuInterpose libraries and coordinator, and the compat image mounted for its CUDA 13.1 `libcuda`.
  Create a configmap `t2-exp` from the scripts here and `vllm/distributed/device_communicators/cuda_communicator.py`
  from `./checkout.sh vllm`, and put Qwen3-8B at `/models/Qwen3-8B`.
- `exp.sh <tag> <inspect|plain|cuinterpose> <0|1> [vllm args]`: start vLLM at TP=2, take a greedy
  completion, `/checkpoint/prepare`, then either inspect shared memory (`inspect`), run an in-place
  CUDA checkpoint cycle (`plain`), or run it between `cuinterpose-coordinator --prepare` and
  `--restore` (`cuinterpose`); then `/checkpoint/restore` and compare the completion. The third
  argument copies `patches/vllm/0003`'s `cuda_communicator.py` over the image's when it is 1.
- `inspect.py`: asks each cuInterpose-preloaded process for its records over the control socket.
- `cudackpt.py`: lock all, checkpoint all, restore all, unlock all through the driver API.
- `series3.sh`, `series4.sh`: the graph, FlashInfer backend, 13.1 driver and cuInterpose runs.

The CRIU runs:

- `pod-criu.yaml`: a privileged 4-GPU pod with a reaping PID 1 (CRIU needs the dumped PIDs free);
  configmap `t5-exp` from `criuexp.sh`, `cudaphase.py`, `noiouring.py` and the patched
  `cuda_communicator.py`.
- `criuexp.sh <tag> <remap 0|1> [vllm args]`: vLLM TP=2 on the first two GPUs under `noiouring.py`,
  prepare, CUDA lock and checkpoint, `criu dump`, `criu restore` in a new network namespace, CUDA
  restore (onto the third and fourth GPUs with remap 1) and unlock, `/checkpoint/restore`, compare.
  NCCL, FlashInfer and graph settings come from the environment and the vLLM arguments.
- `cudaphase.py`: one CUDA checkpoint phase over a list of PIDs, with `--map OLD=NEW,...` for restore.
- `noiouring.py`: installs a seccomp filter that fails `io_uring_setup` with ENOSYS, then execs.
