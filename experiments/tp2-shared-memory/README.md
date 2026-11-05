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
