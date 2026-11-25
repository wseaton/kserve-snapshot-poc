+++
sort_by = "weight"

[extra]
mermaid = true
+++

`snapshotd` is a node daemon. The first replica of a revision cold boots and is checkpointed;
later replicas are restored from that checkpoint, on the same node or on one that pulls it from a
peer. It works with a plain `Deployment` or a KServe `LLMInferenceService`, and needs no
controller, webhook or custom resource.

![Seconds until a DeepSeek-V4.1-Flash replica serves on 8 GPUs](img/headline.svg)

DeepSeek-V4.1-Flash, TP=8 with expert parallel on 8 H200s, captured on one node and restored on
another. One run of each. NVIDIA's figure is on different hardware ([comparison](@/comparison.md)).

## How it works

{% mermaid() %}
flowchart LR
  subgraph source[Source node]
    direction TB
    serve[vLLM, serving]
    prepare["/checkpoint/prepare<br/>free KV cache and weights"]
    cuda["snapshotd: CUDA checkpoint<br/>every process at once"]
    dump["criu dump --stream"]
    send["snapshotd transfer stream"]
    serve --> prepare --> cuda --> dump --> send
  end
  subgraph target[Target node]
    direction TB
    receive["snapshotd transfer receive<br/>image held in RAM"]
    criu[criu restore]
    gpu["snapshotd: CUDA restore<br/>onto this pod's GPUs"]
    restore["/checkpoint/restore<br/>reload weights, rebuild KV cache"]
    ready[vLLM, serving]
    receive --> criu --> gpu --> restore --> ready
  end
  peer[A replica that is serving]
  source == "image over InfiniBand, during the dump" ==> target
  peer == "weights, GPU to GPU" ==> target
{% end %}

1. **Prepare.** vLLM drains and frees its KV cache and weights. About 5 GB stays on each GPU.
2. **Capture.** `snapshotd` checkpoints every CUDA process in parallel. CRIU dumps into a pipe
   that `snapshotd` sends over RDMA as it fills. The source writes nothing.
3. **Restore.** CRIU restores from the image in the target's RAM, the driver puts GPU state back,
   and vLLM pulls its weights from a serving replica, or from the model volume if none is serving.

## Results

| Workload | Cold boot | Restore |
| --- | --- | --- |
| DeepSeek-V4.1-Flash, TP=8 + EP, two nodes, weights from a serving replica | 104.7 s | 20.5 s |
| The same, weights from the model volume | 166.1 s | 42.9 to 101.5 s |
| gpt-oss-120b, TP=4, two nodes, 106 GB checkpoint | 72.5 s | 17.2 to 18.6 s |
| Qwen3-8B, TP=4, one node, onto other GPUs | 48.8 s | 8.0 s |
| Qwen3-8B, TP=1, `snapshotd node` under KServe, pod creation to Ready | 60 s | 8 s same node, 30 s another node |

Capture to a complete image on the second node: 25.5 s for DeepSeek (74 GB), 33.0 s for
gpt-oss-120b (106 GB). Every number, with setup: [benchmarks](@/benchmarks.md).

## Status

The patches are rebased onto vLLM main (nightly `f2f9edca`). On that stack, restores onto other
GPUs give identical output for expert parallel at DP=2 and TP=2, CPU KV offload with and without
async_init, a draft model for speculative decoding, encoder CUDA graphs, and gpt-oss-120b with its
weights released. Prefill and decode under an LWS DisaggregatedSet restore per role, across nodes
and across a revision change, with every request answered right. The table above was measured on the earlier base.
[Validated configurations](@/validated.md) lists every run and what main is not validated on yet.
