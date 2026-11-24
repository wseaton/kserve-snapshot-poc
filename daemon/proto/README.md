# Vendored gRPC schemas

`build.rs` generates clients from these with `protox` and `tonic-prost-build`; no `protoc` is
needed. They are copied unchanged from:

| File | Source | Revision |
| --- | --- | --- |
| `vllm/control.proto` | `rust/proto/control.proto` in [wseaton/vllm](https://github.com/wseaton/vllm) | `b3c279b57a69c494a651ac400c1e8dbd8454e8c3` |
| `cri/api.proto` | `pkg/apis/runtime/v1/api.proto` in [kubernetes/cri-api](https://github.com/kubernetes/cri-api) | `v0.35.0` |
| `podresources/api.proto` | `pkg/apis/podresources/v1/api.proto` in [kubernetes/kubelet](https://github.com/kubernetes/kubelet) | `v0.35.0` |
| `timeslice/snapshot_agent.proto` | `pkg/snapshot-agent/api/v1alpha1/snapshot_agent.proto` in [llm-d-incubation/llm-d-rl-time-slicing](https://github.com/llm-d-incubation/llm-d-rl-time-slicing) | `8757acb1` |

`vllm/control.proto` is upstream's at `a2bcb565` plus the checkpoint RPCs of
[`patches/vllm/0006`](../../patches/vllm).
