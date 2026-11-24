# Deploying

Three paths, all kustomize:

| | Node agent | Checkpoint goes to | Weights after restore |
| --- | --- | --- | --- |
| [`snapshotd/node`](snapshotd/node) | `snapshotd node` DaemonSet | node disk, pulled by peers | model volume |
| [`snapshotd/`](snapshotd) bench pods | `snapshotd` from scripts | node disk, or the target's RAM over InfiniBand | ModelExpress peer, else model volume |
| [`dynamo-snapshot/`](dynamo-snapshot) | Dynamo Snapshot `bc9d216` | a `ReadWriteMany` volume | ModelExpress peer, else model volume |

The daemon path is documented on the [docs site](https://wseaton.github.io/snapshotd/running/).
This page covers the other two. Their manifests have been built and dry-run, but the measured runs
created the same objects from scripts, not from this tree.

## Layout

```
deploy/
  components/images/         digests of the images built here; CI opens a PR to change them
  modelexpress/
    crds/                    ModelExpress CRDs, v0.6.0 (cluster-scoped)
    server/                  ModelExpress server, one per namespace
  dynamo-snapshot/
    agent/                   Dynamo Snapshot operator and agent: its Helm chart, inflated and patched
    companion/               admission webhook and controller that capture on scale-up
    config-layer/            the LLMInferenceServiceConfig a service opts into
    qwen3-8b/namespace/      namespace, model volume and download, config layer, ModelExpress server
    qwen3-8b/service/        the LLMInferenceService
  snapshotd/
    node/                    the snapshotd DaemonSet with peers, from daemon/deploy
    node-openshift/          the same for OpenShift: CRI-O, the privileged SCC, the service CA
    workload/                an example Deployment for the daemon
    workload-pd/             the same as NIXL prefill and decode Deployments
    kserve/                  the daemon under KServe; its README documents the layer
    base/                    namespace and volumes
    pod/                     the bench pod
    components/
      rdma-image-transfer/   xfer container and RAM-backed image directory
      modelexpress-weights/  rdma/ib for the workload, pod-info volume, ModelExpress server
    single-node/             base and one pod
    two-node/                base, two pods on different nodes, both components
    models/                  download Jobs for Qwen3-8B and DeepSeek-V4.1-Flash
```

## Prerequisites

- `kubectl`, `kustomize` 5+, `helm` for the agent's chart.
- GPU nodes (measured on 8x H200, driver 580.126.09, containerd).
- A `ReadWriteMany` storage class (the manifests name `shared-vast`).
- For RDMA: an InfiniBand device plugin offering `rdma/ib`, and transparent huge pages.
- The companion image is private. Its namespace and each service namespace need a pull secret:

  ```bash
  kubectl -n <namespace> create secret docker-registry quay-wseaton-pull \
    --from-file="$HOME/.docker/config.json"
  ```

### Images

| Image | Built from |
| --- | --- |
| `ghcr.io/wseaton/snapshotd/vllm-checkpoint` | [`images/vllm-checkpoint`](../images/vllm-checkpoint): vLLM nightly `f2f9edca` + patches, ModelExpress fork, CUDA 13.1 compat `libcuda` at `/usr/local/cuda-13.1/compat` |
| `ghcr.io/wseaton/snapshotd/snapshotd` | [`images/snapshotd`](../images/snapshotd): both builds of [`daemon/`](../daemon), with NIXL and UCX |
| `ghcr.io/wseaton/snapshotd/snapshot-agent` | Dynamo Snapshot `bc9d216` + [`patches/dynamo-snapshot`](../patches/dynamo-snapshot), CRIU `1a93585` |
| `quay.io/wseaton/kserve-dynamo-snapshot@sha256:77c68762…` | `kserve-dynamo-snapshot/Dockerfile` |
| `ghcr.io/ai-dynamo/snapshot/{operator,agent}:v0.0.0-gbc9d2161` | upstream |
| `nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.6.0` | upstream |

## Dynamo Snapshot through KServe

Needs KServe with `LLMInferenceService` and the ModelExpress annotations
([kserve#6408](https://github.com/kserve/kserve/pull/6408)), its gateway, and cert-manager.

```bash
# chart from the pinned checkout; kustomize only reads charts below the kustomization
git submodule update --init
./checkout.sh dynamo-snapshot
mkdir -p deploy/dynamo-snapshot/agent/charts
cp -RL src/dynamo-snapshot/charts/snapshot deploy/dynamo-snapshot/agent/charts/

# operator and agent, on labelled nodes only
kubectl label node <node> snapshot.neuralmagic.com/agent=true
kustomize build --enable-helm deploy/dynamo-snapshot/agent | kubectl apply --server-side -f -

# companion
kubectl apply -k deploy/dynamo-snapshot/companion

# namespace and model; skip the CRDs if the cluster has them
kubectl apply -k deploy/modelexpress/crds
kubectl apply -k deploy/dynamo-snapshot/qwen3-8b/namespace
kubectl -n kds-demo wait job/download-qwen3-8b --for=condition=Complete --timeout=30m

# service, after the download
kubectl apply -k deploy/dynamo-snapshot/qwen3-8b/service
kubectl -n kds-demo patch llminferenceservice qwen3-8b --type=merge -p '{"spec":{"replicas":2}}'
```

[`agent/values.yaml`](dynamo-snapshot/agent/values.yaml) turns PageBroker off (needs a driver
newer than 580), sets CRIU's `mntnsCompatMode` on and `extMasters` off.

Behavior:

- A snapshot belongs to the Deployment UID plus `pod-template-hash`. Any template change captures
  again; going back to a template still in the revision history reuses its snapshot.
- The first replica of a revision is held unschedulable while a `SnapshotJob` donor captures on
  its capacity. Later replicas restore directly.
- Capture failure: the revision cold boots. Restore failure: the `PodSnapshot` is labelled
  `snapshot.neuralmagic.com/rejected=true` and the pod is replaced with a cold boot.
- Prefill/decode, multi-node, KV events and precise prefix-cache routing cold boot.
- Old revisions' snapshots are not pruned. To recapture:
  `kubectl -n kds-demo delete snapshotjob,podsnapshot <name>`.

## `snapshotd` bench pods

One privileged pod per node, taking all 8 GPUs, driven by
[`experiments/bench`](../experiments/bench).

```bash
kubectl apply -k deploy/snapshotd/single-node
kubectl apply -k deploy/snapshotd/models/qwen3-8b
kubectl -n snapshot-bench wait job/download-qwen3-8b --for=condition=Complete --timeout=30m

kubectl -n snapshot-bench exec restore-bench -- env MODEL=/models/Qwen3-8B TP=4 REMAP=rotate \
  MODE=cuinterpose COMPAT=/usr/local/cuda-13.1/compat \
  /exp/restore-bench.sh qwen-tp4 --max-model-len 4096 --gpu-memory-utilization 0.6
```

It prints one `RESULT` JSON line; `"result": "MATCH"` means the restored server's greedy completion
matched.

Two nodes (delete the single-node pod first):

```bash
kubectl apply -k deploy/modelexpress/crds
kubectl apply -k deploy/snapshotd/two-node
kubectl apply -k deploy/snapshotd/models/deepseek-v4.1-flash

cd experiments/bench
NS=snapshot-bench CONTEXT=$(kubectl config current-context) \
MODEL=/models/DeepSeek-V4.1-Flash TP=8 MODE=cuinterpose COMPAT=/usr/local/cuda-13.1/compat \
MX=1 NET=pod PEER=1 OVERLAP=1 IMG_DIR=/mem/deepseek RESERVE_GIB=76 CRIU_VERBOSITY=1 \
./lifecycle.sh deepseek --enable-expert-parallel --tokenizer-mode deepseek_v41 --language-model-only \
  --engram-config '{"cpu_offload":false}' --max-model-len 32768 --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 --max-num-batched-tokens 16384
```

The pods repel each other; pin nodes with a `nodeSelector` patch in
[`two-node/`](snapshotd/two-node). Results land in `lifecycle-<tag>/`.

### `lifecycle.sh` settings

| Variable | Measured | Meaning |
| --- | --- | --- |
| `OVERLAP` | `1` | `0` dump then copy; `1` stream during the dump; `memory` restore from the receiver's arena |
| `IMG_DIR` | `/mem/<tag>` | Image location on the target; under `/mem` it is RAM-backed |
| `RESERVE_GIB` | image size | Pre-allocate `/mem` so page cache is reclaimed before the dump |
| `SLOTS`, `SLOT_MIB` | 64, 64 | Transfer pool shape |
| `MX`, `NET`, `PEER` | `1`, `pod`, `1` | ModelExpress weights, restore into the pod network, cold boot a peer |

The [`rdma-image-transfer`](snapshotd/components/rdma-image-transfer) component adds the `xfer`
container (`rdma/ib`, `IPC_LOCK`, `SYS_RESOURCE`), a 256Gi memory-backed `/mem`, and the receiver on
TCP 7700. By hand:

```bash
snapshotd-nixl transfer receive --dir /mem/<tag> --listen 0.0.0.0:7700 --mmap     # target
snapshotd-nixl transfer stream --images-dir /mem/<tag> --to <target IP>:7700 \
  --slots 64 --slot-mib 64                                                       # source
```

## Cluster-specific settings

| Setting | Where |
| --- | --- |
| Storage class and volume sizes | `dynamo-snapshot/agent/values.yaml`, `dynamo-snapshot/qwen3-8b/namespace/model.yaml`, `snapshotd/base/storage.yaml` |
| Pull secret name | `companion/companion.yaml`, `config-layer/kustomization.yaml` |
| Image references | `components/images`; `images:` in `companion/` and `config-layer/` |
| Agent image, PageBroker, CRIU options | `dynamo-snapshot/agent/values.yaml` |
| Agent node label | `agent/values.yaml`, `qwen3-8b/service/service.yaml` |
| Gateway namespace label `inference-gateway-access` | `qwen3-8b/namespace/namespace.yaml` |
| GPUs and `rdma/ib` per bench pod | `snapshotd/pod/pod.yaml`, `snapshotd/components/modelexpress-weights/weights.yaml` |
| Namespaces `kds-demo`, `snapshot-bench` | each kustomization's `namespace:`; `NS` for `lifecycle.sh` |

## Removing

```bash
kubectl delete -k deploy/dynamo-snapshot/qwen3-8b/service
kubectl -n kds-demo delete snapshotjobs,podsnapshots --all
kubectl delete -k deploy/dynamo-snapshot/qwen3-8b/namespace
kubectl delete -k deploy/dynamo-snapshot/companion
kustomize build --enable-helm deploy/dynamo-snapshot/agent | kubectl delete -f -
kubectl delete -k deploy/snapshotd/two-node
```

Deleting a namespace kustomization deletes its volumes, downloaded models included.
