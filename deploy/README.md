# Deploying

Two ways to restore a vLLM replica from a checkpoint, both as kustomize:

| | [`dynamo-snapshot/`](dynamo-snapshot) | [`snapshotd/`](snapshotd) |
| --- | --- | --- |
| Node agent | Dynamo Snapshot's, at `bc9d216` | `snapshotd`, called from a script in a privileged pod |
| Driven by | KServe: scaling an `LLMInferenceService` captures and restores | [`experiments/bench`](../experiments/bench) scripts; no KServe integration yet |
| Checkpoint goes to | a shared `ReadWriteMany` volume | the node's disk, or over InfiniBand into the target node's RAM while CRIU dumps |
| Weights after restore | ModelExpress from a serving replica, else the model volume | the same |
| Largest configuration run | Qwen3-8B, TP=1 | DeepSeek-V4.1-Flash, TP=8 with expert parallel, across two nodes |

The manifests hold the configuration of the measured runs in
[docs/benchmarks.md](../docs/benchmarks.md) and [docs/cross-node.md](../docs/cross-node.md). Those
runs created the same objects with scripts: `kserve-dynamo-snapshot/hack/smoke-waldorf.sh`, against
KServe in a vCluster with the agent on the host, and a hand-edited pod spec. This tree has been
built, dry-run against that cluster's API servers, and its build Job has run. Neither path has
been applied end to end from it.

## Layout

```
deploy/
  modelexpress/
    crds/                    ModelExpress CRDs, v0.6.0 (cluster-scoped)
    server/                  ModelExpress server, Kubernetes metadata backend, one per namespace
  dynamo-snapshot/
    agent/                   Dynamo Snapshot operator and agent: its Helm chart, inflated and patched
    companion/               admission webhook and controller that capture on scale-up
    config-layer/            the LLMInferenceServiceConfig a service opts into
    qwen3-8b/namespace/      namespace, model volume and download, config layer, ModelExpress server
    qwen3-8b/service/        the LLMInferenceService
  snapshotd/
    base/                    namespace, volumes, script configmap, snapshotd build Job
    pod/                     the bench pod
    components/
      rdma-image-transfer/   xfer container and RAM-backed image directory
      modelexpress-weights/  rdma/ib for the workload, pod-info volume, ModelExpress server
    single-node/             base and one pod
    two-node/                base, two pods on different nodes, both components
    models/                  download Jobs for Qwen3-8B and DeepSeek-V4.1-Flash
```

## Before either path

- `kubectl`, `kustomize` 5 or later, and `helm` for the agent's chart.
- GPU nodes. Measured on 8x H200, host driver 580.126.09, containerd.
- A `ReadWriteMany` storage class. The manifests name `shared-vast`.
- For the RDMA parts, an InfiniBand device plugin that offers `rdma/ib`, and transparent huge
  pages on the nodes.
- The images under `quay.io/wseaton` are private. Each namespace below needs a pull secret named
  `quay-wseaton-pull`:

  ```bash
  kubectl -n <namespace> create secret docker-registry quay-wseaton-pull \
    --from-file="$HOME/.docker/config.json"
  ```

  To use your own builds, change the `images:` entries and image references listed under
  [Cluster-specific settings](#cluster-specific-settings).

### Images

| Image | Used for | Built from |
| --- | --- | --- |
| `quay.io/wseaton/vllm-checkpoint@sha256:e2659cc1…` | vLLM in both paths | `kserve-dynamo-snapshot/images/vllm-checkpoint`: vLLM nightly `a2bcb565` with files from the patched fork, and ModelExpress from its fork branch |
| `quay.io/wseaton/kserve-dynamo-snapshot@sha256:77c68762…` | companion, snapshot entrypoint | `kserve-dynamo-snapshot/Dockerfile` |
| `quay.io/wseaton/snapshot-agent@sha256:7fe69503…` | Dynamo agent; in the bench pod, an image volume for CRIU, its plugins and cuInterpose | Dynamo Snapshot `bc9d216` with [`patches/dynamo-snapshot/0001`](../patches/dynamo-snapshot) |
| `ghcr.io/ai-dynamo/snapshot/operator:v0.0.0-gbc9d2161`, `…/agent:v0.0.0-gbc9d2161` | Dynamo operator; cuInterpose bundle the companion injects | upstream |
| `quay.io/wseaton/vllm-openai@sha256:20fd867e…` | image volume for the CUDA 13.1 compat `libcuda` | `kserve-dynamo-snapshot/images/vllm-cuda-compat` |
| `quay.io/wseaton/nixl-regbench@sha256:da96ebb4…` | xfer container, and the `snapshotd` build with the `nixl` feature | NIXL with [ai-dynamo/nixl#2672](https://github.com/ai-dynamo/nixl/pull/2672), UCX and Rust. Its Dockerfile is not in this repository. |
| `nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.6.0` | ModelExpress server | upstream |

## Dynamo Snapshot through KServe

Needs KServe with `LLMInferenceService` and the ModelExpress annotations
([kserve/kserve#6408](https://github.com/kserve/kserve/pull/6408)), its gateway, and cert-manager.

### 1. Sources

The agent's chart comes from the pinned Dynamo Snapshot checkout, because no published chart
matches `bc9d216`. Kustomize only reads charts below the kustomization, so copy it in:

```bash
git submodule update --init
./checkout.sh dynamo-snapshot
mkdir -p deploy/dynamo-snapshot/agent/charts
cp -RL src/dynamo-snapshot/charts/snapshot deploy/dynamo-snapshot/agent/charts/
```

### 2. Operator and agent

The agent is a privileged DaemonSet. It runs only on nodes you label, and services restore only
on those nodes:

```bash
kubectl label node <node> snapshot.neuralmagic.com/agent=true
kustomize build --enable-helm deploy/dynamo-snapshot/agent | kubectl apply --server-side -f -
kubectl -n kds-snapshot-system create secret docker-registry quay-wseaton-pull \
  --from-file="$HOME/.docker/config.json"
kubectl -n kds-snapshot-system rollout status daemonset/kds-snapshot-agent
kubectl -n kds-snapshot-system rollout status deployment/kds-snapshot-operator
```

[`agent/values.yaml`](dynamo-snapshot/agent/values.yaml) turns PageBroker off (it needs a newer
driver than 580), sets CRIU's `mntnsCompatMode` on and `extMasters` off, and sizes the checkpoint
volume. The kustomization swaps in the agent built with the CoreWeave patch and removes
`runtimeClassName: nvidia`.

### 3. Companion

```bash
kubectl apply -k deploy/dynamo-snapshot/companion
kubectl -n kserve-dynamo-snapshot create secret docker-registry quay-wseaton-pull \
  --from-file="$HOME/.docker/config.json"
kubectl -n kserve-dynamo-snapshot rollout status deployment/companion
```

### 4. Namespace and model

Skip the first command on a cluster that already has the ModelExpress CRDs.

```bash
kubectl apply -k deploy/modelexpress/crds
kubectl apply -k deploy/dynamo-snapshot/qwen3-8b/namespace
kubectl -n kds-demo create secret docker-registry quay-wseaton-pull \
  --from-file="$HOME/.docker/config.json"
kubectl -n kds-demo wait job/download-qwen3-8b --for=condition=Complete --timeout=30m
```

### 5. Service

Apply it only after the download finishes: the first capture cold-boots from the model volume.

```bash
kubectl apply -k deploy/dynamo-snapshot/qwen3-8b/service
kubectl -n kds-demo get pods,snapshotjobs,podsnapshots --watch
```

The first replica is created unschedulable, labelled
`snapshot.neuralmagic.com/awaiting-capture=true`. The companion runs a `SnapshotJob` donor on the
capacity it would have used, and once the `PodSnapshot` is ready the replica is released and
restores from it. Later replicas restore without the hold:

```bash
kubectl -n kds-demo patch llminferenceservice qwen3-8b --type=merge -p '{"spec":{"replicas":2}}'
kubectl -n kds-demo get pod <new pod> -o jsonpath='{.metadata.annotations.nvidia\.com/restore-from}'
kubectl -n kds-demo logs <new pod> -c main | grep "RDMA transfer complete"
```

Measured for this service: first replica Ready 156 s after creation (cold boot, capture, restore),
later replicas 37 to 41 s after scale-up with weights from a peer, cold boot 89 s.
`kserve-dynamo-snapshot/hack/smoke-waldorf.sh verify` automates the same checks against its own
install. TP>1 has not been rerun on this path since it moved to vLLM's checkpoint routes.

## `snapshotd`

One privileged pod per node holds vLLM, CRIU and `snapshotd`; the scripts in
[`experiments/bench`](../experiments/bench) run the phases and time them. The pod takes a whole
node's 8 GPUs. The cluster needs image volumes (`ImageVolume`).

### One node

```bash
kubectl apply -k deploy/snapshotd/single-node
kubectl -n snapshot-bench create secret docker-registry quay-wseaton-pull \
  --from-file="$HOME/.docker/config.json"
kubectl -n snapshot-bench wait job/build-snapshotd --for=condition=Complete --timeout=15m
kubectl apply -k deploy/snapshotd/models/qwen3-8b
kubectl -n snapshot-bench wait job/download-qwen3-8b --for=condition=Complete --timeout=30m
kubectl -n snapshot-bench wait pod/restore-bench --for=condition=Ready --timeout=10m
```

`build-snapshotd` compiles [`daemon/`](../daemon) from `SNAPSHOTD_REF` of this repository (default
`main`) onto the models volume: `/models/bin/snapshotd`, and `/models/bin/snapshotd-nixl` with
the `nixl` feature.

Cold boot, checkpoint and restore onto the next GPU over, with cuInterpose:

```bash
kubectl -n snapshot-bench exec restore-bench -- env MODEL=/models/Qwen3-8B TP=4 REMAP=rotate \
  MODE=cuinterpose COMPAT=/compat/usr/local/cuda-13.1/compat \
  /exp/restore-bench.sh qwen-tp4 --max-model-len 4096 --gpu-memory-utilization 0.6
```

It prints one `RESULT` line of JSON. `"result": "MATCH"` means the restored server gave the same
greedy completion as before the dump. Measured: `restore_total_s` 8.0 against `cold_boot_s` 48.8.
The path without cuInterpose needs the patched vLLM files as a second configmap; see
[experiments/README.md](../experiments/README.md).

### Two nodes

Capture on one node, stream the image to the other over InfiniBand, restore there with weights
from a server on the first node. Delete a single-node pod first, since a pod's spec cannot change:

```bash
kubectl -n snapshot-bench delete pod -l app=snapshot-bench --ignore-not-found
kubectl apply -k deploy/modelexpress/crds
kubectl apply -k deploy/snapshotd/two-node
kubectl apply -k deploy/snapshotd/models/deepseek-v4.1-flash
kubectl -n snapshot-bench wait job/download-deepseek-v4-1-flash --for=condition=Complete --timeout=6h
kubectl -n snapshot-bench wait pod/restore-bench pod/restore-bench-b --for=condition=Ready --timeout=10m
```

The two pods repel each other, so they land on different nodes. To choose the nodes, add a
`nodeSelector` patch to [`two-node/kustomization.yaml`](snapshotd/two-node/kustomization.yaml) and
[`two-node/target/kustomization.yaml`](snapshotd/two-node/target/kustomization.yaml); the measured
pair shared an InfiniBand leaf group.

The harness runs from a workstation and drives both pods with `kubectl exec`:

```bash
cd experiments/bench
NS=snapshot-bench CONTEXT=$(kubectl config current-context) \
MODEL=/models/DeepSeek-V4.1-Flash TP=8 MODE=cuinterpose COMPAT=/compat/usr/local/cuda-13.1/compat \
MX=1 NET=pod PEER=1 OVERLAP=1 IMG_DIR=/mem/deepseek RESERVE_GIB=76 CRIU_VERBOSITY=1 \
./lifecycle.sh deepseek --enable-expert-parallel --tokenizer-mode deepseek_v41 --language-model-only \
  --engram-config '{"cpu_offload":false}' --max-model-len 32768 --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 --max-num-batched-tokens 16384
```

Results land in `lifecycle-deepseek/`: `capture.json`, `restore.json`, and `lifecycle.html`, the
run drawn as a page. Measured: 25.5 s from the start of the checkpoint to a complete 74 GB image
on the second node, restore 20.5 s, cold boot 104.7 s. A smaller run of the same shape:

```bash
NS=snapshot-bench CONTEXT=$(kubectl config current-context) \
MODEL=/models/Qwen3-8B TP=4 MODE=cuinterpose COMPAT=/compat/usr/local/cuda-13.1/compat \
MX=1 NET=pod PEER=1 OVERLAP=1 IMG_DIR=/mem/qwen \
./lifecycle.sh qwen --max-model-len 4096 --gpu-memory-utilization 0.8
```

## Rollouts

Scaling and rollouts are manual: every run changed `spec.replicas` or the pod template by hand.
Nothing here configures KEDA or WVA. KServe's `spec.scaling` changes the same Deployment's
replicas and the companion acts at pod admission, so an autoscaler should take the same path. That
has not been run.

### Through KServe

A snapshot belongs to one revision of the workload Deployment: the Deployment's UID plus the
ReplicaSet's `pod-template-hash`. The companion decides for each new pod whether it restores,
waits for a capture, or cold-boots, and never denies a pod.

| Change | What the next pods do |
| --- | --- |
| Scale up within a revision | Restore from the revision's snapshot, on any labelled node. No hold. |
| Scale to zero and back | The same revision, so the same snapshot. With no replica serving, the weights come from the model volume. |
| Any pod template change: image, arguments, environment, resources, `kubectl rollout restart` | A new revision with no snapshot. Its first pods are held unscheduled, a donor cold-boots on the capacity they would use and is checkpointed, then they are released and restore. |
| Back to an earlier template | The Deployment reuses that ReplicaSet and its hash while it is still in the revision history, so its snapshot is used again. |
| Service deleted and recreated | A new Deployment UID. The `SnapshotJob` is owned by the Deployment and goes with it; the first pods capture again. |
| New model files under the same URI | Not a new revision. Pods restore a process that was captured with the old files and reloads the new weights. Change the template or delete the snapshot. |

What a rollout costs:

- **One capture per revision.** For Qwen3-8B the first replica of a revision was Ready 156 s after
  creation, against 89 s for a cold boot and 37 to 41 s for the replicas after it. A rollout of a
  one-replica service is slower than without snapshots; the gain is in the replicas that follow.
- **Capacity for the donor.** A held pod holds no GPUs, so the donor needs one replica's worth
  free on a labelled node while the old revision still serves. Without it the donor stays pending
  until the `SnapshotJob` deadline (`--snapshot-job-deadline-seconds`, 3600 by default).
- **Checkpoint storage.** Snapshots of old revisions are not pruned. They go when the service is
  deleted, or with `kubectl delete podsnapshot`.

When something fails, the revision cold-boots:

- **Capture fails or times out.** Held pods are deleted and their replacements cold-boot, as does
  every later pod of that revision.
- **Restore fails.** The `PodSnapshot` is labelled `snapshot.neuralmagic.com/rejected=true` with
  the reason in `snapshot.neuralmagic.com/rejected-reason`, the pod is deleted, and its
  replacement cold-boots. A node whose GPU model or driver differs from the capture node's fails
  this way, so delete snapshots after a driver upgrade.
- **Not eligible.** Services with prefill/decode or multi-node workers, vLLM KV events, or precise
  prefix-cache routing in the endpoint picker cold-boot from the start.

To see which snapshot a revision has, and to capture it again:

```bash
kubectl -n kds-demo get podsnapshots \
  -L snapshot.neuralmagic.com/pod-template-hash,snapshot.neuralmagic.com/rejected
kubectl -n kds-demo delete snapshotjob,podsnapshot <name>
```

With both gone, the next pod created for that revision is held and a new capture runs. Labelling a
`PodSnapshot` rejected by hand turns restores off for its revision without deleting anything,
which is how the smoke test takes its cold-boot comparison. The delete-and-recapture path follows
from the admission logic and has not been exercised.

### `snapshotd`

There is no controller. A capture is one run of the harness, and an image is good for the vLLM
image, arguments, GPU model and driver it was taken with; nothing records or checks that. A
rollout means capturing again.

## RDMA image transfer

`snapshotd transfer` moves the CRIU image between nodes through one registered huge-page pool on
each side ([docs/cross-node.md](../docs/cross-node.md#how-the-copy-works)). The
[`rdma-image-transfer`](snapshotd/components/rdma-image-transfer) component is what a pod needs
for it:

| In the pod | Value | For |
| --- | --- | --- |
| `xfer` container | the NIXL image, running `/models/bin/snapshotd-nixl` | NIXL and UCX stay out of the workload container |
| `rdma/ib` on `xfer` | 1 | the InfiniBand device the copy uses |
| `IPC_LOCK` | added | registering the pool |
| `SYS_RESOURCE` | added | raising CRIU's pipes to 64 MiB; at the 1 MiB unprivileged limit the dump runs at about 400 MB/s |
| `/mem` in both containers | memory-backed `emptyDir`, 256Gi | the image on the target, read by `criu restore` from the workload container |
| `xfer` memory limit | 256Gi | the image in `/mem` is charged to the pod |
| TCP 7700 on the pod IP | receiver | the sender connects to the target pod directly |

`lifecycle.sh` picks the mode from its environment:

| Variable | Measured setting | Meaning |
| --- | --- | --- |
| `OVERLAP` | `1` | `0`: dump to the source's disk, then copy. `1`: `criu dump --stream` into `snapshotd transfer stream`, which sends while CRIU dumps and writes nothing on the source. `memory`: as `1`, and the target serves the restore from its huge-page arena, which made CRIU's restore slower. |
| `IMG_DIR` | `/mem/<tag>` | where the image lands on the target. Under `/mem` the receiver copies slots into RAM-backed files in parallel (`--mmap`); elsewhere it writes with `O_DIRECT`, limited by the disk to about 3.4 GB/s. |
| `RESERVE_GIB` | the image size, 76 for DeepSeek | allocates and frees that much of `/mem` before the dump, so the kernel reclaims page cache then and not under the transfer |
| `SLOTS`, `SLOT_MIB` | 64, 64 | the pool: 64 slots of 64 MiB |
| `XFER` | `/models/bin/snapshotd-nixl` | the binary built with the `nixl` feature |

What the harness runs, to do the same by hand:

```bash
# target
snapshotd-nixl transfer receive --dir /mem/<tag> --listen 0.0.0.0:7700 --mmap
# source, started once CRIU's image directory exists
snapshotd-nixl transfer stream --images-dir /mem/<tag> --to <target pod IP>:7700 --slots 64 --slot-mib 64
```

The Dynamo Snapshot path has no equivalent: its checkpoints go to the shared volume.

## Weights from a peer

Both paths release the weights before the checkpoint (`--checkpoint-release-weights`) and reload
them at restore through ModelExpress: from a replica that is serving, over RDMA, or from the model
volume when there is none.

- **Server.** [`modelexpress/server`](modelexpress/server) keeps its metadata in the namespace it
  runs in, so each workload namespace gets its own. Clients reach it as `modelexpress-server:8001`.
- **Through KServe.** The service carries `serving.kserve.io/exp-modelexpress-mode: layered` and
  `serving.kserve.io/exp-modelexpress-address`, and requests `rdma/ib`. The config layer sets
  `UCX_TLS=^rc_gda` and the pod-info volume ModelExpress reads its identity from.
- **Bench pods.** The [`modelexpress-weights`](snapshotd/components/modelexpress-weights)
  component gives the workload one `rdma/ib` per GPU and the pod-info volume. `MX=1 NET=pod` on
  the scripts loads through ModelExpress and restores into the pod's own network; `PEER=1` cold
  boots a second server on the source node to pull from.

## Cluster-specific settings

| Setting | Where |
| --- | --- |
| Storage class `shared-vast` and volume sizes | `dynamo-snapshot/agent/values.yaml`, `dynamo-snapshot/qwen3-8b/namespace/model.yaml`, `snapshotd/base/storage.yaml` |
| Pull secret name | `agent/values.yaml`, `companion/companion.yaml`, `config-layer/kustomization.yaml`, `snapshotd/base/build.yaml`, `snapshotd/pod/pod.yaml` |
| Image references | `images:` in `agent/`, `companion/` and `config-layer/`; `kserve-dynamo-snapshot/config/llmisvcconfig`; `snapshotd/pod/pod.yaml`; `snapshotd/components/rdma-image-transfer/xfer.yaml`; `snapshotd/base/build.yaml` |
| CoreWeave agent image, no `runtimeClassName` | `dynamo-snapshot/agent/kustomization.yaml` |
| PageBroker and CRIU options for driver 580 | `dynamo-snapshot/agent/values.yaml` |
| Node label for the agent and for services | `agent/values.yaml`, `qwen3-8b/service/service.yaml` |
| Namespace label `inference-gateway-access`, which the test cluster's gateway requires | `qwen3-8b/namespace/namespace.yaml` |
| GPUs for a bench pod (8) and `rdma/ib` for its workload (one per GPU) | `snapshotd/pod/pod.yaml`, `snapshotd/components/modelexpress-weights/weights.yaml` |
| Namespaces `kds-demo` and `snapshot-bench` | the `namespace:` of each kustomization and its `namespace.yaml`; `NS` for `lifecycle.sh` |

## Removing

```bash
kubectl delete -k deploy/dynamo-snapshot/qwen3-8b/service
kubectl -n kds-demo delete snapshotjobs,podsnapshots --all
kubectl delete -k deploy/dynamo-snapshot/qwen3-8b/namespace
kubectl delete -k deploy/dynamo-snapshot/companion
kustomize build --enable-helm deploy/dynamo-snapshot/agent | kubectl delete -f -

kubectl delete -k deploy/snapshotd/two-node
```

Deleting a namespace kustomization deletes its volumes, the downloaded models included.
