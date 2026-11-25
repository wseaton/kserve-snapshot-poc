# The node daemon under KServe

[`config-layer/`](config-layer) holds one `LLMInferenceServiceConfig`, named `snapshotd`. An
`LLMInferenceService` that lists it under `baseRefs` gets its replicas checkpointed and restored by
the `snapshotd` DaemonSet ([Running the daemon](https://wseaton.github.io/snapshotd/running/)). Nothing else is
installed for KServe: no webhook, no controller, no custom resource.

```yaml
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: qwen3-8b
spec:
  baseRefs:
    - name: snapshotd
  model:
    uri: pvc://models/Qwen3-8B
    name: Qwen/Qwen3-8B
  replicas: 1
```

[`qwen3-8b/`](qwen3-8b) is that service in full, with the layer.

A second layer, `snapshotd-pd`, is for a service with separate prefill and decode pods. It applies
the same changes to both of KServe's templates and differs in three ways: vLLM runs with
`--checkpoint-prepare-on-startup` and the NIXL connector in each pod's role, the shim's port is
8090 because the decode pod's routing sidecar and vLLM already have 8000 and 8001, and the prefill
pods are labelled `<service name>-prefill`. [`qwen3-8b-pd/`](qwen3-8b-pd) is an example, and
[Benchmarks](https://wseaton.github.io/snapshotd/benchmarks/#the-daemon) has what was measured.

## Before a service can use it

- KServe with `LLMInferenceService` at `serving.kserve.io/v1alpha2`.
- The daemon on every GPU node the service may land on: `kubectl apply -k deploy/snapshotd/node`.
  It needs cert-manager. A replica on a node without the daemon cold boots after 30 s and is
  never checkpointed.
- RBAC that lets the service's pods read their checkpoints. The daemon restores a pod only if its
  ServiceAccount may `get` `checkpoints.snapshot.llm-d.ai` in the namespace, and nothing grants
  that by default; without it every replica cold boots. To let every pod of a namespace restore:

  ```bash
  kubectl -n <namespace> create rolebinding snapshotd-checkpoint-reader \
    --clusterrole=snapshotd-checkpoint-reader --group=system:serviceaccounts:<namespace>
  ```

- The model on a volume (`pvc://`). The checkpoint does not hold the weights; a restored server
  reads them again from `/mnt/models`. Other storage schemes have not been run.
- The two images the layer names, pinned by [`deploy/components/images`](../../components/images):
  `snapshotd`, which the init container copies the shim out of, and `vllm-checkpoint`, vLLM with
  the checkpoint RPCs on its gRPC control port.

## Installing the layer

```bash
kubectl apply -k deploy/snapshotd/kserve/config-layer -n <namespace>
```

KServe looks a `baseRefs` name up in the service's own namespace first and then in its system
namespace (`kserve`). Apply the layer to a service's namespace to use it there, or once to the
system namespace to offer it to every namespace.

## How KServe combines it

For a service, KServe merges, in this order and each over the last:

1. its own defaults for the kind of service, such as `kserve-config-llm-template`;
2. each entry of `baseRefs`, in the order listed;
3. the service's own `spec`.

Then it runs the result through Go templates (`{{ .ObjectMeta.Name }}`, `{{ .Spec.Model.Name }}`,
`{{ .GlobalConfig.EnableTLS }}`), which is how the layer names things after the service.

Each step is a Kubernetes strategic merge. Three consequences shaped the layer:

- Containers, environment variables, volumes and ports merge by name, so the layer adds to the
  `main` container instead of redefining it, and a service can still add its own.
- A container's `command` is replaced whole. The layer's command stands in for KServe's, see
  [what it leaves out](#what-the-layers-command-leaves-out).
- A field can be changed but not removed. KServe's probes are HTTP, and a layer cannot turn them
  into TCP probes, so the shim's port answers HTTP and the layer only changes the probes' port.

## What the layer sets

| Field | Value | Why |
| --- | --- | --- |
| `spec.labels` | `snapshot.llm-d.ai/workload: <service name>` | The daemon handles pods that carry this label. The checkpoint is named after the Deployment KServe creates from the template (for example `qwen3-8b-kserve`), the namespace and the pod's `pod-template-hash`. The label's value names it only for a pod that no controller owns |
| init container `snapshot-tools` | copies `snapshotd-shim` and `noiouring.py` into an `emptyDir` | The shim has to be PID 1 of the server's container, and the vLLM image does not contain it |
| `main` command | `snapshotd-shim -- bash -c 'exec … vllm serve …'` | The shim starts the server when the daemon says to, or leaves room for a restored one |
| `LD_PRELOAD`, in the command | `/run/snapshotd/tools/cuinterpose/libcuinterpose.so` | The library that lets workers which share GPU memory be checkpointed. The daemon mounts it into the pod before the server starts |
| `LD_LIBRARY_PATH`, in the command | `/usr/local/cuda-13.1/compat` first | cuInterpose needs the CUDA 13.1 `libcuda`, which is newer than the one a 580 driver installs. The image carries it off the default library path |
| `noiouring.py` | launches `vllm` | CRIU cannot dump a process that holds an `io_uring` |
| vLLM flags | `--grpc-port 50051 --enable-sleep-mode --checkpoint-release-weights` | The control port the daemon drives, and the lifecycle it drives through it |
| `VLLM_USE_RUST_FRONTEND` | `1` | The checkpoint RPCs are served by the Rust frontend only |
| ports | `control` 50051, `ready` 8001 | The daemon's and the kubelet's |
| startup, readiness and liveness probes | port 8001, plain HTTP | 8001 is the shim's. It opens when the daemon reports the server serving, so the pod is not Ready during the cold boot that precedes a checkpoint, and the kubelet holds no connection to the server while it is dumped |
| `securityContext` | `runAsUser: 1000` | The image's default user is root, and KServe's template requires a non-root one |
| volume `snapshot-run` at `/run/snapshotd` | `emptyDir` | Where the shim puts its socket and the server's log, writable by uid 1000 |
| volume `dshm` | 64 Gi, in memory | `/dev/shm` travels with the checkpoint. KServe's default of 1 Gi is too small for what goes there |
| `HOME` and the cache variables | under `/dev/shm` | A file the server has open has to exist in the pod it is restored into. Files on the container's own filesystem or in an `emptyDir` do not follow the checkpoint; `/dev/shm` does |
| `VLLM_WORKER_MULTIPROC_METHOD`, the `NCCL_*` and `GLOO_*` variables, `VLLM_HOST_IP` | `spawn`, loopback | Workers that talk to each other over loopback and without InfiniBand, which is what has been checkpointed |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | CUDA numbers the pod's GPUs by bus position, the same way on every node |
| `PYTORCH_ALLOC_CONF` | no pinned-memory cache | Carried over from the configuration every measured run used; not tested without it |
| `KSERVE_INFER_ROCE` | `false` | See below |

## What a service can still set

Anything KServe lets a service set that the table does not list: `replicas`, `model`, resources,
node selection, the router and scheduler, autoscaling. For the server itself:

- `VLLM_ADDITIONAL_ARGS` and the `main` container's `args` are passed to `vllm serve`.
- `spec.parallelism.tensor` and `.expert` become `--tensor-parallel-size` and
  `--enable-expert-parallel`. Read the limit on tensor parallel below a whole node in
  [Limits](https://wseaton.github.io/snapshotd/limits/#tensor-parallel-below-a-whole-node)
  first.
- More environment variables, by name.

What a service should leave alone: the `main` container's `command`, its probes and its
`runAsUser`. Changing any of them in the service overrides the layer.

Any change to the pod template, in the layer or the service, is a new revision: its first replica
on each node cold boots and is checkpointed again.

## What the layer's command leaves out

KServe's default command does more than start vLLM, and the layer's replaces it:

- RoCE discovery (`KSERVE_INFER_ROCE`) is off. The layer keeps NCCL on loopback.
- KV cache offloading is refused: the server exits at start if KServe passes
  `--kv-transfer-config`.
- The version checks around `--disable-access-log-for-endpoints` and `--shutdown-timeout` are
  gone, since the layer pins the vLLM image.
- With TLS enabled the layer passes the certificate and key but not `--enable-ssl-refresh`. TLS
  between the gateway and vLLM has not been run with it.

## Checking that it took

```bash
kubectl get pods -l snapshot.llm-d.ai/workload=<service> -L pod-template-hash
kubectl -n snapshotd logs ds/snapshotd -c snapshotd | grep 'pod{'
```

The daemon on the pod's node logs one line per step. A first replica ends with
`serving … how=cold boot, checkpoint, restore`; a later one with `how=restore`, after
`pulled the checkpoint from a peer` if its node did not have it. `how=cold boot` with a `because=`
means the daemon decided not to checkpoint or restore, and says why.

## Measured

Qwen3-8B at TP=1 on H200, KServe in a vCluster with the daemon on the host cluster:

| Replica | Ready after | How |
| --- | --- | --- |
| first | 85 s | 60 s cold boot, 7.8 s checkpoint (4.3 GiB), 4.3 s restore |
| second, same node | 8 s | restore from the node's checkpoint |
| third and fourth, another node | 30 s each | one 6.7 s pull shared by both, then both restored at once in 12 s |

A completion through the KServe gateway and the endpoint picker was answered by the restored
pods.

## Limits

- Single-node services only. KServe runs those as a Deployment, whose pods carry
  `pod-template-hash`. Multi-node services are `LeaderWorkerSet` pods without it, and
  checkpointing several pods as one has not been tried.
- A layer that sets `spec.prefill` makes any service that uses it a prefill and decode service,
  which is why there are two layers and a service picks one.
- The endpoint picker scrapes the server's metrics port. What it does with a connection it holds
  to a pod that is being checkpointed has not been looked at.
- A service that never scales still pays for one checkpoint on its first replica.

## Removing

Take `snapshotd` out of the service's `baseRefs`; KServe rolls the pods back to its own template.
Then delete the `LLMInferenceServiceConfig`. Checkpoints stay on the nodes' disks, under the
daemon's store, until they are deleted there.
