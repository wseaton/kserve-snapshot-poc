+++
title = "Running the daemon"
weight = 10

[extra]
nav = "Running"
mermaid = true
+++

`snapshotd node` runs as a privileged DaemonSet. Workload pods stay unprivileged and run stock
vLLM. Manifests: [`deploy/snapshotd`](https://github.com/wseaton/snapshotd/tree/main/deploy/snapshotd).

{% mermaid() %}
flowchart LR
  subgraph A["Node A"]
    direction TB
    subgraph WA["Workload pod (unprivileged, stock vLLM image)"]
      SA["snapshotd-shim<br/>PID 1, readiness port"]
      VA["vLLM<br/>HTTP :8000, gRPC control :50051"]
    end
    DA["snapshotd pod<br/>DaemonSet, privileged, hostPID<br/>kube-rbac-proxy sidecar"]
    STA[("checkpoint store<br/>node data disk")]
    RA["CRI socket<br/>kubelet PodResources socket"]
    DA -- "shim.sock through /proc/PID/root" --> SA
    DA -- "CheckpointPrepare / Restore" --> VA
    DA -- "CUDA checkpoint by host PID<br/>CRIU in the pod's namespaces and cgroup" --> WA
    DA <--> STA
    DA --> RA
  end
  subgraph B["Node B"]
    direction TB
    DB["snapshotd pod"]
    STB[("checkpoint store")]
    WB["Workload pod"]
    DB --> WB
    DB <--> STB
  end
  DA <-- "pull through kube-rbac-proxy: TLS, ServiceAccount token<br/>encrypted chunks over NIXL" --> DB
  K["kubelet"] -- "TCP readiness probe" --> SA
  C["clients, through the Service"] --> VA
{% end %}

A workload pod differs from a plain vLLM Deployment in four places
([example](https://github.com/wseaton/snapshotd/tree/main/daemon/deploy/workload)):

- the label `snapshot.llm-d.ai/workload`
- an init container that copies `snapshotd-shim` and the `io_uring` launcher into an `emptyDir`
- the command wrapped in `snapshotd-shim --`
- probes on the shim's port

## Try it

```bash
kubectl apply -k deploy/snapshotd/node              # DaemonSet; needs cert-manager; set its store path
kubectl apply -k deploy/snapshotd/node-openshift    # the same for OpenShift
kubectl apply -k deploy/snapshotd/workload          # example Deployment, Qwen3-8B
kubectl apply -k deploy/snapshotd/kserve/qwen3-8b   # or as an LLMInferenceService
daemon/e2e/run.sh                                   # whole flow on kind, no GPU, stand-in server
daemon/e2e/peers.sh                                 # checkpoints between three kind workers
```

Both e2e scripts run in CI. The stand-in
([`stand_in.rs`](https://github.com/wseaton/snapshotd/blob/main/daemon/examples/stand_in.rs))
serves vLLM's checkpoint RPCs and an instance ID drawn at start, which tells a restore from a boot.

## What it does

For each pod on its node with the workload label:

1. Find the container's PID through CRI and its GPUs through the kubelet's PodResources API.
2. If the node holds a checkpoint for `<namespace>/<workload>/<pod-template-hash>`, restore it and
   tell the shim to report ready.
3. Otherwise let it cold boot, checkpoint it, restore it, publish the checkpoint, report ready.
4. On failure the container exits and cold boots. No second attempt.

It writes nothing to the Kubernetes API. Its one read is its own node's labels, for
`--topology-levels`.

## Checkpoints between nodes

With the [`peers`](https://github.com/wseaton/snapshotd/tree/main/daemon/deploy/components/peers)
component, a daemon without a revision's checkpoint asks the other daemons for it before letting
the pod cold boot.

{% mermaid() %}
sequenceDiagram
  participant A as snapshotd, node B (asks)
  participant P as kube-rbac-proxy, node A
  participant K as API server
  participant H as snapshotd, node A (holds)
  A->>P: TLS, checked against the peers' CA
  A->>P: GET /v1/checkpoints/ns/workload/hash, ServiceAccount token, Upgrade
  P->>K: TokenReview, SubjectAccessReview
  K-->>P: snapshotd may get that path
  P->>H: the request, on localhost
  H-->>A: 101, then a key made for this transfer
  H->>A: control messages, through the proxy
  H-->>A: chunks over NIXL (RDMA or TCP), each encrypted with the key
{% end %}

- Daemons listen on localhost. Peers reach them through kube-rbac-proxy, which checks a projected
  token (pod-bound, 10 minutes, audience `snapshotd-peers`) against the `snapshotd-peers`
  ClusterRole.
- The asker verifies the proxy's certificate against a cert-manager CA before sending its token.
- Chunks go over NIXL, not the proxy, encrypted with a per-transfer AES-256-GCM key. File, offset
  and length are authenticated.
- A node pulls a revision once; pods starting together wait on that pull.
- Each daemon has `--transfer-lanes` (default 2) concurrent transfers. A daemon with no free lane
  answers busy; two daemons refusing each other back off by address.

### Choosing a peer

The asker queries every peer behind the headless Service at once (`GET` on the checkpoint path:
held, free lanes, location) and tries the nearest holder first. Nearness uses Kueue `Topology`
levels, widest first:

```text
--topology-levels   ib.coreweave.cloud/superpod, ib.coreweave.cloud/leafgroup, kubernetes.io/hostname
gf49e9c             0                            11                            gf49e9c
gf41fb2             0                            11                            gf41fb2     2 levels shared with gf49e9c
gf4334a             0                            391                           gf4334a     1 level shared
```

Among equally near holders, one with a free lane goes first. What it does not do yet is on
[Open](@/open.md#peer-selection).

## Store classes

A pod names a class with the annotation `snapshot.llm-d.ai/class` (default `default`). It is part
of the template, so part of the revision. A class the node lacks means cold boot, no checkpoint.

`default` comes from `--store-dir`, `--min-free-gib`, `--namespace-limit-gib` and
`--retain-hours`. Others come from the JSON file in `SNAPSHOTD_CLASSES_FILE`:

```json
{
  "shared": {"dir": "/stores/shared", "min_free_gib": 100, "namespace_limit_gib": 500, "retain_hours": 168}
}
```

- `dir`: where revisions go. Mount a PVC on the DaemonSet for any StorageClass.
- `min_free_gib`: no checkpoint below this much free space (default 0).
- `namespace_limit_gib`: above this, a namespace's new revisions cold boot.
- `retain_hours`: remove a revision no pod has started for this long (default: keep).

A class on a volume every node mounts holds one copy; nodes find it there without pulling.

## Metrics

Prometheus at `:9090/metrics` (`SNAPSHOTD_METRICS_LISTEN`). Stores are scanned and swept every
`SNAPSHOTD_SWEEP_S` seconds (600).

| Metric | Labels | Counts |
| --- | --- | --- |
| `snapshotd_pods_total` | `namespace`, `workload`, `how` | Pods brought up: `restore`, `capture`, `cold_boot`, `started_earlier` |
| `snapshotd_pod_failures_total` | `namespace`, `workload` | Failed attempts |
| `snapshotd_step_seconds` | `step` | `cold_boot`, `checkpoint`, `restore`, `pull` |
| `snapshotd_store_bytes`, `snapshotd_store_revisions` | `class`, `namespace`, `workload` | What a class holds |
| `snapshotd_store_free_bytes` | `class` | Free space |
| `snapshotd_swept_revisions_total`, `snapshotd_swept_bytes_total` | `class` | Removed past retention |
| `snapshotd_timeslice_operations_total` | `action`, `outcome` | [Time-slicing](@/timeslicing.md) operations |
| `snapshotd_timeslice_operation_seconds` | `action` | Their duration |

Every `snapshotd node` flag also reads from `SNAPSHOTD_<FLAG>`; `snapshotd node --help` lists them.

## OpenShift

[`deploy/snapshotd/node-openshift`](https://github.com/wseaton/snapshotd/tree/main/deploy/snapshotd/node-openshift)
swaps four things: the CRI-O socket, the `nvidia` runtime class, a binding to the `privileged` SCC,
and the service CA in place of cert-manager.

SELinux needed two fixes:

- CRIU's work directory and log live in the pod's `/run/snapshotd/work`, because `container_t`
  cannot append to the node store (`container_var_lib_t`).
- [`patches/criu`](https://github.com/wseaton/snapshotd/tree/main/patches/criu) makes CRIU decide
  SELinux from its own label (a pod's mount namespace has no selinuxfs), and the daemon passes the
  pod's context to `criu restore`. Without it the restored server stays `spc_t` and the shim
  cannot signal it.

Not yet applied to a cluster: the overlay's own SCC binding and peers with a service CA
certificate.

## KServe

[`deploy/snapshotd/kserve/config-layer`](https://github.com/wseaton/snapshotd/tree/main/deploy/snapshotd/kserve)
is an `LLMInferenceServiceConfig`. A service opts in with `baseRefs: [{name: snapshotd}]`. The
layer:

- labels the pods with the workload label
- adds the shim init container and wraps `vllm serve` (Rust frontend, gRPC control port, sleep
  mode, released weights)
- points every probe at the shim's port
- runs the server as non-root with `/run/snapshotd` as an `emptyDir`, and keeps caches and `HOME`
  in `/dev/shm`, which travels with the checkpoint

Not covered: TLS between gateway and vLLM, multi-node services (`LeaderWorkerSet` pods have no
`pod-template-hash`; use `--revision-label`), and the endpoint picker's connections to a pod being
checkpointed.

## Prefill and decode

Needs the NIXL connector with [vllm#64371](https://github.com/vllm-project/vllm/pull/64371),
carried as `patches/vllm/0007`. Examples:
[`workload-pd`](https://github.com/wseaton/snapshotd/tree/main/deploy/snapshotd/workload-pd) and
[`kserve/qwen3-8b-pd`](https://github.com/wseaton/snapshotd/tree/main/deploy/snapshotd/kserve/qwen3-8b-pd).

- The server runs with `--checkpoint-prepare-on-startup`, since a NIXL connector can be
  checkpointed only before its agent starts. Every replica, the first included, serves only after
  a restore.
- A restored replica gets a new KV engine id and side-channel address, so
  `VLLM_NIXL_SIDE_CHANNEL_HOST` must be the pod IP, not a name.

All runs so far used `UCX_TLS=tcp,cuda_copy`, so they say nothing about disaggregation cost.
