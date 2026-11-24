+++
title = "Time-slicing"
weight = 20

[extra]
mermaid = true
+++

With `SNAPSHOTD_TIMESLICE_LISTEN` set
([`components/timeslice`](https://github.com/wseaton/snapshotd/tree/main/daemon/deploy/components/timeslice)),
the daemon serves the `SnapshotAgentService` of
[llm-d-rl-time-slicing](https://github.com/llm-d-incubation/llm-d-rl-time-slicing), whose
orchestrator lets jobs take turns on a GPU. The project's orchestrator and clients work against it
unchanged. A snapshot here pauses a job in place: processes stay alive, device memory goes to host
memory, nothing goes to the store.

{% mermaid() %}
sequenceDiagram
  participant J as job B (timeslice client)
  participant O as TimeSlice Orchestrator
  participant D as snapshotd, on the jobs' node
  J->>O: Acquire(job B, group)
  O->>D: Status
  D-->>O: job A running, job B saved
  O->>D: Snapshot(job A)
  D->>D: CUDA lock and checkpoint, every process of A's pods at once
  O->>D: GetOperation, until complete
  O->>D: Restore(job B)
  D->>D: CUDA restore and unlock
  O-->>J: granted
{% end %}

- `Status`: jobs with a pod on the node (label `timeslice.io/job-id`), their state and per-GPU
  memory. Stateless; asks the driver and NVML each time.
- `Snapshot`, `Restore`: start an operation on the job's CUDA processes (`cuda` backend only).
- `GetOperation`: pending, complete or failed, and duration.
- `grpc.health.v1.Health`: serving.

The orchestrator dials `<node name>:9001`. Where node names do not resolve,
[`experiments/timeslice/run.sh`](https://github.com/wseaton/snapshotd/blob/main/experiments/timeslice/run.sh)
creates a headless Service per node.

Without a DRA driver, `run.sh verl` has the two trainer pods open the GPU through the runtime's
device mount instead of a shared `ResourceClaim`. That is its only change to the project's
example. Results: [benchmarks](@/benchmarks.md#time-slicing).
