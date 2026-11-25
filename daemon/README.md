# snapshotd

The node agent for restore on scale-up, as a Cargo workspace of two crates:

- `snapshotd` (this directory). `snapshotd node` is the daemon: it finds opted-in pods through CRI,
  checkpoints the first of a revision, restores the rest, and pulls a checkpoint from another
  node's daemon when its own node has none. It can also serve llm-d-rl-time-slicing's node agent
  API, pausing a job's CUDA processes in place and resuming them. How it fits into a cluster is in
  [Running the daemon](https://wseaton.github.io/snapshotd/running/); manifests are in [`deploy/`](deploy). The other
  subcommands, below, are the daemon's building blocks and are what
  [`experiments/bench`](../experiments/bench) calls.
- `snapshotd-shim` ([`shim/`](shim)). PID 1 of the workload container. It is built static, against
  musl, so it runs in whatever image the workload uses.

Every option of `snapshotd node` can also be set through the environment, as `SNAPSHOTD_` and the
option's name: `--boot-timeout-s` is `SNAPSHOTD_BOOT_TIMEOUT_S`. `snapshotd node --help` lists them.

With `--topology-levels`, a list of node labels as in a Kueue `Topology`, a daemon that has to
pull a checkpoint takes it from the nearest node that holds it.

Linux only. `snapshotd` loads `libcuda.so.1` at run time, so it builds without a GPU or the CUDA
toolkit and uses whichever driver library is first on the library path where it runs. Moving
checkpoints between nodes needs the `p2p` feature and NIXL installed.

```bash
cargo build --release --workspace
cargo test --workspace
e2e/run.sh        # the daemon and shim on kind, with a stand-in for vLLM
e2e/peers.sh      # checkpoints between nodes: three kind workers, NIXL, kube-rbac-proxy
```

[`images/snapshotd`](../images/snapshotd) builds both on Fedora with the toolchain from rustup,
and has the kind tests' images as its `e2e` and `e2e-peers` targets.

## CUDA checkpoint

```bash
snapshotd cuda pids <root-pid>                 # CUDA processes in a process tree
snapshotd cuda state <pid>...
snapshotd cuda checkpoint <pid>...             # lock all, then checkpoint all
snapshotd cuda restore <pid>...                # restore all, then unlock all
snapshotd cuda restore --source GPU-a,GPU-b --target GPU-c,GPU-d --visible <every GPU> <pid>...
```

Every process in a phase is handled in parallel. With `--source` and `--target`, rank i moves from
`source[i]` to `target[i]`; `--visible` lists every GPU the processes can see, because the 580
driver wants a one-to-one map over all of them.

## cuInterpose

For workloads that still share GPU memory between processes at checkpoint time. The bundle
directory holds `libcuinterpose.so`, `libcuinterpose_core.so` and `cuinterpose-coordinator` from
Dynamo Snapshot.

```bash
snapshotd cuinterpose inspect <namespace-pid>...
snapshotd cuinterpose prepare --bundle-dir DIR --checkpoint-dir DIR <pid>...   # before cuda checkpoint
snapshotd cuinterpose restore --bundle-dir DIR --checkpoint-dir DIR            # after cuda restore
```

`prepare` does nothing when no process shares memory. Otherwise it records the hashes of the
libraries the processes loaded and runs the coordinator's prepare phase; `restore` refuses a bundle
whose hashes differ.

## Checkpoints in memory

```bash
snapshotd image hold --images-dir DIR [--reserve-gib N]
```

The other end of `criu dump --stream` and `criu restore --stream`: it takes one dump into
anonymous 2 MiB-aligned huge-page segments and answers restores from them until killed. See
[Internals](https://wseaton.github.io/snapshotd/internals/#huge-page-capture).

## Copy to another node

Built with `--features p2p`, which needs NIXL and UCX installed and links
[ai-dynamo/nixl#2672](https://github.com/ai-dynamo/nixl/pull/2672) for caller-provided buffers.

```bash
snapshotd transfer receive --dir DIR [--direct | --mmap] [--hold] [--trace FILE]   # on the target
snapshotd transfer send --dir DIR --to HOST:7700 [--trace FILE]                    # files on disk
snapshotd transfer stream --images-dir DIR --to HOST:7700 [--trace FILE]           # criu dump --stream
```

Each side registers one huge-page pool with NIXL and moves chunks slot to slot over UCX. `stream`
sends while CRIU dumps and writes nothing on the source. On the target, `--direct` writes with
`O_DIRECT`, `--mmap` copies into RAM-backed files in parallel, and `--hold` keeps the image in
memory and serves the restore itself. `--trace` writes a per-chunk timeline for
`experiments/bench/trace-merge.py`. Results are in
[Benchmarks](https://wseaton.github.io/snapshotd/benchmarks/).
