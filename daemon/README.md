# snapshotd

The node path of the Rust snapshot daemon described in [DESIGN.md](../DESIGN.md), as far as it is
built: a CLI that checkpoints and restores the CUDA state of a process tree through the driver
API, and drives cuInterpose around it. The control loop, the CRI lookup, the CRIU invocation and
the cross-pod mount handling are design only; [`experiments/bench`](../experiments/bench) runs
CRIU from a script and calls this binary for the CUDA phases.

Linux only. It loads `libcuda.so.1` at run time, so it builds without a GPU or the CUDA toolkit
and uses whichever driver library is first on the library path where it runs.

```bash
cargo build --release
cargo test
```

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
