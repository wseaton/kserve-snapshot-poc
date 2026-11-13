# snapshotd

The node path of the Rust snapshot daemon described in [DESIGN.md](../DESIGN.md), as far as it is
built: a CLI that checkpoints and restores the CUDA state of a process tree through the driver
API, drives cuInterpose around it, holds a CRIU checkpoint in memory, and copies one to another
node over RDMA. The control loop, the CRI lookup, the CRIU invocation and
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

## Checkpoints in memory

```bash
snapshotd image hold --images-dir DIR [--reserve-gib N]
```

The other end of `criu dump --stream` and `criu restore --stream`: it takes one dump into
anonymous 2 MiB-aligned huge-page segments and answers restores from them until killed. See
[experiments/memory-capture.md](../experiments/memory-capture.md).

## Copy to another node

Built with `--features nixl`, which needs NIXL and UCX installed and links
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
[experiments/p2p-transfer.md](../experiments/p2p-transfer.md).
