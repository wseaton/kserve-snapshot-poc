# Capturing into huge-page memory: notes

An attempt, on 2026-11-13, to keep the checkpoint off disk: CRIU dumps into anonymous huge-page
memory, that memory is registered for RDMA and sent to the peer, and the peer restores from
memory. It works, and the dump ends up about twice as fast as local disk, but it is parked in
favour of local NVMe plus an RDMA copy until the open items below are dealt with. The code is in
[`daemon/src/image.rs`](../daemon/src/image.rs) (`snapshotd image hold`) and the bench harness has
`STREAM=1` for it.

## How it works

`criu dump --stream` writes no image files. It connects to `streamer-capture.sock` in the images
directory and, for each image file, sends the file name and the read end of a pipe.
`snapshotd image hold` is the other end: it reads every pipe into an arena of anonymous 2 MiB-aligned
segments mapped with `MADV_HUGEPAGE`, the same mapping the NIXL registration benchmark uses to get
transparent huge pages on CoreWeave nodes. `criu restore --stream` asks `streamer-serve.sock` for
files by name and gets them written back into pipes. No `criu-image-streamer` binary is involved;
the protocol is two small protobuf messages and one passed descriptor per file.

## What was measured

All on one 8x H200 node, cuInterpose configuration, restored onto rotated GPUs.

Qwen3-8B, TP=2, 11.8 GB image:

| | CRIU dump | CRIU restore |
| --- | --- | --- |
| Local disk | 5.7 s | 2.2 s |
| Memory, pipes at the default 1 MiB | 30.1 s | 5.4 s |
| Memory, pipes at 64 MiB | 6.6 s | 5.5 s |

DeepSeek-V4.1-Flash, TP=8 with expert parallel, about 70 GB image:

| | CRIU dump | CRIU restore | Huge pages |
| --- | --- | --- | --- |
| Local disk | 37.0 s | 5.1 s | |
| Memory, segments faulted during the dump | 692.9 s | 11.3 s | 79% |
| Memory, 76 GiB faulted in before the dump | 16.9 s | 9.8 s | 100% |

Faulting the 76 GiB in beforehand took 19.4 s with one thread per 4 GiB segment.

## What went wrong, and the fixes

- **Pipe size.** With 1 MiB pipes every 4 MiB chunk CRIU spliced took a steady 10 ms, about
  400 MB/s. Raising each pipe to 64 MiB with `F_SETPIPE_SZ` removed that. It needs
  `CAP_SYS_RESOURCE`; the node's unprivileged limit is 1 MiB.
- **Huge pages allocated during the dump.** The node had 68 to 185 GB free and 1.8 TB of page
  cache. Each first touch of a 2 MiB page waited for the kernel to compact memory: about 75,000
  compaction stalls and 34,000 reclaim stalls over one dump, and each worker took 40 to 140 s
  instead of 4 s. At TP=2 the 12 GB of huge pages were easy to find and nothing showed.
  Faulting the pool in before the dump, in parallel, moves that cost out of the dump and made it
  small. The pool has to exist before the capture starts; a daemon would allocate and register it
  once and reuse it.
- **First touch against rewrite.** Writing into memory that is already faulted in ran at 30 GB/s
  on this node, first touch of huge pages at 4.6 GB/s and of 4 KiB pages at 2.3 GB/s.

## Open

- **Restore from memory is slower than from disk** (9.8 s against 5.1 s at TP=8). From files, CRIU
  restores each process's memory with bulk `preadv` calls inside that process. A stream cannot be
  seeked, so CRIU reads page ranges one after another in one place. Passing pages into the pipe
  without a copy (`vmsplice`) was not tried, and may not help if the limit is CRIU's read path.
- **The dump is still serial.** CRIU dumps one process at a time and moves its memory in 4 MiB
  chunks. 70 GB in 16.9 s is 4.1 GB/s. Going well past that means dumping less (each DeepSeek
  worker still holds 5 GiB of GPU state after prepare, about 40 GB of the image) or dumping
  processes in parallel, which stock CRIU does not do.
- **The RDMA leg from this memory is not written.** `snapshotd transfer` copies files through a
  registered huge-page pool; registering the arena's segments and sending them directly is the
  next step for this path.
- **Pool pre-fault time depends on the node's memory state.** 19.4 s was measured right after a
  run that had already forced the kernel to compact. Dropping the page cache or triggering
  compaction would make it cheap, and both are node-wide, so neither was done on a shared node.
- **Memory cost.** The image needs its size in host memory on both nodes for as long as it is
  held, counted against whichever pod runs `snapshotd`.
- **Host files.** cuInterpose's state and the daemon's manifest are still regular files in the
  images directory and would need to travel with the stream.
