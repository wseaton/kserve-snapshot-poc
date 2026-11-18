#!/usr/bin/env python3
"""Stand in for `criu dump --stream`, and check what a receiver wrote.

    stream-check.py dump <images dir> [--large-gib N] [--seed S]
    stream-check.py verify <received dir> [--large-gib N] [--seed S]
    stream-check.py hold [--gib N] [--procs P] [--huge]
    stream-check.py pages <received dir> [--gib N] [--procs P]

`dump` connects to streamer-capture.sock in the images directory, where
`snapshotd transfer stream` or `snapshotd image hold` listens, and sends the
image files of a made-up dump the way CRIU does: a request naming the file,
the read end of a pipe, then the bytes. A few large files are written with
small ones opened and closed in between, so several pipes are open at once.

Every file is a run of 4 MiB chunks. Each chunk starts with a header naming
its file and its index, so `verify` can tell a chunk that is missing from one
that holds another chunk's bytes, and say whose.

`hold` is a process for a real `criu dump --stream` to checkpoint: P processes,
each holding N GiB of anonymous memory in which every 4 KiB page repeats its
process number and its page number. It prints `HOLDING <pid>` and sleeps.
`pages` reads the pages-*.img files of the received image and checks that
every stamped page is whole and that each one arrived exactly once.
"""

import argparse
import array
import glob
import mmap
import os
import random
import socket
import struct
import sys
import time

CHUNK = 4 << 20
MAGIC = b"SNAPCHK!"
HEADER = struct.Struct("<8sQQ")
SLOT = 64 << 20


def plan(seed, large_gib):
    """The files of the dump, in the order they are opened: (name, size, parent).

    A file with a parent is written while the parent's pipe is still open.
    """
    rng = random.Random(seed)
    large = [
        32 * SLOT,
        3 * (1 << 30) + 12345,
        (large_gib << 30) + 1,
        (large_gib << 30) - 4096,
        SLOT,
        SLOT - 1,
        SLOT + 1,
    ]
    files = []
    small = 0
    for index, size in enumerate(large):
        parent = len(files)
        files.append((f"pages-{index + 1}.img", size, None))
        for _ in range(140):
            files.append((f"core-{small}.img", rng.randint(1, 200_000), parent))
            small += 1
        for _ in range(6):
            files.append((f"pagemap-shmem-{small}.img", rng.randint(1 << 20, 50 << 20), parent))
            small += 1
    files.append(("inventory.img", 0, None))
    files.append(("stats-dump", 37, None))
    return files


def base_block(seed):
    return random.Random(seed ^ 0x5EED).randbytes(CHUNK)


def chunk_bytes(base, file, index, length):
    return (HEADER.pack(MAGIC, file, index) + base[HEADER.size :])[:length]


def open_file(sock, name):
    payload = b"\x0a" + bytes([len(name)]) + name.encode()
    sock.sendall(struct.pack("<I", len(payload)) + payload)
    read_end, write_end = os.pipe()
    sock.sendmsg([b"*"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [read_end]))])
    os.close(read_end)
    return write_end


def write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def write_file(base, fd, file, size, between=None):
    index = 0
    while index * CHUNK < size:
        write_all(fd, chunk_bytes(base, file, index, min(CHUNK, size - index * CHUNK)))
        index += 1
        if between:
            between(index)
    os.close(fd)


def dump(args):
    files = plan(args.seed, args.large_gib)
    base = base_block(args.seed)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(os.path.join(args.dir, "streamer-capture.sock"))
    start = time.time()
    children = {}
    for file, (_, _, parent) in enumerate(files):
        if parent is not None:
            children.setdefault(parent, []).append(file)
    for file, (name, size, parent) in enumerate(files):
        if parent is not None:
            continue
        pending = children.get(file, [])
        chunks = max(1, -(-size // CHUNK))
        every = max(1, chunks // (len(pending) + 1))

        def between(index, pending=pending, every=every):
            if pending and index % every == 0:
                child = pending.pop(0)
                child_name, child_size, _ = files[child]
                write_file(base, open_file(sock, child_name), child, child_size)

        write_file(base, open_file(sock, name), file, size, between)
        for child in pending:
            child_name, child_size, _ = files[child]
            write_file(base, open_file(sock, child_name), child, child_size)
    sock.close()
    total = sum(size for _, size, _ in files)
    seconds = time.time() - start
    print(f"DUMP {len(files)} files, {total / 1e9:.1f} GB in {seconds:.1f} s, {total / 1e9 / seconds:.2f} GB/s")


def describe(found, files):
    if not found.strip(b"\0"):
        return "zeros"
    if found[:8] == MAGIC and len(found) >= HEADER.size:
        _, file, index = HEADER.unpack(found[: HEADER.size])
        name = files[file][0] if file < len(files) else f"file {file}"
        return f"chunk {index} of {name}"
    return "other bytes"


def verify(args):
    files = plan(args.seed, args.large_gib)
    base = base_block(args.seed)
    bad_files = 0
    bad_chunks = 0
    checked = 0
    for file, (name, size, _) in enumerate(files):
        path = os.path.join(args.dir, name)
        try:
            actual = os.path.getsize(path)
        except OSError:
            print(f"MISSING {name}")
            bad_files += 1
            continue
        problems = []
        if actual != size:
            problems.append(f"size {actual}, want {size}")
        with open(path, "rb") as handle:
            index = 0
            while index * CHUNK < min(size, actual):
                want = chunk_bytes(base, file, index, min(CHUNK, size - index * CHUNK))
                found = handle.read(len(want))
                checked += len(found)
                if found != want:
                    bad_chunks += 1
                    if len(problems) < 8:
                        first = next(
                            (i for i in range(0, len(want), 4096) if found[i : i + 4096] != want[i : i + 4096]),
                            0,
                        )
                        problems.append(
                            f"chunk {index} (offset {index * CHUNK}): holds {describe(found, files)}, "
                            f"first wrong page at +{first}, slot offset {(index * CHUNK) % SLOT}"
                        )
                index += 1
        if problems:
            bad_files += 1
            print(f"BAD {name}: " + "; ".join(problems))
    print(f"VERIFY {len(files)} files, {checked / 1e9:.1f} GB read, {bad_files} bad files, {bad_chunks} bad chunks")
    return 1 if bad_files else 0


PAGE = 4096
STAMP = struct.Struct("<4sHHQ")
STAMP_MAGIC = b"PGCK"


def stamp(proc, page):
    return STAMP.pack(STAMP_MAGIC, proc, 0, page) * (PAGE // STAMP.size)


def fill(proc, pages, huge):
    region = mmap.mmap(-1, pages * PAGE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if huge:
        region.madvise(mmap.MADV_HUGEPAGE)
    for page in range(pages):
        region[page * PAGE : (page + 1) * PAGE] = stamp(proc, page)
    return region


def hold(args):
    pages = (args.gib << 30) // PAGE
    filled, done = os.pipe()
    for proc in range(1, args.procs):
        if os.fork() == 0:
            os.close(filled)
            region = fill(proc, pages, args.huge)
            os.write(done, b".")
            os.close(done)
            while region:
                time.sleep(3600)
    os.close(done)
    region = fill(0, pages, args.huge)
    for _ in range(1, args.procs):
        os.read(filled, 1)
    os.close(filled)
    print(f"HOLDING {os.getpid()}", flush=True)
    while region:
        time.sleep(3600)


def pages(args):
    want = (args.gib << 30) // PAGE
    seen = [bytearray(want) for _ in range(args.procs)]
    torn = 0
    strangers = 0
    examples = []
    total = 0
    for path in sorted(glob.glob(os.path.join(args.dir, "pages-*.img"))):
        with open(path, "rb") as handle:
            offset = 0
            while True:
                block = handle.read(PAGE * 1024)
                if not block:
                    break
                for start in range(0, len(block), PAGE):
                    page = block[start : start + PAGE]
                    if page[:4] != STAMP_MAGIC:
                        continue
                    total += 1
                    _, proc, _, index = STAMP.unpack(page[: STAMP.size])
                    if proc >= args.procs or index >= want:
                        strangers += 1
                    elif page != stamp(proc, index):
                        torn += 1
                        if len(examples) < 8:
                            first = next(i for i in range(0, PAGE, STAMP.size) if page[i : i + STAMP.size] != page[: STAMP.size])
                            examples.append(
                                f"{os.path.basename(path)} +{offset + start}: page {index} of process {proc} "
                                f"changes at byte {first} to {page[first : first + STAMP.size].hex()}"
                            )
                    else:
                        seen[proc][index] += 1
                offset += len(block)
    missing = sum(row.count(0) for row in seen)
    repeated = sum(len(row) - row.count(0) - row.count(1) for row in seen)
    for line in examples:
        print("TORN " + line)
    print(
        f"PAGES {total} stamped pages read, want {want * args.procs}: "
        f"{missing} missing, {repeated} repeated, {torn} torn, {strangers} with a bad header"
    )
    return 1 if missing or repeated or torn or strangers else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dump", "verify", "hold", "pages"])
    parser.add_argument("dir", nargs="?")
    parser.add_argument("--large-gib", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gib", type=int, default=4)
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--huge", action="store_true", help="hold: back the memory with transparent huge pages")
    args = parser.parse_args()
    sys.exit({"dump": dump, "verify": verify, "hold": hold, "pages": pages}[args.mode](args))


if __name__ == "__main__":
    main()
