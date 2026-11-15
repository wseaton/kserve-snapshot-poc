#!/usr/bin/env python3
"""Say what a checkpoint of a process tree would contain, before taking it.

  mem-breakdown.py <root pid> [--top N]

Run it in the bench pod after /checkpoint/prepare. For every process under the root it reads
/proc/<pid>/smaps and sums the memory CRIU writes into the image: anonymous pages, dirty pages of
private file mappings, and shared memory (counted once, by Pss). It also prints what each process
still holds on the GPUs, which the CUDA checkpoint moves into anonymous host memory before the
dump.
"""
import argparse
import collections
import os
import re
import subprocess

HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) (\S+) \S+ \S+ \S+\s*(.*)$")


def tree(pid):
    yield pid
    for task in os.listdir(f"/proc/{pid}/task"):
        try:
            with open(f"/proc/{pid}/task/{task}/children") as handle:
                for child in handle.read().split():
                    yield from tree(int(child))
        except FileNotFoundError:
            pass


def kind(name):
    if name.startswith("/dev/shm/") or name.startswith("/memfd:") or name.startswith("/SYSV"):
        return "shared memory"
    if name.startswith("/dev/nvidia"):
        return "nvidia device mapping"
    if name == "[heap]":
        return "heap"
    if name.startswith("[stack") or name in ("[vdso]", "[vvar]", "[vsyscall]"):
        return "stack and kernel"
    if not name or name.startswith("[anon"):
        return "anonymous mmap"
    if ".so" in os.path.basename(name):
        return "library data, " + os.path.basename(name).split(".so")[0]
    return "file, " + name


def read(pid):
    """Bytes CRIU would dump for one process, by kind of mapping, and its largest mappings."""
    by_kind, mappings = collections.Counter(), []
    current = None
    with open(f"/proc/{pid}/smaps") as handle:
        for line in handle:
            match = HEADER.match(line)
            if match:
                current = {"kind": kind(match.group(4)), "name": match.group(4), "shared": "s" in match.group(3), "f": {}}
                mappings.append(current)
            elif current is not None:
                key, _, rest = line.partition(":")
                if rest.strip().endswith("kB"):
                    current["f"][key] = int(rest.split()[0]) * 1024
    sized = []
    for m in mappings:
        f = m["f"]
        if m["kind"] == "shared memory" or m["shared"]:
            size = f.get("Pss", 0)
        else:
            size = f.get("Anonymous", 0) + f.get("Swap", 0)
        by_kind[m["kind"]] += size
        sized.append((size, m["kind"], m["name"]))
    return by_kind, sorted(sized, reverse=True)


def gpu_by_pid():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout
    used = collections.Counter()
    for line in out.splitlines():
        pid, _, mib = line.partition(",")
        if mib.strip().isdigit():
            used[int(pid)] += int(mib) << 20
    return used


def gb(n):
    return f"{n / 1e9:8.2f} GB"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=int)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    gpu = gpu_by_pid()
    total, host_total, gpu_total = collections.Counter(), 0, 0
    print(f"{'pid':>8}  {'host':>11}  {'gpu':>11}  command")
    largest = None
    for pid in tree(args.root):
        try:
            by_kind, sized = read(pid)
            with open(f"/proc/{pid}/cmdline") as handle:
                command = handle.read().replace("\0", " ")[:60]
        except (FileNotFoundError, PermissionError):
            continue
        host = sum(by_kind.values())
        total.update(by_kind)
        host_total += host
        gpu_total += gpu.get(pid, 0)
        print(f"{pid:>8}  {gb(host)}  {gb(gpu.get(pid, 0))}  {command}")
        if largest is None or host > largest[0]:
            largest = (host, pid, sized)
    print(f"\nhost memory {gb(host_total)}, GPU memory {gb(gpu_total)}, expected image {gb(host_total + gpu_total)}")
    print("\nhost memory by kind of mapping, all processes:")
    for name, size in total.most_common(args.top * 2):
        print(f"  {gb(size)}  {name}")
    if largest:
        print(f"\nlargest mappings of pid {largest[1]}:")
        for size, name, path in largest[2][: args.top]:
            print(f"  {gb(size)}  {name}  {path}")


if __name__ == "__main__":
    main()
