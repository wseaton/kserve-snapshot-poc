#!/usr/bin/env python3
"""Merge transfer traces and harness phases into one timeline, and summarize it.

  trace-merge.py -o merged.json [--phases phases.jsonl ...] sender.json receiver.json

The inputs are what `snapshotd transfer ... --trace FILE` writes on each node and the
phases.jsonl that restore-bench.sh leaves in /work/<tag>. The output opens in
https://ui.perfetto.dev: the harness phases of each node (cold boot, prepare, checkpoint,
dump, restore) and the transfer as one span, then one process for each side of the transfer
with a track for each slot of the registered pool. Timestamps are wall-clock, so the
nodes line up as well as their clocks agree.
"""
import argparse
import collections
import json
import sys

FILE_TRACKS = 1_000_000


def load_trace(path, pid):
    with open(path) as handle:
        trace = json.load(handle)
    process = trace.get("process", path)
    events = trace["traceEvents"]
    out = [{"name": "process_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": process}}]
    tracks = sorted({e["tid"] for e in events if e["ph"] == "X"})
    for tid in tracks:
        name = f"file {tid - FILE_TRACKS}" if tid >= FILE_TRACKS else f"slot {tid}"
        out.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": name}})
    for event in events:
        event["pid"] = pid
        out.append(event)
    return process, events, out


def load_phases(path, pid, tid):
    """One track for each node's phases, in the order the nodes first appear, with the
    phases that span others (restore_total) on a row of their own."""
    out = []
    with open(path) as handle:
        phases = [json.loads(line) for line in handle if line.strip()]
    nodes = []
    for phase in phases:
        node = phase.get("node", path)
        if node not in nodes:
            nodes.append(node)
            base = tid + 2 * (len(nodes) - 1)
            out.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": base, "args": {"name": f"phases on {node}"}})
            out.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": base + 1, "args": {"name": f"totals on {node}"}})
        base = tid + 2 * nodes.index(node)
        out.append({
            "name": phase["name"], "ph": "X", "pid": pid,
            "tid": base + (1 if phase["name"].endswith("_total") else 0),
            "ts": phase["start_us"], "dur": phase["end_us"] - phase["start_us"], "args": {},
        })
    return out


def summarize(process, events):
    spans = [e for e in events if e["ph"] == "X"]
    if not spans:
        return
    begin = min(e["ts"] for e in spans)
    end = max(e["ts"] + e["dur"] for e in spans)
    wall = (end - begin) / 1e6
    slots = len({e["tid"] for e in spans if e["tid"] < FILE_TRACKS})
    print(f"\n{process}: {wall:.2f} s wall, {slots} slots")
    print(f"  {'stage':<26}{'spans':>7}{'GB':>9}{'slots busy':>12}{'GB/s per slot':>15}{'GB/s overall':>14}")
    by_name = collections.defaultdict(list)
    for event in spans:
        by_name[event["name"]].append(event)
    for name, group in sorted(by_name.items(), key=lambda item: min(e["ts"] for e in item[1])):
        busy = sum(e["dur"] for e in group) / 1e6
        gigabytes = sum(e["args"].get("bytes", 0) for e in group) / 1e9
        per_slot = f"{gigabytes / busy:.2f}" if gigabytes and busy else ""
        overall = f"{gigabytes / wall:.2f}" if gigabytes else ""
        # "slots busy" is the average number of slots (or files) inside this stage at once.
        print(f"  {name:<26}{len(group):>7}{gigabytes:>9.2f}{busy / wall:>12.2f}{per_slot:>15}{overall:>14}")
    counters = [e for e in events if e["ph"] == "C"]
    if counters:
        name = counters[0]["name"]
        series = sorted((e["ts"], e["args"]["value"]) for e in counters if e["name"] == name)
        print(f"  {name}, GB/s in each second:")
        buckets = collections.defaultdict(int)
        previous = 0
        for ts, value in series:
            buckets[int((ts - begin) / 1e6)] += value - previous
            previous = value
        line = " ".join(f"{buckets[second] / 1e9:.1f}" for second in range(int(wall) + 1))
        print(f"    {line}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--phases", action="append", default=[])
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    merged = [{"name": "process_name", "ph": "M", "pid": 1, "tid": 0, "args": {"name": "harness"}}]
    for index, path in enumerate(args.phases):
        merged += load_phases(path, 1, index * 10)
    traces = []
    for index, path in enumerate(args.traces):
        process, events, out = load_trace(path, index + 2)
        merged += out
        traces.append((process, events))
        summarize(process, events)
    spans = [e for _, events in traces for e in events if e["ph"] == "X"]
    if spans:
        begin = min(e["ts"] for e in spans)
        end = max(e["ts"] + e["dur"] for e in spans)
        gigabytes = max(
            sum(e["args"].get("bytes", 0) for e in events if e["ph"] == "X" and e["name"] == "rdma write")
            for _, events in traces
        ) / 1e9
        merged.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": 900, "args": {"name": "transfer"}})
        merged.append({
            "name": "rdma transfer", "ph": "X", "pid": 1, "tid": 900, "ts": begin, "dur": end - begin,
            "args": {"gb": round(gigabytes, 2), "gb_per_s": round(gigabytes / ((end - begin) / 1e6), 2)},
        })
    with open(args.output, "w") as handle:
        json.dump({"traceEvents": merged, "displayTimeUnit": "ms"}, handle)
    print(f"\n{args.output}: {len(merged)} events; open it at https://ui.perfetto.dev", file=sys.stderr)


if __name__ == "__main__":
    main()
