#!/usr/bin/env python3
"""Draw one lifecycle.sh run as a page that reads on a phone.

  trace-render.py <lifecycle output dir> [-o page.html] [--png page.png] [--title TEXT]

Reads what lifecycle.sh leaves in its output directory: phases.jsonl, the transfer's
STREAM or TRANSFER line, and the receiver's trace for throughput. Writes a standalone HTML
page; with --png it also renders that page with a headless Chrome, if one is installed.
"""
import argparse
import collections
import glob
import html
import json
import os
import shutil
import subprocess

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
W, PAD = 640, 0
NAMES = {
    "prepare": "prepare",
    "cuinterpose_prepare": "cuInterpose prepare",
    "cuda_checkpoint": "CUDA checkpoint",
    "criu_dump": "CRIU dump",
    "criu_restore": "CRIU restore",
    "cuda_restore": "CUDA restore",
    "cuinterpose_restore": "cuInterpose restore",
    "vllm_restore": "vLLM restore",
}
CHROMES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
]


class Svg:
    def __init__(self):
        self.parts, self.y = [], 0

    def text(self, x, y, value, cls="", anchor="start"):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{html.escape(value)}</text>')

    def bar(self, x, y, width, height, color):
        width = max(width, 1.5)
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height}" rx="{min(4, width / 2):.1f}" fill="{color}"/>')

    def axis(self, index, top, bottom, seconds):
        """Gridlines go in at `index`, under everything the panel drew after it."""
        step = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200) if seconds / s <= 7)
        lines, tick = [], 0
        while tick <= seconds:
            x = W * tick / seconds
            lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{bottom}" class="grid"/>')
            anchor = "start" if tick == 0 else "end" if x > W - 12 else "middle"
            self.text(x, bottom + 15, f"{tick} s", "muted", anchor)
            tick += step
        self.parts[index:index] = lines

    def segments(self, title, phases, color, origin, seconds):
        """One row of back-to-back phases; labels that do not fit go on the line below."""
        self.text(0, self.y + 12, title, "label")
        top = self.y + 20
        below = []
        for name, start, end in phases:
            x = W * (start - origin) / 1e6 / seconds
            width = W * (end - start) / 1e6 / seconds - 2
            label = f"{NAMES.get(name, name)} {(end - start) / 1e6:.1f} s"
            self.bar(x, top, width, 24, color)
            if width > 7.0 * len(label) + 14:
                self.text(x + 8, top + 16, label, "onbar")
            else:
                below.append(label)
        if below:
            self.text(0, top + 40, " · ".join(below), "muted")
        self.y = top + (58 if below else 36)


def chrome():
    for candidate in CHROMES:
        path = candidate if os.path.exists(candidate) else shutil.which(candidate)
        if path:
            return path
    return None


def first_json(path, prefix):
    try:
        with open(path) as handle:
            for line in handle:
                if line.startswith(prefix):
                    return json.loads(line[len(prefix):])
    except FileNotFoundError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("-o", "--output")
    parser.add_argument("--png")
    parser.add_argument("--title", default="Checkpoint on one node, restore on another")
    parser.add_argument("--subtitle", default="")
    args = parser.parse_args()
    d = args.directory
    output = args.output or os.path.join(d, "lifecycle.html")

    with open(os.path.join(d, "phases.jsonl")) as handle:
        phases = [json.loads(line) for line in handle if line.strip()]
    nodes = list(collections.OrderedDict.fromkeys(p["node"] for p in phases))
    by_node = {n: {p["name"]: (p["start_us"], p["end_us"]) for p in phases if p["node"] == n} for n in nodes}
    source, target = by_node[nodes[0]], by_node[nodes[-1]]
    capture = [(n, *source[n]) for n in ("prepare", "cuinterpose_prepare", "cuda_checkpoint", "criu_dump") if n in source]
    restore = [(n, *target[n]) for n in ("criu_restore", "cuda_restore", "cuinterpose_restore", "vllm_restore") if n in target]

    stream = first_json(os.path.join(d, "stream.txt"), "STREAM ")
    copy = first_json(os.path.join(d, "transfer.txt"), "TRANSFER ")
    counters = []
    for path in sorted(glob.glob(os.path.join(d, "receive-*.json"))):
        with open(path) as handle:
            events = json.load(handle)["traceEvents"]
        series = sorted((e["ts"], e["args"]["value"]) for e in events if e["ph"] == "C")
        if len(series) > len(counters):
            counters = series

    svg = Svg()
    origin = capture[0][1]
    capture_end = capture[-1][2]
    transfer_end = counters[-1][0] if counters else capture_end
    overlapped = stream is not None
    if not overlapped and copy:
        transfer_end = capture_end + int(copy["seconds"] * 1e6)
    total = (max(capture_end, transfer_end) - origin) / 1e6
    seconds = total * 1.04

    svg.text(0, svg.y + 18, "Capture", "h2")
    svg.text(W, svg.y + 18, f"{total:.1f} s until the image is on the second node", "muted", "end")
    svg.y += 34
    top, index = svg.y, len(svg.parts)
    svg.segments(f"Source node ({nodes[0]})", capture, BLUE, origin, seconds)
    moved = stream or copy
    if moved:
        start = counters[0][0] if overlapped and counters else capture_end
        end = transfer_end
        svg.text(0, svg.y + 12, "RDMA to the second node" + (", while the dump runs" if overlapped else ", after the dump"), "label")
        x, width = W * (start - origin) / 1e6 / seconds, W * (end - start) / 1e6 / seconds
        svg.bar(x, svg.y + 20, width, 24, ORANGE)
        label = f"{moved['gb']:.0f} GB at {moved['mb_per_s'] / 1000:.2f} GB/s"
        if width > 7.0 * len(label) + 14:
            svg.text(x + 8, svg.y + 36, label, "onbar")
        else:
            svg.text(0, svg.y + 60, label, "muted")
            svg.y += 14
        svg.y += 56
        if overlapped and len(counters) > 4:
            bucket = max((end - start) / 1e6 / 60, 0.25)
            n = int((end - start) / 1e6 / bucket) + 1
            sums, previous = [0] * n, 0
            for ts, value in counters:
                sums[min(int((ts - start) / 1e6 / bucket), n - 1)] += value - previous
                previous = value
            rates = [v / 1e9 / bucket for v in sums][:-1]
            peak = max(rates)
            svg.text(0, svg.y + 12, f"Received on the second node, GB/s (peak {peak:.1f})", "label")
            base, height = svg.y + 58, 34
            points = " ".join(
                f"{W * ((start - origin) / 1e6 + (i + 0.5) * bucket) / seconds:.1f},{base - height * r / peak:.1f}"
                for i, r in enumerate(rates)
            )
            first = W * ((start - origin) / 1e6 + 0.5 * bucket) / seconds
            last = W * ((start - origin) / 1e6 + (len(rates) - 0.5) * bucket) / seconds
            svg.parts.append(f'<polygon points="{first:.1f},{base} {points} {last:.1f},{base}" fill="{ORANGE}" opacity="0.12"/>')
            svg.parts.append(f'<polyline points="{points}" fill="none" stroke="{ORANGE}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            svg.y = base + 8
    svg.axis(index, top - 4, svg.y, seconds)
    svg.y += 44

    restore_total = (restore[-1][2] - restore[0][1]) / 1e6
    seconds = restore_total * 1.04
    svg.text(0, svg.y + 18, "Restore", "h2")
    svg.text(W, svg.y + 18, f"{restore_total:.1f} s to a serving replica", "muted", "end")
    svg.y += 34
    top, index = svg.y, len(svg.parts)
    svg.segments(f"Second node ({nodes[-1]})", restore, AQUA, restore[0][1], seconds)
    svg.axis(index, top - 4, svg.y, seconds)
    svg.y += 44

    if "cold_boot" in source:
        cold = (source["cold_boot"][1] - source["cold_boot"][0]) / 1e6
        scale = max(cold, total, restore_total) * 1.02
        svg.text(0, svg.y + 18, "Against a cold boot", "h2")
        svg.y += 34
        for label, value, color in (("Cold boot", cold, GRAY), ("Capture, to the second node", total, BLUE), ("Restore", restore_total, AQUA)):
            svg.text(0, svg.y + 12, label, "label")
            svg.text(W, svg.y + 12, f"{value:.1f} s", "value", "end")
            svg.bar(0, svg.y + 20, W * value / scale, 18, color)
            svg.y += 50

    body = f'<svg viewBox="0 0 {W} {svg.y}" role="img" aria-label="{html.escape(args.title)}">{"".join(svg.parts)}</svg>'
    page = f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(args.title)}</title>
<style>
:root {{ --bg: #fcfcfb; --fg: #0b0b0b; --muted: #6f6e69; --grid: #e6e5e0; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #1a1a19; --fg: #f0efec; --muted: #a3a29b; --grid: #33332f; }} }}
body {{ margin: 0; background: var(--bg); color: var(--fg); font-family: -apple-system, system-ui, "Segoe UI", sans-serif; }}
main {{ max-width: 680px; margin: 0 auto; padding: 24px 16px 32px; }}
h1 {{ font-size: 22px; margin: 0 0 6px; letter-spacing: -0.01em; }}
p {{ font-size: 15px; color: var(--muted); margin: 0 0 24px; line-height: 1.45; }}
svg {{ width: 100%; height: auto; display: block; }}
text {{ font-size: 13px; fill: var(--fg); }}
.muted {{ fill: var(--muted); }} .label {{ fill: var(--muted); font-size: 12.5px; }}
.h2 {{ font-size: 17px; font-weight: 650; }} .value {{ font-weight: 650; }}
.onbar {{ fill: #fff; font-weight: 600; font-size: 12.5px; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
</style>
<main>
<h1>{html.escape(args.title)}</h1>
<p>{html.escape(args.subtitle)}</p>
{body}
</main>
"""
    with open(output, "w") as handle:
        handle.write(page)
    print(output)
    if args.png:
        binary = chrome()
        if not binary:
            raise SystemExit("no Chrome or Chromium found for --png")
        height = int((svg.y + 120) * 680 / W) + 40
        subprocess.run(
            [binary, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
             f"--window-size=680,{height}", f"--screenshot={os.path.abspath(args.png)}", "file://" + os.path.abspath(output)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(args.png)


if __name__ == "__main__":
    main()
