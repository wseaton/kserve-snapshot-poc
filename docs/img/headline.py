#!/usr/bin/env python3
"""Write headline.svg: seconds until a DeepSeek-V4.1-Flash replica serves, by how it starts.

Numbers are from docs/cross-node.md and NVIDIA's 2026-11-10 post. Run it again after changing them.
"""
import os

GRAY, AQUA, ORANGE, VIOLET = "#b9b8b2", "#1baf7a", "#eb6834", "#7a5cd6"
ROWS = [
    ("Cold boot, cold page cache", [(166.1, GRAY)]),
    ("Cold boot, warm page cache", [(104.7, GRAY)]),
    ("Restore, weights from a cold volume", [(13.1, AQUA), (88.4, ORANGE)]),
    ("Restore, weights from page cache", [(12.8, AQUA), (30.1, ORANGE)]),
    ("Dynamo Snapshot restore, published (B200, weights in the checkpoint)", [(31.0, VIOLET)]),
    ("Restore, weights from a serving replica over InfiniBand", [(13.5, AQUA), (7.0, ORANGE)]),
]
LEGEND = [(GRAY, "cold boot"), (AQUA, "CRIU, CUDA and cuInterpose restore"), (ORANGE, "weight reload"), (VIOLET, "published by NVIDIA")]
W, LEFT, RIGHT, ROW = 760, 0, 70, 46
SCALE = (W - LEFT - RIGHT) / 170

parts, y = [], 26
x = 0
for color, label in LEGEND:
    parts.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="3" fill="{color}"/>')
    parts.append(f'<text x="{x + 18}" y="{y}" class="muted">{label}</text>')
    x += 18 + 7.2 * len(label) + 22
y += 22
for label, segments in ROWS:
    parts.append(f'<text x="0" y="{y + 12}" class="label">{label}</text>')
    x = LEFT
    for value, color in segments:
        width = value * SCALE
        parts.append(f'<rect x="{x:.1f}" y="{y + 18}" width="{width - 1.5:.1f}" height="16" rx="4" fill="{color}"/>')
        x += width
    total = sum(v for v, _ in segments)
    parts.append(f'<text x="{x + 6:.1f}" y="{y + 31}" class="value">{total:.1f} s</text>')
    y += ROW
height = y + 4
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" role="img" aria-label="Seconds until a DeepSeek-V4.1-Flash replica serves on 8 GPUs">
<style>
text {{ font: 13px -apple-system, system-ui, "Segoe UI", sans-serif; fill: #0b0b0b; }}
.muted, .label {{ fill: #6f6e69; font-size: 12.5px; }} .value {{ font-weight: 650; }}
@media (prefers-color-scheme: dark) {{ text {{ fill: #f0efec; }} .muted, .label {{ fill: #a3a29b; }} }}
</style>
{chr(10).join(parts)}
</svg>
"""
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "headline.svg"), "w") as handle:
    handle.write(svg)
