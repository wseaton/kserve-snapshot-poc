#!/usr/bin/env python3
"""Write the snapshotd mark: templates/logo.html, static/img/favicon.svg, static/img/logo.svg and
static/img/og.png.

Two six-blade irises of opposite handedness overlap. West of the centre line only the west iris shows,
east of it only the east one. In the header the irises turn in opposite directions, so blades meet at
the seam moving the same way and change colour as they cross.

og.png is rendered with headless Chrome so it uses the site's own Public Sans. Run it again after
changing the mark.
"""
import math
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(os.path.dirname(HERE))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

SEP, R, r, BLADES, GAP, SEAM = 16, 27, 8, 6, 2.2, 2.2
WEST, EAST = 50 - SEP, 50 + SEP


def blades(cx, rot, mirror):
    vs = [(cx + r * math.cos(rot + 2 * math.pi * i / BLADES), 50 + r * math.sin(rot + 2 * math.pi * i / BLADES)) for i in range(BLADES)]
    tips = []
    for i in range(BLADES):
        (ax, ay), (bx, by) = vs[i], vs[(i + 1) % BLADES]
        length = math.hypot(bx - ax, by - ay)
        dx, dy = (bx - ax) / length, (by - ay) / length
        ox, oy = ax - cx, ay - 50
        proj = ox * dx + oy * dy
        t = -proj + math.sqrt(proj * proj - (ox * ox + oy * oy - R * R))
        tips.append((ax + t * dx, ay + t * dy))
    flip = (lambda x, y: (2 * cx - x, y)) if mirror else (lambda x, y: (x, y))
    paths = []
    for i in range(BLADES):
        v1, v2 = flip(*vs[(i + 1) % BLADES]), flip(*vs[(i + 2) % BLADES])
        p0, p1 = flip(*tips[i]), flip(*tips[(i + 1) % BLADES])
        sweep = 0 if mirror else 1
        paths.append(
            f"M{v1[0]:.2f} {v1[1]:.2f}L{p0[0]:.2f} {p0[1]:.2f}"
            f"A{R} {R} 0 0 {sweep} {p1[0]:.2f} {p1[1]:.2f}L{v2[0]:.2f} {v2[1]:.2f}Z"
        )
    return paths


def disk(cx, radius):
    return f"M{cx - radius} 50a{radius} {radius} 0 1 0 {2 * radius} 0a{radius} {radius} 0 1 0 {-2 * radius} 0Z"


def mark(prefix, west_fill, east_fill, spin=False):
    """Inner SVG markup. Fills are attribute strings, e.g. 'fill="#2c2b27"' or 'class="logo-w"'."""
    half = SEAM / 2

    def blade_mask(name, paths):
        body = "".join(f'<path d="{d}" fill="#fff" stroke="#000" stroke-width="{GAP}" stroke-linejoin="round"/>' for d in paths)
        return f'<mask id="{prefix}{name}" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="100">{body}</mask>'

    def side_mask(name, other, x, width):
        return (
            f'<mask id="{prefix}{name}" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="100">'
            f'<rect width="100" height="100" fill="#fff"/><path d="{disk(other, R + GAP)}" fill="#000"/>'
            f'<rect x="{x}" width="{width}" height="100" fill="#fff"/></mask>'
        )

    defs = (
        "<defs>"
        + blade_mask("wb", blades(WEST, 0, mirror=False))
        + blade_mask("eb", blades(EAST, math.pi / 6, mirror=True))
        + side_mask("ws", EAST, 0, 50 - half)
        + side_mask("es", WEST, 50 + half, 50 - half)
        + "</defs>"
    )
    west_spin = ' class="logo-spin-w"' if spin else ""
    east_spin = ' class="logo-spin-e"' if spin else ""
    return (
        defs
        + f'<g mask="url(#{prefix}ws)"><g{west_spin}><rect width="100" height="100" {west_fill} mask="url(#{prefix}wb)"/></g></g>'
        + f'<g mask="url(#{prefix}es)"><g{east_spin}><rect width="100" height="100" {east_fill} mask="url(#{prefix}eb)"/></g></g>'
    )


HEADER_VIEWBOX = "5 21 90 58"

header = (
    f'<svg class="logo" viewBox="{HEADER_VIEWBOX}" aria-hidden="true">'
    + mark("logo-", 'class="logo-w"', 'class="logo-e"', spin=True)
    + "</svg>\n"
)
with open(os.path.join(DOCS, "templates", "logo.html"), "w") as handle:
    handle.write(header)

favicon = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="5 5 90 90">'
    "<style>.w{fill:#2c2b27}.e{fill:#23704c}"
    "@media (prefers-color-scheme: dark){.w{fill:#dcd9ce}.e{fill:#6fc292}}</style>"
    + mark("f", 'class="w"', 'class="e"')
    + "</svg>\n"
)
with open(os.path.join(HERE, "favicon.svg"), "w") as handle:
    handle.write(favicon)

standalone = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{HEADER_VIEWBOX}">'
    "<style>.w{fill:#2c2b27}.e{fill:#23704c}"
    "@media (prefers-color-scheme: dark){.w{fill:#dcd9ce}.e{fill:#6fc292}}</style>"
    + mark("l", 'class="w"', 'class="e"')
    + "</svg>\n"
)
with open(os.path.join(HERE, "logo.svg"), "w") as handle:
    handle.write(standalone)

fonts = os.path.join(DOCS, "static", "fonts")
card = f"""<!doctype html>
<meta charset="utf-8">
<style>
@font-face {{ font-family: "Public Sans"; font-weight: 400; src: url("file://{fonts}/public-sans-latin-400-normal.woff2"); }}
@font-face {{ font-family: "Public Sans"; font-weight: 700; src: url("file://{fonts}/public-sans-latin-700-normal.woff2"); }}
html, body {{ margin: 0; width: 1200px; height: 630px; overflow: hidden; }}
body {{ background: #efede6; color: #2c2b27; font-family: "Public Sans", sans-serif; display: flex; align-items: center; gap: 56px; padding: 0 80px; box-sizing: border-box; }}
svg {{ width: 400px; flex: none; }}
.name {{ font: 26px Menlo, monospace; letter-spacing: 0.18em; text-transform: uppercase; color: #57554c; margin: 0 0 22px; }}
.tag {{ font-size: 46px; font-weight: 700; line-height: 1.14; letter-spacing: -0.01em; margin: 0; }}
</style>
<svg viewBox="{HEADER_VIEWBOX}">{mark("o", 'fill="#2c2b27"', 'fill="#23704c"')}</svg>
<div>
  <p class="name">snapshotd</p>
  <p class="tag">Restore vLLM replicas<br>from CRIU checkpoints<br>instead of cold booting.</p>
</div>
"""
with tempfile.TemporaryDirectory() as scratch:
    page = os.path.join(scratch, "card.html")
    with open(page, "w") as handle:
        handle.write(card)
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "--window-size=1200,630",
            f"--screenshot={os.path.join(HERE, 'og.png')}",
            f"file://{page}",
        ],
        check=True,
        capture_output=True,
    )
