"""Colours for the lettered classes (the ones with a colour of their own), following location.

Lightness follows dimension: classes that fill a region are lightest, those on a wall darker, those on a line
darker still, and those at a point (on lines) darkest. Hue follows place: each class takes the hue family of the
coloured region it borders (F2 or F3, from the nudges in xloci_step1.json); classes touching both sit between
the two, and those touching only X regions form a family of their own. Within a family the hues are spread so
every class keeps a distinct colour. In dark mode the lightness order is reversed (deep areas, bright lines) so
lines stay visible on the dark background. The unlisted X classes keep their shared colour.
"""
from __future__ import annotations

import json

import numpy as np

FAMILY = {"F2": (40, 36), "F3": (165, 34), "both": (100, 14), "other": (275, 50)}   # centre hue, half-width
LIGHT = {3: (0.80, 0.11), 2: (0.64, 0.13), 1: (0.52, 0.14), 0: (0.40, 0.12)}       # dimension: (L, C)
DARK = {3: (0.46, 0.09), 2: (0.62, 0.12), 1: (0.74, 0.13), 0: (0.84, 0.10)}


def _oklch_to_hex(L, C, h):
    def conv(c):
        a, b = c * np.cos(np.radians(h)), c * np.sin(np.radians(h))
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        lc, mc, sc = l_ ** 3, m_ ** 3, s_ ** 3
        return np.array([4.0767416621 * lc - 3.3077115913 * mc + 0.2309699292 * sc,
                         -1.2684380046 * lc + 2.6097574011 * mc - 0.3413193965 * sc,
                         -0.0041960863 * lc - 0.7034186147 * mc + 1.7076147010 * sc])
    c = C
    rgb = conv(c)
    while (rgb.min() < -1e-6 or rgb.max() > 1 + 1e-6) and c > 0:        # out of gamut: lower the chroma
        c -= 0.005
        rgb = conv(c)
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in srgb)


def assign(class_ids):
    """{class id: (light hex, dark hex, family, dimension)} for the lettered classes."""
    r1 = json.load(open("xloci_step1.json"))
    r2 = json.load(open("xloci_step2.json"))
    regions = {t for t, r in r1.items() if r["kept"] == r["of"]}
    info = {}
    for t in class_ids:
        if t in regions:
            d = 3
        elif t in r2 and r2[t]["kind"] == "wall":
            d = 2
        elif t in r2 and r2[t]["kind"] == "line":
            d = 1
        else:
            d = 0
        if t in ("F2", "F3"):
            fam = t
        else:
            nb = set(r1.get(t, {}).get("neighbours", [])) & {"F2", "F3"}
            fam = "both" if len(nb) == 2 else (nb.pop() if nb else "other")
        info[t] = (fam, d)
    out = {}
    for fam, (centre, half) in FAMILY.items():
        members = sorted((t for t in class_ids if info[t][0] == fam and t not in ("F2", "F3")),
                         key=lambda t: (-info[t][1], t))
        if fam in ("F2", "F3"):
            out[fam] = (_oklch_to_hex(*LIGHT[3], centre), _oklch_to_hex(*DARK[3], centre), fam, 3)
        n = len(members)
        for k, t in enumerate(members):
            h = centre + (half * (2 * k / (n - 1) - 1) if n > 1 else 0)
            d = info[t][1]
            out[t] = (_oklch_to_hex(*LIGHT[d], h), _oklch_to_hex(*DARK[d], h), fam, d)
    hexes = [c[0] for c in out.values()]
    assert len(set(hexes)) == len(hexes), "two classes share a colour"
    return out


def css(colors):
    """CSS custom properties --ty-<id> for both themes."""
    light = " ".join(f"--ty-{t}: {c[0]};" for t, c in colors.items())
    dark = " ".join(f"--ty-{t}: {c[1]};" for t, c in colors.items())
    return (f":root {{ {light} }}\n@media (prefers-color-scheme: dark) {{ :root:not([data-theme=\"light\"]) {{ {dark} }} }}\n"
            f":root[data-theme=\"dark\"] {{ {dark} }}")
