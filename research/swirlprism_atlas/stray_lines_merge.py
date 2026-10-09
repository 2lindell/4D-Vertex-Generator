"""Add the lines traced from stray samples (stray_lines.py output) to x_point_lines.json, keyed "<id> stray <k>".
Pieces that are images of an already added piece of the same shape under the 2400-element group are left out (they are
drawn as its copies).

    python stray_lines_merge.py stray_lines_a.json [stray_lines_b.json ...]
"""
import json
import sys

import numpy as np

from cellframe import T, TINV
from normalizer import extended_group

M = np.einsum("ij,gjk,kl->gil", TINV, np.array(extended_group(), float), np.array(T, float).T)


FOLD = [1, 0, 3, 2]


def orbit(a, b):
    """The piece's images (end pairs, each end in the displayed half), for spotting a piece that is another's image."""
    out = []
    for g in M:
        A, B = g @ a, g @ b
        if A.sum() <= 0 or B.sum() <= 0:
            continue
        A, B = A / A.sum(), B / B.sum()
        out += [(A, B), (A[FOLD], B[FOLD])]
    return out


def same_piece(a, b, imgs):
    return any((np.abs(a - A).max() < 1e-7 and np.abs(b - B).max() < 1e-7) or
               (np.abs(a - B).max() < 1e-7 and np.abs(b - A).max() < 1e-7) for A, B in imgs)


def text(e):
    t = e["text"]
    return t if t.startswith("(") else "(" + ", ".join(f"{v:.6g}" for v in json.loads(t)) + ")"


def name(lab):
    return "an unnumbered shape" if lab.startswith("new:") else "the cell face" if lab in ("face", "OUT") else lab


def main(paths):
    d = json.load(open("x_point_lines.json"))
    d = {k: v for k, v in d.items() if " stray " not in k}          # (rebuilt from scratch each time)
    added = {}
    for path in paths:
        for tid, lines in json.load(open(path)).items():
            for r in lines:
                for pc in r["pieces"]:
                    a, b = (np.array(e["beta"], float) for e in pc)
                    if np.linalg.norm(a - b) < 1e-6:
                        continue
                    a, b = a / a.sum(), b / b.sum()
                    if same_piece(a, b, added.get(tid, [])):
                        continue
                    added.setdefault(tid, []).extend(orbit(a, b))
                    k = sum(1 for key in d if key.startswith(f"{tid} stray "))
                    ends = [e.get("beyond", "face") for e in pc]
                    d[f"{tid} stray {k + 1}"] = {
                        "id": tid, "kind": "line", "a": a.tolist(), "b": b.tolist(),
                        "ends": [x for x in ends if not x.startswith("new:") and x not in ("face", "OUT", "ERR")],
                        "where": f"the line from β ∝ {text(pc[0])} ({name(ends[0])} beyond) to β ∝ {text(pc[1])} "
                                 f"({name(ends[1])} beyond), found from samples off its other lines"}
                    print(tid, d[f"{tid} stray {k + 1}"]["where"])
    json.dump(d, open("x_point_lines.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
