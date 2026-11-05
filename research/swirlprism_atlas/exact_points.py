"""Every meeting point of the atlas, solved exactly (exact_points.json).

The points where lines end (each line's ends and the ends of its pieces, exact_defs.json: every point where a wall's
edges meet is one of them) were found in floating point and stored to about 9 digits, so two names for one point could
differ by 1e-8 or more. Each is solved again here at full precision (exact.py, 60 digits): the surfaces through it are
found (library surfaces, or their images under the group where the library has no third one), three of them crossing
are solved together, and every one is checked to hold there to 40 digits. The same point reached from different sources
is kept once, with every source.

Each point: its betas (40 digits), the surfaces through it, its name by the classifier, and where it came from.

    python exact_points.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np

OUT = "exact_points.json"


def sources():
    ex = json.load(open("exact_defs.json"))
    out = []
    for L, v in ex["lines"].items():
        for e in v.get("ends", []):
            out.append((e["at"], f"end of {L} on {e.get('on')}"))
        for P in v["points"]:
            if len(P):
                out.append((P[0], f"end of a piece of {L}"))
                out.append((P[-1], f"end of a piece of {L}"))
    return out


def solve_one(x):
    import exact
    from core import label
    r = exact.corner(x)
    if r is None:
        return {"float": list(map(float, x)), "exact": False, "why": "fewer than three surfaces cross cleanly there"}
    p, on = r
    pf = [float(v) for v in p]
    return {"b": exact.text(p, 40), "float": pf, "exact": True, "surfaces": on, "name": label(np.array(pf))[0],
            "moved": float(np.linalg.norm(np.array(pf) - np.asarray(x, float)))}


def main():
    src = sources()
    reps, tags = [], []
    for x, tag in src:
        x = np.asarray(x, float)
        x = x / x.sum()
        for i, r in enumerate(reps):
            if np.linalg.norm(r - x) < 1e-5:
                tags[i].append(tag)
                break
        else:
            reps.append(x)
            tags.append([tag])
    with Pool() as pool:
        got = pool.map(solve_one, reps, chunksize=4)
    # (two stored points that solve to one exact point are one point)
    pts = []
    for g, t in zip(got, tags):
        same = next((p for p in pts if g["exact"] and p["exact"] and p["b"] == g["b"]), None)
        if same:
            same["from"] = sorted(set(same["from"]) | set(t))
        else:
            pts.append({**g, "from": sorted(set(t))})
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "points": pts}, open(OUT, "w"), indent=0)
    ok = [p for p in pts if p["exact"]]
    print(f"{len(src)} stored points, {len(reps)} distinct, {len(ok)} solved exactly, {len(pts) - len(ok)} not; "
          f"largest move {max(p['moved'] for p in ok):.2e}", flush=True)
    for p in pts:
        if not p["exact"]:
            print("  not solved:", np.round(p["float"], 6).tolist(), p["from"][:3])


if __name__ == "__main__":
    main()
