"""Each wall's boundary, as one piece on the 3-sphere (wall_outline.json): its edges, each a line or another wall, and the
points at their ends. From the walls' own traced edges and corners (edge_lines.json, corner_points.json, the exact
patches' side labels in fexact_patches.json). A side on a face of the half-cell or on the splitting mirror is an edge
only where the wall ends there: a step along the wall across it (mapped back into the half-cell by the group) that
finds the wall again means the wall goes on (into its own image, or its mirror image, as X11 and F4 in their mirrors),
and the side is no edge. A side between two patches of the wall itself is no edge either.

    python wall_outline.py          (needs wall_edges.json for the sides' geometry)
"""
from __future__ import annotations

import collections
import json
from multiprocessing import Pool

import numpy as np

import optimal_seeds as osd
import region_walls as rw
from edge_relabel import lab_edge
from probe import _one

FACES = {"face b1=0": np.eye(4)[0], "face b2=0": np.eye(4)[1], "face b3=0": np.eye(4)[2], "face b4=0": np.eye(4)[3],
         "mirror b3=b4": np.array([0, 0, 1, -1.0])}
ON = 1e-7
CUR = {}
JOBS = []


def face_of(a, b):
    A = np.array([a, b], float)
    A = A / A.sum(1, keepdims=True)
    return [n for n, f in FACES.items() if np.abs(A @ f).max() < ON]


def continues(i):
    """Across a face (or the mirror) from the middle of a side: is the wall there again?"""
    w, a, b, own, fname = JOBS[i]
    fn = FACES[fname]
    m = (np.asarray(a) + np.asarray(b)) / 2
    m = m / m.sum()
    T = osd.tangent(m, [own])
    u = max(T, key=lambda t: abs(t @ fn))
    if abs(u @ fn) < 1e-9:
        return "in it"
    u = -np.sign(u @ fn) * u / np.linalg.norm(u)
    x = osd.home(osd.onto(m + 2e-4 * u, [own]))
    return "continues" if _one(x)[0] == w else "ends"


def main():
    d = rw.page_data()
    dims = {t["id"]: t["dim"] for t in d["types"]}
    planes, curved = osd.library()
    C = {c.name: c for c in curved}
    EL = json.load(open("edge_lines.json"))
    CP = json.load(open("corner_points.json"))
    fp = [p for p in (lambda x: x if isinstance(x, list) else x.values())(json.load(open("fexact_patches.json")))
          if isinstance(p, dict)]
    WE = json.load(open("wall_edges.json"))
    walls = [t for t, dm in dims.items() if dm == "wall"]
    good = lambda w, l: l and l != w and l not in ("ERR", "OUT") and not str(l).startswith(("new:", "face:")) \
        and dims.get(l) in ("line", "wall")
    # (the curved walls' face sides, from wall_edges' cuts: does the wall go on across?)
    jobs, keys = [], []
    for t in WE:
        if "edges" not in t or t["piece"].startswith(("patch", "triangle")):
            continue
        w = t["wall"]
        own = C.get(w) or min(curved, key=lambda c: 0)  # (replaced below for walls on another's surface)
        if w not in C:
            P = np.array([e["from"] for e in t["edges"]] + [e["to"] for e in t["edges"]])
            own = min(curved, key=lambda c: float(np.abs(c.f(P)).max()))
        for e in t["edges"]:
            if e["kind"] == "cut of the view":
                jobs.append((w, e["from"], e["to"], own, e["on"]))
                keys.append((w, "curved", e.get("beyond")))
    # (the flat walls' sides on a face or the mirror)
    for p in fp:
        w = p["target"]
        if w == "T2" and len(p["corners"]) == 3 and abs(np.asarray(p["normal"], float) @ [0, 1, -1, 0]) > 1.4:
            continue                                  # (T2's former region B: X96)
        Cn = np.array(p["corners"], float)
        own = osd.Plane(p["normal"], "own")
        for k, (i, j, lab, *_) in enumerate(p.get("edge_labels") or []):
            fs = face_of(Cn[i], Cn[j])
            if fs:
                jobs.append((w, Cn[i].tolist(), Cn[j].tolist(), own, fs[0]))
                keys.append((w, f"patch {p['wall']}", k))
    JOBS[:] = jobs
    with Pool(3) as pool:
        verdict = pool.map(continues, range(len(jobs)))
    V = collections.defaultdict(list)
    for kk, v in zip(keys, verdict):
        V[kk].append(v)
    out = {}
    corner_jobs, corner_w = [], []
    for w in walls:
        edges, points, cut = collections.Counter(), set(), []
        # curved: its traced edges; a face side only where the wall ends there
        for past, on, *n in EL.get(w, []):
            if str(past).startswith("face:"):
                vs = V.get((w, "curved", past), [])
                if vs and all(v == "continues" for v in vs):
                    cut.append(on)
                    continue
            if good(w, on):
                edges[on] += 1
        # (a face side the traced edges do not cover, where the wall ends: the line on that face)
        for (ww, kind, bey), vs in V.items():
            if ww == w and kind == "curved" and "ends" in vs:
                lab = str(bey)[5:] if str(bey).startswith("face:") else str(bey)
                if good(w, lab) and dims.get(lab) == "line" and not any(str(past) == bey for past, *_ in EL.get(w, [])):
                    edges[lab] += 1
        for c in CP.get(w, []):
            if dims.get(c["shape"]) == "point":
                points.add(c["shape"])
        # flat: each patch's sides; a face side only where the wall ends there (or lies in it)
        for p in fp:
            if p["target"] != w:
                continue
            Cn = np.array(p["corners"], float)
            labs = p.get("edge_labels") or []
            true = []
            for k, (i, j, lab, *_) in enumerate(labs):
                vs = V.get((w, f"patch {p['wall']}", k))
                ok = lab != w and not (vs and all(v == "continues" for v in vs))
                true.append(ok)
                if ok and good(w, lab):
                    edges[lab] += 1
                elif vs and all(v == "continues" for v in vs):
                    cut.append(lab)
            for k, (i, j, lab, *_) in enumerate(labs):
                if true[k] and labs[k - 1][2] != lab:
                    corner_jobs.append(Cn[i] / Cn[i].sum())
                    corner_w.append(w)
        out[w] = {"edges": sorted(edges), "points": points, "goes_on_across": sorted(set(map(str, cut)))}
    with Pool(3) as pool:
        cl = pool.map(lab_edge, corner_jobs)
    for w, l in zip(corner_w, cl):
        if dims.get(l) == "point":
            out[w]["points"].add(l)
    extra = json.load(open("boundaries_extra.json"))
    for w, r in out.items():
        if not r["edges"] and w in extra:             # (no record: its exact geometry)
            r["edges"] = sorted(x for x in extra[w] if dims.get(x) in ("line", "wall"))
            r["points"] |= {x for x in extra[w] if dims.get(x) == "point"}
        r["points"] = sorted(r["points"])
    # (a face or mirror side the wall goes on across is still an edge: what lies across is its copy or its own
    # mirror image (T2 at D1, X26 at D3), not more of the same piece)
    for w, r in out.items():
        add = [x for x in r["goes_on_across"] if x != w and dims.get(x) == "line" and x not in r["edges"]]
        r["edges"] = sorted(set(r["edges"]) | set(add))
    # (walls checked by hand, or as the domain page lists them: wall_outline_checked.json)
    for w, r in json.load(open("wall_outline_checked.json")).items():
        if not w.startswith("_"):
            out[w] = {**out.get(w, {}), **r, "checked": True}
    json.dump(out, open("wall_outline.json", "w"), indent=1)
    for w in walls:
        r = out[w]
        print(f"{w:4s} edges {r['edges']}  points {r['points']}" + (f"   (goes on across {r['goes_on_across']})" if r["goes_on_across"] else ""))


if __name__ == "__main__":
    main()
