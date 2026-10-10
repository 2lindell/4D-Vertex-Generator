"""The shapes on the boundary of each atlas shape (written to boundaries.json, shown in the key's Boundaries column).

Four sources, each exact or nearly so where it applies:
  * lines traced along chords (xloci_lines.json): the shapes just past either end of each run of a line shape;
  * exact wall patches (fexact_patches.json): the line shapes on their edges (label_edges) and the shapes at their
    corners; the corners at the ends of an edge bound that edge's line shape; and the regions on the two sides of
    each patch, found by nudging its centre off the wall;
  * the curved walls' meshes (xsurfaces.json): the shapes past their boundary lines;
  * the nudge test (xloci_step1.json): every region a transitional sample falls into when nudged has that
    transitional shape on its boundary.
A boundary is listed as found, so a shape whose surroundings are only partly traced has a partial list.

    python boundaries.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from fexact import _lab_loose, HALF


def _corner_jobs(patches):
    """Distinct corners of the patches: every corner of a polygon, and the corners of a curved patch that end a
    straight edge (the arc points between them lie on its conic)."""
    out = []
    for key, p in patches.items():
        C = np.array(p["corners"], float)
        C = C / C.sum(axis=1, keepdims=True)
        n = len(C)
        if n < 12:
            ks = range(n)
        else:
            d = [np.linalg.norm(C[(k + 1) % n] - C[k]) for k in range(n)]
            arc = np.median(d)
            ks = {k for k in range(n) if d[k] > 4 * arc} | {(k + 1) % n for k in range(n) if d[k] > 4 * arc}
            labs = p.get("edge_labels", [])        # and where one labelled run of edges gives way to the next
            for prev, cur in zip(labs[-1:] + labs[:-1], labs):
                if prev[2] != cur[2] or len(cur) > 3:
                    ks |= {cur[0] % n} | ({cur[1] % n} if len(cur) > 3 else set())
            ks = sorted(ks)
        for k in ks:
            out.append((key, k, C[k]))
    return out


def _side_jobs(patches, eps=1e-5):
    out = []
    for key, p in patches.items():
        C = np.array(p["corners"], float)
        C = C / C.sum(axis=1, keepdims=True)
        c = C.mean(axis=0)
        n = np.asarray(p["normal"], float)
        n = n - n.mean()          # stay in the plane sum(beta) = 1
        n = n / np.linalg.norm(n)
        for s in (1, -1):
            b = c + s * eps * n
            if np.all(np.array(HALF) @ b > -1e-12) and np.all(b > 0):
                out.append((key, s, b))
    return out


def _lab_job(b):
    return _lab_loose(b)


def main(procs=4):
    bnd = defaultdict(set)

    for tid, chords in json.load(open("xloci_lines.json")).items():
        for ch in chords.values():
            labs = ch.get("labels", [])
            for k, lab in enumerate(labs):
                if lab != tid:
                    continue
                for j in (k - 1, k + 1):
                    if 0 <= j < len(labs) and labs[j] != tid:
                        bnd[tid].add(labs[j])

    patches = json.load(open("fexact_patches.json"))
    corners = _corner_jobs(patches)
    sides = _side_jobs(patches)
    with Pool(procs) as pool:
        clabs = pool.map(_lab_job, [b for _, _, b in corners], chunksize=2)
        slabs = pool.map(_lab_job, [b for _, _, b in sides], chunksize=2)
    at = {(key, k): lab for (key, k, _), lab in zip(corners, clabs)}
    for (key, _, _), lab in zip(corners, clabs):
        # a corner on a curve can fall just past it, into the shape beyond: that shape is not on the boundary
        if lab != patches[key]["target"] and lab != "ERR" and lab not in (patches[key].get("curved") or []):
            bnd[patches[key]["target"]].add(lab)
    for key, p in patches.items():
        tid = p["target"]
        n = len(p["corners"])
        for a, b, lab, *_ in p.get("edge_labels", []):      # _ is ["straight"] for one straight edge
            if lab == tid or lab.startswith("mixed:"):
                continue
            bnd[tid].add(lab)
            if len(_) == 3:                 # a part of a straight edge: only its ends at corners
                ends = ([a] if _[1] < 1e-9 else []) + ([b] if _[2] > 1 - 1e-9 else [])
            else:
                ends = (a, b) if n < 12 or _ else [k for k in range(a, a + 7) if (key, k % n) in at]  # an arc's ends
            for k in ends:
                if at.get((key, k % n), "ERR") not in ("ERR", lab, tid):
                    bnd[lab].add(at[(key, k % n)])
    for key, p in patches.items():      # points where the shape along a straight edge changes (fexact split)
        tid = p["target"]
        labs = p.get("edge_labels", [])
        for a, bb, t, lab in p.get("edge_points", []):
            if lab in ("ERR", tid):
                continue
            bnd[tid].add(lab)
            for l in labs:                  # it ends the parts of that edge on either side
                if len(l) == 6 and l[0] == a and (abs(l[4] - t) < 1e-9 or abs(l[5] - t) < 1e-9) and l[2] != lab:
                    bnd[l[2]].add(lab)
    for (key, _, _), lab in zip(sides, slabs):
        if lab != "ERR" and lab != patches[key]["target"]:
            bnd[lab].add(patches[key]["target"])

    try:                                    # the curved walls X10 and X33 (xsurface.py): the shapes past their edges
        for tid, sf in json.load(open("xsurfaces.json")).items():
            for line in sf["outline"]:
                for p in line:
                    lab = p["beyond"][5:] if p["beyond"].startswith("face:") else p["beyond"]
                    if lab not in ("OUT", "ERR", tid) and not lab.startswith("new:"):   # (a line's tolerance halo)
                        bnd[tid].add(lab)
    except FileNotFoundError:
        pass
    for tid, r in json.load(open("xloci_step1.json")).items():
        if r["kept"] < r["of"]:
            for nb in r["neighbours"]:
                bnd[nb].add(tid)

    out = {t: sorted(s) for t, s in bnd.items()}
    json.dump(out, open("boundaries.json", "w"), indent=1)
    print(f"{len(out)} shapes with boundaries")


if __name__ == "__main__":
    main()
