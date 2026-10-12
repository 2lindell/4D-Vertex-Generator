"""The shapes on the boundary of each atlas shape (written to boundaries.json, shown in the key's Boundaries column).

Four sources, each exact or nearly so where it applies:
  * lines traced along chords (xloci_lines.json): the shapes just past either end of each run of a line shape;
  * exact wall patches (fexact_patches.json): the line shapes on their edges (label_edges) and the shapes at their
    corners; the corners at the ends of an edge bound that edge's line shape; and the regions on the two sides of
    each patch, found by nudging its centre off the wall;
  * the curved walls' meshes (xsurfaces.json): the shapes past their boundary lines; the regions on their two sides
    (surface_sides.json); the lines and points on their edges (edge_shapes.json); the ends of traced curves (xcurves.json);
  * rays out of the thin regions (region_rays.json): the walls they cross;
  * the exact corners of the curved walls (corner_points.json): the shapes there;
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
    sep = defaultdict(set)          # the regions on the two sides of each wall (wall_sides.json)

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
            sep[patches[key]["target"]].add(lab)

    try:                                    # the curved walls X10 and X33 (xsurface.py): the shapes past their edges
        for tid, sf in json.load(open("xsurfaces.json")).items():
            for line in sf["outline"]:
                for p in line:
                    lab = p["beyond"][5:] if p["beyond"].startswith("face:") else p["beyond"]
                    if lab not in ("OUT", "ERR", tid) and not lab.startswith("new:"):   # (a line's tolerance halo)
                        bnd[tid].add(lab)
    except FileNotFoundError:
        pass
    # the regions on either side of each curved wall (surface_sides.py): each bounds the other; a side met only a
    # few times is a passing neighbour near an edge, not a side
    try:
        for tid, sides in json.load(open("surface_sides.json")).items():
            total = sum(sides.values())
            for reg, n in sides.items():
                if n >= max(3, 0.1 * total) and not reg.startswith("new:"):
                    bnd[reg].add(tid)
                    bnd[tid].add(reg)
                    sep[tid].add(reg)
    except FileNotFoundError:
        pass
    # the shapes found on the curved walls' edges (edge_shapes.py): a line or point there bounds the wall
    try:
        for key, ons in json.load(open("edge_shapes.json"))["summary"].items():
            wall, beyond = key.split("|")
            for on, n in ons:
                if on not in (wall, beyond, "ERR") and not on.startswith("new:"):
                    bnd[wall].add(on)
    except FileNotFoundError:
        pass
    # the ends of the traced curves (xcurves.json)
    try:
        for tid, cv in json.load(open("xcurves.json")).items():
            for e in cv.get("ends", []):
                if e and e.get("end_shape"):
                    bnd[tid].add(e["end_shape"])
    except FileNotFoundError:
        pass
    # the exact corners of the curved walls (corner_points.py): the shape there bounds the wall, and the line shapes
    # on the two edges meeting there (edge_shapes.json) end at it
    try:
        E = json.load(open("edge_shapes.json"))["summary"]
        for tid, cs in json.load(open("corner_points.json")).items():
            for c in cs:
                lab = c["shape"]
                if lab in ("ERR", tid) or lab.startswith("new:"):
                    continue
                bnd[tid].add(lab)
                for past in c["edges"]:
                    for on, n in E.get(f"{tid}|{past}", []):
                        if on not in ("ERR", tid, lab) and not on.startswith("new:"):
                            bnd[on].add(lab)
    except FileNotFoundError:
        pass
    # the line shape along each edge of the curved walls (edge_lines.py: each edge's middle, now exact)
    try:
        for tid, pairs in json.load(open("edge_lines.json")).items():
            for past, lab, *_ in pairs:
                if lab not in ("ERR", tid) and not lab.startswith("new:"):
                    bnd[tid].add(lab)
    except FileNotFoundError:
        pass
    # the walls met by rays out of a region (region_rays.py), seen at least twice
    try:
        for tid, r in json.load(open("region_rays.json")).items():
            for on, n in r["on"]:
                if n >= 2 and on not in (tid, "OUT", "ERR") and not on.startswith("new:"):
                    bnd[tid].add(on)
                    bnd[on].add(tid)                 # (and the wall has the region on one side)
                    sep[on].add(tid)
            for (on, _), (past, n) in zip(r["on"], r.get("beyond", [])):   # (the region across each wall, in order)
                if past not in (tid, on, "OUT", "ERR") and not past.startswith("new:"):
                    bnd[on].add(past)
                    bnd[past].add(on)
                    sep[on].add(past)
    except FileNotFoundError:
        pass
    for tid, r in json.load(open("xloci_step1.json")).items():
        if r["kept"] < r["of"]:
            for nb in r["neighbours"]:
                bnd[nb].add(tid)
                sep[tid].add(nb)

    # a curved wall's boundary is what was measured on it: the regions on its two sides, the line along each edge,
    # the point at each corner (other walls past its edges, and labels picked up near it, are not its boundary)
    try:
        import re
        page = open("../../assets/symmetry_domain.html").read()
        m = re.search(r'<script[^>]*id="atlas-data"[^>]*>', page)
        dims = {t["id"]: t.get("dim") for t in json.loads(page[m.end():page.index("</script>", m.end())])["types"]}
        sides = json.load(open("surface_sides.json"))
        lines_ = json.load(open("edge_lines.json"))
        corners_ = json.load(open("corner_points.json"))
        for tid in json.load(open("xsurfaces.json")):
            own = set()
            sd = sides.get(tid, {})
            total = sum(sd.values())
            own |= {r for r, n in sd.items() if n >= max(3, 0.1 * total) and dims.get(r) == "region"}
            # (the shape found on the edge itself, at two of its three samples at least: a line or point shape,
            # or a wall or region the edge runs along (T2, say), but not the wall's own side regions)
            sides_ = {r for r, n in sd.items() if n >= max(3, 0.1 * total)}
            own |= {e[1] for e in lines_.get(tid, []) if (len(e) < 3 or e[2] >= 2) and e[1] not in sides_
                    and not e[1].startswith("new:") and e[1] != "ERR"}
            own |= {c["shape"] for c in corners_.get(tid, []) if dims.get(c["shape"]) in ("line", "point")}
            # every stretch of the outline bisected onto a face of the cell (face:<shape>) has that shape along it,
            # and a stretch whose shape past it is a line shape runs along that line
            for l in json.load(open("xsurfaces.json"))[tid]["outline"]:
                labs_ = [q["beyond"] for q in l]
                for k in range(len(labs_)):
                    lab_ = labs_[k]
                    if k >= 2 and labs_[k - 1] == lab_ and labs_[k - 2] == lab_:      # (three points at least)
                        if lab_.startswith("face:"):
                            own.add(lab_[5:])
                        elif dims.get(lab_) in ("line", "point"):
                            own.add(lab_)
            # the shapes the bisected edge check found on the edge itself (edge_shapes.json)
            for key, ons in json.load(open("edge_shapes.json"))["summary"].items():
                if key.split("|")[0] == tid:
                    own |= {o for o, n in ons if n >= 2 and o not in sides_ and o != "ERR" and not o.startswith("new:")}
            # two curved walls meeting along an edge share its line: the line another wall measured on its edge past
            # this one is on this wall's edge too
            # (not between two pieces of one surface, X10 and X39 on the same quadric: they meet only at points)
            import xsurface
            eq = lambda w: xsurface.surface(w)["equation"]
            for w, es in lines_.items():
                if w != tid and any(e[0] == tid for e in es) and eq(w) == eq(tid):
                    continue
                own |= {e[1] for e in es if w != tid and e[0] == tid and (len(e) < 3 or e[2] >= 2)
                        and dims.get(e[1]) in ("line", "point")}
            own = {o for o in own if not o.startswith("new:") and o not in ("ERR", "OUT")}
            # (another piece of the wall's own surface is not on its boundary: X39 and X52 beside X10 on its quadric)
            own = {o for o in own if not (o in json.load(open("xsurfaces.json")) and eq(o) == eq(tid))}
            own.discard(tid)
            if own:
                bnd[tid] = own
    except (FileNotFoundError, AttributeError):
        pass
    out = {t: sorted(s) for t, s in bnd.items()}
    json.dump(out, open("boundaries.json", "w"), indent=1)
    # (only regions: the page keeps the walls' entries and drops the rest)
    json.dump({t: sorted(r for r in s if r not in ("ERR", "OUT", t) and not r.startswith("new:"))
               for t, s in sep.items()}, open("wall_sides.json", "w"), indent=1)
    print(f"{len(out)} shapes with boundaries")


if __name__ == "__main__":
    main()
