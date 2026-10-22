"""A region's view straight from the database's faces (region_pieces.json): every wall face with the region on one
side and something else on the other. With --check, compared wall by wall with the region's hand-checked colouring
(region_walls.json): the area each has that the other lacks.

    python region_view.py F2 --check
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

FLIP = np.array([1, 1, -1.0])


def view(rid, path="region_pieces.json", kinds=False):
    """{wall: triangles (n, 3, 3) in cell xyz} of the faces the region borders (with kinds: also {wall: [kind of
    each triangle]})."""
    if path not in view.cache:
        if path.endswith(".json") and os.path.exists(path):
            view.cache[path] = json.load(open(path))
        elif os.path.isdir(path):                  # (a folder of <wall>.json.gz, as wall_sections.py writes)
            import glob
            import gzip
            view.cache[path] = {os.path.basename(f).split(".")[0]: json.load(gzip.open(f, "rt"))
                                for f in glob.glob(os.path.join(path, "*.json.gz"))}
        else:
            from region_db import load_pieces
            view.cache[path] = load_pieces()
    fp = view.cache[path]
    out = defaultdict(list)
    kd = defaultdict(list)
    for wid, faces in fp.items():
        for f in faces:
            a, b = f["sides"]
            if f["mirror"] or (a == rid) == (b == rid):
                continue
            # (a face lying in the view's cut is drawn where the region is on the visible side: past the cut, the
            # same place is drawn again at the glued copy)
            if f.get("cut", [False, False])[0 if a == rid else 1] and _in_cut(f["tris"]):
                continue
            out[wid] += f["tris"]
            kd[wid] += [f["kind"]] * len(f["tris"])
    tris = {w: np.array(T).reshape(-1, 3, 3) for w, T in out.items()}
    return (tris, dict(kd)) if kinds else tris


view.cache = {}


def export(regions=None, keep=("F2",), path="region_pieces.json"):
    """region_walls.json (the page's region colouring) for every complete region: one whose pockets touch only walls
    already in the database. The regions in keep stay as they are there (hand-checked)."""
    db = json.load(open("region_db.json"))
    grid = json.load(open("region_grid.json"))
    done = set(db["walls"])
    try:
        out = json.load(open("region_walls.json"))
    except FileNotFoundError:
        out = {}
    ready = [r for r, ws in db.get("touch", {}).items() if set(ws) <= done and r in grid["runs"]]
    if os.path.isdir(path) and regions:
        # (from the wall sections: the regions asked for, whole; a hand-checked colouring is replaced too)
        ready, keep = list(regions), ()
    for r in ready:
        if (regions and r not in regions) or r in keep:
            continue
        tris, kinds = view(r, path=path, kinds=True)
        # (the pieces of each wall triangle merged back, by kind, so the page carries far fewer triangles)
        for w in list(tris):
            parts = []
            kd = np.array(kinds[w])
            for k in sorted(set(kinds[w])):
                M = merge_coplanar(tris[w][kd == k])
                parts.append((M, [k] * len(M)))
            tris[w] = np.concatenate([p[0] for p in parts]) if parts else tris[w]
            kinds[w] = [k for p in parts for k in p[1]]
        T = [t for w in sorted(tris) for t in tris[w]]
        if not T:
            continue
        P = (np.array(T).reshape(-1, 3) * FLIP).round(6)
        out[r] = {"walls": sorted(w for w in tris if len(tris[w])), "pockets": grid["pockets"][r],
                  "tris": P.reshape(-1, 9).tolist(), "ids": [w for w in sorted(tris) for _ in tris[w]],
                  "kinds": [k for w in sorted(tris) for k in kinds[w]],
                  "grid": {"lo": grid["lo"], "h": grid["h"], "shape": grid["shape"], "runs": grid["runs"][r]},
                  "source": "wall_sections" if os.path.isdir(path) else "region_db"}
        print(f"  {r}: {len(T)} triangles on {', '.join(out[r]['walls'])}")
    json.dump(out, open("region_walls.json", "w"), separators=(",", ":"))
    waiting = {r: sorted(set(ws) - done) for r, ws in db.get("touch", {}).items() if not set(ws) <= done}
    return ready, waiting


def merge_coplanar(T, tol=1e-9):
    """The triangles T (n, 3, 3) with the pieces of each original wall triangle merged back: pieces lying in one plane
    and touching are joined in that plane and filled again, so a triangle coloured whole is one triangle again while
    the line where the colour stops stays exact."""
    import shapely
    from shapely.ops import unary_union
    from shapely.geometry import Polygon
    T = np.asarray(T, float)
    if not len(T):
        return T
    N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(N, axis=1)
    keep = ln > 1e-18
    T, N = T[keep], N[keep] / ln[keep][:, None]
    N *= np.where(N[np.arange(len(N)), np.abs(N).argmax(1)] < 0, -1, 1)[:, None]      # (one orientation per plane)
    d = np.einsum("ni,ni->n", N, T[:, 0])
    key = np.c_[np.round(N / 1e-7), np.round(d / 1e-8)].astype(np.int64)
    _, grp = np.unique(key, axis=0, return_inverse=True)
    out = []
    for g in range(grp.max() + 1):
        idx = np.flatnonzero(grp == g)
        if len(idx) == 1:
            out.append(T[idx[0]])
            continue
        n, o = N[idx[0]], T[idx[0], 0]
        e1 = T[idx[0], 1] - o
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        polys = [Polygon(np.c_[(t - o) @ e1, (t - o) @ e2]) for t in T[idx]]
        m = unary_union([p.buffer(1e-9) for p in polys]).buffer(-1e-9).simplify(1e-9, preserve_topology=True)
        for part in (m.geoms if hasattr(m, "geoms") else [m]):
            if part.area <= 0:
                continue
            for tri in shapely.constrained_delaunay_triangles(part).geoms:
                c = np.array(tri.exterior.coords)[:3]
                out.append(o + c[:, :1] * e1 + c[:, 1:2] * e2)
    return np.array(out).reshape(-1, 3, 3)


def _in_cut(tris, tol=1e-6):
    """A face lying in a face of the cell or in the splitting mirror (not one merely beside them)."""
    import region_cells as rc
    B = rc.betas(np.array(tris).reshape(-1, 3))
    return bool((np.abs(B).min(axis=1) < tol).all() and (np.argmin(np.abs(B), axis=1) == np.argmin(np.abs(B[0]))).all()
                or (np.abs(B[:, 2] - B[:, 3]) < tol).all())


def area(T):
    return 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)


def sample(T, step=4e-4, rng=np.random.default_rng(0)):
    """Points spread over the triangles, about one per step² of area."""
    a = area(T)
    n = np.maximum(1, np.ceil(a / step ** 2)).astype(int)
    idx = np.repeat(np.arange(len(T)), n)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = T[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0]), idx


def uncovered(A, B, tol=1.2e-3, step=4e-4):
    """Area of A further than tol from B (both triangles), and the points where."""
    if len(A) == 0:
        return 0.0, np.zeros((0, 3))
    P, idx = sample(A, step)
    w = area(A)[idx] / np.bincount(idx, minlength=len(A))[idx]
    if len(B) == 0:
        return float(w.sum()), P
    Q, _ = sample(B, step / 2)
    d, _ = cKDTree(Q).query(P)
    far = d > tol
    return float(w[far].sum()), P[far]


def check(rid):
    mine = view(rid)
    r = json.load(open("region_walls.json"))[rid]
    T = (np.array(r["tris"]).reshape(-1, 3) * FLIP).reshape(-1, 3, 3)
    ids = np.array(r["ids"])
    ref = {w: T[ids == w] for w in set(ids)}
    print(f"{rid}: view from the database's faces against the checked colouring")
    print(f"  {'wall':6s} {'view':>8s} {'checked':>8s} {'extra':>8s} {'missing':>8s}")
    for w in sorted(set(mine) | set(ref)):
        A, B = mine.get(w, np.zeros((0, 3, 3))), ref.get(w, np.zeros((0, 3, 3)))
        ex, pe = uncovered(A, B)
        mi, pm = uncovered(B, A)
        print(f"  {w:6s} {area(A).sum():8.5f} {area(B).sum():8.5f} {ex:8.5f} {mi:8.5f}")
        check.where = getattr(check, "where", {})
        check.where[w] = (pe, pm)


if __name__ == "__main__":
    if "--export" in sys.argv:
        # (--sections [region,...]: those regions from the wall sections, sec_pieces/)
        if "--sections" in sys.argv:
            k = sys.argv.index("--sections")
            regs = sys.argv[k + 1].split(",") if len(sys.argv) > k + 1 else None
            ready, waiting = export(regions=regs, path="sec_pieces")
        else:
            ready, waiting = export()
        print(f"complete: {', '.join(sorted(ready))}")
        for r, ws in sorted(waiting.items(), key=lambda kv: len(kv[1])):
            print(f"  {r} waits for {', '.join(ws)}")
    elif "--check" in sys.argv:
        check(sys.argv[1])
    else:
        v = view(sys.argv[1])
        print({w: round(float(area(T).sum()), 5) for w, T in v.items()})
