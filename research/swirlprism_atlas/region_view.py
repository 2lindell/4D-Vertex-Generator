"""A region's view straight from the database's faces (region_pieces.json): every wall face with the region on one
side and something else on the other. With --check, compared wall by wall with the region's hand-checked colouring
(region_walls.json): the area each has that the other lacks.

    python region_view.py F2 --check
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

FLIP = np.array([1, 1, -1.0])


def view(rid, path="region_pieces.json", kinds=False):
    """{wall: triangles (n, 3, 3) in cell xyz} of the faces the region borders (with kinds: also {wall: [kind of
    each triangle]})."""
    fp = view.cache.get(path) or view.cache.setdefault(path, json.load(open(path)))
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


def export(regions=None, keep=("F2",)):
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
    for r in ready:
        if (regions and r not in regions) or r in keep:
            continue
        tris, kinds = view(r, kinds=True)
        T = [t for w in sorted(tris) for t in tris[w]]
        if not T:
            continue
        P = (np.array(T).reshape(-1, 3) * FLIP).round(6)
        out[r] = {"walls": sorted(w for w in tris if len(tris[w])), "pockets": grid["pockets"][r],
                  "tris": P.reshape(-1, 9).tolist(), "ids": [w for w in sorted(tris) for _ in tris[w]],
                  "kinds": [k for w in sorted(tris) for k in kinds[w]],
                  "grid": {"lo": grid["lo"], "h": grid["h"], "shape": grid["shape"], "runs": grid["runs"][r]},
                  "source": "region_db"}
        print(f"  {r}: {len(T)} triangles on {', '.join(out[r]['walls'])}")
    json.dump(out, open("region_walls.json", "w"), separators=(",", ":"))
    waiting = {r: sorted(set(ws) - done) for r, ws in db.get("touch", {}).items() if not set(ws) <= done}
    return ready, waiting


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
        ready, waiting = export()
        print(f"complete: {', '.join(sorted(ready))}")
        for r, ws in sorted(waiting.items(), key=lambda kv: len(kv[1])):
            print(f"  {r} waits for {', '.join(ws)}")
    elif "--check" in sys.argv:
        check(sys.argv[1])
    else:
        v = view(sys.argv[1])
        print({w: round(float(area(T).sum()), 5) for w, T in v.items()})
