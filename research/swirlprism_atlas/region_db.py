"""A database of the regions and the walls between them (region_db.json): for each wall, its drawn pieces split into
faces wherever another wall meets it, and the region on each side of every face; for each region, the wall faces it
borders. Built from the page as drawn (every wall piece at the page's resolution, copies included).

  * a flat piece is split in its own plane along the traces every other wall leaves on it (region_walls.split_flat),
    and each face is classified on both sides at a point well inside it;
  * a curved piece's triangles are cut where other walls cross them, and each part's sides are looked up in the
    voxel pockets of the half-cell cut by all the walls; each pocket is labelled by classifying points deep inside
    it (a pocket whose points disagree is marked mixed); where the voxels give up (a part next to a face of the cell,
    or wedged between walls), the part is classified directly.

Sides are named from the drawn half: "outside" is past a face of the cell, "mirror" past the splitting mirror (what
lies there is drawn beside the wall's copy).

    python region_db.py T2,X3,X9,X10,X11,X14,X21        (the walls to add or redo; reads ../../assets/symmetry_domain.html)
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import region_cells as rc
import region_walls as rw

CACHE = "region_walls_cache.jsonl"          # (beta -> shape, shared with region_walls.py)
OUT = "region_db.json"
PIECES = "region_pieces.json"


class Labels:
    """The classifier, cached on disk."""

    def __init__(self, path=CACHE):
        self.path, self.cache = path, {}
        if os.path.exists(path):
            for line in open(path):
                k, v = json.loads(line)
                self.cache[k] = v

    key = staticmethod(lambda b: ",".join(f"{x:.10f}" for x in b / b.sum()))

    def __call__(self, B):
        """Shapes at betas B (n, 4); "outside" past a face of the cell, "mirror" past the splitting mirror."""
        B = np.asarray(B, float)
        out = [None] * len(B)
        todo = {}
        for i, b in enumerate(B):
            if b.min() <= 0:
                out[i] = "outside"
            elif b[2] < b[3]:
                out[i] = "mirror"
            else:
                k = self.key(b)
                if k in self.cache:
                    out[i] = self.cache[k]
                else:
                    todo.setdefault(k, []).append(i)
        if todo:
            ks = sorted(todo)
            with Pool(4) as pool, open(self.path, "a") as f:
                for n_, (k, l) in enumerate(zip(ks, pool.imap(_lab_timed, [np.array([float(x) for x in k.split(",")]) for k in ks],
                                                                chunksize=1))):
                    if l != "TIMEOUT":
                        self.cache[k] = l
                        f.write(json.dumps([k, l]) + "\n")
                        f.flush()
                    for i in todo[k]:
                        out[i] = l
                    if (n_ + 1) % 100 == 0:
                        print(f"    classified {n_ + 1} of {len(ks)}", flush=True)
        return out


def _lab_timed(b, seconds=30):
    """The classifier with a time limit: a hull that will not settle is "TIMEOUT"."""
    import signal

    def stop(*_):
        raise TimeoutError
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)
    try:
        return rc._lab_job(b)
    except TimeoutError:
        return "TIMEOUT"
    finally:
        signal.alarm(0)


def voxels(pieces, h=0.0025):
    lo, hi = rc.CORNERS.min(axis=0) - 3 * h, rc.CORNERS.max(axis=0) + 3 * h
    shape = tuple(np.ceil((hi - lo) / h).astype(int) + 1)
    vox = lambda X: np.clip(np.floor((np.asarray(X) - lo) / h).astype(int), 0, np.array(shape) - 1)
    wall = np.zeros(shape, bool)
    for _, _, _, T in pieces:
        for t in T:
            e = max(np.linalg.norm(t[1] - t[0]), np.linalg.norm(t[2] - t[1]), np.linalg.norm(t[0] - t[2]))
            n = max(1, int(np.ceil(e / (0.4 * h))))
            a_, b_ = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            k = a_ + b_ <= n
            a, b = a_[k] / n, b_[k] / n
            idx = vox(t[0] + a[:, None] * (t[1] - t[0]) + b[:, None] * (t[2] - t[0]))
            wall[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    g = np.stack(np.meshgrid(*[lo[k] + h * (np.arange(shape[k]) + 0.5) for k in range(3)], indexing="ij"), -1)
    Bg = rc.betas(g.reshape(-1, 3)).reshape(shape + (4,))
    lab, _ = ndimage.label(rc.inside(Bg) & ~wall)
    return lab, lo, h


def pocket_labels(lab, lo, h, pockets, labels, per=3):
    """Each pocket's shape, from points spread deep inside it ("mixed: a/b" when they disagree)."""
    objs = ndimage.find_objects(lab)
    pts, owner = [], []
    for p in pockets:
        sl = objs[p - 1]
        m = lab[sl] == p
        d = ndimage.distance_transform_edt(m)
        idx = np.argwhere(m & (d >= min(2, d.max())))
        pick = [idx[np.argmax(d[tuple(idx.T)])]]
        far = np.linalg.norm(idx - pick[0], axis=1)
        while len(pick) < per and far.max() > 3:
            pick.append(idx[np.argmax(far)])
            far = np.minimum(far, np.linalg.norm(idx - pick[-1], axis=1))
        off = np.array([s.start for s in sl])
        for q in pick:
            pts.append(rc.betas((lo + h * (q + off + 0.5))[None])[0])
            owner.append(p)
    got = labels(np.array(pts))
    by = defaultdict(list)
    for p, l in zip(owner, got):
        by[p].append(l)
    by = {p: [l for l in ls if l not in ("TIMEOUT", "ERR")] or ["TIMEOUT"] for p, ls in by.items()}
    return {p: (ls[0] if len(set(ls)) == 1 else "mixed: " + "/".join(sorted(set(ls)))) for p, ls in by.items()}


def main(walls):
    targets = set(walls.split(","))
    d = rw.page_data()
    dims = {t["id"]: t["dim"] for t in d["types"]}
    all_walls = {w["id"] for w in d["xwalls"]} | {"T2"}
    print(f"{len(all_walls)} walls on the page; building every drawn piece", flush=True)
    pieces = rw.pieces_of(d, all_walls)
    others = [(w, T) for w, _, _, T in pieces]
    labels = Labels()
    lab, lo, h = voxels(pieces)
    shape = np.array(lab.shape)
    print(f"voxels: {lab.max()} pockets", flush=True)

    def pocket_at(X):
        i = np.floor((X - lo) / h).astype(int)
        ok = np.all((i >= 0) & (i < shape), axis=1)
        v = np.full(len(X), -1)
        v[ok] = lab[tuple(i[ok].T)]
        B = rc.betas(X)
        v[B.min(axis=1) <= 0] = -2                    # (outside the cell)
        v[(B.min(axis=1) > 0) & (B[:, 2] < B[:, 3])] = -3   # (past the mirror)
        return v

    def pocket_sides(C, N):
        res = []
        for s in (1, -1):
            r = np.full(len(C), -1)
            for k in (1.5, 2.5, 3.5, 4.5):
                q = pocket_at(C + s * k * h * N)
                m = (r == -1) & (q != -1) & (q != 0)
                r[m] = q[m]
            res.append(r)
        return np.stack(res, 1)

    db = json.load(open(OUT)) if os.path.exists(OUT) else {"walls": {}, "regions": {}}
    curved_parts = []                                   # (wall, kind, polygon, normal)
    flat_faces = []                                     # (wall, kind, polygon (shapely), frame, normal, inner point, thin)
    for wid, kind, idx, T in pieces:
        if wid not in targets:
            continue
        if rw.is_flat(T):
            for g, frame, n, P3, thin in rw.split_flat(wid, T, others):
                flat_faces.append((wid, kind, g, frame, n, P3, thin))
        else:
            parts = _cut_parts(T, wid, pieces)
            curved_parts += [(wid, kind, pg, n) for pg, n in parts]
    print(f"{len(flat_faces)} flat faces, {len(curved_parts)} curved parts", flush=True)
    # curved parts: pockets on both sides, then the pockets' shapes
    C = np.array([pg.mean(0) for _, _, pg, _ in curved_parts]) if curved_parts else np.zeros((0, 3))
    Nn = np.array([n for *_, n in curved_parts]) if curved_parts else np.zeros((0, 3))
    ps = pocket_sides(C, Nn) if len(C) else np.zeros((0, 2), int)
    pk = sorted({int(p) for p in ps.ravel() if p > 0})
    print(f"  {len(pk)} pockets beside them; labelling", flush=True)
    plab = pocket_labels(lab, lo, h, pk, labels)
    name = lambda p: plab.get(int(p)) if p > 0 else {-2: "outside", -3: "mirror"}.get(int(p))
    side_names = [[name(a), name(b)] for a, b in ps]
    # where the voxels give up, or a pocket is mixed: the classifier, a short step off the part
    redo = [i for i, sn in enumerate(side_names) if None in sn or any(str(x).startswith("mixed") for x in sn)]
    if redo:
        step = 4e-4
        B = np.concatenate([rc.betas(C[redo] + s * step * Nn[redo]) for s in (1, -1)])
        got = labels(B)
        for j, i in enumerate(redo):
            side_names[i] = [got[j], got[len(redo) + j]]
    print(f"  {len(redo)} parts classified directly", flush=True)
    # flat faces: the same, from a point well inside each face (the classifier where the voxels give up)
    P3 = np.array([f[5] for f in flat_faces]) if flat_faces else np.zeros((0, 3))
    N3 = np.array([f[4] for f in flat_faces]) if flat_faces else np.zeros((0, 3))
    fps = pocket_sides(P3, N3) if len(P3) else np.zeros((0, 2), int)
    new_pk = sorted({int(p) for p in fps.ravel() if p > 0} - set(plab))
    plab.update(pocket_labels(lab, lo, h, new_pk, labels))
    flat_sides = [[name(a), name(b)] for a, b in fps]
    redo = [i for i, sn in enumerate(flat_sides) if None in sn or any(str(x).startswith("mixed") for x in sn)]
    if redo:
        B = np.concatenate([rc.betas(P3[redo] + s * 4e-4 * N3[redo]) for s in (1, -1)])
        got = labels(B)
        for j, i in enumerate(redo):
            flat_sides[i] = [got[j], got[len(redo) + j]]
    print(f"  flat faces: {len(redo)} classified directly", flush=True)
    # gather: per wall, per pair of sides, the area and the faces (polygons in betas, for drawing and checking)
    area3 = lambda pg: 0.5 * np.linalg.norm(sum(np.cross(pg[m] - pg[0], pg[m + 1] - pg[0]) for m in range(1, len(pg) - 1)))
    acc = defaultdict(lambda: defaultdict(lambda: {"area": 0.0, "pieces": Counter(), "points": []}))
    for (wid, kind, pg, n), sn in zip(curved_parts, side_names):
        key = "|".join(sorted(str(x) for x in sn))
        e = acc[wid][key]
        e["area"] += area3(pg)
        e["pieces"][kind] += 1
        if len(e["points"]) < 6:
            e["points"].append(np.round(rc.betas(pg.mean(0)[None])[0], 6).tolist())
    for (wid, kind, g, (o, e1, e2), n, P, thin), sn in zip(flat_faces, flat_sides):
        key = "|".join(sorted(str(x) for x in sn))
        e = acc[wid][key]
        e["area"] += g.area
        e["pieces"][kind] += 1
        if len(e["points"]) < 6:
            e["points"].append(np.round(rc.betas(P[None])[0], 6).tolist())
    # the faces themselves, for drawing any region from them (region_pieces.json: cell xyz triangles, fanned)
    fp = json.load(open(PIECES)) if os.path.exists(PIECES) else {}
    for wid in targets:
        fp[wid] = []
    for (wid, kind, pg, n), sn in zip(curved_parts, side_names):
        fp[wid].append({"sides": [str(x) for x in sn], "kind": kind, "mirror": _in_mirror(pg),
                        "tris": [np.r_[pg[0], pg[m], pg[m + 1]].round(7).tolist() for m in range(1, len(pg) - 1)]})
    for (wid, kind, g, frame, n, P, thin), sn in zip(flat_faces, flat_sides):
        T = _fill(g, frame)
        fp[wid].append({"sides": [str(x) for x in sn], "kind": kind, "mirror": _in_mirror(T.reshape(-1, 3)),
                        "tris": T.reshape(-1, 9).round(7).tolist()})
    json.dump(fp, open(PIECES, "w"), separators=(",", ":"))
    for wid in targets:
        db["walls"][wid] = {"dim": dims.get(wid), "sides": {k: {"area": round(v["area"], 6), "pieces": dict(v["pieces"]),
                                                               "points": v["points"]} for k, v in acc[wid].items()}}
    # regions: the wall faces each borders (a pair with the region on one side only; both sides: the wall lies inside it)
    regs = defaultdict(lambda: defaultdict(lambda: {"area": 0.0, "across": Counter(), "inside": 0.0}))
    for wid, w in db["walls"].items():
        for key, v in w["sides"].items():
            a, b = key.split("|")
            for r, other in ((a, b), (b, a)):
                if dims.get(r) != "region":
                    continue
                e = regs[r][wid]
                if r == other:
                    e["inside"] += v["area"]
                else:
                    e["area"] += v["area"]
                    e["across"][other] += v["area"]
    db["regions"] = {r: {wid: {"area": round(v["area"], 6), "across": {k: round(x, 6) for k, x in v["across"].most_common()},
                               "inside": round(v["inside"], 6)} for wid, v in ws.items()} for r, ws in regs.items()}
    json.dump(db, open(OUT, "w"), indent=1)
    for wid in sorted(targets):
        print(wid)
        for k, v in sorted(db["walls"][wid]["sides"].items(), key=lambda kv: -kv[1]["area"]):
            print(f"    {k:28s} area {v['area']:.5f}  {v['pieces']}")


def _in_mirror(P):
    """Lying in the splitting mirror (such a face is drawn by the wall's copy past it, not here)."""
    B = rc.betas(np.asarray(P))
    return bool(np.abs(B[:, 2] - B[:, 3]).max() < 1e-6)


def _fill(g, frame):
    """A flat face (shapely polygon in its plane) as triangles in cell xyz."""
    import shapely
    o, e1, e2 = frame
    out = []
    for part in (g.geoms if hasattr(g, "geoms") else [g]):
        for t in shapely.constrained_delaunay_triangles(part).geoms:
            c = np.array(t.exterior.coords)[:3]
            out.append(o + c[:, :1] * e1 + c[:, 1:2] * e2)
    return np.array(out).reshape(-1, 3, 3)


def _cut_parts(T, wid, pieces, reach=0.03, max_parts=64):
    """A curved piece's triangles, each cut along the other walls crossing it: [(polygon, normal)]."""
    if not hasattr(_cut_parts, "index"):
        allT = np.concatenate([P for *_, P in pieces])
        wall_of = np.concatenate([[w] * len(P) for w, *_, P in pieces])
        rad = np.max(np.linalg.norm(allT - allT.mean(axis=1)[:, None], axis=2), axis=1)
        small = rad < reach
        _cut_parts.index = (allT, wall_of, cKDTree(allT[small].mean(axis=1)), np.flatnonzero(small), np.flatnonzero(~small))
    allT, wall_of, tree, sidx, big = _cut_parts.index
    out = []
    N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(N, axis=1)
    N = N / np.maximum(ln, 1e-300)[:, None]
    for j, t in enumerate(T):
        if ln[j] < 1e-15:
            continue
        rt = np.max(np.linalg.norm(t - t.mean(0), axis=1))
        cand = [c for c in list(sidx[tree.query_ball_point(t.mean(0), rt + reach)]) + list(big)
                if wall_of[c] != wid and rw._crosses(t, allT[c])]
        polys = [list(t)]
        for c in cand:
            u = allT[c]
            nu = np.cross(u[1] - u[0], u[2] - u[0])
            polys = [q for pg in polys for q in (rw._clip(pg, u[0], nu), rw._clip(pg, u[0], -nu)) if len(q) >= 3]
            if len(polys) > max_parts:
                break
        for pg in polys:
            pg = np.array(pg)
            if np.linalg.norm(sum(np.cross(pg[m] - pg[0], pg[m + 1] - pg[0]) for m in range(1, len(pg) - 1))) > 1e-12:
                out.append((pg, N[j]))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
