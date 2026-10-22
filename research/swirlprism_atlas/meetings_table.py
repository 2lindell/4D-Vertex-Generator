"""Every place two walls meet, curve by curve, for a set of walls: what kind of meeting it is and where it runs.

For each wall A asked for and every other wall B, their meshes (the page's, originals and copies) are intersected and
the pieces joined into curves. Each curve is one of:

    named line   it runs along a line the page already has (a traced line, segment or ring): defined
    B ends on A  it runs along B's own edge: a line on A, where A is cut for a region changing across it
    A ends on B  it runs along A's own edge: A's own boundary, nothing new on A
    crossing     along neither wall's edge: the walls pass through each other, a line to create on both

with its length and its two ends (a named point where there is one, else beta), so the lines still to make can be
read off. Writes meetings_table.json.

    python meetings_table.py X3,X9,X10,X11,X14,X21,T2
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

import region_cells as rc
import region_walls as rw
from meetings import mesh_edges, sample_segs, tri_tri

FLIP = np.array([1, 1, -1.0])
TOL = 1.5e-3


def named_things(d):
    lines = []
    for key in ("xlines", "segments", "rings"):
        for l in d.get(key, []):
            if isinstance(l, dict) and isinstance(l.get("pts"), list) and len(l["pts"]) > 1:
                p = np.array(l["pts"], float) * FLIP
                lines.append((l["id"] + (" (copy)" if l.get("copy") else ""), cKDTree(sample_segs(np.stack([p[:-1], p[1:]], 1)))))
    pts = {}
    for key in ("special", "regular", "uniform"):
        for q in d.get(key, []):
            if isinstance(q, dict) and "id" in q and "q" in q:
                pts.setdefault(q["id"] + (" (copy)" if q.get("copy") else ""), np.array(q["q"], float) * FLIP)
    names = list(pts)
    return lines, names, cKDTree(np.array([pts[k] for k in names]))


def main(walls):
    want = walls.split(",")
    d = rw.page_data()
    pieces = rw.pieces_of(d, {w["id"] for w in d["xwalls"]} | {"T2"})
    lines, pnames, ptree = named_things(d)

    def where(x):
        dd, i = ptree.query(x)
        if dd < 2e-3:
            return pnames[i]
        return "β (" + ", ".join(f"{v:.4f}" for v in rc.betas(x[None])[0]) + ")"
    edge = {}
    for pid, (w, k, i, T) in enumerate(pieces):
        E = sample_segs(mesh_edges(T))
        edge[pid] = cKDTree(E) if len(E) else None
    allT = np.concatenate([T for *_, T in pieces])
    owner = np.concatenate([[pid] * len(T) for pid, (*_, T) in enumerate(pieces)])
    wall_of = np.array([pieces[p][0] for p in owner])
    C = allT.mean(1)
    R = np.linalg.norm(allT - C[:, None], axis=2).max(1)
    rs = float(np.quantile(R, 0.95))
    small = np.flatnonzero(R <= rs)
    pairs = small[cKDTree(C[small]).query_pairs(2 * rs, output_type="ndarray")]
    tree = cKDTree(C)
    extra = [(b, j) for b in np.flatnonzero(R > rs) for j in tree.query_ball_point(C[b], R[b] + rs)]
    if extra:
        pairs = np.vstack([pairs, np.array(extra)])
    pairs = pairs[wall_of[pairs[:, 0]] != wall_of[pairs[:, 1]]]
    pairs = pairs[np.isin(wall_of[pairs[:, 0]], want) | np.isin(wall_of[pairs[:, 1]], want)]
    seg, ok = tri_tri(allT[pairs[:, 0]], allT[pairs[:, 1]])
    seg, pairs = seg[ok], pairs[ok]
    # (each meeting piece, seen from wall A's side)
    by = defaultdict(list)
    for (i, j), s in zip(pairs, seg):
        for a, b in ((i, j), (j, i)):
            if wall_of[a] in want:
                by[(owner[a], owner[b])].append(s)
    seen_pair = set()
    out = []
    for (pa, pb), S in by.items():
        S = np.array(S)
        P3 = np.stack([S[:, 0], S.mean(1), S[:, 1]], 1)
        on_named = [None] * len(S)
        for n, tr in lines:
            dd = tr.query(P3.reshape(-1, 3))[0].reshape(-1, 3).max(1)
            for k in np.flatnonzero(dd < TOL):
                on_named[k] = on_named[k] or n
        de_a = edge[pa].query(P3.reshape(-1, 3))[0].reshape(-1, 3).max(1) if edge[pa] else np.full(len(S), np.inf)
        de_b = edge[pb].query(P3.reshape(-1, 3))[0].reshape(-1, 3).max(1) if edge[pb] else np.full(len(S), np.inf)
        kind = np.where([x is not None for x in on_named], "named line",
                        np.where(de_b < TOL, "B ends on A", np.where(de_a < TOL, "A ends on B", "crossing")))
        # (curves: pieces of one kind joined end to end)
        for kd in np.unique(kind):
            m = np.flatnonzero(kind == kd)
            ends = np.concatenate([S[m, 0], S[m, 1]])
            parent = list(range(len(m)))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for p, q in cKDTree(ends).query_pairs(2e-3):
                parent[find(p % len(m))] = find(q % len(m))
            comps = defaultdict(list)
            for k in range(len(m)):
                comps[find(k)].append(m[k])
            for ks in comps.values():
                L = float(np.linalg.norm(S[ks, 1] - S[ks, 0], axis=1).sum())
                if L < 1e-3:
                    continue
                E = S[ks].reshape(-1, 3)
                D = np.linalg.norm(E[:, None] - E[None], axis=2)
                i0, j0 = np.unravel_index(D.argmax(), D.shape)
                names = sorted({on_named[k] for k in ks if on_named[k]})
                out.append({"A": f"{pieces[pa][0]} ({pieces[pa][1]})", "B": f"{pieces[pb][0]} ({pieces[pb][1]})",
                            "kind": str(kd), "line": ", ".join(names), "length": round(L, 4),
                            "from": where(E[i0]), "to": where(E[j0])})
    # (the same curve seen from two pieces of a wall listed once)
    uniq = {}
    for r in out:
        k = (r["A"], r["B"], r["kind"], r["line"], r["length"], r["from"], r["to"])
        uniq[k] = r
    out = sorted(uniq.values(), key=lambda r: (r["A"], r["B"], r["kind"], -r["length"]))
    json.dump(out, open("meetings_table.json", "w"), indent=1)
    for r in out:
        print(f"{r['A']:13s} x {r['B']:13s} {r['kind']:12s} {r['line'][:28]:28s} {r['length']:.4f}  {r['from']}  ->  {r['to']}")


if __name__ == "__main__":
    main(sys.argv[1])
