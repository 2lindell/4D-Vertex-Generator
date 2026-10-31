"""Clean a curved wall's corners: where the shape past the wall's edge changes (a corner, or a stretch where the edge
touches a face of the cell), the mesh from xsurface.py is cut off in chords and the corner is clipped. Around each such
place the edge is followed exactly on the wall's surface, step by step, through the corners (each refined to where the
shape past the edge changes), and the mesh there is rebuilt between the kept mesh and the followed edge.

Following the edge: from a point on it, a small circle on the surface around the point is searched for where the
wall stops (the classifier, bisected around the circle and then outward); of the crossings, the one ahead is the
next point. A corner lies between two points whose shapes past the edge differ, and is bisected along the edge.

    python wall_conform.py X42 [out.json]          (out: a file merged into xsurfaces.json later)
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from cell_atlas2 import ear_clip
from fexact import _lab_loose
from xsurface import surface, project, grad

STEP = 0.003


def in_half(b):
    return b.min() > -1e-12 and b[2] >= b[3] - 1e-12


_SF, _TID = None, None


def _init(tid):
    global _SF, _TID
    _SF, _TID = surface(tid), tid


def lab(b):
    b = np.asarray(b, float)
    return _lab_loose(b / b.sum()) if in_half(b) else "OUT"


def tangent_frame(x):
    """Two orthonormal sum-zero directions tangent to the surface at x."""
    n = grad(_SF, x)
    U = np.linalg.svd(np.vstack([np.ones(4), n]))[2][2:]
    return U


def edge_point(inside, outside):
    """The point on the wall's edge between a point on the wall and one off it (both on the surface), and what is past
    the edge there: the shape just past it, or "face:<shape on the face>" where the edge is the cell's face."""
    a, b = np.asarray(inside, float), np.asarray(outside, float)
    beyond = lab(b)
    for _ in range(30):
        m = project(_SF, (a + b) / 2)
        l = lab(m)
        if l == _TID:
            a = m
        else:
            b, beyond = m, l
    x = project(_SF, (a + b) / 2)
    if beyond == "OUT":
        beyond = "face:" + _lab_loose(np.clip(x, 0, None) / np.clip(x, 0, None).sum())
    return x, beyond


def next_point(p, prev, r=STEP, m=12):
    """The next point along the edge from p (away from prev): where a circle of radius r around p crosses the edge,
    taking the crossing most nearly straight ahead."""
    U = tangent_frame(p)
    ang = np.linspace(0, 2 * np.pi, m + 1)
    at = lambda t: project(_SF, p + r * (np.cos(t) * U[0] + np.sin(t) * U[1]))
    pts = [at(t) for t in ang[:-1]]
    on = [lab(q) == _TID for q in pts]
    fwd = (p - prev) if prev is not None else None
    cands = []
    for k in range(m):
        if on[k] != on[(k + 1) % m]:
            mid = at((ang[k] + ang[k + 1]) / 2)
            cands.append((k, (mid - p) @ fwd if fwd is not None else 0.0))
    if not cands:
        return None
    k = max(cands, key=lambda c: c[1])[0]
    ta, tb = (ang[k], ang[k + 1]) if on[k] else (ang[k + 1], ang[k])
    for _ in range(10):                      # around the circle first, then outward to the edge
        tm = (ta + tb) / 2
        if lab(at(tm)) == _TID:
            ta = tm
        else:
            tb = tm
    return edge_point(at(ta), at(tb))


def refine_corner(a, la, b, lb, centre):
    """The corner between edge points a (shape la past it) and b (lb): bisected along the edge, each trial point pushed
    to the edge from a point well inside the wall."""
    for _ in range(18):
        m = project(_SF, (a + b) / 2)
        d = m - centre
        out = project(_SF, centre + 1.6 * d)
        while lab(out) == _TID:
            out = project(_SF, centre + 1.5 * (out - centre))
        x, l = edge_point(centre, out)
        if l == la:
            a = x
        elif l == lb:
            b = x
        else:
            return x, l                       # a third shape: a short stretch of edge of its own; keep it
    return project(_SF, (a + b) / 2), lb


def _follow(args):
    """Follow the edge from start (a point on it) toward goal, through corners, until within one step of goal."""
    tid, start, goal, prev, centre = args
    _init(tid) if _SF is None else None
    p, goal, prev, centre = (np.asarray(v, float) for v in (start, goal, prev, centre))
    lp = edge_point(centre, project(_SF, p + 1e-4 * (p - centre) / np.linalg.norm(p - centre)))[1]
    pts = [(p.tolist(), lp)]
    for _ in range(int(4 * np.linalg.norm(goal - p) / STEP) + 40):
        if np.linalg.norm(goal - p) < 1.5 * STEP:
            break
        nx = next_point(p, prev)
        if nx is None:
            break
        q, lq = nx
        if lq != lp:                          # a corner between p and q
            c, lc = refine_corner(p, lp, q, lq, centre)
            pts.append((c.tolist(), "corner:" + lp + ">" + lc))
        pts.append((q.tolist(), lq))
        prev, p, lp = p, q, lq
    return pts


def _key(p):
    return tuple(np.round(p, 9))


def _rotate_closed(line):
    """A closed outline started in the middle of its longest run, so no corner zone wraps round its start."""
    L = [q["beyond"] for q in line[:-1]]
    n = len(L)
    best, bk = -1, 0
    k = 0
    while k < n:
        j = k
        while j + 1 < n and L[j + 1] == L[k]:
            j += 1
        if j - k > best:
            best, bk = j - k, (k + j) // 2
        k = j + 1
    body = line[:-1]
    body = body[bk:] + body[:bk]
    return body + [body[0]]


def main(tid, out="xsurfaces.json"):
    _init(tid)
    X = json.load(open("xsurfaces.json"))
    sf = X[tid]
    h = sf.get("h", 0.012)
    outlines = [(_rotate_closed(l) if np.allclose(l[0]["beta"], l[-1]["beta"]) and len(l) > 3 else l) for l in sf["outline"]]
    tris = [np.asarray(t, float) for t in sf["tris"]]
    V = np.array([v for t in tris for v in t])
    zones = []                     # (outline, j0, j1, followed edge start, goal, prev, centre)
    for li, line in enumerate(outlines):
        B = np.array([q["beta"] for q in line])
        L = [q["beyond"] for q in line]
        n = len(line)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(B, axis=0), axis=1))]
        trans = [k for k in range(1, n) if L[k] != L[k - 1]]
        if not trans:
            continue
        groups = [[trans[0]]]       # transitions within 4h along the edge make one zone
        for k in trans[1:]:
            (groups[-1].append(k) if s[k] - s[groups[-1][-1]] < 4 * h else groups.append([k]))
        covered = []
        for g in groups:
            if any(lo <= g[0] <= hi for lo, hi in covered):
                continue
            # drop the triangles within 1.5h of the edge from h before the zone's first transition to h after its last;
            # where the wall is narrow (a tip) that cuts across to the edge further round: widen the zone over it
            a = max(0, int(np.searchsorted(s, s[g[0]] - h)) - 1)
            b = min(n - 1, int(np.searchsorted(s, s[g[-1]] + h)))
            idx = {_key(B[j]): j for j in range(n)}
            for _ in range(6):
                core = B[a:b + 1]
                near = [np.min(np.linalg.norm(t[:, None, :] - core[None], axis=2)) < 1.5 * h for t in tris]
                gone = [t for t, f in zip(tris, near) if f]
                keep = [t for t, f in zip(tris, near) if not f]
                # the cut: edges of kept triangles shared with dropped ones, chained into one path between two edge points
                kept_edges = {}
                for t in keep:
                    for c in range(3):
                        e = tuple(sorted((_key(t[c]), _key(t[(c + 1) % 3]))))
                        kept_edges[e] = kept_edges.get(e, 0) + 1
                adj = {}
                for t in gone:
                    for c in range(3):
                        e = tuple(sorted((_key(t[c]), _key(t[(c + 1) % 3]))))
                        if kept_edges.get(e) == 1 and e[1] not in adj.get(e[0], []):
                            adj.setdefault(e[0], []).append(e[1])
                            adj.setdefault(e[1], []).append(e[0])
                ends = [v for v, nb in adj.items() if len(nb) == 1]
                if len(ends) == 2 and all(e in idx for e in ends):
                    break
                js = [idx[e] for e in ends if e in idx]
                if not js:
                    break
                a, b = max(0, min([a] + js) - 2), min(n - 1, max([b] + js) + 2)
            if len(ends) != 2 or not all(e in idx for e in ends):
                print(f"  outline {li} zone at {g}: the cut is not one path between two edge points; left as it was", flush=True)
                continue
            covered.append((a, b))
            j0, j1 = sorted(idx[e] for e in ends)
            path, seen = [B[j0]], {_key(B[j0])}
            while True:
                nb = [v for v in adj[_key(path[-1])] if v not in seen]
                if not nb:
                    break
                path.append(np.asarray(nb[0], float))
                seen.add(nb[0])
            tris = keep
            mid = B[g[len(g) // 2]]
            far = np.min(np.linalg.norm(V[:, None, :] - B[None, ::2], axis=2), axis=1) > 2 * h
            centre = V[far][np.argmin(np.linalg.norm(V[far] - mid, axis=1))] if far.any() else V.mean(axis=0)
            prev = B[j0 - 1] if j0 > 0 else B[j0] - (B[j0 + 1] - B[j0])
            zones.append((li, j0, j1, path, B[j0].tolist(), B[j1].tolist(), prev.tolist(), centre.tolist()))
    print(f"{tid}: {len(zones)} corner zones", flush=True)
    with Pool(4, initializer=_init, initargs=(tid,)) as pool:
        traced = pool.map(_follow, [(tid, *z[4:]) for z in zones])
    new_outline = [[dict(q) for q in l] for l in outlines]
    cuts = {li: [] for li in range(len(outlines))}
    for (li, j0, j1, path, *_), tr in zip(zones, traced):
        print(f"  outline {li} [{j0}:{j1}]: {len(tr)} points; corners "
              f"{[(np.round(np.asarray(p) / max(p), 4).tolist(), l[7:]) for p, l in tr if l.startswith('corner:')]}", flush=True)
        B = [np.asarray(q["beta"], float) for q in outlines[li]]
        pts = [np.asarray(p, float) for p, _ in tr]
        if np.linalg.norm(pts[-1] - B[j1]) > 1e-12:
            pts.append(B[j1])
        poly = pts + [np.asarray(p) for p in path[::-1]]
        clean = []
        for p in poly:
            if not clean or np.linalg.norm(p - clean[-1]) > 1e-10:
                clean.append(p)
        while len(clean) > 1 and np.linalg.norm(clean[0] - clean[-1]) < 1e-10:
            clean.pop()
        nrm = grad(_SF, np.mean(clean, axis=0))
        tris += [np.array([clean[a], clean[b], clean[c]]) for a, b, c in ear_clip(clean, nrm)]
        seg = [{"beta": p, "beyond": l.split(">")[-1] if l.startswith("corner:") else l} for p, l in tr]
        cuts[li].append((j0, j1, seg))
    for li, cs in cuts.items():
        line = new_outline[li]
        for j0, j1, seg in sorted(cs, reverse=True):          # (from the end, so earlier indices stay valid)
            line[j0:j1] = seg
    res = json.load(open(out)) if os.path.exists(out) else {}
    res[tid] = {**sf, "tris": [t.tolist() for t in tris], "outline": new_outline, "conformed": True,
                "corners": [[p, l[7:]] for tr in traced for p, l in tr if l.startswith("corner:")]}
    json.dump(res, open(out, "w"))
    print(f"{tid}: {len(sf['tris'])} -> {len(tris)} triangles", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
