"""Each wall's edges and cuts from its extent (arrangement.py), as exact curves, and the checks (arrangement_edges.json).

Where the wall stops between two neighbouring points of its scaffold, the curve it stops along (a copy of a wall it
meets, or of one of its lines, or a seam) is solved exactly between them; these points, joined across the scaffold's
triangles, are the edge. A cut is the same where the wall lies on both sides of a copy of a cutting wall that is there.
Each end of an edge is put on the exact point it reaches (exact_points.json: the meeting point of the surfaces there,
every image of it) or onto the face of the half-cell it reaches. The checks, all from the symmetry and the exact data:
  * every end of every edge is an exact point or on a face of the half-cell;
  * the exact points at the wall's corners are the points the working data lists for it;
  * every edge the working data lists is found, and nothing else bounds the wall;
  * W's edge along Y lies on Y as grown (Y's edge, or Y's cut by W).
Nothing in the working data is changed.

    python arrangement_draw.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

import arrangement as ar
from arrangement import CACHE, CELL, GG, H, LIB, N, Wall, crossing, fvals, off, onto
import core as osd
import __main__
__main__.Img = ar.Img                                     # (the setup was saved by arrangement.py run as a script)

OUT = "arrangement_edges.json"
FACEN = {"b1=0": np.eye(4)[0], "b2=0": np.eye(4)[1], "b4=0": np.eye(4)[3], "b3=b4": np.array([0, 0, 1, -1.0])}


def exact_point_images():
    pts = [p for p in json.load(open("exact_points.json"))["points"] if p["exact"]]
    X, names = [], []
    for p in pts:
        Y = GG @ np.array(p["float"])
        s = Y.sum(1)
        ok = np.abs(s) > 1e-12
        Y = Y[ok] / s[ok, None]
        Y = Y[(Y @ CELL.T >= -1e-9).all(1)]
        Y = np.unique(Y.round(12), axis=0)
        X.append(Y)
        names += [p["name"]] * len(Y)
    return np.vstack(X), names


def chains(segs):
    """Segments [(key_a, key_b)] joined into chains of keys."""
    adj = defaultdict(list)
    for a, b in segs:
        adj[a].append(b)
        adj[b].append(a)
    seen, out = set(), []
    starts = [k for k in adj if len(adj[k]) != 2] + list(adj)
    for s in starts:
        if s in seen:
            continue
        ch, prev, cur = [s], None, s
        seen.add(s)
        while True:
            nx = [k for k in adj[cur] if k != prev and k not in seen]
            if not nx:
                if len(ch) > 2 and s in adj[cur] and prev is not None:
                    ch.append(s)
                break
            prev, cur = cur, nx[0]
            seen.add(cur)
            ch.append(cur)
        out.append(ch)
    return out


def sheet_curves(W, sn, walls):
    """The wall's edges and cuts on one of its surfaces: [(kind, source, points (betas))]."""
    sh = W.sheets[sn]
    V, T, ins, funcs, sg = sh["V"], sh["T"], sh["ins"], sh["funcs"], sh["sg"]
    S = LIB[sn]
    inw = W.inwall[sn]
    out = []
    Ei = sh["E"][ins[sh["E"][:, 0]] & ins[sh["E"][:, 1]]]
    for k, (kind, src, f, P) in enumerate(funcs):
        if kind == "seam":
            continue                                       # (the wall goes on onto its next plane there)
        if kind == "line":
            # (the line's own exact points, each stretch kept where the steps it passes between have the wall on one
            # side only)
            # (steps with one end in the half-cell: a line lying in one of its faces is passed by the steps out of it)
            Ea = sh["E"][ins[sh["E"][:, 0]] | ins[sh["E"][:, 1]]]
            wa, wb = inw[Ea[:, 0]] & ins[Ea[:, 0]], inw[Ea[:, 1]] & ins[Ea[:, 1]]
            keep = [bool(len(st) and (wa[st] != wb[st]).mean() > 0.5) for st in ar.line_steps(V, Ea, P, S, per=True)]
            run = []
            for i, kp in enumerate(keep):
                if kp:
                    run = run or [P[i]]
                    run.append(P[i + 1])
                elif run:
                    out.append((kind, src, np.array(run), None))
                    run = []
            if run:
                out.append((kind, src, np.array(run), None))
            continue
        s = sg[:, k]
        tri_ch = (s[T] != s[T[:, [1, 2, 0]]])                              # (each triangle's edges 01, 12, 20)
        rows = np.flatnonzero(tri_ch.sum(1) == 2)
        if not len(rows):
            continue
        E = np.concatenate([T[rows][:, [0, 1]], T[rows][:, [1, 2]], T[rows][:, [2, 0]]])
        ch = tri_ch[rows].T.reshape(-1)
        E = np.sort(E[ch], 1)
        keys, inv = np.unique(E, axis=0, return_inverse=True)
        both_in = ins[keys[:, 0]] & ins[keys[:, 1]]
        w0, w1 = inw[keys[:, 0]], inw[keys[:, 1]]
        if kind == "cut":
            want = both_in & w0 & w1
        else:
            want = both_in & (w0 != w1)
        if not want.any():
            continue
        X = np.full((len(keys), 4), np.nan)
        X[want] = crossing(V, keys[want], f, S)
        real = ar.real_at(kind, src, P, X[want], walls)
        good = np.zeros(len(keys), bool)
        good[np.flatnonzero(want)[real]] = True
        # (each triangle's two crossing edges, in the order E was made: every 01 edge, then 12, then 20)
        m = len(rows)
        idx = -np.ones((3, m), int)
        idx[ch.reshape(3, m)] = inv.reshape(-1)
        segs = []
        for j in range(m):
            e2 = idx[:, j][idx[:, j] >= 0]
            if len(e2) == 2 and good[e2[0]] and good[e2[1]]:
                segs.append((int(e2[0]), int(e2[1])))
        for c in chains(segs):
            out.append((kind, src, X[c], f))
    return out


def finish_end(x, S, f, EP, names):
    """An end of an edge: the exact point it reaches, or the face of the half-cell, or None."""
    d = np.linalg.norm(EP - x, axis=1)
    near = np.flatnonzero(d < 2.5 * H)
    best = None
    for i in near:
        if off(S, EP[i])[0] < 1e-9 and (f is None or off(f, EP[i])[0] < 1e-9) and (best is None or d[i] < d[best]):
            best = i
    if best is not None:
        return EP[best], names[best]
    for nm, n in FACEN.items():
        if abs(x @ n) < 2.5 * H:
            y = onto([S, osd.Plane(n, nm)] + ([f] if f is not None else []), x[None])[0]
            if abs(y @ n) < 1e-12 and np.linalg.norm(y - x) < 3 * H:
                return y, "face " + nm
    return None, None


def main():
    work = json.load(open("working_data.json"))["walls"]
    got = pickle.load(open(os.path.join(CACHE, f"setup_{N}s.pkl"), "rb"))
    z = np.load(os.path.join(CACHE, "walls.npz"))
    walls = {}
    for w, sn, V, E, T, ins, funcs, sg in got:
        walls.setdefault(w, {})[sn] = {"V": V, "E": E, "T": T, "ins": ins, "funcs": funcs, "sg": sg}
    walls = {w: Wall(w, sh) for w, sh in walls.items()}
    for w, W in walls.items():
        W.inwall = {sn: z[f"{w}|{sn}"] for sn in W.sheets}
    EP, names = exact_point_images()
    out, report = {}, {}
    for w, W in walls.items():
        rec = {"solid": [], "dashed": []}
        found_edges, corners, loose = set(), set(), []
        for sn in W.sheets:
            S = LIB[sn]
            for kind, src, X, f in sheet_curves(W, sn, walls):
                ends = []
                for k in (0, -1):
                    if len(X) > 2 and np.linalg.norm(X[0] - X[-1]) < 1e-12:
                        ends.append("loop")
                        continue
                    y, nm = finish_end(X[k], S, f, EP, names)
                    if y is None:
                        loose.append(f"{kind} {src}: end at {np.round(X[k], 4).tolist()} reaches no exact point")
                        ends.append("?")
                        continue
                    ends.append(nm)
                    if not nm.startswith("face"):
                        corners.add(nm)
                    X = np.vstack([y[None], X]) if k == 0 else np.vstack([X, y[None]])
                if kind != "cut":
                    found_edges.add("seam" if kind == "seam" else src)
                (rec["dashed"] if kind == "cut" else rec["solid"]).append(
                    {"kind": kind, "along": src, "surface": sn, "ends": ends, "points": X.round(12).tolist()})
        wd = work[w]
        listed = set(wd.get("points", [])) | set(wd.get("boundary_points", []))
        edges = set(wd["edges"])
        found_named = found_edges
        report[w] = {"edges_listed": sorted(edges), "edges_found": sorted(found_named),
                     "missing_edges": sorted(edges - found_named), "extra": sorted(found_named - edges - {"seam"}),
                     "corners_found": sorted(corners), "points_listed": sorted(listed),
                     "corners_not_listed": sorted(corners - listed), "listed_not_corners": sorted(listed - corners),
                     "loose_ends": loose[:10], "n_loose": len(loose),
                     "share": {sn: [int(W.inwall[sn].sum()), int(W.sheets[sn]["ins"].sum())] for sn in W.sheets}}
        out[w] = rec
        r = report[w]
        flag = "ok" if not (r["missing_edges"] or r["extra"] or r["n_loose"] or r["corners_not_listed"]
                            or r["listed_not_corners"]) else "CHECK"
        print(f"{w}: {flag}; edges {r['edges_found']} (missing {r['missing_edges']}, extra {r['extra']}); corners "
              f"{r['corners_found']} (listed {r['points_listed']}); loose ends {r['n_loose']}", flush=True)
    json.dump({"walls": out}, open(OUT, "w"))
    json.dump(report, open("arrangement_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
