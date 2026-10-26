"""Each wall's edges as curves, from the working data alone (edge_curves.json), and whether the wall's mesh reaches them.

A wall is one surface on the 3-sphere; the half-cell holds pieces of it: its mesh (xsurfaces.json for a curved wall,
its exact patches in fexact_patches.json for a flat one) carried by every element of the extended group, clipped to
the half-cell (b >= 0, b3 >= b4). So a piece that goes over a face of the cell and comes back elsewhere is tested
where it lands. Its edges are those in working_data.json, each a line or another wall:
  * a line: the line's traced curve (every piece of it the page draws), carried by the group the same way;
  * a wall: the points of the wall's boundary that lie on the other wall (on its pieces, all images).
For each edge, where the line touches the wall's pieces:
  reached  the wall's boundary is there (the mesh stops on the line);
  past     the line runs through the inside of a piece (the mesh goes on past its edge);
  short    the line misses the pieces (the mesh stops short of it, by the distance given).
And the wall's own boundary that lies on none of its edges: where a piece stops on nothing (a side on a face of the
half-cell or the mirror is a cut of the view, not counted).

    python edge_curves.py [X14,F4,...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

import region_walls as rw
from t2_pieces import extended

TOL = 4e-3                   # (boundary to curve: reached)
ON = 2.5e-3                  # (a point on a piece)
FOLD = np.eye(4)[[1, 0, 3, 2]]
HALF = [np.eye(4)[k] for k in range(4)] + [np.array([0, 0, 1, -1.0])]


def nrm(b):
    b = np.asarray(b, float)
    return b / b.sum(-1, keepdims=True)


def dens(P, h=2e-3):
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        k = max(1, int(np.linalg.norm(b - a) / h))
        out += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    return np.array(out)


def clip(poly, h):
    out = []
    for k in range(len(poly)):
        p, q = poly[k], poly[(k + 1) % len(poly)]
        fp, fq = h @ p, h @ q
        if fp >= -1e-12:
            out.append(p)
        if (fp >= -1e-12) != (fq >= -1e-12):
            out.append(nrm(p + (q - p) * (fp / (fp - fq))))
    return out


GG = None


def group():
    global GG
    if GG is None:
        G = extended()
        GG = np.concatenate([G, G @ FOLD])
    return GG


def images(T):
    """Triangles (n, 3, 4) carried by every group element, clipped to the half-cell (duplicates dropped)."""
    out = []
    if not len(T):
        return np.zeros((0, 3, 4))
    for g in group():
        Y = nrm(T @ g.T)
        ins = (Y > -1e-9).all(-1) & (Y[..., 2] >= Y[..., 3] - 1e-9)
        full = ins.all(1)
        if full.any():
            out.append(Y[full])
        for t in Y[ins.any(1) & ~full]:                                  # (straddling a face: clipped)
            poly = list(t)
            for h in HALF:
                poly = clip(poly, h)
                if len(poly) < 3:
                    break
            if len(poly) >= 3:
                out.append(np.array([[poly[0], poly[k], poly[k + 1]] for k in range(1, len(poly) - 1)]))
    T2 = np.concatenate(out) if out else np.zeros((0, 3, 4))
    seen, keep = set(), []
    for i, t in enumerate(np.round(T2, 7)):
        k = tuple(sorted(map(tuple, t)))
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return T2[keep]


def page_lines(d):
    Ci = np.asarray(d["cellInv"], float)
    out = {}
    for src in ("xlines", "rings", "segments", "main"):
        for x in d.get(src, []):
            if isinstance(x, dict) and x.get("id") and x.get("pts") and not x.get("outline"):
                V = np.array(x["pts"], float) * [1, 1, -1.0]
                out.setdefault(x["id"], []).append(dens(nrm(np.c_[V, np.ones(len(V))] @ Ci.T)))
    return out


def line_images(pieces):
    P = np.vstack(pieces)
    out = []
    for g in group():
        Y = nrm(P @ g.T)
        out.append(Y[(Y > -1e-9).all(1) & (Y[:, 2] >= Y[:, 3] - 1e-9)])
    return np.unique(np.round(np.vstack(out), 6), axis=0)


def base_mesh(w, X, fp):
    tris = []
    if w in X:
        tris += list(np.array(X[w]["tris"], float))
    for p in fp:
        if p["target"] == w:
            C = nrm(np.array(p["corners"], float))
            tris += [np.array([C[0], C[k], C[k + 1]]) for k in range(1, len(C) - 1)]
    if w == "X96":
        PHI = (1 + 5 ** 0.5) / 2
        tris.append(nrm([[0, 1, 1, 0], [0, 1, 1, 1], [1, PHI ** 2, PHI ** 2, 0]]))
    return nrm(np.array(tris)) if tris else np.zeros((0, 3, 4))


def point_images(P):
    """Points carried by every group element, kept where they land in the half-cell."""
    out = []
    for g in group():
        Y = nrm(P @ g.T)
        out.append(Y[(Y > -1e-9).all(1) & (Y[:, 2] >= Y[:, 3] - 1e-9)])
    return np.unique(np.round(np.vstack(out), 6), axis=0) if out else np.zeros((0, 4))


class Wall:
    """A wall's pieces in the half-cell, as points: on it (a curved mesh's vertices, exact on its surface; points spread
    over a flat patch) and on its boundary (a curved mesh's traced outline, a flat patch's sides), all carried by the
    group. A boundary point on a face of the half-cell or on the mirror is a cut of the view."""

    def __init__(self, w, X, fp):
        self.w = w
        on, bd = [], []
        if w in X:
            on.append(nrm(np.array(X[w]["tris"], float).reshape(-1, 4)))
            for line in X[w].get("outline", []):
                P = nrm(np.array([q["beta"] for q in line], float))
                if len(P) > 1:
                    bd.append(dens(P))
        polys = []
        for p in fp:
            if p["target"] != w:
                continue
            C = nrm(np.array(p["corners"], float))
            if w == "T2" and {e[2] for e in p.get("edge_labels") or []} == {"T1", "X5", "C2a"}:
                continue                                  # (T2's former region B and its copy: the wall X96)
            labs = p.get("edge_labels") or [[k, (k + 1) % len(C), None] for k in range(len(C))]
            polys.append((C, [(i_, j_) for i_, j_, lab, *_ in labs if lab != w]))   # (a side on itself: a seam)
        if w == "X96":
            PHI = (1 + 5 ** 0.5) / 2
            C = nrm([[0, 1, 1, 0], [0, 1, 1, 1], [1, PHI ** 2, PHI ** 2, 0]])
            polys.append((C, [(0, 1), (1, 2), (2, 0)]))
        for C, sides in polys:
            for i_, j_ in sides:
                bd.append(dens(np.array([C[i_], C[j_]])))
            for k in range(1, len(C) - 1):
                t = np.array([C[0], C[k], C[k + 1]])
                n = max(1, int(max(np.linalg.norm(t[1] - t[0]), np.linalg.norm(t[2] - t[0])) / 3e-3))
                u, v_ = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
                m = (u + v_) <= 1
                on.append(t[0] + np.outer(u[m], t[1] - t[0]) + np.outer(v_[m], t[2] - t[0]))
        self.T = np.vstack(on) if on else np.zeros((0, 4))
        self.V = point_images(self.T) if len(self.T) else np.zeros((0, 4))
        B = point_images(np.vstack(bd)) if bd else np.zeros((0, 4))
        self.h = float(X[w].get("h", 0.012)) if w in X else 3e-3
        self.treeV = cKDTree(self.V) if len(self.V) else None
        self.face = ((np.abs(B) < 1e-7).any(1) | (np.abs(B[:, 2] - B[:, 3]) < 1e-7)) if len(B) else np.zeros(0, bool)
        self.B = B
        self.treeB = cKDTree(B) if len(B) else None


def main(only=None):
    d = rw.page_data()
    work = json.load(open("working_data.json"))
    X = json.load(open("xsurfaces.json"))
    fp = [p for p in (lambda x: x if isinstance(x, list) else x.values())(json.load(open("fexact_patches.json")))
          if isinstance(p, dict)]
    L0 = page_lines(d)
    need = set(work["walls"]) if not only else set(only) | {e for w in only for e in work["walls"][w]["edges"]}
    walls = {w: Wall(w, X, fp) for w in work["walls"] if w in need}
    Lc, out = {}, {}
    for w, v in work["walls"].items():
        if only and w not in only:
            continue
        W = walls[w]
        if not len(W.T):
            out[w] = {"error": "no mesh"}
            print(f"{w}: no mesh")
            continue
        used = np.zeros(len(W.B), bool)
        rec = {}
        for e in v["edges"]:
            if e in L0:
                if e not in Lc:
                    Lc[e] = line_images(L0[e])
                C, kind = Lc[e], "line"
            elif e in walls and walls[e].treeV is not None and len(W.B):
                C, kind = W.B[walls[e].treeV.query(W.B)[0] < max(ON, 0.7 * walls[e].h)], "wall"
            else:
                rec[e] = {"kind": "?", "status": "no curve (not a drawn line or a meshed wall)"}
                continue
            if not len(C):
                rec[e] = {"kind": kind, "status": "missing: no boundary of the wall lies on it"}
                continue
            dB = W.treeB.query(C)[0] if W.treeB is not None else np.full(len(C), np.inf)
            used |= cKDTree(C).query(W.B)[0] < TOL
            if kind == "wall":
                rec[e] = {"kind": kind, "status": "reached", "points": int(len(C)), "curve": C[:80].round(6).tolist()}
                continue
            on = W.treeV.query(C)[0] < max(ON, 0.7 * W.h)
            if not on.any():
                rec[e] = {"kind": kind, "status": f"short by {dB.min():.4f}",
                          "curve": C[np.argsort(dB)][:40].round(6).tolist()}
                continue
            inner = dB[on] > TOL
            rec[e] = {"kind": kind, "points": int(on.sum()),
                      "status": "reached" if inner.mean() < 0.05 else f"past ({inner.mean():.0%} of it inside a piece)",
                      "curve": C[on][:: max(1, int(on.sum()) // 80)].round(6).tolist(),
                      "inside": C[on][inner][:: max(1, int(inner.sum()) // 20)].round(5).tolist() if inner.any() else []}
        loose = ~used & ~W.face
        out[w] = {"pieces": int(len(W.T)), "edges": rec,
                  "loose": round(float(loose.sum() / max(1, (~W.face).sum())), 3),
                  "loose_sample": W.B[loose][:: max(1, int(loose.sum()) // 10)].round(4).tolist()}
        print(f"{w}: " + "; ".join(f"{e} {r['status']}" for e, r in rec.items())
              + f" | boundary on no edge: {out[w]['loose']}", flush=True)
    old = json.load(open("edge_curves.json")) if only and os.path.exists("edge_curves.json") else {}
    old.update(out)
    json.dump(old, open("edge_curves.json", "w"))


if __name__ == "__main__":
    main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None)
