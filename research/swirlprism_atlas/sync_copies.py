"""Copies kept in sync with their main pieces: each copy section takes the names of the main section its points are
images of. A copy (a curved wall's copy mesh, a flat wall's copy patch, T2's copy patches) is its wall carried by
elements of the extended group, folded into the half-cell; one copy piece can hold images of several main pieces (a
main piece cut off by a face of the cell continues, glued, as another), so each point is matched on its own: carried
by every element, it is looked for inside the main pieces' sections (exactly: within the section's triangles, on its
surface); the names of the section it lands in are the copy section's, by majority over its points. A line on a copy
that is only the image of a cell face separates nothing: the sections either side get the same names.

    python sync_copies.py [--dry] [walls]     (rewrites sections/<wall>.json.gz; prints what changed)
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

import region_cells as rc
import region_walls as rw
from cut_walls import FOLD, sample_tris
from t2_pieces import extended

OUT = "sections"
BEYOND = "(beyond the half-cell)"


def is_copy_piece(d, piece):
    _, kind, i, _ = piece
    if kind == "copy":
        return True
    if kind == "orig":
        return bool(d["xwalls"][i].get("copy"))
    if kind == "tpatch":
        return bool(d["tpatches"][i].get("copy"))
    return False


def inside_tris(X, T, tree, R, tol):
    """For each point, the index of a triangle of T it lies in (within tol of it), else -1."""
    out = np.full(len(X), -1)
    for i, x in enumerate(X):
        for t in tree.query_ball_point(x, R):
            a, b, c = T[t]
            e1, e2 = b - a, c - a
            n = np.cross(e1, e2)
            nn = np.linalg.norm(n)
            if nn < 1e-14:
                continue
            n /= nn
            dpl = (x - a) @ n
            if abs(dpl) > tol:
                continue
            y = x - dpl * n - a
            M = np.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]])
            u, v = np.linalg.solve(M, [y @ e1, y @ e2])
            if u > -1e-6 and v > -1e-6 and u + v < 1 + 1e-6:
                out[i] = t
                break
    return out


def _counts(f):
    return [{k: v for k, v in f.get("checked", {}).get(str(sd), {}).items() if not k.startswith("(")} for sd in (0, 1)]


def mixed(f):
    """The classifier met more than one region on a side of the section."""
    return any(len(c) > 1 for c in _counts(f))


def whole(f):
    """The classifier met one region on each side of the section, the one it is named with."""
    c = _counts(f)
    return all(len(x) == 1 for x in c) and [next(iter(x)) for x in c] == list(f["sides"])


def main(dry=False, walls=None):
    d = rw.page_data()
    pieces = rw.pieces_of(d, {w["id"] for w in d["xwalls"]} | {"T2"})
    G = extended()
    GG = np.array([F @ M for M in G for F in (np.eye(4), FOLD)])
    files = sorted(glob.glob(f"{OUT}/*.json.gz"))
    for fpath in files:
        w = os.path.basename(fpath).split(".")[0]
        if walls and w not in walls:
            continue
        faces = json.load(gzip.open(fpath, "rt"))
        cp = [k for k, f in enumerate(faces) if is_copy_piece(d, pieces[f["piece"]])]
        mn = [k for k, f in enumerate(faces) if k not in cp and BEYOND not in f["sides"]]
        if not cp or not mn:
            continue
        T0 = np.concatenate([np.array(faces[k]["tris"]).reshape(-1, 3, 3) for k in mn])
        lab0 = np.concatenate([[k] * (len(faces[k]["tris"])) for k in mn])
        C0 = T0.mean(1)
        R = float(np.linalg.norm(T0 - C0[:, None], axis=2).max()) + 1e-9
        tree = cKDTree(C0)
        curved = any(pieces[faces[k]["piece"]][1] == "copy" for k in cp) or w not in {p[0] for p in pieces if p[1] == "orig"} \
            or len(T0) > 400
        tol = 3e-4 if curved else 1e-7
        changed = False
        for k in cp:
            f = faces[k]
            if BEYOND in f["sides"] or f["area"] < 1e-5:
                continue
            T = np.array(f["tris"]).reshape(-1, 3, 3)
            X = sample_tris(T, max(0.002, np.sqrt(f["area"]) / 6))
            X = X[np.random.default_rng(0).choice(len(X), min(len(X), 24), replace=False)]
            Y = np.einsum("gij,nj->gni", GG, rc.betas(X))
            s = Y.sum(2, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                B = Y / s
            ok = (np.abs(s[..., 0]) > 1e-12) & (B.min(2) > -1e-9) & (B[..., 2] >= B[..., 3] - 1e-9)
            votes = Counter()
            for g in np.flatnonzero(ok.any(1)):
                pts = np.flatnonzero(ok[g])
                Xg = rc.xyz(B[g, pts])
                hit = inside_tris(Xg, T0, tree, R, tol)
                for h in hit[hit >= 0]:
                    votes[int(lab0[h])] += 1
            if not votes:
                continue
            top, n_top = votes.most_common(1)[0]
            src = faces[top]["sides"]
            share = n_top / sum(votes.values())
            # (a main section the classifier found mixed, part one region and part another on a side, is short of
            # a cut; a copy the classifier found whole on both sides keeps its own names rather than take the mixed
            # main's majority)
            if set(src) != set(f["sides"]) and mixed(faces[top]) and whole(f):
                print(f"  {w} copy section {k} (piece {f['piece']}): keeps {'|'.join(f['sides'])}; its main section "
                      f"{top} ({'|'.join(src)}) is mixed: {faces[top].get('checked')}")
                continue
            if set(src) != set(f["sides"]):
                print(f"  {w} copy section {k} (piece {f['piece']}, {f['area']:.4f}): {'|'.join(f['sides'])} -> "
                      f"{'|'.join(src)}  (lands in main section {top}, {share:.0%} of {sum(votes.values())} hits)")
                if not dry:
                    f.setdefault("own_sides", f["sides"])
                    f["sides"], f["named"] = list(src), "as its main piece"
                    changed = True
            elif share < 0.8:
                print(f"  {w} copy section {k}: {'|'.join(f['sides'])} agrees, but only {share:.0%} of hits in one section")
        if changed:
            json.dump(faces, gzip.open(fpath, "wt"), separators=(",", ":"))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry"]
    main(dry="--dry" in sys.argv, walls=args[0].split(",") if args else None)
