"""Is every region of the octahedral swirl domain closed off by recorded walls? (region_closure_oct.json)

Each piece of a region is exact (its sign conditions, region_forms_oct.json), so where a piece ends is exact too. From
points of each piece (a regular grid in the betas, and the piece's example seed), walks go out along a fixed, evenly
spread set of directions in small steps until the piece ends; the end is bisected to 1e-12, and there:
  * the surface: the factor (region_forms' surfaces) that vanishes at the point;
  * beyond: the piece a small step past it, or the domain's edge, or no piece at all (a gap in the model).
An end between two different shapes on a surface that walls_verified_oct.json does not record as a wall between
those two shapes is a piece of wall missing; an end into no piece is a piece of region missing. An end between two
pieces of one shape is a relabelling, not a wall.

    SWIRL_GROUP=oct python region_closure.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

import atlas_model as am
import domain as D

OUT = f"region_closure_{D.KEY}.json"
STEP0, NEAR, STEP1, MAXLEN = 2e-4, 0.01, 1e-3, 1.2


def grid(N):
    pts = []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            for k in range(N + 1 - i - j):
                pts.append([i, j, k, N - i - j - k])
    B = np.array(pts, float) / N
    return B[am.in_domain(B)]


def directions(n):
    U = np.linalg.svd(np.ones((1, 4)))[2][1:]
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([r * np.cos(a), r * np.sin(a), z], 1) @ U


def spread(P, n):
    """Up to n of the points P, spread out (each the farthest from those chosen)."""
    out = [P[0]]
    while len(out) < min(n, len(P)):
        d = np.min([np.linalg.norm(P - q, axis=1) for q in out], axis=0)
        out.append(P[int(d.argmax())])
    return out


def main():
    model = am.MODEL
    surfs = [am.Surf(s["factor"]) for s in model["surfaces"]]
    ver = json.load(open(f"walls_verified_{D.KEY}.json"))["walls"]
    recorded = defaultdict(set)                     # (surface text) -> {frozenset of the two shapes}
    for w in ver:
        if w["verdict"] == "wall":
            for pair in w["separates"]:
                recorded[w["surface"]].add(frozenset(pair.split("|")))
    G = grid(60)
    gp = am.piece_at(G)
    D_ = directions(60)
    ts = np.concatenate([np.arange(STEP0, NEAR, STEP0), np.arange(NEAR, MAXLEN, STEP1)])
    res = {}
    for pi, p in enumerate(am.PIECES):
        starts = list(G[gp == pi])
        ex = np.asarray(json.loads(model["pieces"][p]["example_beta"]), float) if isinstance(model["pieces"][p]["example_beta"], str) \
            else np.asarray(model["pieces"][p]["example_beta"], float)
        ex = ex / ex.sum()
        starts = spread(np.array([ex] + starts), 6)
        starts = [s for s in starts if am.piece_at(s[None])[0] == pi]
        ends = Counter()
        where = {}
        allpts = {}
        for s in starts:
            for d in D_:
                R = s[None] + ts[:, None] * d[None]
                inside = am.in_domain(R)
                P = am.piece_at(R)
                k = int(np.argmax((P != pi) | ~inside)) if ((P != pi) | ~inside).any() else None
                if k is None:
                    ends[("not ended", "", "")] += 1
                    continue
                if not inside[k]:
                    ends[("domain edge", "", "")] += 1
                    continue
                lo, hi = (s if k == 0 else R[k - 1]), R[k]
                for _ in range(45):
                    m = (lo + hi) / 2
                    if am.piece_at(m[None])[0] == pi:
                        lo = m
                    else:
                        hi = m
                x = (lo + hi) / 2
                beyond = am.piece_at((x + 1e-7 * d)[None])[0]
                vals = np.array([abs(float(sf.val(x)[0])) / (np.linalg.norm(sf.grad(x)[0] - sf.grad(x)[0].mean()) + 1e-300)
                                 for sf in surfs])
                si = int(vals.argmin())
                surf = model["surfaces"][si]["factor"] if vals[si] < 1e-8 else f"none within 1e-8 (nearest {vals[si]:.0e})"
                if beyond < 0:
                    kind = "no piece beyond (gap)"
                    b = ""
                else:
                    b = am.PIECES[beyond]
                    a_t, b_t = am.TYPE[p], am.TYPE[b]
                    if a_t == b_t:
                        kind = "same shape (relabelling)"
                    elif frozenset((a_t, b_t)) in recorded.get(surf, set()):
                        kind = "recorded wall"
                    elif surf in recorded:
                        kind = "wall surface, but not recorded between these two"
                    else:
                        kind = "surface not recorded as a wall"
                key = (kind, surf, b)
                ends[key] += 1
                where.setdefault(key, x.round(8).tolist())
                allpts.setdefault(key, []).append(x.round(10).tolist())
        res[p] = {"type": am.TYPE[p], "starts": len(starts),
                  "ends": [{"kind": k[0], "surface": k[1], "beyond": k[2], "beyond_type": am.TYPE.get(k[2], ""),
                            "walks": n, "at": where.get(k), "points": allpts.get(k, [])} for k, n in ends.most_common()]}
        bad = [e for e in res[p]["ends"] if e["kind"] not in ("recorded wall", "domain edge", "same shape (relabelling)")]
        print(f"{p} ({am.TYPE[p]}): {len(starts)} starts; " +
              ("closed by recorded walls" if not bad else "OPEN: " + "; ".join(
                  f"{e['kind']} on {e['surface'][:40]} to {e['beyond']} x{e['walks']}" for e in bad)), flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "pieces": res}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
