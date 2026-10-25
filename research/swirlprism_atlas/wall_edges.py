"""Every wall's edges, exactly and in one form: each main piece of a wall as a cycle of edges, each edge with its two
corners, the exact surface it lies on (and how closely its points fit it), the line shape along it and what lies
beyond it. Written to wall_edges.json with an audit of what is missing (wall_edges_audit.txt).

Sources: the curved walls' exact outlines and corners (xsurfaces.json: outline, corner_points), the flat walls' exact
patches (fexact_patches.json: corners, conics), and X96's exact triangle. An edge's surface is found in the exact
library (optimal_seeds.library: the planes of the half-cell under the extended group, every curved wall's surface
and its images, the conic edges) and the half-cell's own faces: the one its points fit best, other than the wall's
own surface. Its line shape is the classification of its middle point with the hull tolerance relaxed.

    python wall_edges.py
"""
from __future__ import annotations

import json
from collections import Counter
from multiprocessing import Pool

import numpy as np

import optimal_seeds as osd
from edge_relabel import lab_edge

PHI = (1 + 5 ** 0.5) / 2
EXACT = 1e-9                # (an edge's points all within this of its surface: the surface is exact for it)
FACES = [osd.Plane(np.eye(4)[k], f"face b{k + 1}=0") for k in range(4)] + [osd.Plane([0, 0, 1, -1.0], "mirror b3=b4")]


def nrm(b):
    b = np.asarray(b, float)
    return b / b.sum(-1, keepdims=True)


def resid(s, P):
    return np.array([abs(s.f(b)[0]) / (np.linalg.norm(s.grad(b)) + 1e-30) for b in P])


def best_surface(P, own, cands):
    """The candidate surface the edge's points fit best (not the wall's own): (surface, max off, median off)."""
    best = None
    for s in cands:
        if s.name in own:
            continue
        r = resid(s, P)
        if best is None or r.max() < best[1]:
            best = (s, float(r.max()), float(np.median(r)))
    return best


def pieces():
    """(wall, piece name, corner list (betas), runs: list of point arrays between corners, own surface names,
    beyond labels per run)."""
    out = []
    X = json.load(open("xsurfaces.json"))
    for tid, sf in X.items():
        corners = [nrm(c[0]) for c in sf.get("corner_points", [])]
        for li, line in enumerate(sf.get("outline", [])):
            P = nrm([q["beta"] for q in line])
            bey = [q.get("beyond") for q in line]
            # (split the closed outline at the corners on it)
            cut = sorted({int(np.argmin(np.linalg.norm(P - c, axis=1))) for c in corners
                          if np.linalg.norm(P - c, axis=1).min() < 1e-6})
            if not cut:
                cut = [0]
            runs, beyond = [], []
            for a, b in zip(cut, cut[1:] + [cut[0] + len(P) - 1]):
                idx = [k % (len(P) - 1) for k in range(a, b + 1)] if len(P) > 1 else [0]
                runs.append(P[idx])
                beyond.append(Counter(bey[k] for k in idx).most_common(1)[0][0])
            out.append((tid, f"outline {li}", [P[c] for c in cut], runs, {tid}, beyond, len(P)))
    d = json.load(open("fexact_patches.json"))
    for p in (d if isinstance(d, list) else d.values()):
        if not isinstance(p, dict) or "corners" not in p:
            continue
        C = nrm(json.loads(p["corners"]) if isinstance(p["corners"], str) else p["corners"])
        n = len(C)
        runs = [np.array([C[k], C[(k + 1) % n]]) for k in range(n)]
        out.append((p["target"], f"patch in {p['wall']}", list(C), runs, {f"{p['target']}:{p['wall']}"},
                    [None] * n, n))
    tri = nrm([[0, 1, 1, 0], [0, 1, 1, 1], [1, PHI ** 2, PHI ** 2, 0]])
    out.append(("X96", "triangle in b2=b3", list(tri), [np.array([tri[k], tri[(k + 1) % 3]]) for k in range(3)],
                {"X96"}, [None] * 3, 3))
    return out


def mid_point(run, s_edge, s_own):
    """The middle of an edge, put back exactly on its edge surface and the wall's own surface where both known."""
    m = run[len(run) // 2] if len(run) > 2 else nrm(run.mean(0))
    cons = [c for c in (s_edge, s_own) if c is not None]
    return osd.home(osd.onto(m, cons)) if cons else m


def main():
    planes, curved = osd.library()
    conics = [c for pcs in osd.conic_pairs().values() for pc in pcs for c in pc if isinstance(c, osd.Conic)]
    cands = FACES + list(planes) + list(curved) + conics
    own_surface = {c.name: c for c in curved if ":" not in c.name}
    table, jobs = [], []
    for tid, name, corners, runs, own, beyond, npts in pieces():
        s_own = own_surface.get(tid)
        own_names = set(own) | ({tid} if s_own else set())
        if s_own is None:                         # (a flat piece: its own plane is not an edge)
            P = np.concatenate(runs)
            n = np.linalg.svd(P)[2][-1]
            own_names |= {s.name for s in planes if abs(abs(n @ s.n / np.linalg.norm(s.n)) - 1) < 1e-9}
        edges = []
        for k, run in enumerate(runs):
            s, mx, md = best_surface(run, own_names, cands)
            exact = mx < EXACT
            m = mid_point(run, s if exact else None, s_own)
            edges.append({"from": corners[k].tolist() if k < len(corners) else None,
                          "to": corners[(k + 1) % len(corners)].tolist() if corners else None,
                          "points": len(run), "surface": s.name, "max_off": mx, "median_off": md, "exact": exact,
                          "beyond": beyond[k], "mid": m.tolist()})
            jobs.append(m)
        table.append({"wall": tid, "piece": name, "outline_points": npts, "corners": [c.tolist() for c in corners],
                      "edges": edges})
    with Pool(3) as pool:
        labs = pool.map(lab_edge, jobs)
    it = iter(labs)
    for t in table:
        for e in t["edges"]:
            e["line"] = next(it)
    json.dump(table, open("wall_edges.json", "w"), indent=1)
    # (the audit)
    walls = sorted({t["wall"] for t in table}, key=lambda w: (w[0], int("".join(ch for ch in w[1:] if ch.isdigit()) or 0)))
    lines = []
    bad = 0
    for w in walls:
        ts = [t for t in table if t["wall"] == w]
        E = [e for t in ts for e in t["edges"]]
        inexact = [e for e in E if not e["exact"]]
        unnamed = [e for e in E if e["line"] in ("ERR", w) or str(e["line"]).startswith("new:")]
        small = [t for t in ts if t["outline_points"] < 12 and t["piece"].startswith("outline")]
        ok = not inexact and not unnamed and not small
        bad += not ok
        lines.append(f"{w:4s} {'ok ' if ok else 'GAP'} pieces {len(ts)}, edges {len(E)}: lines "
                     f"{sorted({str(e['line'])[:40] for e in E})}"
                     + ("; not exact: " + str([(e["surface"], "%.1e" % e["max_off"]) for e in inexact]) if inexact else "")
                     + (f"; unnamed line on {len(unnamed)} edge(s): {[str(e['line'])[:60] for e in unnamed]}" if unnamed else "")
                     + (f"; {len(small)} stray small outline loop(s)" if small else ""))
    lines.append(f"{len(walls)} walls, {bad} with gaps")
    open("wall_edges_audit.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
