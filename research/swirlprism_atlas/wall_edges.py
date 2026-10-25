"""Every wall's edges, exactly and in one form (wall_edges.json, audit in wall_edges_audit.txt).

Each main piece of a wall is a cycle of edges, each edge either
  * a cut of the view: it lies on a face of the half-cell or on the splitting mirror; the wall goes on across it on
    the 3-sphere (into an image of itself), so it is a seam, not an end of the wall; or
  * a true edge: the wall ends there, against the exact surface it lies on (another wall's surface, an image of its
    own, a plane or a conic edge of the library), with what lies beyond it.
Each true edge is checked: from its middle, a step along the wall's own surface each way across it must give the
wall on one side and something else on the other.

Sources: the curved walls' exact outlines (xsurfaces.json; runs split at their corners and where what lies beyond
changes), the flat walls' exact patches (fexact_patches.json; their sides), X96's exact triangle.

    python wall_edges.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np

import optimal_seeds as osd
from probe import _one

PHI = (1 + 5 ** 0.5) / 2
EXACT = 1e-8                # (an edge's points within this of a surface (|f| over |grad f|): it lies on it)
ON_FACE = 1e-5              # (an edge this close to a face of the half-cell or the mirror lies on it)
STEP = 2e-4
LIB = None


def nrm(b):
    b = np.asarray(b, float)
    return b / b.sum(-1, keepdims=True)


def vres(s, B, e=1e-7):
    B = np.atleast_2d(B)
    f = s.f(B)
    g = np.stack([(s.f(B + e * np.eye(4)[k]) - s.f(B - e * np.eye(4)[k])) / (2 * e) for k in range(4)], 1)
    return np.abs(f) / (np.linalg.norm(g, axis=1) + 1e-30)


def face_of(B):
    """The face of the half-cell or the mirror all the points lie on, or None."""
    for k in range(4):
        if np.abs(B[:, k]).max() < ON_FACE:
            return f"face b{k + 1}=0"
    if np.abs(B[:, 2] - B[:, 3]).max() < ON_FACE:
        return "mirror b3=b4"
    return None


def fit(B, own):
    """The library surface the points lie on best (not the wall's own): (surface, max off)."""
    planes, curved, conics, PN = LIB
    rp = np.abs(B @ PN.T).max(0)
    j = int(np.argmin([r if planes[i].name not in own else np.inf for i, r in enumerate(rp)]))
    best = (planes[j], float(rp[j]))
    for s in list(curved) + conics:
        if s.name in own:
            continue
        m = float(vres(s, B).max())
        if m < best[1]:
            best = (s, m)
    return best


def runs_of_outline(line, corners):
    """A curved wall's closed outline split at its corners and where what lies beyond it changes (a change over
    fewer than 4 points is a blip)."""
    P = nrm([q["beta"] for q in line])
    bey = [q.get("beyond") for q in line]
    cut = {0}
    for c in corners:
        d = np.linalg.norm(P - c, axis=1)
        if d.min() < 1e-6:
            cut.add(int(d.argmin()))
    k = 0
    while k < len(P):
        j = k
        while j < len(P) and bey[j] == bey[k]:
            j += 1
        if j - k >= 4 and k > 0:
            cut.add(k)
        k = j
    cut = sorted(cut)
    out = []
    for a, b in zip(cut, cut[1:] + [len(P) - 1]):
        if b - a >= 2:
            lab = max(set(bey[a:b + 1]), key=bey[a:b + 1].count)
            out.append((P[a:b + 1], lab))
    return out


def pieces():
    X = json.load(open("xsurfaces.json"))
    out = []
    for tid, sf in X.items():
        corners = [nrm(c[0]) for c in sf.get("corner_points", [])]
        for li, line in enumerate(sf.get("outline", [])):
            if len(line) <= 15:
                out.append({"wall": tid, "piece": f"outline {li}", "stray": len(line)})
                continue
            out.append({"wall": tid, "piece": f"outline {li}", "own": tid, "runs": runs_of_outline(line, corners)})
    d = json.load(open("fexact_patches.json"))
    for p in (d if isinstance(d, list) else d.values()):
        if not isinstance(p, dict) or "corners" not in p:
            continue
        C = nrm(json.loads(p["corners"]) if isinstance(p["corners"], str) else p["corners"])
        if p["target"] == "T2" and len(C) == 3 and abs(np.asarray(p["normal"], float) @ [0, 1, -1, 0]) > 1.4:
            continue                                  # (T2's former region B, in b2 = b3: the wall X96 now)
        out.append({"wall": p["target"], "piece": f"patch in {p['wall']}", "own": None, "normal": p["normal"],
                    "conics": [f"conic {c['label']} in {p['wall']}" for c in p.get("conics") or []],
                    "runs": [(np.array([C[k], C[(k + 1) % len(C)]]), None) for k in range(len(C))]})
    tri = nrm([[0, 1, 1, 0], [0, 1, 1, 1], [1, PHI ** 2, PHI ** 2, 0]])
    out.append({"wall": "X96", "piece": "triangle in b2=b3", "own": None, "normal": [0, 1, -1, 0],
                "runs": [(np.array([tri[k], tri[(k + 1) % 3]]), None) for k in range(3)]})
    return out


JOBS = []


def across(i):
    """From the edge's middle m, a step along the wall each way across the edge: (one side, the other)."""
    m, own, edge = JOBS[i]
    S = [own, edge] if own is not None else [edge]
    q = osd.onto(m, [s for s in S if s is not None])
    T = osd.tangent(q, [own]) if own is not None else None
    D = osd.tangent(q, [s for s in S if s is not None])
    if T is None or not len(D) or not len(T):
        return "?", "?"
    d = D[0]
    u = T[0] - (T[0] @ d) * d if len(T) else None
    if len(T) > 1 and np.linalg.norm(u) < 1e-6:
        u = T[1] - (T[1] @ d) * d
    u = u / np.linalg.norm(u)
    lab = lambda x: _one(osd.home(osd.onto(x, [own])))[0]
    return lab(q + STEP * u), lab(q - STEP * u)


def main():
    global LIB
    planes, curved = osd.library()
    planes = [osd.Plane(np.eye(4)[k], f"face b{k + 1}=0") for k in range(4)] + \
        [osd.Plane([0, 0, 1, -1.0], "mirror b3=b4")] + list(planes)
    conics = [c for pcs in osd.conic_pairs().values() for pc in pcs for c in pc if isinstance(c, osd.Conic)]
    PN = np.array([p.n for p in planes], float)
    PN = PN / np.linalg.norm(PN, axis=1, keepdims=True)
    LIB = (planes, curved, conics, PN)
    by_name = {c.name: c for c in curved}
    table, jobs, where = [], [], []
    for pc in pieces():
        if "stray" in pc:
            table.append(pc)
            continue
        own_s, own = None, set()
        if pc["own"]:
            # (the surface the wall lies on: its own, or the one it shares with another wall (X47 on X14's))
            P = np.concatenate([R for R, _ in pc["runs"]])[::7]
            own_s = by_name.get(pc["own"]) or min(curved, key=lambda c: float(vres(c, P).max()))
            own = {own_s.name, pc["own"]}
        if pc.get("normal") is not None:              # (a flat piece: its own plane is not an edge)
            n = np.asarray(pc["normal"], float)
            n = n / np.linalg.norm(n)
            own |= {p.name for p in planes if abs(abs(n @ (p.n / np.linalg.norm(p.n))) - 1) < 1e-9}
            own_s = osd.Plane(n, "own plane")
        runs = pc["runs"]
        if pc.get("normal") is not None and len(runs) > 6:
            # (a curved side given as many short chords: chords lying on one surface joined into one edge)
            # (a curved side's points are a drawing of its conic, kept to ~1e-4: a run of them within 2e-3 of one of
            # the patch's conics is that conic, exactly, between its two ends)
            def on_conic(Q):
                best_c = None
                for c in [c for c in conics if c.name in pc.get("conics", ())]:
                    r = float(vres(c, Q).max())
                    if r < 2e-3 and (best_c is None or r < best_c[1]):
                        best_c = (c, r)
                return best_c
            joined = []
            for R, lab in runs:
                if joined:
                    Q = np.vstack([joined[-1][0], R[1:]])
                    if (face_of(Q) or fit(Q, own)[1] < EXACT or (len(Q) >= 3 and on_conic(Q))):
                        joined[-1] = (Q, lab)
                        continue
                joined.append((R, lab))
            runs = joined
        edges = []
        for R, lab in runs:
            mid = R[len(R) // 10: len(R) - len(R) // 10] if len(R) > 20 else R
            face = face_of(R)
            if face:
                e = {"kind": "cut of the view", "on": face, "points": len(R), "beyond": lab}
            else:
                cc = None
                if pc.get("normal") is not None and len(R) >= 3:
                    for c in [c for c in conics if c.name in pc.get("conics", ())]:
                        r = float(vres(c, R).max())
                        if r < 2e-3 and (cc is None or r < cc[1]):
                            cc = (c, r)
                if cc is not None:                    # (a conic side: exact by its equation; its points a drawing)
                    s, off = cc[0], 0.0
                elif len(R) == 2:                     # (a straight side: the plane through it, judged all along it)
                    L = R[0] + np.linspace(0, 1, 9)[:, None] * (R[1] - R[0])
                    planes_, _, _, PN_ = LIB
                    rp = np.abs(nrm(L) @ PN_.T).max(0)
                    j = int(np.argmin([r if planes_[i].name not in own else np.inf for i, r in enumerate(rp)]))
                    s, off = planes_[j], float(rp[j])
                else:
                    s, off = fit(mid, own)
                    if off >= EXACT:                  # (a traced curve: exact when its points sit on it to 1e-7)
                        q = np.array([osd.onto(x, [own_s, s]) if own_s is not None else x for x in mid[::max(1, len(mid) // 8)]])
                        moved = float(np.linalg.norm(q - mid[::max(1, len(mid) // 8)], axis=1).max())
                        if moved < 1e-7:
                            off = 0.0
                e = {"kind": "edge", "surface": s.name, "max_off": off, "exact": off < EXACT, "points": len(R),
                     "beyond": lab}
                m = R[len(R) // 2] if len(R) > 2 else nrm(R.mean(0))
                jobs.append((m, own_s, s))
                where.append(e)
            e["from"], e["to"] = R[0].round(10).tolist(), R[-1].round(10).tolist()
            edges.append(e)
        table.append({"wall": pc["wall"], "piece": pc["piece"], "edges": edges})
    JOBS[:] = jobs                                    # (the workers read them by forking)
    with Pool(3) as pool:
        res = pool.map(across, range(len(jobs)))
    for e, (a, b) in zip(where, res):
        e["across"] = [a, b]
    # (the audit: per wall, its true edges, and what does not make sense)
    lines, n_bad = [], 0
    walls = []
    for t in table:
        if t["wall"] not in walls:
            walls.append(t["wall"])
    for w in walls:
        ts = [t for t in table if t["wall"] == w]
        E = [e for t in ts for e in t.get("edges", [])]
        true = [e for e in E if e["kind"] == "edge"]
        prob = []
        true_ = true
        for e in true:
            if not e["exact"]:
                prob.append(f"edge (beyond {e['beyond']}) not exact: best {e['surface']} {e['max_off']:.0e}")
            a, b = e.get("across", ["?", "?"])
            if a == w and b == w:
                e["kind"] = "seam"                    # (between two pieces of the wall: it goes on across it)
                continue
            if not ((a == w) ^ (b == w)):
                prob.append(f"edge on {e['surface'][:28]} (beyond {e['beyond']}): across it {a}|{b}, not the wall "
                            f"on one side only")
        stray = [t for t in ts if "stray" in t]
        if stray:
            prob.append(f"{len(stray)} stray loop(s) (dropped)")
        n_bad += bool(prob)
        seams = sum(e["kind"] == "seam" for e in E)
        lines.append(f"{w:4s} {'ok ' if not prob else 'GAP'} {len(true) - seams} edges, {seams} seams, "
                     f"{sum(e['kind'] == 'cut of the view' for e in E)} cuts of the view"
                     + (": " + "; ".join(prob) if prob else ""))
    lines.append(f"{len(walls)} walls, {n_bad} with something to fix")
    open("wall_edges_audit.txt", "w").write("\n".join(lines) + "\n")
    json.dump(table, open("wall_edges.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
