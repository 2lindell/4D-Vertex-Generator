"""Exact curves where walls meet walls, from the atlas's exact definitions, checked on the map.

`exact_defs.json` gives, for each wall W and each wall E it meets (an edge of W on E, or a cut of W
by E), the exact surface that meets W along the curve. So the curve is W's own surface met by that
surface: two exact equations. Each such curve is traced exactly through the domain (every point
solved onto both surfaces; ends solved onto the domain's face where it leaves). At points all along
it, the map's exact incidence (`incidence.Incidence.round_line`) names what surrounds it. The
meeting is confirmed where both W and E are named round the curve. The stretch where they are has
its ends compared with the atlas's exact meeting points (`exact_points.json`).
"""
from __future__ import annotations

import json
import re

import numpy as np

from .chords import FACE_N
from .frame import data_dir, load_json, point_from_beta, source_dir, surface_by_name, surfaces
from .incidence import Incidence
from .surface_images import image_of

STEP = 2e-3
CHECK_EVERY = 10          # incidence checked every CHECK_EVERY steps (~2e-2)
NEAR_POINT = 1e-2         # (R^4) points this close to a named point are not checked: corners there


def resolve(name: str):
    """A surface by its atlas name: a library name, or 'BASE:map[:fold]' (BASE's image under map)."""
    by = surface_by_name()
    if name in by:
        return surfaces()[by[name]]
    m = re.fullmatch(r"(.+?):(\d+)(:fold)?", name)
    if m:
        base = m.group(1) + (m.group(3) or "")
        if base not in by:              # a wall's name for its own (single) surface
            own = load_json("exact_defs.json")["walls"].get(m.group(1), {}).get("surface", [])
            if len(own) == 1:
                base = own[0] + (m.group(3) or "")
        if base in by:
            return image_of(surfaces()[by[base]], int(m.group(2)), name)
    raise KeyError(name)


def _g(s, x):
    g = s.grad(x[None])[0]
    return g - g.mean()


def _solve(fs, x, iters=30):
    """x moved onto every f in fs (each (value, gradient) of x), keeping sum(beta) = 1."""
    x = np.asarray(x, float).copy()
    for _ in range(iters):
        v = np.array([f(x)[0] for f in fs])
        if np.abs(v).max() < 1e-15:
            break
        J = np.array([f(x)[1] for f in fs])
        x = x + np.linalg.lstsq(np.vstack([J, np.ones(4)]), np.r_[-v, 1.0 - x.sum()], rcond=None)[0]
    ok = np.abs([f(x)[0] for f in fs]).max() < 1e-12
    return x, ok


def _sf(s):
    return lambda x: (float(s.f(x[None])[0]), _g(s, x))


def _face(n):
    return lambda x: (float(n @ x), n - n.mean())


def _inside(x, tol=1e-12):
    return bool((x @ FACE_N.T >= -tol).all())


def trace(a, b, n_seeds=3000, rng=None):
    """Every arc of the curve a = b = 0 in the domain, as polylines in beta (ends on the domain's
    faces, solved exactly there), or closed loops."""
    rng = rng or np.random.default_rng(0)
    fa, fb = _sf(a), _sf(b)
    seeds = []
    for x0 in rng.dirichlet(np.ones(4), n_seeds):
        x0[2], x0[3] = max(x0[2], x0[3]), min(x0[2], x0[3])
        x, ok = _solve([fa, fb], x0)
        if ok and _inside(x, 1e-9):
            seeds.append(x)
    arcs, done = [], np.zeros(len(seeds), bool)
    seeds = np.array(seeds) if seeds else np.zeros((0, 4))
    for i in range(len(seeds)):
        if done[i]:
            continue
        halves = []
        closed = False
        for sgn in (1, -1):
            pts, x = [seeds[i]], seeds[i]
            t_prev = None
            for _ in range(5000):
                t = np.linalg.svd(np.vstack([np.ones(4), fa(x)[1], fb(x)[1]]))[2][-1]
                if t_prev is None:
                    t = sgn * t
                elif t @ t_prev < 0:
                    t = -t
                t_prev = t
                y, ok = _solve([fa, fb], x + STEP * t / np.linalg.norm(t))
                if not ok:
                    break
                if not _inside(y):
                    # leave through a face: solve onto it
                    k = int(np.argmin(y @ FACE_N.T))
                    z, ok = _solve([fa, fb, _face(FACE_N[k])], x)
                    if ok and _inside(z, 1e-9):
                        pts.append(z)
                    break
                if len(pts) > 20 and np.linalg.norm(y - seeds[i]) < 1.5 * STEP:
                    closed = True
                    pts.append(seeds[i])
                    break
                pts.append(y)
                x = y
            halves.append(pts)
            if closed:
                break
        poly = np.array(halves[0][::-1] + (halves[1][1:] if len(halves) > 1 else []))
        d = np.min(np.linalg.norm(seeds[:, None] - poly[None], axis=2), axis=1)
        done |= d < 3 * STEP
        arcs.append({"points": poly, "closed": closed})
    return arcs


def meetings():
    """Every wall-wall meeting the exact definitions record: (wall, other wall, kind, surface)."""
    d = load_json("exact_defs.json")
    walls = set(d["walls"])
    out = []
    for w, v in d["walls"].items():
        for kind in ("edges", "cuts"):
            for e, rec in v.get(kind, {}).items():
                if e in walls:
                    out.append({"wall": w, "other": e, "kind": kind[:-1],
                                "surface": rec.get("surface"), "status": rec.get("status"),
                                "between": rec.get("between")})
    return out


def check(meet, inc, wall_surfs, named):
    """Trace a meeting's curve(s) and name what surrounds them all along."""
    w, e = meet["wall"], meet["other"]
    if meet["surface"]:
        pairs = [(a, meet["surface"]) for a in wall_surfs[w]]
    else:                               # unverified in the data: W's surfaces met by E's
        pairs = [(a, b) for a in wall_surfs[w] for b in wall_surfs[e]]
    found = []
    for an, bn in pairs:
        a, b = resolve(an), resolve(bn)
        if a.kind == b.kind == "plane" and abs(abs(a.normal @ b.normal) - 1) < 1e-12:
            continue
        for arc in trace(a, b):
            pts = arc["points"]
            run, runs = [], []
            for j in range(1, len(pts) - 1, CHECK_EVERY):
                x = pts[j]
                if np.min(np.linalg.norm(named[:, 1:] - point_from_beta(x), axis=1)) < NEAR_POINT:
                    continue
                r = inc.round_line(x, pts[j + 1] - pts[j - 1])
                names = set() if r is None else {wl["wall"] for wl in r["walls"] if wl["wall"]}
                here = w in names and e in names
                if here:
                    run.append((j, names, r))
                elif run:
                    runs.append(run)
                    run = []
            if run:
                runs.append(run)
            for run in runs:
                j0, j1 = run[0][0], run[-1][0]
                regions = sorted({s["region"] for _, _, r in run for s in r["sectors"]
                                  if s["region"]})
                others = sorted(set().union(*(n for _, n, _ in run)) - {w, e})
                found.append({"surfaces": [an, bn], "from": _nearest(pts[j0], named),
                              "to": _nearest(pts[j1], named), "checked_points": len(run),
                              "regions_round": regions, "other_walls_round": others,
                              "arc_ends": [_nearest(pts[0], named), _nearest(pts[-1], named)]})
    return found


def _nearest(x, named):
    p = point_from_beta(x)
    d = np.linalg.norm(named[:, 1:] - p, axis=1)
    i = int(np.argmin(d))
    return {"beta": [round(float(t), 6) for t in x], "nearest_point": NAMES[int(named[i, 0])],
            "distance": float(d[i])}


NAMES: list[str] = []


def run() -> dict:
    inc = Incidence()
    d = load_json("exact_defs.json")
    wall_surfs = {w: v["surface"] for w, v in d["walls"].items()}
    pts = json.loads((source_dir() / "exact_points.json").read_text())["points"]
    NAMES[:] = sorted({p["name"] for p in pts})
    named = np.array([[NAMES.index(p["name"]), *point_from_beta(np.asarray(p["float"], float))]
                      for p in pts])
    out = []
    for m in meetings():
        res = check(m, inc, wall_surfs, named)
        out.append({**m, "found": res})
        ends = "; ".join(f"{r['from']['nearest_point']}({r['from']['distance']:.0e})"
                         f" -> {r['to']['nearest_point']}({r['to']['distance']:.0e}) n={r['checked_points']}"
                         for r in res) or "NOT FOUND"
        print(f"{m['wall']:4s} {m['kind']:4s} {m['other']:4s} [{(m['status'] or '')[:10]}] {ends}",
              flush=True)
    (data_dir() / "sphere_map" / "wall_meets.json").write_text(json.dumps(
        {"_about": __doc__, "meetings": out}, indent=1))
    return out


if __name__ == "__main__":
    run()
