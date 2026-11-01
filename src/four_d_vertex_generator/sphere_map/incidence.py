"""Exact incidence of lines and walls: what surrounds each line, read from the exact surfaces.

At a point x of a line, the library surfaces through x are found and x is solved onto all of them.
In the plane normal to the line each such surface is a straight line through x (its gradient
fixes it), so the directions round the line split into sectors in an exact order; each sector has
an exact sign on every surface through x, and with the signs of all the other surfaces at x that
names a piece of the arrangement, and so a region. Between two neighbouring sectors lies one
surface, and the wall there is the wall patch between those two pieces (named from the working
data). Nothing is sampled at a distance from the line: the only lookup is of pieces by sign.
A sector whose piece the exploration never met is reported as unknown.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from .chords import FACE_N
from .frame import data_dir, load_json, locate, point_from_beta


def _g0(s, x):
    g = s.grad(x[None])[0]
    return g - g.mean()


def _solve_onto(surfs, idx, x, iters=20):
    x = np.asarray(x, float).copy()
    for _ in range(iters):
        f = np.array([float(surfs[i].f(x[None])[0]) for i in idx])
        if np.abs(f).max() < 1e-15:
            break
        J = np.array([_g0(surfs[i], x) for i in idx])
        x = x + np.linalg.lstsq(np.vstack([J, np.ones(4)]), np.r_[-f, 0.0], rcond=None)[0]
    return x


class Incidence:
    def __init__(self, root=None):
        from .atlas_db import _library_surfaces, _piece_keys

        self.root = root or data_dir() / "sphere_map"
        self.surfs = _library_surfaces(self.root)
        self.keys = _piece_keys(self.root)
        pieces = json.loads((self.root / "pieces.json").read_text())["pieces"]
        self.region = [p["region"] for p in pieces]
        patches = json.loads((self.root / "patches.json").read_text())["patches"]
        self.patch_wall = {}
        for p in patches:
            if p["face"] is None and p["wall"]:
                a, b = p["pieces"]
                self.patch_wall[(min(a, b), max(a, b), p["surface"])] = p["wall"]
        lib = json.loads((self.root / "surface_library.json").read_text())["surfaces"]
        self.family = [d["family"] for d in lib]
        from .build import wall_naming
        self.naming = wall_naming()
        self.wall_surfaces = self._wall_surfaces()
        self.wall_edges = {k: set(v["edges"]) for k, v in load_json("working_data.json")["walls"].items()}
        self.line = None                     # the atlas line being checked, if any
        self.planes = np.array([s.normal if s.kind == "plane" else np.zeros(4) for s in self.surfs])
        self.is_plane = np.array([s.kind == "plane" for s in self.surfs])

    def values(self, x):
        v = np.empty(len(self.surfs))
        v[self.is_plane] = self.planes[self.is_plane] @ x
        for i in np.flatnonzero(~self.is_plane):
            v[i] = self.surfs[i].f(x[None])[0]
        return v

    def through(self, x, tol):
        v = self.values(x)
        sc = np.array([np.linalg.norm(_g0(s, x)) for s in self.surfs]) + 1e-300
        return [int(i) for i in np.flatnonzero(np.abs(v) / sc < tol)]

    def round_line(self, x, tangent, tol=1e-6):
        """Sectors round the line at x: [(sector direction angle, piece, region)], and the walls
        between neighbouring sectors [(surface, region a, region b, wall)]."""
        x = x / x.sum()
        on = self.through(x, tol)
        if len(on) < 2:
            return None
        x = _solve_onto(self.surfs, on, x)
        x = x / x.sum()
        basis = np.linalg.svd(np.vstack([np.ones(4), tangent]))[2][2:]     # normal plane
        base = np.sign(self.values(x)).astype(np.int8)
        # each surface through x: its trace in the normal plane is the line g.d = 0
        cuts = []
        for i in on:
            g = _g0(self.surfs[i], x)
            gp = np.array([g @ basis[0], g @ basis[1]])
            if np.linalg.norm(gp) < 1e-9:
                continue                       # (the surface contains the whole normal plane?)
            ang = np.arctan2(gp[1], gp[0]) + np.pi / 2                       # direction of g.d = 0
            cuts += [(ang % (2 * np.pi), i), ((ang + np.pi) % (2 * np.pi), i)]
        cuts.sort()
        if not cuts:
            return None
        sectors = []
        for k in range(len(cuts)):
            a0 = cuts[k][0]
            a1 = cuts[(k + 1) % len(cuts)][0] + (2 * np.pi if k + 1 == len(cuts) else 0)
            mid = (a0 + a1) / 2
            d = np.cos(mid) * basis[0] + np.sin(mid) * basis[1]
            sign = base.copy()
            for i in on:
                sign[i] = 1 if _g0(self.surfs[i], x) @ d > 0 else -1
            piece, copy = self._piece(x, d, sign)
            sectors.append({"angle": float(mid), "piece": piece, "copy": copy,
                            "region": None if piece is None else self.region[piece],
                            "after_surface": cuts[k][1]})
        walls = []
        for k in range(len(sectors)):
            a, b = sectors[k - 1], sectors[k]
            s = cuts[k][1]                     # the surface between sector k-1 and sector k
            wall = None
            if a["piece"] is not None and b["piece"] is not None and a["copy"] == b["copy"] == 0:
                wall = self.patch_wall.get((min(a["piece"], b["piece"]),
                                            max(a["piece"], b["piece"]), s))
            how = "patch" if wall else None
            if wall is None and a["region"] and b["region"] and a["region"] != b["region"]:
                wall, how = self._name_by_data(a["region"], b["region"], s)
            if a["region"] != b["region"] or wall:
                walls.append({"surface": s, "between": [a["region"], b["region"]], "wall": wall,
                              "named_by": how})
        return {"point": x, "surfaces_through": on, "sectors": sectors, "walls": walls}

    def _wall_surfaces(self):
        """Each wall's own surfaces (exact_defs), as library indices: a wall lies on these."""
        defs = load_json("exact_defs.json")["walls"]
        idx = {s.name: i for i, s in enumerate(self.surfs)}
        lib = json.loads((self.root / "surface_library.json").read_text())["surfaces"]
        for i, d in enumerate(lib):
            for a in d.get("aliases", []):
                idx[a] = i
        out = {}
        for w, v in defs.items():
            got = set()
            for n in v["surface"]:
                if n in idx:
                    got.add(idx[n])
                elif n.startswith("plane"):
                    nv = np.array([float(t) for t in n[7:-1].split(",")])
                    nv /= np.linalg.norm(nv)
                    got |= {i for i, s in enumerate(self.surfs)
                            if s.kind == "plane" and abs(abs(s.normal @ nv) - 1) < 1e-5}
            out[w] = got
        return out

    def _name_by_data(self, ra, rb, s):
        """The wall between regions ra and rb across surface s, from the working data: the one
        wall separating them; of several, the ones whose own exact surfaces include s (a wall
        lies on its surfaces); of those, the one with the line being checked as an edge; then by
        surface family."""
        cands = [w for w, v in self.naming.items() if ra in v["regions"] and rb in v["regions"]]
        if len(cands) == 1:
            return cands[0], "data: the one wall between these regions"
        on = [w for w in cands if s in self.wall_surfaces.get(w, ())]
        if len(on) == 1:
            return on[0], "data: regions, and the wall lies on this surface"
        if self.line and on:
            edge = [w for w in on if self.line in self.wall_edges.get(w, ())]
            if len(edge) == 1:
                return edge[0], "data: regions, surface, and the line is its edge"
        fam = [w for w in cands if self.family[s] in self.naming[w]["families"]]
        if len(fam) == 1:
            return fam[0], "data: regions and surface family"
        return None, None

    def _piece(self, x, d, sign):
        y = x + 1e-9 * d
        if (y @ FACE_N.T >= 0).all():
            return self.keys.get(sign.tobytes()), 0
        # beyond the domain's face: the sector is in a neighbouring copy (looked up there)
        k, yb = locate(point_from_beta(y / y.sum())[None], tol=1e-13)
        v = np.sign(self.values(yb[0])).astype(np.int8)
        return self.keys.get(v.tobytes()), int(k[0])


def line_report(step=2.5e-2) -> dict:
    """Walls round every atlas line, at points all along each of its domain pieces."""
    from .frame import surface_by_name, surfaces
    from .line_atlas import _exact_resample

    inc = Incidence()
    w = load_json("working_data.json")
    named = np.array([point_from_beta(np.asarray(p["float"], float)) for p in w["exact"]["points"]])
    lib_by_name = {n: surfaces()[i] for n, i in surface_by_name().items()}
    out = {}
    for name, info in w["exact"]["lines"].items():
        walls, unknown, n_pts = Counter(), 0, 0
        inc.line = name
        for pi, poly in enumerate(info["points"]):
            poly = np.asarray(poly, float)
            poly = poly / poly.sum(axis=1, keepdims=True)
            sn = info["surfaces"][pi] if pi < len(info["surfaces"]) else info["surfaces"][0]
            pair = [lib_by_name.get(n) for n in sn]
            if len(pair) == 2 and all(pair):
                poly = _exact_resample(poly, pair, step=step)
            for i in range(1, len(poly) - 1):
                x = poly[i]
                if np.min(np.linalg.norm(named - point_from_beta(x), axis=1)) < 1e-2:
                    continue
                r = inc.round_line(x, poly[i + 1] - poly[i - 1])
                if r is None:
                    continue
                n_pts += 1
                unknown += sum(s["piece"] is None for s in r["sectors"])
                for wl in r["walls"]:
                    walls[wl["wall"] or "unnamed " + "|".join(sorted(map(str, wl["between"])))] += 1
        out[name] = {"points": n_pts, "walls": dict(walls), "unknown_sectors": unknown}
    return out


if __name__ == "__main__":
    rep = line_report()
    w = load_json("working_data.json")
    for name, r in sorted(rep.items()):
        data = set(w["lines"][name].get("bounds_walls", []))
        found = {k for k in r["walls"] if not k.startswith("unnamed")}
        print(f"{name:4s} pts {r['points']:3d} unknown {r['unknown_sectors']:3d} | data {sorted(data)} | "
              f"found {dict(sorted(r['walls'].items(), key=lambda kv: -kv[1]))} | "
              f"{'MATCH' if found == data else 'missing ' + str(sorted(data - found)) + ' extra ' + str(sorted(found - data))}")
