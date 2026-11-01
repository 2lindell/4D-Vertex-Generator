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


def _solve_onto(surfs, idx, x, iters=20, max_move=1e-6):
    """x moved onto every surface in idx (they meet along the line through x). Many surfaces can
    pass through one curve, so the Jacobian is rank-deficient: small singular values are cut, and
    a correction is kept only if it is tiny (x is already on the line to that accuracy)."""
    x0 = np.asarray(x, float)
    x = x0.copy()
    for _ in range(iters):
        f = np.array([float(surfs[i].f(x[None])[0]) for i in idx])
        if np.abs(f).max() < 1e-15:
            break
        J = np.array([_g0(surfs[i], x) for i in idx])
        x = x + np.linalg.lstsq(np.vstack([J, np.ones(4)]), np.r_[-f, 0.0], rcond=1e-8)[0]
    return x if np.linalg.norm(x - x0) < max_move else x0


def _quad_roots(H):
    """Directions theta where (cos, sin) H (cos, sin)^T = 0."""
    a, b, c = H[0, 0], H[0, 1], H[1, 1]
    out = []
    if abs(c) < 1e-14:                               # theta = pi/2 is a root
        out.append(np.pi / 2)
        if abs(b) > 1e-14:
            out.append(np.arctan2(-a, 2 * b))
    else:
        disc = b * b - a * c
        if disc < 0:
            return []
        for t in ((-b + np.sqrt(disc)) / c, (-b - np.sqrt(disc)) / c):
            out.append(np.arctan(t))
    return [o for a_ in out for o in (a_, a_ + np.pi)]


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
        from .frame import cell_vertices, group
        self.T = cell_vertices()
        self.Tinv = np.linalg.inv(self.T)
        self.g = group()
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
        """Sectors round the line at x: [(sector direction angle, copy, piece, region)], and the
        walls between neighbouring sectors [(surface, region a, region b, wall)].

        Where the line runs on the domain's boundary the ring round it passes through several
        copies. Each copy k is exactly the domain seen through beta_k = beta . M_k
        (M_k = T g_k T^-1), so the surfaces through the line in copy k are copy k's own library
        surfaces, pulled back through M_k; each sector is read in the copy it lies in."""
        x = x / x.sum()
        on0 = self.through(x, tol)
        if on0:
            x = _solve_onto(self.surfs, on0, x)
            x = x / x.sum()
        basis = np.linalg.svd(np.vstack([np.ones(4), tangent]))[2][2:]     # normal plane
        frames = self._frames_round(x, basis)
        cuts = []
        self._local = {}                              # (k, i) -> ('lin', gp) or ('quad', H)
        for k, (M, xk, on) in frames.items():
            for i in on:
                kind, c = self._local_form(i, M, xk, basis)
                self._local[(k, i)] = (kind, c)
                if kind == "lin":                     # the trace is the line gp . (cos, sin) = 0
                    ang = np.arctan2(c[1], c[0]) + np.pi / 2
                    angs = [ang, ang + np.pi]
                elif kind == "quad":                  # singular here: a cone, q(theta) = 0
                    angs = _quad_roots(c)
                else:
                    angs = []
                cuts += [(a % (2 * np.pi), k, i) for a in angs]
        if not cuts:
            return None
        cuts.sort()
        groups = []                                   # cut directions, equal ones merged
        for c in cuts:
            if groups and c[0] - groups[-1][0] < 1e-9:
                groups[-1][1].append(c[1:])
            else:
                groups.append((c[0], [c[1:]]))
        if len(groups) > 1 and groups[0][0] + 2 * np.pi - groups[-1][0] < 1e-9:
            groups[0][1].extend(groups.pop()[1])
        sectors = []
        for j in range(len(groups)):
            a0 = groups[j][0]
            a1 = groups[(j + 1) % len(groups)][0] + (2 * np.pi if j + 1 == len(groups) else 0)
            mid = (a0 + a1) / 2
            d = np.cos(mid) * basis[0] + np.sin(mid) * basis[1]
            k = self._copy_of(frames, d)
            piece = None
            if k is not None:
                M, xk, on = frames[k]
                sign = np.sign(self.values(xk)).astype(np.int8)
                dk = d @ M
                u = np.array([np.cos(mid), np.sin(mid)])
                for i in on:
                    kind, c = self._local[(k, i)]
                    if kind == "lin":
                        v = c @ u
                    elif kind == "quad":
                        v = u @ c @ u
                    else:
                        v = float(self.surfs[i].f((xk + 1e-4 * dk)[None])[0])
                    sign[i] = 1 if v > 0 else -1
                piece = self.keys.get(sign.tobytes())
            sectors.append({"angle": float(mid), "piece": piece, "copy": k,
                            "region": None if piece is None else self.region[piece]})
        walls = []
        for j in range(len(sectors)):
            a, b = sectors[j - 1], sectors[j]
            here = groups[j][1]
            same = [i for k, i in here if k == a["copy"] == b["copy"]]
            # across the domain's face (sectors in two copies) the wall lies in the face itself
            face = [i for k, i in here if k in (a["copy"], b["copy"])]
            s = same[0] if same else face[0] if face else None
            wall = None
            if s is not None and a["piece"] is not None and b["piece"] is not None:
                for t in same:
                    wall = self.patch_wall.get((min(a["piece"], b["piece"]),
                                                max(a["piece"], b["piece"]), t))
                    if wall:
                        s = t
                        break
            how = "patch" if wall else None
            if wall is None and a["region"] and b["region"] and a["region"] != b["region"]:
                wall, how = self._name_by_data(a["region"], b["region"], s)
            if a["region"] != b["region"] or wall:
                walls.append({"surface": s, "copy": a["copy"], "between": [a["region"], b["region"]],
                              "wall": wall, "named_by": how})
        return {"point": x, "surfaces_through": on0, "sectors": sectors, "walls": walls}

    def _local_form(self, i, M, xk, basis, h=1e-5):
        """Surface i at the line point, in the normal plane (coordinates along basis, copy-0
        frame): its gradient there ('lin'), or where the gradient vanishes (the surface is
        singular along the line) its second-order form ('quad'), from exact gradients."""
        s = self.surfs[i]
        g = M @ s.grad(xk[None])[0]
        gp = np.array([g @ basis[0], g @ basis[1]])
        if np.linalg.norm(gp) > 1e-8:
            return "lin", gp
        if s.kind == "plane":
            return "none", None
        H = np.empty((2, 2))
        for b in range(2):
            e = basis[b] @ M
            gd = M @ (s.grad((xk + h * e)[None])[0] - s.grad((xk - h * e)[None])[0]) / (2 * h)
            H[:, b] = [gd @ basis[0], gd @ basis[1]]
        H = (H + H.T) / 2
        if np.abs(H).max() < 1e-8:
            return "high", None
        return "quad", H

    def _frames_round(self, x, basis, n=720):
        """The copies the ring round the line passes through: {k: (M_k, beta_k of x, surfaces
        through it)}."""
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        y = x + 1e-7 * (np.cos(ang)[:, None] * basis[0] + np.sin(ang)[:, None] * basis[1])
        ks, _ = locate(point_from_beta(y / y.sum(axis=1, keepdims=True)), tol=1e-13)
        out = {}
        for k in sorted(set(int(t) for t in ks)):
            M = self.T @ self.g[k] @ self.Tinv
            xk = x @ M
            xk = xk / xk.sum()
            out[k] = (M, xk, self.through(xk, 1e-6))
        return out

    @staticmethod
    def _copy_of(frames, d):
        """The copy whose domain the direction d (at x) points into."""
        for k, (M, xk, _) in frames.items():
            dk = d @ M
            act = np.abs(xk @ FACE_N.T) < 1e-10
            if (dk @ FACE_N[act].T > 0).all():
                return k
        return None

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
        on = [w for w in cands if s is not None and s in self.wall_surfaces.get(w, ())]
        if len(on) == 1:
            return on[0], "data: regions, and the wall lies on this surface"
        if self.line:
            edge = [w for w in (on or cands) if self.line in self.wall_edges.get(w, ())]
            if len(edge) == 1:
                return edge[0], "data: regions, " + ("surface, " if on else "") + "and the line is its edge"
        if s is None:
            return None, None
        fam = [w for w in cands if self.family[s] in self.naming[w]["families"]]
        if len(fam) == 1:
            return fam[0], "data: regions and surface family"
        return None, None


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
