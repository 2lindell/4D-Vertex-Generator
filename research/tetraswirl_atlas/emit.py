"""Turn build2's pickle into the JSON used by the page.   SWIRL_GROUP=tet python emit.py data_tet.pkl out.json"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter

import numpy as np
from skimage.measure import marching_cubes

import build2 as B
import domain as D

PAL = ["#e07a5f", "#3d85c6", "#81b29a", "#f2cc8f", "#b07cc6", "#d4a373", "#5fa8a8", "#c96f9b", "#9aa15a", "#6f7fd1"]


def short(sig):
    head, rest = sig.split(": cells ")
    cells, _, _ = rest.partition(" | faces ")
    c = Counter()
    for part in cells.split("+"):
        n, shape = part.split("x[")
        c[shape.rstrip("]")] += int(n)
    names = {"6v8f": "octahedra", "4v4f": "tetrahedra", "6v5f": "triangular prisms", "5v6f": "square pyramids",
             "8v6f": "cubes/hexahedra"}
    body = " + ".join(f"{n} {names.get(k, k)}" for k, n in sorted(c.items()))
    return f"{head} vertices; {body}"


def tris_to_json(v, f, nd=4):
    return {"v": np.round(v, nd).ravel().tolist(), "f": f.ravel().tolist()}


def reindex(v, f):
    used, inv = np.unique(f, return_inverse=True)
    return v[used], inv.reshape(f.shape)


def project(P, ms, co, p):
    p = np.array(p, float)
    for _ in range(40):
        val = B.design(p, ms) @ co
        g = np.array([(B.design(p + e * 1e-6, ms) - B.design(p - e * 1e-6, ms)) @ co / 2e-6 for e in np.eye(3)]).ravel()
        if abs(val[0]) < 1e-14:
            break
        p = p - g * val[0] / (g @ g)
    return p


def info(u, tol=1e-7):
    s = D.signature(D.classify(D.seed_from_chart(np.asarray(u)), tol))
    return s


def main(pk, out):
    r = pickle.load(open(pk, "rb"))
    h, lo, shape = r["h"], r["lo"], r["shape"]
    fits = r["reps"]
    sig_names = {}

    def tname(sig):
        if sig not in sig_names:
            sig_names[sig] = f"T{len(sig_names) + 1}"
        return sig_names[sig]

    for rg in r["regions"]:
        tname(rg["type_sig"])
    types = {}
    # regions
    step = 2
    regions = []
    Hn = B.HS
    for i, rg in enumerate(r["regions"], 1):
        mask = rg["mask"]
        field = np.where(mask, 1.0, -1.0)
        from scipy import ndimage
        field = ndimage.gaussian_filter(field, 0.8) if mask.sum() > 400 else field
        field = np.where(r["margin3"] > -0.5 * h, field, -1)
        if mask.sum() < 400:
            step_i = 1
        else:
            step_i = step
        v, f, _, _ = marching_cubes(field, 0.0, step_size=step_i)
        v = v * h + lo
        v, f = reindex(v, f)
        # example: the deepest interior voxel
        dm = ndimage.distance_transform_edt(mask)
        k = np.unravel_index(np.argmax(dm), mask.shape)
        u = np.array(k) * h + lo
        t = tname(rg["type_sig"])
        regions.append({"id": rg["id"], "type": t, "volume": round(float(rg["voxels"] * h ** 3), 4), "u": np.round(u, 5).tolist(),
                        "mesh": tris_to_json(v, f)})
    # walls
    walls = []
    for k, w in enumerate(r["wall_surf"], 1):
        d_, ms, co = fits[w["surface"]]
        tris = w["tris"]
        v, f = reindex(w["verts"], tris)
        cen = v[f].mean(1)
        area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
        mid = project(None, ms, co, cen[np.argmax(area)])
        s = info(mid)
        t = tname(s) if s != "none" else None
        a, b = w["pair"]
        walls.append({"id": f"W{k}", "surface": int(w["surface"]), "between": [f"R{a}", f"R{b}"], "area": round(w["area"], 4),
                      "type": t, "u": np.round(mid, 5).tolist(), "mesh": tris_to_json(v, f)})
    # lines: intersections of two surfaces' meshes, found on the wall meshes
    segs = {}
    nsurf = len(fits)
    for a in range(nsurf):
        for b in range(a + 1, nsurf):
            pts = []
            for (v, f, _, _, _), (src, other) in ((r["meshes"][a], (a, b)), (r["meshes"][b], (b, a))):
                d_, ms, co = fits[other]
                val = B.design(v, ms) @ co
                s = np.sign(val)
                for tri in f:
                    ss = s[tri]
                    if ss.min() == ss.max():
                        continue
                    pp = []
                    for i0, i1 in ((0, 1), (1, 2), (2, 0)):
                        if ss[i0] != ss[i1]:
                            t0 = val[tri[i0]] / (val[tri[i0]] - val[tri[i1]])
                            pp.append(v[tri[i0]] + t0 * (v[tri[i1]] - v[tri[i0]]))
                    if len(pp) == 2 and B.margin(np.array(pp)).min() > -h:
                        pts.append(pp)
                break
            if pts:
                segs[(a, b)] = pts
    lines = []
    for (a, b), pts in segs.items():
        flat = np.array(pts).reshape(-1, 3)
        # polish onto both surfaces, then record each segment
        polished = []
        for p in flat:
            q = np.array(p)
            for _ in range(8):
                for c in (a, b):
                    q = project(None, fits[c][1], fits[c][2], q)
            polished.append(q)
        P = np.array(polished).reshape(-1, 2, 3)
        keep = np.array([B.margin(x).min() > -1e-6 for x in P.reshape(-1, 3)]).reshape(-1, 2).all(1)
        P = P[keep]
        if len(P) == 0:
            continue
        mid = P[len(P) // 2].mean(0)
        s = info(mid)
        lines.append({"id": f"L{len(lines) + 1}", "kind": f"walls {a + 1} and {b + 1} meet", "type": tname(s) if s != "none" else None,
                      "u": np.round(mid, 5).tolist(), "segs": np.round(P, 4).reshape(len(P), 6).tolist()})
    # fixed-circle lines: edges of the domain where the Voronoi cell is cut, and the chart axis
    V = D.domain_vertices()
    points = []
    seen = []
    for p in V:
        if any(np.linalg.norm(p - q) < 1e-6 for q in seen):
            continue
        seen.append(p)
        s = info(p, 1e-9)
        points.append({"id": f"P{len(points) + 1}", "u": np.round(p, 5).tolist(), "type": tname(s) if s != "none" else None})
    zs = [-1, 1]
    ax = np.linspace(V[:, 2].min(), V[:, 2].max(), 2)
    axis_pts = np.array([[0, 0, z] for z in ax])
    s = info(axis_pts[0] * 0.3 + axis_pts[1] * 0.7)
    lines.append({"id": f"L{len(lines) + 1}", "kind": "axis of the order-3 anchor (fixed circle)", "type": tname(s) if s != "none" else None,
                  "u": [0, 0, round(float(ax.mean() + 0.1), 5)], "segs": [np.round(axis_pts, 4).ravel().tolist()]})
    keyed = []
    for sig, t in sig_names.items():
        keyed.append({"type": t, "desc": short(sig), "full": sig})
    # seeds
    def seed(u):
        return np.round(D.seed_from_chart(np.array(u)), 6).tolist()

    for coll in (regions, walls, lines, points):
        for it in coll:
            it["seed"] = seed(it["u"])
    domain = {"halfspaces": np.round(D.domain_halfspaces(), 6).tolist(), "vertices": np.round(V, 5).tolist()}
    hull = __import__("scipy.spatial", fromlist=["ConvexHull"]).ConvexHull(V)
    domain["faces"] = hull.simplices.tolist()
    # Colour by type, order of first use
    ctypes = sorted(sig_names.values(), key=lambda x: int(x[1:]))
    colors = {t: PAL[i % len(PAL)] for i, t in enumerate(ctypes)}
    out_d = {"group": D.NAME, "order": int(len(D.ELEMENTS)), "key": D.KEY, "types": keyed, "colors": colors, "regions": regions,
             "walls": walls, "lines": lines, "points": points, "domain": domain, "grid_h": round(h, 4)}
    json.dump(out_d, open(out, "w"), separators=(",", ":"))
    print(len(regions), "regions", len(walls), "walls", len(lines), "lines", len(points), "points", len(keyed), "types")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
