"""Shapes around a 2400-symmetry axis, each shown once, exploded (axis_view.json).

The axis: the great circle fixed by one of the extra half-turns (the 2400-element group's elements
outside the swirlprism group); here the one X38 runs along. The view is gnomonic at the point of the
axis nearest the shapes, the axis a straight line through it.

Each shape is shown with only the extent every seed it holds needs once: a region or wall the
2400-element group carries onto itself by an element other than the identity is cut to a part
no such element carries onto itself (X38, turned onto itself by the axis's half-turn: the half on
one side of the axis); a region another of its copies is carried onto (X68's two sides of X38,
swapped by the half-turn) is shown once. Shapes are laid out apart from one another, each moved
straight out from the axis, so each can be seen whole.

A region is drawn as its shell (the wall patches bounding it, regions.json), a wall as its patches;
each patch's triangles are those of the wall pieces (domain_pieces) found between that patch's two
pieces of the domain.

    python -m four_d_vertex_generator.sphere_map.axis_view
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .frame import data_dir, group, point_from_beta, source_dir

AXIS_ELEMENT = 1200 + 368          # the half-turn whose axis X38 runs along (in the 2400-element group)
# the explosion: the whole picture scaled by EXPLODE about the view's centre (the axis's point the view
# is centred on) while each shape keeps its own size: each shape moved so its middle (a wall's area, an
# edge's length) goes from m to EXPLODE * m. Every shape then keeps its place against its neighbours,
# with a gap: a corner just beyond its edges' ends, an edge just off its wall's side.
# A region moves with the walls bounding it (the mean of their moves), and each of those walls then moves
# off it, along the wall's normal away from the region's volume, by WALL_STEP: the walls curve round
# the axis, so moving by the region's middle (here almost on the axis) would slide it along them.
EXPLODE = 1.8
WALL_STEP = 0.05
NEAR = ("X38", "X68", "X99")              # the shapes whose edges and corners are shown with them


def _patch_triangles(D, doc, z):
    """Each wall patch's triangles (on R^4, in the half-cell's own copy): every wall piece's triangles,
    each named by the patch between the two pieces of the domain on either side of its middle."""
    from . import domain_pieces as dp
    vals = lambda X: np.sign(np.stack([sf.f(X) for sf in D.lib], 1)).astype(np.int8)
    out = defaultdict(list)
    for f in doc["faces"]:
        X = z["verts"][f["verts"][0]:f["verts"][1]]
        F = z["tris"][f["tris"][0]:f["tris"][1]]
        cen = X[F].mean(1)
        if f["sheet"].startswith("face"):
            continue                      # (pieces in a face of the half-cell: none in this view yet)
        s = D.lib[int(f["sheet"])]
        n = s.grad(cen)
        n = n - n.mean(1, keepdims=True)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        # (each middle moved onto the surface first: a flat triangle's middle is off a curved one)
        lifted = [dp._lift(s, y, nn) for y, nn in zip(cen, n)]
        cen = np.array([y if l_ is None else l_ for y, l_ in zip(cen, lifted)])
        key = int(f["sheet"])
        A = [D.keys.get(r.tobytes()) for r in vals(cen + 1e-7 * n)]
        B = [D.keys.get(r.tobytes()) for r in vals(cen - 1e-7 * n)]
        for t, (a, b) in enumerate(zip(A, B)):
            if a is None or b is None or a == b:
                continue
            p = D.pw.get((min(a, b), max(a, b), key)) or D.pw.get((min(a, b), max(a, b), None))
            if p is not None:
                out[p["id"]].append(point_from_beta(X[F[t]]))
    return {k: np.array(v) for k, v in out.items()}


def _to_path(X, M):
    """Each point of X's distance to the polyline M (a single point when M has one)."""
    if len(M) == 1:
        return np.linalg.norm(X - M[0], axis=1)
    a, b = M[:-1], M[1:]
    ab = b - a
    t = np.clip(np.einsum("nsk,sk->ns", X[:, None] - a[None], ab) / np.maximum(np.einsum("sk,sk->s", ab, ab), 1e-30), 0, 1)
    return np.linalg.norm(X[:, None] - (a[None] + t[..., None] * ab[None]), axis=2).min(1)


def _same(L, M, tol=1e-4):
    """Whether two sampled curves (or points) on R^4 are the same set, within tol (each sample of
    one measured to the other's polyline, as their samples need not fall at the same places)."""
    return len(L) == len(M) == 1 and np.linalg.norm(L[0] - M[0]) < tol or \
        len(L) > 1 and len(M) > 1 and max(_to_path(L, M).max(), _to_path(M, L).max()) < tol


def _half_path(L, u):
    """A curve cut to x . u >= 0 (the end on the axis found exactly)."""
    s = L @ u
    keep = []
    for i in range(len(L)):
        if s[i] >= 0:
            keep.append(L[i])
        if i + 1 < len(L) and (s[i] >= 0) != (s[i + 1] >= 0):
            keep.append(L[i] + s[i] / (s[i] - s[i + 1]) * (L[i + 1] - L[i]))
    return np.array(keep)


def _edges_and_corners(H, u, half):
    """The named lines and points bounding the near shapes (shape_map3d.json), each placed copy once:
    a copy the half-turn carries onto itself is cut to its half on the kept side (x . u >= 0); of two
    copies it swaps, the one on the kept side is shown."""
    sm = json.loads((data_dir() / "sphere_map" / "shape_map3d.json").read_text())
    G = group()
    names = {(k, x["id"]): x["name"] for k in ("lines", "points") for x in sm[k]}
    pc = {p["id"]: np.array(p["c"]) for p in sm["points"]}
    found = {"line": [], "point": []}               # (name, geometry on R^4)
    for kind in ("walls", "regions"):
        for x in sm[kind]:
            if x["name"] not in NEAR:
                continue
            for n in x["nb"]:
                if n["kind"] not in found:
                    continue
                for piece, g in n["touch"]:
                    if n["kind"] == "line":
                        geo = np.array(sm["line_geo"][piece][0]) @ G[g].T
                    else:
                        geo = (pc[piece] @ G[g].T)[None]
                    nm = names[(n["kind"] + "s", n["node"])]
                    if not any(_same(geo, o) for _, o in found[n["kind"]]):
                        found[n["kind"]].append((nm, geo))
    out = []
    for kind, items in found.items():
        done = set()
        for i, (nm, geo) in enumerate(items):
            if i in done:
                continue
            img = geo @ H.T
            if _same(img, geo):
                if kind == "point":
                    out.append({"id": nm, "kind": kind, "note": "on the axis: the half-turn holds it fixed", "at": geo[0]})
                else:
                    out.append({"id": nm, "kind": kind, "note": "half: it crosses the axis, and the half-turn turns it end for end",
                                "path": _half_path(geo, u)})
                continue
            j = next((j for j, (_, o) in enumerate(items) if j != i and _same(img, o)), None)
            if j is not None:
                done.add(j)
                if (items[j][1] @ u).mean() > (geo @ u).mean():
                    geo = items[j][1]
                note = "one of two copies (the half-turn carries each onto the other)"
            else:
                note = "whole: no symmetry here repeats it"
            if kind == "point":
                out.append({"id": nm, "kind": kind, "note": note, "at": geo[0]})
            else:
                out.append({"id": nm, "kind": kind, "note": note, "path": geo})
    return out


def build() -> dict:
    root = data_dir() / "sphere_map"
    from . import domain_pieces as dp
    D = dp.Data()
    doc = json.loads((root / "domain_pieces.json").read_text())
    z = np.load(root / "domain_pieces.npz")
    G = group()
    Q = np.load(source_dir() / "normalizer_q.npy")
    H = G[AXIS_ELEMENT - 1200] @ Q
    tris = _patch_triangles(D, doc, z)
    patches = json.loads((root / "patches.json").read_text())["patches"]
    regions = json.loads((root / "regions.json").read_text())["regions"]

    # the axis: the 2-plane the half-turn fixes; its other 2-plane is turned half round
    w, V = np.linalg.eigh((H + H.T) / 2)
    A, P = V[:, w > 0.5].T, V[:, w < -0.5].T
    # shapes (on R^4): X38's patches in the half-cell's copy, and X68's side through its first orbit
    x38 = [p["id"] for p in patches if p["wall"] == "X38" and p["id"] in tris]
    X38 = np.concatenate([tris[i] for i in x38])                      # (n, 3, 4)
    o = regions["X68"]["orbits"][0]
    shell = [(k, pid) for b in o["boundary"] for k, pid in b["parts"] if pid in tris]
    X68 = np.concatenate([np.einsum("ij,ntj->nti", G[k], tris[pid]) for k, pid in shell])
    x68_faces = defaultdict(list)                 # (its shell's faces, one per wall)
    for b in o["boundary"]:
        for k, pid in b["parts"]:
            if pid in tris:
                x68_faces[b["wall"]].append(np.einsum("ij,ntj->nti", G[k], tris[pid]))
    # (X68's middle: the middle of its volume, from its pieces' samples weighted by their volumes)
    pcs = {pc["id"]: pc for pc in json.loads((root / "pieces.json").read_text())["pieces"]}
    mem = o["members"] if isinstance(o["members"], list) else json.loads(o["members"])
    wts = np.array([pcs[pid]["volume_share"] for _, pid in mem])
    pts = np.array([point_from_beta(np.array(pcs[pid]["sample_beta"])) @ G[k].T for k, pid in mem])
    x68_mid = (wts[:, None] * pts).sum(0)
    x68_mid /= np.linalg.norm(x68_mid)
    # the view: gnomonic at the axis's point nearest X38, the axis along x
    m = X38.reshape(-1, 4).mean(0)
    c = A.T @ (A @ m)
    c /= np.linalg.norm(c)
    e1 = A.T @ (A @ np.array([1.0, 0, 0, 0]))
    e1 = A[0] if np.linalg.norm(e1) < 1e-9 else e1
    e1 = e1 - (e1 @ c) * c
    e1 /= np.linalg.norm(e1)
    # X38's half: the side of the axis it reaches out to most (the half-turn turns u to -u)
    Pm = (X38.reshape(-1, 4) @ P.T)
    u = P.T @ np.linalg.svd(Pm - 0, full_matrices=False)[2][0]
    u /= np.linalg.norm(u)
    e2 = u
    e3 = P.T @ (P @ np.ones(4))
    e3 = e3 - (e3 @ e2) * e2
    e3 = e3 if np.linalg.norm(e3) > 1e-9 else P[1]
    e3 /= np.linalg.norm(e3)
    chart = lambda X: np.stack([X @ e1, X @ e2, X @ e3], -1) / (X @ c)[..., None]

    def half(T):                              # triangles cut to x . u >= 0
        out = []
        for tri in T:
            s = tri @ u
            if (s >= -1e-12).all():
                out.append(tri)
            elif (s > 0).any():
                pts = []
                for i in range(3):
                    a, b = tri[i], tri[(i + 1) % 3]
                    sa, sb = a @ u, b @ u
                    if sa >= 0:
                        pts.append(a)
                    if (sa >= 0) != (sb >= 0):
                        pts.append(a + sa / (sa - sb) * (b - a))
                for k in range(1, len(pts) - 1):
                    out.append(np.array([pts[0], pts[k], pts[k + 1]]))
        return np.array(out)

    shapes = [
        {"id": "X38", "kind": "wall", "note": "half: the axis's half-turn carries X38 onto itself",
         "T": half(X38), "bounds": "X68"},
        {"id": "X99", "kind": "wall", "note": "one of two copies, one bounding each side of X68 (the half-turn carries each onto the other)",
         "T": np.concatenate(x68_faces["X99"]), "bounds": "X68"},
        {"id": "X68", "kind": "region", "note": "one of its two sides of X38 (the half-turn carries each onto the other)",
         "T": X68, "mid": x68_mid},
    ]
    shapes += _edges_and_corners(H, u, half)
    # first every shape's own middle and its move out from the view's centre
    for s_ in shapes:
        key = "T" if "T" in s_ else "path" if "path" in s_ else "at"
        T3 = chart(np.asarray(s_[key]))
        if key == "T":                                       # (area-weighted)
            ar = np.linalg.norm(np.cross(T3[:, 1] - T3[:, 0], T3[:, 2] - T3[:, 0]), axis=1)
            cen = (ar[:, None] * T3.mean(1)).sum(0) / ar.sum()
        elif key == "path":                                  # (length-weighted)
            ln = np.linalg.norm(np.diff(T3, axis=0), axis=1)
            cen = (ln[:, None] * (T3[1:] + T3[:-1]) / 2).sum(0) / max(ln.sum(), 1e-12)
        else:
            cen = T3
        s_.update(key=key, T3=T3, shift=(EXPLODE - 1) * cen)
    by = {s_["id"]: s_ for s_ in shapes}
    # a region with its walls; each of its walls then off it, away from its volume
    for s_ in shapes:
        walls = [w for w in shapes if w.get("bounds") == s_["id"]]
        if s_["kind"] == "region" and walls:
            s_["shift"] = np.mean([w["shift"] for w in walls], 0)
    for w in shapes:
        if "bounds" in w:
            r = by[w["bounds"]]
            P = w["T3"].reshape(-1, 3)
            n = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][2]     # (its best plane's normal)
            n *= -np.sign(n @ (chart(r["mid"]) - P.mean(0)))
            w["shift"] = w["shift"] + WALL_STEP * n
    out = []
    for s_ in shapes:
        key, T3 = s_["key"], s_["T3"] + s_["shift"]
        rec = {"id": s_["id"], "kind": s_["kind"], "note": s_["note"]}
        if key == "T":
            V3, F3 = np.unique(T3.reshape(-1, 3).round(6), axis=0, return_inverse=True)
            rec.update(verts=V3.round(5).tolist(), faces=F3.reshape(-1, 3).tolist())
        elif key == "path":
            rec["path"] = T3.round(5).tolist()
        else:
            rec["at"] = T3.round(5).tolist()
        out.append(rec)
    L = np.linspace(-0.35, 0.35, 2)
    return {"_about": __doc__, "axis": [[float(t), 0.0, 0.0] for t in L], "shapes": out}


def run() -> str:
    path = data_dir() / "sphere_map" / "axis_view.json"
    data = build()
    path.write_text(json.dumps(data, separators=(",", ":")))
    for s in data["shapes"]:
        size = f"{len(s['faces'])} triangles" if "faces" in s else f"{len(s['path'])} steps" if "path" in s else "point"
        print(f"{s['id']} ({s['kind']}): {size}; {s['note']}")
    return str(path)


if __name__ == "__main__":
    print(run())
