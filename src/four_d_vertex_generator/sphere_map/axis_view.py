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
GAP = 0.18                         # how far each shape is moved out from the axis, in the view's units


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
         "T": half(X38)},
        {"id": "X68", "kind": "region", "note": "one of its two sides of X38 (the half-turn carries each onto the other)",
         "T": X68},
    ]
    out = []
    for s_ in shapes:
        T3 = chart(s_["T"])
        cen = T3.reshape(-1, 3).mean(0)
        d = np.array([0.0, cen[1], cen[2]])                  # (straight out from the axis)
        d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else np.array([0, 1.0, 0])
        # (walls nearest the axis, regions further: a region's shell holds its walls, so each is
        # moved out its own distance to stand clear of them)
        T3 = T3 + GAP * (1 if s_["kind"] == "wall" else 3.2) * d
        V3, F3 = np.unique(T3.reshape(-1, 3).round(6), axis=0, return_inverse=True)
        out.append({"id": s_["id"], "kind": s_["kind"], "note": s_["note"],
                    "verts": V3.round(5).tolist(), "faces": F3.reshape(-1, 3).tolist()})
    L = np.linspace(-0.6, 0.6, 2)
    return {"_about": __doc__, "axis": [[float(t), 0.0, 0.0] for t in L], "shapes": out}


def run() -> str:
    path = data_dir() / "sphere_map" / "axis_view.json"
    data = build()
    path.write_text(json.dumps(data, separators=(",", ":")))
    for s in data["shapes"]:
        print(f"{s['id']} ({s['kind']}): {len(s['faces'])} triangles; {s['note']}")
    return str(path)


if __name__ == "__main__":
    print(run())
