"""The Symmetry Domain page's walls, lines and regions, from the 3-sphere map's wall pieces.

Input: the sphere map's `domain_pieces.json` / `.npz` (branch claude/sharp-rubin-c0puiw, module
`sphere_map.domain_pieces`). There every wall surface in the half-cell is cut by its exact edges
only: the named lines (stretches cut and snapped at the exact named points) and the curves where
another wall meets it, with the half-cell's own faces; each face of that graph is one wall piece
in the half-cell, named from the map's pieces, and triangulated inside its own exact loop (the loop
points the only rim vertices, every inner vertex on the exact surface).

This page keeps everything else it shows (frame, mirrors, axes, samples, seeds, key). Its three
views are each a projective chart of the 3-sphere; each chart is read off the page itself:

  - the half-cell: fitted to the page's own seed points (each listed with its point on the sphere);
  - the dodecahedron round V1 (and its tenth, the chambers): the gnomonic chart at V1, turned to
    match the page's samples there; the walls carried there by every element of the group and cut
    exactly at the dodecahedron's faces (the tenth: at its own faces, as the page draws them).

Wall pieces are drawn as they are; a piece's edges on named lines solid, where another wall meets
it dashed, nothing along the half-cell's faces. The named lines are their exact stretches. A
region shows the wall pieces with it on one side and their edges.

    PYTHONPATH=../../src python domain_view.py <sphere_map data dir> ../../assets/symmetry_domain.html
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

T = np.load(Path(__file__).with_name("cell_vertices.npy"))
RINGS = ("C2a", "C2b", "C3", "D1", "D2", "D3")
SEL_ONLY = {"T1", "E1", "E2"}


def to_sphere(B):
    X = np.atleast_2d(B) @ T
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def blocks(html):
    out = {}
    for m in re.finditer(r'<script id="([^"]+)" type="application/json">', html):
        s = m.end()
        out[m.group(1)] = (s, html.index("</script>", s))
    return out


def seed_of(hover):
    m = re.search(r"seed ([-0-9.e]+), ([-0-9.e]+), ([-0-9.e]+), ([-0-9.e]+)", hover)
    if not m:
        return None
    x = np.array([float(v) for v in m.groups()])
    return x / np.linalg.norm(x)


# ---------------------------------------------------------------- charts

class Chart:
    """u = (X M) / (X . p) for points X on the sphere (X . p > 0)."""

    def __init__(self, M, p):
        self.M, self.p = M, p

    def __call__(self, X):
        X = np.atleast_2d(X)
        return (X @ self.M) / (X @ self.p)[:, None]


def fit_cell_chart(DC):
    """The half-cell view's chart, from the page's seed points (point on the sphere, place drawn)."""
    X, Q = [], []
    for u in DC["uniform"] + DC["special"]:
        x = seed_of(u.get("hover", ""))
        if x is not None:
            X.append(x)
            Q.append(u["q"])
    X, Q = np.array(X), np.array(Q)
    rows = []
    for x, q in zip(X, Q):
        for k in range(3):
            r = np.zeros(16)
            r[4 * k:4 * k + 4] = x
            r[12:] = -q[k] * x
            rows.append(r)
    sol = np.linalg.svd(np.array(rows))[2][-1]
    M, p = sol[:12].reshape(3, 4).T, sol[12:]
    if (X @ p).mean() < 0:
        M, p = -M, -p
    ch = Chart(M, p)
    err = np.abs(ch(X) - Q).max()
    return ch, err


def fit_v1_chart(DC, V1, G):
    """The dodecahedron views' chart: gnomonic at V1 = beta (1, 0, 0, 0), turned (any linear map of
    the chart) so the page's samples there are images of the half-cell's samples."""
    c = to_sphere([[1, 0, 0, 0]])[0]
    B0 = np.linalg.svd(c[None])[2][1:]
    w0 = lambda X: (np.atleast_2d(X) @ B0.T) / (np.atleast_2d(X) @ c)[:, None]
    near = dodeca_planes(c, G)
    src = {k: [seed_of(p[3]) for p in v] for k, v in DC["samples"].items()}
    pairs = []
    for k, v in V1["samples"].items():
        for p in v:
            x = src[k][p[3]]
            if x is None:
                continue
            Y = x @ G.transpose(0, 2, 1)                       # every image g x
            Y = Y[(Y @ near.T >= -1e-9).all(1) & (Y @ c > 0)]
            if len(Y):
                pairs.append((np.array(p[:3]), w0(Y)))
    rng = np.random.default_rng(0)
    pairs = [pairs[i] for i in rng.permutation(len(pairs))[:400]]
    best = (-1, None)
    u1, W1 = pairs[0]
    for i2 in range(1, 6):
        for i3 in range(6, 12):
            for a in W1[:1]:
                for b in pairs[i2][1]:
                    for c3 in pairs[i3][1]:
                        A = np.vstack([a, b, c3])
                        if abs(np.linalg.det(A)) < 1e-6:
                            continue
                        R = np.linalg.solve(A, np.vstack([u1, pairs[i2][0], pairs[i3][0]])).T
                        good = sum(np.linalg.norm(W @ R.T - u, axis=1).min() < 2e-3 for u, W in pairs)
                        if good > best[0]:
                            best = (good, R)
    R = best[1]
    ch = Chart((R @ B0).T, c)
    return ch, best[0] / len(pairs), near


def dodeca_planes(c, G):
    """The 12 faces of the dodecahedron round c: X . (c - n) >= 0 for its 12 nearest images n."""
    orbit = np.unique(np.round(G @ c, 9), axis=0)
    near = orbit[np.argsort(-(orbit @ c))[1:13]]
    return c - near


def chamber_planes(CH, chart):
    """The tenth's faces (as the page draws them), as half-spaces X . h >= 0 on the sphere."""
    from scipy.spatial import ConvexHull
    P = np.unique(np.round(np.array([p for f in CH["dodecaFaces"] for p in f]), 5), axis=0)
    H = ConvexHull(P)
    out = []
    Bm, c = chart.M, chart.p
    for a0, a1, a2, b in H.equations:                # inside: a . u + b <= 0
        a = np.array([a0, a1, a2])
        out.append(-(Bm @ a + b * c))
    return np.unique(np.round(np.array(out), 12), axis=0)


# ---------------------------------------------------------------- cutting on the sphere

def clip_polygon(P, H):
    """A convex polygon of points on R^4 (straight in every gnomonic chart) cut by X . h >= 0."""
    for h in H:
        if not P:
            return []
        out = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            fa, fb = a @ h, b @ h
            if fa >= 0:
                out.append(a)
            if (fa >= 0) != (fb >= 0):
                out.append(a + fa / (fa - fb) * (b - a))
        P = out
    return P


def clip_tris(X, F, H):
    """Triangles (vertices X on R^4, faces F) cut by the half-spaces: a list of triangles (n,3,4)."""
    S = X @ H.T
    allin = (S[F] >= -1e-12).all((1, 2))
    out = [X[F[allin]]]
    for t in np.flatnonzero(~allin & ~(S[F] < -1e-12).all(1).any(1)):
        P = clip_polygon([X[i] for i in F[t]], H)
        for k in range(1, len(P) - 1):
            out.append(np.array([[P[0], P[k], P[k + 1]]]))
    return np.concatenate(out) if out else np.zeros((0, 3, 4))


def clip_line(P, H):
    """A polyline on R^4 cut by the half-spaces: its runs inside."""
    runs, cur = [], []
    for a, b in zip(P[:-1], P[1:]):
        lo, hi = 0.0, 1.0
        for h in H:
            fa, fb = a @ h, b @ h
            if fa < 0 and fb < 0:
                lo, hi = 1, 0
                break
            if fa < 0:
                lo = max(lo, fa / (fa - fb))
            elif fb < 0:
                hi = min(hi, fa / (fa - fb))
        if hi <= lo:
            if len(cur) > 1:
                runs.append(np.array(cur))
            cur = []
            continue
        p, q = a + lo * (b - a), a + hi * (b - a)
        if not cur or np.linalg.norm(cur[-1] - p) > 1e-12:
            if len(cur) > 1:
                runs.append(np.array(cur))
            cur = [p]
        cur.append(q)
    if len(cur) > 1:
        runs.append(np.array(cur))
    return runs


def orient(F):
    """Faces turned so neighbours agree (each shared edge run opposite ways in the two), so the
    surface is shaded as one sheet."""
    from collections import deque
    F = [list(f) for f in F]
    by = defaultdict(list)
    for t, f in enumerate(F):
        for k in range(3):
            by[frozenset((f[k], f[(k + 1) % 3]))].append(t)
    seen = [False] * len(F)
    for t0 in range(len(F)):
        if seen[t0]:
            continue
        seen[t0] = True
        q = deque([t0])
        while q:
            t = q.popleft()
            f = F[t]
            for k in range(3):
                a, b = f[k], f[(k + 1) % 3]
                for u in by[frozenset((a, b))]:
                    if u == t or seen[u]:
                        continue
                    g = F[u]
                    same = any(g[j] == a and g[(j + 1) % 3] == b for j in range(3))
                    if same:
                        F[u] = [g[0], g[2], g[1]]
                    seen[u] = True
                    q.append(u)
    return F


def mesh(Q, nd):
    """Triangles (n,3,3) as shared vertices and faces, rounded to nd places, turned consistently."""
    Q = np.round(Q, nd).reshape(-1, 3)
    V, inv = np.unique(Q, axis=0, return_inverse=True)
    F = inv.ravel().reshape(-1, 3)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return V.tolist(), orient(F.tolist())


def rounded(o, n):
    if isinstance(o, float):
        return round(o, n)
    if isinstance(o, dict):
        return {k: rounded(v, n) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rounded(v, n) for v in o]
    return o


def thin(P, tol):
    """Points of a polyline kept where it turns by more than tol (ends kept)."""
    P = np.asarray(P, float)
    if len(P) <= 2:
        return P
    keep = [0]
    for i in range(1, len(P) - 1):
        a, c = P[keep[-1]], P[i + 1]
        d = c - a
        L = np.linalg.norm(d)
        x = P[keep[-1] + 1:i + 1] - a
        if L < 1e-12 or np.max(np.linalg.norm(x - np.outer(x @ d / L ** 2, d), axis=1)) > tol:
            keep.append(i)
    keep.append(len(P) - 1)
    return P[keep]


# ---------------------------------------------------------------- the page

def main(src_dir, out_path):
    src_dir = Path(src_dir)
    doc = json.loads((src_dir / "domain_pieces.json").read_text())
    z = np.load(src_dir / "domain_pieces.npz")
    html = Path(out_path).read_text()
    pos = blocks(html)
    DC = json.loads(html[slice(*pos["atlas-data"])])
    V1 = json.loads(html[slice(*pos["atlas-v1"])])
    CH = json.loads(html[slice(*pos["atlas-chambers"])])
    names = {t["id"]: t["name"] for t in DC["types"]}
    by = {t["id"]: t for t in DC["types"]}
    G = np.stack(group_elements(named_symmetry("h4_swirlprism")))

    cell, err = fit_cell_chart(DC)
    v1c, share, dplanes = fit_v1_chart(DC, V1, G)
    chplanes = chamber_planes(CH, v1c)
    print(f"charts: half-cell fitted to the page's seeds within {err:.1e}; dodecahedron: "
          f"{share:.0%} of the page's samples there matched", flush=True)

    verts, tris = z["verts"], z["tris"]
    ep, el = z["edge_pts"], z["edge_len"]
    eoff = np.r_[0, np.cumsum(el)]
    edge_pts = lambda i: ep[eoff[i]:eoff[i + 1]]
    E = doc["edges"]

    # every face: its triangles on the sphere, its drawn edges
    faces = []
    for f in doc["faces"]:
        Xv = to_sphere(verts[f["verts"][0]:f["verts"][1]])
        F = tris[f["tris"][0]:f["tris"][1]]
        drawn = []
        for i in f["edges"]:
            if E[i]["kind"] in ("line", "meet", "cross"):
                drawn.append((E[i]["kind"], E[i]["name"], to_sphere(edge_pts(i)), i))
        a, b = f["sides"]
        lines = sorted({n for k, n, _, _ in drawn if k == "line"})
        meets = sorted({n for k, n, _, _ in drawn if k != "line"})
        hv = (f"{f['wall']} · {names.get(f['wall'], f['wall'])}<br>wall piece between {a} and {b}"
              + (f"<br>edges on {', '.join(lines)}" if lines else "")
              + (f"<br>met by {', '.join(meets)}" if meets else ""))
        rec = {"f": f, "X": Xv, "F": F, "drawn": drawn, "hover": hv}
        if "coarse_verts" in f:                    # (the coarser mesh, for the many-image views)
            rec["Xc"] = to_sphere(z["coarse_verts"][f["coarse_verts"][0]:f["coarse_verts"][1]])
            rec["Fc"] = z["coarse_tris"][f["coarse_tris"][0]:f["coarse_tris"][1]]
        faces.append(rec)

    # an edge two pieces of one wall share is where the wall is cut (dashed: the wall goes on across
    # it); an edge only one piece of the wall has is where the wall ends (solid). Each drawn once.
    ecount = defaultdict(int)
    for fc in faces:
        for i in {d[3] for d in fc["drawn"]}:
            ecount[(fc["f"]["wall"], i)] += 1

    # (a face whose mesh is far denser than the rest, its rim's short steps crowding it: shown with
    # its coarser mesh in the half-cell too)
    Bz = np.linalg.svd(np.ones((1, 4)))[2][1:]
    dens = []
    for fc in faces:
        Xb = verts[fc["f"]["verts"][0]:fc["f"]["verts"][1]] @ Bz.T
        F_ = fc["F"]
        a = np.linalg.norm(np.cross(Xb[F_[:, 1]] - Xb[F_[:, 0]], Xb[F_[:, 2]] - Xb[F_[:, 0]]), axis=1).sum() / 2
        dens.append(len(F_) / max(a, 1e-12))
    med = float(np.median(dens))
    for fc, d_ in zip(faces, dens):
        fc["dense"] = d_ > 1.8 * med and "Xc" in fc

    def outlines(fc, draw, done):
        out = []
        w = fc["f"]["wall"]
        for kind, nm, P, i in fc["drawn"]:
            if (w, i) in done:
                continue
            done.add((w, i))
            cut = ecount[(w, i)] >= 2
            for Q in draw(P):
                if len(Q) < 2:
                    continue
                what = (f"on {nm}" if kind == "line" else f"where {nm} meets it")
                out.append({"id": w, "outline": True, "copy": cut, "pts": Q,
                            "hover": f"{w} · {'cut ' + what + ', between its pieces' if cut else 'edge of the wall, ' + what}"})
        return out

    # the named lines: every stretch once
    seen, line_runs = set(), []
    for i, e in enumerate(E):
        if e["kind"] != "line":
            continue
        P = edge_pts(i)
        key = (e["name"], tuple(np.round(np.sort(np.r_[P[0], P[-1], P[len(P) // 2]]), 6)))
        if key in seen:
            continue
        seen.add(key)
        line_runs.append((e["name"], to_sphere(P)))

    def build_view(chart, planes, nd, line_tol, images, inside_only=False):
        xwalls, xlines, segments, rings, main_ = [], [], [], [], []
        shells = defaultdict(lambda: {"tris": [], "rim": [], "walls": set()})
        drawn_once = {}
        for fc in faces:
            parts = []
            X0, F0 = (fc["Xc"], fc["Fc"]) if (len(images) > 1 or fc.get("dense")) and "Xc" in fc else (fc["X"], fc["F"])
            for g in images:
                Xg = X0 @ g.T
                if planes is not None:
                    if not (Xg @ planes.T >= -1e-9).all(1).any():
                        continue
                    Q = clip_tris(Xg, F0, planes)
                else:
                    Q = Xg[F0]
                if len(Q):
                    parts.append((g, Q))
            if not parts:
                continue
            Q3 = chart(np.concatenate([q for _, q in parts]).reshape(-1, 4)).reshape(-1, 3, 3)
            v_, f_ = mesh(Q3, nd)
            xwalls.append({"id": fc["f"]["wall"], "hover": fc["hover"], "copy": False, "verts": v_, "faces": f_})
            # (a piece lying in a face of the half-cell has only its inside side in this view's domain)
            sides = fc["f"]["sides"][:1] if inside_only and fc["f"]["sheet"].startswith("face:") else fc["f"]["sides"]
            for g, _ in parts:
                draw = (lambda P, g=g: [chart(R) for R in (clip_line(P @ g.T, planes) if planes is not None else [P @ g.T])])
                done = drawn_once.setdefault(id(g), set())
                for o in outlines(fc, draw, done):
                    xlines.append({**o, "pts": thin(o["pts"], line_tol).tolist()})
                # (a region's edges: every drawn edge of its pieces, each piece's own)
                for kind, nm, P, i in fc["drawn"]:
                    for Q in draw(P):
                        for r in sides:
                            shells[r]["rim"].append(thin(Q, line_tol).tolist())
            for r in set(sides):
                shells[r]["tris"].append(Q3)
                shells[r]["walls"].add(fc["f"]["wall"])
        for nm, P in line_runs:
            for g in images:
                runs = clip_line(P @ g.T, planes) if planes is not None else [P @ g.T]
                for R in runs:
                    Q = thin(chart(R), line_tol / 4).tolist()
                    if nm in RINGS:
                        rings.append({"id": nm, "pts": Q})
                    elif nm == "B1":
                        main_.append({"id": nm, "pts": Q})
                    else:
                        hv = f"{nm} · {names.get(nm, nm)}<br>named line (each point on its two exact surfaces)"
                        if nm in SEL_ONLY or by.get(nm, {}).get("listed"):
                            segments.append({"id": nm, "copy": False, "selOnly": nm in SEL_ONLY, "pts": Q, "hover": hv})
                        else:
                            xlines.append({"id": nm, "copy": False, "pts": Q, "hover": hv})
        return {"xwalls": xwalls, "xlines": xlines, "segments": segments, "rings": rings, "main": main_}, shells

    ident = [np.eye(4)]
    cv, cshell = build_view(cell, None, 4, 1e-3, ident, inside_only=True)
    vv, vshell = build_view(v1c, dplanes, 3, 8e-3, list(G))
    chv, chshell = build_view(v1c, chplanes, 3, 6e-3, list(G))

    region_files = {}
    for vk, view, shells in (("cell", DC, cshell), ("v1", V1, vshell), ("chambers", CH, chshell)):
        view.update({k: v for k, v in (cv if vk == "cell" else vv if vk == "v1" else chv).items()})
        view["regionShells"], view["regionEdges"] = {}, {}
        ids = []
        for r, sh in shells.items():
            if by.get(r, {}).get("dim") != "region":
                continue
            v_, f_ = mesh(np.concatenate(sh["tris"]), 4 if vk == "cell" else 3)
            region_files[f"{vk}/{r}.json"] = {"verts": v_, "faces": f_, "walls": sorted(sh["walls"])}
            if vk == "cell":
                view["regionEdges"][r] = sh["rim"]
            ids.append(r)
        view["regionWallsIds"] = sorted(ids)
        view["tpatches"] = []

    dump = lambda o, n: json.dumps(rounded(o, n), separators=(",", ":"))
    rdir = Path(out_path).parent / "symmetry_domain_regions"
    shutil.rmtree(rdir, ignore_errors=True)
    for f, m in region_files.items():
        (rdir / f).parent.mkdir(parents=True, exist_ok=True)
        (rdir / f).write_text(dump(m, 4 if f.startswith("cell/") else 3))
    new = {"atlas-data": dump(DC, 4), "atlas-v1": dump(V1, 3), "atlas-chambers": dump(CH, 3)}
    for name in sorted(new, key=lambda n: -pos[n][0]):
        s, e = pos[name]
        html = html[:s] + new[name] + html[e:]
    # what is not drawn yet: walls with a patch of the map on a surface where no piece was built
    patches = json.loads((src_dir / "patches.json").read_text())["patches"]
    have = defaultdict(set)
    for f in doc["faces"]:
        have[f["sheet"]].add(f["wall"])
    whole, part = set(), defaultdict(int)
    built = {f["wall"] for f in doc["faces"]}
    for p in patches:
        if not p["wall"]:
            continue
        sheet = str(p["surface"]) if p["face"] is None else "face:" + p["face"]
        if p["wall"] not in built:
            whole.add(p["wall"])
        elif p["wall"] not in have[sheet]:
            part[(p["wall"], sheet)] += 1
    partial = sorted({w for w, _ in part})
    note = ("Walls are drawn from the 3-sphere map's wall pieces, each bounded by its exact edges. "
            + (f"Not drawn yet: {', '.join(sorted(whole))}" if whole else "")
            + (("; " if whole else "Not drawn yet: ") + f"part of {', '.join(partial)}" if partial else "")
            + ("." if whole or partial else ""))
    tag = f'<p id="dv-note" class="lede">{note}</p>'
    html = re.sub(r'<p id="dv-note"[^>]*>.*?</p>', "", html)
    anchor = "map from several fundamental domains."
    i = html.find(anchor)
    if i >= 0:
        j = html.find("</p>", i) + 4
        html = html[:j] + "\n      " + tag + html[j:]
    print(note, flush=True)
    Path(out_path).write_text(html)
    n = lambda D: sum(len(w["faces"]) for w in D["xwalls"])
    print(f"{out_path}: {len(html) / 1e6:.1f} MB; wall pieces {len(DC['xwalls'])}; triangles: half-cell {n(DC)}, "
          f"dodecahedron {n(V1)}, tenth {n(CH)}; regions {len(DC['regionWallsIds'])}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
