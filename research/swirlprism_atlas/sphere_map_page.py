"""The Symmetry Domain page's walls, lines and regions, drawn from the 3-sphere map's exact pieces.

The page (assets/symmetry_domain.html) keeps everything else it shows (the frame, mirrors, axes, the fundamental
domain, the regular cells, the samples, the key); what is redrawn comes from the sphere map built on the branch
claude/sharp-rubin-c0puiw (data/swirlprism/sphere_map/domain_mesh.npz and .json, `sphere_map.domain_mesh`), which
follows DOMAIN_PLAN.md:

  - the curves are exact: every wall patch is the part of its exact surface where every other surface has the sign
    its two pieces share; its rim is solved exactly onto the surface and the crossing surface (and onto both
    crossing surfaces at its corners);
  - the pieces come before any mesh: a wall piece is a connected part of one wall, cut only where another wall meets
    it or a named line crosses it (over all 1200 copies, as the sphere map counts them); each is one mesh;
  - each piece is triangulated inside its exact rim, every vertex on the exact surface;
  - a piece's rim on a named line is drawn solid (the wall's edge), where another wall meets it dashed; seams (where
    the wall goes on: no curve) and the half-cell's faces are not drawn;
  - lines: every named line's stretches in the half-cell (the sphere map's line pieces), every point solved onto the
    line's two exact surfaces; the cross ring (C2a, C2b, C3, D1, D2, D3) and the main ring B1 as rings;
  - a region's shell is the wall pieces with it on one side, its edges their drawn rims.
The dodecahedron views (V1 and the chambers) carry the same pieces there under the group.

    python sphere_map_page.py <sphere_map data dir> ../../assets/symmetry_domain.html
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import chamber_domain
import dodeca_view as dv
from cellframe import T as TCELL
from core import CHART

HALF = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, -1.0]])
CORNERS = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, .5, .5]])
G_CELL = 0.016          # interior vertices merged on this grid (betas); rim vertices kept exactly
G_CELL_RIM = 0.004     # (rim vertices merged on this finer grid: each kept vertex an exact rim point)
G_DODECA = 0.2         # (... for the dodecahedron views, which hold about 20 half-cells)
G_DODECA_RIM = 0.03   # (... and their rims, small at that scale)
LINE_TOL = 1e-3         # polylines kept where they turn by more than this (view coordinates)
RINGS = ("C2a", "C2b", "C3", "D1", "D2", "D3")
SEL_ONLY = {"T1", "E1", "E2"}


# ---------------------------------------------------------------- projection (as the page always drew it)

def frame():
    A = CORNERS @ TCELL
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    n = np.linalg.solve(A, np.ones(4))
    p = n / np.linalg.norm(n)
    U = np.linalg.svd(p[None, :])[2][1:]
    X = A / (A @ p)[:, None]
    return p, U, float(np.linalg.norm(X @ U.T, axis=1)[0])


PV, UV, RV = frame()


def _gnom(B):
    X = np.atleast_2d(np.asarray(B, float)) @ TCELL
    return (X / (X @ PV)[:, None]) @ UV.T / RV


def turn():
    F = _gnom([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, .5, .5]])
    z = np.cross(F[1] - F[0], F[2] - F[0])
    z /= np.linalg.norm(z)
    inner = _gnom([[.3, .3, .3, .1]])[0]
    if (inner - F[0]) @ z > 0:
        z = -z
    c = _gnom(CORNERS).mean(0)
    y = _gnom([[1, 0, 0, 0]])[0] - c
    y -= (y @ z) * z
    y /= np.linalg.norm(y)
    return np.vstack([np.cross(y, z), y, z]), c


ROT, CEN = turn()


def flip():
    m = (_gnom([[0, 0, .5, .5]])[0] - CEN) @ ROT.T
    m = np.array([-m[0], m[1]])
    a = np.arctan2(m[0], m[1])
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rz @ np.diag([-1.0, 1, 1])


FLIP = flip()


def view(B):
    return (_gnom(B) - CEN) @ ROT.T @ FLIP.T


def chart(B):
    B = np.atleast_2d(np.asarray(B, float))
    return (B / B.sum(1, keepdims=True)) @ CHART


# ---------------------------------------------------------------- meshes and polylines

def coarsen(T, G, GB=None):
    """Triangles (n,3,4) as vertices and faces, interior vertices merged on a grid G in betas (rim vertices, on the
    piece's boundary, kept exactly where they are)."""
    Vu, inv = np.unique(T.reshape(-1, 4).round(12), axis=0, return_inverse=True)
    F = inv.ravel().reshape(-1, 3)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    e, c = np.unique(E, axis=0, return_counts=True)
    bnd = np.zeros(len(Vu), bool)
    bnd[e[c == 1].ravel()] = True
    key = np.round(Vu[:, :3] / G).astype(np.int64)
    key[bnd] = (np.round(Vu[bnd, :3] / GB).astype(np.int64) if GB
                else np.c_[np.flatnonzero(bnd), np.zeros((bnd.sum(), 2), np.int64)])
    key = np.c_[key, bnd]
    _, first, cl = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V, I = Vu[first], cl.ravel()[F]
    good = (I[:, 0] != I[:, 1]) & (I[:, 1] != I[:, 2]) & (I[:, 0] != I[:, 2])
    I = I[good]
    _, k = np.unique(np.sort(I, 1), axis=0, return_index=True)
    return V, I[np.sort(k)]


def chains(segs, nd=8):
    key = lambda p: tuple(np.round(p, nd) + 0.0)
    uniq = {}
    for a, b in segs:
        ka, kb = key(a), key(b)
        if ka != kb:
            uniq.setdefault(tuple(sorted((ka, kb))), (a, b))
    segs = list(uniq.values())
    adj = defaultdict(list)
    for n, (a, b) in enumerate(segs):
        adj[key(a)].append(n)
        adj[key(b)].append(n)
    used, out = set(), []
    order = sorted(range(len(segs)), key=lambda n: min(len(adj[key(segs[n][0])]), len(adj[key(segs[n][1])])))
    for n in order:
        if n in used:
            continue
        used.add(n)
        line = [segs[n][0], segs[n][1]]
        for end in (1, 0):
            while True:
                at = key(line[-1] if end else line[0])
                m = next((m for m in adj.get(at, []) if m not in used), None)
                if m is None:
                    break
                used.add(m)
                a, b = segs[m]
                nxt = b if key(a) == at else a
                line.append(nxt) if end else line.insert(0, nxt)
        out.append(np.array(line))
    return out


def thin_line(P, tol):
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


def mesh(Q, nd):
    at, verts, faces = {}, [], []
    for t in Q:
        f = []
        for p in t:
            k = tuple(round(float(c), nd) for c in p)
            if k not in at:
                at[k] = len(verts)
                verts.append(list(k))
            f.append(at[k])
        if len(set(f)) == 3:
            faces.append(f)
    return verts, faces


def rounded(o, n):
    if isinstance(o, float):
        return round(o, n)
    if isinstance(o, dict):
        return {k: rounded(v, n) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rounded(v, n) for v in o]
    return o


# ---------------------------------------------------------------- the page

def blocks(html):
    out = {}
    for m in re.finditer(r'<script id="([^"]+)" type="application/json">', html):
        s = m.end()
        out[m.group(1)] = (s, html.index("</script>", s))
    return out


def main(src_dir, out_path):
    src_dir = Path(src_dir)
    doc = json.loads((src_dir / "domain_mesh.json").read_text())
    z = np.load(src_dir / "domain_mesh.npz")
    html = Path(out_path).read_text()
    pos = blocks(html)
    DC = json.loads(html[slice(*pos["atlas-data"])])
    V1old = json.loads(html[slice(*pos["atlas-v1"])])
    CHold = json.loads(html[slice(*pos["atlas-chambers"])])
    by = {t["id"]: t for t in DC["types"]}
    work = json.load(open("working_data.json"))
    names = {k: v["name"] for k, v in work["shapes"].items()}

    tris, tp = z["tris"].astype(float), z["tri_piece"]
    rims, ri = z["rims"].astype(float), z["rim_info"]
    kinds, rnames = doc["rim_kinds"], doc["rim_names"]
    order = np.argsort(tp, kind="stable")
    tris, tp = tris[order], tp[order]
    starts = np.searchsorted(tp, np.arange(len(doc["pieces"]) + 1))
    rim_by = defaultdict(list)
    for (a, b), (pc, kd, nm, cont, _p) in zip(rims, ri):
        rim_by[pc].append((kinds[kd], rnames[nm], bool(cont), a, b))

    cell = {"xwalls": [], "xlines": [], "segments": [], "regionEdges": {}}
    src = {"xwalls": [], "xlines": [], "segments": []}
    shells = defaultdict(lambda: {"tris": [], "rim": []})
    region_walls = defaultdict(set)
    n_pieces = defaultdict(int)
    for pc in doc["pieces"]:
        n_pieces[pc["wall"]] += 1
    for pc in doc["pieces"]:
        k, w = pc["id"], pc["wall"]
        Tb = tris[starts[k]:starts[k + 1]]
        if not len(Tb):
            continue
        between = pc["between"][0].split("|") if len(pc["between"]) == 1 else None
        hv = (f"{w} · {names.get(w, w)}<br>wall piece between {' and '.join(between) if between else ', '.join(pc['between'])}"
              + (f"<br>on the 3-sphere: {pc['classes_on_sphere']} pieces like it" if pc.get("classes_on_sphere") else "")
              + f"<br>{n_pieces[w]} pieces of {w} in the half-cell")
        V, F = coarsen(Tb, G_CELL, G_CELL_RIM)
        Q = view(V[F].reshape(-1, 4)).reshape(-1, 3, 3)
        verts, faces = mesh(Q, 4)
        cell["xwalls"].append({"id": w, "hover": hv, "copy": False, "verts": verts, "faces": faces})
        Vd, Fd = coarsen(Tb, G_DODECA, G_DODECA_RIM)
        src["xwalls"].append({"id": w, "hover": hv, "copy": False,
                              "tris": chart(Vd[Fd].reshape(-1, 4)).reshape(-1, 3, 3).tolist()})
        drawn = defaultdict(list)
        for kind, nm, cont, a, b in rim_by[k]:
            if kind in ("line", "meet"):
                drawn[(kind, nm)].append((a, b))
        for (kind, nm), segs in drawn.items():
            for line in chains(segs):
                if kind == "meet":
                    rec = {"id": w, "outline": True, "copy": True,
                           "hover": f"{w} · where {nm} meets it" if nm else f"{w} · where another wall meets it"}
                else:
                    rec = {"id": w, "outline": True, "copy": False,
                           "hover": f"{w} · edge of the wall, on {nm}" if nm else f"{w} · edge of the wall"}
                cell["xlines"].append({**rec, "pts": thin_line(view(line), LINE_TOL / 2).tolist()})
                src["xlines"].append({**rec, "pts": thin_line(chart(line), LINE_TOL / 6).tolist()})
        for r in set(between or []):
            shells[r]["tris"].append(Q)
            shells[r]["rim"] += [s for segs in drawn.values() for s in segs]
            region_walls[r].add(w)
            shells[r]["src"] = shells[r].get("src", []) + [Vd[Fd]]

    # lines
    lp, ll = z["line_pts"].astype(float), z["line_len"]
    offs = np.r_[0, np.cumsum(ll)]
    rings, main_ = [], []
    for i, L in enumerate(doc["lines"]):
        P = lp[offs[i]:offs[i + 1]]
        name = L["line"]
        if name in RINGS:
            rings.append({"id": name, "pts": view(P).tolist(), "_b": P})
            continue
        if name == "B1":
            main_.append({"id": name, "pts": view(P).tolist(), "_b": P})
            continue
        t = by.get(name, {})
        hv = f"{name} · {names.get(name, name)}<br>exact line (each point on its two surfaces)"
        key = "segments" if name in SEL_ONLY or t.get("listed") else "xlines"
        extra = {"selOnly": name in SEL_ONLY} if key == "segments" else {}
        cell[key].append({"id": name, "copy": False, **extra, "pts": thin_line(view(P), LINE_TOL / 4).tolist(), "hover": hv})
        src[key].append({"id": name, "copy": False, **extra, "pts": thin_line(chart(P), LINE_TOL / 12).tolist(), "hover": hv})
    DC["rings"] = [{"id": r["id"], "pts": thin_line(r["pts"], LINE_TOL / 4).tolist()} for r in rings]
    DC["main"] = [{"id": r["id"], "pts": thin_line(r["pts"], LINE_TOL / 4).tolist()} for r in main_]

    # regions
    region_files = {}
    for r, sh in shells.items():
        if by.get(r, {}).get("dim") != "region":
            continue
        verts, faces = mesh(np.concatenate(sh["tris"]), 4)
        region_files[f"cell/{r}.json"] = {"verts": verts, "faces": faces, "walls": sorted(region_walls[r])}
        cell["regionEdges"][r] = [thin_line(view(c), LINE_TOL).tolist() for c in chains(sh["rim"], 7)]
        src["xwalls"].append({"id": f"__region_{r}", "hover": "",
                              "tris": chart(np.concatenate(sh["src"]).reshape(-1, 4)).reshape(-1, 3, 3).tolist()})
    DC.update(cell)
    DC["regionShells"], DC["regionWallsIds"] = {}, sorted(f.split("/")[1][:-5] for f in region_files)

    # the dodecahedron views: the same walls and lines under the group, in each view
    empty = {"samples": {}, "uniform": [], "special": [], "rings": [], "main": [], "qaxes": [], "regular": [],
             "tpatches": [], "mirrors": [], "fdomain": [], "fcentre": [0, 0, 0]}
    S = {**empty, **src}
    S["rings"] = [{"id": r["id"], "pts": chart(r["_b"]).tolist()} for r in rings]
    S["main"] = [{"id": r["id"], "pts": chart(r["_b"]).tolist()} for r in main_]
    dv.use_centre((1, 0, 0, 0))
    v1 = dv.view(S)
    ch = chamber_domain.view(v1)
    dv.use_centre((0, 0, 1, 1))
    length = lambda P: float(np.linalg.norm(np.diff(np.asarray(P, float), axis=0), axis=1).sum())
    for vk, old, new in (("v1", V1old, v1), ("chambers", CHold, ch)):
        dv.tidy_lines(new, keys=("rings", "main", "segments", "xlines"))
        old["rings"], old["main"] = new["rings"], new["main"]
        dv.split_edges_on_rings(old)
        for w in [w for w in new["xwalls"] if w["id"].startswith("__region_")]:
            verts, faces = mesh(w["tris"], 3)
            r = w["id"][9:]
            region_files[f"{vk}/{r}.json"] = {"verts": verts, "faces": faces, "walls": sorted(region_walls[r])}
        old["regionWallsIds"] = sorted(f.split("/")[1][:-5] for f in region_files if f.startswith(vk + "/"))
        old["xwalls"] = []
        for w in new["xwalls"]:
            if not w["id"].startswith("__region_"):
                verts, faces = mesh(w["tris"], 3)
                old["xwalls"].append({**{a: b for a, b in w.items() if a != "tris"}, "verts": verts, "faces": faces})
        old["xlines"] = [{**s, "pts": thin_line(s["pts"], 4 * LINE_TOL).tolist()} for s in new["xlines"]
                         if length(s["pts"]) >= 0.03]
        old["segments"] = [{**s, "pts": thin_line(s["pts"], LINE_TOL / 4).tolist()} for s in new["segments"]]
        old["regionEdges"], old["regionShells"] = {}, {}

    dump = lambda o: json.dumps(rounded(o, 4), separators=(",", ":"))
    dump3 = lambda o: json.dumps(rounded(o, 3), separators=(",", ":"))
    import shutil
    rdir = Path(out_path).parent / "symmetry_domain_regions"
    shutil.rmtree(rdir, ignore_errors=True)
    for f, m in region_files.items():
        (rdir / f).parent.mkdir(parents=True, exist_ok=True)
        (rdir / f).write_text(dump(m) if f.startswith("cell/") else dump3(m))
    new_blocks = {"atlas-data": dump(DC), "atlas-v1": dump3(V1old), "atlas-chambers": dump3(CHold)}
    for name in sorted(new_blocks, key=lambda n: -pos[n][0]):
        s, e = pos[name]
        html = html[:s] + new_blocks[name] + html[e:]
    Path(out_path).write_text(html)
    n = lambda D: sum(len(w["faces"]) for w in D["xwalls"])
    print(f"{out_path}: {len(html) / 1e6:.1f} MB; wall pieces {len(DC['xwalls'])}; triangles: cell {n(DC)}, "
          f"dodecahedron {n(V1old)}, chambers {n(CHold)}; regions {len(DC['regionWallsIds'])}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
