"""Merged piece counts on the whole 3-sphere: wall pieces (faces) and edges.

Edges are the atlas's line pieces (`atlas.json: line_pieces`, cut at their exact meeting points
and joined across copies) together with every wall-wall meeting curve (`wall_meets.json`) that
lies on no atlas line. A meeting curve is rebuilt exactly (traced on its two surfaces) between
the exact meeting points at its ends, and identified with the other records of the same curve by
its images under the group: two stretches are one edge class when some element carries one onto
the other. The number of pieces on the sphere is the number of distinct images of a class's
representative under all 1200 elements.

Wall pieces are the faces of `atlas.json` (a piece of one wall between two region volumes, joined
across copies), counted per wall and per pair of regions it separates.
"""
from __future__ import annotations

import json

import numpy as np

from .chords import FACE_N, FACE_NAMES
from .frame import data_dir, group, point_from_beta, source_dir
from .wall_meets import resolve, trace

ON_LINE = 1e-7          # (R^4) a meeting stretch lies on an atlas line if every point is this close
END = 1e-9              # named end points lie exactly on the curve


def _key(P):
    """An orientation-free key for a curve on S^3 (R^4 polyline): its ends and its middle."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    mid = P[int(np.searchsorted(d, d[-1] / 2))]
    ends = sorted(tuple(np.round(p, 5) + 0.0) for p in (P[0], P[-1]))
    return (ends[0], ends[1], tuple(np.round(mid, 3) + 0.0))


def _orbit_keys(P, g):
    imgs = np.einsum("gij,nj->gni", g, P)
    return {_key(Q) for Q in imgs}


def _named_points():
    pts = json.loads((source_dir() / "exact_points.json").read_text())["points"]
    return [(p["name"], np.asarray(p["float"], float)) for p in pts]


def _atlas_line_points():
    """Every named line's pieces in the domain (the central database's line pieces, mirror twins
    included), densely, in R^4."""
    atlas = json.loads((data_dir() / "sphere_map" / "atlas.json").read_text())
    g_all = group()
    out = []
    for lp in atlas["line_pieces"]:
        for seg in lp["segments"]:
            poly = np.asarray(seg["polyline_beta"], float)
            poly = poly / poly.sum(axis=1, keepdims=True)
            fine = [poly[0]]
            for a, b in zip(poly[:-1], poly[1:]):
                n = max(1, int(np.ceil(np.linalg.norm(b - a) / 1e-3)))
                fine += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
            out.append((lp["line_shape"], point_from_beta(np.array(fine)) @ g_all[seg["copy"]].T))
    return out


def corner_points(atlas, g):
    """Every image on S^3 of every exact meeting point of the atlas (exact_points.json), as a
    KD-tree (the vertices where edge pieces stop), with the name of each image."""
    from scipy.spatial import cKDTree
    named = _named_points()
    P = np.array([point_from_beta(x) for _, x in named])
    class _Tree(cKDTree):
        pass
    tree = _Tree(np.einsum("gij,nj->gni", g, P).reshape(-1, 4))
    tree.names = [n for _ in range(len(g)) for n, _ in named]
    return tree


def _signature(r, keep):
    """What surrounds a curve point, as the walls named round it with their regions (names only,
    so the same in every copy)."""
    return tuple(sorted((str(w["wall"]), tuple(sorted(map(str, w["between"]))))
                        for w in r["walls"])) if r else ()


def _pulled(surf, M):
    """Surface surf of copy k, in copy 0's coordinates (beta_k = beta . M): value, gradient."""
    return lambda x: (float(surf.f((x @ M)[None])[0]),
                      (lambda gr: gr - gr.mean())(M @ surf.grad((x @ M)[None])[0]))


def meeting_pieces(w1, w2, inc, wall_surfs, log=print, gaps=None):
    """Pieces (in the domain) of the curve where walls w1 and w2 meet: every pair of their own
    surfaces traced, each arc cut exactly where another library surface crosses it, and the
    stretches where both walls are named round the curve kept, joined where what surrounds them
    is the same. Each piece: beta polyline, its signature, and its two ends ('face' where the
    arc leaves the domain, else 'corner')."""
    from .wall_meets import _sf, _solve
    named_beta = [(n, x / x.sum()) for n, x in _named_points()]
    out = []
    gaps = [] if gaps is None else gaps
    lib = inc.surfs
    for sa in wall_surfs[w1]:
        for sb in wall_surfs[w2]:
            a = lib[sa] if isinstance(sa, int) else resolve(sa)
            b = lib[sb] if isinstance(sb, int) else resolve(sb)
            an, bn = a.name, b.name
            if a.kind == b.kind == "plane" and abs(abs(a.normal @ b.normal) - 1) < 1e-9:
                continue
            for arc in trace(a, b):
                P = arc["points"] / arc["points"].sum(axis=1, keepdims=True)
                # the march can step back and forth on its last step at a face: drop that
                while len(P) > 3 and np.linalg.norm(P[-1] - P[-3]) < 1e-9:
                    P = P[:-2]
                while len(P) > 3 and np.linalg.norm(P[0] - P[2]) < 1e-9:
                    P = P[2:]
                if len(P) < 3:
                    continue
                # the march stops short of a point where many surfaces meet (its step fails there):
                # an end inside the domain next to an exact meeting point is that point
                if not arc["closed"]:
                    for side in (0, -1):
                        d, x = min(((np.linalg.norm(x - P[side]), x) for _, x in named_beta),
                                   key=lambda t: t[0])
                        if d < 1e-4:
                            P[side] = x               # (an end at an exact point: exactly on it)
                        elif (P[side] @ FACE_N.T).min() > 1e-9 and d < 2e-3:
                            P = np.vstack([P, x]) if side == -1 else np.vstack([x, P])
                # the surfaces that can cut the arc: copy 0's, and where the arc lies in the
                # domain's boundary, those of every copy on the other side, pulled back (M_k)
                frames = [np.eye(4)]
                if (np.abs(P @ FACE_N.T) < 1e-12).all(axis=0).any():
                    m = len(P) // 2
                    basis = np.linalg.svd(np.vstack([np.ones(4), P[m + 1] - P[m - 1]]))[2][2:]
                    frames += [M for k, (M, _, _) in inc._frames_round(P[m], basis).items() if k]
                cuts = []                                                   # (index, exact point)
                for M in frames:
                    Q = P @ M
                    V = np.stack([s.f(Q) for s in lib], axis=1)             # (n, S)
                    G = np.stack([np.linalg.norm(s.grad(Q), axis=1) for s in lib], axis=1) + 1e-300
                    rel = np.abs(V) / G
                    on = rel.max(axis=0) < 1e-9                             # through the whole arc
                    sg = np.sign(V)
                    for si in np.flatnonzero(~on):
                        real = rel[:, si] > 1e-11            # a value at zero has no sign
                        for i in np.flatnonzero((sg[:-1, si] * sg[1:, si] < 0) & real[:-1] & real[1:]):
                            t = V[i, si] / (V[i, si] - V[i + 1, si])
                            x, ok = _solve([_sf(a), _sf(b), _pulled(lib[si], M)],
                                           P[i] + t * (P[i + 1] - P[i]))
                            step = np.linalg.norm(P[i + 1] - P[i])
                            if not ok or (np.linalg.norm(x - P[i]) + np.linalg.norm(x - P[i + 1])
                                          > 1.01 * step + 1e-9):
                                continue              # solved onto some other crossing
                            if min(np.linalg.norm(x - P[0]), np.linalg.norm(x - P[-1])) < 1e-6:
                                continue              # at the arc's own end
                            cuts.append((int(i), x))
                cuts.sort(key=lambda c: (c[0], np.linalg.norm(c[1] - P[c[0]])))   # along the arc
                # sub-arcs between consecutive cut points (and the arc's own ends)
                knots = [(0, P[0], "face" if not arc["closed"] else "corner")]
                for i, x in cuts:                    # several surfaces crossing at one corner: one cut
                    if np.linalg.norm(x - knots[-1][1]) > 1e-6:
                        knots.append((i, x, "corner"))
                end = (len(P) - 1, P[-1], "face" if not arc["closed"] else "corner")
                if np.linalg.norm(end[1] - knots[-1][1]) <= 1e-6 and len(knots) > 1:
                    knots[-1] = (knots[-1][0], end[1], end[2] if end[2] == "face" else knots[-1][2])
                else:
                    knots.append(end)
                pieces = []
                for (i0, x0, k0), (i1, x1, k1) in zip(knots[:-1], knots[1:]):
                    seg = np.vstack([x0, P[i0 + 1:i1 + 1], x1]) if i1 > i0 else np.vstack([x0, x1])
                    if np.linalg.norm(np.diff(seg, axis=0), axis=1).sum() < 1e-6:
                        continue
                    m = len(seg) // 2
                    if len(seg) < 3:
                        mid, _ = _solve([_sf(a), _sf(b)], (x0 + x1) / 2)
                        tan = x1 - x0
                    else:
                        mid, tan = seg[m], seg[min(m + 1, len(seg) - 1)] - seg[max(m - 1, 0)]
                    r = inc.round_line(mid, tan)
                    names = {w["wall"] for w in r["walls"]} if r else set()
                    gap = bool(r) and any(sc["piece"] is None for sc in r["sectors"])
                    pieces.append({"seg": seg, "ends": [k0, k1], "present": w1 in names and w2 in names,
                                   "sig": _signature(r, None), "surfaces": [an, bn], "gap": gap})
                # a stretch with a sector the exploration never met (a sliver of the arrangement
                # missing from the map) carries no information: it takes the surroundings of the
                # stretches on either side when they agree
                for j, pc in enumerate(pieces):
                    if pc["gap"] and 0 < j < len(pieces) - 1:
                        a_, b_ = pieces[j - 1], pieces[j + 1]
                        if not a_["gap"] and a_["sig"] == b_["sig"]:
                            pc.update(sig=a_["sig"], present=a_["present"])
                            gaps.append({"walls": f"{w1}/{w2}", "at": np.round(pc["seg"][len(pc["seg"]) // 2], 5).tolist()})
                # join neighbours with the same surroundings
                run = None
                for pc in pieces:
                    if pc["present"] and run and run["sig"] == pc["sig"]:
                        run["seg"] = np.vstack([run["seg"], pc["seg"][1:]])
                        run["ends"][1] = pc["ends"][1]
                        continue
                    if run:
                        out.append(run)
                    run = dict(pc) if pc["present"] else None
                    if run:
                        run["ends"] = list(run["ends"])
                if run:
                    out.append(run)
    log(f"  {w1}/{w2}: {len(out)} domain pieces")
    return out


def _glue(pieces, g, corners, tol=1e-7):
    """Whole pieces on S^3: every domain piece carried to all copies (equal images merged), joined
    end to end where two images with the same surroundings meet at a point of the domain's
    boundary. Points are matched by distance (tol), not by rounding."""
    from scipy.spatial import cKDTree
    imgs = []                                         # (piece, start, mid, end) in R^4
    for pi, pc in enumerate(pieces):
        R = point_from_beta(pc["seg"])
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(R, axis=0), axis=1))]
        j = min(max(int(np.searchsorted(d, d[-1] / 2)), 1), len(R) - 1)
        t = (d[-1] / 2 - d[j - 1]) / max(d[j] - d[j - 1], 1e-300)
        mid = R[j - 1] + t * (R[j] - R[j - 1])
        for k in range(len(g)):
            imgs.append((pi, g[k] @ R[0], g[k] @ mid, g[k] @ R[-1]))
    n = len(imgs)
    # equal images: same piece class, same middle point, same ends (either way round)
    mids = np.array([m for _, _, m, _ in imgs])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)
    for i, j in cKDTree(mids).query_pairs(1e-4):
        (pa, a0, _, a1), (pb, b0, _, b1) = imgs[i], imgs[j]
        same_ends = (min(np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1),
                         np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0)) < 1e-6)
        if same_ends and pieces[pa]["sig"] == pieces[pb]["sig"]:
            union(i, j)
    # joins: at an end that is not one of the atlas's exact meeting points, the image ends meeting
    # there (after equal images are merged) are joined when what surrounds them is the same (the
    # curve goes on into the next copy, or nothing round it changes). Where what surrounds it
    # changes the signatures differ and nothing is joined.
    root = [find(i) for i in range(n)]
    pts, owner = [], []
    for ii, (pi, a, _, b) in enumerate(imgs):
        if root[ii] != ii:
            continue
        for side, p in ((0, a), (1, b)):
            if corners is not None and corners.query(p)[0] < 1e-7:
                continue
            pts.append(p)
            owner.append(ii)
    if pts:
        for i, j in cKDTree(np.array(pts)).query_pairs(tol):
            a, b = owner[i], owner[j]
            if a != b and pieces[imgs[a][0]]["sig"] == pieces[imgs[b][0]]["sig"]:
                union(a, b)
    comps = {}
    for ii in range(n):
        comps.setdefault(find(ii), set()).add(imgs[ii][0])
    cls = list(range(len(pieces)))

    def cf(i):
        while cls[i] != i:
            i = cls[i]
        return i
    for members in comps.values():
        ms = sorted(members)
        for m in ms[1:]:
            cls[cf(m)] = cf(ms[0])
    per_class = {}
    for members in comps.values():
        c = cf(next(iter(members)))
        per_class[c] = per_class.get(c, 0) + 1
    return per_class, cf


def build() -> dict:
    g = group()
    lines = _atlas_line_points()
    atlas = json.loads((data_dir() / "sphere_map" / "atlas.json").read_text())
    meets = json.loads((data_dir() / "sphere_map" / "wall_meets.json").read_text())["meetings"]

    # ---- meeting curves ------------------------------------------------------------------------
    from .incidence import Incidence
    inc = Incidence()
    # every library surface a wall's patches lie on (inside the half-cell a wall lies on images of
    # its atlas surface too, and in the domain's own faces), so every meeting curve is traced
    patches = json.loads((data_dir() / "sphere_map" / "patches.json").read_text())["patches"]
    face_plane = {}
    for i, sf in enumerate(inc.surfs):
        if sf.kind == "plane":
            for fi, fname in enumerate(FACE_NAMES):
                nf = FACE_N[fi] / np.linalg.norm(FACE_N[fi])
                if abs(abs(sf.normal @ nf) - 1) < 1e-9:
                    face_plane[fname] = i
    wall_surfs = {}
    for pt in patches:
        if not pt["wall"]:
            continue
        sid = pt["surface"] if pt["surface"] is not None else face_plane.get(pt["face"])
        if sid is not None:
            wall_surfs.setdefault(pt["wall"], set()).add(int(sid))
    wall_surfs = {w: sorted(v) for w, v in wall_surfs.items()}
    pairs = sorted({tuple(sorted((m["wall"], m["other"]))) for m in meets if m["found"]})
    corner_tree = corner_points(atlas, g)
    meeting_edges = []
    map_gaps = []
    from scipy.spatial import cKDTree
    by_line = {}
    for name, L in lines:
        by_line.setdefault(name, []).append(L)
    line_trees = {n: cKDTree(np.vstack(v)) for n, v in by_line.items()}
    for w1, w2 in pairs:
        pcs = meeting_pieces(w1, w2, inc, wall_surfs, gaps=map_gaps)
        if not pcs:
            meeting_edges.append({"walls": f"{w1}/{w2}", "found": False})
            continue
        per_class, cf = _glue(pcs, g, corner_tree)
        for c, n in sorted(per_class.items()):
            members = [i for i in range(len(pcs)) if cf(i) == c]
            on = None
            # on a named line: some copy of one of its pieces carries the whole stretch (tested on
            # the sphere, not in the domain's coordinates: a stretch on the domain's boundary is
            # a copy of the line's recorded points, not those points)
            X = np.vstack([point_from_beta(pcs[i]["seg"]) for i in members])[::3]
            for name, tree in line_trees.items():
                if any(tree.query(X @ g[h])[0].max() < 5e-4 for h in range(len(g))):
                    on = name
                    break
            ends = []
            for i in members:
                for side, p in ((0, pcs[i]["seg"][0]), (1, pcs[i]["seg"][-1])):
                    dist, k = corner_tree.query(point_from_beta(p))
                    if dist < 1e-7:
                        ends.append(corner_tree.names[k])
                    elif pcs[i]["ends"][side] == "corner":
                        ends.append(f"unnamed {np.round(p, 4).tolist()}")
            meeting_edges.append({
                "walls": f"{w1}/{w2}", "found": True, "on_atlas_line": on,
                "surroundings": [list(t) for t in pcs[members[0]]["sig"]],
                "domain_pieces": len(members), "ends": ends, "pieces_on_sphere": n,
                "domain_polylines": [[[round(float(t), 9) for t in x] for x in pcs[i]["seg"]]
                                     for i in members]})

    # ---- atlas line pieces ---------------------------------------------------------------------
    per_line = {}
    for lp in atlas["line_pieces"]:
        e = per_line.setdefault(lp["line_shape"], {"classes": 0, "pieces_on_sphere": 0,
                                                    "classes_detail": []})
        e["classes"] += 1
        e["pieces_on_sphere"] += lp["copies_on_sphere"]
        e["classes_detail"].append({"id": lp["id"], "ends": lp["end_points"],
                                    "on_sphere": lp["copies_on_sphere"]})

    # ---- new edges: meeting curves on no atlas line, per meeting ------------------------------
    new_edges = {}
    for e in meeting_edges:
        if not e["found"] or e["on_atlas_line"]:
            continue
        d = new_edges.setdefault(e["walls"], {"classes": 0, "pieces_on_sphere": 0, "classes_detail": []})
        d["classes"] += 1
        d["pieces_on_sphere"] += e["pieces_on_sphere"]
        d["classes_detail"].append({"ends": e["ends"], "on_sphere": e["pieces_on_sphere"]})

    # ---- walls -----------------------------------------------------------------------------------
    walls = {w: {"face_types": v["piece_types"], "classes": v["face_classes"],
                 "faces_on_sphere": v["faces_on_sphere"],
                 "removed_fragments": v["removed_fragments"],
                 "by_regions": {"|".join(p["between"]): p["on_sphere"] for p in v["pieces"]}}
             for w, v in atlas["counts_per_wall"].items()}

    out = {"_about": __doc__,
           "walls": walls,
           "edges": {"atlas_lines": per_line, "wall_meetings_off_atlas_lines": new_edges},
           "wall_meeting_curves": meeting_edges,
           "map_gaps_bridged": map_gaps,
           "totals": {
               "wall_pieces_on_sphere": sum(v["faces_on_sphere"] for v in walls.values()),
               "atlas_line_pieces_on_sphere": sum(v["pieces_on_sphere"] for v in per_line.values()),
               "meeting_edge_pieces_on_sphere": sum(e["pieces_on_sphere"] for e in meeting_edges
                                                    if e["found"] and not e["on_atlas_line"]),
               "points_on_sphere": sum(p["copies_on_sphere"] for p in atlas["points"])}}
    (data_dir() / "sphere_map" / "merged_counts.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = build()
    for e in r["wall_meeting_curves"]:
        if not e["found"]:
            print(e["walls"], "NOT FOUND")
            continue
        print(f"{e['walls']:9s} on line {str(e['on_atlas_line']):5s} x{e['pieces_on_sphere']:5d} "
              f"domain pieces {e['domain_pieces']} ends {e['ends']}")
    print(json.dumps(r["totals"], indent=1))
