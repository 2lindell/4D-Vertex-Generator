"""Line pieces from the atlas's own lines.

The working data holds every line exactly (``exact.lines``: polylines in the domain, each on the
surfaces it lies on) and how each of its domain pieces ends (``line_ends.json``): on a named point,
on another line, or going on as the same line into the next copy. A line is split only at known
points: the atlas's exact meeting points (``exact.points``), wherever a copy of one lies on it.
Pieces are joined where the line just goes on into the next copy. The pieces that result are
grouped into classes under the group and then into congruence types under G and G.Q.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .frame import load_json, point_from_beta

MIN_PIECE = 5e-3     # line stretches shorter than this are rounding, not pieces


def densify(r4: np.ndarray, step: float = 2e-4) -> np.ndarray:
    """Points along a polyline on S^3 at about `step` apart (chords split and renormalised)."""
    out = [r4[:1]]
    for a, b in zip(r4[:-1], r4[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        t = np.linspace(0, 1, n + 1)[1:, None]
        seg = a * (1 - t) + b * t
        out.append(seg / np.linalg.norm(seg, axis=1, keepdims=True))
    return np.vstack(out)


def _exact_resample(poly: np.ndarray, pair, step: float = 2e-3) -> np.ndarray:
    """The curve on both surfaces of `pair`, sampled every ~step: chords split, each point moved
    onto both surfaces (min-norm Newton); ends kept as recorded (they are exact points)."""
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            x = a * (1 - t) + b * t
            y = _polish_pair(pair, x)
            out.append(y if y is not None else x)
    out[-1] = poly[-1]
    return np.array(out)


def _polish_pair(pair, x, iters=12):
    x = np.asarray(x, float).copy()
    for _ in range(iters):
        f = np.array([float(s.f(x[None])[0]) for s in pair])
        if np.abs(f).max() < 1e-15:
            break
        J = np.array([s.grad(x[None])[0] - s.grad(x[None])[0].mean() for s in pair])
        x = x + np.linalg.lstsq(np.vstack([J, np.ones(4)]), np.r_[-f, 0.0], rcond=None)[0]
    return x if np.abs([float(s.f(x[None])[0]) for s in pair]).max() < 1e-12 else None


def _on_line(descriptions, line) -> bool:
    """Whether the data's notes on a point put it on this line."""
    pat = re.compile(rf"(\bon {re.escape(line)}\b|\bof (a piece of )?{re.escape(line)}\b)")
    return any(pat.search(d) for d in descriptions)


def _physical_length(members, subs, grp) -> float:
    """Length of a piece on the sphere, each physical stretch counted once (a piece on the
    domain's boundary is listed once per copy it belongs to)."""
    seen, total = set(), 0.0
    for k, s in members:
        r4 = point_from_beta(subs[s]["polyline"]) @ grp.g[k].T
        key = tuple(np.round(r4[len(r4) // 2], 6) + 0.0)
        if key in seen:
            continue
        seen.add(key)
        total += _arc(r4)
    return total


def _arc(pts):
    return float(np.sum(np.arccos(np.clip(np.sum(pts[1:] * pts[:-1], axis=1), -1, 1))))


def build_line_pieces(grp, line_segs_doc, log, traced_edges=()):
    """Returns (pieces, place) — pieces: list of class records; place(line, sub, copy) -> ref."""
    from .build import _components, sphere_edges

    w = load_json("working_data.json")
    lines = w["exact"]["lines"]
    ends_doc = load_json("line_ends.json")["lines"]
    named = [(p["name"], point_from_beta(np.asarray(p["float"], float))) for p in w["exact"]["points"]]
    named_from = [p["from"] for p in w["exact"]["points"]]
    all_named = np.vstack([grp.g @ x for _, x in named])            # every copy of every point
    from .frame import cell_vertices, surface_by_name, surfaces
    cell_inv = np.linalg.inv(cell_vertices())
    lib_by_name = {n: surfaces()[i] for n, i in surface_by_name().items()}
    exact_pieces = 0
    named_of = [n for n, _ in named for _ in range(grp.n)]
    ntree = cKDTree(all_named)

    # 1. cut each domain piece of each line at every copy of a named point lying on it
    subs = []                # (line, atlas piece, polyline beta, start name, end name)
    for name, info in lines.items():
        for pi, poly in enumerate(info["points"]):
            poly = np.asarray(poly, float)
            poly = poly / poly.sum(axis=1, keepdims=True)
            keep = np.r_[True, np.linalg.norm(np.diff(poly, axis=0), axis=1) > 1e-12]
            poly = poly[keep]                    # (repeated points in the record)
            surf_names = info["surfaces"][pi] if pi < len(info["surfaces"]) else info["surfaces"][0]
            exact_pair = [lib_by_name.get(n) for n in surf_names]
            if len(exact_pair) == 2 and all(x is not None for x in exact_pair):
                poly = _exact_resample(poly, exact_pair)
                exact_pieces += 1
            # named points the data puts on this line ('end of X8 on X5', 'end of X5 on C4'),
            # wherever a copy of one lies on the curve: within 1e-5 of the (exactly resampled)
            # polyline, and on both of its surfaces where those are known exactly
            r4 = point_from_beta(poly)
            mine = [j for j in range(len(named)) if _on_line(named_from[j], name)]
            inserts, seen = [], set()
            for a in range(len(r4) - 1):
                xa, xb = r4[a], r4[a + 1]
                span = np.linalg.norm(xb - xa)
                for jj in ntree.query_ball_point((xa + xb) / 2, span / 2 + 1e-5):
                    if jj // grp.n not in mine:
                        continue
                    x = all_named[jj]
                    t = float(np.clip((x - xa) @ (xb - xa) / max(span ** 2, 1e-300), 0, 1))
                    off = np.linalg.norm(xa + t * (xb - xa) - x)
                    bx = x @ cell_inv
                    bx = bx / bx.sum()
                    if all(e is not None for e in exact_pair) and len(exact_pair) == 2:
                        if any(abs(float(sf.f(bx[None])[0])) > 1e-9 for sf in exact_pair):
                            continue
                    elif off > max(1e-5, span ** 2):
                        continue
                    key = tuple(np.round(x, 9))
                    if key not in seen:
                        seen.add(key)
                        inserts.append((off, a, t, bx, x, jj // grp.n))
            kept = []
            for off, a, t, bx, x, pi_ in sorted(inserts, key=lambda z: (z[0], z[5])):
                if all(np.linalg.norm(x - y[4]) > 1e-7 for y in kept):
                    kept.append((off, a, t, bx, x, pi_))
            for _off, a, t, bx, _x, _pi in sorted(kept, key=lambda z: -(z[1] + z[2])):
                if t <= 1e-9:
                    poly[a] = bx
                elif t >= 1 - 1e-9:
                    poly[a + 1] = bx
                else:
                    poly = np.insert(poly, a + 1, bx, axis=0)
            # snap: the curve meets each named point exactly (samples a hair off it are dropped)
            for *_rest, x, _pi in kept:
                bx = x @ cell_inv
                bx = bx / bx.sum()
                near = np.linalg.norm(point_from_beta(poly) - x, axis=1)
                drop = (near < 1e-4) & (near > 1e-12)
                if drop.any() and (near <= 1e-12).any():
                    poly = poly[~drop]
            r4 = point_from_beta(poly)
            cut_at = []
            for idx, x in enumerate(r4):
                hit = [jj for jj in ntree.query_ball_point(x, 1e-6) if jj // grp.n in mine]
                if hit and all(np.linalg.norm(x - r4[c]) > 5e-3 for c, _ in cut_at):
                    cut_at.append((idx, named_of[hit[0]]))
            # the piece's own ends on named points (line_ends 'on')
            ends = {e["side"]: e for e in ends_doc.get(name, []) if e["piece"] == pi}
            start = ends.get("start", {}).get("on")
            stop = ends.get("end", {}).get("on")
            bounds = [(0, start)] + [c for c in cut_at if 0 < c[0] < len(poly) - 1] + \
                [(len(poly) - 1, stop)]
            for (a, na), (b, nb) in zip(bounds[:-1], bounds[1:]):
                if b - a < 1 or _arc(point_from_beta(poly[a:b + 1])) < MIN_PIECE:
                    continue      # (rounding left between a cut and a known point: lines run
                                  #  all the way to their known points, so it is not a piece)
                subs.append({"line": name, "atlas_piece": pi, "from": a, "to": b,
                             "polyline": poly[a:b + 1], "start_point": na, "end_point": nb,
                             "surfaces": info["surfaces"][pi] if pi < len(info["surfaces"]) else
                             info["surfaces"][0]})
    log(f"atlas lines: {exact_pieces} domain pieces resampled exactly on their two surfaces; "
        f"{len(subs)} pieces after cutting at named points")

    # 2. join across ends that are not on a named point (the same line going on)
    end_x, end_owner = [], []
    for si, sp in enumerate(subs):
        for side, idx, named_end in (("start", 0, sp["start_point"]), ("end", -1, sp["end_point"])):
            if named_end:
                continue
            end_x.append(point_from_beta(sp["polyline"][idx]))
            end_owner.append(si)
    end_x = np.array(end_x)
    all_x = np.einsum("gij,nj->gni", grp.g, end_x).reshape(-1, 4)
    tree = cKDTree(all_x)
    n_end = len(end_owner)
    joins = []
    for i, (si, x) in enumerate(zip(end_owner, end_x)):
        for j in tree.query_ball_point(x, 1e-6):
            k, i2 = divmod(j, n_end)
            s2 = end_owner[i2]
            if (k == 0 and i2 == i) or subs[s2]["line"] != subs[si]["line"]:
                continue
            joins.append((si, int(k), s2))
    # the same physical piece seen from several copies: a piece lying on the domain's boundary
    # (or on a symmetry's fixed curve) is carried by some group elements onto itself or onto
    # another piece of copy 0; those are one piece, not several
    def probe(sp):
        r4 = point_from_beta(sp["polyline"])
        return np.vstack([r4[len(r4) // 3], r4[len(r4) // 2], r4[(2 * len(r4)) // 3]])

    probes = np.array([probe(sp) for sp in subs])                 # (subs, 3, 4)
    ptree = cKDTree(probes[:, 1])
    same = 0
    for si, pr in enumerate(probes):
        imgs = np.einsum("gij,nj->gni", grp.g, pr)                # (copies, 3, 4)
        for k in range(grp.n):
            for s2 in ptree.query_ball_point(imgs[k, 1], 1e-7):
                if subs[s2]["line"] != subs[si]["line"] or (k == 0 and s2 == si):
                    continue
                # all three probes of the carried piece lie on the other piece
                other = densify(point_from_beta(subs[s2]["polyline"]), 1e-4)
                if cKDTree(other).query(imgs[k])[0].max() < 1e-6:
                    # G[k] carries piece si onto piece s2: (copy k, si) is (copy 0, s2)
                    joins.append((s2, k, si))
                    same += 1
    log(f"atlas lines: {same} placements of a piece onto itself or another (shared by copies)")
    comp = _components(grp.n * len(subs), *sphere_edges([], joins, len(subs)))
    comp = comp.reshape(grp.n, len(subs))

    # 3. classes (orbits) of whole pieces
    placed, classes = {}, []
    for c in np.unique(comp):
        c = int(c)
        if c in placed:
            continue
        ks, js = np.nonzero(comp == c)
        k0, s0 = int(ks[0]), int(js[0])
        images = comp[grp.mult[:, k0], s0]
        stab = int(np.sum(images == c))
        for h in range(grp.n):
            placed.setdefault(int(images[h]), (len(classes), h))
        members = [[int(k), int(s)] for k, s in zip(ks, js)]
        names = Counter(subs[s]["line"] for _, s in members)
        pts = np.vstack([densify(point_from_beta(subs[s]["polyline"])) @ grp.g[k].T
                         for k, s in members])
        ends = sorted({subs[s][e] for _, s in members for e in ("start_point", "end_point")
                       if subs[s][e]})
        classes.append({
            "id": len(classes), "line_shape": names.most_common(1)[0][0],
            "segments": [{"copy": k, "line": subs[s]["line"], "atlas_piece": subs[s]["atlas_piece"],
                          "points": [subs[s]["from"], subs[s]["to"]],
                          "equations": subs[s]["surfaces"]} for k, s in members],
            "end_points": ends,
            "closed": not any(subs[s]["start_point"] or subs[s]["end_point"] for _, s in members),
            "length": _physical_length(members, subs, grp),
            "stabilizer_order": stab, "copies_on_sphere": grp.n // stab,
            "_cloud": pts,
        })
    log(f"atlas line pieces: {len(classes)} classes on the sphere")

    # 4. congruence types under G and G.Q, within one line shape
    from .frame import source_dir
    maps = np.concatenate([grp.g, grp.g @ np.load(source_dir() / "normalizer_q.npy")])
    by_shape = defaultdict(list)
    for c in classes:
        by_shape[c["line_shape"]].append(c["id"])
    n_types = 0
    for ids in by_shape.values():
        reps = []
        for i in ids:
            ci = classes[i]
            hit = None
            for t in reps:
                ct = classes[t]
                if abs(ct["length"] - ci["length"]) > 1e-3 * max(ct["length"], 1e-3):
                    continue
                tb = cKDTree(ci["_cloud"])
                sub = ct["_cloud"][:: max(1, len(ct["_cloud"]) // 120)]
                for mi, m in enumerate(maps):
                    d, _ = tb.query(sub @ m.T, distance_upper_bound=1e-3)
                    if np.isfinite(d).all() and d.max() < 3e-4:
                        mapped = ct["_cloud"] @ m.T
                        if tb.query(mapped)[0].max() < 3e-4 and \
                                cKDTree(mapped).query(ci["_cloud"])[0].max() < 3e-4:
                            hit = (t, mi)
                            break
                if hit:
                    break
            if hit is None:
                reps.append(i)
                ci["piece_type"], ci["congruent_to"] = n_types, None
                n_types += 1
            else:
                ci["piece_type"] = classes[hit[0]]["piece_type"]
                ci["congruent_to"] = {"line_piece": hit[0], "map": hit[1]}
    log(f"atlas line piece types: {n_types}")
    for c in classes:
        del c["_cloud"]

    # which atlas sub-piece each traced edge lies on (for face outlines): most of the edge's points
    # within 1e-4 of the sub-piece on the sphere (the edge in copy 0, sub-pieces in copy 0)
    dense, owner = [], []
    for si, sp in enumerate(subs):
        d = densify(point_from_beta(sp["polyline"]), 5e-5)
        dense.append(d)
        owner += [si] * len(d)
    stree = cKDTree(np.vstack(dense))
    edge_sub = defaultdict(Counter)
    for e in traced_edges:
        pts = densify(point_from_beta(np.asarray(e["polyline_beta"], float)), 2e-4)
        if _arc(pts) < MIN_PIECE:
            continue                 # (touches a corner only: not an edge along a line)
        d, j = stree.query(pts)
        close = d < 1e-4
        if close.mean() > 0.8:
            for jj in j[close]:
                edge_sub[e["id"]][owner[jj]] += 1
    del line_segs_doc

    def place_edge(copy, edge):
        if edge not in edge_sub:
            return None
        si = edge_sub[edge].most_common(1)[0][0]
        cls, h = placed[int(comp[copy, si])]
        return {"class": cls, "g": int(h)}

    return classes, place_edge
