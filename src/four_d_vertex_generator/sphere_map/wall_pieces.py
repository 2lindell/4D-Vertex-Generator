"""Wall pieces on the whole 3-sphere, cut only where they should be.

A wall piece is a connected part of one wall, cut only by the curves where another wall meets it
and by the atlas's named lines (every crease of a wall lies on a named line). It is built from
the map's wall patches (`patches.json`) over all 1200 copies:

* two patches of the wall are next to each other when the pieces on their two sides are the same
  or neighbours (across one surface), or continue into each other across a domain face;
* the curve they share is solved exactly (the wall's surface met by the crossing surface, by the
  wall's other surface at a crease, or by the domain face), and checked against the sign of
  every other surface on both patches;
* they are joined unless that curve lies on a named line or on a wall-wall meeting curve
  (`merged_counts.json`), read in whichever copy it lies in.

Each class of pieces is reported with its count on the sphere, the regions it separates, and any
inconsistency (a piece whose regions change without a cut, or a count other than 1200 / its
stabiliser).
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .chords import FACE_N, FACE_NAMES
from .frame import data_dir, load_json, multiplication_table

TOL_ON = 1e-7


def _seg_tree(polys):
    """KD-tree over dense points of polylines (beta), plus the segments for exact distances."""
    from scipy.spatial import cKDTree
    pts, segs = [], []
    for P in polys:
        P = np.asarray(P, float)
        P = P / P.sum(axis=1, keepdims=True)
        for a, b in zip(P[:-1], P[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / 2e-4)))
            for t in np.linspace(0, 1, n + 1)[:-1]:
                pts.append(a + (b - a) * t)
                segs.append((a, b))
        pts.append(P[-1])
        segs.append((P[-1], P[-1]))
    return cKDTree(np.array(pts)), segs


def _dist(tree, segs, x, k=6):
    d, idx = tree.query(x, k=k)
    best = np.inf
    for i in np.atleast_1d(idx):
        a, b = segs[i]
        ab = b - a
        t = 0.0 if ab @ ab == 0 else float(np.clip((x - a) @ ab / (ab @ ab), 0, 1))
        best = min(best, float(np.linalg.norm(x - (a + t * ab))))
    return best


class Splits:
    """The curves that cut walls: named lines and wall-wall meeting curves, in the domain."""

    def __init__(self, inc):
        w = load_json("working_data.json")
        polys = [p for info in w["exact"]["lines"].values() for p in info["points"]]
        mc = json.loads((data_dir() / "sphere_map" / "merged_counts.json").read_text())
        polys += [p for e in mc["wall_meeting_curves"] if e.get("found")
                  for p in e["domain_polylines"]]
        self.tree, self.segs = _seg_tree(polys)
        self.inc = inc

    def on(self, x, tangent):
        """Whether the curve point x (domain beta) lies on a cutting curve, in any copy it touches."""
        x = x / x.sum()
        if _dist(self.tree, self.segs, x) < TOL_ON:
            return True
        if (x @ FACE_N.T).min() > 1e-9:
            return False
        basis = np.linalg.svd(np.vstack([np.ones(4), tangent]))[2][2:]
        for k, (M, xk, _) in self.inc._frames_round(x, basis).items():
            if k and _dist(self.tree, self.segs, xk / xk.sum()) < TOL_ON:
                return True
        return False


def _solve(fs, x, iters=40):
    x = np.asarray(x, float).copy()
    for _ in range(iters):
        v = np.array([f(x)[0] for f in fs])
        if np.abs(v).max() < 1e-15:
            break
        J = np.array([f(x)[1] for f in fs])
        x = x + np.linalg.lstsq(np.vstack([J, np.ones(4)]), np.r_[-v, 1.0 - x.sum()], rcond=1e-10)[0]
    return x, np.abs([f(x)[0] for f in fs]).max() < 1e-11


def build(walls=None, log=print) -> dict:
    from .incidence import Incidence
    from .wall_meets import _sf
    inc = Incidence()
    surfs = inc.surfs
    root = data_dir() / "sphere_map"
    patches = json.loads((root / "patches.json").read_text())["patches"]
    pieces = json.loads((root / "pieces.json").read_text())["pieces"]
    sign = np.array([[1 if c == "+" else -1 if c == "-" else 0 for c in p["signs"]] for p in pieces],
                    np.int8)
    nb = defaultdict(dict)                       # piece -> {neighbour piece: surface}
    cont = defaultdict(list)                     # piece -> [(face, copy, piece)]
    for p in pieces:
        for n in p["neighbours"]:
            nb[p["id"]][n["piece"]] = n["surface"]
        for f in p["faces"]:
            cont[p["id"]].append((f["face"], f["copy"], f["piece"]))
    mult = multiplication_table()
    splits = Splits(inc)
    face_f = {n: (lambda nn: (lambda x: (float(nn @ x), nn - nn.mean())))(FACE_N[i])
              for i, n in enumerate(FACE_NAMES)}
    by_wall = defaultdict(list)
    for p in patches:
        if p["wall"]:
            by_wall[p["wall"]].append(p)

    def close(a, b):
        return a == b or b in nb[a]

    def valid(x, piece_list, zero):
        """x lies on the closure of patches of these pieces: every surface not through x has the
        pieces' sign there."""
        v = inc.values(x / x.sum())
        g = np.array([np.linalg.norm(s.grad((x / x.sum())[None])[0]) for s in surfs]) + 1e-300
        live = np.abs(v) / g > 1e-9
        live[list(zero)] = False
        for pc in piece_list:
            if (np.sign(v[live]) != sign[pc][live]).any():
                return False
        return True

    out = {}
    for wall in sorted(walls or by_wall):
        ps = by_wall[wall]
        idx = {p["id"]: i for i, p in enumerate(ps)}
        interior = [p for p in ps if p["face"] is None]
        facep = [p for p in ps if p["face"] is not None]
        by_piece = defaultdict(list)
        for p in interior:
            for q in p["pieces"]:
                by_piece[q].append(p)
        links = []                               # (patch i, patch j, copy shift of j, cut?)
        n_cut = n_join = n_bad = 0
        # interior - interior, same copy
        for p in interior:
            a, b = p["pieces"]
            cands = {id(q): q for x in [a, *nb[a]] for q in by_piece.get(x, ())}
            for q in cands.values():
                if q["id"] <= p["id"]:
                    continue
                c, d = q["pieces"]
                if not ((close(a, c) and close(b, d)) or (close(a, d) and close(b, c))):
                    continue
                if {a, b} == {c, d}:
                    continue
                s1, s2 = p["surface"], q["surface"]
                if s1 == s2:
                    t = nb[a].get(c, nb[a].get(d, nb[b].get(c, nb[b].get(d))))
                    if t is None:
                        continue
                    fs, zero = [_sf(surfs[s1]), _sf(surfs[t])], {s1, t}
                else:
                    fs, zero = [_sf(surfs[s1]), _sf(surfs[s2])], {s1, s2}
                x0 = (np.asarray(p["sample_beta"]) + np.asarray(q["sample_beta"])) / 2
                x, ok = _solve(fs, x0)
                if not ok or (x @ FACE_N.T).min() < -1e-9 or not valid(x, [a, c], zero):
                    n_bad += 1
                    continue
                tan = np.linalg.svd(np.vstack([np.ones(4), fs[0](x)[1], fs[1](x)[1]]))[2][-1]
                cut = splits.on(x, tan)
                n_cut += cut
                n_join += not cut
                links.append((idx[p["id"]], idx[q["id"]], 0, cut))
        # interior - interior across a domain face (into copy c), and interior - face patch
        for p in interior:
            a, b = p["pieces"]
            for fa, ca, a2 in cont[a]:
                for fb, cb, b2 in cont[b]:
                    if fa != fb or ca != cb:
                        continue
                    for q in by_piece.get(a2, ()):
                        if set(q["pieces"]) != {a2, b2}:
                            continue
                        fs = [_sf(surfs[p["surface"]]), face_f[fa]]
                        x, ok = _solve(fs, np.asarray(p["sample_beta"]))
                        if not ok:
                            n_bad += 1
                            continue
                        tan = np.linalg.svd(np.vstack([np.ones(4), fs[0](x)[1], fs[1](x)[1]]))[2][-1]
                        cut = splits.on(x, tan)
                        n_cut += cut
                        n_join += not cut
                        links.append((idx[p["id"]], idx[q["id"]], ca, cut))
        # face patches: join to interior patches sharing piece a and whose other side continues to b
        for p in facep:
            a, b = p["pieces"]
            c = p["side_b_copy_shift"]
            for q in by_piece.get(a, ()):
                other = q["pieces"][1] if q["pieces"][0] == a else q["pieces"][0]
                if not any(f == p["face"] and cc == c and pc == b for f, cc, pc in cont[other]) \
                        and not (other in nb and any(pc == b for _, cc, pc in cont[other] if cc == c)):
                    continue
                fs = [_sf(surfs[q["surface"]]), face_f[p["face"]]]
                x, ok = _solve(fs, np.asarray(p["sample_beta"]))
                if not ok:
                    n_bad += 1
                    continue
                tan = np.linalg.svd(np.vstack([np.ones(4), fs[0](x)[1], fs[1](x)[1]]))[2][-1]
                cut = splits.on(x, tan)
                links.append((idx[p["id"]], idx[q["id"]], 0, cut))
            # the same patch seen from copy c: pieces (b, a) with the inverse shift
            for q in facep:
                if q["pieces"] == [b, a] and mult[c, q["side_b_copy_shift"]] == 0:
                    links.append((idx[p["id"]], idx[q["id"]], c, False))
        # union-find over (copy, patch)
        n, K = len(ps), len(mult)
        parent = np.arange(n * K)

        def find(i):
            r = i
            while parent[r] != r:
                r = parent[r]
            while parent[i] != r:
                parent[i], i = r, parent[i]
            return r
        for i, j, shift, cut in links:
            if cut:
                continue
            for k in range(K):
                u, v = find(k * n + i), find(int(mult[k, shift]) * n + j)
                if u != v:
                    parent[v] = u
        roots = np.array([find(i) for i in range(n * K)])
        comp_patches = defaultdict(set)
        for i in range(n):
            for k in range(K):
                comp_patches[roots[k * n + i]].add(i)
        # classes: patches appearing together in one piece
        cls = list(range(n))

        def cf(i):
            while cls[i] != i:
                i = cls[i]
            return i
        for members in comp_patches.values():
            ms = sorted(members)
            for m in ms[1:]:
                cls[cf(m)] = cf(ms[0])
        per_class = defaultdict(lambda: {"count": 0, "patches": set()})
        for members in comp_patches.values():
            c = cf(next(iter(members)))
            per_class[c]["count"] += 1
            per_class[c]["patches"] |= members
        classes = []
        for c, v in per_class.items():
            regs = sorted({tuple(sorted(ps[i]["regions"])) for i in v["patches"]})
            classes.append({"count_on_sphere": v["count"], "patches": len(v["patches"]),
                            "regions": ["|".join(r) for r in regs],
                            "consistent": len(regs) == 1,
                            "patch_ids": sorted(ps[i]["id"] for i in v["patches"])})
        classes.sort(key=lambda c: (-c["patches"], c["regions"]))
        out[wall] = {"classes": classes, "links": {"joined": n_join, "cut": n_cut, "unsolved": n_bad},
                     "_links": [(ps[i]["id"], ps[j]["id"], int(sh), bool(c)) for i, j, sh, c in links]}
        log(f"{wall:5s} {len(classes):2d} classes  "
            + "  ".join(f"{c['regions']}x{c['count_on_sphere']}({c['patches']}p)" for c in classes)
            + f"  [links joined {n_join} cut {n_cut} unsolved {n_bad}]")
    return out


if __name__ == "__main__":
    import sys
    res = build(sys.argv[1:] or None)
    for v in res.values():
        v.pop("_links", None)
    (data_dir() / "sphere_map" / "wall_pieces.json").write_text(json.dumps(
        {"_about": __doc__, "walls": res}, indent=1))


def split_faces(walls, log=print) -> dict:
    """For each face of these walls (atlas faces: one wall between two region volumes), the
    number of parts it falls into when cut along the named lines and meeting curves: its own
    patches, joined only where the curve between them is on neither."""
    res = build(walls, log=lambda *_: None)
    mult = multiplication_table()
    faces = json.loads((data_dir() / "sphere_map" / "faces.json").read_text())["faces"]
    out = {}
    for label, info in faces.items():
        wall = label.split("|")[0].strip()
        if wall not in res:
            continue
        links = res[wall]["_links"]
        for o in info["orbits"]:
            if (o.get("congruence_with_first_orbit") or {}).get("status") == "not congruent":
                continue
            mem = {(int(k), int(p)) for k, p in o["members"]}
            idx = {m: i for i, m in enumerate(sorted(mem))}

            def parts(use_cuts):
                parent = list(range(len(idx)))

                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i
                for a, b, sh, cut in links:
                    if cut and use_cuts:
                        continue
                    for (k, pa) in mem:
                        if pa != a:
                            continue
                        other = (int(mult[k, sh]), b)
                        if other in idx:
                            parent[find(idx[other])] = find(idx[(k, pa)])
                    if sh == 0:
                        for (k, pb) in mem:
                            if pb == b and (k, a) in idx:
                                parent[find(idx[(k, pb)])] = find(idx[(k, a)])
                return len({find(i) for i in range(len(idx))})
            out[f"{label} #{o['orbit']}"] = {"members": len(mem), "parts_uncut": parts(False),
                                             "parts_cut": parts(True)}
            log(f"{label} #{o['orbit']}: {len(mem)} patches, parts without cuts {parts(False)}, "
                f"with cuts {parts(True)}")
    return out


def face_cuts(log=print) -> dict:
    """For every face orbit (atlas faces: one wall between two region volumes), the extra parts it
    falls into when cut along named lines and meeting curves running through it. Uses the build's
    own surface-walk links between the face's patches: a link records the exact points where the
    walk crossed from one patch into the next, on the curve they share; the link is cut when those
    points lie on a named line or meeting curve. Parts are counted with and without the cut
    links; the difference is what the cuts add."""
    from .build import _load
    from .incidence import Incidence
    root = data_dir() / "sphere_map"
    state = _load(root / ".cache" / "checkpoint.pkl")
    march_links, exit_glues = state["march_links"], state["exit_glues"]
    inc = Incidence()
    surfs = inc.surfs
    splits = Splits(inc)
    patches = json.loads((root / "patches.json").read_text())["patches"]
    index = {(min(p["pieces"]), max(p["pieces"]), p["surface"]): p["id"]
             for p in patches if p["face"] is None}
    mult = multiplication_table()
    inner = defaultdict(list)                    # patch -> [(other patch, cut?)]
    cut_cache = {}
    for (ka, kb), recs in march_links.items():
        a, b = index.get(ka), index.get(kb)
        if a is None or b is None or patches[a]["wall"] != patches[b]["wall"] or not patches[a]["wall"]:
            continue
        cut = False
        for crossed, x in recs[:3]:
            gs = [surfs[ka[2]].grad(x[None])[0]] + [surfs[t].grad(x[None])[0] for t in crossed[:1]]
            tan = np.linalg.svd(np.vstack([np.ones(4), *gs]))[2][-1]
            if splits.on(np.asarray(x, float), tan):
                cut = True
                break
        inner[a].append((b, cut))
        inner[b].append((a, cut))
    across = defaultdict(list)
    for ka, c, kb in exit_glues:
        a, b = index.get(ka), index.get(kb)
        if a is not None and b is not None and patches[a]["wall"] == patches[b]["wall"]:
            across[a].append((b, int(c)))
    faces = json.loads((root / "faces.json").read_text())["faces"]
    out = {}
    for label, info in faces.items():
        for o in info["orbits"]:
            if (o.get("congruence_with_first_orbit") or {}).get("status") == "not congruent":
                continue
            mem = sorted({(int(k), int(p)) for k, p in o["members"]})
            idx = {m: i for i, m in enumerate(mem)}

            def parts(use_cuts):
                parent = list(range(len(mem)))

                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i
                for (k, pa), i in idx.items():
                    for pb, cut in inner.get(pa, ()):
                        if use_cuts and cut:
                            continue
                        j = idx.get((k, pb))
                        if j is not None:
                            parent[find(j)] = find(i)
                    for pb, c in across.get(pa, ()):
                        j = idx.get((int(mult[k, c]), pb))
                        if j is not None:
                            parent[find(j)] = find(i)
                return len({find(i) for i in range(len(mem))})
            p0, p1 = parts(False), parts(True)
            out[f"{label} #{o['orbit']}"] = {"patches": len(mem), "parts_links": p0,
                                             "parts_cut": p1, "extra_pieces": p1 - p0}
            if p1 != p0:
                log(f"{label} #{o['orbit']}: {len(mem)} patches; parts {p0} -> {p1} when cut")
    return out
