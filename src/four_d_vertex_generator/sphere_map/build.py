"""Build the swirlprism 3-sphere map from the atlas data.

Steps (each from the data, nothing named by hand):

1. The complete surface library in the half-cell (``surface_images``).
2. Pieces: cells of that arrangement, found along random chords, each named by classifying a point
   inside it (two points, as a check) and matching the signature against the atlas's shape list.
3. Gluing: where each piece meets each face of the domain, and which piece of which neighbouring
   copy it continues into.
4. The sphere: 1200 copies of every piece; pieces of one region joined across seams (surfaces that
   are not walls there) and across domain faces give each region's connected components on S^3.
5. Walls: every place two different regions meet across a surface is a wall patch, named from the
   working data (the wall whose surfaces and regions match); patches joined along their edges and
   across domain faces give each wall's components. Each region component's boundary is the list
   of wall patches round it, by wall and by the region on the other side.
6. Lines and points: the exact atlas curves and points carried to every copy, joined at shared ends.

Run ``python -m four_d_vertex_generator.sphere_map.build`` to write ``data/swirlprism/sphere_map``.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .chords import FACE_N, FACE_NAMES
from .classify import (
    counts_match,
    key_of_beta,
    shape_signatures,
    split_name,
    split_parent,
    splits,
)
from .frame import (
    cell_vertices,
    data_dir,
    group,
    load_json,
    locate,
    multiplication_table,
    point_from_beta,
    surface_by_name,
    surface_scale,
    surface_values,
)
from .march import glue_exits, march
from .pieces import Exploration, explore
from .surface_images import Library, images_through, source_families

ATLAS_COMMIT = "8488da9c0cf86cef7fbb4f49c6ca797ccfe46b42"
ATLAS_BRANCH = "claude/laughing-noether-qydcwe"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------ labels


def _key_job(args):
    beta, tol = args
    return key_of_beta(beta, hull_tol=tol)


def classify_points(points, hull_tol: float, workers: int, cache: dict, need_key=None):
    """(signature, vertex key) at every point, from the cache where it has them.

    ``need_key(sig)`` says whether a cached signature without its key must be recomputed.
    """
    keys = [(np.asarray(b, float).tobytes(), hull_tol) for b in points]
    todo = []
    for (kb, tol), b in zip(keys, points):
        hit = cache.get((kb, tol)) if tol != 1e-9 else cache.get((kb, tol), cache.get(kb))
        if hit is None or (isinstance(hit, str) and need_key is not None and need_key(hit)):
            todo.append(((kb, tol), b))
    if todo:
        with Pool(workers) as pool:
            res = pool.map(_key_job, [(b, hull_tol) for _, b in todo], chunksize=32)
        for (k, _), r in zip(todo, res):
            cache[k] = r
    out = []
    for kb, tol in keys:
        hit = cache.get((kb, tol)) if tol != 1e-9 else cache.get((kb, tol), cache.get(kb))
        out.append((hit, None) if isinstance(hit, str) else tuple(hit))
    return out


def region_signatures() -> dict[str, str]:
    shapes = load_json("working_data.json")["shapes"]
    out = {}
    for name, sig in shape_signatures().items():
        if shapes[name]["dim"] == "region":
            if not counts_match(name, sig):
                raise RuntimeError(f"region {name}: its seed does not give its recorded counts")
            out.setdefault(sig, split_parent(name))
    return out


def label_pieces(ex: Exploration, workers: int, cache: dict | None = None):
    """Region name of every piece (from up to four interior points), and the per-piece check.

    Where a name covers more than one polytope (shape_splits.json), the vertex key picks the part.
    """
    by_sig = region_signatures()
    cache = {} if cache is None else cache
    split = set(splits())
    pts = [b for lst in ex.samples for _, b in lst]
    res = classify_points(pts, 1e-9, workers, cache,
                          need_key=lambda sig: by_sig.get(sig) in split)
    labels, checks = [], []
    it = iter(res)
    for lst in ex.samples:
        names = []
        for _ in lst:
            sig, key = next(it)
            name = by_sig.get(sig, "?" + sig)
            names.append(split_name(name, key) if name in split else name)
        labels.append(Counter(names).most_common(1)[0][0])
        checks.append({"samples": len(names), "agree": len(set(names)) == 1,
                       "names": sorted(set(names))})
    return labels, checks


def _name_at(beta, by_sig):
    sig, key = key_of_beta(beta)
    name = by_sig.get(sig)
    return split_name(name, key) if name in splits() else name


def _transition(p1, p2, name1, by_sig, steps=44):
    lo, hi = np.asarray(p1, float), np.asarray(p2, float)
    for _ in range(steps):
        mid = (lo + hi) / 2
        if _name_at(mid, by_sig) == name1:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _refine_job(args):
    pid, p1, n1, p2, n2, families = args
    by_sig = region_signatures()
    x = _transition(p1, p2, n1, by_sig)
    hits = images_through(x, families, tol=1e-7) if families else []
    if not hits:
        hits = images_through(x, None, tol=1e-7)
    return pid, n1, n2, x, hits


def refine(ex: Exploration, labels, checks, lib: Library, workers: int) -> tuple[int, list]:
    """Add the images that pass where a piece's samples change shape. Returns (added, unexplained)."""
    by_sig = region_signatures()
    naming = wall_naming()
    jobs = []
    for pid, c in enumerate(checks):
        if c["agree"]:
            continue
        lst = ex.samples[pid]
        names = [_name_at(b, by_sig) for _, b in lst]
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if names[i] and names[j] and names[i] != names[j]:
                    fams = set()
                    for v in naming.values():
                        if names[i] in v["regions"] and names[j] in v["regions"]:
                            fams |= v["families"]
                    jobs.append((pid, lst[i][1], names[i], lst[j][1], names[j], fams or None))
    added, unexplained = 0, []
    with Pool(workers) as pool:
        for pid, n1, n2, x, hits in pool.map(_refine_job, jobs, chunksize=1):
            if not hits:
                unexplained.append({"piece": pid, "regions": [n1, n2], "point": x.tolist()})
                continue
            best = min(hits, key=lambda h: h[2])
            _, new = lib.add_image(best[0], best[1], x)
            added += int(new)
    return added, unexplained


# ----------------------------------------------------------------------------------------- volumes


def piece_volumes(ex: Exploration, surfs, n: int = 400_000, seed: int = 5) -> np.ndarray:
    """Share of the domain's spherical volume in each piece (Monte Carlo, radially weighted)."""
    rng = np.random.default_rng(seed)
    t = cell_vertices()
    acc = np.zeros(len(ex.signs))
    total = 0.0
    for lo in range(0, n, 50_000):
        b = rng.dirichlet(np.ones(4), min(50_000, n - lo))
        swap = b[:, 2] < b[:, 3]
        b[swap] = b[swap][:, [1, 0, 3, 2]]
        w = 1.0 / np.linalg.norm(b @ t, axis=1) ** 4       # flat tetrahedron -> sphere
        signs = np.sign(surface_values(b, surfs)).astype(np.int8)
        total += w.sum()
        for row, weight in zip(signs, w):
            pid = ex.keys.get(row.tobytes())
            if pid is not None:
                acc[pid] += weight
    return acc / total


# ------------------------------------------------------------------------------------------ gluing


def glue_faces(ex: Exploration, surfs) -> tuple[list[tuple[int, int, int, int]], dict]:
    """(piece, face, neighbour copy, neighbour piece) for every observed crossing of a domain face."""
    out: set[tuple[int, int, int, int]] = set()
    stats = Counter()
    items = [(pid, face, beta) for (pid, face), pts in ex.face_points.items() for beta in pts]
    for lo in range(0, len(items), 4000):
        chunk = items[lo:lo + 4000]
        outside = np.array([beta - 1e-7 * FACE_N[f] / np.linalg.norm(FACE_N[f])
                            for _, f, beta in chunk])
        copies, inner = locate(point_from_beta(outside), tol=1e-13)
        vals = surface_values(inner, surfs)
        dist = np.abs(vals) / (surface_scale(inner, surfs) + 1e-300)
        for (pid, face, _), k, row, d in zip(chunk, copies, vals, dist):
            if k == 0:
                stats["stayed"] += 1
                continue
            if d.min() < 1e-9:
                stats["on a surface"] += 1
                continue
            q = ex.keys.get(np.sign(row).astype(np.int8).tobytes())
            if q is None:
                stats["unseen piece"] += 1
                continue
            out.add((pid, face, int(k), q))
            stats["glued"] += 1
    return sorted(out), dict(stats)


# -------------------------------------------------------------------------------------- components


def _components(n_nodes: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_nodes, n_nodes))
    return connected_components(graph, directed=False)[1]


def sphere_edges(pairs: list[tuple[int, int]], glues: list[tuple[int, int, int]], n: int):
    """Edges between nodes k * n + i over all copies k: same-copy pairs and copy-shifting glues."""
    mult = multiplication_table()
    copies = np.arange(len(mult))
    rows, cols = [], []
    if pairs:
        a = np.array(pairs)
        rows.append((copies[:, None] * n + a[:, 0]).ravel())
        cols.append((copies[:, None] * n + a[:, 1]).ravel())
    if glues:
        g = np.array(glues)                                   # (i, shift copy, j)
        rows.append((copies[:, None] * n + g[:, 0]).ravel())
        cols.append((mult[:, g[:, 1]] * n + g[:, 2]).ravel())
    if not rows:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(rows), np.concatenate(cols)


def region_components(labels, adjacency, glued, n_pieces):
    seams = [(a, b) for (a, b, s) in adjacency if labels[a] == labels[b]]
    glues = [(p, k, q) for (p, f, k, q) in glued if labels[p] == labels[q]]
    mismatched = [(p, f, k, q) for (p, f, k, q) in glued if labels[p] != labels[q]]
    rows, cols = sphere_edges(seams, glues, n_pieces)
    comp = _components(1200 * n_pieces, rows, cols)
    return comp.reshape(1200, n_pieces), mismatched


def describe_components(comp: np.ndarray, members_of: dict[str, list[int]]):
    """Per name: its components on S^3 (orbits under G, stabiliser size, copies spanned)."""
    out = {}
    for name, items in members_of.items():
        ids = np.unique(comp[:, items])
        # component -> members (copy, item)
        sub = comp[:, items]
        info = {}
        for cid in ids:
            ks, js = np.nonzero(sub == cid)
            info[int(cid)] = (ks, np.asarray(items)[js])
        # orbits: components containing (k, item) for one item, over all k, are one orbit
        orbit_of: dict[int, int] = {}
        orbits = []
        for j, item in enumerate(items):
            cids = set(map(int, np.unique(comp[:, item])))
            known = {orbit_of[c] for c in cids if c in orbit_of}
            if known:
                o = min(known)
            else:
                o = len(orbits)
                orbits.append(set())
            for c in cids:
                orbit_of[c] = o
            orbits[o] |= cids
        reps = []
        for o, cids in enumerate(orbits):
            cid = min(cids)
            ks, its = info[cid]
            # stabiliser: g with g . component = component, i.e. comp[mult(g, k), item] == cid
            k0, i0 = int(ks[0]), int(its[0])
            mult = multiplication_table()
            stab = [int(g) for g in range(1200) if comp[mult[g, k0], i0] == cid]
            pieces_in_rep = sorted(set(map(int, its)))
            reps.append({
                "orbit": o,
                "components": len(cids),
                "stabilizer_order": len(stab),
                "stabilizer": stab,
                "representative": int(cid),
                "copies_spanned": int(len(set(map(int, ks)))),
                "domain_pieces": pieces_in_rep,
                "members": [[int(k), int(i)] for k, i in zip(ks, its)],
            })
        out[name] = {"components": int(len(ids)), "orbits": reps}
    return out


# ------------------------------------------------------------------------------------------- walls


def wall_naming():
    """For each wall: the surface families it lies on and the regions it separates."""
    w = load_json("working_data.json")
    fam = source_families()
    by_name = surface_by_name()
    out = {}
    for wall, surfs in w["exact"]["wall_surfaces"].items():
        fams = {fam[by_name[s]] for s in surfs}
        out[wall] = {"families": fams, "regions": set(w["walls"][wall]["regions"])}
    return out


def name_patch(region_a, region_b, family, naming):
    cands = [k for k, v in naming.items()
             if region_a in v["regions"] and region_b in v["regions"] and family in v["families"]]
    if len(cands) == 1:
        return cands[0], "data"
    if not cands:
        loose = [k for k, v in naming.items() if region_a in v["regions"] and region_b in v["regions"]]
        return None, ("regions-only:" + ",".join(loose)) if loose else "no wall in data"
    return None, "ambiguous:" + ",".join(cands)


def verify_patch_names(patches, workers: int, cache: dict) -> Counter:
    """Classify a point on every patch and settle its wall from the recorded counts.

    The data names a patch's wall from its surface family and the two regions; where two walls
    fit (they share both), the patch point's polytope decides, and where a name covers two
    polytopes (shape_splits.json) its vertex key does. Every name is then checked: the point must
    give the wall's recorded vertex / cell / face / edge counts.
    """
    shapes = load_json("working_data.json")["shapes"]
    walls = [k for k, v in shapes.items() if v["dim"] == "wall"]
    regions_of = {k: v["regions"] for k, v in load_json("working_data.json")["walls"].items()}
    idx = [i for i, p in enumerate(patches) if p["point"] is not None]
    res = classify_points([patches[i]["point"] for i in idx], 1e-7, workers, cache)
    got = dict(zip(idx, res))
    out = Counter()
    for i, p in enumerate(patches):
        sig, key = got.get(i, ("ERR no point", None))
        p["signature"], p["vertex_key"] = sig, key
        if sig.startswith("ERR"):
            p["verified"] = None
            out["no classification"] += 1
            continue
        if p["wall"] is not None:
            p["verified"] = counts_match(p["wall"], sig)
            if p["verified"]:
                if split_parent(p["wall"]) in splits():
                    new = split_name(p["wall"], key)
                    if new != p["wall"]:
                        p["wall"], p["naming"] = new, p["naming"] + "+vertex key"
                        out["moved to its split part"] += 1
                out["data name confirmed"] += 1
                continue
            # the data's name does not fit: settle among the walls that border both regions
            ra, rb = p["regions"]
            fit = [w for w in walls if counts_match(w, sig)
                   and {ra, rb} <= set(regions_of.get(w, ()))]
            if len({split_parent(w) for w in fit}) == 1 and fit:
                name = fit[0] if len(fit) == 1 else split_name(split_parent(fit[0]), key)
                p["naming"] = f"data said {p['wall']}+classified"
                p["wall"], p["verified"] = name, True
                out["data name corrected by classification"] += 1
            else:
                out["data name contradicted"] += 1
            continue
        how = p["naming"]
        cands = how.split(":", 1)[1].split(",") if ":" in how and how.split(":", 1)[1] else walls
        fit = [w for w in cands if counts_match(w, sig)] or [w for w in walls if counts_match(w, sig)]
        parents = {split_parent(w) for w in fit}
        if len(fit) == 1 or (len(parents) == 1 and key and fit):
            name = fit[0] if len(fit) == 1 else split_name(parents.pop(), key)
            p["wall"], p["verified"] = name, True
            p["naming"] = how.split(":")[0] + "+classified"
            out["settled by classification"] += 1
        else:
            p["verified"] = False
            out["still unnamed"] += 1
    return out


def flip(sign: np.ndarray, t: int) -> bytes:
    s = sign.copy()
    s[t] = -s[t]
    return s.tobytes()


def face_family(face: int, lib) -> int | None:
    """Family of the library plane that a domain face lies on (None if no surface lies there)."""
    n = FACE_N[face] / np.linalg.norm(FACE_N[face])
    for k in lib:
        if k.surface.kind == "plane" and abs(abs(float(k.surface.normal @ n)) - 1) < 1e-9:
            return k.family
    return None


def wall_patches(ex, labels, adjacency, mismatched, lib):
    """Interior patches (two pieces across a surface) and face patches (across a domain face).

    A patch has side A (piece a in copy k) and side B (piece b in copy mult[k, shift]).
    """
    naming = wall_naming()
    patches = []
    for (a, b, s), (count, point) in adjacency.items():
        if labels[a] == labels[b]:
            continue
        name, how = name_patch(labels[a], labels[b], lib[s].family, naming)
        patches.append({"pieces": (a, b), "shift": 0, "surface": s, "face": None,
                        "regions": (labels[a], labels[b]), "wall": name, "naming": how,
                        "hits": count, "point": point})
    inv = {}
    mult = multiplication_table()
    for g in range(1200):
        inv[g] = int(np.flatnonzero(mult[g] == 0)[0])
    seen = set()
    for (p, f, k, q) in mismatched:
        key = min((p, q, k), (q, p, inv[k]))
        if key in seen:
            continue
        seen.add(key)
        name, how = name_patch(labels[p], labels[q], face_family(f, lib), naming)
        pts = ex.face_points.get((p, f), [None])      # (a point of the same patch, up to G)
        patches.append({"canon": key, "pieces": key[:2], "shift": int(key[2]), "surface": None,
                        "face": FACE_NAMES[f],
                        "regions": (labels[key[0]], labels[key[1]]), "wall": name, "naming": how,
                        "hits": len(pts), "point": pts[0]})
    return patches


def face_continuation_links(ex, surfs, patches, face_cont):
    """Links from interior patches to face patches, where a walk continues in a domain face."""
    mult = multiplication_table()
    inv = np.array([int(np.flatnonzero(mult[g] == 0)[0]) for g in range(len(mult))])
    index = {(min(p["pieces"]), max(p["pieces"]), p["surface"]): i
             for i, p in enumerate(patches) if p["face"] is None}
    canon = {p["canon"]: i for i, p in enumerate(patches) if p["face"] is not None}
    out, stats = [], Counter()
    if not face_cont:
        return out, dict(stats)
    betas = np.array([b for _, _, b, _ in face_cont])
    inward = np.array([FACE_N[f] / np.linalg.norm(FACE_N[f]) for _, _, _, f in face_cont])
    inside = betas + 1e-7 * inward
    outside = betas - 1e-7 * inward
    sa = np.sign(surface_values(inside, surfs)).astype(np.int8)
    k2, beyond = locate(point_from_beta(outside), tol=1e-13)
    sb = np.sign(surface_values(beyond, surfs)).astype(np.int8)
    for (key, copy, _, _), ra, shift, rb in zip(face_cont, sa, k2, sb):
        i = index.get(key)
        a, q = ex.keys.get(ra.tobytes()), ex.keys.get(rb.tobytes())
        if i is None or a is None or q is None or shift == 0:
            stats["unresolved"] += 1
            continue
        fwd, rev = (a, q, int(shift)), (q, a, int(inv[shift]))
        if fwd in canon:
            out.append((i, copy, canon[fwd]))
        elif rev in canon:
            out.append((i, int(mult[copy, shift]), canon[rev]))
        else:
            stats["no face patch there"] += 1
            continue
        stats["linked"] += 1
    return out, dict(stats)


def face_walk_links(ex, surfs, patches):
    """Links among face patches (and from them to interior patches) from walks in the faces."""
    from .march import march_faces

    mult = multiplication_table()
    inv = np.array([int(np.flatnonzero(mult[g] == 0)[0]) for g in range(len(mult))])
    canon = {p["canon"]: i for i, p in enumerate(patches) if p["face"] is not None}
    inner = {(min(p["pieces"]), max(p["pieces"]), p["surface"]): i
             for i, p in enumerate(patches) if p["face"] is None}

    def canon_of(a, q, shift):
        if (a, q, shift) in canon:
            return (0, (a, q, shift))
        rev = (q, a, int(inv[shift]))
        if rev in canon:
            return (shift, rev)
        return None

    starts = []
    for p in patches:
        if p["face"] is not None and p["point"] is not None:
            starts.append((FACE_NAMES.index(p["face"]), np.asarray(p["point"], float)))
    links, stats = march_faces(surfs, ex.keys, starts, canon_of)
    out = []
    for (k1, key1), c1, (k2, key2), c2 in links:
        i = canon.get(key1) if k1 == "face" else inner.get(key1)
        j = canon.get(key2) if k2 == "face" else inner.get(key2)
        if i is None or j is None:
            continue
        out.append((i, int(mult[inv[c1], c2]), j))
    return out, stats


def _nearby_on_surface(surf, pt, sign_other, s_index, surfs, rng, n=4):
    """Points on the same surface near pt, with every other surface's sign unchanged."""
    from .march import _project, _signs

    out = []
    for delta in (2e-3, 4e-3, 7e-3, 1.1e-2, 1.6e-2, 2.2e-2)[: 2 * n]:
        u = rng.normal(size=4)
        u -= u.mean()
        x = _project(surf, (np.asarray(pt, float) + delta * u / np.linalg.norm(u))[None])
        sg = _signs(surfs, x)[0]
        sg[s_index] = 0
        if (x @ FACE_N.T >= 0).all() and np.array_equal(sg, sign_other):
            out.append(x[0])
        if len(out) == n:
            break
    return out


def walls_inside_regions(ex, labels, workers: int, cache: dict, surfs):
    """Seams (one region on both sides) where the surface is a different polytope: walls that run
    inside a region (the working data's 'same_both_sides'). A seam counts only when its point and
    most of a few nearby points on the same patch name the same wall (single points near another
    surface can classify as a neighbouring shape)."""
    shapes = load_json("working_data.json")["shapes"]
    walls = [k for k, v in shapes.items() if v["dim"] == "wall"]
    by_sig = region_signatures()
    jobs = [(key, pt) for key, (_, pt) in ex.adjacency.items()
            if labels[key[0]] == labels[key[1]] and pt is not None]
    res = classify_points([pt for _, pt in jobs], 1e-7, workers, cache)

    def wall_of(sig, vkey):
        if sig.startswith("ERR"):
            return "ERR"
        fit = [w for w in walls if counts_match(w, sig)]
        if len({split_parent(w) for w in fit}) == 1 and len(fit) > 1 and vkey:
            fit = [split_name(fit[0], vkey)]
        return fit[0] if len(fit) == 1 else None

    stats = Counter()
    candidates = []
    for (key, pt), (sig, vkey) in zip(jobs, res):
        region = labels[key[0]]
        if sig.startswith("ERR"):
            stats["no classification"] += 1
        elif by_sig.get(sig) == split_parent(region):
            stats["region itself"] += 1
        else:
            candidates.append((key, pt, sig, vkey, wall_of(sig, vkey)))
    rng = np.random.default_rng(11)
    extra_pts, owners = [], []
    for c, (key, pt, *_rest) in enumerate(candidates):
        sign = ex.signs[key[0]].copy()
        sign[key[2]] = 0
        for x in _nearby_on_surface(surfs[key[2]], pt, sign, key[2], surfs, rng):
            extra_pts.append(x)
            owners.append(c)
    extra = classify_points(extra_pts, 1e-7, workers, cache)
    votes = defaultdict(list)
    for c, (sig, vkey) in zip(owners, extra):
        votes[c].append(wall_of(sig, vkey) if by_sig.get(sig) != split_parent(labels[candidates[c][0][0]])
                        else "region")
    out = []
    for c, (key, pt, sig, vkey, name) in enumerate(candidates):
        v = votes.get(c, [])
        agree = sum(1 for x in v if x == name)
        if name is None or agree < max(1, (len(v) + 1) // 2):
            stats["single-point shape, not confirmed nearby"] += 1
            continue
        stats["wall inside a region"] += 1
        region = labels[key[0]]
        out.append({"pieces": key[:2], "shift": 0, "surface": key[2], "face": None,
                    "regions": (region, region), "wall": name,
                    "naming": f"inside region+classified ({agree}/{len(v)} nearby agree)",
                    "hits": ex.adjacency[key][0], "point": pt, "signature": sig,
                    "vertex_key": vkey, "verified": True})
    return out, dict(stats)


def patch_links(ex, patches, march_links, exit_glues, labels, face_links=()):
    """Patch-to-patch links: from the surface walks (inside the domain and across faces), and for
    face patches, along a surface crossing the face."""
    index = {}
    for i, p in enumerate(patches):
        if p["face"] is None:
            index[(min(p["pieces"]), max(p["pieces"]), p["surface"])] = i
    inner = [(index[a], index[b]) for a, b in march_links if a in index and b in index]
    across = [(index[a], k, index[b]) for a, k, b in exit_glues if a in index and b in index]
    across += list(face_links)
    adj_keys = set(ex.adjacency)
    face_by = defaultdict(list)
    for i, p in enumerate(patches):
        if p["face"] is not None:
            face_by[(p["pieces"][0], p["shift"])].append(i)
    for i, p in enumerate(patches):
        if p["face"] is None:
            continue
        a = p["pieces"][0]
        for t in np.flatnonzero(ex.signs[a] != 0):
            a2 = ex.keys.get(flip(ex.signs[a], int(t)))
            if a2 is None or (min(a, a2), max(a, a2), int(t)) not in adj_keys:
                continue
            for j in face_by.get((a2, p["shift"]), ()):
                if j > i and patches[j]["face"] == p["face"]:
                    inner.append((i, j))
    return inner, across


# ------------------------------------------------------------------------------------ lines/points


def carried_curves():
    """Every atlas line piece and point carried to all 1200 copies, with copy counts."""
    w = load_json("working_data.json")
    g = group()
    t = cell_vertices()
    out_lines = {}
    for name, line in w["exact"]["lines"].items():
        pieces = [np.asarray(p, float) for p in line["points"]]
        ends = []
        for p in pieces:
            for e in (p[0], p[-1]):
                ends.append(point_from_beta(e))
        ends = np.array(ends)
        imgs = np.einsum("gij,nj->gni", g, ends)              # (1200, ends, 4)
        keys = np.round(imgs, 7).reshape(len(g), -1)
        distinct = len({tuple(k) for k in keys})
        # join the carried pieces at shared ends: components of the curve on S^3
        n_pc = len(pieces)
        node_end = {}
        rows, cols = [], []
        for k in range(len(g)):
            for j in range(n_pc):
                for side in (0, 1):
                    key = tuple(np.round(imgs[k, 2 * j + side], 7) + 0.0)
                    node = k * n_pc + j
                    other = node_end.setdefault(key, node)
                    if other != node:
                        rows.append(node)
                        cols.append(other)
        comp = _components(len(g) * n_pc, np.array(rows, int), np.array(cols, int))
        comps = comp.reshape(len(g), n_pc)
        n_comp = len(np.unique(comps))
        mult = multiplication_table()
        stab = [int(x) for x in range(len(g)) if comps[mult[x, 0], 0] == comps[0, 0]]
        out_lines[name] = {
            "domain_pieces": [p.tolist() for p in pieces],
            "carried_copies": distinct,
            "components": int(n_comp),
            "stabilizer_order": len(stab),
            "component_members": [[int(k), int(j)] for k, j in zip(*np.nonzero(comps == comps[0, 0]))],
        }
    out_points = {}
    for p in w["exact"]["points"]:
        x = point_from_beta(np.asarray(p["float"], float))
        imgs = np.round(g @ x, 7)
        distinct = len({tuple(r + 0.0) for r in imgs})
        out_points.setdefault(p["name"], []).append({
            "beta": p["b"], "beta_float": p["float"], "copies_on_sphere": distinct,
            "stabilizer_order": 1200 // distinct, "surfaces": p["surfaces"],
        })
    del t
    return out_lines, out_points


# ------------------------------------------------------------------------------------------ driver


def _save(obj, path) -> None:
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def _load(path):
    import pickle
    with open(path, "rb") as handle:
        return pickle.load(handle)


def build(n_chords: int, workers: int, seed: int = 0, rounds: int = 6,
          resume_library: bool = False, from_checkpoint: bool = False) -> dict:
    """Build the map. ``from_checkpoint`` reuses the saved pieces and surface walks (the slow
    part) and redoes the naming and everything after it, e.g. after the working data changes."""
    out_dir = data_dir() / "sphere_map"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / ".cache" / "classify.pkl"
    checkpoint = out_dir / ".cache" / "checkpoint.pkl"
    cache: dict = _load(cache_path) if cache_path.exists() else {}
    if cache:
        log(f"classification cache: {len(cache)} points")
    if from_checkpoint:
        state = _load(checkpoint)
        log(f"checkpoint: {len(state['ex'].signs)} pieces")
    else:
        state = explore_and_walk(n_chords, workers, seed, rounds, resume_library, out_dir, cache)
        _save(cache, cache_path)
        _save(state, checkpoint)
    try:
        return finish(state, workers, cache, out_dir)
    finally:
        _save(cache, cache_path)


def explore_and_walk(n_chords, workers, seed, rounds, resume_library, out_dir, cache) -> dict:
    saved = out_dir / "surface_library.json"
    if resume_library and saved.exists():
        with open(saved, encoding="utf-8") as handle:
            lib = Library.from_json(json.load(handle))
    else:
        lib = Library()
    unexplained: list = []
    for rnd in range(rounds):
        surfs = lib.surfaces()
        log(f"round {rnd}: {len(lib)} surfaces ({sum(k.in_source for k in lib)} from surfaces.json)")
        ex = explore(n_chords, seed=seed, surfs=surfs, keep_face_points=24)
        log(f"chords: {ex.chords}, pieces: {len(ex.signs)}, adjacencies: {len(ex.adjacency)}, "
            f"rejected chords: {ex.rejected}")
        labels, checks = label_pieces(ex, workers, cache)
        bad = [i for i, c in enumerate(checks) if not c["agree"]]
        log(f"labelled {len(labels)} pieces; {len(bad)} with samples naming different regions")
        if not bad or rnd == rounds - 1:
            break
        added, unexplained = refine(ex, labels, checks, lib, workers)
        log(f"added {added} surface images; {len(unexplained)} shape changes on no atlas surface")
        if not added:
            break
    lib.save(out_dir / "surface_library.json")
    surfs = lib.surfaces()
    starts = defaultdict(list)
    for (a, b, sidx), (_, pt) in ex.adjacency.items():
        starts[sidx].append(pt)
    march_links, exits, observed, march_stats = march(surfs, ex.keys, starts)
    added_pairs = 0
    for k, pt in observed.items():
        if k not in ex.adjacency:
            ex.adjacency[k] = [0, pt]
            added_pairs += 1
    exit_glues, face_cont, exit_stats = glue_exits(surfs, ex.keys, exits)
    log(f"surface walks: {march_stats}; {added_pairs} new piece pairs; exits {exit_stats}")
    return dict(ex=ex, lib=lib, unexplained=unexplained, march_links=march_links,
                march_stats=march_stats, exit_glues=exit_glues, face_cont=face_cont,
                exit_stats=exit_stats)


def finish(state, workers, cache, out_dir) -> dict:
    ex, lib = state["ex"], state["lib"]
    surfs = lib.surfaces()
    march_links, exit_glues, face_cont = (state["march_links"], state["exit_glues"],
                                          state["face_cont"])
    march_stats, exit_stats, unexplained = (state["march_stats"], dict(state["exit_stats"]),
                                            state["unexplained"])
    labels, checks = label_pieces(ex, workers, cache)
    log(f"labels: {len(labels)} pieces, {sum(not c['agree'] for c in checks)} with samples "
        f"naming different regions, {sum(x.startswith('?') for x in labels)} unknown; "
        f"{len(set(labels))} regions")
    volumes = piece_volumes(ex, surfs)
    glued, glue_stats = glue_faces(ex, surfs)
    log(f"face gluing: {len(glued)} links {glue_stats}")
    n = len(ex.signs)
    comp, mismatched = region_components(labels, ex.adjacency, glued, n)
    log(f"region components on S^3: {len(np.unique(comp))}; glue label mismatches: {len(mismatched)}")
    region_pieces: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(labels):
        region_pieces[name].append(i)
    regions = describe_components(comp, region_pieces)
    patches = wall_patches(ex, labels, ex.adjacency, mismatched, lib)
    naming_check = verify_patch_names(patches, workers, cache)
    inside, inside_stats = walls_inside_regions(ex, labels, workers, cache, surfs)
    patches += inside
    log(f"seams classified: {inside_stats}")
    log(f"wall patches: {len(patches)}; naming: {Counter(p['naming'] .split(':')[0] for p in patches)}"
        f"; check: {dict(naming_check)}")
    face_links, face_link_stats = face_continuation_links(ex, surfs, patches, face_cont)
    exit_stats = dict(exit_stats, face_patch_links=face_link_stats)
    walk_links, face_walk_stats = face_walk_links(ex, surfs, patches)
    exit_stats = dict(exit_stats, face_walks=face_walk_stats)
    log(f"face walks: {face_walk_stats}")
    inner, across = patch_links(ex, patches, march_links, exit_glues, labels,
                                list(face_links) + walk_links)
    wall_nodes: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(patches):
        wall_nodes[p["wall"] or f"unnamed {'|'.join(sorted(p['regions']))}"].append(i)
    keyname = {i: name for name, items in wall_nodes.items() for i in items}
    same_inner = [(i, j) for i, j in inner if keyname[i] == keyname[j]]
    same_across = [(i, k, j) for i, k, j in across if keyname[i] == keyname[j]]
    wcomp = _components(1200 * len(patches), *sphere_edges(same_inner, same_across, len(patches)))
    wcomp = wcomp.reshape(1200, len(patches))
    walls = describe_components(wcomp, wall_nodes)
    log(f"wall components: {sum(v['components'] for v in walls.values())} over {len(walls)} walls")
    lines, points = carried_curves()
    log("lines and points carried")
    result = dict(ex=ex, labels=labels, checks=checks, volumes=volumes, glued=glued,
                  glue_stats=glue_stats, comp=comp, mismatched=mismatched, regions=regions,
                  region_pieces=region_pieces, patches=patches, inner=inner, across=across,
                  wcomp=wcomp, walls=walls, lines=lines, points=points, lib=lib,
                  unexplained=unexplained, march_stats=march_stats, exit_stats=exit_stats,
                  naming_check=dict(naming_check))
    from .export import write_all
    write_all(result, out_dir)
    log(f"written to {out_dir}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chords", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=6, help="refinement rounds at most")
    ap.add_argument("--resume-library", action="store_true",
                    help="start from the saved surface_library.json instead of surfaces.json")
    ap.add_argument("--from-checkpoint", action="store_true",
                    help="reuse the saved pieces and surface walks; redo naming and the rest")
    args = ap.parse_args()
    build(args.chords, args.workers, args.seed, rounds=args.rounds,
          resume_library=args.resume_library, from_checkpoint=args.from_checkpoint)


if __name__ == "__main__":
    main()
