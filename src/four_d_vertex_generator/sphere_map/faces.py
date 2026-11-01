"""Faces, edges, corners and line pieces of the 3-sphere map.

A face is one connected piece of one wall sheet between one region component and another: the
wall patches joined along surface walks (inside the domain and across its faces) that keep the
same wall and the same region component on each side.

An edge is where a face ends. The surface walks record each crossing: walking on the face's
surface s, the walk crosses surface t into something that is not the same face. The edge lies on
the curve s = t = 0 and is traced exactly along it (Newton onto both equations at every step) from
that crossing in both directions, until a third surface changes sign (a corner, solved exactly on
s = t = u = 0) or the curve leaves the domain. Around an edge are four pieces (the signs of s and
t); their regions and the walls between them say what meets there, which is how lines split into
pieces: one piece per stretch with the same walls and regions round it.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .chords import FACE_N
from .frame import multiplication_table, point_from_beta


# ------------------------------------------------------------------------------------- geometry


def _grad0(surf, x):
    g = surf.grad(x[None])[0]
    return g - g.mean()


def polish(surfs, idx: list[int], x: np.ndarray, iters: int = 12) -> np.ndarray | None:
    """x moved onto the common zero set of the surfaces idx (min-norm Newton, sum of betas kept)."""
    x = np.asarray(x, float).copy()
    for _ in range(iters):
        f = np.array([surfs[i].f(x[None])[0] for i in idx])
        if np.abs(f).max() < 1e-15:
            break
        jac = np.array([_grad0(surfs[i], x) for i in idx])
        step = np.linalg.lstsq(jac, -f, rcond=None)[0]
        x = x + step
    f = np.array([surfs[i].f(x[None])[0] for i in idx])
    return x if np.abs(f).max() < 1e-11 else None


def edge_tangent(surfs, s: int, t: int, x: np.ndarray) -> np.ndarray:
    m = np.vstack([np.ones(4), _grad0(surfs[s], x), _grad0(surfs[t], x)])
    v = np.linalg.svd(m)[2][-1]
    return v / np.linalg.norm(v)


class _Fast:
    """All surface values at one point: planes as one product, curved ones in a short loop."""

    def __init__(self, surfs):
        self.surfs = surfs
        self.plane_ix = [s.index for s in surfs if s.kind == "plane"]
        self.normals = np.array([surfs[i].normal for i in self.plane_ix])
        self.curved = [s for s in surfs if s.kind != "plane"]

    def values(self, x):
        out = np.empty(len(self.surfs))
        out[self.plane_ix] = self.normals @ x
        for c in self.curved:
            out[c.index] = c.f(x[None])[0]
        return out


_FAST: dict = {}


def _signs(surfs, x):
    fast = _FAST.get(id(surfs))
    if fast is None:
        fast = _FAST[id(surfs)] = _Fast(surfs)
    return np.sign(fast.values(x)).astype(np.int8)


def _face_of_surface(surfs, i):
    sf = surfs[i]
    if sf.kind != "plane":
        return None
    for f, n in enumerate(FACE_N):
        if abs(abs(float(sf.normal @ n)) / np.linalg.norm(n) - 1) < 1e-9:
            return f
    return None


def _regions_round(surfs, s, t, x, keys, labels):
    """Regions in the four quadrants round the curve at x (inside the domain), or None."""
    gs, gt = _grad0(surfs[s], x), _grad0(surfs[t], x)
    out = []
    for a in (1, -1):
        for b in (1, -1):
            d = np.linalg.lstsq(np.vstack([gs, gt, np.ones(4)]), np.array([a, b, 0.0]),
                                rcond=None)[0]
            y = x + 1e-7 * d / np.linalg.norm(d)
            if (y @ FACE_N.T < 0).any():             # (a quadrant in the neighbouring copy)
                from .frame import locate
                _, yb = locate(point_from_beta(y / y.sum())[None], tol=1e-13)
                y = yb[0]
            pc = keys.get(_signs(surfs, y).tobytes())
            out.append(None if pc is None else labels[pc])
    return tuple(out)


def trace_edge(surfs, s: int, t: int, x0: np.ndarray, step: float = 1e-3, max_steps: int = 4000,
               keys=None, labels=None):
    """Trace the curve s = t = 0 from x0 both ways.

    Where another surface crosses the curve, the regions round it are compared on both sides:
    if they are the same it is only a cut (the surface is the extension of some wall that is not
    there) and the trace goes on; otherwise it is a corner and the trace stops there.

    Returns (polyline, ends): ends are dicts {kind: corner|domain|stopped, surface, point, cuts}.
    """
    x0 = polish(surfs, [s, t], x0)
    if x0 is None:
        return None, []
    base = _signs(surfs, x0)
    base[[s, t]] = 0
    # surfaces through the curve itself (other walls meeting along this line) carry no sign here
    fast = _FAST.get(id(surfs)) or _Fast(surfs)
    vals = fast.values(x0)
    scale = np.array([np.linalg.norm(_grad0(sf, x0)) for sf in surfs]) + 1e-300
    on_curve = np.flatnonzero(np.abs(vals) / scale < 1e-9)
    base[on_curve] = 0
    own_faces = [f for f in (_face_of_surface(surfs, s), _face_of_surface(surfs, t)) if f is not None]
    halves, ends = [], []
    base0 = base.copy()
    cuts = []
    for direction in (1.0, -1.0):
        base = base0.copy()
        pts = [x0]
        x = x0
        v = direction * edge_tangent(surfs, s, t, x0)
        end = {"kind": "stopped", "surface": None, "point": None}
        for _ in range(max_steps):
            y = polish(surfs, [s, t], x + step * v)
            if y is None:
                break
            sg = _signs(surfs, y)
            sg[[s, t]] = 0
            dom = y @ FACE_N.T
            dom[own_faces] = 1.0
            changed = np.flatnonzero((sg != base) & (base != 0))
            if len(changed) and not (dom < 0).any() and keys is not None:
                rr0 = _regions_round(surfs, s, t, x, keys, labels)
                rr1 = _regions_round(surfs, s, t, y, keys, labels)
                if None not in rr0 and rr0 == rr1:
                    base = sg.copy()           # only a cut: carry on with the new signs
                    base[on_curve] = 0
                    cuts.append([int(k) for k in changed])
                    x = y
                    pts.append(y)
                    nv = edge_tangent(surfs, s, t, y)
                    v = nv if nv @ v > 0 else -nv
                    continue
            if len(changed) or (dom < 0).any():
                # bisect to the event, then solve the corner exactly
                lo, hi = x, y
                for _ in range(50):
                    mid = polish(surfs, [s, t], (lo + hi) / 2)
                    if mid is None:
                        break
                    sm = _signs(surfs, mid)
                    sm[[s, t]] = 0
                    dm = mid @ FACE_N.T
                    dm[own_faces] = 1.0
                    bad = ((sm != base) & (base != 0)).any() or (dm < 0).any()
                    lo, hi = (lo, mid) if bad else (mid, hi)
                if len(changed):
                    real = [int(k) for k in changed if _face_of_surface(surfs, int(k)) is None]
                    u = real[0] if real else int(changed[0])
                    c = polish(surfs, [s, t, u], (lo + hi) / 2)
                    pt = c if c is not None else (lo + hi) / 2
                    # a corner only where a real surface meets the curve; a domain face alone
                    # just cuts the curve where the copy ends (it goes on in the next copy)
                    vals = fast.values(pt)
                    sc = np.array([np.linalg.norm(_grad0(sf, pt)) for sf in surfs]) + 1e-300
                    through = [int(k) for k in np.flatnonzero(np.abs(vals) / sc < 1e-8)
                               if k not in (s, t) and k not in on_curve
                               and _face_of_surface(surfs, int(k)) is None]
                    kind = "corner" if (real or through) else "domain"
                    end = {"kind": kind, "surface": u, "point": pt,
                           "exact": c is not None, "also": [int(k) for k in changed if k != u],
                           "through": through}
                    if kind == "domain":
                        end["face"] = _face_of_surface(surfs, u)
                else:
                    f = int(np.argmin(dom))
                    end = {"kind": "domain", "face": f, "point": (lo + hi) / 2}
                pts.append(end["point"])
                break
            nv = edge_tangent(surfs, s, t, y)
            v = nv if nv @ v > 0 else -nv
            x = y
            pts.append(y)
        halves.append(pts)
        ends.append(end)
    line = halves[1][::-1] + halves[0][1:]
    for e in ends:
        e["cuts_crossed"] = len(cuts)
    return np.array(line), ends


def pieces_round(keys: dict, surfs, s: int, t: int, x: np.ndarray, eps: float = 1e-6):
    """The pieces round the curve s = t = 0 at x, found by stepping off it in the four quadrants
    of (s, t) (other surfaces through the same curve sort themselves out). A quadrant outside the
    domain is looked up in the copy it lies in: {(sign s, sign t): (piece, copy, beta there)}."""
    from .frame import locate

    gs, gt = _grad0(surfs[s], x), _grad0(surfs[t], x)
    out = {}
    for a in (1, -1):
        for b in (1, -1):
            d = np.linalg.lstsq(np.vstack([gs, gt, np.ones(4)]), np.array([a, b, 0.0]),
                                rcond=None)[0]
            y = x + eps * d / np.linalg.norm(d)
            copy = 0
            if (y @ FACE_N.T < 0).any():
                k, yb = locate(point_from_beta(y / y.sum())[None], tol=1e-13)
                copy, y = int(k[0]), yb[0]
            out[(a, b)] = (keys.get(_signs(surfs, y).tobytes()), copy, y)
    return out


# ---------------------------------------------------------------------------------------- faces


def face_components(patches, comp, wcomp_inputs, region_comp_sides):
    """Face id of every (copy, patch): wall links kept only where both sides' region components
    agree. Region components are equivariant under the group, so agreeing in copy 0 means
    agreeing in every copy."""
    from .build import _components, sphere_edges

    ca, cb = region_comp_sides
    same_inner, same_across = wcomp_inputs
    mult = multiplication_table()

    def same(i, k, j):
        a = {int(ca[0, i]), int(cb[0, i])}
        b = {int(ca[k, j]), int(cb[k, j])}
        return a == b

    inner = [(i, j) for i, j in same_inner if same(i, 0, j)]
    across = [(i, k, j) for i, k, j in same_across if same(i, k, j)]
    del mult
    fc = _components(1200 * len(patches), *sphere_edges(inner, across, len(patches)))
    return fc.reshape(1200, len(patches))


def polish_many(surfs, s: int, t: int, x: np.ndarray, iters: int = 14) -> np.ndarray:
    """Many points moved onto s = t = 0 at once (min-norm Newton); rows that fail are NaN."""
    x = np.asarray(x, float).copy()
    S, T = surfs[s], surfs[t]
    for _ in range(iters):
        f = np.stack([S.f(x), T.f(x)], axis=1)                      # (n, 2)
        gs, gt = S.grad(x), T.grad(x)
        gs = gs - gs.mean(axis=1, keepdims=True)
        gt = gt - gt.mean(axis=1, keepdims=True)
        J = np.stack([gs, gt], axis=1)                              # (n, 2, 4)
        JJ = J @ J.transpose(0, 2, 1)                               # (n, 2, 2)
        det = JJ[:, 0, 0] * JJ[:, 1, 1] - JJ[:, 0, 1] * JJ[:, 1, 0]
        ok = np.abs(det) > 1e-24
        inv = np.zeros_like(JJ)
        inv[ok, 0, 0], inv[ok, 1, 1] = JJ[ok, 1, 1] / det[ok], JJ[ok, 0, 0] / det[ok]
        inv[ok, 0, 1], inv[ok, 1, 0] = -JJ[ok, 0, 1] / det[ok], -JJ[ok, 1, 0] / det[ok]
        lam = np.einsum("nij,nj->ni", inv, -f)
        x = x + np.einsum("nij,ni->nj", J, lam)
        x[~ok] = np.nan
    f = np.stack([S.f(x), T.f(x)], axis=1)
    bad = ~(np.abs(f).max(axis=1) < 1e-11)
    x[bad] = np.nan
    return x


def _enumerate_job(args):
    """Every piece of the curve s = t = 0 in the domain: seeds dropped onto it, each new one
    traced corner to corner."""
    s, t, n_seeds, seed = args
    surfs, keys, labels = _POOL["surfs"], _POOL["keys"], _POOL["labels"]
    rng = np.random.default_rng(seed)
    b = rng.dirichlet(np.ones(4) * 0.7, n_seeds)
    swap = b[:, 2] < b[:, 3]
    b[swap] = b[swap][:, [1, 0, 3, 2]]
    x = polish_many(surfs, s, t, b)
    x = x[~np.isnan(x).any(axis=1)]
    x = x[(x @ FACE_N.T >= -1e-12).all(axis=1)]
    out, seen_pts = [], np.zeros((0, 4))
    for p0 in x:
        if len(seen_pts) and np.min(np.linalg.norm(seen_pts - p0, axis=1)) < 4e-3:
            continue
        line, ends = trace_edge(surfs, s, t, p0, step=2e-3, keys=keys, labels=labels)
        if line is None or len(line) < 2:
            continue
        seen_pts = np.vstack([seen_pts, line])
        out.append((s, t, line, ends))
    return out


_POOL = {}


def _init(surfs, keys, labels):
    _POOL.update(surfs=surfs, keys=keys, labels=labels)


def _trace_job(args):
    s, t, x0 = args
    line, ends = trace_edge(_POOL["surfs"], s, t, x0, step=2e-3, keys=_POOL["keys"],
                            labels=_POOL["labels"])
    return s, t, line, ends


def _key(x, digits=9):
    return tuple(np.round(np.asarray(x, float) / np.sum(x), digits) + 0.0)


def build_faces(ex, lib, patches, comp, same_inner, same_across, march_links, labels,
                workers: int, ended=()) -> dict:
    """Faces (one record per orbit), edges with exact curves, corners, and the congruence test."""
    from multiprocessing import Pool

    from .build import describe_components, log
    from .classify import counts_match, key_of_beta
    from .export import _patch_instances
    from .frame import group, load_json
    from .march import _face_surfaces

    surfs = lib.surfaces()
    mult = multiplication_table()
    inv = np.array([int(np.flatnonzero(mult[g_] == 0)[0]) for g_ in range(len(mult))])
    ca, cb = _patch_instances(patches, comp)
    wall_of = [p["wall"] or f"unnamed {'|'.join(sorted(p['regions']))}" for p in patches]
    fcomp = face_components(patches, comp, (same_inner, same_across), (ca, cb))
    groups = defaultdict(list)
    for i, p in enumerate(patches):
        groups[" | ".join([wall_of[i]] + sorted(p["regions"]))].append(i)
    faces = describe_components(fcomp, groups)
    log(f"faces: {sum(v['components'] for v in faces.values())} on the sphere in "
        f"{sum(len(v['orbits']) for v in faces.values())} orbits ({len(faces)} labels)")

    # ---------------------------------------------------------------- edges: where faces end
    if not isinstance(march_links, dict):           # (walks recorded before crossing details)
        march_links = {k: [] for k in march_links}
    index = {(min(p["pieces"]), max(p["pieces"]), p["surface"]): i
             for i, p in enumerate(patches) if p["face"] is None}
    canon = {p["canon"]: i for i, p in enumerate(patches) if p["face"] is not None}
    face_surface = {f: si for si, f in _face_surfaces(surfs).items()}
    by_pair = defaultdict(list)
    for (k1, k2), recs in march_links.items():
        i, j = index.get(k1), index.get(k2)
        if i is None and j is None:
            continue
        if i is not None and j is not None and fcomp[0, i] == fcomp[0, j]:
            continue
        for crossed, point in recs:
            # (at a line where several surfaces meet, the walk crosses them together; the edge
            # is the curve s = t = 0 for any one of them)
            s_, t_ = k1[2], crossed[0]
            by_pair[(min(s_, t_), max(s_, t_))].append(point)
    # walls that stop exactly at a domain face: edge on the wall's surface and the face's plane
    n_face_starts = 0
    for key, f, xn in ended:
        if key in index and f in face_surface:
            s_, t_ = key[2], face_surface[f]
            by_pair[(min(s_, t_), max(s_, t_))].append(np.asarray(xn, float))
            n_face_starts += 1
    jobs, seen_cfg = [], set()
    for (s_, t_), pts in by_pair.items():
        for x0 in pts:
            x1 = polish(surfs, [s_, t_], x0)
            if x1 is None:
                continue
            rnd = pieces_round(ex.keys, surfs, s_, t_, x1)
            cfg = (s_, t_, tuple(sorted((k, (-1 if v[0] is None else v[0], v[1]))
                                        for k, v in rnd.items())))
            if cfg in seen_cfg:
                continue
            seen_cfg.add(cfg)
            jobs.append((s_, t_, x1))
    log(f"edge curves from the walks: {len(jobs)} ({n_face_starts} where a wall meets a face)")
    # every curve where two wall-carrying surfaces (or one and a domain face) meet, enumerated
    carrying = sorted({p["surface"] for p in patches if p["surface"] is not None}
                      | set(face_surface.values()))
    pairs = [(a, b, 400, 1000 + n) for n, (a, b) in enumerate(
        (a, b) for i, a in enumerate(carrying) for b in carrying[i + 1:])]
    log(f"enumerating meeting curves of {len(pairs)} surface pairs")
    with Pool(workers, initializer=_init, initargs=(surfs, ex.keys, labels)) as pool:
        traced = pool.map(_trace_job, jobs, chunksize=4)
        for chunk in pool.imap_unordered(_enumerate_job, pairs, chunksize=8):
            traced.extend(chunk)
    log(f"curve pieces traced: {len(traced)}")

    def patch_between(qa, qb, on):
        """(patch, copy of its side A) between two quadrant pieces, or (None, None)."""
        (a, ka, ya), (b, kb, yb) = qa, qb
        if a is None or b is None:
            return None, None
        if ka == 0 and kb == 0:
            pid = index.get((min(a, b), max(a, b), on))
            return (pid, 0) if pid is not None else (None, None)
        if ka == kb:                           # both in one neighbouring copy: its own surface
            sa, sb = _signs(surfs, ya), _signs(surfs, yb)
            diff = np.flatnonzero(sa != sb)
            if len(diff) != 1:
                return None, None
            pid = index.get((min(a, b), max(a, b), int(diff[0])))
            return (pid, ka) if pid is not None else (None, None)
        for (p_, kp), (q_, kq) in (((a, ka), (b, kb)), ((b, kb), (a, ka))):
            shift = int(mult[inv[kp], kq])
            if (p_, q_, shift) in canon:
                return canon[(p_, q_, shift)], kp
        return None, None

    edges, seg_keys = [], {}
    for s_, t_, line, ends in traced:
        if line is None or len(line) < 2:
            continue
        k = (s_, t_, _key(line[0], 7), _key(line[-1], 7))
        k2 = (s_, t_, _key(line[-1], 7), _key(line[0], 7))
        if k in seg_keys or k2 in seg_keys:
            continue
        seg_keys[k] = len(edges)
        pr = pieces_round(ex.keys, surfs, s_, t_, line[len(line) // 2])
        walls_round = []
        for pair, on in ((((1, 1), (-1, 1)), s_), (((1, -1), (-1, -1)), s_),
                         (((1, 1), (1, -1)), t_), (((-1, 1), (-1, -1)), t_)):
            qa, qb = pr[pair[0]], pr[pair[1]]
            if qa[0] is None or qb[0] is None:
                continue
            pid, pk = patch_between(qa, qb, on)
            if labels[qa[0]] != labels[qb[0]] or pid is not None:
                walls_round.append({"on": int(on), "between": [labels[qa[0]], labels[qb[0]]],
                                    "patch": pid, "patch_copy": pk,
                                    "wall": None if pid is None else wall_of[pid]})
        if not any(w["patch"] is not None for w in walls_round):
            del seg_keys[k]                  # (a curve with no wall along it: not an edge)
            continue
        edges.append({"id": len(edges), "surfaces": [int(s_), int(t_)],
                      "pieces_round": {f"{a:+d}{b:+d}": (None if v[0] is None else
                                                          [int(v[0]), int(v[1])])
                                       for (a, b), v in pr.items()},
                      "regions_round": sorted({labels[v[0]] for v in pr.values()
                                               if v[0] is not None}),
                      "walls_round": walls_round,
                      "polyline": line, "ends": ends})
    log(f"edges: {len(edges)}")

    # ---------------------------------------------------------------- what each edge is
    shapes = load_json("working_data.json")["shapes"]
    line_names = [k for k, v in shapes.items() if v["dim"] in ("line", "point")]
    for e in edges:
        mid = e["polyline"][len(e["polyline"]) // 2]
        sig, _ = key_of_beta(mid, hull_tol=1e-7)
        fit = [n for n in line_names if not sig.startswith("ERR") and counts_match(n, sig)]
        e["line_shape"] = fit[0] if len(fit) == 1 else (fit or None)
        e["signature"] = sig

    # ---------------------------------------------------------------- corners
    atlas = load_json("working_data.json")["exact"]["points"]
    atlas_pts = [(p["name"], np.asarray(p["float"], float) / np.sum(p["float"])) for p in atlas]
    g = group()
    corners, corner_ix = [], {}
    for e in edges:
        for end in e["ends"]:
            if end["kind"] != "corner":
                continue
            x = np.asarray(end["point"], float)
            k = _key(x, 8)
            if k not in corner_ix:
                xp = point_from_beta(x)
                name = None
                for n, a in atlas_pts:     # an atlas point, in this copy or any other
                    ap = point_from_beta(a)
                    if np.min(np.linalg.norm(g @ ap - xp, axis=1)) < 1e-7:
                        name = n
                        break
                corner_ix[k] = len(corners)
                corners.append({"id": len(corners), "beta": x, "exact": bool(end.get("exact")),
                                "surfaces": [*e["surfaces"], end["surface"]], "atlas_point": name})
            end["corner"] = corner_ix[k]

    # ---------------------------------------------------------------- faces: outlines, congruence
    edges_of_patch = defaultdict(set)      # patch -> {(edge, copy of the patch instance)}
    for e in edges:
        for w in e["walls_round"]:
            if w["patch"] is not None:
                edges_of_patch[w["patch"]].add((e["id"], int(w["patch_copy"])))
    maps = np.concatenate([g, g @ np.load(_q_path())])
    out_faces = {}
    for label, info in faces.items():
        recs = []
        for o in info["orbits"]:
            members = o["members"]
            outline = sorted({(int(mult[k, inv[kk]]), eid) for k, i in members
                              for eid, kk in edges_of_patch.get(i, ())})
            cpts, perim = [], 0.0
            for k, eid in outline:
                e = edges[eid]
                line = point_from_beta(e["polyline"]) @ g[k].T
                perim += float(np.sum(np.arccos(np.clip(np.sum(line[1:] * line[:-1], axis=1),
                                                        -1, 1))))
                for end in e["ends"]:
                    if end["kind"] == "corner":
                        cpts.append(point_from_beta(np.asarray(end["point"], float)) @ g[k].T)
            cpts = np.unique(np.round(np.array(cpts), 9), axis=0) if cpts else np.zeros((0, 4))
            recs.append(dict({k: v for k, v in o.items() if k != "stabilizer"},
                             stabilizer=o["stabilizer"], outline=outline,
                             corners=cpts, perimeter=perim))
        # congruence: every orbit with this label against the first, under G and G.Q
        ref = recs[0]
        for r in recs:
            r["congruent_to_first"] = _congruent(ref["corners"], r["corners"], maps,
                                                 ref["perimeter"], r["perimeter"])
        out_faces[label] = {"components_on_sphere": info["components"], "orbits": recs}
    return {"faces": out_faces, "fcomp": fcomp, "edges": edges, "corners": corners}


def _q_path():
    from .frame import source_dir
    return source_dir() / "normalizer_q.npy"


def _congruent(a: np.ndarray, b: np.ndarray, maps: np.ndarray, pa: float, pb: float):
    """Index of a map in G or G.Q carrying corner set a onto b (and equal perimeters), or None."""
    from scipy.spatial import cKDTree

    # (perimeters are polyline lengths: the same curve sampled from different points differs
    # by ~1e-5 of its length; corners are exact points)
    if len(a) != len(b) or abs(pa - pb) > 2e-3 * max(pa, pb, 1e-3):
        return None
    if len(a) == 0:
        return -1
    tree = cKDTree(b)
    for mi, m in enumerate(maps):
        d, _ = tree.query(a @ m.T)
        if d.max() < 1e-6:
            return int(mi)
    return None


def split_atlas_lines(edges, surfs) -> dict:
    """Each atlas line cut into pieces by the traced edge it lies on (exact membership: both of
    the edge's surfaces vanish there and the four pieces round it are the edge's)."""
    from scipy.spatial import cKDTree

    from .frame import load_json

    verts, owner = [], []
    for e in edges:
        for x in e["polyline"]:
            verts.append(np.asarray(x) / np.sum(x))
            owner.append(e["id"])
    tree = cKDTree(np.array(verts))
    out = {}
    for name, line in load_json("working_data.json")["exact"]["lines"].items():
        pieces = []
        for pi, pts in enumerate(line["points"]):
            pts = np.asarray(pts, float)
            pts = pts / pts.sum(axis=1, keepdims=True)
            ids = []
            for x in pts:
                cand = {owner[j] for j in tree.query(x, k=12)[1]}
                hit = None
                for eid in cand:
                    s, t = edges[eid]["surfaces"]
                    if abs(surfs[s].f(x[None])[0]) < 1e-7 and abs(surfs[t].f(x[None])[0]) < 1e-7:
                        hit = eid
                        break
                ids.append(hit)
            run_start = 0
            for k in range(1, len(ids) + 1):
                if k == len(ids) or ids[k] != ids[run_start]:
                    eid = ids[run_start]
                    pieces.append({"atlas_piece": pi, "from": run_start, "to": k - 1,
                                   "edge": eid,
                                   "walls_round": None if eid is None else
                                   edges[eid]["walls_round"],
                                   "regions_round": None if eid is None else
                                   edges[eid]["regions_round"]})
                    run_start = k
        out[name] = pieces
    return out
