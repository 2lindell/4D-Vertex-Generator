"""Walk along each library surface to see how its patches join.

A chord crosses a surface at a point; from there we march along the surface itself (steps in a
tangent direction, each projected back onto the surface by Newton). At every step the other
surfaces' signs give the pair of pieces on either side; where one of them changes sign the walk
has crossed from one patch to the next along their shared edge. Where the walk leaves the domain
through a face, the next point (still on the surface) is carried into the neighbouring copy, where
it lies on an image of the surface: that gives the patch it continues into.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .chords import FACE_N
from .frame import locate, point_from_beta


def _project(surf, x: np.ndarray, iters: int = 3) -> np.ndarray:
    for _ in range(iters):
        f = surf.f(x)
        g = surf.grad(x)
        g = g - g.mean(axis=1, keepdims=True)
        gg = np.einsum("ij,ij->i", g, g)
        x = x - (f / np.where(gg > 0, gg, 1.0))[:, None] * g
    return x


def _tangent(surf, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    g = surf.grad(x)
    g = g - g.mean(axis=1, keepdims=True)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-300
    u = u - u.mean(axis=1, keepdims=True)
    u = u - np.einsum("ij,ij->i", u, g)[:, None] * g
    return u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-300)


def _signs(surfs, x: np.ndarray) -> np.ndarray:
    return np.sign(np.stack([s.f(x) for s in surfs], axis=1)).astype(np.int8)


def pair_at(keys: dict, sign: np.ndarray, s: int) -> tuple[int | None, int | None]:
    a = sign.copy()
    b = sign.copy()
    a[s], b[s] = 1, -1
    return keys.get(a.tobytes()), keys.get(b.tobytes())


def march(surfs, keys: dict, starts: dict[int, list[np.ndarray]], step: float = 2e-3,
          max_steps: int = 400, rng: np.random.Generator | None = None, directions: int = 2):
    """Walk every surface from its start points.

    Returns (links, exits, observed, stats):
      links:    {(key1, key2): [(crossed surfaces, point near the crossing), ...]} for patch/seam
                keys (a, b, s) joined along a crossing inside the domain (a few points each)
      exits:    (key, face index, x_next beta outside the domain) where a walk leaves
      observed: keys seen on the way (pairs of pieces across s), with a point near each
    """
    rng = rng or np.random.default_rng(0)
    links: dict[tuple[tuple, tuple], list] = {}
    exits: list[tuple[tuple, int, np.ndarray]] = []
    observed: dict[tuple, np.ndarray] = {}
    stats = defaultdict(int)

    def key(a, b, s):
        return (min(a, b), max(a, b), s)

    for s, pts in starts.items():
        surf = surfs[s]
        x0 = np.repeat(np.asarray(pts, float), directions, axis=0)
        x = _project(surf, x0)
        u = rng.normal(size=x.shape)
        u = _tangent(surf, x, u)
        u[1::2] = -u[0::2] if directions == 2 else u[1::2]
        alive = np.ones(len(x), bool)
        sign = _signs(surfs, x)
        sign[:, s] = 0
        prev_key = []
        for i in range(len(x)):
            a, b = pair_at(keys, sign[i], s)
            prev_key.append(key(a, b, s) if a is not None and b is not None else None)
            if prev_key[-1] is not None:
                observed.setdefault(prev_key[-1], x[i].copy())
        for _ in range(max_steps):
            idx = np.flatnonzero(alive)
            if not len(idx):
                break
            xn = _project(surf, x[idx] + step * u[idx])
            moved = xn - x[idx]
            un = _tangent(surf, xn, moved)
            dom = xn @ FACE_N.T
            out = (dom < 0).any(axis=1)
            sn = _signs(surfs, xn)
            sn[:, s] = 0
            changed = (sn != sign[idx]).any(axis=1)
            for j in np.flatnonzero(out | changed):
                i = idx[j]
                if out[j]:
                    f = int(np.argmin(dom[j]))
                    if prev_key[i] is not None:
                        exits.append((prev_key[i], f, xn[j].copy()))
                    alive[i] = False
                    stats["left domain"] += 1
                    continue
                if (sn[j] != sign[i]).sum() > 1:
                    # several crossings in one step: walk it again in small sub-steps
                    sub = _project(surf, x[i][None] + np.linspace(0, 1, 17)[1:, None]
                                   * (xn[j] - x[i])[None])
                    ss = _signs(surfs, sub)
                    ss[:, s] = 0
                    prev_pts = np.vstack([x[i][None], sub[:-1]])
                    seq = [(r, (p0 + p1) / 2) for r, p0, p1 in zip(ss, prev_pts, sub)
                           if (r != sign[i]).any()]
                else:
                    seq = [(sn[j], (x[i] + xn[j]) / 2)]
                for row, where in seq:
                    nflip = int((row != sign[i]).sum())
                    if nflip == 0:
                        continue
                    a, b = pair_at(keys, row, s)
                    k = key(a, b, s) if a is not None and b is not None else None
                    if k is not None and k not in observed:
                        observed[k] = where.copy()
                    if k is not None and prev_key[i] is not None and k != prev_key[i]:
                        rec = links.setdefault((min(prev_key[i], k), max(prev_key[i], k)), [])
                        if len(rec) < 4:
                            rec.append((tuple(int(t) for t in np.flatnonzero(row != sign[i])),
                                        where.copy()))
                        stats["crossings" if nflip == 1 else "crossings at a meeting curve"] += 1
                    elif k is None:
                        stats["unseen pieces"] += 1
                    prev_key[i] = k
                    sign[i] = row
            keep = ~out
            x[idx[keep]] = xn[keep]
            u[idx[keep]] = un[keep]
        stats["walks"] += len(x)
        stats["still walking at the end"] += int(alive.sum())
    return links, exits, observed, dict(stats)


def _face_surfaces(surfs) -> dict[int, int]:
    """Library surfaces that are domain-face planes: surface index -> face index."""
    out = {}
    for s in surfs:
        if s.kind != "plane":
            continue
        for f, n in enumerate(FACE_N):
            if abs(abs(float(s.normal @ n)) / np.linalg.norm(n) - 1) < 1e-9:
                out[s.index] = f
    return out


def glue_exits(surfs, keys: dict, exits):
    """Patches/seams in neighbouring copies that the walks continue into across a face.

    Returns (glues, face_continuations, stats, ended). ``ended`` are walks whose surface carries
    no library surface in the next copy (the walk's patch ends at that face). A face continuation is a walk that continues, in
    the neighbouring copy, in a face of the domain (a wall lying in the face there):
    (key, copy, beta there, face index).
    """
    from .frame import surface_scale, surface_values

    faces = _face_surfaces(surfs)
    face_cont = []
    ended = []           # walks whose surface carries nothing in the next copy: they end at the face
    out = set()
    stats: dict = defaultdict(int)
    for lo in range(0, len(exits), 4000):
        chunk = exits[lo:lo + 4000]
        xs = np.array([xn / xn.sum() for _, _, xn in chunk])
        copies, betas = locate(point_from_beta(xs), tol=1e-13)
        vals = surface_values(betas, surfs)
        dist = np.abs(vals) / (surface_scale(betas, surfs) + 1e-300)
        for (k, f_exit, xn), copy, row, d, beta in zip(chunk, copies, vals, dist, betas):
            if copy == 0:
                stats["stayed in copy 0"] += 1
                continue
            s2 = int(np.argmin(d))
            if d[s2] > 1e-8:
                stats["not on a library surface there"] += 1
                ended.append((k, int(f_exit), xn.copy()))
                continue
            if s2 in faces:
                face_cont.append((k, int(copy), beta.copy(), faces[s2]))
                stats["continues in a domain face"] += 1
                continue
            sign = np.sign(row).astype(np.int8)
            sign[s2] = 0
            a, b = pair_at(keys, sign, s2)
            if a is None or b is None:
                stats["unseen pieces"] += 1
                continue
            out.add((k, int(copy), (min(a, b), max(a, b), s2)))
            stats["glued"] += 1
    return sorted(out), face_cont, dict(stats), ended



def march_faces(surfs, keys: dict, starts: list[tuple[int, np.ndarray]], canon_of, step: float = 2e-3,
                max_steps: int = 400, rng: np.random.Generator | None = None):
    """Walk along the domain's faces, where walls lying in a face live.

    In a face, the two sides of a wall are a piece of this copy and a piece of the neighbouring
    copy across the face. ``canon_of(a, q, shift)`` gives (instance copy, face patch key) or None:
    the copy, relative to the walk's own, in which the stored patch has its side A. Walking
    past an edge of the domain, the walk is carried into the copy beyond, where it lies in that
    copy's face (a face patch) or on one of its surfaces (an interior patch).

    Returns links as ((kind, key), copy, (kind, key), copy) with kind "face" or "inner".
    """
    from .frame import multiplication_table, surface_values

    rng = rng or np.random.default_rng(1)
    mult = multiplication_table()
    faces = _face_surfaces(surfs)
    links = set()
    stats: dict = defaultdict(int)

    def sides(x, f):
        """Face patch key (instance copy, key) at points x lying in face f of copy 0."""
        n = FACE_N[f] / np.linalg.norm(FACE_N[f])
        sa = np.sign(surface_values(x + 1e-9 * n, surfs)).astype(np.int8)
        k, beyond = locate(point_from_beta(x - 1e-9 * n), tol=1e-13)
        sb = np.sign(surface_values(beyond, surfs)).astype(np.int8)
        out = []
        for ra, kk, rb in zip(sa, k, sb):
            a, q = keys.get(ra.tobytes()), keys.get(rb.tobytes())
            if a is None or q is None or kk == 0:
                out.append(None)
                continue
            out.append(canon_of(a, q, int(kk)))
        return out

    by_face = defaultdict(list)
    for f, x in starts:
        by_face[f].append(x)
    for f, pts in by_face.items():
        n = FACE_N[f] - FACE_N[f].mean()
        n = n / np.linalg.norm(n)
        x = np.repeat(np.asarray(pts, float), 2, axis=0)
        x = x - np.outer(x @ FACE_N[f] / (n @ FACE_N[f]), n)
        u = rng.normal(size=x.shape)
        u -= u.mean(axis=1, keepdims=True)
        u -= np.outer(u @ n, n)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        u[1::2] = -u[0::2]
        cur = sides(x, f)
        alive = np.ones(len(x), bool)
        for _ in range(max_steps):
            idx = np.flatnonzero(alive)
            if not len(idx):
                break
            xn = x[idx] + step * u[idx]
            dom = xn @ FACE_N.T
            dom[:, f] = 0.0
            out = (dom < 0).any(axis=1)
            inside = idx[~out]
            if len(inside):
                new = sides(xn[~out], f)
                for i, k in zip(inside, new):
                    if k != cur[i]:
                        if k is not None and cur[i] is not None:
                            links.add((("face", cur[i][1]), cur[i][0], ("face", k[1]), k[0]))
                            stats["face crossings"] += 1
                        cur[i] = k
                x[inside] = xn[~out]
            for j in np.flatnonzero(out):
                i = idx[j]
                alive[i] = False
                stats["left the face"] += 1
                if cur[i] is None:
                    continue
                copy, beta = locate(point_from_beta(xn[j][None] / xn[j].sum()), tol=1e-13)
                copy, beta = int(copy[0]), beta[0]
                vals = surface_values(beta[None], surfs)[0]
                # in the copy beyond: in one of its faces, or on one of its surfaces?
                fd = beta @ FACE_N.T / np.linalg.norm(FACE_N, axis=1)
                f2 = int(np.argmin(np.abs(fd)))
                if abs(fd[f2]) < 1e-9:
                    k2 = sides(beta[None], f2)[0]
                    if k2 is not None:
                        links.add((("face", cur[i][1]), cur[i][0], ("face", k2[1]),
                                   int(mult[copy, k2[0]])))
                        stats["into a face beyond the edge"] += 1
                    continue
                s2 = int(np.argmin(np.abs(vals)))
                if abs(vals[s2]) > 1e-8 or s2 in faces:
                    stats["beyond the edge: no wall surface"] += 1
                    continue
                sign = np.sign(vals).astype(np.int8)
                sign[s2] = 0
                a, b = pair_at(keys, sign, s2)
                if a is None or b is None:
                    stats["unseen pieces"] += 1
                    continue
                links.add((("face", cur[i][1]), cur[i][0], ("inner", (min(a, b), max(a, b), s2)),
                           copy))
                stats["into a surface beyond the edge"] += 1
    return links, dict(stats)
