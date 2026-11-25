"""The closed map's faces joined into the pieces the page needs, as the swirlprism domain was built: a curve is an edge
only where the shape changes across it.

a4_closed cuts every surface along its meeting curve with every other surface, so its faces lie each on one side of
all of them. That is more than a wall piece needs: a surface that changes nothing where it crosses a face is no edge
of it (and near the hubs, where all the surfaces meet all but tangent, those crossings are where the curve graph is
hardest to get right). Here, on each surface, two faces sharing a segment are one piece when they are the same shape;
a piece's edges are then the segments with a different shape (or the chamber's outside) across them.

Checks, per piece:
  (a) one shape all through: every probe of every face in it reads the same shape (probes that read no shape at all,
      next to a chamber edge where the hull test fails, are left out but counted);
  (b) the swap's image of the piece is the same shape;
  (c) every edge exact (on all its surfaces to 1e-7) and the edges closing into loops (no loose ends).

    python a4_merge.py kidecaic
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import a4_pieces as P  # noqa: E402
import a4_snub as SN  # noqa: E402

REL = lambda S_, x: abs(S_.f(x)) / (np.linalg.norm(S_.g(x)) + 1e-300)  # noqa: E731


def merge(X, surfs, sub, log=print):
    segs, faces = X["segments"], X["faces"]
    kn = X.get("keyname", {})

    # (a probe within 1e-4 of a chamber wall other than the face's own reads the shape unreliably: there vertices of
    #  the hull all but coincide. Such probes are left out of a face's shape when it has others)
    for F in faces:
        k = F["surface"]
        good = []
        for x, kk in zip(F["probes"], F["keys"]):
            x = np.asarray(x, float)
            near = min(x[j] for j in range(4) if j != k) < 1e-4
            good.append(kk if not near else "near")
        if any(g not in ("near", None) for g in good):
            F["vote"] = [g for g in good if g != "near"]
        else:
            F["vote"] = list(F["keys"])

    # (a face's shape: the most common of its probes' shapes, probes reading none left out)
    def key_of(F):
        ks = [k for k in F["vote"] if k is not None]
        return Counter(ks).most_common(1)[0][0] if ks else None
    fkey = [key_of(F) for F in faces]

    # (the faces on each side of each segment, per surface)
    side = {}
    for fi, F in enumerate(faces):
        for i, fw in F["cycle"]:
            side[(F["surface"], i, fw)] = fi
    par = list(range(len(faces)))

    def root(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for (k, i, fw), fi in side.items():
        fj = side.get((k, i, not fw))
        if fj is not None and fj != fi and fkey[fi] is not None and fkey[fi] == fkey[fj]:
            a, b = root(fi), root(fj)
            if a != b:
                par[max(a, b)] = min(a, b)
    pieces = defaultdict(list)
    for fi in range(len(faces)):
        pieces[root(fi)].append(fi)

    # (a piece's edges: its faces' segments with another shape, or nothing, across them)
    out = []
    for r, members in pieces.items():
        k = faces[r]["surface"]
        edges = []
        for fi in members:
            for i, fw in faces[fi]["cycle"]:
                fj = side.get((k, i, not fw))
                if fj is None or root(fj) != r:
                    edges.append((i, fw))
        keys = [kk for fi in members for kk in faces[fi]["vote"]]
        shapes = Counter(kk for kk in keys if kk is not None)
        out.append({"surface": k, "faces": members, "edges": edges, "key": fkey[r] if fkey[r] else
                    (shapes.most_common(1)[0][0] if shapes else None),
                    "shapes": dict(shapes), "unread": sum(kk is None for kk in keys)})

    # (a) one shape all through
    mixed = [p for p in out if len(p["shapes"]) > 1]
    unread = [p for p in out if not p["shapes"]]
    # (b) the swap's image: each piece's probes reversed read the same shape
    pr, owner = [], []
    for pi, p in enumerate(out):
        for fi in p["faces"][:3]:
            for x in faces[fi]["probes"][:2]:
                pr.append(np.asarray(x, float)[::-1])
                owner.append(pi)
    labs = SN.classify_many(pr, sub, tol=1e-7)
    swap_keys = defaultdict(Counter)
    for pi, (_, kk) in zip(owner, labs):
        if kk is not None:
            swap_keys[pi][kk] += 1
    bad_swap = [pi for pi, p in enumerate(out) if p["key"] is not None and swap_keys[pi]
                and swap_keys[pi].most_common(1)[0][0] != p["key"]]
    # (c) the edges: loose ends per surface, exactness
    loose = 0
    for k in range(len(surfs)):
        deg = Counter()
        for p in out:
            if p["surface"] != k:
                continue
            for i, fw in p["edges"]:
                a, b = segs[i]["ends"]
                deg[a] += 1
                deg[b] += 1
        loose += sum(1 for d in deg.values() if d % 2)
    edge_segs = {i for p in out for i, _ in p["edges"]}
    off = [i for i in edge_segs
           if max(REL(surfs[k], x) for x in np.asarray(segs[i]["pts"], float) for k in segs[i]["on"]) > 1e-7]
    report = {
        "faces": len(faces),
        "pieces": len(out),
        "pieces whose probes read more than one shape": len(mixed),
        "pieces no probe of which reads a shape": len(unread),
        "probes reading no shape": sum(p["unread"] for p in out),
        "pieces whose swap image is another shape": len(bad_swap),
        "edge segments": len(edge_segs),
        "edge points of odd degree (loose ends)": loose,
        "edge segments off their surfaces by more than 1e-7": len(off),
        "pieces by shape": dict(Counter(kn.get(p["key"], "other" if p["key"] else "none") for p in out)),
    }
    for p in mixed:
        p["mixed"] = True
    for pi in bad_swap:
        out[pi]["bad swap"] = True
    return out, report


# ------------------------------------------------------------------------------------------------ repair
def _mirrors():
    import a4_exact as E
    return [E.Surf(E.B[i], f"b{i + 1}") for i in range(4)]


def separating(X, surfs, F):
    """A surface not on the face's outline whose sign parts the face's probes by shape (one shape each side): the
    curve where it meets the face's surface is missing inside the face. (surface, probe of one shape, probe of the
    other), or None."""
    segs = X["segments"]
    k = F["surface"]
    pts = [(np.asarray(x, float), kk) for x, kk in zip(F["probes"], F["keys"]) if kk is not None]
    ks = {kk for _, kk in pts}
    if len(ks) < 2:
        return None
    on_outline = {m for i, _ in F["cycle"] for m in segs[i]["on"]}
    for m, C in enumerate(surfs):
        if m == k or m in on_outline:
            continue
        sg = defaultdict(set)
        sp = {}
        for x, kk in pts:
            r = C.f(x) / (np.linalg.norm(C.g(x)) + 1e-300)
            if abs(r) > 1e-9:
                sg[kk].add(int(np.sign(r)))
                sp.setdefault(int(np.sign(r)), x)
        if all(len(v) == 1 for v in sg.values()) and len({next(iter(v)) for v in sg.values()}) == 2:
            return m, sp[1], sp[-1]
    return None


def seed_between(S, k, C, a, b):
    """The point where C changes sign between a and b on surface k, solved on both (None if it cannot be)."""
    import a4_exact as E
    fa = C.f(a)
    for _ in range(60):
        mid = (a + b) / 2
        mid = P.lift_normal(S, mid) if k >= 4 else mid
        if mid is None:
            return None
        fm = C.f(mid)
        if (fm > 0) == (fa > 0):
            a, fa = mid, fm
        else:
            b = mid
        if np.linalg.norm(b - a) < 1e-12:
            break
    z, ok = E.newton((a + b) / 2, [S, C])
    return z if ok else None


def add_curve(X, surfs, k, m, seed, log=print):
    """The curve where surfaces k and m meet, traced through seed, added to the map: cut where third surfaces cross
    it (segments_of), its points joined to points of the map already there (3e-5), and every segment of the map cut at
    its new points. The surfaces whose segments changed are returned."""
    import a4_closed as Cl
    import a4_exact as E
    from scipy.spatial import cKDTree
    pts, closed = E.trace(seed, surfs[k], surfs[m], _mirrors())
    if len(pts) < 2:
        return set()
    npts, nsegs = Cl.segments_of(surfs, [{"pair": (k, m), "pts": np.array(pts), "closed": closed}], lambda *a: None)
    Pts, segs = X["points"], X["segments"]
    live = sorted({e for s_ in segs for e in s_["ends"]})
    tree = cKDTree(np.array([Pts[p] for p in live]))
    remap = {}
    for j, x in enumerate(npts):
        d, i = tree.query(x)
        if d < 3e-5:
            remap[j] = live[i]
        else:
            Pts.append(np.asarray(x, float))
            remap[j] = len(Pts) - 1
    touched = set()
    new_ids = set(remap.values())
    for s_ in nsegs:
        s_["ends"] = (remap[s_["ends"][0]], remap[s_["ends"][1]])
        if s_["ends"][0] == s_["ends"][1] and np.linalg.norm(np.diff(s_["pts"], axis=0), axis=1).sum() < 1e-3:
            continue
        Q = np.asarray(s_["pts"], float)
        Q[0], Q[-1] = Pts[s_["ends"][0]], Pts[s_["ends"][1]]
        s_["pts"] = Q
        segs.append(s_)
        touched |= set(s_["on"])
    # (existing segments cut at the new points lying on them; the first part keeps the segment's place, so faces of
    #  surfaces not touched keep their indices)
    for i in range(len(segs)):
        t = segs[i]
        Q = np.asarray(t["pts"], float)
        for p in new_ids:
            if p in t["ends"]:
                continue
            x = Pts[p]
            if min(np.linalg.norm(x - Q[0]), np.linalg.norm(x - Q[-1])) < 1e-5:
                continue
            A_, d_ = Q[:-1], Q[1:] - Q[:-1]
            u = np.clip(np.einsum("ij,ij->i", x - A_, d_) / (np.einsum("ij,ij->i", d_, d_) + 1e-300), 0, 1)
            dist = np.linalg.norm(A_ + u[:, None] * d_ - x, axis=1)
            j = int(np.argmin(dist))
            # (the point is exact, solved on the segment's surfaces; the segment's points are a chain along the
            #  curve, which can sag from it by more than 1e-6 between points)
            if dist[j] < 2e-4 and all(REL(surfs[q], x) < 1e-8 for q in t["on"]):
                segs[i] = dict(t, ends=(t["ends"][0], p), pts=np.vstack([Q[:j + 1], x]))
                segs.append(dict(t, ends=(p, t["ends"][1]), pts=np.vstack([x, Q[j + 1:]])))
                touched |= set(t["on"])
                t, Q = segs[i], np.asarray(segs[i]["pts"], float)
    log(f"    curve {k}x{m}: {len(nsegs)} segments, {len(new_ids)} points; surfaces re-walked: {sorted(touched)}")
    return touched


def rewalk(X, surfs, sub, ks):
    """The faces of surfaces ks walked, probed and classified again."""
    import a4_closed as Cl
    import a4_faces as A
    segs, Pts = X["segments"], X["points"]
    keep = [F for F in X["faces"] if F["surface"] not in ks]
    new = []
    for k in sorted(ks):
        S = surfs[k]
        for loops, hc in A.faces_on(k, S, segs, Pts):
            cp = Cl.probes_of(S, k, loops[0], hc[0], segs, frac=0.02)
            pr = Cl.probes_of(S, k, loops[0], hc[0], segs, minlen=2e-3) or cp
            if pr:
                new.append({"surface": k, "loop": loops[0], "cycle": hc[0], "probes": pr, "cut probes": cp or pr})
    labs = SN.classify_many([p for F in new for p in F["probes"]], sub, tol=1e-7)
    i = 0
    for F in new:
        F["keys"] = [labs[i + j][1] for j in range(len(F["probes"]))]
        i += len(F["probes"])
        F["key"] = Counter(F["keys"]).most_common(1)[0][0]
    X["faces"] = keep + new


def repair(X, surfs, sub, log=print, rounds=4):
    """Two-shape faces with a surface parting their shapes: the missing curve added, until none is left."""
    import a4_faces as A
    A.SURFS = surfs
    for rnd in range(rounds):
        pieces, report = merge(X, surfs, sub)
        todo = []
        for p in pieces:
            if not p.get("mixed"):
                continue
            for fi in p["faces"]:
                s = separating(X, surfs, X["faces"][fi])
                if s:
                    todo.append((X["faces"][fi]["surface"], s))
        log(f"  round {rnd}: {report['pieces whose probes read more than one shape']} mixed pieces; "
            f"{len(todo)} faces with a surface parting their shapes")
        if not todo:
            return pieces, report
        touched = set()
        done = set()
        for k, (m, a, b) in todo:
            if (min(k, m), max(k, m), tuple(np.round(a, 3))) in done:
                continue
            seed = seed_between(surfs[k], k, surfs[m], a, b)
            if seed is None:
                log(f"    {k}x{m}: no seed")
                continue
            done.add((min(k, m), max(k, m), tuple(np.round(a, 3))))
            touched |= add_curve(X, surfs, k, m, seed, log)
        rewalk(X, surfs, sub, touched)
    return merge(X, surfs, sub)


if __name__ == "__main__":
    for g in sys.argv[1:]:
        X = pickle.load(open(os.path.join(HERE, f"a4_closed_{g}.pkl"), "rb"))
        surfs = [P.surf_of(n) for n in X["surfaces"]]
        import a4_exact as E
        E.SING = 1e-8
        pieces, report = repair(X, surfs, SN.SUB[g])
        pickle.dump(X, open(os.path.join(HERE, f"a4_repaired_{g}.pkl"), "wb"))
        pickle.dump(pieces, open(os.path.join(HERE, f"a4_merged_{g}.pkl"), "wb"))
        json.dump(report, open(os.path.join(HERE, f"a4_merged_{g}.json"), "w"), indent=1, default=str)
        print(json.dumps(report, indent=1, default=str))
