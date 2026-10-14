"""A denser search for regions the golden grid missed: random seeds spread through the half-cell, biased toward
its faces and edges, and clustered at its corners (where the grid is sparsest). Every seed is classified; each
new shape is nudged 16 ways to tell regions from lower-dimensional loci. Writes extra_samples.json (one sample
per new shape, in the golden-sample format) and dense_search.json (all results).

    python dense_search.py [n_per_kind]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from cell_atlas import to_upper
from probe import _one
from xloci import directions


def seeds(n):
    rng = np.random.default_rng(11)
    out = list(rng.dirichlet(np.ones(4), n))                        # even in the barycentric simplex
    out += list(rng.dirichlet(np.full(4, 0.35), n))                 # toward faces, edges and corners
    corners = [np.eye(4)[0], np.eye(4)[1], np.eye(4)[2], np.array([0, 0, 0.5, 0.5])]
    for k in range(n):                                              # clustered at the half-cell's corners
        c = corners[k % 4]
        eps = rng.uniform(0.002, 0.25)
        out.append((1 - eps) * c + eps * rng.dirichlet(np.ones(4)))
    return [to_upper(b / b.sum()) for b in out]


def _nudged(job):
    b, lab = job
    D = directions(5)[:8]
    return sum(_one(b + s * 1e-5 * d)[0] == lab for d in D for s in (1, -1))


def main(n=800):
    from cell_atlas2 import identify, load_refs
    refs = load_refs()[0]
    known = set(json.load(open("atlas_xids.json")).values())
    B = seeds(n)
    with Pool(4) as pool:
        res = pool.map(_one, B, chunksize=8)
        new = {}
        for b, (lab, sig) in zip(B, res):
            if lab.startswith("new:"):
                new.setdefault(sig, []).append(b)
        kept = pool.map(_nudged, [(bs[0], "new:" + sig) for sig, bs in new.items()])
    from collections import Counter
    counts = Counter(lab for lab, _ in res)
    print(len(B), "seeds;", sum(v for k, v in counts.items() if not k.startswith("new:")), "in known shapes;",
          sum(v for k, v in counts.items() if k.startswith("new:")), "in", len(new), "new shapes")
    extra = []
    for (sig, bs), k in zip(new.items(), kept):
        _, xkey, _ = identify(sig, refs)
        kind = "region" if k == 16 else ("lower-dimensional" if k == 0 else f"unclear ({k}/16 kept)")
        print(f"  {len(bs):3d} hits, {kind}: {sig.split(' | val')[0][6:]} | val {sig.split(' | val')[-1]}  e.g. β {np.round(bs[0], 4).tolist()}")
        extra.append({"beta": bs[0].tolist(), "golden": False, "sig": sig, "kind": kind, "hits": len(bs)})
    print("known shapes never hit:", sorted(known - set(counts)))
    json.dump(extra, open("extra_samples.json", "w"), indent=1)
    json.dump([[b.tolist(), lab] for b, (lab, _) in zip(B, res)], open("dense_search.json", "w"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 800)
