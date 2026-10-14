"""Find transitional (T1/T2) lines in the golden survey: collinear runs verified by midpoints."""
import json
import sys
from multiprocessing import Pool

import numpy as np
from catalog import match
from cellframe import seed_from_beta
from classify import classify, signature


def to_upper(b):
    b = np.asarray(b, float); b = b / b.sum()
    return b[[1, 0, 3, 2]] if b[2] < b[3] - 1e-12 else b

def candidate_runs(samples, t):
    B = np.array([to_upper(s["beta"]) for s in samples]); isT = np.array([match(s["sig"]) == t for s in samples])
    TB = []
    for b in B[isT]:
        if not any(np.allclose(b, c, atol=1e-12) for c in TB): TB.append(b)
    TB = np.array(TB); runs, used = [], set()
    for i in range(len(TB)):
        for j in range(i + 1, len(TB)):
            if (i, j) in used: continue
            d = TB[j] - TB[i]; d = d / np.linalg.norm(d)
            relT = TB - TB[i]; offT = np.linalg.norm(relT - np.outer(relT @ d, d), axis=1)
            idx = [int(k) for k in np.argsort(relT @ d) if offT[k] < 1e-9]
            for a in idx:
                for b in idx: used.add((min(a, b), max(a, b)))
            if len(idx) < 3: continue
            # all samples on the line, in order: a gap is bad if a non-T sample lies between two T samples
            rel = B - TB[i]; off = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
            online = [(float(rel[k] @ d), bool(isT[k])) for k in range(len(B)) if off[k] < 1e-9]
            online.sort(); run = []
            for pos, t_ok in online:
                if t_ok: run.append(pos)
                else:
                    if len(run) >= 3: runs.append((TB[i], d, run))
                    run = []
            if len(run) >= 3: runs.append((TB[i], d, run))
    return runs

def mid_type(beta):
    try: return match(signature(classify(seed_from_beta(beta))))
    except Exception: return None

if __name__ == "__main__":
    samples = json.load(open(sys.argv[1])); out = []
    for t in ("T1", "T2"):
        runs = candidate_runs(samples, t)
        mids = {}
        for base, d, run in runs:
            for a, b in zip(run, run[1:]):
                m = base + d * (a + b) / 2; mids[tuple(np.round(m, 12))] = m
        print(t, "candidate runs:", len(runs), "midpoints to check:", len(mids), flush=True)
        if len(sys.argv) > 2 and sys.argv[2] == "count": continue
        keys = list(mids)
        with Pool(4) as pool: types = dict(zip(keys, pool.map(mid_type, [mids[k] for k in keys])))
        for base, d, run in runs:
            seg = [run[0]]
            for a, b in zip(run, run[1:]):
                if types[tuple(np.round(base + d * (a + b) / 2, 12))] == t: seg.append(b)
                else:
                    if len(seg) >= 2: out.append({"id": t, "beta": [(base + d * seg[0]).tolist(), (base + d * seg[-1]).tolist()]})
                    seg = [b]
            if len(seg) >= 2: out.append({"id": t, "beta": [(base + d * seg[0]).tolist(), (base + d * seg[-1]).tolist()]})
        print(t, "verified segments:", sum(1 for o in out if o["id"] == t), flush=True)
    if not (len(sys.argv) > 2 and sys.argv[2] == "count"): json.dump(out, open("tlines.json", "w"))
