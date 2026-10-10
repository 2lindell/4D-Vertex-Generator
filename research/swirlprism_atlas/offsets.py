import json
import re
from multiprocessing import Pool

import numpy as np
from catalog import parse_signature
from classify import classify, signature
from mplanes import REPS
from survey import seed_from_q

NORMALS = {k: np.array(v) / np.linalg.norm(v) for k, v in REPS.items()}
NORMALS.update({"az0": np.array([0.0, 1.0, 0.0]), "az36": np.array([-np.sin(np.radians(36)), np.cos(np.radians(36)), 0.0]), "z0": np.array([0.0, 0.0, 1.0])})
d = json.loads(re.search(r'<script id="atlas-data" type="application/json">(.*?)</script>', open("atlas.html").read(), re.S).group(1))
key = {(t["cells"], t["faces"], t["edges"]): k for k, t in d["types"].items()}
def name(sig):
    p = parse_signature(sig)
    return "ERR" if p is None else key.get(tuple("+".join(map(str, x)) for x in p[1:]), "new")
rows = json.load(open("planes_0012.json")) + json.load(open("mirrors_0015.json"))
cands = [k for k, t in d["types"].items() if not t["volumeSamples"] and t["vertices"] == 1200]
jobs = []
for k in cands:
    mine = [r for r in rows if name(r["sig"]) == k]
    for r in mine[:: max(1, len(mine) // 3)][:3]:
        n = NORMALS[r["plane"]]
        for eps in (-0.003, -0.0005, 0.0005, 0.003):
            jobs.append((k, r["plane"], tuple(r["q"]), eps, tuple(np.array(r["q"]) + eps * n)))
def run(j):
    k, plane, q, eps, q2 = j
    try: return (k, plane, q, eps, name(signature(classify(seed_from_q(np.array(q2))))))
    except Exception: return (k, plane, q, eps, "ERR")
if __name__ == "__main__":
    with Pool(4) as pool: res = pool.map(run, jobs)
    from collections import defaultdict
    by = defaultdict(list)
    for k, plane, q, eps, t in res: by[(k, plane, q)].append((eps, t))
    for (k, plane, q), v in sorted(by.items()):
        v.sort(); print(f"{k:<4} on {plane:<16} offsets -> " + "  ".join(f"{e:+.4f}:{t}" for e, t in v))
