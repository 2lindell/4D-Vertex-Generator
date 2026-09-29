import json
from multiprocessing import Pool

import numpy as np
from axes import name
from classify import classify, signature
from survey import seed_from_q

rays = json.load(open("axes.json"))
def sig_at(r, s, tol=1e-5):
    try: return name(signature(classify(seed_from_q(s * np.array(r)), hull_tol=tol)))
    except Exception: return "ERR"
def bisect(args):
    kind, r, lo, hi = args
    slo, shi = sig_at(r, lo, 1e-9), sig_at(r, hi, 1e-9)
    for _ in range(40):
        mid = (lo + hi) / 2; sm = sig_at(r, mid, 1e-9)
        if sm == slo: lo = mid
        elif sm == shi: hi = mid
        else: break
    t = (lo + hi) / 2
    return kind, r, t, slo, shi, sig_at(r, t, 1e-5), sig_at(r, t, 1e-4)
jobs = []
for ray in rays:
    if ray["dir"][2] == 0.0 or abs(ray["dir"][2]) < 1e-9: continue           # cross rings: already mapped
    runs = [x for x in ray["runs"] if x[2] != "ERR" and "240" not in x[2][:8]]
    for (a0, b0, t0, _), (a1, b1, t1, _) in zip(runs, runs[1:]):
        if t0 != t1 and b0 < a1: jobs.append((ray["kind"], ray["dir"], b0, a1))
with Pool(4) as pool: res = pool.map(bisect, jobs)
for kind, r, t, slo, shi, at5, at4 in res:
    print(f"{kind} ray {np.round(r, 3).tolist()} at {np.degrees(np.arctan(t)):.4f}°: {slo} -> {shi}\n   at the point: {at5}\n   (looser merge: {at4})")
