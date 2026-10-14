from multiprocessing import Pool

import numpy as np
from axes import name
from classify import classify, signature
from survey import seed_from_q


def sig(r, s):
    try: return name(signature(classify(seed_from_q(s * r))))
    except Exception: return "ERR"
def find(args):
    label, r, lo_deg, hi_deg = args
    r = np.array(r) / np.linalg.norm(r); lo, hi = np.tan(np.radians(lo_deg)), np.tan(np.radians(hi_deg))
    slo, shi = sig(r, lo), sig(r, hi); seen = []
    for _ in range(45):
        mid = (lo + hi) / 2; sm = sig(r, mid)
        if sm == slo: lo = mid
        elif sm == shi: hi = mid
        else: seen.append(sm); hi = mid          # a third type appeared: keep narrowing towards it
    t = (lo + hi) / 2
    return label, np.degrees(np.arctan(t)), slo, shi, sig(r, t), seen[-1:] , (t * r).tolist()
jobs = [
    ("W4->W5", [-0.725, 0.0, -0.689], 16.82, 17.00),
    ("W4->U6", [-0.94, 0.0, -0.341], 14.35, 14.54),
    ("W1->U17", [0.545, 0.0, 0.839], 18.82, 19.02),
    ("W4->W5 (b)", [-0.545, 0.0, -0.839], 16.00, 16.20),
    ("W5->U23", [-0.545, 0.0, -0.839], 19.81, 20.01),
    ("U20->U25", [1.0, 0.0, 0.0], 13.11, 13.32),
    ("U20->U27", [-1.0, 0.0, 0.0], 10.81, 11.02),
    ("3fold 13.07", [-0.577, 0.795, 0.188], 13.04, 13.23),
]
with Pool(4) as pool:
    for label, deg, slo, shi, at, seen, q in pool.map(find, jobs):
        extra = f" | third type seen while narrowing: {seen[0]}" if seen else ""
        print(f"{label:<12} at {deg:.5f}°: {slo} -> {shi}; at the point: {at}{extra}")
