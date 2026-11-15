"""The ixoic half-turns' fixed lines traced whole (fixed_circles.json, fixed_circles.md).

fixed_lines.py sampled each line piece by piece, one sheet's chamber at a time (120 points each). Here each line is
taken whole: a half-turn g fixes a plane of seed vectors x (g x = x), a circle of seeds; the central inversion is in
ixoic, so x and -x give the same polychoron and half the circle is the whole line. The half-turns fall into two
conjugacy classes (180 and 90 of them), so two circles cover every half-turn line of every sheet. Each circle is
sampled at N angles (labelled at hull tolerance 1e-7, the seed beta = X^-1 x taken with whichever sign sums positive),
every change of key is bisected to 1e-12 in angle, and a key met only at a bisection, different from both sides, is
a point. The keys are then compared with what the atlas recorded (fixed_lines.json, ixoic_shapes.json, the closure
checks): any key not there was missed.

    HYIC_SUB=ixo0 python fixed_circles.py [N]      (N = 2000)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import group as G  # noqa: E402

CIRC = None


def lab(th):
    u, v = CIRC
    x = np.cos(th) * u + np.sin(th) * v
    b = np.linalg.solve(G.X, x)
    if abs(b.sum()) < 1e-9:
        return None
    b = b / b.sum()
    for t in (1e-7, 1e-8):
        try:
            k = G.label(b, t)[1]
            if k:
                return k
        except Exception:
            pass
    return None


def init(c):
    global CIRC
    CIRC = c


def bisect(item):
    a, b, ka, kb = item
    mids = []
    for _ in range(45):
        m = (a + b) / 2
        km = lab(m)
        if km == ka:
            a = m
        elif km == kb:
            b = m
        else:                                   # (a third key between: a point or a short stretch)
            mids.append((m, km))
            b = m
            kb = km
    m = (a + b) / 2
    return {"at": m, "left": ka, "right": kb, "between": [[x, k] for x, k in mids]}


def classes():
    E, I = G.ELEMENTS, np.eye(4)
    inv = [i for i, g in enumerate(E) if np.allclose(g @ g, I) and not np.allclose(g, I) and not np.allclose(g, -I)]
    idx = {tuple(np.round(g, 6).ravel()): i for i, g in enumerate(E)}
    left, out = set(inv), []
    while left:
        i = min(left)
        cl = {idx[tuple(np.round(h @ E[i] @ h.T, 6).ravel())] for h in E}
        left -= cl
        out.append((i, len(cl)))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    known = {s["key"] for s in json.load(open(os.path.join(HERE, "ixoic_shapes.json")))["shapes"]}
    recorded = {}
    for sub, o in json.load(open(os.path.join(HERE, "fixed_lines.json"))).items():
        for k, r in o["shapes"].items():
            recorded.setdefault(k, set()).add(r["kind"])
    for f in glob.glob(os.path.join(HERE, "closure_ixo*.json")):
        if not f.endswith(".partial.json"):
            known |= set(json.load(open(f))["new_regions"])
    out = {"sheet": G.SUB, "samples": n, "circles": []}
    md = ["# Ixoic half-turn lines traced whole", "", __doc__.strip().split("\n\n")[1], ""]
    for gi, size in classes():
        g = G.ELEMENTS[gi]
        _, s, Vt = np.linalg.svd(g - np.eye(4))
        u, v = Vt[2], Vt[3]
        th = np.linspace(0, np.pi, n, endpoint=False)
        ck = os.path.join(HERE, f"fixed_circles_{size}.partial.json")    # (the samples kept, so a rerun resumes)
        with Pool(int(os.environ.get("NPROC", "2")), initializer=init, initargs=((u, v),)) as P:
            if os.path.exists(ck) and json.load(open(ck))["n"] == n:
                keys = json.load(open(ck))["keys"]
            else:
                keys = P.map(lab, th, chunksize=8)
                json.dump({"n": n, "keys": keys}, open(ck, "w"))
            keys.append(keys[0])                # (the circle closes: pi is 0 again, x -> -x)
            th = np.append(th, np.pi)
            ch = [(th[i], th[i + 1], keys[i], keys[i + 1]) for i in range(n) if keys[i] != keys[i + 1]]
            B = P.map(bisect, ch, chunksize=1)
        stretches, cur = [], None
        for t, k in zip(th, keys):
            if not cur or cur["key"] != k:
                cur = {"key": k, "from": float(t), "samples": 0}
                stretches.append(cur)
            cur["samples"] += 1
        pts = {}
        for b in B:
            for x, k in b["between"]:
                pts.setdefault(k, x)
        found = ({s_["key"] for s_ in stretches} | set(pts)) - {None}
        miss = sorted(k for k in found if k not in recorded and k not in known)
        line_keys = {s_["key"] for s_ in stretches if s_["key"]}
        kind_wrong = sorted(k for k in found if k in recorded and
                            (("line" in recorded[k]) != (k in line_keys)))
        c = {"class_size": size, "element": int(gi), "plane": [u.tolist(), v.tolist()],
             "stretches": [{**s_, "key": s_["key"]} for s_ in stretches], "changes": len(ch),
             "points": {k: float(x) for k, x in pts.items()}, "keys": sorted(k for k in found if k),
             "not_recorded": miss, "kind_differs": kind_wrong}
        out["circles"].append(c)
        md += [f"## Half-turns of the class of {size}", "",
               f"{n} samples round the line ({sum(k is None for k in keys)} failed to label), {len(ch)} changes of key, {len(stretches)} stretches, {len(pts)} points "
               f"between stretches. Keys met: {len(c['keys'])}; recorded by fixed_lines.py or the atlas: "
               f"{len(c['keys']) - len(miss)}; not recorded: {len(miss)}" + (f" ({', '.join(miss)})" if miss else "")
               + (f". Kind differs from the record: {', '.join(kind_wrong)}" if kind_wrong else "") + ".", ""]
        print(f"class {size}: {len(stretches)} stretches, {len(pts)} points, {len(c['keys'])} keys, not recorded {miss}, "
              f"kind differs {kind_wrong}", flush=True)
        json.dump(out, open(os.path.join(HERE, "fixed_circles.json"), "w"), indent=1)
    open(os.path.join(HERE, "fixed_circles.md"), "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
