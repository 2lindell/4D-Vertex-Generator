"""What do the X shapes still listed as points really fill? Every test learned so far, at each of their samples:

  * region: the shape in (nearly) every random direction, at 1e-5 and 1e-3;
  * wall: chords through the sample whose two ends differ are bisected; boundary points that read as the shape (loose
    classifier) mean the shape fills the wall there;
  * lines: the directions toward the shape's own sample images and the cell's corners and centre along which the shape
    goes on just past the sample (as for the stray samples, stray_lines.py);
  * the neighbours seen around it.

    python xpoint_survey.py out.json X15 X18 ...        (samples from the atlas page data, page path in $ATLAS_PAGE)
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

from fexact import _lab, _lab_loose
from stray_lines import images

PHI = (1 + 5 ** 0.5) / 2
U = np.linalg.svd(np.ones((1, 4)))[2][1:]
rng = np.random.default_rng(7)
DIRS = [U.T @ v / np.linalg.norm(v) for v in rng.normal(size=(24, 3))]
EXTRA = [np.eye(4)[k] for k in range(4)] + [np.array(v, float) / sum(v) for v in
                                            ([1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1])]


def golden_value(t):
    t = t.strip()
    t = re.sub(r"(\d)φ", r"\1*p", t).replace("φ", "p")
    return float(eval(t, {"p": PHI}))


def parse_beta(text):
    inner = text[text.index("(") + 1:text.rindex(")")]
    parts, depth, cur = [], 0, ""
    for ch in inner:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            depth += ch == "("
            depth -= ch == ")"
            cur += ch
    parts.append(cur)
    b = np.array([golden_value(x) for x in parts])
    return b / b.sum()


def samples_of(page, tids):
    s = open(page).read()
    i = s.find('<script id="atlas-data"')
    i = s.find(">", i) + 1
    d = json.loads(s[i:s.find("</script>", i)])
    out = {}
    for t in tids:
        out[t] = [parse_beta(x[3].split("<br>")[0].split("∝")[1]) for x in d["samples"].get(t, [])]
    return out


def lab(x):
    x = np.asarray(x, float)
    if x.min() < -1e-12 or x[2] < x[3] - 1e-12:
        return "OUT"
    return _lab(x / x.sum())


def _chord(args):
    a, b = (np.asarray(v) for v in args)
    la, lb = lab(a), lab(b)
    if la == lb or "OUT" in (la, lb):
        return la, lb, None
    lo, hi = 0.0, 1.0
    for _ in range(36):
        m = (lo + hi) / 2
        if lab(a + m * (b - a)) == la:
            lo = m
        else:
            hi = m
    x = a + (lo + hi) / 2 * (b - a)
    return la, lb, _lab_loose(x / x.sum())


def short(l):
    return l if not l.startswith("new:") else "new:" + l[4:44]


def main(out, tids, page):
    S = samples_of(page, tids)
    res = {}
    with Pool(4) as pool:
        for tid in tids:
            res[tid] = []
            imgs = []
            for b in S[tid]:
                for v in images(b):
                    if not any(np.abs(v - w).max() < 1e-9 for w in imgs):
                        imgs.append(v)
            for q in S[tid][:3]:
                r = {"beta": q.tolist(), "at": lab(q)}
                for rad in (1e-5, 1e-3):
                    r[f"around {rad:g}"] = Counter(pool.map(lab, [q + rad * d for d in DIRS])).most_common()
                chords = []
                for d in DIRS[:12]:
                    e = rng.normal(size=3)
                    off = U.T @ e / np.linalg.norm(e) * 3e-4
                    chords.append(((q + off - 1e-3 * d).tolist(), (q + off + 1e-3 * d).tolist()))
                walls = pool.map(_chord, chords)
                r["wall points"] = Counter(w[2] for w in walls if w[2]).most_common()
                dirs = []
                for c in imgs + EXTRA:
                    d = c - q
                    if np.linalg.norm(d) < 1e-9:
                        continue
                    d = d / np.linalg.norm(d)
                    for sgn in (1, -1):
                        if not any(np.abs(sgn * d - e).max() < 1e-6 for e in dirs):
                            dirs.append(sgn * d)
                labs = pool.map(lab, [q + 1e-4 * d for d in dirs])
                r["line directions"] = [d.tolist() for d, l in zip(dirs, labs) if l == tid]
                n_reg = sum(n for l, n in r["around 0.001"] if l == tid)
                n_wall = sum(n for l, n in r["wall points"] if l == tid)
                r["verdict"] = ("region" if n_reg >= 20 else "wall" if n_wall >= 2 else
                                f"{len(r['line directions']) // 2} line(s)" if r["line directions"] else "point (no extent found)")
                res[tid].append(r)
                print(f"{tid} at {np.round(q / q.max(), 5).tolist()} ({r['at'][:12]}): {r['verdict']}; around 1e-3 "
                      f"{[(short(l), n) for l, n in r['around 0.001']]}; wall points {[(short(l), n) for l, n in r['wall points']]}; "
                      f"{len(r['line directions'])} line directions", flush=True)
                json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:], os.environ.get("ATLAS_PAGE", "../../assets/symmetry_domain.html"))
