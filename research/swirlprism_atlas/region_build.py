"""Every region from the wall sections (sections/, cut_walls.py), checked, and the database made from them.

  1. Each section (above a sliver's area) is checked by the classifier: points well inside it, a little off it on
     each side, brought into the half-cell by the group and classified. A side whose points all agree names the
     region there; a side whose points disagree has a cut missing (a wall or line on it not yet found); a label that
     is not a known shape is a new shape. Where the classifier and the section's stored sides disagree, the
     classifier's names are kept (the voxels are coarse beside thin regions) and the change is listed.
  2. Each region's walls: the walls with a section beside it; partial where some section of the wall (outside the
     face and mirror views) is not.
  3. Compared with the voxel survey's walls for each region (region_db.json touch) and with the region types of the
     page: regions with no sections, walls the survey lists that no section confirms, and the reverse.

Writes region_defs.json (every region), region_check.json (all findings) and the sections' checked sides.

    python region_build.py            (then: python region_view.py --defs --nocut)
"""
from __future__ import annotations

import glob
import gzip
import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

import region_cells as rc
import region_walls as rw
from region_db import home

OUT = "sections"
SLIVER = 1e-4          # (sections smaller than this are not checked or counted)
OFF = (3e-4, 6e-4, 1.2e-3, 2.5e-3)  # (steps off the section: each point is named by the first region it meets)
NPTS = 5
BEYOND = "(beyond the half-cell)"


def _classify(b):
    if np.isnan(b).any():
        return "OUT"
    return rc._lab_job(b)


def inner_points(T, n=NPTS):
    """Points well inside a section, spread over it: of points sampled over it, the farthest from its own edge,
    taken apart from each other; with the normal there and the distance to the edge."""
    from cut_walls import sample_tris
    key = lambda p: tuple(np.round(p, 7))
    cnt = Counter()
    for t in T:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            cnt[frozenset((key(t[a]), key(t[b])))] += 1
    E = [np.array(list(e)) for e, k in cnt.items() if k == 1 and len(e) == 2]
    X, owner = sample_tris(T, 0.0015, owner=True)
    N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-300)
    far = cKDTree(np.concatenate([a + np.linspace(0, 1, 40)[:, None] * (b - a) for a, b in E])).query(X)[0] if E \
        else np.ones(len(X))
    pick = []
    for i in np.argsort(-far):
        if all(np.linalg.norm(X[i] - X[j]) > 0.5 * far[i] for j in pick):
            pick.append(i)
        if len(pick) == n:
            break
    pick = np.array(pick)
    return X[pick], N[owner[pick]], far[pick]


def main():
    d = rw.page_data()
    regions = [t["id"] for t in d["types"] if t.get("dim") == "region"]
    known = {t["id"] for t in d["types"]}
    secs = {os.path.basename(f).split(".")[0]: json.load(gzip.open(f, "rt")) for f in sorted(glob.glob(f"{OUT}/*.json.gz"))}
    jobs, where = [], []
    for w, faces in secs.items():
        for k, f in enumerate(faces):
            if f["area"] < SLIVER:
                continue
            T = np.array(f["tris"]).reshape(-1, 3, 3)
            P, N, far = inner_points(T)
            face = BEYOND in f["sides"]
            if face:
                # (a piece in a face or the mirror: only its inside, stepped into the half-cell in betas)
                Bp = rc.betas(P)
                fk = [j for j in range(4) if np.abs(Bp[:, j]).max() < 1e-6]
                into = np.eye(4)[fk[0]] if fk else np.array([0, 0, 0.5, -0.5])
            for sd, sg in ((0, 1), (1, -1)):
                if face and sd == 1:
                    continue
                for pi_ in range(len(P)):
                    for oi, off in enumerate(OFF):
                        off = min(off, 0.45 * float(far[pi_]))
                        if face:
                            Y = rc.betas(P[pi_][None]) + 0.6 * off * into
                            Y = Y / Y.sum(1, keepdims=True)
                        else:
                            Y = rc.betas((P[pi_] + sg * off * N[pi_])[None])
                        jobs.append(home(Y)[0])
                        where.append((w, k, sd, pi_, oi))
    print(f"{len(jobs)} points to classify on {sum(len(v) for v in secs.values())} sections of {len(secs)} walls", flush=True)
    with Pool(4) as pool:
        labs = pool.map(_classify, jobs, chunksize=20)
    # (each point named by the first region met stepping out from the section; a wall's or line's own name, or an
    # error, at a small step is passed over)
    region_set = set(regions)
    steps = defaultdict(dict)
    for (w, k, sd, pi_, oi), lab in zip(where, labs):
        steps[(w, k, sd, pi_)][oi] = lab
    got = defaultdict(lambda: defaultdict(Counter))
    for (w, k, sd, pi_), by_off in steps.items():
        seq = [by_off[o] for o in sorted(by_off)]
        first = next((x for x in seq if x in region_set), None)
        if first is None:
            first = next((x for x in seq if x not in ("ERR", "TIMEOUT", "OUT")), seq[-1])
        got[(w, k)][sd][first] += 1
    findings = {"changed": [], "mixed": [], "new_shapes": [], "same_both_sides": [], "unclassified": []}
    for (w, k), by in got.items():
        f = secs[w][k]
        face = BEYOND in f["sides"]
        names = []
        for sd in ((0,) if face else (0, 1)):
            c = Counter({k_: v_ for k_, v_ in by[sd].items() if k_ not in ("ERR", "TIMEOUT")}) or by[sd]
            top, n_top = c.most_common(1)[0]
            tot = sum(c.values())
            if n_top < 0.7 * tot:
                findings["mixed"].append({"wall": w, "section": k, "area": f["area"], "side": sd, "labels": dict(c),
                                          "stored": f["sides"]})
            if top not in known and top not in ("ERR", "OUT", "TIMEOUT"):
                findings["new_shapes"].append({"wall": w, "section": k, "label": top, "count": n_top})
            if top in ("ERR", "OUT", "TIMEOUT"):
                findings["unclassified"].append({"wall": w, "section": k, "side": sd, "labels": dict(c)})
            names.append(top)
        if face:
            names.append(BEYOND)
        stored = f["sides"]
        if set(names) != set(stored) and not any(n in ("ERR", "OUT", "TIMEOUT") for n in names):
            findings["changed"].append({"wall": w, "section": k, "area": f["area"], "piece": f["piece"], "kind": f["kind"],
                                        "stored": stored, "classified": names})
            f["voxel_sides"] = stored
            f["sides"] = names
            f["named"] = "classified"
        if not face and names[0] == names[1]:
            findings["same_both_sides"].append({"wall": w, "section": k, "area": f["area"], "region": names[0]})
        f["checked"] = {str(sd): dict(c) for sd, c in by.items()}
    for w, faces in secs.items():
        json.dump(faces, gzip.open(os.path.join(OUT, f"{w}.json.gz"), "wt"), separators=(",", ":"))
    # (2) each region's walls, and the partial ones
    walls_of, part_of = defaultdict(set), defaultdict(set)
    for w, faces in secs.items():
        for f in faces:
            if f["area"] < SLIVER:
                continue
            for r in f["sides"]:
                if r != BEYOND:
                    walls_of[r].add(w)
    for r, ws in walls_of.items():
        for w in ws:
            if any(r not in f["sides"] and BEYOND not in f["sides"] and f["area"] >= SLIVER for f in secs[w]):
                part_of[r].add(w)
    # (3) against the survey and the page's regions
    db = json.load(open("region_db.json"))
    touch = db.get("touch", {})
    compare = {}
    for r in sorted(set(regions) | set(walls_of)):
        if r in ("ERR", "OUT", "TIMEOUT", "unknown"):
            continue
        a, b = walls_of.get(r, set()), set(touch.get(r, []))
        compare[r] = {"walls": sorted(a), "partial": sorted(part_of.get(r, set())),
                      "survey_only": sorted(b - a), "sections_only": sorted(a - b),
                      "listed": r in regions}
    findings["regions"] = compare
    findings["regions_without_sections"] = sorted(r for r in regions if r not in walls_of)
    json.dump(findings, open("region_check.json", "w"), indent=1)
    defs = json.load(open("region_defs.json"))
    about = defs.get("_about")
    defs = {"_about": about}
    for r, c in compare.items():
        if c["walls"] and c["listed"]:
            defs[r] = {"walls": c["walls"]}
    open("region_defs.json", "w").write(json.dumps(defs, indent=1, ensure_ascii=False) + "\n")
    for k in ("changed", "mixed", "new_shapes", "same_both_sides", "unclassified"):
        print(f"{k}: {len(findings[k])}")
    print(f"regions without sections: {findings['regions_without_sections']}")
    for r, c in compare.items():
        print(f"  {r}: {', '.join(w + (' (part)' if w in c['partial'] else '') for w in c['walls'])}"
              + (f"   survey only: {', '.join(c['survey_only'])}" if c["survey_only"] else "")
              + ("" if c["listed"] else "   NOT A LISTED REGION"))


if __name__ == "__main__":
    main()
