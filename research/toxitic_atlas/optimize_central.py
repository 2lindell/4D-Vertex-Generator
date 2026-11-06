"""Optimized and central seeds of the toxitic and oxitic shapes (opt_central_<sub>.json; optimized_off/toxitic|oxitic,
central_off/toxitic|oxitic).

The shapes are those of toxitic_shapes.json with the group's own symmetry or the icopyritic shapes toxitic meets on its
mirrors (sebtidi, prissi, pysni, sadi; oxitic: every shape of symmetry 576). Each is run under its own sheet's group
(CONTIC_SUB = tox, ox or oxb: an oxb shape's polychoron is the mirror image of an oxitic one) with the contic
optimiser (contic_atlas/optimize_seeds.py: R = longest / shortest edge minimised on the shape's locus while its sign
conditions hold and its arrangement key is kept) and the centre finder (decaic_atlas/central_seeds.py: the locus's
deepest point inside the shape). The locus: a mirror face or edge (the zero betas, kept zero), the fold line b1 = b4,
b2 = b3, or the curved wall surface(s) of the map that vanish at the shape's seed. Unnamed shapes are numbered
(oxitic_region_NN, oxitic_wall_NN, in the order of toxitic_shapes.json).

    CONTIC_SUB=tox|ox|oxb python optimize_central.py      (then without CONTIC_SUB to gather: python optimize_central.py gather)
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONTIC = os.path.join(ROOT, "contic_atlas")


def shapes_for(sub):
    d = json.load(open(os.path.join(HERE, "toxitic_shapes.json")))
    out = []
    for group, lim in (("toxitic", 576), ("oxitic", 576)):
        count = {}
        for s in d[group]:
            if s["symmetry"] > lim:
                continue
            count[s["kind"]] = count.get(s["kind"], 0) + 1
            name = s["name"] or f"{group}_{s['kind']}_{count[s['kind']]:02d}"
            if s["sheet"] != sub:
                continue
            out.append({"group": group, "name": name, "kind": s["kind"],
                        "degrees": {"region": 3, "wall": 2, "line": 1, "point": 0}[s["kind"]], "seed": s["seed"],
                        "key": s["key"], "hull_tol": 1e-9 if s["kind"] in ("region", "line") else 1e-7})
    return out


def locus(s, surfaces):
    import numpy as np
    b = np.asarray(s["seed"], float)
    b = b / b.sum()
    if s["kind"] == "region":
        return []
    if s["kind"] == "line" and min(b) > 1e-12:
        return ["b1 - b4", "b2 - b3"]
    if min(b) < 1e-12:
        return []                                  # (on a mirror face or edge: the zero betas stay zero)
    return [S.text for S in surfaces if S.dist(b) < 1e-7]


def job(s):
    import numpy as np
    import central_seeds as CSR
    C = CSR.C
    CS = CSR.CS
    CS.O, CS.G, CS.F = C.O, C.G, C.F
    try:
        r = C.O.optimise(s)
    except Exception as e:                          # (a failed run keeps the seed)
        r = {"start": s["seed"], "best": s["seed"], "off_seed": s["seed"], "R": None, "start_R": None,
             "status": f"optimiser failed: {e}"}
    CS.OPT = {s["name"]: r}
    try:
        c = CS.centre(s)
    except Exception as e:
        c = {"centre": s["seed"], "depth": None, "status": f"centre finder failed: {e}"}
    return r, c


def run(sub):
    sys.path.insert(0, CONTIC)
    os.chdir(CONTIC)
    import numpy as np
    import group as G
    from snub_map import Surface
    from four_d_vertex_generator.local_view import hull_edges
    from four_d_vertex_generator.off import to_4off
    M = json.load(open(os.path.join(CONTIC, G.fname("snub_map"))))
    surfaces = [Surface(t) for t in M["surfaces"]]
    S = shapes_for(sub)
    for s in S:
        s["locus"] = locus(s, surfaces)
    print(f"{sub}: {len(S)} shapes", flush=True)
    with Pool(4) as pool:
        R = pool.map(job, S, chunksize=1)
    out = []
    for s, (r, c) in zip(S, R):
        row = {k: s[k] for k in ("group", "name", "kind", "degrees", "key", "locus")}
        row.update(sheet=sub, seed=list(map(float, s["seed"])), start_R=r.get("start_R"), R=r.get("R"),
                   optimized=list(map(float, r.get("off_seed", r.get("best", s["seed"])))), optimized_status=r["status"],
                   central=list(map(float, c["centre"])), depth=c.get("depth"), central_status=c["status"])
        V0, _, _, _, cells0 = G.polytope(np.asarray(row["seed"]) / np.sum(row["seed"]), s["hull_tol"])
        for folder, key, tag in (("optimized_off", "optimized", "opt"), ("central_off", "central", "central")):
            b = np.asarray(row[key], float)
            b = b / b.sum()
            V, Vn, idx, faces, cells = G.polytope(b, s["hull_tol"])
            if (len(V), len(cells)) != (len(V0), len(cells0)) or G.label(b, s["hull_tol"])[1] != s["key"]:
                row[f"{tag}_note"] = (f"rebuilt as {len(V)}v {len(cells)}c / another key: the seed is written instead")
                b = np.asarray(row["seed"]) / np.sum(row["seed"])
                V, Vn, idx, faces, cells = G.polytope(b, s["hull_tol"])
            d = os.path.join(ROOT, folder, s["group"])
            os.makedirs(d, exist_ok=True)
            f = os.path.join(d, f"{s['name']}_{len(V)}v_{len(cells)}c_{tag}.off")
            open(f, "w").write(to_4off(Vn, faces, cells))
            row[f"{tag}_off"] = os.path.relpath(f, ROOT)
            row[f"{tag}_counts"] = [len(V), len(cells)]
            if tag == "opt":
                E = [np.linalg.norm(Vn[a] - Vn[c_]) for a, c_ in (tuple(e) for e in hull_edges(faces))]
                row["off_R"] = float(max(E) / min(E))
        out.append(row)
        print(f"{row['name']:30} {row['kind']:6} R {row['start_R'] or 0:.4f} -> {row['off_R']:.6f}  "
              f"{row['optimized_status'][:40]:40} central {row['central_status'][:30]:30} {row['opt_counts']}"
              f"{'  ' + row.get('opt_note', '') + row.get('central_note', '')}", flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": out},
              open(os.path.join(HERE, f"opt_central_{sub}.json"), "w"), indent=1)


def gather():
    rows = []
    for sub in ("tox", "ox", "oxb"):
        rows += json.load(open(os.path.join(HERE, f"opt_central_{sub}.json")))["shapes"]
    for folder, tag, title in (("optimized_off", "opt", "Optimized"), ("central_off", "central", "Central")):
        for group in ("toxitic", "oxitic"):
            L = [f"# {title} {group} seeds", "", "Made by research/toxitic_atlas/optimize_central.py (see its docstring); "
                 "shapes from toxitic_shapes.md. oxb: the mirror-image sheet (the file is the mirror image of an oxitic "
                 "orbit, the same shape).", ""]
            if tag == "opt":
                L += ["| shape | kind | sheet | R seed | R | status | beta | file |", "|---|---|---|---|---|---|---|---|"]
                for r in rows:
                    if r["group"] == group:
                        L.append(f"| {r['name']} | {r['kind']} | {r['sheet']} | {r['start_R'] or 0:.6f} | {r['off_R']:.8f} | "
                                 f"{r['optimized_status']}{'; ' + r['opt_note'] if 'opt_note' in r else ''} | "
                                 f"{[round(v, 9) for v in r['optimized']]} | {r['opt_off'].split('/')[-1]} |")
            else:
                L += ["| shape | kind | sheet | depth | status | beta | file |", "|---|---|---|---|---|---|---|"]
                for r in rows:
                    if r["group"] == group:
                        dd = "-" if r["depth"] is None else f"{r['depth']:.2e}"
                        L.append(f"| {r['name']} | {r['kind']} | {r['sheet']} | {dd} | {r['central_status']}"
                                 f"{'; ' + r['central_note'] if 'central_note' in r else ''} | "
                                 f"{[round(v, 9) for v in r['central']]} | {r['central_off'].split('/')[-1]} |")
            open(os.path.join(ROOT, folder, group, "README.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gather":
        gather()
    else:
        run(os.environ["CONTIC_SUB"])
