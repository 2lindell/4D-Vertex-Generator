"""Each example seed of the F4 maps, pasted into the generator with its symmetry: the orbit's vertex count must be
its shape's (python check_seeds.py)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from four_d_vertex_generator.generation import group_elements  # noqa: E402
from four_d_vertex_generator.library import named_symmetry  # noqa: E402

GEN = {"icoic": "f4", "kiicoic": "f4+", "contic": "f4_extended", "kicontic": "f4_chiral_extended",
       "iocontic": "iocontic", "icopyritic": "icopyritic", "toxitic": "f4_double_diminished",
       "oxitic": "f4_extended_double_diminished"}
bad = 0
for g, name in GEN.items():
    path = os.path.join(HERE, f"f4_map_{g}.json")
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    E = np.array(group_elements(named_symmetry(name)))
    nv = {t["id"]: t.get("vertices") for t in d["types"]}
    n = 0
    for e in d["examples"]:
        s = np.array([float(v) for v in e["seed"].split(",")])
        P = E @ s
        k = len(np.unique(np.round(P, 7), axis=0))
        if k != nv[e["id"]]:
            bad += 1
            print(f"{g}: {e['id']} at {e['beta']}: {k} vertices, not {nv[e['id']]}")
        n += 1
    print(f"{g} ({name}, order {len(E)}): {n} example seeds checked")
sys.exit(1 if bad else 0)
