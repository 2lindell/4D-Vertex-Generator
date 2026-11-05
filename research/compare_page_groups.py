"""Fingerprint every page symmetry and the research groups: order, reflections, -I, and the multiset of
(determinant, characteristic polynomial) -- a basis-free description of the group's elements up to conjugacy in O(4)."""
import os, sys, subprocess, json
from collections import Counter
import numpy as np
ROOT = "/home/user/4D-Vertex-Generator"
sys.path.insert(0, ROOT + "/src")

def fp(E):
    E = np.asarray(E)
    c = Counter()
    refl = 0
    for g in E:
        d = int(round(np.linalg.det(g)))
        poly = tuple(np.round(np.poly(g).real, 6) + 0.0)
        c[(d, poly)] += 1
        w = np.linalg.eigvals(g)
        if d == -1 and np.sum(np.abs(w - 1) < 1e-6) == 3:
            refl += 1
    has_minus_I = any(np.allclose(g, -np.eye(4)) for g in E)
    return {"order": len(E), "reflections": refl, "minus_I": has_minus_I, "classes": c}

def page():
    from four_d_vertex_generator.library import named_symmetry, PAGE_SYMMETRY_NAMES
    from four_d_vertex_generator.generation import group_elements
    out = {}
    for n in PAGE_SYMMETRY_NAMES:
        try:
            out[n] = fp(group_elements(named_symmetry(n), 20000))
        except Exception as e:
            out[n] = {"error": repr(e)}
    return out

def research(atlas, env, sub):
    code = f"""
import sys, json, numpy as np
sys.path.insert(0, {ROOT + '/research/' + atlas!r}); sys.path.insert(0, {ROOT + '/src'!r})
import group as G
if hasattr(G, 'ELEMENTS'):
    E = G.ELEMENTS
else:
    B = G.BASIS
    E = np.array([B @ (s * np.eye(5)[list(p)]) @ B.T for s, p in G.ELEM])
np.save(sys.argv[1], E)
"""
    f = f"/tmp/claude-0/-home-user-4D-Vertex-Generator/53780574-1e50-5e89-8221-45dc649ee9f4/scratchpad/{atlas}_{sub}.npy"
    subprocess.run([sys.executable, "-c", code, f], env={**os.environ, env: sub}, check=True, cwd=ROOT + "/research/" + atlas)
    return np.load(f)

P = page()
R = {}
for name, atlas, env, sub in [("pennic", "decaic_atlas", "DECAIC_SUB", "pen"), ("kipennic", "decaic_atlas", "DECAIC_SUB", "kipen"),
                              ("decaic", "decaic_atlas", "DECAIC_SUB", "full"), ("kidecaic", "decaic_atlas", "DECAIC_SUB", "ki"),
                              ("iodecaic", "decaic_atlas", "DECAIC_SUB", "io"),
                              ("tessic", "tessic_atlas", "TESSIC_SUB", "full"), ("kitessic", "tessic_atlas", "TESSIC_SUB", "ki"),
                              ("demitessic", "tessic_atlas", "TESSIC_SUB", "demi"), ("kidemitessic", "tessic_atlas", "TESSIC_SUB", "kidemi"),
                              ("tessipyritic", "tessic_atlas", "TESSIC_SUB", "pyr"),
                              ("contic", "contic_atlas", "CONTIC_SUB", "full"), ("kicontic", "contic_atlas", "CONTIC_SUB", "ki"),
                              ("iocontic", "contic_atlas", "CONTIC_SUB", "io")]:
    E = research(atlas, env, sub)
    R[name] = fp(E)
    if name == "contic":                     # (icoic = F4, kiicoic = F4+: the contic elements without the swap)
        inF4 = [g for g in E if np.abs(g @ np.array([1, 1, 0, 0.]) - np.round(g @ np.array([1, 1, 0, 0.]))).max() < 1e-9]
        R["icoic"] = fp(inF4)
        R["kiicoic"] = fp([g for g in inF4 if np.linalg.det(g) > 0])
for n in P:
    p = P[n]
    if "error" in p:
        print(f"{n:14} page ERROR {p['error']}")
        continue
    line = f"{n:14} page: order {p['order']:5} refl {p['reflections']:4} -I {str(p['minus_I']):5}"
    if n in R:
        r = R[n]
        same = p["classes"] == r["classes"]
        line += f" | research: order {r['order']:5} refl {r['reflections']:4} -I {str(r['minus_I']):5} | {'SAME' if same else 'DIFFERENT'}"
        if not same and p["order"] == r["order"]:
            # which research group does the page group match?
            hit = [m for m, q in R.items() if q["classes"] == p["classes"]]
            line += f" (page group matches research {hit or 'none'})"
    else:
        line += " | (no research counterpart)"
    print(line)
