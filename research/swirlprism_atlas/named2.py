import numpy as np
from catalog import match
from classify import classify, signature
from scipy.optimize import brentq
from survey import BASIS, FULL, A

from four_d_vertex_generator.generation import generate_vertices_from_seed as g
from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry
from four_d_vertex_generator.off import compute_convex_hull

H4 = named_symmetry("h4_icosian")
E = np.stack(group_elements(FULL))
def slerp(a, b, f):
    w = np.arccos(np.clip(a @ b, -1, 1)); return (np.sin((1 - f) * w) * a + np.sin(f * w) * b) / np.sin(w)
def truncation_point(v, n1, n2):
    """Uniform truncation of vertex v: leftover edge v->n1 equals the cut edge between edges v->n1 and v->n2."""
    h = lambda f: np.linalg.norm(slerp(v, n1, f) - slerp(n1, v, f)) - np.linalg.norm(slerp(v, n1, f) - slerp(v, n2, f))
    f = brentq(h, 1e-6, 0.5 - 1e-9); return slerp(v, n1, f), f
def into_cell(p):
    X = E @ p; x = X[int(np.argmax(X @ A))]; return BASIS @ (x / (x @ A)), x
def report(label, p):
    q, x = into_cell(p); s = signature(classify(x)); VH = g(p, H4, tol=1e-7)
    print(f"{label}: uniform orbit {len(VH)}, swirlprism orbit {s.split(':')[0]}, match: {match(s) or 'unlisted'}")
    print(f"   {s}\n   atlas position q = {np.round(q, 5)}  seed {','.join(f'{c:.7f}' for c in x / np.linalg.norm(x))}")
# 600-cell: anchor, two adjacent tilted neighbours (not on the main ring)
V600 = g(A, H4, tol=1e-7); d = np.linalg.norm(V600 - A, axis=1); nb = V600[np.isclose(d, np.sort(d)[1])]
main = np.array([0, -0.5257311121, 0, -0.8506508084])
tilted = [u for u in nb if abs(abs((u - (u @ A) * A) @ main / np.linalg.norm(u - (u @ A) * A)) - 1) > 1e-6]
n1 = tilted[0]; n2 = min((u for u in tilted if not np.allclose(u, n1)), key=lambda u: np.linalg.norm(u - n1))
p, f = truncation_point(A, n1, n2); print(f"600-cell edge cut at fraction {f:.6f}")
report("truncated 600-cell vertex on a non-main-ring edge", p)
pm, _ = truncation_point(A, nb[np.argmax([abs((u - (u @ A) * A) @ main) for u in nb])], n1)
print("   (on a main-ring edge instead:", len(g(pm, FULL, tol=1e-6)), "vertices under swirlprism)")
# 120-cell: vertex = a cell centre of the 600-cell, its 4 neighbours

faces, cells = compute_convex_hull(V600)
c0 = V600[sorted({v for fi in cells[0] for v in faces[fi]})].mean(0); c0 /= np.linalg.norm(c0)
V120 = g(c0, H4, tol=1e-7); d = np.linalg.norm(V120 - c0, axis=1); nb120 = V120[np.isclose(d, np.sort(d)[1])]
print(f"120-cell: {len(V120)} vertices, {len(nb120)} neighbours per vertex")
found = set()
for i in range(len(nb120)):
    others = [u for j, u in enumerate(nb120) if j != i]
    p, f = truncation_point(c0, nb120[i], others[0])
    n = len(g(p, FULL, tol=1e-6))
    key = n
    if key in found: continue
    found.add(key); print(f"120-cell edge {i} cut at fraction {f:.6f}")
    report(f"truncated 120-cell vertex (edge {i})", p)

print("\nall four edges at a 120-cell vertex:")
halves = []
for i in range(len(nb120)):
    others = [u for j, u in enumerate(nb120) if j != i]
    p, f = truncation_point(c0, nb120[i], others[0])
    s = signature(classify(into_cell(p)[1]))
    VG = g(p, FULL, tol=1e-6); halves.append(VG)
    print(f"   edge {i}: {s}")
fp = lambda V: np.round(np.sort(np.linalg.norm(V - V[0], axis=1)), 6)
print("   edge 0 and edge 1..3 congruent:", [np.allclose(fp(halves[0]), fp(h), atol=1e-6) for h in halves[1:]])
