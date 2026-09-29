"""Half-cell fundamental domain: barycentric coordinates in one 600-cell tetrahedron, beta1 >= beta2."""
import numpy as np

PHI = (1 + 5 ** 0.5) / 2
T = np.load("cell_vertices.npy")          # rows: the cell's 4 vertices V1..V4 (half-turn swaps 1<->2, 3<->4)
_EXACT = np.array([0.0, 0.5, PHI / 2, 1 / (2 * PHI), 1.0])
T = np.sign(T) * _EXACT[np.abs(np.abs(T)[..., None] - _EXACT).argmin(-1)]   # snap stored float residue to exact golden values
TINV = np.linalg.inv(T.T)
def seed_from_beta(beta):
    p = np.asarray(beta, float) @ T; return p / np.linalg.norm(p)
def beta_from_seed(p):
    b = TINV @ p; return b / b.sum()
def in_half(beta, eps=1e-9):
    b = np.asarray(beta); return bool(np.all(b >= -eps) and b[0] >= b[1] - eps)
def golden_form(x, maxc=6):
    """Express x as (a + b*phi)/c with small integers, if possible."""
    best = None
    for c in range(1, maxc + 1):
        for b in range(-maxc, maxc + 1):
            a = round(x * c - b * PHI)
            if abs(a + b * PHI - x * c) < 1e-7 and abs(a) <= 3 * maxc:
                cand = (abs(a) + abs(b) + c, a, b, c)
                best = min(best, cand) if best else cand
    if best is None: return f"{x:.6f}"
    _, a, b, c = best
    s = (f"{a}" if a else "") + (("+" if b > 0 and a else "-" if b < 0 else "") + (f"{abs(b)}φ" if abs(b) != 1 else "φ") if b else "")
    s = s or "0"
    return s if c == 1 else f"({s})/{c}"
