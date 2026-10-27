"""The Symmetry Domain page's overlays, from the group alone (no shape data): all in betas of the half-cell.

  * the frame: the half-cell's edges, its top face (the splitting mirror b3 = b4) and the H4 mirrors through the cell
    (b_i = b_j), cut to the half-cell;
  * the extra-symmetry axes: the circles fixed by the half-turns of the coset G.Q (normalizer.py), and the order-3
    ghost girdle of the 3600-element group (supergroups.py);
  * a fundamental domain of the 2400-element group: the Dirichlet domain about b (2, 2, 2, 1) (dirichlet.py);
  * the regular cells: the lines where pentagonal prisms or antiprisms are regular (regular_cells.py extend_lines,
    regular_lines.json) and where decagon faces are regular (decagons.py, regular_decagons.json), each with its pieces
    in the half-cell under the group and its dashed copies under Q; the shapes along each piece named by the classifier;
  * the exact points where both prism lines end (on T2) and where the regular decagons are untwisted.
"""
from __future__ import annotations

import json
import os

import numpy as np

from core import TINV, beta_text, fmt17, label, seed_from_beta
from dirichlet import dirichlet, display_faces
from dodeca_view import E, clip_images
from normalizer import coset_axes, extended_group, halfturn
from supergroups import girdle_families, girdle_segments

PHI = (1 + 5 ** 0.5) / 2
H_CELL = [TINV[i] for i in range(4)] + [TINV[2] - TINV[3]]          # (beta_i >= 0 and beta3 >= beta4, in seed space)


def clip_upper(poly):
    """A convex polygon (betas) cut to b3 >= b4."""
    out = []
    for i in range(len(poly)):
        a, b = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % len(poly)], float)
        fa, fb = a[2] - a[3], b[2] - b[3]
        if fa >= -1e-12:
            out.append(a)
        if (fa >= -1e-12) != (fb >= -1e-12):
            out.append(a + fa / (fa - fb) * (b - a))
    return out


def frame():
    """(edges, top face, mirrors), betas."""
    C = np.eye(4)
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            a, c = C[i], C[j]
            fa, fc = a[2] - a[3], c[2] - c[3]
            if fa < 0 and fc < 0:
                continue
            if fa < 0 or fc < 0:
                lo, hi = (a, c) if fa < fc else (c, a)
                flo, fhi = min(fa, fc), max(fa, fc)
                a, c = lo + (hi - lo) * (-flo / (fhi - flo)), hi
            edges.append([a, c])
    split = [C[0], C[1], (C[2] + C[3]) / 2]
    mirrors = []
    for i in range(4):
        for j in range(i + 1, 4):
            if {i, j} == {2, 3}:
                continue
            k, m = [x for x in range(4) if x not in (i, j)]
            poly = clip_upper([C[k], C[m], (C[i] + C[j]) / 2])
            if len(poly) >= 3:
                mirrors.append(poly)
    return edges, split, mirrors


def axes():
    """The extra-symmetry axes: [{a, b (betas), order, hover}]."""
    family = girdle_families()
    out = []
    for a, b in coset_axes():
        fam = family(a, b)
        bowers = " (Bowers' 30 ghost girdles with skew 20-gonal symmetry)" if fam == 30 else ""
        out.append({"a": a, "b": b, "order": 2, "hover": "Axis of an extra half-turn (2400-element group)<br>"
                    f"one of {fam} such circles{bowers}<br>seeds on it are their own copy; their polytopes have 2400 "
                    f"symmetries<br>from β = {np.round(a, 5).tolist()}<br>to β = {np.round(b, 5).tolist()}"})
    for a, b in girdle_segments():
        out.append({"a": a, "b": b, "order": 3, "hover": "Axis of an order-3 rotation of the 3600-element group<br>"
                    "one of Bowers' 20 ghost girdles with 30/3-gyrogonic symmetry<br>seeds on it give 1200-vertex "
                    f"polytopes (X12) with 3600 symmetries<br>from β = {np.round(a, 5).tolist()}"
                    f"<br>to β = {np.round(b, 5).tolist()}"})
    return out


def fundamental_domain():
    """The 2400-element group's Dirichlet domain about b (2, 2, 2, 1): ([{kind, pts, axis, hover}], centre)."""
    group_n = extended_group()
    centre = np.array([2, 2, 2, 1.0])
    domain = dirichlet(seed_from_beta(centre), group_n)
    glue = {"q-axis": "folded onto itself across a light purple axis (a half-turn of the extra coset)",
            "ring-axis": "folded onto itself across a cross-ring axis (a half-turn of the swirlprism group)"}
    faces = []
    for face in display_faces(domain, group_n, len(group_n) // 2):
        how = glue.get(face["kind"], "glued to the other face of the same colour by "
                       + ("an element of the extra coset" if face["coset"] else "a rotation of the swirlprism group"))
        faces.append({"kind": face["kind"], "pts": [np.asarray(b, float) for b in face["corners"]],
                      "axis": [np.asarray(b, float) for b in face["axis"]],
                      "hover": f"Fundamental domain of the 2400-element group<br>{len(face['corners'])}-sided face, {how}"})
    return faces, centre / centre.sum()


def regular(names):
    """The regular-cell lines: [{kind, copy, pts (betas), hover}], and the exact points on them: [{id, b, hover, ...}]."""
    Qm = halfturn()
    out, special = [], []

    def to_betas(piece):
        B = [TINV @ x for x in piece]
        return [b / b.sum() for b in B]

    def shapes_on(B):
        found = []
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            k = t * (len(B) - 1)
            i = int(k)
            p = B[i] + (k - i) * (B[min(i + 1, len(B) - 1)] - B[i])
            lab = label(p)[0]
            if lab in names and lab not in found:
                found.append(lab)
        return ", ".join(f"{t} · {names[t]}" for t in found)

    def arc(a, b, n=96, whole=False):
        if whole:                                          # (the cells stay regular all the way round the circle)
            u = seed_from_beta(a)
            v = seed_from_beta(b)
            v = v - (v @ u) * u
            v /= np.linalg.norm(v)
            th = np.linspace(0, 2 * np.pi, 8 * n + 1)
            return np.outer(np.cos(th), u) + np.outer(np.sin(th), v)
        return np.array([seed_from_beta(a + (b - a) * k / n) for k in range(n + 1)])

    def on_lines(B, lines, tol=1e-5):
        for p in B[1:-1] or B:
            if not any(np.linalg.norm(u + np.clip((p - u) @ (v - u) / max((v - u) @ (v - u), 1e-30), 0, 1) * (v - u) - p) < tol
                       for L in lines for u, v in zip(L[:-1], L[1:])):
                return False
        return True

    def length(B):
        return float(np.linalg.norm(np.diff(np.array(B), axis=0), axis=1).sum())

    def hover(kind, B, copy):
        name = {"prism": "prisms", "antiprism": "antiprisms"}[kind]
        return (f"Regular pentagonal {name}{' (copy under the extra half-turn)' if copy else ''}<br>shapes along this "
                f"piece: {shapes_on(B)}<br>from β ∝ {beta_text(np.clip(B[0], 0, None))}"
                f"<br>to β ∝ {beta_text(np.clip(B[-1], 0, None))}")

    def add_family(kind, specs):
        originals, copies = [], []
        for a, b, whole in specs:
            for pc in clip_images(arc(a, b, whole=whole), H_CELL):
                B = to_betas(pc)
                if length(B) < 1e-5 or on_lines(B, originals):
                    continue                               # (a corner, or the same line reached by another element)
                originals.append(B)
                out.append({"kind": kind, "copy": False, "pts": B, "hover": hover(kind, B, False)})
        for a, b, whole in specs:
            for pc in clip_images(arc(a, b, whole=whole) @ Qm.T, H_CELL):
                B = to_betas(pc)
                if length(B) < 1e-5 or on_lines(B, originals) or on_lines(B, copies):
                    continue                               # (on an original line: that line, not a separate copy)
                copies.append(B)
                out.append({"kind": kind, "copy": True, "pts": B, "hover": hover(kind, B, True)})

    def circle(a, b):
        u = seed_from_beta(a)
        v = seed_from_beta(b)
        v = v - (v @ u) * u
        v /= np.linalg.norm(v)
        return np.outer(u, u) + np.outer(v, v)
    fams, kept = {"prism": [], "antiprism": []}, []
    for line in json.load(open("regular_lines.json")) if os.path.exists("regular_lines.json") else []:
        whole = max(abs(t) for t in line["t"]) >= 2.99
        a, b = np.array(line["a"]), np.array(line["b"])
        plane = circle(a, b)
        same = [Pk for k, Pk in kept if k == line["kind"]]
        if (not any(np.allclose(g @ Pk @ g.T, plane, atol=1e-7) for Pk in same for g in E)
                and any(np.allclose(g @ Qm @ Pk @ Qm.T @ g.T, plane, atol=1e-7) for Pk in same for g in E)):
            continue                                       # (the half-turn copy of a line already kept)
        kept.append((line["kind"], plane))
        fams[line["kind"]].append((a, b, whole))
    add_family("prism", fams["prism"])
    add_family("antiprism", fams["antiprism"])
    for st in json.load(open("regular_decagons.json")) if os.path.exists("regular_decagons.json") else []:
        B = to_betas(np.array(st["seeds"]))
        copy = bool(st.get("copy"))
        out.append({"kind": "decagon", "copy": copy, "pts": B,
                    "hover": f"Regular decagon faces (two regular pentagons merged in one plane)"
                             f"{' (copy under the extra half-turn)' if copy else ''}<br>shapes along this piece: "
                             f"{shapes_on(B)}<br>from β ∝ {beta_text(np.clip(B[0], 0, None))}"
                             f"<br>to β ∝ {beta_text(np.clip(B[-1], 0, None))}"})
    pt2 = np.array([2.0, 3.0, 2.0, 2 - PHI])
    special.append({"id": label(pt2)[0], "b": pt2 / pt2.sum(), "prismT2": True,
                    "hover": "Both regular prism lines end here: the prisms become corealmic with their neighbours; an "
                             f"antiprism line passes through<br>β ∝ {beta_text(pt2)}<br>seed "
                             + ", ".join(fmt17(v) for v in seed_from_beta(pt2))})
    for p, copy in ((np.array([PHI ** 2, 1, 1, 1]), False), (np.array([PHI ** 2, 1, 2, 1]), True)):
        special.append({"id": label(p)[0], "b": p / p.sum(), "decagon": True,
                        "hover": "Untwisted regular decagons: each decagon's cell is a pentagonal cupola whose pentagon "
                                 "lines up with the decagon; a geometric coincidence, not extra symmetry"
                                 f"{' (copy under the extra half-turn)' if copy else ''}<br>β ∝ {beta_text(p)}<br>seed "
                                 + ", ".join(fmt17(v) for v in seed_from_beta(p))})
    return out, special
