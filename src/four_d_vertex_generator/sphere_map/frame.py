"""The swirlprism atlas frame: the half-cell domain, the group, and the exact surfaces.

Everything here is read from the atlas source data (``data/swirlprism/source``, copied from the
atlas branch). Coordinates inside the domain are barycentric weights ``beta`` on the four corners
V1..V4 of one 600-cell tetrahedron; the fundamental domain of ``h4_swirlprism`` is the half of that
cell with ``beta3 >= beta4``. The 1200 group elements carry it onto 1200 copies that tile the
3-sphere exactly once.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..generation import _finite_group_elements
from ..library import named_symmetry

PHI = (1 + 5**0.5) / 2
SYMMETRY = "h4_swirlprism"

# The five faces of the half-cell, as linear forms n . beta >= 0.
DOMAIN_FACES: dict[str, np.ndarray] = {
    "b1=0": np.array([1.0, 0.0, 0.0, 0.0]),
    "b2=0": np.array([0.0, 1.0, 0.0, 0.0]),
    "b3=0": np.array([0.0, 0.0, 1.0, 0.0]),
    "b4=0": np.array([0.0, 0.0, 0.0, 1.0]),
    "b3=b4": np.array([0.0, 0.0, 1.0, -1.0]),
}


def data_dir() -> Path:
    """The swirlprism data folder (``$SWIRLPRISM_DATA`` overrides the repository default)."""
    env = os.environ.get("SWIRLPRISM_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data" / "swirlprism"


def source_dir() -> Path:
    return data_dir() / "source"


def load_json(name: str):
    with open(source_dir() / name, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def cell_vertices() -> np.ndarray:
    """Rows V1..V4 of the cell in R^4, snapped to their exact golden values."""
    t = np.load(source_dir() / "cell_vertices.npy")
    exact = np.array([0.0, 0.5, PHI / 2, 1 / (2 * PHI), 1.0])
    return np.sign(t) * exact[np.abs(np.abs(t)[..., None] - exact).argmin(-1)]


def point_from_beta(beta: np.ndarray) -> np.ndarray:
    """Unit point(s) on S^3 for barycentric weight(s) ``beta`` (shape (..., 4))."""
    p = np.asarray(beta, float) @ cell_vertices()
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


def beta_from_point(p: np.ndarray) -> np.ndarray:
    """Barycentric weights (sum 1) of point(s) ``p`` in the cell's frame."""
    b = np.asarray(p, float) @ np.linalg.inv(cell_vertices())
    return b / b.sum(axis=-1, keepdims=True)


def in_domain(beta: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    b = np.atleast_2d(beta)
    return (b > -tol).all(axis=1) & (b[:, 2] >= b[:, 3] - tol)


@lru_cache(maxsize=1)
def group() -> np.ndarray:
    """The 1200 elements of h4_swirlprism as 4x4 matrices on R^4, element 0 the identity."""
    action = named_symmetry(SYMMETRY)
    elements = _finite_group_elements(tuple(g.tobytes() for g in action.generators), 5000)
    if elements is None or len(elements) != 1200:
        raise RuntimeError("h4_swirlprism did not close at order 1200")
    e = np.stack(elements)
    e[np.abs(e) < 1e-15] = 0.0
    return e


def element_keys(elements: np.ndarray) -> list[tuple]:
    return [tuple(np.round(m, 6).ravel() + 0.0) for m in elements]


@lru_cache(maxsize=1)
def element_index() -> dict[tuple, int]:
    return {k: i for i, k in enumerate(element_keys(group()))}


def index_of(matrix: np.ndarray) -> int:
    return element_index()[tuple(np.round(matrix, 6).ravel() + 0.0)]


@lru_cache(maxsize=1)
def multiplication_table() -> np.ndarray:
    """table[i, j] = index of G[i] @ G[j]."""
    g = group()
    idx = element_index()
    prods = np.einsum("aij,bjk->abik", g, g)
    keys = np.round(prods.reshape(len(g), len(g), 16), 6) + 0.0
    out = np.empty((len(g), len(g)), dtype=np.int32)
    for a in range(len(g)):
        for b in range(len(g)):
            out[a, b] = idx[tuple(keys[a, b])]
    return out


def locate(points: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """For each point on S^3: (copy index k, beta of G[k]^-1 p in the domain).

    Copy k is the domain carried by G[k]; points on a shared face go to the lowest index.
    """
    p = np.atleast_2d(points)
    g = group()
    tinv = np.linalg.inv(cell_vertices())
    b = np.einsum("gji,nj->gni", g, p) @ tinv      # G^T = G^-1 (orthogonal)
    inside = (b > -tol).all(-1) & (b[..., 2] >= b[..., 3] - tol)
    if not inside.any(axis=0).all():
        raise ValueError("a point lies in no copy of the domain")
    k = inside.argmax(axis=0)
    beta = b[k, np.arange(len(p))]
    return k, beta / beta.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------- surfaces


@dataclass(frozen=True)
class Surface:
    """One exact surface of the atlas library: a plane n . beta = 0 or a homogeneous polynomial."""

    index: int
    name: str
    kind: str                       # "plane" | "curved"
    normal: np.ndarray | None       # plane normal (beta coordinates)
    mons: tuple[tuple[int, ...], ...] = ()
    coeffs: np.ndarray | None = None
    a: np.ndarray | None = None     # polynomial taken at A @ beta
    exact: object = None
    degree: int = 1
    aliases: tuple[str, ...] = ()   # other library names for the same surface (e.g. negated normal)

    def f(self, beta: np.ndarray) -> np.ndarray:
        b = np.atleast_2d(beta)
        if self.kind == "plane":
            return b @ self.normal
        y = b @ self.a.T
        return sum(c * np.prod(y[:, list(m)], axis=1) for c, m in zip(self.coeffs, self.mons))

    def grad(self, beta: np.ndarray) -> np.ndarray:
        b = np.atleast_2d(beta)
        if self.kind == "plane":
            return np.broadcast_to(self.normal, b.shape).copy()
        y = b @ self.a.T
        gy = np.zeros_like(y)
        for c, m in zip(self.coeffs, self.mons):
            m = list(m)
            for pos, var in enumerate(m):
                rest = m[:pos] + m[pos + 1:]
                gy[:, var] += c * np.prod(y[:, rest], axis=1)
        return gy @ self.a

    def to_json(self) -> dict:
        out = {"index": self.index, "name": self.name, "kind": self.kind, "degree": self.degree}
        if self.kind == "plane":
            out["normal"] = self.normal.tolist()
        else:
            out["monomials"] = [list(m) for m in self.mons]
            out["coefficients"] = self.coeffs.tolist()
            out["A"] = self.a.tolist()
        out["exact"] = self.exact
        out["aliases"] = list(self.aliases)
        out["equation"] = "n . beta = 0" if self.kind == "plane" else "sum_c c * prod_i (A beta)_i = 0"
        return out


@lru_cache(maxsize=1)
def surfaces() -> tuple[Surface, ...]:
    """The library's distinct surfaces (a plane stored twice, once per normal sign, is one)."""
    data = load_json("surfaces.json")
    raw: list[Surface] = []
    for p in data["planes"]:
        raw.append(Surface(0, p["name"], "plane", np.asarray(p["n"], float), exact=p.get("exact")))
    for c in data["curved"]:
        mons = tuple(tuple(m) for m in c["mons"])
        raw.append(Surface(0, c["name"], "curved", None, mons,
                           np.asarray(c["coeffs"], float), np.asarray(c["A"], float),
                           exact=c.get("exact"), degree=max(len(m) for m in mons)))
    probe = np.random.default_rng(12345).dirichlet(np.ones(4), 64)
    vals = np.stack([s.f(probe) for s in raw])
    vals /= np.linalg.norm(vals, axis=1, keepdims=True)
    out: list[Surface] = []
    kept_vals: list[np.ndarray] = []
    aliases: dict[int, list[str]] = {}
    for s, v in zip(raw, vals):
        dup = next((i for i, w in enumerate(kept_vals) if abs(abs(float(v @ w)) - 1) < 1e-12), None)
        if dup is None:
            out.append(s)
            kept_vals.append(v)
        else:
            aliases.setdefault(dup, []).append(s.name)
    return tuple(
        Surface(i, s.name, s.kind, s.normal, s.mons, s.coeffs, s.a, s.exact, s.degree,
                tuple(aliases.get(i, ())))
        for i, s in enumerate(out)
    )


@lru_cache(maxsize=1)
def surface_by_name() -> dict[str, int]:
    out = {}
    for s in surfaces():
        out[s.name] = s.index
        for a in s.aliases:
            out[a] = s.index
    return out


def surface_values(beta: np.ndarray, surfs=None) -> np.ndarray:
    """(n, S) values of every surface (default: the source library) at the (normalised) betas."""
    b = np.atleast_2d(np.asarray(beta, float))
    b = b / b.sum(axis=1, keepdims=True)
    return np.stack([s.f(b) for s in (surfs or surfaces())], axis=1)


def surface_scale(beta: np.ndarray, surfs=None) -> np.ndarray:
    """(n, S) gradient norms (in the sum-zero directions), to turn values into distances."""
    b = np.atleast_2d(np.asarray(beta, float))
    b = b / b.sum(axis=1, keepdims=True)
    out = []
    for s in (surfs or surfaces()):
        g = s.grad(b)
        g = g - g.mean(axis=1, keepdims=True)
        out.append(np.linalg.norm(g, axis=1))
    return np.stack(out, axis=1)
