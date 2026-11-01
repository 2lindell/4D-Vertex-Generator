"""The surface library the map is cut by: the atlas surfaces plus the images the walls need.

``surfaces.json`` holds the surfaces the atlas's walls were traced on, with some of their images in
the half-cell, but not all: some walls in the domain lie on images that are missing. Each was found
where one arrangement piece classified as two different shapes; the point where the shape changes
lies on an image of one of the atlas surfaces under the 2400 maps of G and G.Q (the shapes, and so
the walls, are invariant under both: Q maps every orbit to a congruent one). Only those images are
added; carrying every surface everywhere would add about a thousand images per curved surface,
almost all of them cutting regions where no wall is.

Each added image keeps its provenance (base surface, map index) and the points that called for it,
so it stays exact: the image of the surface F(beta) = 0 under the map M is F(M^-1 beta) = 0.
Surfaces that are images of one another form a family (used to name walls from the working data).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .frame import Surface, cell_vertices, data_dir, group, source_dir, surfaces


@lru_cache(maxsize=1)
def beta_maps() -> np.ndarray:
    """The 2400 maps of G and G.Q acting on (unnormalised) betas; 0..1199 are G."""
    t = cell_vertices()
    tinv_t = np.linalg.inv(t.T)
    e = group()
    q = np.load(source_dir() / "normalizer_q.npy")
    ee = np.concatenate([e, e @ q])
    return np.einsum("ij,gjk,kl->gil", tinv_t, ee, t.T)


@lru_cache(maxsize=1)
def _inverse_maps() -> np.ndarray:
    return np.linalg.inv(beta_maps())


@dataclass(frozen=True)
class SurfaceImage:
    surface: Surface
    base: str                 # source surface it is an image of
    map_index: int            # index into beta_maps(): the surface is base(M^-1 beta) = 0
    in_source: bool           # present in surfaces.json
    family: int               # surfaces that are images of one another share a family
    evidence: tuple = field(default=())   # points (betas) where a shape change called for it


def image_of(s: Surface, mi: int, name: str | None = None) -> Surface:
    inv = _inverse_maps()[mi]
    if s.kind == "plane":
        n = inv.T @ s.normal
        return Surface(0, name or f"{s.name} @ {mi}", "plane", n / np.linalg.norm(n))
    return Surface(0, name or f"{s.name} @ {mi}", "curved", None, s.mons, s.coeffs, s.a @ inv,
                   degree=s.degree)


def _all_image_values(s: Surface, pts: np.ndarray) -> np.ndarray:
    """Values (maps, points) of every image of s (planes with unit normals)."""
    inv = _inverse_maps()
    if s.kind == "plane":
        normals = np.einsum("gji,j->gi", inv, s.normal)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return normals @ pts.T
    a_stack = np.einsum("ij,gjk->gik", s.a, inv)
    y = np.matmul(pts[None], a_stack.transpose(0, 2, 1))
    cols = [y[:, :, i] for i in range(4)]
    out = np.zeros(y.shape[:2])
    for c, m in zip(s.coeffs, s.mons):
        term = cols[m[0]] * c
        for v in m[1:]:
            term = term * cols[v]
        out += term
    return out


_SIG_PTS = np.random.default_rng(99).dirichlet(np.ones(4), 48)


def _signature(img: Surface) -> np.ndarray:
    v = img.f(_SIG_PTS)
    v = v / np.linalg.norm(v)
    return -v if v[np.argmax(np.abs(v))] < 0 else v


@lru_cache(maxsize=1)
def source_families() -> tuple[int, ...]:
    """Family of each source surface: sources that are images of one another share one."""
    src = surfaces()
    sigs = np.stack([_signature(s) for s in src])
    parent = list(range(len(src)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, s in enumerate(src):
        vals = _all_image_values(s, _SIG_PTS)                    # (maps, 48)
        vals /= np.linalg.norm(vals, axis=1, keepdims=True)
        match = np.abs(vals @ sigs.T) > 1 - 1e-10                # (maps, sources)
        for j in np.flatnonzero(match.any(axis=0)):
            if src[j].kind == s.kind:
                parent[find(int(j))] = find(i)
    roots = sorted({find(i) for i in range(len(src))})
    rank = {r: k for k, r in enumerate(roots)}
    return tuple(rank[find(i)] for i in range(len(src)))


def images_through(x: np.ndarray, families: set[int] | None = None,
                   tol: float = 1e-7) -> list[tuple[int, int, float]]:
    """(source index, map index, distance) of images of source surfaces passing through x."""
    src = surfaces()
    fam = source_families()
    out = []
    x = np.asarray(x, float) / np.sum(x)
    for i, s in enumerate(src):
        if families is not None and fam[i] not in families:
            continue
        vals = _all_image_values(s, x[None])[:, 0]
        if s.kind == "curved":
            for mi in np.flatnonzero(np.abs(vals) < 1e-3):
                img = image_of(s, int(mi))
                g = img.grad(x[None])[0]
                g = g - g.mean()
                d = abs(vals[mi]) / (np.linalg.norm(g) + 1e-300)
                if d < tol:
                    out.append((i, int(mi), float(d)))
        else:
            for mi in np.flatnonzero(np.abs(vals) < tol):
                out.append((i, int(mi), float(abs(vals[mi]))))
    return out


class Library:
    """The working surface library: the source surfaces plus the images added for walls."""

    def __init__(self, images: list[SurfaceImage] | None = None) -> None:
        if images is None:
            fam = source_families()
            images = [SurfaceImage(s, s.name, 0, True, fam[s.index]) for s in surfaces()]
        self.images: list[SurfaceImage] = []
        self._sigs: list[np.ndarray] = []
        for im in images:
            self._append(im)

    def _append(self, im: SurfaceImage) -> int:
        i = len(self.images)
        s = im.surface
        surf = Surface(i, s.name, s.kind, s.normal, s.mons, s.coeffs, s.a, s.exact, s.degree,
                       s.aliases)
        self.images.append(SurfaceImage(surf, im.base, im.map_index, im.in_source, im.family,
                                        tuple(im.evidence)))
        self._sigs.append(_signature(surf))
        return i

    def find(self, surf: Surface) -> int | None:
        v = _signature(surf)
        for i, w in enumerate(self._sigs):
            if abs(float(v @ w)) > 1 - 1e-10:
                return i
        return None

    def add_image(self, source_index: int, map_index: int, evidence) -> tuple[int, bool]:
        src = surfaces()[source_index]
        img = image_of(src, map_index)
        found = self.find(img)
        if found is not None:
            return found, False
        fam = source_families()[source_index]
        return self._append(SurfaceImage(img, src.name, map_index, False, fam,
                                         (tuple(map(float, evidence)),))), True

    def surfaces(self) -> tuple[Surface, ...]:
        return tuple(im.surface for im in self.images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> SurfaceImage:
        return self.images[i]

    def __iter__(self):
        return iter(self.images)

    # ------------------------------------------------------------------------- persistence

    def to_json(self) -> dict:
        return {
            "_about": "The surfaces the map is cut by: the atlas surfaces (surfaces.json, duplicates "
                      "merged) and the images of them that walls in the domain lie on, found where "
                      "one piece classified as two shapes. An image of base F under map M "
                      "(group.npz beta_maps) is F(M^-1 beta) = 0. 'family' groups surfaces that are "
                      "images of one another; 'evidence' lists the points that called for an image.",
            "surfaces": [dict(im.surface.to_json(), base=im.base, map_index=im.map_index,
                              in_source=im.in_source, family=im.family,
                              evidence=[list(map(float, e)) for e in im.evidence])
                         for im in self.images],
        }

    @classmethod
    def from_json(cls, data: dict) -> Library:
        out = []
        for d in data["surfaces"]:
            if d["kind"] == "plane":
                surf = Surface(d["index"], d["name"], "plane", np.asarray(d["normal"], float),
                               exact=d.get("exact"), aliases=tuple(d.get("aliases", ())))
            else:
                surf = Surface(d["index"], d["name"], "curved", None,
                               tuple(tuple(m) for m in d["monomials"]),
                               np.asarray(d["coefficients"], float), np.asarray(d["A"], float),
                               exact=d.get("exact"), degree=d["degree"],
                               aliases=tuple(d.get("aliases", ())))
            out.append(SurfaceImage(surf, d["base"], d["map_index"], d["in_source"], d["family"],
                                    tuple(tuple(e) for e in d.get("evidence", ()))))
        return cls(out)

    def save(self, path=None) -> None:
        path = path or data_dir() / "sphere_map" / "surface_library.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_json(), handle)
