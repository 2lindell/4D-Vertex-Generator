"""Read the swirlprism 3-sphere map: the data a viewer pulls from.

    from four_d_vertex_generator.sphere_map import SphereMap
    m = SphereMap.load()
    m.locate(point)                 # copy, piece, region, region component at any point of S^3
    m.domain_view(k)                # everything in copy k of the domain, and the map to place it
    m.region_component("F2")        # one whole region: its (copy, piece) members
    m.boundary("F2")                # its faces: which wall, which region across, which parts
    m.wall_component("T2")          # one whole wall sheet: its (copy, patch) members
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frame import data_dir


def _load(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class SphereMap:
    root: Path
    manifest: dict
    elements: np.ndarray            # (1200, 4, 4)
    cell_vertices: np.ndarray       # (4, 4)
    multiplication: np.ndarray      # (1200, 1200)
    inverse: np.ndarray             # (1200,)
    surfaces: list[dict]
    pieces: list[dict]
    patches: list[dict]
    regions: dict[str, dict]
    walls: dict[str, dict]
    lines: dict[str, dict]
    points: dict
    region_component_table: np.ndarray   # (1200, pieces)
    wall_component_table: np.ndarray     # (1200, patches)

    @classmethod
    def load(cls, root: str | Path | None = None) -> SphereMap:
        root = Path(root) if root else data_dir() / "sphere_map"
        g = np.load(root / "group.npz")
        c = np.load(root / "components.npz")
        return cls(
            root=root,
            manifest=_load(root / "manifest.json"),
            elements=g["elements"],
            cell_vertices=g["cell_vertices"],
            multiplication=g["multiplication"].astype(np.int64),
            inverse=g["inverse"].astype(np.int64),
            surfaces=_load(root / "surface_library.json")["surfaces"],
            pieces=_load(root / "pieces.json")["pieces"],
            patches=_load(root / "patches.json")["patches"],
            regions=_load(root / "regions.json")["regions"],
            walls=_load(root / "walls.json")["walls"],
            lines=_load(root / "lines.json")["lines"],
            points=_load(root / "points.json"),
            region_component_table=c["region_component"],
            wall_component_table=c["wall_component"],
        )

    # ------------------------------------------------------------------ coordinates and copies

    def point(self, beta, copy: int = 0) -> np.ndarray:
        """The point of S^3 with weights beta in the domain, placed in copy ``copy``."""
        p = np.asarray(beta, float) @ self.cell_vertices
        p = p / np.linalg.norm(p, axis=-1, keepdims=True)
        return p @ self.elements[copy].T

    def placement(self, copy: int) -> np.ndarray:
        """4x4 map taking beta (domain weights) to R^4 points of copy ``copy`` (before scaling)."""
        return self.elements[copy] @ self.cell_vertices.T

    def surface_values(self, beta) -> np.ndarray:
        b = np.atleast_2d(np.asarray(beta, float))
        b = b / b.sum(axis=1, keepdims=True)
        out = []
        for s in self.surfaces:
            if s["kind"] == "plane":
                out.append(b @ np.asarray(s["normal"]))
            else:
                y = b @ np.asarray(s["A"]).T
                out.append(sum(c * np.prod(y[:, m], axis=1)
                               for c, m in zip(s["coefficients"], s["monomials"])))
        return np.stack(out, axis=1)

    def piece_of_beta(self, beta) -> int | None:
        if not hasattr(self, "_by_signs"):
            self._by_signs = {p["signs"]: p["id"] for p in self.pieces}
        v = self.surface_values(beta)[0]
        key = "".join("+" if x > 0 else "-" if x < 0 else "0" for x in v)
        return self._by_signs.get(key)

    def locate(self, point) -> dict:
        """Copy, domain weights, piece, region and region component of a point of S^3."""
        p = np.asarray(point, float)
        p = p / np.linalg.norm(p)
        tinv = np.linalg.inv(self.cell_vertices)
        b = np.einsum("gji,j->gi", self.elements, p) @ tinv
        inside = (b > -1e-12).all(axis=1) & (b[:, 2] >= b[:, 3] - 1e-12)
        k = int(np.argmax(inside))
        beta = b[k] / b[k].sum()
        piece = self.piece_of_beta(beta)
        out = {"copy": k, "beta": beta, "piece": piece}
        if piece is not None:
            out["region"] = self.pieces[piece]["region"]
            out["region_component"] = int(self.region_component_table[k, piece])
        return out

    # ------------------------------------------------------------------------------- views

    def domain_view(self, copy: int) -> dict:
        """Everything in one copy of the domain: pieces with their region components, patches
        with their wall components, and the placement map."""
        return {
            "copy": copy,
            "placement": self.placement(copy),
            "pieces": [{"piece": p["id"], "region": p["region"],
                        "region_component": int(self.region_component_table[copy, p["id"]])}
                       for p in self.pieces],
            "patches": [{"patch": q["id"], "wall": q["wall"],
                         "wall_component": int(self.wall_component_table[copy, q["id"]])}
                        for q in self.patches],
        }

    def region_component(self, region: str, orbit: int = 0, element: int = 0) -> list[tuple]:
        """Members (copy, piece) of one whole component: the orbit's representative carried by
        group element ``element``."""
        rep = self.regions[region]["orbits"][orbit]["members"]
        return [(int(self.multiplication[element, k]), int(i)) for k, i in rep]

    def components_of(self, region: str, orbit: int = 0) -> list[int]:
        """Group elements carrying the representative onto each distinct component (cosets)."""
        o = self.regions[region]["orbits"][orbit]
        k0, i0 = o["members"][0]
        seen, reps = set(), []
        for g in range(len(self.elements)):
            cid = int(self.region_component_table[self.multiplication[g, k0], i0])
            if cid not in seen:
                seen.add(cid)
                reps.append(g)
        return reps

    def boundary(self, region: str, orbit: int = 0) -> list[dict]:
        return self.regions[region]["orbits"][orbit]["boundary"]

    def wall_component(self, wall: str, orbit: int = 0, element: int = 0) -> list[tuple]:
        rep = self.walls[wall]["orbits"][orbit]["members"]
        return [(int(self.multiplication[element, k]), int(i)) for k, i in rep]
