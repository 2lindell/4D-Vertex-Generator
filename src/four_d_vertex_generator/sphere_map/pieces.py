"""Pieces of the half-cell: cells of the surface arrangement, found along chords.

A piece is the set of domain points with one sign vector over the library surfaces. The chords give
every piece they pass through, which pieces meet across which surface (and where), and which pieces
reach which face of the domain (and where).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .chords import FACE_NAMES, cast, random_directions, random_domain_points


@dataclass
class Exploration:
    keys: dict[bytes, int] = field(default_factory=dict)        # sign vector -> piece id
    signs: list[np.ndarray] = field(default_factory=list)
    hits: list[int] = field(default_factory=list)               # chord intervals seen per piece
    # the longest chord intervals through each piece: (length, midpoint), longest first
    samples: list[list[tuple[float, np.ndarray]]] = field(default_factory=list)
    keep_samples: int = 4
    # (piece a, piece b, surface) -> [count, sample crossing point]
    adjacency: dict[tuple[int, int, int], list] = field(default_factory=dict)
    # (piece, face index) -> [count, sample point]
    face_hits: dict[tuple[int, int], list] = field(default_factory=dict)
    face_points: dict[tuple[int, int], list] = field(default_factory=lambda: defaultdict(list))
    chords: int = 0
    rejected: int = 0
    multi_crossings: int = 0

    def piece(self, sign: np.ndarray) -> int:
        k = sign.tobytes()
        pid = self.keys.get(k)
        if pid is None:
            pid = len(self.signs)
            self.keys[k] = pid
            self.signs.append(sign.copy())
            self.hits.append(0)
            self.samples.append([])
        return pid

    def offer(self, pid: int, length: float, point: np.ndarray) -> None:
        self.hits[pid] += 1
        lst = self.samples[pid]
        if len(lst) < self.keep_samples or length > lst[-1][0]:
            lst.append((length, point))
            lst.sort(key=lambda x: -x[0])
            del lst[self.keep_samples:]

    @property
    def best(self) -> list[tuple[float, np.ndarray]]:
        return [lst[0] for lst in self.samples]


def explore(n_chords: int, seed: int = 0, batch: int = 5000, ex: Exploration | None = None,
            keep_face_points: int = 12, surfs=None) -> Exploration:
    rng = np.random.default_rng(seed)
    ex = ex or Exploration()
    done = 0
    while done < n_chords:
        m = min(batch, n_chords - done)
        r = cast(random_domain_points(m, rng), random_directions(m, rng), surfs)
        done += m
        ex.chords += m
        starts = np.searchsorted(r.chord, np.arange(m + 1))
        span = np.linalg.norm(r.end_beta - r.start_beta, axis=1)
        for c in range(m):
            if not r.valid[c]:
                ex.rejected += 1
                continue
            lo, hi = starts[c], starts[c + 1]
            us = np.concatenate([[0.0], r.u[lo:hi], [1.0]])
            sign = r.start_signs[c].copy()
            prev = ex.piece(sign)
            a, b = r.start_beta[c], r.end_beta[c]
            ex.offer(prev, (us[1] - us[0]) * span[c], a + (us[0] + us[1]) / 2 * (b - a))
            key = (prev, int(r.start_face[c]))
            _face(ex, key, a, keep_face_points)
            j = lo
            while j < hi:
                # crossings at the same spot (surfaces meeting on the chord) are taken together
                k = j
                while k + 1 < hi and r.u[k + 1] - r.u[j] < 1e-12:
                    k += 1
                for q in range(j, k + 1):
                    sign[r.surface[q]] = -sign[r.surface[q]]
                cur = ex.piece(sign)
                if k == j:
                    s = int(r.surface[j])
                    akey = (min(prev, cur), max(prev, cur), s)
                    rec = ex.adjacency.get(akey)
                    if rec is None:
                        ex.adjacency[akey] = [1, r.beta[j].copy()]
                    else:
                        rec[0] += 1
                else:
                    ex.multi_crossings += 1
                i0 = k - lo + 1
                ex.offer(cur, (us[i0 + 1] - us[i0]) * span[c],
                         a + (us[i0] + us[i0 + 1]) / 2 * (b - a))
                prev = cur
                j = k + 1
            _face(ex, (prev, int(r.end_face[c])), b, keep_face_points)
    return ex


def _face(ex: Exploration, key: tuple[int, int], point: np.ndarray, keep: int) -> None:
    rec = ex.face_hits.get(key)
    if rec is None:
        ex.face_hits[key] = [1, point.copy()]
    else:
        rec[0] += 1
    pts = ex.face_points[key]
    if len(pts) < keep:
        pts.append(point.copy())


def face_name(i: int) -> str:
    return FACE_NAMES[i]
