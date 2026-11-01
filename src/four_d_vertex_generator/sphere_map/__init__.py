"""The swirlprism 3-sphere map: every region, wall, line and point of the h4_swirlprism atlas,
carried from the half-cell domain to all 1200 copies that tile S^3.

Build with ``python -m four_d_vertex_generator.sphere_map.build``; read with ``SphereMap.load()``.
"""
from .api import SphereMap

__all__ = ["SphereMap"]
