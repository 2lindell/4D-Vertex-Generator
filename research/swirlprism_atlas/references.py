"""Reference signatures (with edge valences) for every shape identified so far, from exact seeds."""
import json

import numpy as np
from classify import classify, signature

from four_d_vertex_generator.library import h4_swirlprism_predefined_seed as ps

PHI = (1 + 5 ** 0.5) / 2
AT = lambda x: float(np.degrees(np.arctan(x)))
ring = lambda t: np.array([np.cos(np.radians(t)), 0.0, np.sin(np.radians(t)), 0.0])
REFS = {
    "Hexacosichoron (600-cell)": ring(0),
    "Hecatonicosachoron (120-cell)": ring(45),
    "Partially-rectified small swirlprism": ring(AT(PHI ** -2)),
    "Swirlprismatodiminished rectified hexacosichoron": ring(AT(1 / PHI)),
    "Subsymmetrical icosafold icosidodecaswirlchoron": ring(AT(PHI) / 2),
    "Bigyroprismatic transitional didecafold icosidodecaswirlchoron": ring(AT(PHI ** 2)),
    "Cross ring: 120+600+600+600+1200 cells (1200 3-valent edges)": ring(6.0),
    "Cross ring: Pentagonal-gyroprismatic triacosihexecontachoron": ring(17.0),
    "Cross ring: 120+120+1200 cells (600+1200 4-valent edges)": ring(23.0),
    "Cross ring: antiprisms split 120+120 (not in the wiki list)": ring(28.0),
    "Cross ring: 120+600+600+600+1200 cells (600 3-valent edges)": ring(63.0),
    "Cross ring: tetrahedra split 600+600 (not in the wiki list)": ring(72.0),
    "Icosafold icosaswirlchoron (main ring, 240)": ps(0, 0, 9.0),
}
def sig(p):
    info = classify(p)
    return signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items())
if __name__ == "__main__":
    out = {}
    for name, p in REFS.items():
        s = sig(p); out[s] = name; print(f"{name}\n   {s}")
    json.dump(out, open("references.json", "w"), indent=1)
