"""Sample one H4 mirror plane from each class that crosses the dodecahedron."""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from survey import in_cell, work

REPS = {  # plane normals in gnomonic coordinates (all pass through the centre)
    "mirror_vertical": [-0.309017, 0.951057, 0.0],     # contains the main ring, azimuth 18°
    "mirror_tilt_a": [0.0, 0.525731, 0.850651],
    "mirror_tilt_b": [0.5, 0.688191, 0.525731],
}

def plane_points(spacing):
    pts = []
    for name, n in REPS.items():
        n = np.array(n) / np.linalg.norm(n)
        u = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0]); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        for s in np.arange(-0.45, 0.45 + 1e-9, spacing):
            for t in np.arange(-0.45, 0.45 + 1e-9, spacing):
                q = s * u + t * v
                if in_cell(q): pts.append((name, tuple(round(float(c), 7) for c in q)))
    return pts

def job(item):
    name, q = item; r = work(q); r["plane"] = name; return r

if __name__ == "__main__":
    pts = plane_points(float(sys.argv[1])); print(len(pts), "mirror-plane points", flush=True)
    out, t0 = [], time.time()
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(job, pts, chunksize=4)):
            out.append(r)
            if (i + 1) % 250 == 0:
                print(f"{i + 1}/{len(pts)} {time.time() - t0:.0f}s", flush=True); json.dump(out, open(sys.argv[2], "w"))
    json.dump(out, open(sys.argv[2], "w")); print("done", flush=True)
