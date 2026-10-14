"""Sample the special planes through the centre that the volume grid straddles."""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from survey import in_cell, work

PLANES = {"az0": 0.0, "az36": 36.0}  # vertical planes through the main ring, by azimuth


def plane_points(spacing: float) -> list[tuple[str, tuple[float, float, float]]]:
    pts = []
    radii = np.arange(0, 0.45 + 1e-9, spacing)
    heights = np.arange(0, 0.45 + 1e-9, spacing)  # z < 0 is a copy under a half-turn
    for name, az in PLANES.items():
        direction = np.array([np.cos(np.radians(az)), np.sin(np.radians(az))])
        for s in radii:
            for z in heights:
                q = (round(float(s * direction[0]), 6), round(float(s * direction[1]), 6), round(float(z), 6))
                if in_cell(np.array(q)):
                    pts.append((name, q))
    for x in np.arange(-0.45, 0.45 + 1e-9, spacing):  # horizontal plane z = 0, wedge 0°–72°
        for y in np.arange(0, 0.45 + 1e-9, spacing):
            az = np.degrees(np.arctan2(y, x)) % 360
            q = (round(float(x), 6), round(float(y), 6), 0.0)
            if 0 < az < 72 and in_cell(np.array(q)):
                pts.append(("z0", q))
    return pts


def job(item: tuple[str, tuple[float, float, float]]) -> dict:
    name, q = item
    result = work(q)
    result["plane"] = name
    return result


if __name__ == "__main__":
    pts = plane_points(float(sys.argv[1]))
    out_path = sys.argv[2]
    print(len(pts), "plane points", flush=True)
    results: list[dict] = []
    t0 = time.time()
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(job, pts, chunksize=4)):
            results.append(r)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(pts)} {time.time() - t0:.0f}s", flush=True)
                json.dump(results, open(out_path, "w"))
    json.dump(results, open(out_path, "w"))
    print("done", flush=True)
