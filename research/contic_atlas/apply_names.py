"""Rename the contic / kicontic / iocontic optimized and central .off files and their JSON entries to the accepted
names (given_names.py), without re-running the optimisers. The atlas's own name is kept as "atlas_name".

    python apply_names.py
"""
from __future__ import annotations

import json
import os

import given_names

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FOLDER = {"": "contic", "_ki": "kicontic", "_io": "iocontic"}
SUB = {"": "full", "_ki": "ki", "_io": "io"}


def new_name(tag, old):
    if tag == "":
        return given_names.contic(old[:-5]) if old.endswith("-like") else old
    return given_names.snub(SUB[tag], old)


def main():
    for tag, folder in FOLDER.items():
        for stem, top, suffix in (("optimized_seeds", "optimized_off", "_opt"), ("central_seeds", "central_off", "_central")):
            path = os.path.join(HERE, f"{stem}{tag}.json")
            if not os.path.exists(path):
                continue
            d = json.load(open(path))
            for r in d["shapes"]:
                old = r.get("atlas_name", r["name"])
                new = new_name(tag, old)
                r["atlas_name"], r["name"] = old, new
                if "off" not in r:
                    continue
                src = os.path.join(ROOT, r["off"])
                fname = f"{new}_{r['vertices']}v_{r['cells']}c{suffix}.off"
                dst = os.path.join(ROOT, top, folder, fname)
                if os.path.exists(src) and src != dst:
                    os.rename(src, dst)
                r["off"] = os.path.relpath(dst, ROOT)
                print(f"{folder:9} {old:28} -> {new}")
            json.dump(d, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
