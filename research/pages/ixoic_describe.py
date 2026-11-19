"""Describe every ixoic shape (merged, crossing-search, fixed-line and closure shapes) with describe.py, cached in
ixoic_described.json (key -> describe.py's record plus kind, sheet and seed). Used by ixoic_suggest.py.

    python ixoic_describe.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ixoic_match import atlas  # noqa: E402

PATH = os.path.join(HERE, "ixoic_described.json")


def main():
    A = atlas()
    C = json.load(open(PATH)) if os.path.exists(PATH) else {}
    by = {}
    for a in A:
        if a["key"] not in C:
            by.setdefault(a["sheet"], []).append(a)

    def one(sub):
        L = by[sub]
        items = [[a["key"], a["seed"], 1e-9 if a["kind"] == "region" else 1e-7] for a in L]
        r = subprocess.run([sys.executable, os.path.join(HERE, "describe.py"), json.dumps(items)], capture_output=True,
                           text=True, env={**os.environ, "HYIC_SUB": sub})
        if r.returncode:
            print(sub, "FAILED", r.stderr[-600:], flush=True)
            return {}
        D = json.loads(r.stdout)
        return {a["key"]: {**D[a["key"]], "kind": a["kind"], "sheet": sub, "seed": a["seed"], "source": a["source"]}
                for a in L if a["key"] in D}

    with ThreadPoolExecutor(4) as ex:
        for sub, o in zip(by, ex.map(one, list(by))):
            C.update(o)
            json.dump(C, open(PATH, "w"), indent=1)
            print(f"{sub}: {len(o)} described", flush=True)


if __name__ == "__main__":
    main()
