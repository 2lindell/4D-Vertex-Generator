import json
from multiprocessing import Pool

import numpy as np
from t2trace import trace

T = json.load(open("t2trace.json"))
jobs = []
for tr in T:
    if tr["status"] == "max steps":
        p = np.array(tr["path"]); t = (p[-1] - p[-2]) / np.linalg.norm(p[-1] - p[-2])
        jobs.append((tr["name"], p[-1].tolist(), t.tolist(), 0.004, 80))
with Pool(2) as pool:
    for name, path, status, last in pool.imap_unordered(trace, jobs):
        for tr in T:
            if tr["name"] == name: tr["path"] += path[1:]; tr["status"] = status
        L = sum(np.linalg.norm(np.subtract(path[i + 1], path[i])) for i in range(len(path) - 1))
        print(f"{name}: extended by {L:.4f}, stopped: {status} (last label {last}), end {np.round(path[-1], 4)}", flush=True)
        json.dump(T, open("t2trace.json", "w"))
