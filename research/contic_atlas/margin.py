import json, numpy as np, sys
sys.path.insert(0,'.')
import group as G
def margin(b):
    V5, V, faces, cells = G.polytope(b)
    m = np.inf
    for c in cells:
        vs = sorted({v for f in c for v in faces[f]})
        P = V[vs]
        n = np.linalg.svd(P[1:] - P[0])[2][-1]
        d = (V - P[0]) @ n
        if d[np.setdiff1d(np.arange(len(V)), vs)].max() > 0: d = -d
        others = np.setdiff1d(np.arange(len(V)), vs)
        m = min(m, -d[others].max())
    return m
d = json.load(open(G.fname('snub_map')))
for t in d['types']:
    if t['grid_regions']:
        print(t['type'], t['grid_regions'], 'margin %.2e' % margin(t['example']))
