"""How far the pentagon of a regular decagon's cell is turned against the decagon, along the decagon lines.

twist(beta) lists, for each cell with a decagon and a pentagon face (both turned onto themselves by the same
5-fold rotation R), the angle in R's turning plane from the decagon's vertices to the pentagon's, modulo 36 degrees.
0 or 18 is an untwisted (mirror-symmetric) cell; 18 lines the pentagon's edges up with the decagon's, as in a cupola.

    python decagon_twist.py lo hi n     (scan the dashed decagon line by its sample index)
"""
import collections
import json
import sys
from multiprocessing import Pool

import numpy as np
from cellframe import seed_from_beta, TINV
from dodeca_view import E
from regular_cells import hull
FIVES=[]
for R in E:
    u,s,vt=np.linalg.svd(R-np.eye(4))
    if np.sum(s>1e-9)==2 and np.allclose(np.linalg.matrix_power(R,5),np.eye(4),atol=1e-9):
        W=vt[:2]; a=W@(R@W[0]); 
        if abs(np.degrees(np.arctan2(a[1],a[0]))-72)<1e-6: FIVES.append((R,W))
def twist(beta):
    v,faces,cells=hull(beta)
    out=[]
    for ci,c in enumerate(cells):
        sizes=[len(faces[f]) for f in c]
        if 10 not in sizes or 5 not in sizes: continue
        dec=v[faces[[f for f in c if len(faces[f])==10][0]]]
        pents=[v[faces[f]] for f in c if len(faces[f])==5]
        for R,W in FIVES:
            if all(np.linalg.norm(dec-(R@p),axis=1).min()<1e-7 for p in dec): break
        else: continue
        ad=np.degrees(np.arctan2(*(dec@W.T)[:, ::-1].T))
        for pen in pents:
            if not all(np.linalg.norm(pen-(R@p),axis=1).min()<1e-7 for p in pen): continue
            ap=np.degrees(np.arctan2(*(pen@W.T)[:, ::-1].T))
            off=min(((x-y)%36 for x in ap for y in ad))
            out.append((ci, round(off,9), collections.Counter(sizes)))
    return out
L=json.load(open("regular_decagons.json")); S=np.array(L[2]["seeds"])
def at(t):
    i=min(int(t),len(S)-2); x=S[i]+(t-i)*(S[i+1]-S[i]); b=TINV@x; return b/b.sum()
def job(t):
    b=at(t)
    try: tw=twist(b)
    except Exception: return t, None
    return t, (sorted({z[1] for z in tw}) if tw else None)

if __name__ == "__main__":
    lo,hi,n=map(float,sys.argv[1:4])
    with Pool(4) as p:
        for t,o in p.map(job, np.linspace(lo,hi,int(n))):
            b=at(t); print(round(t,5), np.round(b/b.max(),6).tolist(), o)
