import pickle,sys,numpy as np
import build2 as B, domain as D
r=pickle.load(open(sys.argv[1],'rb'))
pts=B.random_points(500)
lg=r['lab_grid']; tid=r['tid']
ok=bad=0
for p in pts:
    idx=np.clip(np.round((p-r['lo'])/r['h']).astype(int),0,np.array(lg.shape)-1)
    L=lg[tuple(idx)]
    if L==0: continue
    t=tid[r['regions'][L-1]['type_sig']]
    s=tid.get(B.sig(p),-1)
    if s==t: ok+=1
    else: bad+=1
print(ok,bad)
