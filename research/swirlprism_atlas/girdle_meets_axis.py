"""Where the order-3 ghost girdle (green) crosses a 2400-symmetry axis (purple): the seed there has 7200
symmetries. Prints the crossing of every girdle/axis pair and writes examples/X12_girdle_meets_axis_swirlprism_1200.off."""
import numpy as np
from normalizer import coset_axes
from supergroups import girdle_families, girdle_segments

fam=girdle_families()
G=girdle_segments()
for a,b in G:
    for c,d in coset_axes():
        # solve a+s(b-a) = λ(c+t(d-c)) homogeneous: beta rays; use normalized sums=1 so lines in simplex
        M=np.column_stack([b-a, -(d-c)]); rhs=c-a
        st,res,_,_=np.linalg.lstsq(M,rhs,rcond=None); s,t=st
        p=a+s*(b-a); q=c+t*(d-c); gap=np.linalg.norm(p-q)
        print(f"s={s:.6f} t={t:.6f} gap={gap:.2e} fam={fam(c,d)}", np.round(p/p[p>1e-9].min(),9))

from cellframe import seed_from_beta
from classify import classify, signature
from probe import _one
from scipy.spatial import cKDTree
from supergroups import supergroup

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.library import named_symmetry
from four_d_vertex_generator.off import compute_convex_hull, to_4off

a,b=G[0]; c,d=coset_axes()[0]
M=np.column_stack([b-a,-(d-c)]); s,t=np.linalg.lstsq(M,c-a,rcond=None)[0]
p=a+s*(b-a); p=p/p.sum()
x=seed_from_beta(p)
print("label", _one(p)[0]); info=classify(x); print(signature(info), info["valence"])
V=generate_vertices_from_seed(x, named_symmetry("h4_swirlprism"), tol=1e-9)
T=cKDTree(V)
for k in (2,3,6):
    E=np.stack(supergroup(k)); n=sum(T.query(V@g.T)[0].max()<1e-7 for g in E); print(f"G_{k} ({len(E)}): {n} preserve")
faces,cells=compute_convex_hull(V,tol=1e-9)
open("examples/X12_girdle_meets_axis_swirlprism_1200.off","w").write(to_4off(V,faces,cells))
print("beta", repr(p.tolist())); print("seed", ", ".join(f"{v:.17g}" for v in x))
print(len(V), len(faces), len(cells))
