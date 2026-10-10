import numpy as np
from scipy.spatial import cKDTree
from supergroups import girdle_segments, supergroup
from cellframe import seed_from_beta
from symcheck import FULL
from four_d_vertex_generator.generation import generate_vertices_from_seed as g, group_elements
from four_d_vertex_generator.library import named_symmetry
G3=np.stack(supergroup(3)); 
H4=np.stack(group_elements(named_symmetry("h4")))
def count(V,E):
    T=cKDTree(V); return sum(T.query(V@M.T)[0].max()<1e-7 for M in E)
for a,b in girdle_segments():
    for p in (a,b,(a+b)/2):
        V=g(seed_from_beta(p),FULL,tol=1e-6)
        print(np.round(p/p[p>1e-9].min(),5), len(V), 'G3', count(V,G3), 'H4', count(V,H4))
