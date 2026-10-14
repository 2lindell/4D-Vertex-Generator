# Tetrahedral and octahedral swirl atlases

This directory maps the combinatorial type of the convex hull of every generic
orbit seed in one gnomonic fundamental domain for two tubical symmetry groups:

- `tet`: the index-3 tetrahedral group
  `h4_tubical_tetrahedral_cyclic_index3_1` (order 24).
- `oct`: the index-6 octahedral group
  `h4_tubical_octahedral_dihedral_index6_6n_1` (order 48).

`survey.py` samples and classifies a domain grid; `dense.py` independently
samples random interior points; and `build2.py` bisects transitions, fits their
coplanarity surfaces, and builds region and wall meshes. `emit.py` converts
that build into compact JSON, and `render_atlas.py` embeds the JSON into the
interactive static page used by GitHub Pages.

Run these commands from this directory. The final `data_<group>.pkl` files are
large, local build intermediates and are deliberately not committed.

```bash
SWIRL_GROUP=tet python survey.py 0.021 survey_tet.json
SWIRL_GROUP=oct python survey.py 0.021 survey_oct.json
SWIRL_GROUP=tet python dense.py 60000 17 dense_tet_60k.json
SWIRL_GROUP=oct python dense.py 60000 17 dense_oct_60k.json

SWIRL_GROUP=tet python build2.py survey_tet.json data_tet
SWIRL_GROUP=tet python emit.py data_tet.pkl out_tet.json
SWIRL_GROUP=tet python render_atlas.py out_tet.json ../../assets/tetraswirl_tet_domain.html

SWIRL_GROUP=oct python build2.py survey_oct.json data_oct
SWIRL_GROUP=oct python emit.py data_oct.pkl out_oct.json
SWIRL_GROUP=oct python render_atlas.py out_oct.json ../../assets/tetraswirl_oct_domain.html

python ../../tools/build_pages.py
```

Validate a mesh classification against direct hull computation with:

```bash
SWIRL_GROUP=tet python check.py data_tet.pkl
SWIRL_GROUP=oct python check.py data_oct.pkl
```
