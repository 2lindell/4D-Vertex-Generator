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

The page (`../../assets/tetraswirl_<group>_domain.html`) is laid out like the Symmetry
Domain page of the swirlprism atlas: `atlas_template.html` shares its styles, key and
controls, and `build_page.py` fills it from whatever has been gathered so far (the
domain from `domain.py`, the survey samples, and the traced walls, lines and points
once a later stage writes `atlas_<group>.json`; its format is in `build_page.py`).
Region ids are kept in `ids_<group>.json` once given.

```bash
SWIRL_GROUP=oct python survey.py 0.05 survey_oct.json
SWIRL_GROUP=oct python build_page.py
```

Status (Oct 2026): the octahedral domain is provisional (its normalizer is still to be
found); `survey_oct.json` was re-run on the present domain; no walls are traced yet.
The older commands below are kept from the first attempt.

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
