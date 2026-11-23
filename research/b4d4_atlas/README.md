# B4 and D4 symmetry domains

The maps behind the B4 and D4 Symmetry Domain pages (`assets/b4_domain.html`, `assets/d4_domain.html`), drawn by the
A4 page's viewer (`../a4_atlas/a4_template.html`).

- `b4d4_map.py`: the five groups of the tessic family, each on its chamber's fifteen faces (no group has walls inside
  the B4 chamber: `../tessic_atlas/tessic_subgroups.md`). Each face is classified by `../tessic_atlas/group.py`
  (TESSIC_SUB), checked at three more seeds, and named from the tessic list by the atlas's seeds; a shape whose
  subgroup orbit is the whole tessic one takes the tessic name (proh under tessipyritic).
  - B4 page: tessic, kitessic, tessipyritic, on the B4 chamber.
  - D4 page: demitessic, kidemitessic, on the D4 chamber: the B4 chamber and its mirror image in beta_4 = 0 (a fold,
    no D4 mirror), every shape drawn in both halves, and a piece joined to its image where it goes on across the fold.

```
python b4d4_map.py && cd ../a4_atlas && python build_page.py b4 d4
```
