# A4 symmetry domain viewer

One page (`../../assets/a4_domain.html`, the app's "A4 Symmetry Domain" tab) for the A4 symmetries, laid out like the
swirl atlases' Symmetry Domain pages: choose a symmetry and its map of the A4 chamber (the tetrahedron between the
four mirrors of [3,3,3]) is drawn, with its key of regions, walls, lines and points.

- `a4_map.py` builds each symmetry's 3-sphere map (`a4_map_<group>.json`), in the gnomonic chart at the chamber's centre:
  - pennic [3,3,3]: the chamber's fifteen faces (the Wythoffian polychora of the 5-cell family);
  - decaic [[3,3,3]]: the exact plane arrangement of `../decaic_atlas/arrangement.json`, every face classified
    (`../decaic_atlas/shapes.py`, names from `decaic_shapes.json`), carried by the swap onto the chamber's other half.
- `build_page.py` fills `a4_template.html` with every map built so far. Kipennic, kidecaic and iodecaic are listed
  but not mapped yet: their maps have curved walls (`../decaic_atlas/snub_map.py`).

Each example seed is s = R⁻¹β (R the generator's simple roots for A4), so it pastes into the generator's Coordinates
mode with the same symmetry; every one is checked to give its polychoron's vertex and cell counts there.

```bash
python a4_map.py           # pennic and decaic
python build_page.py
python ../../tools/build_pages.py
```
