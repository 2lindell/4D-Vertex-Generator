import json

import numpy as np
import pytest

from four_d_vertex_generator.sphere_map import frame
from four_d_vertex_generator.sphere_map.classify import counts_match, signature_of_beta

MAP_DIR = frame.data_dir() / "sphere_map"
needs_map = pytest.mark.skipif(not (MAP_DIR / "manifest.json").exists(),
                               reason="sphere map not built")


def test_half_cell_copies_tile_the_sphere_once() -> None:
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(500, 4))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    g = frame.group()
    tinv = np.linalg.inv(frame.cell_vertices())
    b = np.einsum("gji,nj->gni", g, pts) @ tinv
    inside = (b > -1e-12).all(-1) & (b[..., 2] >= b[..., 3] - 1e-12)
    assert (inside.sum(axis=0) == 1).all()


def test_point_beta_round_trip() -> None:
    beta = np.array([0.4, 0.1, 0.3, 0.2])
    assert np.allclose(frame.beta_from_point(frame.point_from_beta(beta)), beta)


def test_region_seed_gives_recorded_counts() -> None:
    shapes = frame.load_json("working_data.json")["shapes"]
    assert counts_match("F2", signature_of_beta(shapes["F2"]["seed"]))


def test_duplicate_planes_merged() -> None:
    names = frame.surface_by_name()
    s = frame.surfaces()
    assert len(set(names.values())) == len(s)
    assert any(x.aliases for x in s)


@needs_map
def test_map_regions_cover_the_sphere() -> None:
    from four_d_vertex_generator.sphere_map import SphereMap

    m = SphereMap.load()
    total = sum(r["volume_share"] for r in m.regions.values())
    assert total == pytest.approx(1.0, abs=2e-3)
    shapes = frame.load_json("working_data.json")["shapes"]
    assert set(m.regions) == {k for k, v in shapes.items() if v["dim"] == "region"}
    for r in m.regions.values():
        for o in r["orbits"]:
            assert o["components"] * o["stabilizer_order"] == 1200


@needs_map
def test_locate_matches_classifier() -> None:
    from four_d_vertex_generator.sphere_map import SphereMap

    m = SphereMap.load()
    shapes = frame.load_json("working_data.json")["shapes"]
    g = frame.group()
    for name in ("F2", "F3", "X49"):
        p = g[17] @ frame.point_from_beta(np.asarray(shapes[name]["seed"], float))
        assert m.locate(p)["region"] == name


@needs_map
def test_region_component_is_carried_by_the_group() -> None:
    from four_d_vertex_generator.sphere_map import SphereMap

    m = SphereMap.load()
    members = m.region_component("F2", element=5)
    cids = {int(m.region_component_table[k, i]) for k, i in members}
    assert len(cids) == 1


@needs_map
def test_manifest_counts_every_region() -> None:
    manifest = json.loads((MAP_DIR / "manifest.json").read_text())
    shapes = frame.load_json("working_data.json")["shapes"]
    assert manifest["counts"]["regions"] == sum(v["dim"] == "region" for v in shapes.values())


def test_vertex_key_splits_shapes_with_equal_counts() -> None:
    from four_d_vertex_generator.sphere_map.classify import key_of_beta, split_name

    shapes = frame.load_json("working_data.json")["shapes"]
    for name in ("X72", "X102"):
        _, key = key_of_beta(shapes[name]["seed"])
        assert split_name(name, key) == name


@needs_map
def test_atlas_database_is_closed_for_faces_and_regions() -> None:
    path = MAP_DIR / "atlas.json"
    if not path.exists():
        pytest.skip("atlas.json not built")
    db = json.loads(path.read_text())
    c = db["closure"]
    assert c["faces_with_two_region_sides"] == c["faces"]
    assert c["regions_with_faces"] == c["regions"]
    walls = {k for k, v in frame.load_json("working_data.json")["shapes"].items()
             if v["dim"] == "wall"}
    assert set(db["counts_per_wall"]) == walls
