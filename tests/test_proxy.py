import pytest
from app.geometry.proxy import build_proxy


def setup(semantic, paths, **properties):
    state = {"source": {"units":"mm"}, "entities":{"entity_1":{"semantic":semantic, **properties}}}
    geometry = [{"id":"entity_1", "paths":paths}]
    return state, geometry


@pytest.mark.parametrize("semantic,height", [("wall",2800),("furniture",750),("furniture",820)])
def test_actual_concave_footprint_and_height(semantic, height):
    outline = [[0,0],[2000,0],[2000,500],[500,500],[500,1500],[0,1500],[0,0]]
    proxy = build_proxy(*setup(semantic,[outline],height=height,base_elevation=100))
    assert proxy["bounds"] == [[0,0,100],[2000,1500,100+height]]
    assert len(proxy["meshes"][0]["outer"]) == 7
    assert proxy["meshes"][0]["dimension_source"] == "cad_footprint"


def test_floor_hole_and_elevation():
    outer = [[0,0],[100,0],[100,100],[0,100],[0,0]]
    hole = [[20,20],[80,20],[80,80],[20,80],[20,20]]
    proxy = build_proxy(*setup("floor",[hole,outer],elevation=100,thickness=150))
    assert proxy["bounds"] == [[0,0,-50],[100,100,100]]
    assert len(proxy["meshes"][0]["holes"]) == 1


def test_open_walls_thickness_is_explicit():
    state, geometry = setup("wall", [[[0,0],[1000,0]]], height=2800)
    proxy = build_proxy(state, geometry)
    assert proxy["meshes"][0]["kind"] == "surface"
    assert proxy["bounds"] == [[0,0,0],[1000,0,2800]]
    state["proxy_settings"] = {"line_wall_thickness":150}
    proxy = build_proxy(state,geometry)
    assert proxy["bounds"] == [[0,-75,0],[1000,75,2800]]
    assert proxy["meshes"][0]["dimension_source"] == "explicit_proxy_thickness"


def test_invalid_and_open_footprints_and_void_are_not_boxes():
    for semantic in ("floor","furniture","void"):
        proxy = build_proxy(*setup(semantic,[[[0,0],[100,0],[100,100]]],height=800,thickness=150))
        assert not proxy["meshes"] and proxy["warnings"]
    proxy = build_proxy(*setup("floor",[[[0,0],[100,100],[100,0],[0,100],[0,0]]],thickness=150))
    assert not proxy["meshes"] and "invalid" in proxy["warnings"][0]


def test_furniture_clustering_subobjects():
    # 2 sub-objects (e.g. a table and a separate chair) composed of open adjoining line segments
    table_lines = [[[0,0],[1000,0]], [[1000,0],[1000,800]], [[1000,800],[0,800]], [[0,800],[0,0]]]
    chair_lines = [[[1500,200],[1900,200]], [[1900,200],[1900,600]], [[1900,600],[1500,600]], [[1500,600],[1500,200]]]
    state, geometry = setup("furniture", [*table_lines, *chair_lines], height=750)
    proxy = build_proxy(state, geometry)
    assert len(proxy["meshes"]) == 2
    assert all(m["kind"] == "solid" for m in proxy["meshes"])
    assert all(m["dimension_source"] == "cad_footprint" for m in proxy["meshes"])
    assert proxy["bounds"] == [[0,0,0],[1900,800,750]]


def test_furniture_oriented_box_mode():
    table_lines = [[[0,0],[1000,0]], [[1000,0],[1000,800]], [[1000,800],[0,800]], [[0,800],[0,0]]]
    chair_lines = [[[1500,200],[1900,200]], [[1900,200],[1900,600]], [[1900,600],[1500,600]], [[1500,600],[1500,200]]]
    state, geometry = setup("furniture", [*table_lines, *chair_lines], height=750)
    state["proxy_settings"] = {"furniture_style": "box"}
    proxy = build_proxy(state, geometry)
    assert len(proxy["meshes"]) == 2
    assert all(m["dimension_source"] == "oriented_box" for m in proxy["meshes"])


def test_automatic_ceiling_from_max_wall_height():
    floor_poly = [[0,0],[5000,0],[5000,4000],[0,4000],[0,0]]
    wall_poly = [[0,0],[5000,0],[5000,200],[0,200],[0,0]]
    state = {
        "source": {"units": "mm"},
        "entities": {
            "floor_1": {"semantic": "floor", "elevation": 0, "thickness": 150},
            "wall_1": {"semantic": "wall", "height": 2800, "base_elevation": 0},
            "wall_2": {"semantic": "wall", "height": 3000, "base_elevation": 0}
        },
        "proxy_settings": {"auto_ceiling": True}
    }
    geometry = [
        {"id": "floor_1", "paths": [floor_poly]},
        {"id": "wall_1", "paths": [wall_poly]},
        {"id": "wall_2", "paths": [wall_poly]}
    ]
    proxy = build_proxy(state, geometry)
    ceiling_meshes = [m for m in proxy["meshes"] if m["semantic"] == "ceiling"]
    assert len(ceiling_meshes) == 1
    assert ceiling_meshes[0]["z_min"] == 3000
    assert ceiling_meshes[0]["z_max"] == 3050
    assert ceiling_meshes[0]["outer"] == floor_poly

    state["proxy_settings"]["auto_ceiling"] = False
    proxy_no_ceil = build_proxy(state, geometry)
    assert not any(m["semantic"] == "ceiling" for m in proxy_no_ceil["meshes"])


def test_void_offset_0_5cm():
    # In meters: 0.5 cm = 0.005 m
    state_m = {
        "source": {"units": "m"},
        "entities": {
            "win_1": {"semantic": "void", "category": "window", "sill_height": 0.5, "opening_height": 1.5}
        }
    }
    geom_m = [{"id": "win_1", "paths": [[[0.5, 1.0], [0.7, 1.0], [0.7, 3.0], [0.5, 3.0], [0.5, 1.0]]]}]
    proxy_m = build_proxy(state_m, geom_m)
    assert len(proxy_m["meshes"]) == 1
    m = proxy_m["meshes"][0]
    xs = [p[0] for p in m["outer"]]
    ys = [p[1] for p in m["outer"]]
    assert min(xs) == pytest.approx(0.495)
    assert max(xs) == pytest.approx(0.705)
    assert min(ys) == pytest.approx(0.995)
    assert max(ys) == pytest.approx(3.005)

    # In mm: 0.5 cm = 5 mm
    state_mm = {
        "source": {"units": "mm"},
        "entities": {
            "win_2": {"semantic": "void", "category": "window", "sill_height": 500, "opening_height": 1500}
        }
    }
    geom_mm = [{"id": "win_2", "paths": [[[500, 1000], [700, 1000], [700, 3000], [500, 3000], [500, 1000]]]}]
    proxy_mm = build_proxy(state_mm, geom_mm)
    assert len(proxy_mm["meshes"]) == 1
    m2 = proxy_mm["meshes"][0]
    xs2 = [p[0] for p in m2["outer"]]
    ys2 = [p[1] for p in m2["outer"]]
    assert min(xs2) == pytest.approx(495.0)
    assert max(xs2) == pytest.approx(705.0)
    assert min(ys2) == pytest.approx(995.0)
    assert max(ys2) == pytest.approx(3005.0)



