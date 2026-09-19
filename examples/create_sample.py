"""Generate an explicitly synthetic sample DXF; never added to the user's workspace."""
from pathlib import Path
import ezdxf


def main():
    doc = ezdxf.new("R2010")
    doc.units = 4
    model = doc.modelspace()
    model.add_lwpolyline([(0, 0), (6000, 0), (6000, 4000), (0, 4000)], close=True, dxfattribs={"layer": "FLOOR_OUTLINE"})
    model.add_lwpolyline([(-150, -150), (6150, -150), (6150, 4150), (-150, 4150)], close=True, dxfattribs={"layer": "WALL_OUTLINE"})
    sofa = doc.blocks.new("SOFA_A01")
    sofa.add_lwpolyline([(0, 0), (2100, 0), (2100, 850), (0, 850)], close=True)
    sofa.add_line((700, 0), (700, 850))
    sofa.add_line((1400, 0), (1400, 850))
    model.add_blockref("SOFA_A01", (700, 2600), dxfattribs={"layer": "FURNITURE"})
    model.add_lwpolyline([(1100, 1300), (2300, 1300), (2300, 2100), (1100, 2100)], close=True, dxfattribs={"layer": "FURNITURE"})
    model.add_circle((4500, 2500), 600, dxfattribs={"layer": "FURNITURE"})
    model.add_arc((5000, 0), 850, 0, 90, dxfattribs={"layer": "DOOR_SWING"})
    model.add_line((5000, 0), (5000, 850), dxfattribs={"layer": "DOOR_SWING"})
    model.add_polyline2d([(3700, 3500), (5500, 3500), (5500, 3900), (3700, 3900)], close=True, dxfattribs={"layer": "FURNITURE"})
    path = Path(__file__).with_name("living_room.dxf")
    doc.saveas(path)
    print(path)


if __name__ == "__main__":
    main()
