"""Keep CAD coordinates unchanged. UI dimensions use the same declared units."""
UNIT_NAMES = {0: "unknown", 1: "in", 2: "ft", 4: "mm", 5: "cm", 6: "m"}
COMMON_UNITS = ("unknown", "mm", "cm", "m", "in", "ft")


def drawing_units(doc):
    code = int(doc.header.get("$INSUNITS", 0))
    return UNIT_NAMES.get(code, f"unknown (INSUNITS={code})")


def from_mm(value, unit):
    factors = {"mm": 1, "cm": 10, "m": 1000, "in": 25.4, "ft": 304.8}
    return value / factors.get(unit, 1)
