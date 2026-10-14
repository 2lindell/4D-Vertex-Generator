"""The wiki's 1200-vertex entries for H3●I2(10), as (cells, faces, edges) class sizes."""
from __future__ import annotations


def _p(text: str) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in text.split("+")))


WIKI_1200 = {
    "W1": ("120+120+600+600+600+600+1200", "240+600+1200+1200+1200+1200+1200+1200+1200",
           "600+600+600+600+600+1200+1200+1200"),
    "W2": ("120+120+240+600+600+1200+1200", "240+240+1200+1200+1200+1200+1200+1200+1200+1200",
           "600+600+600+600+1200+1200+1200+1200"),
    "W3": ("120+120+600+600+600+600+1200+1200", "240+1200+1200+1200+1200+1200+1200+1200+1200+1200",
           "600+600+600+600+600+600+1200+1200+1200"),
    "W4": ("120+120+600+600+600+600+1200+1200", "240+600+1200+1200+1200+1200+1200+1200+1200+1200+1200",
           "600+600+600+600+600+1200+1200+1200+1200"),
    "W5": ("120+120+600+600+600+600+1200+1200+1200",
           "240+600+1200+1200+1200+1200+1200+1200+1200+1200+1200+1200+1200",
           "600+600+600+600+600+1200+1200+1200+1200+1200"),
    "T1": ("120+120+600+600", "240+600+1200+1200+1200", "600+600+600+1200+1200"),
    "T2": ("120+120+600+600+1200", "240+1200+1200+1200+1200+1200+1200",
           "600+600+600+600+1200+1200+1200"),
}
WIKI_1200 = {k: tuple(_p(x) for x in v) for k, v in WIKI_1200.items()}

LABELS = {
    "W1": "Polychoron with 120+120+600+600+600+600+1200 cells",
    "W2": "Polychoron with 120+120+240+600+600+1200+1200 cells",
    "W3": "Polychoron with 120+120+600+600+600+600+1200+1200 cells (240+1200×9 faces)",
    "W4": "Polychoron with 120+120+600+600+600+600+1200+1200 cells (240+600+1200×9 faces)",
    "W5": "Polychoron with 120+120+600+600+600+600+1200+1200+1200 cells",
    "T1": "Transitional polychoron with 120+120+600+600 cells",
    "T2": "Transitional polychoron with 120+120+600+600+1200 cells",
}


def parse_signature(sig: str) -> tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    """'1200: cells a+b | faces … | edges … | val …' -> (vertices, cells, faces, edges)."""
    if sig.startswith("ERR"):
        return None
    head, rest = sig.split(": ", 1)
    parts = [p.strip() for p in rest.split("|")]
    cells = _p(parts[0].removeprefix("cells "))
    faces = _p(parts[1].removeprefix("faces "))
    edges = _p(parts[2].removeprefix("edges "))
    return int(head), cells, faces, edges


def match(sig: str) -> str | None:
    parsed = parse_signature(sig)
    if parsed is None:
        return None
    _, cells, faces, edges = parsed
    for key, entry in WIKI_1200.items():
        if entry == (cells, faces, edges):
            return key
    return None
