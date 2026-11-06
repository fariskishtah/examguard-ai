"""Pure geometry helpers (no external dependencies)."""
from __future__ import annotations

Box = tuple[float, float, float, float]


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    return box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def expand_box(b: Box, frac: float) -> Box:
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - w * frac, b[1] - h * frac, b[2] + w * frac, b[3] + h * frac)


def containment(inner: Box, outer: Box) -> float:
    """Fraction of `inner` area that lies inside `outer`."""
    a = box_area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def point_in_polygon(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    """Ray casting; boundary points count as inside-ish (not relied upon)."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside
