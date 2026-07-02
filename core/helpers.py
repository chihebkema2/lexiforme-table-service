"""
Shared Helper Functions
"""

from typing import List


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def round2(value: float) -> float:
    return round(value, 2)


def sort_rows(rows):
    return sorted(rows, key=lambda r: r["y_start"])


def sort_columns(columns):
    return sorted(columns, key=lambda c: c["x_start"])


def sort_cells(cells):
    return sorted(
        cells,
        key=lambda c: (
            c.get("row", 0),
            c.get("col", 0),
        ),
    )


def unique(values: List[float]) -> List[float]:
    return sorted(set(values))
