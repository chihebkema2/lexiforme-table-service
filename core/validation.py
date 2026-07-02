"""
Geometry Validation Layer
"""

from typing import List, Dict, Tuple


def validate_rows(rows: List[Dict]) -> List[Dict]:
    rows = [
        r for r in rows
        if r["y_end"] > r["y_start"]
    ]

    rows.sort(key=lambda r: r["y_start"])

    for i, row in enumerate(rows):
        row["index"] = i

    return rows


def validate_columns(columns: List[Dict]) -> List[Dict]:
    columns = [
        c for c in columns
        if c["x_end"] > c["x_start"]
    ]

    columns.sort(key=lambda c: c["x_start"])

    for i, col in enumerate(columns):
        col["index"] = i

    return columns


def validate_geometry(
    rows: List[Dict],
    columns: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:

    rows = validate_rows(rows)
    columns = validate_columns(columns)

    return rows, columns
