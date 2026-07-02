"""
Logical Grid Builder
"""

from typing import List, Dict


def create_empty_matrix(rows: int, cols: int):
    return [[None for _ in range(cols)] for _ in range(rows)]


def find_row_index(cell, rows):

    cy = (cell["bbox"][1] + cell["bbox"][3]) / 2

    for row in rows:
        if row["y_start"] <= cy <= row["y_end"]:
            return row["index"]

    return None


def find_column_index(cell, columns):

    cx = (cell["bbox"][0] + cell["bbox"][2]) / 2

    for col in columns:
        if col["x_start"] <= cx <= col["x_end"]:
            return col["index"]

    return None


def build_matrix(rows, columns, cells):

    matrix = create_empty_matrix(
        len(rows),
        len(columns)
    )

    for cell in cells:

        r = find_row_index(cell, rows)
        c = find_column_index(cell, columns)

        if r is None or c is None:
            continue

        cell["logical_row"] = r
        cell["logical_column"] = c

        matrix[r][c] = cell

    return matrix
