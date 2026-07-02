"""
Snapping Core

Deterministic geometry alignment for Table Transformer outputs.
"""

from typing import List


DEFAULT_EPSILON = 5.0


def cluster_coordinates(
    values: List[float],
    epsilon: float = DEFAULT_EPSILON,
) -> List[float]:
    """
    Group nearby coordinates into master separators.
    """

    if not values:
        return []

    values = sorted(values)

    clusters = [[values[0]]]

    for value in values[1:]:

        if abs(value - clusters[-1][-1]) <= epsilon:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    masters = [
        sum(cluster) / len(cluster)
        for cluster in clusters
    ]

    return masters


def snap_to_nearest(
    value: float,
    masters: List[float],
) -> float:

    if not masters:
        return value

    return min(
        masters,
        key=lambda x: abs(x - value),
    )


def snap_bbox(
    bbox: List[float],
    master_x: List[float],
    master_y: List[float],
) -> List[float]:

    x1, y1, x2, y2 = bbox

    return [
        snap_to_nearest(x1, master_x),
        snap_to_nearest(y1, master_y),
        snap_to_nearest(x2, master_x),
        snap_to_nearest(y2, master_y),
    ]


def build_master_separators(
    rows,
    columns,
):

    xs = []

    ys = []

    for c in columns:

        xs.append(c["x_start"])
        xs.append(c["x_end"])

    for r in rows:

        ys.append(r["y_start"])
        ys.append(r["y_end"])

    master_x = cluster_coordinates(xs)

    master_y = cluster_coordinates(ys)

    return master_x, master_y
