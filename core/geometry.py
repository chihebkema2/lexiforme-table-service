"""
Geometry Utilities
"""

from typing import List


def bbox_width(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0])


def bbox_height(bbox: List[float]) -> float:
    return max(0.0, bbox[3] - bbox[1])


def bbox_area(bbox: List[float]) -> float:
    return bbox_width(bbox) * bbox_height(bbox)


def bbox_center(bbox: List[float]):
    return (
        (bbox[0] + bbox[2]) / 2,
        (bbox[1] + bbox[3]) / 2,
    )


def intersection_area(a: List[float], b: List[float]) -> float:

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def iou(a: List[float], b: List[float]) -> float:

    inter = intersection_area(a, b)

    if inter == 0:
        return 0.0

    union = bbox_area(a) + bbox_area(b) - inter

    if union == 0:
        return 0.0

    return inter / union
