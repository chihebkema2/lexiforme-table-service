"""
Shared Data Models
"""

from dataclasses import dataclass
from typing import List


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_list(self) -> List[float]:
        return [
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        ]
