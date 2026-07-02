"""
LexiForme Table Detection Service
Uses Microsoft Table Transformer to detect tables, cells, and structure
from document images, returning precise coordinate-based JSON.
"""

import io
import base64
import logging
from typing import List, Dict, Any
from core.validation import validate_geometry
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("table-service")

app = FastAPI(title="LexiForme Table Detection Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Loading Table Transformer detection model...")
detection_processor = AutoImageProcessor.from_pretrained(
    "microsoft/table-transformer-detection"
)
if hasattr(detection_processor, "size") and isinstance(detection_processor.size, dict):
    if "shortest_edge" not in detection_processor.size and "height" not in detection_processor.size:
        longest = detection_processor.size.get("longest_edge", 1000)
        detection_processor.size = {"shortest_edge": longest, "longest_edge": longest}

detection_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection"
).to(DEVICE)

logger.info("Loading Table Transformer structure recognition model...")
structure_processor = AutoImageProcessor.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
)
if hasattr(structure_processor, "size") and isinstance(structure_processor.size, dict):
    if "shortest_edge" not in structure_processor.size and "height" not in structure_processor.size:
        longest = structure_processor.size.get("longest_edge", 1000)
        structure_processor.size = {"shortest_edge": longest, "longest_edge": longest}

structure_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
).to(DEVICE)

logger.info(f"Models loaded successfully on {DEVICE}.")


class DetectRequest(BaseModel):
    image_base64: str


class Column(BaseModel):
    index: int
    x_start: float
    x_end: float
    width_pct: float


class Row(BaseModel):
    index: int
    y_start: float
    y_end: float
    height_pct: float


class Cell(BaseModel):
    row: int
    col: int
    rowspan: int
    colspan: int
    bbox: List[float]


class TableResult(BaseModel):
    bbox: List[float]
    columns: List[Column]
    rows: List[Row]
    cells: List[Cell]


class DetectResponse(BaseModel):
    tables: List[TableResult]
    image_width: int
    image_height: int


def decode_image(image_base64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def detect_tables(image: Image.Image) -> List[Dict[str, Any]]:
    """Run table detection to find bounding boxes of tables in the image."""
    inputs = detection_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = detection_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = detection_processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes
    )[0]

    tables = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label_name = detection_model.config.id2label[label.item()]
        if label_name == "table":
            tables.append({"bbox": box.tolist(), "score": score.item()})
    return tables


def split_merged_tables(
    image: Image.Image, tables: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    If only one table is detected but the document has two stacked grids
    (e.g. Semestre 1 + Semestre 2), split it at the largest horizontal gap.
    """
    if len(tables) != 1:
        return tables

    table = tables[0]
    x1, y1, x2, y2 = table["bbox"]
    padding = 10
    crop_x1 = max(0, x1 - padding)
    crop_y1 = max(0, y1 - padding)
    crop_x2 = min(image.width, x2 + padding)
    crop_y2 = min(image.height, y2 + padding)
    cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    inputs = structure_processor(images=cropped, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = structure_model(**inputs)

    target_sizes = torch.tensor([cropped.size[::-1]])
    results = structure_processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes
    )[0]

    row_intervals = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label_name = structure_model.config.id2label[label.item()]
        if label_name == "table row":
            _, ry1, _, ry2 = box.tolist()
            abs_ry1 = ry1 + crop_y1
            abs_ry2 = ry2 + crop_y1
            row_intervals.append((abs_ry1, abs_ry2))

    if len(row_intervals) < 4:
        return tables

    row_intervals.sort(key=lambda r: r[0])

    gaps = []
    for i in range(1, len(row_intervals)):
        gap_start = row_intervals[i - 1][1]
        gap_end = row_intervals[i][0]
        gap_size = gap_end - gap_start
        gaps.append((gap_size, i, gap_start, gap_end))

    if not gaps:
        return tables

    avg_row_height = sum(r[1] - r[0] for r in row_intervals) / len(row_intervals)
    max_gap = max(gaps, key=lambda g: g[0])
    max_gap_size, split_idx, gap_start, gap_end = max_gap

    if max_gap_size < avg_row_height * 1.5:
        logger.info(
            f"[Split] Gap ({max_gap_size:.1f}px) < 1.5x avg row ({avg_row_height:.1f}px). Not splitting."
        )
        return tables

    split_y = (gap_start + gap_end) / 2
    logger.info(
        f"[Split] Splitting at y={split_y:.1f} (gap={max_gap_size:.1f}px)"
    )

    table1 = {"bbox": [x1, y1, x2, split_y], "score": table["score"]}
    table2 = {"bbox": [x1, split_y, x2, y2], "score": table["score"]}
    return [table1, table2]


def recognize_structure(image: Image.Image, table_bbox: List[float]) -> Dict[str, Any]:
    """Crop to the table region and detect rows, columns, and spanning cells."""
    x1, y1, x2, y2 = table_bbox
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)

    cropped = image.crop((x1, y1, x2, y2))

    inputs = structure_processor(images=cropped, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = structure_model(**inputs)

    target_sizes = torch.tensor([cropped.size[::-1]])
    results = structure_processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes
    )[0]

    rows_raw = []
    cols_raw = []
    spanning_cells_raw = []

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label_name = structure_model.config.id2label[label.item()]
        bx1, by1, bx2, by2 = box.tolist()
        abs_box = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]

        if label_name == "table row":
            rows_raw.append(abs_box)
        elif label_name == "table column":
            cols_raw.append(abs_box)
        elif label_name in ("table spanning cell", "table projected row header"):
            spanning_cells_raw.append(abs_box)

    rows_raw.sort(key=lambda b: b[1])
    cols_raw.sort(key=lambda b: b[0])

    table_width = x2 - x1
    table_height = y2 - y1

    columns = []
    for i, box in enumerate(cols_raw):
        cx1, _, cx2, _ = box
        width_pct = round(((cx2 - cx1) / table_width) * 100, 2)
        columns.append(
            {"index": i, "x_start": cx1, "x_end": cx2, "width_pct": width_pct}
        )

    rows = []
    for i, box in enumerate(rows_raw):
        _, ry1, _, ry2 = box
        height_pct = round(((ry2 - ry1) / table_height) * 100, 2)
        rows.append(
            {"index": i, "y_start": ry1, "y_end": ry2, "height_pct": height_pct}
        )

    cells = build_cell_grid(rows_raw, cols_raw, spanning_cells_raw)

    return {
        "bbox": [x1, y1, x2, y2],
        "columns": columns,
        "rows": rows,
        "cells": cells,
    }


def build_cell_grid(
    rows_raw: List[List[float]],
    cols_raw: List[List[float]],
    spanning_cells_raw: List[List[float]],
) -> List[Dict[str, Any]]:
    n_rows = len(rows_raw)
    n_cols = len(cols_raw)

    occupied = [[False] * n_cols for _ in range(n_rows)]
    cells = []

    def overlaps(a, b) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    def find_span(box):
        r_indices = [i for i, r in enumerate(rows_raw) if overlaps(box, r)]
        c_indices = [i for i, c in enumerate(cols_raw) if overlaps(box, c)]
        if not r_indices or not c_indices:
            return None
        return min(r_indices), max(r_indices), min(c_indices), max(c_indices)

    for box in spanning_cells_raw:
        span = find_span(box)
        if span is None:
            continue
        r_min, r_max, c_min, c_max = span
        rowspan = r_max - r_min + 1
        colspan = c_max - c_min + 1
        if rowspan <= 1 and colspan <= 1:
            continue
        already_taken = any(
            occupied[r][c]
            for r in range(r_min, r_max + 1)
            for c in range(c_min, c_max + 1)
        )
        if already_taken:
            continue
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                occupied[r][c] = True
        cells.append(
            {"row": r_min, "col": c_min, "rowspan": rowspan, "colspan": colspan, "bbox": box}
        )

    for r in range(n_rows):
        for c in range(n_cols):
            if occupied[r][c]:
                continue
            row_box = rows_raw[r]
            col_box = cols_raw[c]
            cell_bbox = [col_box[0], row_box[1], col_box[2], row_box[3]]
            cells.append({"row": r, "col": c, "rowspan": 1, "colspan": 1, "bbox": cell_bbox})

    cells.sort(key=lambda c: (c["row"], c["col"]))
    return cells


@app.get("/")
def health_check():
    return {"status": "ok", "service": "lexiforme-table-detection", "device": DEVICE}


@app.post("/detect", response_model=DetectResponse)
def detect(payload: DetectRequest):
    image = decode_image(payload.image_base64)
    logger.info(f"Received image: {image.width}x{image.height}")

    detected_tables = detect_tables(image)
    logger.info(f"Found {len(detected_tables)} table(s) before split check")

    detected_tables = split_merged_tables(image, detected_tables)
    logger.info(f"Found {len(detected_tables)} table(s) after split check")

    results = []
    for t in detected_tables:
        structure = recognize_structure(image, t["bbox"])
        results.append(structure)

    return {
        "tables": results,
        "image_width": image.width,
        "image_height": image.height,
    }
