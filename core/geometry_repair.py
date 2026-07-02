import logging
from typing import List, Dict, Any, Tuple

from core.geometry import bbox_area, intersection_area, iou  #[cite: 10]
from core.geometry_validation import GeometryValidationEngine

logger = logging.getLogger("table-service.gre")

class GeometryRepairEngine:
    @staticmethod
    def apply(
        rows: List[Dict[str, Any]],
        cols: List[Dict[str, Any]],
        cells: List[Dict[str, Any]],
        table_bbox: List[float],
        report: Any
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Applies target coordinate repairs to table features based on a read-only GVE report.
        Maintains data lineage by treating raw_bbox and snapped_bbox as immutable records.
        """
        # Read-only structural gate check
        if not getattr(report, "repairable", True):
            logger.warning("GRE Abort: Table structure marked unrepairable by GVE. Passing geometry unchanged.")
            return rows, cols, cells

        # Deep-copy input structures to isolate state mutations safely
        repaired_rows = [dict(r) for r in rows]
        repaired_cols = [dict(c) for c in cols]
        repaired_cells = [dict(c) for c in cells]

        # Invariant Deterministic Fixed Repair Sequence
        repair_map: Dict[str, List[Any]] = {
            "NORMALIZE_TRACK_BOUNDARIES": [],
            "ALIGN_TRACK_EXTENTS": [],
            "MERGE_DUPLICATE_BOUNDARIES": [],
            "REGENERATE_INTERSECTION_BOUNDS": [],
            "TRUNCATE_CELL_COLLISION_SPANS": [],
            "SYNTHESIZE_MISSING_GRID_CELL": [],
            "RECLAIM_ORPHAN_GRID": []
        }

        for issue in report.issues:
            if issue.suggested_repair in repair_map:
                repair_map[issue.suggested_repair].append(issue)

        # 1. Track Normalization & Alignment Passes
        for issue in repair_map["NORMALIZE_TRACK_BOUNDARIES"]:
            GeometryRepairEngine._repair_track_variations(repaired_rows, repaired_cols, issue)
        for issue in repair_map["ALIGN_TRACK_EXTENTS"]:
            GeometryRepairEngine._repair_track_drift(repaired_rows, repaired_cols, table_bbox, issue)

        # 2. Duplicate Removal Layer
        for issue in repair_map["MERGE_DUPLICATE_BOUNDARIES"]:
            GeometryRepairEngine._repair_duplicates(repaired_cells, issue)
            
        # Post-duplicate extraction cleanup step: immediately strip out inactive references
        repaired_cells = [c for c in repaired_cells if c.get("is_active", True)]

        # 3. Overlap Repair Layer
        for issue in repair_map["REGENERATE_INTERSECTION_BOUNDS"]:
            GeometryRepairEngine._repair_overlaps(repaired_cells, issue)

        # 4. Span Repair Layer
        for issue in repair_map["TRUNCATE_CELL_COLLISION_SPANS"]:
            GeometryRepairEngine._repair_invalid_spans(repaired_cells, repaired_rows, repaired_cols, issue)

        # 5. Missing Cell Synthesis Layer
        for issue in repair_map["SYNTHESIZE_MISSING_GRID_CELL"]:
            GeometryRepairEngine._repair_missing_cells(repaired_cells, repaired_rows, repaired_cols, issue)

        # 6. Orphan Recovery Layer
        for issue in repair_map["RECLAIM_ORPHAN_GRID"]:
            GeometryRepairEngine._repair_orphans(repaired_cells, repaired_rows, repaired_cols, issue)

        # Final cell pruning sweep for any leftover references
        final_cells = [c for c in repaired_cells if c.get("is_active", True)]

        # Post-Repair Structural Validation Feedback Loop
        post_repair_report = GeometryValidationEngine.analyze(
            [[r["bbox"][0], r["bbox"][1], r["bbox"][2], r["bbox"][3]] for r in repaired_rows],
            [[c["bbox"][0], c["bbox"][1], c["bbox"][2], c["bbox"][3]] for c in repaired_cols],
            [[s["bbox"][0], s["bbox"][1], s["bbox"][2], s["bbox"][3]] for s in final_cells],
            table_bbox
        )

        if post_repair_report.errors > 0:
            logger.error(
                f"GRE Hardening Warning: Post-repair validation detected {post_repair_report.errors} unresolved error state(s). "
                f"Final structural quality confidence metric scored at {post_repair_report.confidence:.4f}."
            )
            for issue in post_repair_report.issues:
                logger.error(f" -> Unresolved issue [{issue.issue_id}]: {issue.description}")
        else:
            logger.info(f"GRE Hardening Success: Post-repair checks clear. Verified final confidence: {post_repair_report.confidence:.4f}")

        return repaired_rows, repaired_cols, final_cells

    @staticmethod
    def _repair_track_variations(rows: List[Dict[str, Any]], cols: List[Dict[str, Any]], issue: Any) -> None:
        """
        Isolated Reference Median Track Normalization.
        Computes median sizes exclusively from tracks that are not flagged as anomalous by the GVE.
        """
        issue_id = issue.issue_id

        if issue.affected_rows and len(rows) > len(issue.affected_rows):
            valid_heights = sorted([
                r["bbox"][3] - r["bbox"][1] for idx, r in enumerate(rows)
                if idx not in issue.affected_rows
            ])
            median_height = valid_heights[len(valid_heights) // 2] if valid_heights else 20.0
            
            for r_idx in issue.affected_rows:
                if r_idx < len(rows):
                    current_y1 = rows[r_idx]["bbox"][1]
                    rows[r_idx]["bbox"][3] = current_y1 + median_height
                    logger.info(f"[{issue_id}] Normalized row {r_idx} height using static reference median ({median_height:.2f}px).")

        if issue.affected_columns and len(cols) > len(issue.affected_columns):
            valid_widths = sorted([
                c["bbox"][2] - c["bbox"][0] for idx, c in enumerate(cols)
                if idx not in issue.affected_columns
            ])
            median_width = valid_widths[len(valid_widths) // 2] if valid_widths else 60.0
            
            for c_idx in issue.affected_columns:
                if c_idx < len(cols):
                    current_x0 = cols[c_idx]["bbox"][0]
                    cols[c_idx]["bbox"][2] = current_x0 + median_width
                    logger.info(f"[{issue_id}] Normalized column {c_idx} width using static reference median ({median_width:.2f}px).")

    @staticmethod
    def _repair_track_drift(rows: List[Dict[str, Any]], cols: List[Dict[str, Any]], table_bbox: List[float], issue: Any) -> None:
        """Snaps drifting track endpoints cleanly to the table boundary frame."""
        tx1, ty1, tx2, ty2 = table_bbox
        issue_id = issue.issue_id

        for r_idx in issue.affected_rows:
            if r_idx < len(rows):
                rows[r_idx]["bbox"][0] = tx1
                rows[r_idx]["bbox"][2] = tx2
                logger.info(f"[{issue_id}] Reset row {r_idx} horizontal extents onto parent table boundary bounds.")

        for c_idx in issue.affected_columns:
            if c_idx < len(cols):
                cols[c_idx]["bbox"][1] = ty1
                cols[c_idx]["bbox"][3] = ty2
                logger.info(f"[{issue_id}] Reset column {c_idx} vertical extents onto parent table boundary bounds.")

    @staticmethod
    def _repair_duplicates(cells: List[Dict[str, Any]], issue: Any) -> None:
        """Filters duplicates by retaining the higher confidence element and flag-marking others for sweep cleanup."""
        if len(issue.affected_cells) < 2:
            return
        issue_id = issue.issue_id

        def sorting_key(cell_idx: int) -> Tuple[float, float]:
            if cell_idx >= len(cells):
                return (-1.0, -1.0)
            cell = cells[cell_idx]
            confidence = float(cell.get("score", cell.get("confidence", 0.0)))
            area = bbox_area(cell["bbox"])  #[cite: 10]
            return (confidence, area)

        sorted_indices = sorted(issue.affected_cells, key=sorting_key, reverse=True)
        keep_idx = sorted_indices[0]
        drop_indices = sorted_indices[1:]

        for drop_idx in drop_indices:
            if drop_idx < len(cells):
                cells[drop_idx]["is_active"] = False
                logger.info(f"[{issue_id}] Consolidated duplicate cell index {drop_idx} into cell index {keep_idx}.")

    @staticmethod
    def _repair_overlaps(cells: List[Dict[str, Any]], issue: Any) -> None:
        """Resolves overlapping bounding regions by cleanly splitting boundaries along their dominant overlap axis."""
        if len(issue.affected_cells) < 2:
            return
        issue_id = issue.issue_id
        idx_a, idx_b = issue.affected_cells[0], issue.affected_cells[1]
        if idx_a >= len(cells) or idx_b >= len(cells):
            return
            
        box_a = cells[idx_a]["bbox"]
        box_b = cells[idx_b]["bbox"]

        x1_inter = max(box_a[0], box_b[0])
        y1_inter = max(box_a[1], box_b[1])
        x2_inter = min(box_a[2], box_b[2])
        y2_inter = min(box_a[3], box_b[3])

        inter_width = x2_inter - x1_inter
        inter_height = y2_inter - y1_inter

        if inter_width <= 0 or inter_height <= 0:
            return

        if inter_width > inter_height:
            mid_y = (y1_inter + y2_inter) / 2.0
            if box_a[1] < box_b[1]:
                box_a[3] = mid_y
                box_b[1] = mid_y
            else:
                box_a[1] = mid_y
                box_b[3] = mid_y
            logger.info(f"[{issue_id}] Split horizontal cell boundary segment between ({idx_a}, {idx_b}) at line y={mid_y:.2f}.")
        else:
            mid_x = (x1_inter + x2_inter) / 2.0
            if box_a[0] < box_b[0]:
                box_a[2] = mid_x
                box_b[0] = mid_x
            else:
                box_a[0] = mid_x
                box_b[2] = mid_x
            logger.info(f"[{issue_id}] Split vertical cell boundary segment between ({idx_a}, {idx_b}) at line x={mid_x:.2f}.")

    @staticmethod
    def _repair_orphans(cells: List[Dict[str, Any]], rows: List[Dict[str, Any]], cols: List[Dict[str, Any]], issue: Any) -> None:
        """Resolves orphan cell mapping states by reattaching them to the closest row and column tracking centers."""
        issue_id = issue.issue_id
        for cell_idx in issue.affected_cells:
            if cell_idx >= len(cells):
                continue
            cell = cells[cell_idx]
            c_x = (cell["bbox"][0] + cell["bbox"][2]) / 2.0
            c_y = (cell["bbox"][1] + cell["bbox"][3]) / 2.0

            best_r_idx = 0
            min_r_dist = float("inf")
            for r_idx, row in enumerate(rows):
                r_center = (row["bbox"][1] + row["bbox"][3]) / 2.0
                dist = abs(c_y - r_center)
                if dist < min_r_dist:
                    min_r_dist = dist
                    best_r_idx = r_idx

            best_c_idx = 0
            min_c_dist = float("inf")
            for c_idx, col in enumerate(cols):
                c_center = (col["bbox"][0] + col["bbox"][2]) / 2.0
                dist = abs(c_x - c_center)
                if dist < min_c_dist:
                    min_c_dist = dist
                    best_c_idx = c_idx

            cell["row"] = best_r_idx
            cell["col"] = best_c_idx
            cell["bbox"] = [cols[best_c_idx]["bbox"][0], rows[best_r_idx]["bbox"][1], cols[best_c_idx]["bbox"][2], rows[best_r_idx]["bbox"][3]]
            logger.info(f"[{issue_id}] Reclaimed orphan cell index {cell_idx} into track coordinate (Row: {best_r_idx}, Col: {best_c_idx}).")

    @staticmethod
    def _repair_missing_cells(cells: List[Dict[str, Any]], rows: List[Dict[str, Any]], cols: List[Dict[str, Any]], issue: Any) -> None:
        """Verifies cell span coverage before synthesizing standard padding blocks within empty tracking fields."""
        issue_id = issue.issue_id
        for r_idx in issue.affected_rows:
            for c_idx in issue.affected_columns:
                
                is_spatially_covered = False
                for cell in cells:
                    if not cell.get("is_active", True):
                        continue
                    
                    c_row, c_col = cell["row"], cell["col"]
                    c_rowspan, c_colspan = cell["rowspan"], cell["colspan"]
                    
                    if (c_row <= r_idx < c_row + c_rowspan) and (c_col <= c_idx < c_col + c_colspan):
                        is_spatially_covered = True
                        break
                        
                if is_spatially_covered:
                    logger.info(f"[{issue_id}] Skipped cell synthesis at grid coordinate (Row: {r_idx}, Col: {c_idx}). Region is covered by an active span.")
                    continue

                row_track = rows[r_idx]["bbox"]
                col_track = cols[c_idx]["bbox"]
                synthesized_bbox = [col_track[0], row_track[1], col_track[2], row_track[3]]
                
                new_cell = {
                    "row": r_idx,
                    "col": c_idx,
                    "rowspan": 1,
                    "colspan": 1,
                    "raw_bbox": synthesized_bbox.copy(),
                    "snapped_bbox": synthesized_bbox.copy(),
                    "bbox": synthesized_bbox.copy(),
                    "is_active": True
                }
                cells.append(new_cell)
                logger.info(f"[{issue_id}] Restored layout void by generating synthetic single cell at (Row: {r_idx}, Col: {c_idx}).")

    @staticmethod
    def _repair_invalid_spans(cells: List[Dict[str, Any]], rows: List[Dict[str, Any]], cols: List[Dict[str, Any]], issue: Any) -> None:
        """Dynamically re-evaluates physical rowspan and colspan properties from active track intersections."""
        issue_id = issue.issue_id
        for cell_idx in range(len(cells)):
            cell = cells[cell_idx]
            if not cell.get("is_active", True):
                continue
                
            c_box = cell["bbox"]
            cell_area = bbox_area(c_box)  #[cite: 10]
            if cell_area == 0.0:
                continue

            r_matches = []
            for r_idx, row in enumerate(rows):
                overlap = intersection_area(c_box, [c_box[0], row["bbox"][1], c_box[2], row["bbox"][3]])  #[cite: 10]
                if (overlap / cell_area) > 0.15:
                    r_matches.append(r_idx)

            c_matches = []
            for c_idx, col in enumerate(cols):
                overlap = intersection_area(c_box, [col["bbox"][0], c_box[1], col["bbox"][2], c_box[3]])  #[cite: 10]
                if (overlap / cell_area) > 0.15:
                    c_matches.append(c_idx)

            if r_matches and c_matches:
                orig_rowspan, orig_colspan = cell["rowspan"], cell["colspan"]
                cell["row"] = min(r_matches)
                cell["col"] = min(c_matches)
                cell["rowspan"] = max(r_matches) - min(r_matches) + 1
                cell["colspan"] = max(c_matches) - min(c_matches) + 1
                
                if cell["rowspan"] != orig_rowspan or cell["colspan"] != orig_colspan:
                    logger.info(f"[{issue_id}] Recalculated valid span boundaries for cell index {cell_idx} (Rowspan: {cell['rowspan']}, Colspan: {cell['colspan']}).")
