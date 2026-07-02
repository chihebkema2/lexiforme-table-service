from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import logging

from core.geometry import iou, bbox_area  #[cite: 10]

logger = logging.getLogger("table-service.gve")

class SeverityLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class ValidationStage(str, Enum):
    TRACK_VALIDATION = "TRACK_VALIDATION"
    CELL_VALIDATION = "CELL_VALIDATION"
    TOPOLOGY = "TOPOLOGY"
    SPAN_VALIDATION = "SPAN_VALIDATION"

class GeometryIssue:
    """
    Represents a single, strongly-typed layout anomaly detected by the GVE.
    Uses entirely human-readable, deterministic identifiers rather than generic UUIDs.
    """
    def __init__(
        self,
        stage: ValidationStage,
        issue_type: str,
        severity: SeverityLevel,
        sequence_num: int,
        affected_rows: List[int],
        affected_columns: List[int],
        affected_cells: List[int],
        description: str,
        suggested_repair: str
    ):
        self.stage: ValidationStage = stage
        self.issue_type: str = issue_type
        self.severity: SeverityLevel = severity
        self.issue_id: str = f"{issue_type}_{sequence_num:03d}"
        self.affected_rows: List[int] = affected_rows
        self.affected_columns: List[int] = affected_columns
        self.affected_cells: List[int] = affected_cells
        self.description: str = description
        self.suggested_repair: str = suggested_repair

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "stage": self.stage.value,
            "issue_type": self.issue_type,
            "severity": self.severity.value,
            "affected_rows": self.affected_rows,
            "affected_columns": self.affected_columns,
            "affected_cells": self.affected_cells,
            "description": self.description,
            "suggested_repair": self.suggested_repair,
        }

class ValidationReport:
    """
    Maintains structural metrics, summary aggregates, and dynamically computes
    the normalized quality confidence and criteria-based repairability parameters.
    """
    def __init__(self, table_bbox: List[float], total_rows: int, total_cols: int, total_cells: int):
        self.table_bbox: List[float] = table_bbox
        self.issues: List[GeometryIssue] = []
        
        # Summary Counter blocks
        self.errors: int = 0
        self.warnings: int = 0
        self.info: int = 0
        self.repairable: bool = True
        self.confidence: float = 1.0
        
        # Structural Grid Metadata
        self.total_rows: int = total_rows
        self.total_cols: int = total_cols
        self.total_cells: int = total_cells
        self.total_elements: int = total_rows + total_cols + total_cells

    def add_issue(self, issue: GeometryIssue) -> None:
        self.issues.append(issue)
        if issue.severity == SeverityLevel.ERROR:
            self.errors += 1
        elif issue.severity == SeverityLevel.WARNING:
            self.warnings += 1
        elif issue.severity == SeverityLevel.INFO:
            self.info += 1

    def finalize_report(self) -> None:
        """
        Derives a normalized structural quality score from the ratio of weighted 
        detected anomalies to total physical table elements, preventing penalty bias on large scales.
        """
        if self.total_elements == 0:
            self.confidence = 0.0
            self.repairable = False
            return

        weighted_error_score = self.errors * 1.0
        weighted_warning_score = self.warnings * 0.4
        weighted_info_score = self.info * 0.1
        
        total_anomaly_weight = weighted_error_score + weighted_warning_score + weighted_info_score
        anomaly_ratio = total_anomaly_weight / self.total_elements
        
        self.confidence = max(0.0, min(1.0, 1.0 - anomaly_ratio))
        self._evaluate_structural_repairability()

    def _evaluate_structural_repairability(self) -> None:
        """
        Evaluates system repairability utilizing structural topology criteria 
        instead of hardcoded scalar constants.
        """
        # Criteria 1: Critical Axis Track Depletion
        if self.total_rows == 0 or self.total_cols == 0:
            self.repairable = False
            logger.error("GVE Flag: Unrepairable grid topology. Row or Column track counts are missing entirely.")
            return

        # Criteria 2: Complete Topological Degradation
        if self.errors >= self.total_cells and self.total_cells > 0:
            self.repairable = False
            logger.error("GVE Flag: Unrepairable topology. Severe collision count outpaces valid cell population.")
            return

        # Criteria 3: Quality Floor Breach
        if self.confidence < 0.20:
            self.repairable = False
            logger.error(f"GVE Flag: Grid degradation beyond logical recovery limits. Confidence: {self.confidence:.4f}")
            return

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_bbox": self.table_bbox,
            "summary": {
                "errors": self.errors,
                "warnings": self.warnings,
                "info": self.info,
                "repairable": self.repairable
            },
            "confidence": round(self.confidence, 4),
            "issues": [issue.to_dict() for issue in self.issues]
        }

class GeometryValidationEngine:
    @staticmethod
    def analyze(
        rows: List[List[float]],
        cols: List[List[float]],
        cells: List[List[float]],
        table_bbox: List[float],
        iob_threshold: float = 0.60
    ) -> ValidationReport:
        """
        Performs read-only validation of table structures. Encapsulates track variations, 
        boundary drift, overlaps, and cell omissions into a strongly-typed tracking report.
        """
        report = ValidationReport(
            table_bbox=table_bbox,
            total_rows=len(rows),
            total_cols=len(cols),
            total_cells=len(cells)
        )
        counters: Dict[str, int] = {}

        def get_next_seq(issue_type: str) -> int:
            counters[issue_type] = counters.get(issue_type, 0) + 1
            return counters[issue_type]

        # STAGE 1: TRACK_VALIDATION
        GeometryValidationEngine._check_track_variations(rows, is_row=True, report=report, get_seq=get_next_seq)
        GeometryValidationEngine._check_track_variations(cols, is_row=False, report=report, get_seq=get_next_seq)
        GeometryValidationEngine._check_boundary_drift(rows, table_bbox, is_row=True, report=report, get_seq=get_next_seq)
        GeometryValidationEngine._check_boundary_drift(cols, table_bbox, is_row=False, report=report, get_seq=get_next_seq)

        # STAGE 2: CELL_VALIDATION
        GeometryValidationEngine._check_cell_anomalies(cells, report=report, get_seq=get_next_seq)

        # STAGE 3: TOPOLOGY & SPAN_VALIDATION
        GeometryValidationEngine._check_missing_coverage(rows, cols, cells, iob_threshold, report=report, get_seq=get_next_seq)

        report.finalize_report()
        return report

    @staticmethod
    def _check_track_variations(tracks: List[List[float]], is_row: bool, report: ValidationReport, get_seq: Any) -> None:
        if len(tracks) <= 1:
            return
        sizes: List[float] = [t[3] - t[1] if is_row else t[2] - t[0] for t in tracks]
        sorted_sizes = sorted(sizes)
        median_size = sorted_sizes[len(sorted_sizes) // 2]
        
        if median_size == 0:
            return

        issue_type = "INCONSISTENT_ROW_HEIGHT" if is_row else "INCONSISTENT_COLUMN_WIDTH"
        for idx, size in enumerate(sizes):
            if size > 3.0 * median_size or size < (median_size / 3.0):
                report.add_issue(GeometryIssue(
                    stage=ValidationStage.TRACK_VALIDATION,
                    issue_type=issue_type,
                    severity=SeverityLevel.INFO,
                    sequence_num=get_seq(issue_type),
                    affected_rows=[idx] if is_row else [],
                    affected_columns=[] if is_row else [idx],
                    affected_cells=[],
                    description=f"{'Row' if is_row else 'Column'} index {idx} size varies significantly from the track median ({size:.2f}px vs {median_size:.2f}px).",
                    suggested_repair="NORMALIZE_TRACK_BOUNDARIES"
                ))

    @staticmethod
    def _check_boundary_drift(tracks: List[List[float]], table_bbox: List[float], is_row: bool, report: ValidationReport, get_seq: Any) -> None:
        tx1, ty1, tx2, ty2 = table_bbox
        issue_type = "ROW_DRIFT" if is_row else "COLUMN_DRIFT"
        
        for idx, track in enumerate(tracks):
            if is_row:
                drift_start = abs(track[0] - tx1)
                drift_end = abs(track[2] - tx2)
            else:
                drift_start = abs(track[1] - ty1)
                drift_end = abs(track[3] - ty2)
                
            if drift_start > 8.0 or drift_end > 8.0:
                report.add_issue(GeometryIssue(
                    stage=ValidationStage.TRACK_VALIDATION,
                    issue_type=issue_type,
                    severity=SeverityLevel.WARNING,
                    sequence_num=get_seq(issue_type),
                    affected_rows=[idx] if is_row else [],
                    affected_columns=[] if is_row else [idx],
                    affected_cells=[],
                    description=f"{'Row' if is_row else 'Column'} axis bound drift detected at tracking index {idx} relative to table boundary hull bounds.",
                    suggested_repair="ALIGN_TRACK_EXTENTS"
                ))

    @staticmethod
    def _check_cell_anomalies(cells: List[List[float]], report: ValidationReport, get_seq: Any) -> None:
        for i in range(len(cells)):
            cell_a = cells[i]
            area_a = bbox_area(cell_a)  #[cite: 10]
            if area_a == 0:
                continue
                
            for j in range(i + 1, len(cells)):
                cell_b = cells[j]
                overlap_iou = iou(cell_a, cell_b)  #[cite: 10]
                
                if overlap_iou > 0.85:
                    issue_type = "DUPLICATE_CELL"
                    report.add_issue(GeometryIssue(
                        stage=ValidationStage.CELL_VALIDATION,
                        issue_type=issue_type,
                        severity=SeverityLevel.ERROR,
                        sequence_num=get_seq(issue_type),
                        affected_rows=[],
                        affected_columns=[],
                        affected_cells=[i, j],
                        description=f"Duplicate spatial cells matched at array indexes ({i}, {j}) with IoU intersection score of {overlap_iou:.2f}.",
                        suggested_repair="MERGE_DUPLICATE_BOUNDARIES"
                    ))
                elif overlap_iou > 0.15:
                    issue_type = "OVERLAPPING_CELL"
                    report.add_issue(GeometryIssue(
                        stage=ValidationStage.CELL_VALIDATION,
                        issue_type=issue_type,
                        severity=SeverityLevel.WARNING,
                        sequence_num=get_seq(issue_type),
                        affected_rows=[],
                        affected_columns=[],
                        affected_cells=[i, j],
                        description=f"Overlapping bounding configurations detected between cell elements ({i}, {j}) with IoU intersection score of {overlap_iou:.2f}.",
                        suggested_repair="REGENERATE_INTERSECTION_BOUNDS"
                    ))

    @staticmethod
    def _check_missing_coverage(
        rows: List[List[float]],
        cols: List[List[float]],
        cells: List[List[float]],
        iob_threshold: float,
        report: ValidationReport,
        get_seq: Any
    ) -> None:
        n_rows = len(rows)
        n_cols = len(cols)
        grid_coverage = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
        
        for cell_idx, cell in enumerate(cells):
            cell_area = bbox_area(cell)  #[cite: 10]
            if cell_area == 0.0:
                continue
                
            matched_any = False
            for r_idx, row in enumerate(rows):
                for c_idx, col in enumerate(cols):
                    track_box = [col[0], row[1], col[2], row[3]]  #[cite: 1]
                    
                    x1 = max(cell[0], track_box[0])
                    y1 = max(cell[1], track_box[1])
                    x2 = min(cell[2], track_box[2])
                    y2 = min(cell[3], track_box[3])
                    
                    if x2 > x1 and y2 > y1:
                        inter_area = (x2 - x1) * (y2 - y1)
                        if (inter_area / cell_area) > iob_threshold:
                            grid_coverage[r_idx][c_idx] += 1
                            matched_any = True
                            
            if not matched_any:
                issue_type = "ORPHAN_CELL"
                report.add_issue(GeometryIssue(
                    stage=ValidationStage.TOPOLOGY,
                    issue_type=issue_type,
                    severity=SeverityLevel.WARNING,
                    sequence_num=get_seq(issue_type),
                    affected_rows=[],
                    affected_columns=[],
                    affected_cells=[cell_idx],
                    description=f"Isolated cell element detected at index {cell_idx} sitting outside standard row/column tracks.",
                    suggested_repair="RECLAIM_ORPHAN_GRID"
                ))
                
        for r in range(n_rows):
            for c in range(n_cols):
                if grid_coverage[r][c] == 0:
                    issue_type = "MISSING_CELL"
                    report.add_issue(GeometryIssue(
                        stage=ValidationStage.TOPOLOGY,
                        issue_type=issue_type,
                        severity=SeverityLevel.ERROR,
                        sequence_num=get_seq(issue_type),
                        affected_rows=[r],
                        affected_columns=[c],
                        affected_cells=[],
                        description=f"Structural tracking void detected at intersection coordinate map (Row: {r}, Col: {c}).",
                        suggested_repair="SYNTHESIZE_MISSING_GRID_CELL"
                    ))
                elif grid_coverage[r][c] > 1:
                    issue_type = "INVALID_SPAN"
                    report.add_issue(GeometryIssue(
                        stage=ValidationStage.SPAN_VALIDATION,
                        issue_type=issue_type,
                        severity=SeverityLevel.ERROR,
                        sequence_num=get_seq(issue_type),
                        affected_rows=[r],
                        affected_columns=[c],
                        affected_cells=[],
                        description=f"Multiple cell segments intersecting matrix block (Row: {r}, Col: {c}) without clear structural spanning parameters.",
                        suggested_repair="TRUNCATE_CELL_COLLISION_SPANS"
                    ))
