import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("validator")

class DataQualityValidator:
    """Automated Data Quality & Integrity Validator for Battery Datasets."""
    
    REQUIRED_COLUMNS = [
        "cell_id", "cycle", "capacity", "soh", "voltage_mean",
        "voltage_max", "voltage_min", "current_mean", "temp_mean"
    ]
    
    PHYSICAL_BOUNDS = {
        "voltage_min": (2.0, 4.0),
        "voltage_max": (3.5, 4.5),
        "temp_min": (0.0, 60.0),
        "temp_max": (10.0, 80.0),
        "capacity": (0.1, 5.0),
        "soh": (0.05, 1.20)
    }
    
    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Perform rigorous data quality checks on battery dataframe."""
        total_records = len(df)
        report: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "total_records": total_records,
            "cells": sorted(df["cell_id"].unique().tolist()) if "cell_id" in df else [],
            "missing_columns": [],
            "nan_counts": {},
            "infinite_counts": {},
            "duplicate_cycles": 0,
            "broken_cycles": 0,
            "out_of_bounds_violations": {},
            "status": "PASS"
        }
        
        # Check required columns
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                report["missing_columns"].append(col)
        if report["missing_columns"]:
            report["status"] = "FAIL"
            return report
            
        # Check NaNs and Inf values
        for col in df.columns:
            nan_cnt = int(df[col].isna().sum())
            if nan_cnt > 0:
                report["nan_counts"][col] = nan_cnt
            if np.issubdtype(df[col].dtype, np.number):
                inf_cnt = int(np.isinf(df[col]).sum())
                if inf_cnt > 0:
                    report["infinite_counts"][col] = inf_cnt
                    
        # Check duplicate or non-monotonic cycles per cell
        for cell_id, group in df.groupby("cell_id"):
            dups = int(group["cycle"].duplicated().sum())
            if dups > 0:
                report["duplicate_cycles"] += dups
            is_monotonic = group["cycle"].is_monotonic_increasing
            if not is_monotonic:
                report["broken_cycles"] += 1
                
        # Check physical boundary conditions
        for col, (lower, upper) in self.PHYSICAL_BOUNDS.items():
            if col in df.columns:
                violations = int(((df[col] < lower) | (df[col] > upper)).sum())
                if violations > 0:
                    report["out_of_bounds_violations"][col] = violations
                    
        if report["nan_counts"] or report["infinite_counts"] or report["duplicate_cycles"] > 0:
            report["status"] = "WARNING" if report["status"] != "FAIL" else "FAIL"
        if report["out_of_bounds_violations"]:
            report["status"] = "WARNING"
            
        return report

    def run_full_validation(
        self,
        nasa_df: pd.DataFrame,
        calce_df: pd.DataFrame,
        report_output_path: str = "results/metrics/dataset_quality_report.json"
    ) -> Dict[str, Any]:
        """Run end-to-end data validation and serialize JSON quality report."""
        nasa_report = self.validate_dataframe(nasa_df, "NASA_PCoE")
        calce_report = self.validate_dataframe(calce_df, "CALCE_CS2")
        
        full_report = {
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "overall_status": "PASS" if (nasa_report["status"] == "PASS" and calce_report["status"] == "PASS") else "WARNING",
            "datasets": {
                "nasa_pcoe": nasa_report,
                "calce_cs2": calce_report
            }
        }
        
        out_p = Path(report_output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
            
        logger.info(f"Data quality report saved to {report_output_path} (Overall Status: {full_report['overall_status']})")
        return full_report
