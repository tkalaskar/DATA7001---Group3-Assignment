"""
Data validation and provenance tracking for APEX.
Ensures data integrity and produces the provenance table
required by the UQ DATA7001 rubric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("apex.data.validators")


@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    message: str
    severity: str = "warning"  # info | warning | critical


class DataValidator:
    """
    Validates data quality, schema conformance, and statistical properties.
    """

    def validate(
        self,
        df: pd.DataFrame,
        expected_columns: list[str] | None = None,
        min_rows: int = 100,
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []

        # Row count
        if len(df) < min_rows:
            results.append(ValidationResult(
                "min_rows", False,
                f"Dataset has {len(df)} rows (minimum: {min_rows})",
                severity="critical",
            ))
        else:
            results.append(ValidationResult(
                "min_rows", True,
                f"Dataset has {len(df)} rows (>= {min_rows})",
            ))

        # Schema
        if expected_columns:
            missing = set(expected_columns) - set(df.columns)
            if missing:
                results.append(ValidationResult(
                    "schema", False,
                    f"Missing expected columns: {missing}",
                    severity="critical",
                ))
            else:
                results.append(ValidationResult(
                    "schema", True,
                    f"All {len(expected_columns)} expected columns present",
                ))

        # Missing values
        total_missing = int(df.isnull().sum().sum())
        total_cells = int(np.prod(df.shape))
        missing_pct = total_missing / max(total_cells, 1) * 100
        if missing_pct > 20:
            results.append(ValidationResult(
                "missing_values", False,
                f"{missing_pct:.1f}% missing values ({total_missing} cells)",
                severity="warning",
            ))
        else:
            results.append(ValidationResult(
                "missing_values", True,
                f"{missing_pct:.1f}% missing values ({total_missing} cells)",
            ))

        # Duplicates
        n_dup = int(df.duplicated().sum())
        if n_dup > 0:
            results.append(ValidationResult(
                "duplicates", False,
                f"{n_dup} duplicate rows detected",
                severity="warning",
            ))
        else:
            results.append(ValidationResult(
                "duplicates", True,
                "No duplicate rows",
            ))

        # Constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                results.append(ValidationResult(
                    "constant_column", False,
                    f"Column '{col}' is constant (only 1 unique value)",
                    severity="warning",
                ))

        # Infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            n_inf = int(np.isinf(df[col]).sum())
            if n_inf > 0:
                results.append(ValidationResult(
                    "infinite_values", False,
                    f"Column '{col}' has {n_inf} infinite values",
                    severity="critical",
                ))

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        logger.info("Validation: %d/%d checks passed", passed, total)
        return results

    @staticmethod
    def provenance_table() -> pd.DataFrame:
        """
        Generate the data provenance table for the UQ DATA7001
        presentation slide.
        """
        return pd.DataFrame([
            {
                "Dataset": "DrugBank 5.0",
                "Source": "drugbank.ca",
                "Access": "Open Academic",
                "Records": "14,000+",
                "Licence": "CC BY-NC 4.0",
                "DOI": "10.1093/nar/gkx1037",
            },
            {
                "Dataset": "ChEMBL 34",
                "Source": "ebi.ac.uk/chembl",
                "Access": "Free Download",
                "Records": "2.4M molecules",
                "Licence": "CC BY-SA 3.0",
                "DOI": "10.1093/nar/gkad1004",
            },
            {
                "Dataset": "STRING v12",
                "Source": "string-db.org",
                "Access": "Free Download",
                "Records": "67M interactions",
                "Licence": "CC BY 4.0",
                "DOI": "10.1093/nar/gkac1000",
            },
            {
                "Dataset": "DisGeNET v7",
                "Source": "disgenet.org",
                "Access": "Open Academic",
                "Records": "1.1M associations",
                "Licence": "ODbL",
                "DOI": "10.1093/nar/gkz1021",
            },
            {
                "Dataset": "CMIP6",
                "Source": "esgf-node.llnl.gov",
                "Access": "Free",
                "Records": "Multi-model ensemble",
                "Licence": "Open",
                "DOI": "10.5194/gmd-9-1937-2016",
            },
            {
                "Dataset": "ERA5",
                "Source": "cds.climate.copernicus.eu",
                "Access": "Free Academic",
                "Records": "Hourly 1940-present",
                "Licence": "Copernicus",
                "DOI": "10.1002/qj.3803",
            },
            {
                "Dataset": "Materials Project",
                "Source": "materialsproject.org",
                "Access": "Free API",
                "Records": "154,000+ materials",
                "Licence": "CC BY 4.0",
                "DOI": "10.1063/1.4812323",
            },
            {
                "Dataset": "SuperCon (NIMS)",
                "Source": "supercon.nims.go.jp",
                "Access": "Free Download",
                "Records": "33,000 superconductors",
                "Licence": "Open",
                "DOI": "N/A",
            },
        ])
