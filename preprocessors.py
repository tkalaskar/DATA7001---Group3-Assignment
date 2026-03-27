"""
Data preprocessing pipeline for APEX.
Handles missing values, outliers, normalisation, feature engineering,
and data quality documentation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger("apex.data.preprocessors")


class DataPreprocessor:
    """
    End-to-end data preprocessing with full audit trail.
    Produces a before/after quality report for the UQ DATA7001
    'data fit for use' rubric requirement.
    """

    def __init__(self):
        self.scaler: Any = None
        self.imputer: Any = None
        self._before_stats: dict[str, Any] = {}
        self._after_stats: dict[str, Any] = {}
        self._transformations: list[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
        scaling: str = "standard",
        impute_strategy: str = "median",
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline:
          1. Record before-stats
          2. Handle missing values
          3. Remove outliers (optional)
          4. Scale features
          5. Record after-stats
        """
        self._before_stats = self._compute_stats(df)
        self._transformations = []

        features = [c for c in df.columns if c != target_col]
        result = df.copy()

        # Step 1: missing values
        n_missing_before = int(result[features].isnull().sum().sum())
        if n_missing_before > 0:
            self.imputer = SimpleImputer(strategy=impute_strategy)
            result[features] = self.imputer.fit_transform(result[features])
            self._transformations.append(
                f"Imputed {n_missing_before} missing values using {impute_strategy}"
            )
            logger.info("Imputed %d missing values (%s)", n_missing_before, impute_strategy)

        # Step 2: remove outliers
        if remove_outliers:
            n_before = len(result)
            for col in features:
                if result[col].dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
                    mean = result[col].mean()
                    std = result[col].std()
                    if std > 0:
                        z = np.abs((result[col] - mean) / std)
                        result = result[z < outlier_threshold]
            n_removed = n_before - len(result)
            if n_removed > 0:
                self._transformations.append(
                    f"Removed {n_removed} outlier rows (z-score > {outlier_threshold})"
                )
                logger.info("Removed %d outlier rows", n_removed)
            result = result.reset_index(drop=True)

        # Step 3: scale features
        numeric_features = result[features].select_dtypes(include=[np.number]).columns.tolist()
        if numeric_features:
            if scaling == "standard":
                self.scaler = StandardScaler()
            elif scaling == "robust":
                self.scaler = RobustScaler()
            elif scaling == "minmax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = None

            if self.scaler:
                result[numeric_features] = self.scaler.fit_transform(result[numeric_features])
                self._transformations.append(
                    f"Scaled {len(numeric_features)} numeric features using {scaling}"
                )

        self._after_stats = self._compute_stats(result)
        return result

    def quality_report(self) -> dict[str, Any]:
        """
        Generate a before/after data quality report.
        Directly feeds the UQ DATA7001 'data fit for use' rubric.
        """
        return {
            "before": self._before_stats,
            "after": self._after_stats,
            "transformations_applied": self._transformations,
        }

    def quality_report_dataframe(self) -> pd.DataFrame:
        """Return the quality report as a presentable DataFrame."""
        rows = []
        before = self._before_stats
        after = self._after_stats

        rows.append({
            "Metric": "Total rows",
            "Before": before.get("n_rows", "—"),
            "After": after.get("n_rows", "—"),
        })
        rows.append({
            "Metric": "Total columns",
            "Before": before.get("n_cols", "—"),
            "After": after.get("n_cols", "—"),
        })
        rows.append({
            "Metric": "Missing values",
            "Before": before.get("n_missing", "—"),
            "After": after.get("n_missing", "—"),
        })
        rows.append({
            "Metric": "Duplicate rows",
            "Before": before.get("n_duplicates", "—"),
            "After": after.get("n_duplicates", "—"),
        })

        for tx in self._transformations:
            rows.append({"Metric": "Transformation", "Before": "—", "After": tx})

        return pd.DataFrame(rows)

    @staticmethod
    def _compute_stats(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_missing": int(df.isnull().sum().sum()),
            "n_duplicates": int(df.duplicated().sum()),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "numeric_summary": df.describe().to_dict() if len(df) > 0 else {},
        }
