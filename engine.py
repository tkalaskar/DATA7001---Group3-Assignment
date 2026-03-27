"""
APEX Engine — The central orchestrator that sequences all six layers
into a single self-improving closed loop.

Pipeline flow:
  Data → L1 (Causal) → L2 (Evolutionary) → L3 (XAI) →
  L4 (Neuro-symbolic) → L5 (Physics) → L6 (Agent) → loop back
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import APEXConfig

logger = logging.getLogger("apex.engine")


@dataclass
class LayerResult:
    """Standardised output from any APEX layer."""
    layer_name: str
    status: str  # "success" | "partial" | "failed" | "skipped"
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineState:
    """Mutable state passed between layers so each layer can constrain the next."""
    causal_graph: dict[str, Any] | None = None
    population: list[dict] | None = None
    best_model: Any = None
    shap_values: np.ndarray | None = None
    feature_names: list[str] | None = None
    symbolic_rules: list[dict] | None = None
    physics_violations: list[str] | None = None
    hypotheses: list[str] | None = None
    fitness_functions: list[str] | None = None
    iteration: int = 0
    history: list[LayerResult] = field(default_factory=list)


class APEXEngine:
    """
    Orchestrates the six-layer APEX pipeline.

    Usage::

        config = APEXConfig("config/apex_config.yaml")
        engine = APEXEngine(config)
        engine.load_data(df)
        results = engine.run(n_iterations=1)
    """

    def __init__(self, config: APEXConfig | None = None):
        self.config = config or APEXConfig()
        self.state = PipelineState()
        self._data: pd.DataFrame | None = None
        self._target_col: str | None = None

        self._layers: dict[str, Any] = {}
        self._results: list[LayerResult] = []

        logger.info("APEX Engine initialised (domain=%s)", self.config.domain)

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------
    def load_data(
        self,
        data: pd.DataFrame,
        target_col: str = "binding_affinity",
        feature_cols: list[str] | None = None,
    ) -> "APEXEngine":
        """Load and validate the primary dataset."""
        self._data = data.copy()
        self._target_col = target_col

        if feature_cols:
            self.state.feature_names = feature_cols
        else:
            self.state.feature_names = [
                c for c in data.columns if c != target_col
            ]

        n_rows, n_cols = data.shape
        n_missing = int(data.isnull().sum().sum())
        logger.info(
            "Data loaded: %d rows × %d cols (%d missing values)",
            n_rows, n_cols, n_missing,
        )
        return self

    # ------------------------------------------------------------------
    # Layer registration
    # ------------------------------------------------------------------
    def register_layer(self, name: str, layer_instance: Any) -> "APEXEngine":
        """Register a layer implementation to be called during pipeline execution."""
        self._layers[name] = layer_instance
        logger.info("Registered layer: %s", name)
        return self

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self, n_iterations: int = 1) -> list[LayerResult]:
        """Execute the full APEX pipeline for *n_iterations* self-improving loops."""
        all_results: list[LayerResult] = []

        for iteration in range(1, n_iterations + 1):
            self.state.iteration = iteration
            logger.info("═" * 60)
            logger.info("APEX iteration %d / %d", iteration, n_iterations)
            logger.info("═" * 60)

            layer_order = [
                "causal_discovery",
                "evolutionary_search",
                "explainability",
                "neuro_symbolic",
                "physics_informed",
                "foundation_agent",
            ]

            for layer_name in layer_order:
                result = self._run_layer(layer_name)
                all_results.append(result)
                self.state.history.append(result)

        self._results = all_results
        return all_results

    def run_layer(self, layer_name: str) -> LayerResult:
        """Run a single named layer (useful for interactive / notebook usage)."""
        return self._run_layer(layer_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_layer(self, name: str) -> LayerResult:
        layer = self._layers.get(name)
        if layer is None:
            logger.warning("Layer '%s' not registered — skipping", name)
            return LayerResult(layer_name=name, status="skipped")

        logger.info("▶ Running layer: %s", name)
        t0 = time.perf_counter()
        try:
            result: LayerResult = layer.execute(
                data=self._data,
                target_col=self._target_col,
                state=self.state,
                config=self.config,
            )
            result.elapsed_seconds = round(time.perf_counter() - t0, 3)
            logger.info(
                "✓ %s completed in %.1fs — status=%s",
                name, result.elapsed_seconds, result.status,
            )
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 3)
            logger.exception("✗ %s failed after %.1fs: %s", name, elapsed, exc)
            result = LayerResult(
                layer_name=name,
                status="failed",
                metrics={"error": str(exc)},
                elapsed_seconds=elapsed,
            )
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising all layer results."""
        rows = []
        for r in self._results:
            rows.append({
                "layer": r.layer_name,
                "status": r.status,
                "elapsed_s": r.elapsed_seconds,
                "timestamp": r.timestamp,
                **{f"metric_{k}": v for k, v in r.metrics.items()},
            })
        return pd.DataFrame(rows)

    def export_results(self, output_dir: str | Path = "outputs") -> Path:
        """Persist all results and artifacts to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary_path = out / "pipeline_summary.csv"
        self.summary().to_csv(summary_path, index=False)

        if self.state.causal_graph:
            with open(out / "causal_graph.json", "w") as f:
                json.dump(self.state.causal_graph, f, indent=2)

        if self.state.hypotheses:
            with open(out / "hypotheses_report.md", "w") as f:
                f.write("# APEX Hypotheses Report\n\n")
                for i, h in enumerate(self.state.hypotheses, 1):
                    f.write(f"## Hypothesis {i}\n{h}\n\n")

        logger.info("Results exported to %s", out)
        return out
