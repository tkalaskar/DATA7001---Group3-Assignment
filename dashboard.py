"""
Dashboard data generator — prepares all visualization data
for both the Matplotlib static exports and the React frontend.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from apex.core.engine import PipelineState

logger = logging.getLogger("apex.viz.dashboard")


class DashboardGenerator:
    """Generates complete dashboard data from pipeline state."""

    def generate(self, state: PipelineState) -> dict[str, Any]:
        """Produce the full JSON payload for the frontend dashboard."""
        dashboard: dict[str, Any] = {
            "iteration": state.iteration,
            "layers": {},
            "summary": {},
        }

        # Causal graph
        if state.causal_graph:
            dashboard["layers"]["causal_discovery"] = {
                "nodes": state.causal_graph.get("nodes", []),
                "edges": state.causal_graph.get("edges", []),
                "n_edges": len(state.causal_graph.get("edges", [])),
            }

        # Evolution / Pareto
        if state.population:
            pareto = [p for p in state.population if p.get("rank", 99) == 0]
            dashboard["layers"]["evolutionary_search"] = {
                "population_size": len(state.population),
                "pareto_front_size": len(pareto),
                "pareto_front": pareto[:50],
                "all_solutions": [
                    {"fitness": p["fitness"], "rank": p.get("rank", -1)}
                    for p in state.population
                ],
            }

        # XAI
        for result in state.history:
            if result.layer_name == "explainability" and result.status == "success":
                dashboard["layers"]["explainability"] = {
                    "metrics": result.metrics,
                    "shap_result": result.artifacts.get("shap_result", {}),
                    "counterfactuals": result.artifacts.get("counterfactuals", []),
                    "sae_info": result.artifacts.get("sae_info", {}),
                }
                break

        # Neuro-symbolic
        for result in state.history:
            if result.layer_name == "neuro_symbolic" and result.status == "success":
                dashboard["layers"]["neuro_symbolic"] = {
                    "metrics": result.metrics,
                    "per_rule_compliance": result.artifacts.get("per_rule_compliance", {}),
                    "rules_applied": result.artifacts.get("rules_applied", []),
                }
                break

        # Physics
        if state.physics_violations:
            dashboard["layers"]["physics_informed"] = {
                "violations": state.physics_violations,
            }

        # Agent hypotheses
        if state.hypotheses:
            dashboard["layers"]["foundation_agent"] = {
                "hypotheses": state.hypotheses,
                "new_rules": state.symbolic_rules or [],
                "fitness_suggestions": state.fitness_functions or [],
            }

        # Summary metrics from history
        for result in state.history:
            if result.status == "success":
                dashboard["summary"][result.layer_name] = {
                    "status": result.status,
                    "elapsed_seconds": result.elapsed_seconds,
                    "key_metrics": {
                        k: v for k, v in result.metrics.items()
                        if isinstance(v, (int, float, str, bool))
                    },
                }

        return dashboard

    def export_json(self, state: PipelineState, output_path: str | Path = "outputs/dashboard.json") -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.generate(state)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

        logger.info("Dashboard JSON exported to %s", path)
        return path
