"""
Pydantic schemas for the APEX API — request/response models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    domain: str = Field("drug_discovery", description="Target domain")
    n_iterations: int = Field(1, ge=1, le=10)
    layers: list[str] = Field(
        default=[
            "causal_discovery",
            "evolutionary_search",
            "explainability",
            "neuro_symbolic",
            "physics_informed",
            "foundation_agent",
        ]
    )
    population_size: int = Field(20, ge=4, le=200)
    n_generations: int = Field(30, ge=5, le=500)


class LayerResultResponse(BaseModel):
    layer_name: str
    status: str
    metrics: dict
    elapsed_seconds: float
    timestamp: str


class PipelineStatusResponse(BaseModel):
    status: str
    iteration: int
    layers_completed: list[str]
    layers_remaining: list[str]
    results: list[LayerResultResponse]


# ── Causal ────────────────────────────────────────────────────────────────────

class CausalGraphResponse(BaseModel):
    nodes: list[str]
    edges: list[dict]
    n_edges: int
    validated_edges: list[dict] = []


# ── Evolution ─────────────────────────────────────────────────────────────────

class ParetoFrontResponse(BaseModel):
    pareto_front: list[dict]
    population_size: int
    pareto_front_size: int
    generation_log: list[dict] = []


# ── Explainability ────────────────────────────────────────────────────────────

class SHAPResponse(BaseModel):
    feature_ranking: list[str]
    mean_abs_shap: dict[str, float]
    expected_value: float
    consistency: float
    counterfactuals: list[dict] = []


# ── Dashboard ─────────────────────────────────────────────────────────────────

class DashboardResponse(BaseModel):
    iteration: int
    layers: dict
    summary: dict


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    domain: str
    layers_registered: list[str]
