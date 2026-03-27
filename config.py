"""
Centralised configuration management for the APEX engine.
Loads YAML config and environment variables, validates constraints,
and exposes typed accessors to every layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CausalConfig:
    algorithm: str = "notears"
    lambda1: float = 0.1
    loss_type: str = "l2"
    max_iter: int = 100
    h_tol: float = 1e-8
    w_threshold: float = 0.3
    refutation_tests: list[str] = field(
        default_factory=lambda: ["random_common_cause", "placebo_treatment", "data_subset"]
    )
    confidence_level: float = 0.05


@dataclass
class ObjectiveSpec:
    name: str = "accuracy"
    weight: float = 1.0
    target: str = "maximize"


@dataclass
class EvolutionConfig:
    algorithm: str = "nsga3"
    population_size: int = 40
    n_generations: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    objectives: list[ObjectiveSpec] = field(default_factory=list)
    architecture_space: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplainabilityConfig:
    shap_n_background: int = 100
    shap_max_features: int = 20
    shap_consistency_threshold: float = 0.9
    shap_n_bootstrap: int = 5
    lime_n_features: int = 10
    lime_n_samples: int = 5000
    dice_n_counterfactuals: int = 5
    dice_desired_range: list[float] = field(default_factory=lambda: [0.8, 1.0])
    sae_latent_dim: int = 64
    sae_sparsity_penalty: float = 0.01
    sae_alignment_threshold: float = 0.7


@dataclass
class AgentConfig:
    model: str = "gemini-2.0-flash"
    fallback_model: str = "gemini-2.0-flash"
    max_tokens: int = 4096
    temperature: float = 0.3
    max_iterations: int = 5


@dataclass
class PhysicsConfig:
    conservation_laws: list[str] = field(
        default_factory=lambda: ["mass_balance", "energy_conservation"]
    )
    pde_loss_weight: float = 1.0
    boundary_loss_weight: float = 10.0
    violation_threshold: float = 0.01
    kill_on_violation: bool = True


@dataclass
class EvaluationConfig:
    confidence: float = 0.95
    mc_dropout_samples: int = 50
    uncertainty_method: str = "conformal_prediction"


class APEXConfig:
    """Loads and validates the full APEX configuration tree."""

    def __init__(self, config_path: str | Path = "config/apex_config.yaml"):
        self._path = Path(config_path)
        self._raw: dict[str, Any] = {}
        self._load()

        self.causal = self._build_causal()
        self.evolution = self._build_evolution()
        self.explainability = self._build_explainability()
        self.physics = self._build_physics()
        self.agent = self._build_agent()
        self.evaluation = self._build_evaluation()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, "r") as f:
                self._raw = yaml.safe_load(f) or {}
        else:
            self._raw = {}

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Access nested config values via dot-notation, e.g. 'evolution.population_size'."""
        keys = dotpath.split(".")
        node = self._raw
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    @property
    def domain(self) -> str:
        return self.get("project.domain", "drug_discovery")

    @property
    def data_paths(self) -> dict[str, str]:
        return self.get(f"data.{self.domain}", {})

    @property
    def api_host(self) -> str:
        return os.getenv("APEX_API_HOST", self.get("api.host", "0.0.0.0"))

    @property
    def api_port(self) -> int:
        return int(os.getenv("APEX_API_PORT", self.get("api.port", 8000)))

    @property
    def anthropic_api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def gemini_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY")

    # ------------------------------------------------------------------
    def _build_causal(self) -> CausalConfig:
        c = self.get("causal", {})
        return CausalConfig(
            algorithm=c.get("algorithm", "notears"),
            lambda1=c.get("lambda1", 0.1),
            loss_type=c.get("loss_type", "l2"),
            max_iter=c.get("max_iter", 100),
            h_tol=c.get("h_tol", 1e-8),
            w_threshold=c.get("w_threshold", 0.3),
            refutation_tests=c.get("refutation_tests", ["random_common_cause"]),
            confidence_level=c.get("confidence_level", 0.05),
        )

    def _build_evolution(self) -> EvolutionConfig:
        e = self.get("evolution", {})
        objectives = [
            ObjectiveSpec(**obj) for obj in e.get("objectives", [])
        ]
        return EvolutionConfig(
            algorithm=e.get("algorithm", "nsga3"),
            population_size=e.get("population_size", 40),
            n_generations=e.get("n_generations", 100),
            crossover_prob=e.get("crossover_prob", 0.9),
            mutation_prob=e.get("mutation_prob", 0.2),
            objectives=objectives,
            architecture_space=e.get("architecture_space", {}),
        )

    def _build_explainability(self) -> ExplainabilityConfig:
        x = self.get("explainability", {})
        shap_cfg = x.get("shap", {})
        lime_cfg = x.get("lime", {})
        dice_cfg = x.get("dice", {})
        mech = x.get("mechanistic", {}).get("sparse_autoencoder", {})
        return ExplainabilityConfig(
            shap_n_background=shap_cfg.get("n_background_samples", 100),
            shap_max_features=shap_cfg.get("max_display_features", 20),
            shap_consistency_threshold=shap_cfg.get("consistency_threshold", 0.9),
            shap_n_bootstrap=shap_cfg.get("n_bootstrap", 5),
            lime_n_features=lime_cfg.get("n_features", 10),
            lime_n_samples=lime_cfg.get("n_samples", 5000),
            dice_n_counterfactuals=dice_cfg.get("n_counterfactuals", 5),
            dice_desired_range=dice_cfg.get("desired_range", [0.8, 1.0]),
            sae_latent_dim=mech.get("latent_dim", 64),
            sae_sparsity_penalty=mech.get("sparsity_penalty", 0.01),
            sae_alignment_threshold=mech.get("alignment_threshold", 0.7),
        )

    def _build_physics(self) -> PhysicsConfig:
        p = self.get("physics", {})
        return PhysicsConfig(
            conservation_laws=p.get("conservation_laws", []),
            pde_loss_weight=p.get("pde_loss_weight", 1.0),
            boundary_loss_weight=p.get("boundary_loss_weight", 10.0),
            violation_threshold=p.get("violation_threshold", 0.01),
            kill_on_violation=p.get("kill_on_violation", True),
        )

    def _build_agent(self) -> AgentConfig:
        a = self.get("agent", {})
        return AgentConfig(
            model=a.get("model", "gemini-2.0-flash"),
            fallback_model=a.get("fallback_model", "gemini-2.0-flash"),
            max_tokens=a.get("max_tokens", 4096),
            temperature=a.get("temperature", 0.3),
            max_iterations=a.get("max_iterations", 5),
        )

    def _build_evaluation(self) -> EvaluationConfig:
        ev = self.get("evaluation", {})
        unc = ev.get("uncertainty", {})
        return EvaluationConfig(
            confidence=unc.get("confidence", 0.95),
            mc_dropout_samples=unc.get("mc_dropout_samples", 50),
            uncertainty_method=unc.get("method", "conformal_prediction"),
        )
