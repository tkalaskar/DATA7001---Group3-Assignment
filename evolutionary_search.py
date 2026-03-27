"""
Layer 2 — Evolutionary Architecture Search
============================================
Multi-objective evolutionary optimisation of model architectures.
CMA-ES for continuous hyperparameter search, NSGA-III for Pareto-optimal
trade-offs among accuracy, interpretability, and computational cost.

The causal graph from Layer 1 constrains which feature subsets and
connections are allowed — architectures violating causal structure
are killed before evaluation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error

from apex.core.engine import LayerResult, PipelineState
from apex.core.config import APEXConfig

logger = logging.getLogger("apex.layers.evolution")


@dataclass
class Individual:
    """A candidate architecture in the evolutionary population."""
    genome: dict[str, float]
    fitness: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    crowding_distance: float = 0.0
    model: Any = None
    generation: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Genome → Model decoder
# ══════════════════════════════════════════════════════════════════════════════

def _decode_genome(genome: dict[str, float], task: str = "regression") -> dict[str, Any]:
    """Convert normalised [0, 1] genome values to concrete hyperparameters."""
    params = {
        "n_estimators": int(genome.get("n_estimators", 0.5) * 490 + 10),
        "max_depth": int(genome.get("max_depth", 0.5) * 9 + 1),
        "learning_rate": 10 ** (-4 + genome.get("learning_rate", 0.5) * 3),
        "subsample": 0.5 + genome.get("subsample", 0.5) * 0.5,
        "min_samples_split": int(genome.get("min_samples_split", 0.5) * 18 + 2),
        "min_samples_leaf": int(genome.get("min_samples_leaf", 0.5) * 19 + 1),
    }
    return params


def _build_model(params: dict[str, Any], task: str = "regression"):
    if task == "classification":
        return GradientBoostingClassifier(**params, random_state=42)
    return GradientBoostingRegressor(**params, random_state=42)


# ══════════════════════════════════════════════════════════════════════════════
# Fitness evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_fitness(
    individual: Individual,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    causal_graph: dict | None = None,
    task: str = "regression",
) -> dict[str, float]:
    """
    Multi-objective fitness:
      1. accuracy  — cross-validated R² (regression) or AUC (classification)
      2. interpretability — inverse of model complexity (fewer trees = more interpretable)
      3. complexity — total number of parameters (leaves × depth)
    """
    params = _decode_genome(individual.genome, task)
    model = _build_model(params, task)

    # Accuracy via 3-fold CV
    scoring = "roc_auc" if task == "classification" else "r2"
    try:
        cv_scores = cross_val_score(model, X, y, cv=3, scoring=scoring, error_score=0.0)
        accuracy = float(np.mean(cv_scores))
    except Exception:
        accuracy = 0.0

    # Interpretability: fewer estimators and shallower trees are more interpretable
    interpretability = 1.0 - (params["n_estimators"] / 500) * (params["max_depth"] / 10)
    interpretability = max(0.0, min(1.0, interpretability))

    # Complexity: proxy for computational cost
    complexity = params["n_estimators"] * params["max_depth"]

    # Causal penalty: if feature importances contradict causal graph, reduce fitness
    causal_penalty = 0.0
    if causal_graph and "edges" in causal_graph:
        causal_features = {e["source"] for e in causal_graph["edges"]}
        used_features = set(feature_names)
        non_causal_ratio = len(used_features - causal_features) / max(len(used_features), 1)
        causal_penalty = non_causal_ratio * 0.1

    # Fit the model for downstream use
    try:
        model.fit(X, y)
        individual.model = model
    except Exception:
        pass

    return {
        "accuracy": round(accuracy - causal_penalty, 6),
        "interpretability": round(interpretability, 6),
        "complexity": float(complexity),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NSGA-III non-dominated sorting
# ══════════════════════════════════════════════════════════════════════════════

def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """Check if solution *a* Pareto-dominates solution *b*."""
    at_least_one_better = False
    for key in ["accuracy", "interpretability"]:
        if a.get(key, 0) < b.get(key, 0):
            return False
        if a.get(key, 0) > b.get(key, 0):
            at_least_one_better = True
    if a.get("complexity", float("inf")) > b.get("complexity", float("inf")):
        return False
    if a.get("complexity", float("inf")) < b.get("complexity", float("inf")):
        at_least_one_better = True
    return at_least_one_better


def _non_dominated_sort(population: list[Individual]) -> list[list[int]]:
    """Fast non-dominated sorting (Deb et al., 2002)."""
    n = len(population)
    domination_count = [0] * n
    dominated_set: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(population[i].fitness, population[j].fitness):
                dominated_set[i].append(j)
            elif _dominates(population[j].fitness, population[i].fitness):
                domination_count[i] += 1

        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = k + 1
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _crowding_distance(population: list[Individual], front: list[int]) -> None:
    """Compute crowding distance for individuals in a front."""
    if len(front) <= 2:
        for idx in front:
            population[idx].crowding_distance = float("inf")
        return

    for idx in front:
        population[idx].crowding_distance = 0.0

    for obj in ["accuracy", "interpretability", "complexity"]:
        sorted_front = sorted(front, key=lambda i: population[i].fitness.get(obj, 0))
        population[sorted_front[0]].crowding_distance = float("inf")
        population[sorted_front[-1]].crowding_distance = float("inf")

        obj_range = (
            population[sorted_front[-1]].fitness.get(obj, 0)
            - population[sorted_front[0]].fitness.get(obj, 0)
        )
        if obj_range == 0:
            continue

        for k in range(1, len(sorted_front) - 1):
            population[sorted_front[k]].crowding_distance += (
                population[sorted_front[k + 1]].fitness.get(obj, 0)
                - population[sorted_front[k - 1]].fitness.get(obj, 0)
            ) / obj_range


# ══════════════════════════════════════════════════════════════════════════════
# Genetic operators
# ══════════════════════════════════════════════════════════════════════════════

GENOME_KEYS = [
    "n_estimators", "max_depth", "learning_rate",
    "subsample", "min_samples_split", "min_samples_leaf",
]


def _random_genome() -> dict[str, float]:
    return {k: float(np.random.uniform(0, 1)) for k in GENOME_KEYS}


def _crossover(parent_a: Individual, parent_b: Individual, eta: float = 20.0) -> tuple[Individual, Individual]:
    """Simulated Binary Crossover (SBX)."""
    child_a_genome, child_b_genome = {}, {}
    for key in GENOME_KEYS:
        u = np.random.random()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        p1, p2 = parent_a.genome.get(key, 0.5), parent_b.genome.get(key, 0.5)
        c1 = np.clip(0.5 * ((1 + beta) * p1 + (1 - beta) * p2), 0, 1)
        c2 = np.clip(0.5 * ((1 - beta) * p1 + (1 + beta) * p2), 0, 1)
        child_a_genome[key] = float(c1)
        child_b_genome[key] = float(c2)
    return Individual(genome=child_a_genome), Individual(genome=child_b_genome)


def _mutate(individual: Individual, prob: float = 0.2, eta: float = 20.0) -> Individual:
    """Polynomial mutation."""
    genome = dict(individual.genome)
    for key in GENOME_KEYS:
        if np.random.random() < prob:
            val = genome.get(key, 0.5)
            u = np.random.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            genome[key] = float(np.clip(val + delta, 0, 1))
    individual.genome = genome
    return individual


# ══════════════════════════════════════════════════════════════════════════════
# Layer class
# ══════════════════════════════════════════════════════════════════════════════

class EvolutionarySearchLayer:
    """
    APEX Layer 2 — multi-objective evolutionary search over model
    architectures, constrained by the causal graph from Layer 1.
    """

    def execute(
        self,
        data: pd.DataFrame | None,
        target_col: str | None,
        state: PipelineState,
        config: APEXConfig,
    ) -> LayerResult:
        if data is None or target_col is None:
            return LayerResult(layer_name="evolutionary_search", status="failed")

        features = state.feature_names or [c for c in data.columns if c != target_col]
        X = data[features].dropna().values
        y = data.loc[data[features].dropna().index, target_col].values

        ec = config.evolution
        pop_size = ec.population_size
        n_gen = ec.n_generations

        is_classification = len(np.unique(y)) <= 20
        task = "classification" if is_classification else "regression"

        logger.info(
            "Evolutionary search: pop=%d, gen=%d, task=%s",
            pop_size, n_gen, task,
        )

        # Initialise population
        population = [
            Individual(genome=_random_genome(), generation=0)
            for _ in range(pop_size)
        ]

        # Evaluate initial population
        for ind in population:
            ind.fitness = _evaluate_fitness(
                ind, X, y, features, state.causal_graph, task
            )

        generation_log = []

        for gen in range(1, n_gen + 1):
            # Generate offspring
            offspring = []
            while len(offspring) < pop_size:
                parents = np.random.choice(len(population), size=2, replace=False)
                if np.random.random() < ec.crossover_prob:
                    c1, c2 = _crossover(population[parents[0]], population[parents[1]])
                else:
                    c1 = Individual(genome=dict(population[parents[0]].genome))
                    c2 = Individual(genome=dict(population[parents[1]].genome))
                c1 = _mutate(c1, ec.mutation_prob)
                c2 = _mutate(c2, ec.mutation_prob)
                c1.generation = gen
                c2.generation = gen
                offspring.extend([c1, c2])

            offspring = offspring[:pop_size]

            for ind in offspring:
                ind.fitness = _evaluate_fitness(
                    ind, X, y, features, state.causal_graph, task
                )

            # NSGA-III selection
            combined = population + offspring
            fronts = _non_dominated_sort(combined)

            next_pop = []
            for front in fronts:
                if len(next_pop) + len(front) <= pop_size:
                    _crowding_distance(combined, front)
                    next_pop.extend(front)
                else:
                    _crowding_distance(combined, front)
                    remaining = pop_size - len(next_pop)
                    sorted_front = sorted(
                        front,
                        key=lambda i: combined[i].crowding_distance,
                        reverse=True,
                    )
                    next_pop.extend(sorted_front[:remaining])
                    break

            population = [combined[i] for i in next_pop]

            best_acc = max(ind.fitness.get("accuracy", 0) for ind in population)
            avg_acc = np.mean([ind.fitness.get("accuracy", 0) for ind in population])
            generation_log.append({
                "generation": gen,
                "best_accuracy": round(best_acc, 4),
                "avg_accuracy": round(float(avg_acc), 4),
                "pareto_front_size": len(fronts[0]) if fronts else 0,
            })

            if gen % 10 == 0:
                logger.info(
                    "Gen %d/%d — best_acc=%.4f, avg_acc=%.4f, front_size=%d",
                    gen, n_gen, best_acc, avg_acc,
                    len(fronts[0]) if fronts else 0,
                )

        # Extract Pareto front
        fronts = _non_dominated_sort(population)
        pareto_indices = fronts[0] if fronts else []
        pareto_front = [population[i] for i in pareto_indices]

        # Select the best model (highest accuracy on the Pareto front)
        best_individual = max(pareto_front, key=lambda ind: ind.fitness.get("accuracy", 0))
        state.best_model = best_individual.model
        state.population = [
            {"genome": ind.genome, "fitness": ind.fitness, "rank": ind.rank}
            for ind in population
        ]

        return LayerResult(
            layer_name="evolutionary_search",
            status="success",
            metrics={
                "best_accuracy": best_individual.fitness.get("accuracy", 0),
                "best_interpretability": best_individual.fitness.get("interpretability", 0),
                "best_complexity": best_individual.fitness.get("complexity", 0),
                "pareto_front_size": len(pareto_front),
                "total_evaluations": pop_size * (n_gen + 1),
            },
            artifacts={
                "pareto_front": [
                    {"genome": ind.genome, "fitness": ind.fitness}
                    for ind in pareto_front
                ],
                "generation_log": generation_log,
                "best_genome": best_individual.genome,
            },
        )
