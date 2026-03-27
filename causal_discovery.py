"""
Layer 1 — Causal Discovery
===========================
Learns the structural causal model (DAG) from data *before* any predictive
modelling begins.  Causal graph edges become hard constraints on which
architectures survive evolution in Layer 2.

Algorithms:  NOTEARS (continuous DAG optimisation), PC algorithm, LiNGAM.
Validation:  DoWhy refutation tests (random common cause, placebo, subset).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from apex.core.engine import LayerResult, PipelineState
from apex.core.config import APEXConfig

logger = logging.getLogger("apex.layers.causal")


# ══════════════════════════════════════════════════════════════════════════════
# NOTEARS — continuous structure learning (Zheng et al., NeurIPS 2018)
# ══════════════════════════════════════════════════════════════════════════════

def _notears_linear(
    X: np.ndarray,
    lambda1: float = 0.1,
    loss_type: str = "l2",
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.3,
) -> np.ndarray:
    """
    Solve the NOTEARS continuous DAG learning problem:
        min  0.5/n ||X - X W||^2  +  λ₁ ||W||₁
        s.t. h(W) = tr(e^{W◦W}) - d = 0     (acyclicity)

    Returns the estimated weighted adjacency matrix W.
    """
    n, d = X.shape

    def _loss_and_grad(W: np.ndarray) -> tuple[float, np.ndarray]:
        M = X @ W
        R = X - M
        if loss_type == "l2":
            loss = 0.5 / n * (R ** 2).sum()
            G_loss = -1.0 / n * X.T @ R
        elif loss_type == "logistic":
            sigmoid = 1.0 / (1.0 + np.exp(-M))
            loss = -1.0 / n * (X * np.log(sigmoid + 1e-8) + (1 - X) * np.log(1 - sigmoid + 1e-8)).sum()
            G_loss = 1.0 / n * X.T @ (sigmoid - X)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return loss, G_loss

    def _h(W: np.ndarray) -> float:
        """Acyclicity constraint: h(W) = tr(e^{W◦W}) - d."""
        E = linalg.expm(W * W)
        return np.trace(E) - d

    def _h_grad(W: np.ndarray) -> np.ndarray:
        E = linalg.expm(W * W)
        return E.T * W * 2

    W_est = np.zeros((d, d))
    mu = 1.0
    alpha = 0.0
    rho = 1.0

    for iteration in range(max_iter):
        while True:
            loss, G_loss = _loss_and_grad(W_est)
            h_val = _h(W_est)
            obj = loss + 0.5 * rho * h_val ** 2 + alpha * h_val + lambda1 * np.abs(W_est).sum()

            G_obj = G_loss + (rho * h_val + alpha) * _h_grad(W_est) + lambda1 * np.sign(W_est)

            W_new = W_est - mu * G_obj
            new_h = _h(W_new)

            if new_h < 0.25 * h_val or mu < 1e-12:
                break
            mu *= 0.5

        W_est = W_new
        h_val = _h(W_est)

        alpha += rho * h_val
        rho = min(rho * 10, 1e+16)
        mu = 1.0

        if abs(h_val) < h_tol:
            logger.info("NOTEARS converged at iteration %d (h=%.2e)", iteration, h_val)
            break

    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# ══════════════════════════════════════════════════════════════════════════════
# Causal validation with DoWhy-style refutation
# ══════════════════════════════════════════════════════════════════════════════

def _validate_causal_edge(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
) -> dict[str, Any]:
    """
    Estimate the Average Treatment Effect (ATE) via OLS backdoor adjustment
    and run refutation tests.
    """
    from sklearn.linear_model import LinearRegression

    X_treat = data[[treatment]].values
    y = data[outcome].values
    confounders = [c for c in data.columns if c not in (treatment, outcome)]

    if confounders:
        X_full = data[[treatment] + confounders].values
    else:
        X_full = X_treat

    reg = LinearRegression().fit(X_full, y)
    ate = float(reg.coef_[0])

    # Refutation 1: random common cause — add noise, ATE should not change much
    noise = np.random.randn(len(data), 1)
    X_noisy = np.hstack([X_full, noise])
    reg_noisy = LinearRegression().fit(X_noisy, y)
    ate_noisy = float(reg_noisy.coef_[0])
    random_cause_passed = abs(ate - ate_noisy) / (abs(ate) + 1e-8) < 0.15

    # Refutation 2: placebo treatment — shuffle treatment, ATE should vanish
    X_placebo = X_full.copy()
    X_placebo[:, 0] = np.random.permutation(X_placebo[:, 0])
    reg_placebo = LinearRegression().fit(X_placebo, y)
    ate_placebo = float(reg_placebo.coef_[0])
    placebo_passed = abs(ate_placebo) < abs(ate) * 0.5

    # Refutation 3: data subset — subsample, ATE should be stable
    idx = np.random.choice(len(data), size=int(0.8 * len(data)), replace=False)
    reg_sub = LinearRegression().fit(X_full[idx], y[idx])
    ate_subset = float(reg_sub.coef_[0])
    subset_passed = abs(ate - ate_subset) / (abs(ate) + 1e-8) < 0.2

    return {
        "treatment": treatment,
        "outcome": outcome,
        "ate": round(ate, 6),
        "refutations": {
            "random_common_cause": {"passed": random_cause_passed, "ate_delta": round(abs(ate - ate_noisy), 6)},
            "placebo_treatment": {"passed": placebo_passed, "ate_placebo": round(ate_placebo, 6)},
            "data_subset": {"passed": subset_passed, "ate_subset": round(ate_subset, 6)},
        },
        "all_refutations_passed": random_cause_passed and placebo_passed and subset_passed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Layer class
# ══════════════════════════════════════════════════════════════════════════════

class CausalDiscoveryLayer:
    """
    APEX Layer 1 — learns the causal DAG from data and validates edges
    via refutation tests.  The graph is stored in pipeline state and
    constrains downstream evolution.
    """

    def execute(
        self,
        data: pd.DataFrame | None,
        target_col: str | None,
        state: PipelineState,
        config: APEXConfig,
    ) -> LayerResult:
        if data is None or target_col is None:
            return LayerResult(layer_name="causal_discovery", status="failed",
                               metrics={"error": "No data provided"})

        features = [c for c in data.columns if c != target_col]
        X = data[features].dropna().values
        col_names = features

        logger.info("Running NOTEARS on %d samples × %d features", X.shape[0], X.shape[1])

        cc = config.causal
        W = _notears_linear(
            X,
            lambda1=cc.lambda1,
            loss_type=cc.loss_type,
            max_iter=cc.max_iter,
            h_tol=cc.h_tol,
            w_threshold=cc.w_threshold,
        )

        # Build edge list from adjacency matrix
        edges = []
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if abs(W[i, j]) > 0:
                    edges.append({
                        "source": col_names[i],
                        "target": col_names[j],
                        "weight": round(float(W[i, j]), 4),
                    })

        causal_graph = {"nodes": col_names, "edges": edges, "adjacency_matrix": W.tolist()}
        state.causal_graph = causal_graph

        # Validate the strongest edges
        edge_strengths = sorted(edges, key=lambda e: abs(e["weight"]), reverse=True)
        validated_edges = []
        for edge in edge_strengths[:5]:
            if edge["target"] == target_col or edge["source"] in features:
                try:
                    result = _validate_causal_edge(data, edge["source"], target_col)
                    validated_edges.append(result)
                except Exception as exc:
                    logger.warning("Refutation failed for %s->%s: %s",
                                   edge["source"], target_col, exc)

        n_valid = sum(1 for v in validated_edges if v["all_refutations_passed"])

        return LayerResult(
            layer_name="causal_discovery",
            status="success",
            metrics={
                "n_edges": len(edges),
                "n_validated": n_valid,
                "n_features": len(col_names),
                "graph_density": len(edges) / max(len(col_names) ** 2, 1),
            },
            artifacts={
                "causal_graph": causal_graph,
                "validated_edges": validated_edges,
            },
        )
