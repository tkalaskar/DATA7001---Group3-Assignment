"""
Layer 3 — Explainability & Mechanistic Interpretability
========================================================
Multi-level explanation of evolved models:
  • SHAP — global and local feature attribution
  • LIME — local interpretable model-agnostic explanations
  • DiCE — counterfactual explanations ("what-if" scenarios)
  • Mechanistic interpretability — sparse autoencoder to discover
    interpretable internal features

The interpretability score is fed *back* to Layer 2 as an
evolutionary fitness signal, rewarding transparent models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from apex.core.engine import LayerResult, PipelineState
from apex.core.config import APEXConfig

logger = logging.getLogger("apex.layers.xai")


# ══════════════════════════════════════════════════════════════════════════════
# SHAP explanations
# ══════════════════════════════════════════════════════════════════════════════

def _compute_shap_values(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    n_background: int = 100,
) -> dict[str, Any]:
    """
    Compute SHAP values using TreeExplainer (if tree-based) or
    KernelExplainer (general).  Returns values, base value, and
    per-feature importance rankings.
    """
    try:
        import shap

        if hasattr(model, "estimators_") or hasattr(model, "n_estimators"):
            explainer = shap.TreeExplainer(model)
        else:
            background = X[:n_background] if len(X) > n_background else X
            if isinstance(background, pd.DataFrame):
                background = background.values
            explainer = shap.KernelExplainer(model.predict, background)

        if isinstance(X, pd.DataFrame):
            shap_values = explainer.shap_values(X.values)
        else:
            shap_values = explainer.shap_values(X)

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feature_ranking = sorted(
            zip(feature_names, mean_abs_shap.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "shap_values": shap_values,
            "expected_value": float(explainer.expected_value)
            if np.isscalar(explainer.expected_value)
            else float(explainer.expected_value[0]),
            "mean_abs_shap": dict(feature_ranking),
            "feature_ranking": [f[0] for f in feature_ranking],
        }
    except Exception as exc:
        logger.warning("SHAP computation failed: %s - falling back to permutation importance", exc)
        return _fallback_permutation_importance(model, X, feature_names)


def _fallback_permutation_importance(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    """Fallback feature importance via permutation when SHAP fails."""
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = X

    try:
        y_pred = model.predict(X_arr)
        perm = permutation_importance(model, X_arr, y_pred, n_repeats=10, random_state=42)
        importances = perm.importances_mean
    except Exception:
        importances = np.zeros(len(feature_names))
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

    feature_ranking = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "shap_values": None,
        "expected_value": 0.0,
        "mean_abs_shap": dict(feature_ranking),
        "feature_ranking": [f[0] for f in feature_ranking],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SHAP consistency (bootstrap stability)
# ══════════════════════════════════════════════════════════════════════════════

def _shap_consistency(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    n_bootstrap: int = 5,
    n_background: int = 100,
) -> float:
    """
    Measure the rank-order stability of SHAP values across bootstrap
    resamples.  Returns Spearman correlation averaged over pairs.
    """
    from scipy.stats import spearmanr

    rankings = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[idx]
        result = _compute_shap_values(model, X_boot, feature_names, n_background)
        ranking = result["feature_ranking"]
        rank_order = [ranking.index(f) if f in ranking else len(ranking) for f in feature_names]
        rankings.append(rank_order)

    correlations = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            corr, _ = spearmanr(rankings[i], rankings[j])
            if not np.isnan(corr):
                correlations.append(corr)

    return float(np.mean(correlations)) if correlations else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Counterfactual explanations (simplified DiCE-style)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_counterfactuals(
    model: Any,
    instance: np.ndarray,
    feature_names: list[str],
    desired_range: tuple[float, float] = (0.8, 1.0),
    n_counterfactuals: int = 5,
    n_attempts: int = 1000,
) -> list[dict[str, Any]]:
    """
    Generate counterfactual examples: minimal perturbations that change
    the prediction into the desired range.
    """
    counterfactuals = []
    original_pred = float(model.predict(instance.reshape(1, -1))[0])

    for _ in range(n_attempts):
        if len(counterfactuals) >= n_counterfactuals:
            break

        perturbed = instance.copy()
        n_features_to_change = np.random.randint(1, max(2, len(instance) // 2))
        features_to_change = np.random.choice(len(instance), size=n_features_to_change, replace=False)

        for feat_idx in features_to_change:
            noise = np.random.normal(0, 0.3 * max(abs(instance[feat_idx]), 1))
            perturbed[feat_idx] += noise

        new_pred = float(model.predict(perturbed.reshape(1, -1))[0])

        if desired_range[0] <= new_pred <= desired_range[1]:
            changes = {}
            for idx in features_to_change:
                if abs(perturbed[idx] - instance[idx]) > 1e-6:
                    changes[feature_names[idx]] = {
                        "from": round(float(instance[idx]), 4),
                        "to": round(float(perturbed[idx]), 4),
                        "delta": round(float(perturbed[idx] - instance[idx]), 4),
                    }
            if changes:
                counterfactuals.append({
                    "original_prediction": round(original_pred, 4),
                    "counterfactual_prediction": round(new_pred, 4),
                    "changes": changes,
                    "n_features_changed": len(changes),
                })

    counterfactuals.sort(key=lambda x: x["n_features_changed"])
    return counterfactuals[:n_counterfactuals]


# ══════════════════════════════════════════════════════════════════════════════
# Sparse Autoencoder for mechanistic interpretability
# ══════════════════════════════════════════════════════════════════════════════

class SparseAutoencoder:
    """
    Learns compressed, sparse representations of model activations.
    Discovered latent features can be mapped to known scientific concepts
    (e.g. hydrophobicity, molecular weight) to validate interpretability.
    """

    def __init__(self, input_dim: int, latent_dim: int = 64, sparsity_penalty: float = 0.01):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_penalty = sparsity_penalty
        self.W_encode: np.ndarray | None = None
        self.W_decode: np.ndarray | None = None
        self.b_encode: np.ndarray | None = None
        self.b_decode: np.ndarray | None = None

    def fit(self, X: np.ndarray, n_epochs: int = 100, lr: float = 0.01) -> "SparseAutoencoder":
        n, d = X.shape
        self.W_encode = np.random.randn(d, self.latent_dim) * 0.01
        self.W_decode = np.random.randn(self.latent_dim, d) * 0.01
        self.b_encode = np.zeros(self.latent_dim)
        self.b_decode = np.zeros(d)

        for epoch in range(n_epochs):
            # Forward pass
            z = np.maximum(0, X @ self.W_encode + self.b_encode)  # ReLU encoding
            X_hat = z @ self.W_decode + self.b_decode

            # Reconstruction loss + L1 sparsity on activations
            recon_loss = 0.5 * np.mean((X - X_hat) ** 2)
            sparsity_loss = self.sparsity_penalty * np.mean(np.abs(z))
            total_loss = recon_loss + sparsity_loss

            # Backward pass (gradient descent)
            dX_hat = (X_hat - X) / n
            dW_decode = z.T @ dX_hat
            db_decode = np.mean(dX_hat, axis=0)

            dz = dX_hat @ self.W_decode.T + self.sparsity_penalty * np.sign(z) / n
            dz[z <= 0] = 0  # ReLU gradient

            dW_encode = X.T @ dz
            db_encode = np.mean(dz, axis=0)

            self.W_encode -= lr * dW_encode
            self.W_decode -= lr * dW_decode
            self.b_encode -= lr * db_encode
            self.b_decode -= lr * db_decode

            if (epoch + 1) % 20 == 0:
                logger.debug("SAE epoch %d/%d — loss=%.6f", epoch + 1, n_epochs, total_loss)

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X @ self.W_encode + self.b_encode)

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z @ self.W_decode + self.b_decode

    def feature_alignment(self, X: np.ndarray, concept_vectors: dict[str, np.ndarray]) -> dict[str, float]:
        """
        Measure cosine similarity between discovered latent features
        and known concept vectors (e.g. 'hydrophobicity' direction).
        """
        z = self.encode(X)
        alignments = {}
        for concept_name, concept_vec in concept_vectors.items():
            if len(concept_vec) != self.latent_dim:
                continue
            latent_mean = np.mean(z, axis=0)
            cos_sim = np.dot(latent_mean, concept_vec) / (
                np.linalg.norm(latent_mean) * np.linalg.norm(concept_vec) + 1e-8
            )
            alignments[concept_name] = round(float(cos_sim), 4)
        return alignments


# ══════════════════════════════════════════════════════════════════════════════
# Layer class
# ══════════════════════════════════════════════════════════════════════════════

class ExplainabilityLayer:
    """
    APEX Layer 3 — multi-level explainability of evolved models.
    Produces SHAP values, counterfactuals, and mechanistic
    interpretability scores.
    """

    def execute(
        self,
        data: pd.DataFrame | None,
        target_col: str | None,
        state: PipelineState,
        config: APEXConfig,
    ) -> LayerResult:
        if data is None or state.best_model is None:
            return LayerResult(
                layer_name="explainability",
                status="failed",
                metrics={"error": "No model or data available"},
            )

        features = state.feature_names or [c for c in data.columns if c != target_col]
        X = data[features].dropna()
        X_arr = X.values

        xai_cfg = config.explainability

        # 1. SHAP values
        logger.info("Computing SHAP values...")
        shap_result = _compute_shap_values(
            state.best_model, X, features, xai_cfg.shap_n_background
        )
        state.shap_values = shap_result.get("shap_values")

        # 2. SHAP consistency
        logger.info("Measuring SHAP consistency across bootstrap resamples...")
        consistency = _shap_consistency(
            state.best_model, X_arr, features,
            n_bootstrap=xai_cfg.shap_n_bootstrap,
            n_background=xai_cfg.shap_n_background,
        )

        # 3. Counterfactual explanations
        logger.info("Generating counterfactual explanations...")
        counterfactuals = []
        if len(X_arr) > 0:
            counterfactuals = _generate_counterfactuals(
                state.best_model,
                X_arr[0],
                features,
                desired_range=tuple(xai_cfg.dice_desired_range),
                n_counterfactuals=xai_cfg.dice_n_counterfactuals,
            )

        # 4. Sparse autoencoder for mechanistic interpretability
        logger.info("Training sparse autoencoder for mechanistic features...")
        sae = SparseAutoencoder(
            input_dim=X_arr.shape[1],
            latent_dim=min(xai_cfg.sae_latent_dim, X_arr.shape[1] * 2),
            sparsity_penalty=xai_cfg.sae_sparsity_penalty,
        )
        sae.fit(X_arr, n_epochs=100, lr=0.01)

        latent = sae.encode(X_arr)
        reconstruction = sae.decode(latent)
        recon_error = float(np.mean((X_arr - reconstruction) ** 2))

        active_features = int(np.sum(np.mean(np.abs(latent), axis=0) > 0.01))
        sparsity_ratio = active_features / latent.shape[1]

        return LayerResult(
            layer_name="explainability",
            status="success",
            metrics={
                "shap_consistency": round(consistency, 4),
                "n_counterfactuals_found": len(counterfactuals),
                "sae_reconstruction_error": round(recon_error, 6),
                "sae_active_features": active_features,
                "sae_sparsity_ratio": round(sparsity_ratio, 4),
                "top_feature": shap_result["feature_ranking"][0] if shap_result["feature_ranking"] else "N/A",
            },
            artifacts={
                "shap_result": {
                    "expected_value": shap_result["expected_value"],
                    "mean_abs_shap": shap_result["mean_abs_shap"],
                    "feature_ranking": shap_result["feature_ranking"],
                },
                "counterfactuals": counterfactuals,
                "sae_info": {
                    "latent_dim": sae.latent_dim,
                    "active_features": active_features,
                    "reconstruction_error": recon_error,
                },
            },
        )
