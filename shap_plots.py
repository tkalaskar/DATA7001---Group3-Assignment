"""
SHAP visualization — waterfall charts, beeswarm plots,
and feature importance bar charts with APEX styling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("apex.viz.shap")

APEX_COLORS = {
    "positive": "#E17055",
    "negative": "#00B894",
    "bar": "#6C5CE7",
    "background": "#0A0A1A",
    "surface": "#1A1A2E",
    "text": "#EAEAEA",
    "grid": "#2d3436",
}


class SHAPVisualizer:
    """Publication-quality SHAP explanation visualizations."""

    def waterfall(
        self,
        shap_values: np.ndarray,
        expected_value: float,
        feature_names: list[str],
        instance_idx: int = 0,
        max_display: int = 15,
        output_path: str | Path | None = None,
        title: str = "APEX XAI — SHAP Waterfall Explanation",
    ) -> Any:
        """
        Horizontal waterfall chart showing how each feature pushes
        the prediction from the base value.
        """
        import matplotlib.pyplot as plt

        if shap_values.ndim > 1:
            values = shap_values[instance_idx]
        else:
            values = shap_values

        indices = np.argsort(np.abs(values))[::-1][:max_display]
        sorted_vals = values[indices]
        sorted_names = [feature_names[i] for i in indices]

        sorted_vals = sorted_vals[::-1]
        sorted_names = sorted_names[::-1]

        fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_names) * 0.4)),
                                facecolor=APEX_COLORS["background"])
        ax.set_facecolor(APEX_COLORS["background"])

        colors = [APEX_COLORS["positive"] if v > 0 else APEX_COLORS["negative"]
                  for v in sorted_vals]

        bars = ax.barh(range(len(sorted_names)), sorted_vals, color=colors, alpha=0.85, height=0.6)

        for bar, val in zip(bars, sorted_vals):
            x_pos = bar.get_width()
            ha = "left" if val >= 0 else "right"
            offset = 0.01 * max(abs(sorted_vals)) * (1 if val >= 0 else -1)
            ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}", va="center", ha=ha,
                    color=APEX_COLORS["text"], fontsize=9, fontweight="bold")

        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=11, color=APEX_COLORS["text"])
        ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=12,
                       color=APEX_COLORS["text"], fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold",
                      color=APEX_COLORS["text"], pad=20)
        ax.axvline(x=0, color=APEX_COLORS["grid"], linewidth=1.5, linestyle="-")
        ax.tick_params(axis="x", colors=APEX_COLORS["text"])
        ax.spines["bottom"].set_color(APEX_COLORS["grid"])
        ax.spines["left"].set_color(APEX_COLORS["grid"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.text(0.02, 0.98, f"E[f(x)] = {expected_value:.4f}",
                transform=ax.transAxes, fontsize=10, color=APEX_COLORS["text"],
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=APEX_COLORS["surface"],
                          edgecolor=APEX_COLORS["bar"], alpha=0.8))

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300,
                        facecolor=APEX_COLORS["background"], bbox_inches="tight")
            logger.info("SHAP waterfall saved to %s", output_path)

        return fig

    def importance_bar(
        self,
        mean_abs_shap: dict[str, float],
        output_path: str | Path | None = None,
        title: str = "APEX XAI — Global Feature Importance (mean |SHAP|)",
        max_display: int = 15,
    ) -> Any:
        """Horizontal bar chart of global mean absolute SHAP values."""
        import matplotlib.pyplot as plt

        items = sorted(mean_abs_shap.items(), key=lambda x: x[1], reverse=True)[:max_display]
        items = items[::-1]
        names = [i[0] for i in items]
        values = [i[1] for i in items]

        fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.4)),
                                facecolor=APEX_COLORS["background"])
        ax.set_facecolor(APEX_COLORS["background"])

        bars = ax.barh(range(len(names)), values, color=APEX_COLORS["bar"],
                       alpha=0.85, height=0.6, edgecolor=APEX_COLORS["bar"])

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.005 * max(values), bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", color=APEX_COLORS["text"], fontsize=9)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=11, color=APEX_COLORS["text"])
        ax.set_xlabel("Mean |SHAP Value|", fontsize=12,
                       color=APEX_COLORS["text"], fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold",
                      color=APEX_COLORS["text"], pad=20)
        ax.tick_params(axis="x", colors=APEX_COLORS["text"])
        ax.spines["bottom"].set_color(APEX_COLORS["grid"])
        ax.spines["left"].set_color(APEX_COLORS["grid"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300,
                        facecolor=APEX_COLORS["background"], bbox_inches="tight")

        return fig

    def to_plotly_waterfall(
        self,
        shap_values: np.ndarray,
        expected_value: float,
        feature_names: list[str],
        instance_idx: int = 0,
        max_display: int = 15,
    ) -> dict:
        """Return Plotly-compatible JSON for waterfall chart."""
        if shap_values.ndim > 1:
            values = shap_values[instance_idx]
        else:
            values = shap_values

        indices = np.argsort(np.abs(values))[::-1][:max_display]
        sorted_vals = [float(values[i]) for i in indices]
        sorted_names = [feature_names[i] for i in indices]

        colors = [APEX_COLORS["positive"] if v > 0 else APEX_COLORS["negative"]
                  for v in sorted_vals]

        return {
            "type": "bar",
            "orientation": "h",
            "x": sorted_vals,
            "y": sorted_names,
            "marker": {"color": colors},
            "expected_value": expected_value,
        }
