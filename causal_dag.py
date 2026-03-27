"""
Causal DAG visualization — directed acyclic graph of discovered
causal relationships with edge weights coloured by ATE magnitude.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("apex.viz.dag")


class CausalDAGVisualizer:
    """Renders causal graphs as publication-quality DAG diagrams."""

    APEX_COLORS = {
        "node": "#6C5CE7",
        "edge_positive": "#00B894",
        "edge_negative": "#E17055",
        "background": "#0A0A1A",
        "text": "#EAEAEA",
        "highlight": "#00CEC9",
    }

    def plot(
        self,
        causal_graph: dict[str, Any],
        output_path: str | Path | None = None,
        title: str = "APEX Causal Discovery — Structural Causal Model",
        figsize: tuple[int, int] = (14, 10),
    ) -> Any:
        """
        Render the causal DAG using NetworkX and Matplotlib.

        Parameters
        ----------
        causal_graph : dict with 'nodes' and 'edges' keys
        output_path : path to save the figure (optional)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx

        nodes = causal_graph.get("nodes", [])
        edges = causal_graph.get("edges", [])

        G = nx.DiGraph()
        G.add_nodes_from(nodes)

        edge_weights = []
        edge_colors = []
        for e in edges:
            w = e.get("weight", 0)
            G.add_edge(e["source"], e["target"], weight=abs(w))
            edge_weights.append(abs(w))
            edge_colors.append(
                self.APEX_COLORS["edge_positive"] if w > 0
                else self.APEX_COLORS["edge_negative"]
            )

        fig, ax = plt.subplots(figsize=figsize, facecolor=self.APEX_COLORS["background"])
        ax.set_facecolor(self.APEX_COLORS["background"])

        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

        max_w = max(edge_weights) if edge_weights else 1
        widths = [1 + 5 * (w / max_w) for w in edge_weights]

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            width=widths,
            arrows=True,
            arrowsize=25,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            alpha=0.8,
        )

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=self.APEX_COLORS["node"],
            node_size=2000,
            edgecolors=self.APEX_COLORS["highlight"],
            linewidths=2,
            alpha=0.9,
        )

        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=10,
            font_weight="bold",
            font_color=self.APEX_COLORS["text"],
            font_family="monospace",
        )

        edge_labels = {
            (e["source"], e["target"]): f'{e["weight"]:.3f}'
            for e in edges
        }
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=8,
            font_color=self.APEX_COLORS["text"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=self.APEX_COLORS["background"], alpha=0.7),
        )

        ax.set_title(
            title,
            fontsize=16, fontweight="bold",
            color=self.APEX_COLORS["text"],
            pad=20,
        )

        positive_patch = mpatches.Patch(color=self.APEX_COLORS["edge_positive"], label="Positive causal effect")
        negative_patch = mpatches.Patch(color=self.APEX_COLORS["edge_negative"], label="Negative causal effect")
        ax.legend(
            handles=[positive_patch, negative_patch],
            loc="lower right",
            facecolor=self.APEX_COLORS["background"],
            edgecolor=self.APEX_COLORS["highlight"],
            labelcolor=self.APEX_COLORS["text"],
            fontsize=10,
        )

        ax.axis("off")
        plt.tight_layout()

        if output_path:
            fig.savefig(
                output_path, dpi=300,
                facecolor=self.APEX_COLORS["background"],
                bbox_inches="tight",
            )
            logger.info("Causal DAG saved to %s", output_path)

        return fig

    def to_plotly(self, causal_graph: dict[str, Any]) -> dict:
        """Convert causal graph to Plotly-compatible JSON for the frontend."""
        import networkx as nx

        nodes = causal_graph.get("nodes", [])
        edges = causal_graph.get("edges", [])

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for e in edges:
            G.add_edge(e["source"], e["target"], weight=e.get("weight", 0))

        pos = nx.spring_layout(G, k=2.5, seed=42)

        node_trace = {
            "x": [pos[n][0] for n in nodes],
            "y": [pos[n][1] for n in nodes],
            "text": nodes,
            "mode": "markers+text",
            "marker": {"size": 30, "color": self.APEX_COLORS["node"]},
            "textposition": "top center",
            "textfont": {"color": self.APEX_COLORS["text"], "size": 12},
        }

        edge_traces = []
        for e in edges:
            src, tgt = e["source"], e["target"]
            edge_traces.append({
                "x": [pos[src][0], pos[tgt][0]],
                "y": [pos[src][1], pos[tgt][1]],
                "mode": "lines",
                "line": {
                    "width": 1 + 4 * abs(e.get("weight", 0)),
                    "color": self.APEX_COLORS["edge_positive"] if e.get("weight", 0) > 0
                    else self.APEX_COLORS["edge_negative"],
                },
                "hovertext": f'{src} → {tgt} (w={e.get("weight", 0):.3f})',
            })

        return {"nodes": node_trace, "edges": edge_traces}
