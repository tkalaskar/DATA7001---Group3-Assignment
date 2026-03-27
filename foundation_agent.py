"""
Layer 6 — Foundation Model Agent (Google Gemini API)
=====================================================
Reads all upstream layer outputs and:
  1. Synthesises natural-language scientific hypotheses
  2. Proposes new symbolic rules based on mechanistic evidence
  3. Proposes new fitness functions for evolution
  4. Generates research directions and experimental suggestions

This closes the APEX self-improving loop: the agent's proposals are
fed back to Layers 2 and 4 in the next iteration.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from apex.core.engine import LayerResult, PipelineState
from apex.core.config import APEXConfig

logger = logging.getLogger("apex.layers.agent")


# ══════════════════════════════════════════════════════════════════════════════
# Prompt engineering
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are APEX Scientific Advisor -- an elite AI research scientist embedded
in the APEX (Autonomous Paradigm-fusing Explanation Engine) pipeline.

Your role is to synthesise outputs from five upstream AI layers:
1. Causal Discovery -- structural causal graphs learned from data
2. Evolutionary Search -- Pareto-optimal model architectures
3. Explainability -- SHAP values, counterfactuals, mechanistic features
4. Neuro-Symbolic -- domain constraint compliance results
5. Physics-Informed -- conservation law violation analysis

Based on these inputs, you must:
A) Generate 3-5 testable scientific hypotheses grounded in the evidence
B) Propose 1-3 new symbolic rules that should be added to the domain knowledge base
C) Suggest improvements to the evolutionary fitness function
D) Identify the most promising research directions

Your output must be rigorous, quantitative, and actionable.
Every claim must reference the specific upstream evidence that supports it."""


def _build_analysis_prompt(state: PipelineState, config: APEXConfig) -> str:
    """Build a comprehensive prompt from all upstream layer results."""
    sections = []

    sections.append(f"## Domain: {config.domain}")
    sections.append(f"## Iteration: {state.iteration}")

    if state.causal_graph:
        edges = state.causal_graph.get("edges", [])
        top_edges = sorted(edges, key=lambda e: abs(e.get("weight", 0)), reverse=True)[:10]
        sections.append("## Layer 1 -- Causal Graph")
        sections.append(f"Nodes: {state.causal_graph.get('nodes', [])}")
        sections.append("Top causal edges (by weight):")
        for e in top_edges:
            sections.append(f"  {e['source']} -> {e['target']} (w={e['weight']})")

    if state.population:
        pareto = [p for p in state.population if p.get("rank", 99) == 0]
        sections.append(f"\n## Layer 2 -- Evolutionary Search")
        sections.append(f"Population size: {len(state.population)}")
        sections.append(f"Pareto front size: {len(pareto)}")
        if pareto:
            best = max(pareto, key=lambda p: p.get("fitness", {}).get("accuracy", 0))
            sections.append(f"Best solution fitness: {best.get('fitness', {})}")
            sections.append(f"Best genome: {best.get('genome', {})}")

    for result in state.history:
        if result.layer_name == "explainability" and result.status == "success":
            sections.append("\n## Layer 3 -- Explainability")
            sections.append(f"SHAP consistency: {result.metrics.get('shap_consistency', 'N/A')}")
            sections.append(f"Top feature: {result.metrics.get('top_feature', 'N/A')}")
            arts = result.artifacts or {}
            if "shap_result" in arts:
                shap_r = arts["shap_result"]
                sections.append(f"Feature importance ranking: {shap_r.get('feature_ranking', [])[:10]}")
                sections.append(f"Mean |SHAP|: {dict(list(shap_r.get('mean_abs_shap', {}).items())[:10])}")
            if "counterfactuals" in arts:
                cfs = arts["counterfactuals"]
                sections.append(f"Counterfactuals generated: {len(cfs)}")
                if cfs:
                    sections.append(f"Example counterfactual: {json.dumps(cfs[0], indent=2)}")
            break

    for result in state.history:
        if result.layer_name == "neuro_symbolic" and result.status == "success":
            sections.append("\n## Layer 4 -- Neuro-Symbolic")
            sections.append(f"Rules applied: {result.metrics.get('n_rules', 0)}")
            sections.append(f"Compliance rate: {result.metrics.get('compliance_rate', 'N/A')}")
            arts = result.artifacts or {}
            sections.append(f"Per-rule compliance: {arts.get('per_rule_compliance', {})}")
            break

    if state.physics_violations:
        sections.append("\n## Layer 5 -- Physics Constraints")
        for v in state.physics_violations:
            sections.append(f"  WARNING: {v}")

    return "\n".join(sections)


# ══════════════════════════════════════════════════════════════════════════════
# Gemini API interaction
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini_api(
    system: str,
    user_message: str,
    model: str = "gemini-2.0-flash",
    max_tokens: int = 4096,
    temperature: float = 0.3,
    api_key: str | None = None,
) -> str | None:
    """Call the Google Gemini API and return the text response."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        logger.warning("GEMINI_API_KEY not set - using offline fallback")
        return None

    try:
        from google import genai

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=f"{system}\n\n---\n\n{user_message}",
            config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return response.text
    except Exception as exc:
        logger.error("Gemini API call failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Offline fallback hypothesis generator
# ══════════════════════════════════════════════════════════════════════════════

def _offline_hypothesis_generator(state: PipelineState, config: APEXConfig) -> dict[str, Any]:
    """
    Generate structured hypotheses without API access, using
    heuristics derived from upstream layer outputs.
    """
    hypotheses = []
    new_rules = []
    fitness_suggestions = []

    if state.causal_graph:
        edges = state.causal_graph.get("edges", [])
        top_edges = sorted(edges, key=lambda e: abs(e.get("weight", 0)), reverse=True)[:3]
        for e in top_edges:
            hypotheses.append(
                f"HYPOTHESIS: {e['source']} has a causal effect on {e['target']} "
                f"(estimated weight = {e['weight']}). This should be validated "
                f"experimentally by intervening on {e['source']} and measuring "
                f"the change in {e['target']}."
            )

    for result in state.history:
        if result.layer_name == "explainability" and result.status == "success":
            top_feat = result.metrics.get("top_feature", "unknown")
            consistency = result.metrics.get("shap_consistency", 0)
            hypotheses.append(
                f"HYPOTHESIS: {top_feat} is the dominant predictive feature "
                f"(SHAP consistency = {consistency}). This feature should "
                f"be prioritised in experimental design for the next research cycle."
            )
            if consistency < 0.8:
                fitness_suggestions.append(
                    "Add a SHAP stability penalty to the fitness function: "
                    "penalise models with SHAP consistency below 0.9."
                )
            break

    for result in state.history:
        if result.layer_name == "neuro_symbolic" and result.status == "success":
            compliance = result.metrics.get("compliance_rate", 1.0)
            if compliance < 0.8:
                new_rules.append({
                    "name": "adaptive_compliance_filter",
                    "description": f"Generated rule: tighten compliance threshold (current rate: {compliance})",
                    "conditions": ["compliance_score >= 0.7"],
                    "action": "enforce_stricter_filtering",
                    "confidence": 0.75,
                    "source": "llm_generated",
                })
            break

    if not hypotheses:
        hypotheses.append(
            "HYPOTHESIS: Insufficient upstream data to generate specific hypotheses. "
            "Recommendation: ensure at least Layers 1-3 complete successfully before "
            "invoking the Foundation Agent."
        )

    return {
        "hypotheses": hypotheses,
        "new_rules": new_rules,
        "fitness_suggestions": fitness_suggestions,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Parse LLM response
# ══════════════════════════════════════════════════════════════════════════════

def _parse_agent_response(response: str) -> dict[str, Any]:
    """Extract structured hypotheses and suggestions from the LLM response."""
    result: dict[str, Any] = {
        "hypotheses": [],
        "new_rules": [],
        "fitness_suggestions": [],
        "raw_response": response,
    }

    current_section = None
    buffer: list[str] = []

    for line in response.split("\n"):
        line_stripped = line.strip()
        lower = line_stripped.lower()

        if "hypothesis" in lower or "hypothes" in lower:
            if current_section and buffer:
                _flush_buffer(result, current_section, buffer)
            current_section = "hypotheses"
            buffer = []
        elif "rule" in lower and ("new" in lower or "propos" in lower or "suggest" in lower):
            if current_section and buffer:
                _flush_buffer(result, current_section, buffer)
            current_section = "new_rules"
            buffer = []
        elif "fitness" in lower and ("suggest" in lower or "propos" in lower or "improv" in lower):
            if current_section and buffer:
                _flush_buffer(result, current_section, buffer)
            current_section = "fitness_suggestions"
            buffer = []

        if line_stripped:
            buffer.append(line_stripped)

    if current_section and buffer:
        _flush_buffer(result, current_section, buffer)

    if not result["hypotheses"]:
        result["hypotheses"] = [response[:500]]

    return result


def _flush_buffer(result: dict, section: str, buffer: list[str]) -> None:
    text = "\n".join(buffer)
    if section == "hypotheses":
        result["hypotheses"].append(text)
    elif section == "new_rules":
        result["new_rules"].append({
            "name": f"llm_rule_{len(result['new_rules'])+1}",
            "description": text[:200],
            "conditions": [],
            "action": "llm_suggested_action",
            "confidence": 0.6,
            "source": "llm_generated",
        })
    elif section == "fitness_suggestions":
        result["fitness_suggestions"].append(text)


# ══════════════════════════════════════════════════════════════════════════════
# Layer class
# ══════════════════════════════════════════════════════════════════════════════

class FoundationAgentLayer:
    """
    APEX Layer 6 — Foundation model agent that synthesises upstream
    outputs into scientific hypotheses and proposes self-improvement
    actions (new rules, new fitness functions).
    """

    def execute(
        self,
        data: pd.DataFrame | None,
        target_col: str | None,
        state: PipelineState,
        config: APEXConfig,
    ) -> LayerResult:
        ac = config.agent
        analysis_prompt = _build_analysis_prompt(state, config)

        logger.info("Invoking foundation agent (model=%s)...", ac.model)

        response = _call_gemini_api(
            system=SYSTEM_PROMPT,
            user_message=analysis_prompt,
            model=ac.model,
            max_tokens=ac.max_tokens,
            temperature=ac.temperature,
            api_key=config.gemini_api_key,
        )

        if response:
            parsed = _parse_agent_response(response)
            mode = "online"
        else:
            parsed = _offline_hypothesis_generator(state, config)
            mode = "offline"

        state.hypotheses = parsed.get("hypotheses", [])
        state.symbolic_rules = parsed.get("new_rules", [])
        state.fitness_functions = parsed.get("fitness_suggestions", [])

        return LayerResult(
            layer_name="foundation_agent",
            status="success",
            metrics={
                "mode": mode,
                "n_hypotheses": len(state.hypotheses),
                "n_new_rules": len(state.symbolic_rules),
                "n_fitness_suggestions": len(state.fitness_functions),
            },
            artifacts={
                "hypotheses": state.hypotheses,
                "new_rules": state.symbolic_rules,
                "fitness_suggestions": state.fitness_functions,
                "raw_response": parsed.get("raw_response", ""),
            },
        )
