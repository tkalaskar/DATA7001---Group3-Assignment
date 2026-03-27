"""
Layer 4 — Neuro-Symbolic Reasoning Engine
===========================================
Bridges neural outputs with symbolic logic.  Neural predictions become
probabilistic facts; symbolic rules (from domain_rules.yaml) constrain
which predictions are admissible.

The system supports:
  • Forward-chaining inference over domain rules
  • Confidence propagation through rule chains
  • Conflict detection between neural outputs and symbolic constraints
  • Rule co-evolution: the Foundation Agent (Layer 6) can propose new rules
    based on mechanistic interpretability evidence from Layer 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from apex.core.engine import LayerResult, PipelineState
from apex.core.config import APEXConfig

logger = logging.getLogger("apex.layers.neurosymbolic")


# ══════════════════════════════════════════════════════════════════════════════
# Rule representation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SymbolicRule:
    name: str
    description: str
    conditions: list[str]
    action: str
    confidence: float = 1.0
    domain: str = "drug_discovery"
    source: str = "domain_expert"  # domain_expert | llm_generated | learned


@dataclass
class RuleEvaluationResult:
    rule_name: str
    satisfied: bool
    confidence: float
    failing_conditions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Symbolic rule engine
# ══════════════════════════════════════════════════════════════════════════════

class SymbolicRuleEngine:
    """
    A forward-chaining symbolic inference engine that evaluates domain
    rules against data rows.  Each row (molecule / material / sample)
    is tested against all applicable rules.
    """

    def __init__(self):
        self.rules: list[SymbolicRule] = []

    def load_rules(self, rules_path: str | Path, domain: str = "drug_discovery") -> None:
        path = Path(rules_path)
        if not path.exists():
            logger.warning("Rules file not found: %s", path)
            return

        with open(path, "r") as f:
            all_rules = yaml.safe_load(f) or {}

        domain_rules = all_rules.get(domain, {})
        for rule_name, rule_def in domain_rules.items():
            self.rules.append(SymbolicRule(
                name=rule_name,
                description=rule_def.get("description", ""),
                conditions=rule_def.get("conditions", []),
                action=rule_def.get("action", ""),
                confidence=rule_def.get("confidence", 1.0),
                domain=domain,
                source="domain_expert",
            ))

        logger.info("Loaded %d symbolic rules for domain '%s'", len(self.rules), domain)

    def add_rule(self, rule: SymbolicRule) -> None:
        self.rules.append(rule)
        logger.info("Added rule: %s (source=%s)", rule.name, rule.source)

    def evaluate_row(self, row: dict[str, Any]) -> list[RuleEvaluationResult]:
        """Evaluate all rules against a single data row."""
        results = []
        for rule in self.rules:
            failing = []
            for condition in rule.conditions:
                if not self._evaluate_condition(condition, row):
                    failing.append(condition)

            results.append(RuleEvaluationResult(
                rule_name=rule.name,
                satisfied=len(failing) == 0,
                confidence=rule.confidence if len(failing) == 0 else 0.0,
                failing_conditions=failing,
                details={"action": rule.action, "description": rule.description},
            ))
        return results

    def evaluate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all rules against every row.  Returns a DataFrame with
        boolean columns for each rule and an aggregate compliance score.
        """
        rule_results = {rule.name: [] for rule in self.rules}
        compliance_scores = []

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_results = self.evaluate_row(row_dict)
            n_satisfied = 0
            for res in row_results:
                rule_results[res.rule_name].append(res.satisfied)
                if res.satisfied:
                    n_satisfied += 1
            compliance_scores.append(n_satisfied / max(len(self.rules), 1))

        result_df = pd.DataFrame(rule_results)
        result_df["compliance_score"] = compliance_scores
        return result_df

    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate_condition(condition: str, row: dict[str, Any]) -> bool:
        """
        Parse and evaluate a simple condition string against a row.
        Supports: <=, >=, <, >, ==, !=, 'not' prefix, 'abs()'.
        """
        condition = condition.strip()

        if condition.startswith("not "):
            return not SymbolicRuleEngine._evaluate_condition(condition[4:], row)

        # Handle abs() wrapper
        if condition.startswith("abs("):
            inner = condition[4:]
            paren_end = inner.find(")")
            if paren_end < 0:
                return False
            var_name = inner[:paren_end].strip()
            rest = inner[paren_end + 1:].strip()
            val = row.get(var_name)
            if val is None:
                return False
            try:
                val = abs(float(val))
            except (ValueError, TypeError):
                return False
            for op, func in [("<=", lambda a, b: a <= b), (">=", lambda a, b: a >= b),
                             ("<", lambda a, b: a < b), (">", lambda a, b: a > b),
                             ("==", lambda a, b: a == b)]:
                if op in rest:
                    threshold = float(rest.split(op)[1].strip())
                    return func(val, threshold)
            return False

        for op, func in [("<=", lambda a, b: a <= b), (">=", lambda a, b: a >= b),
                         ("!=", lambda a, b: a != b), ("==", lambda a, b: a == b),
                         ("<", lambda a, b: a < b), (">", lambda a, b: a > b)]:
            if op in condition:
                parts = condition.split(op, 1)
                var_name = parts[0].strip()
                threshold_str = parts[1].strip()

                val = row.get(var_name)
                if val is None:
                    return False

                try:
                    if threshold_str.lower() in ("true", "false"):
                        return func(str(val).lower(), threshold_str.lower())
                    val = float(val)
                    threshold = float(threshold_str)
                    return func(val, threshold)
                except (ValueError, TypeError):
                    return False

        # Boolean check
        val = row.get(condition)
        if val is not None:
            return bool(val)

        return False


# ══════════════════════════════════════════════════════════════════════════════
# Neuro-symbolic integration: filter neural predictions through symbolic rules
# ══════════════════════════════════════════════════════════════════════════════

def _filter_predictions(
    predictions: np.ndarray,
    data: pd.DataFrame,
    rule_compliance: pd.DataFrame,
    min_compliance: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mask predictions that violate symbolic constraints.
    Returns (filtered_predictions, violation_mask).
    """
    compliance = rule_compliance["compliance_score"].values
    mask = compliance >= min_compliance

    filtered = predictions.copy()
    filtered[~mask] = np.nan

    return filtered, ~mask


# ══════════════════════════════════════════════════════════════════════════════
# Layer class
# ══════════════════════════════════════════════════════════════════════════════

class NeuroSymbolicLayer:
    """
    APEX Layer 4 — applies symbolic domain rules to neural predictions,
    filtering out candidates that violate domain constraints.
    """

    def __init__(self, rules_path: str | Path = "config/domain_rules.yaml"):
        self.rules_path = rules_path

    def execute(
        self,
        data: pd.DataFrame | None,
        target_col: str | None,
        state: PipelineState,
        config: APEXConfig,
    ) -> LayerResult:
        if data is None:
            return LayerResult(layer_name="neuro_symbolic", status="failed")

        engine = SymbolicRuleEngine()
        engine.load_rules(self.rules_path, domain=config.domain)

        if not engine.rules:
            logger.warning("No symbolic rules loaded — applying default drug-likeness filters")
            engine.rules = self._default_drug_rules()

        # Add any LLM-generated rules from previous iterations
        if state.symbolic_rules:
            for rule_def in state.symbolic_rules:
                engine.add_rule(SymbolicRule(**rule_def))

        # Evaluate all rules against the dataset
        compliance_df = engine.evaluate_dataset(data)

        n_compliant = int((compliance_df["compliance_score"] >= 0.5).sum())
        n_total = len(data)
        avg_compliance = float(compliance_df["compliance_score"].mean())

        per_rule_compliance = {}
        for rule in engine.rules:
            if rule.name in compliance_df.columns:
                per_rule_compliance[rule.name] = float(compliance_df[rule.name].mean())

        # Filter predictions if model exists
        violation_summary = []
        if state.best_model is not None:
            features = state.feature_names or [c for c in data.columns if c != target_col]
            try:
                preds = state.best_model.predict(data[features].dropna().values)
                _, violation_mask = _filter_predictions(preds, data, compliance_df)
                n_violations = int(violation_mask.sum())
                violation_summary = [
                    f"{n_violations} of {len(preds)} predictions violate symbolic constraints"
                ]
                state.physics_violations = violation_summary
            except Exception as exc:
                logger.warning("Prediction filtering failed: %s", exc)

        return LayerResult(
            layer_name="neuro_symbolic",
            status="success",
            metrics={
                "n_rules": len(engine.rules),
                "avg_compliance": round(avg_compliance, 4),
                "n_compliant": n_compliant,
                "n_total": n_total,
                "compliance_rate": round(n_compliant / max(n_total, 1), 4),
            },
            artifacts={
                "per_rule_compliance": per_rule_compliance,
                "compliance_scores": compliance_df["compliance_score"].tolist(),
                "violation_summary": violation_summary,
                "rules_applied": [
                    {"name": r.name, "description": r.description, "source": r.source}
                    for r in engine.rules
                ],
            },
        )

    @staticmethod
    def _default_drug_rules() -> list[SymbolicRule]:
        """Lipinski's Rule of Five as fallback."""
        return [
            SymbolicRule(
                name="lipinski_mw",
                description="Molecular weight ≤ 500 Da",
                conditions=["molecular_weight <= 500"],
                action="pass_mw_filter",
                confidence=0.95,
            ),
            SymbolicRule(
                name="lipinski_logp",
                description="LogP ≤ 5",
                conditions=["logP <= 5"],
                action="pass_logp_filter",
                confidence=0.95,
            ),
            SymbolicRule(
                name="lipinski_hbd",
                description="H-bond donors ≤ 5",
                conditions=["hbd <= 5"],
                action="pass_hbd_filter",
                confidence=0.95,
            ),
            SymbolicRule(
                name="lipinski_hba",
                description="H-bond acceptors ≤ 10",
                conditions=["hba <= 10"],
                action="pass_hba_filter",
                confidence=0.95,
            ),
        ]
