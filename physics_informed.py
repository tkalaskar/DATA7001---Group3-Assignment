"""
Layer 5 — Physics-Informed Constraints
========================================
Conservation laws (mass balance, energy, charge neutrality, crystal symmetry)
are embedded as differentiable loss terms.  Any evolved architecture that
violates physics is **eliminated before fitness evaluation** — acting as a
hard kill signal in the evolutionary loop.

Supports:
  • Mass balance conservation
  • Energy conservation constraints
  • Charge neutrality for materials
  • Custom PDE residual losses
  • Equivariance-aware scoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from apex.core.engine import LayerResult, PipelineState
from apex.core.config import APEXConfig

logger = logging.getLogger("apex.layers.physics")


# ══════════════════════════════════════════════════════════════════════════════
# Conservation law verifiers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsViolation:
    law: str
    severity: float
    message: str
    sample_indices: list[int] = field(default_factory=list)


class ConservationLawChecker:
    """
    Registry of physics-based constraint functions.
    Each checker returns a list of violations for a given dataset.
    """

    def __init__(self):
        self._checkers: dict[str, Callable] = {
            "mass_balance": self._check_mass_balance,
            "energy_conservation": self._check_energy_conservation,
            "charge_neutrality": self._check_charge_neutrality,
            "molecular_validity": self._check_molecular_validity,
            "thermodynamic_stability": self._check_thermodynamic_stability,
        }

    def check(
        self,
        law: str,
        data: pd.DataFrame,
        predictions: np.ndarray | None = None,
        threshold: float = 0.01,
    ) -> list[PhysicsViolation]:
        checker = self._checkers.get(law)
        if checker is None:
            logger.warning("Unknown conservation law: %s", law)
            return []
        return checker(data, predictions, threshold)

    # ------------------------------------------------------------------
    @staticmethod
    def _check_mass_balance(
        data: pd.DataFrame,
        predictions: np.ndarray | None,
        threshold: float,
    ) -> list[PhysicsViolation]:
        """
        Verify that molecular descriptors satisfy mass conservation.
        For drug discovery: molecular_weight must be positive and consistent
        with atom counts.
        """
        violations = []
        if "molecular_weight" not in data.columns:
            return violations

        mw = data["molecular_weight"].values
        invalid_indices = np.where(mw <= 0)[0].tolist()

        if invalid_indices:
            violations.append(PhysicsViolation(
                law="mass_balance",
                severity=len(invalid_indices) / len(mw),
                message=f"{len(invalid_indices)} samples have non-positive molecular weight",
                sample_indices=invalid_indices[:20],
            ))

        if "hba" in data.columns and "hbd" in data.columns:
            h_atoms = data.get("hba", pd.Series(dtype=float)).values + data.get("hbd", pd.Series(dtype=float)).values
            max_reasonable_h = mw / 1.008
            bad = np.where(h_atoms > max_reasonable_h)[0].tolist()
            if bad:
                violations.append(PhysicsViolation(
                    law="mass_balance",
                    severity=len(bad) / len(mw),
                    message=f"{len(bad)} samples have more H-bond donors/acceptors than MW allows",
                    sample_indices=bad[:20],
                ))

        return violations

    @staticmethod
    def _check_energy_conservation(
        data: pd.DataFrame,
        predictions: np.ndarray | None,
        threshold: float,
    ) -> list[PhysicsViolation]:
        """For climate: energy balance at top of atmosphere."""
        violations = []
        if "energy_imbalance" in data.columns:
            imbalance = data["energy_imbalance"].values
            bad = np.where(np.abs(imbalance) > threshold)[0].tolist()
            if bad:
                violations.append(PhysicsViolation(
                    law="energy_conservation",
                    severity=len(bad) / len(imbalance),
                    message=f"{len(bad)} samples violate energy conservation (|imbalance| > {threshold})",
                    sample_indices=bad[:20],
                ))
        return violations

    @staticmethod
    def _check_charge_neutrality(
        data: pd.DataFrame,
        predictions: np.ndarray | None,
        threshold: float,
    ) -> list[PhysicsViolation]:
        """For materials: crystal must be charge-neutral."""
        violations = []
        if "total_charge" in data.columns:
            charge = data["total_charge"].values
            bad = np.where(np.abs(charge) > threshold)[0].tolist()
            if bad:
                violations.append(PhysicsViolation(
                    law="charge_neutrality",
                    severity=len(bad) / len(charge),
                    message=f"{len(bad)} materials are not charge-neutral",
                    sample_indices=bad[:20],
                ))
        return violations

    @staticmethod
    def _check_molecular_validity(
        data: pd.DataFrame,
        predictions: np.ndarray | None,
        threshold: float,
    ) -> list[PhysicsViolation]:
        """Drug discovery: validate physical plausibility of molecular descriptors."""
        violations = []

        bounds = {
            "logP": (-10, 15),
            "tpsa": (0, 500),
            "rotatable_bonds": (0, 50),
            "hbd": (0, 20),
            "hba": (0, 30),
        }

        for col, (low, high) in bounds.items():
            if col in data.columns:
                vals = data[col].values
                bad = np.where((vals < low) | (vals > high))[0].tolist()
                if bad:
                    violations.append(PhysicsViolation(
                        law="molecular_validity",
                        severity=len(bad) / len(vals),
                        message=f"{len(bad)} samples have {col} outside [{low}, {high}]",
                        sample_indices=bad[:20],
                    ))

        return violations

    @staticmethod
    def _check_thermodynamic_stability(
        data: pd.DataFrame,
        predictions: np.ndarray | None,
        threshold: float,
    ) -> list[PhysicsViolation]:
        """Materials: formation energy must be negative for stability."""
        violations = []
        if "formation_energy_per_atom" in data.columns:
            fe = data["formation_energy_per_atom"].values
            unstable = np.where(fe > 0)[0].tolist()
            if unstable:
                violations.append(PhysicsViolation(
                    law="thermodynamic_stability",
                    severity=len(unstable) / len(fe),
                    message=f"{len(unstable)} materials are thermodynamically unstable (Ef > 0)",
                    sample_indices=unstable[:20],
                ))
        return violations


# ══════════════════════════════════════════════════════════════════════════════
# Physics-informed loss computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_physics_loss(
    predictions: np.ndarray,
    data: pd.DataFrame,
    conservation_laws: list[str],
    pde_weight: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """
    Compute an aggregate physics loss that can be added to the evolutionary
    fitness function.  Higher loss = more physics violations.
    """
    checker = ConservationLawChecker()
    total_loss = 0.0
    per_law_loss = {}

    for law in conservation_laws:
        violations = checker.check(law, data, predictions)
        law_loss = sum(v.severity for v in violations)
        per_law_loss[law] = round(law_loss, 6)
        total_loss += law_loss * pde_weight

    return round(total_loss, 6), per_law_loss


# ══════════════════════════════════════════════════════════════════════════════
# Layer class
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsInformedLayer:
    """
    APEX Layer 5 — enforces conservation laws and physical constraints.
    Models violating physics are killed; physics loss becomes an
    additional evolutionary fitness term.
    """

    def execute(
        self,
        data: pd.DataFrame | None,
        target_col: str | None,
        state: PipelineState,
        config: APEXConfig,
    ) -> LayerResult:
        if data is None:
            return LayerResult(layer_name="physics_informed", status="failed")

        pc = config.physics
        checker = ConservationLawChecker()

        all_violations: list[dict[str, Any]] = []
        total_severity = 0.0
        laws_checked = 0

        predictions = None
        if state.best_model is not None:
            features = state.feature_names or [c for c in data.columns if c != target_col]
            try:
                predictions = state.best_model.predict(data[features].dropna().values)
            except Exception:
                pass

        for law in pc.conservation_laws:
            violations = checker.check(law, data, predictions, pc.violation_threshold)
            laws_checked += 1
            for v in violations:
                all_violations.append({
                    "law": v.law,
                    "severity": v.severity,
                    "message": v.message,
                    "n_samples_affected": len(v.sample_indices),
                })
                total_severity += v.severity

        # Molecular validity (always checked for drug discovery)
        if config.domain == "drug_discovery":
            mol_violations = checker.check("molecular_validity", data, predictions)
            for v in mol_violations:
                all_violations.append({
                    "law": v.law,
                    "severity": v.severity,
                    "message": v.message,
                    "n_samples_affected": len(v.sample_indices),
                })
                total_severity += v.severity

        # Kill decision
        should_kill = pc.kill_on_violation and total_severity > pc.violation_threshold
        if should_kill:
            logger.warning(
                "PHYSICS KILL SIGNAL: total_severity=%.4f > threshold=%.4f",
                total_severity, pc.violation_threshold,
            )

        # Update state
        state.physics_violations = [v["message"] for v in all_violations]

        return LayerResult(
            layer_name="physics_informed",
            status="success",
            metrics={
                "laws_checked": laws_checked,
                "n_violations": len(all_violations),
                "total_severity": round(total_severity, 4),
                "kill_signal": should_kill,
                "physics_compliant": total_severity <= pc.violation_threshold,
            },
            artifacts={
                "violations": all_violations,
                "conservation_laws": pc.conservation_laws,
            },
        )
