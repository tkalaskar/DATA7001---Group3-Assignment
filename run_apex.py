"""
APEX Quick Start — Run the full pipeline from the command line.

Usage:
    python run_apex.py                    # Run with defaults (synthetic data)
    python run_apex.py --domain drug_discovery --generations 50
    python run_apex.py --serve            # Start the API server
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

from apex.core.config import APEXConfig
from apex.core.engine import APEXEngine
from apex.data.loaders import DataLoader
from apex.data.preprocessors import DataPreprocessor
from apex.data.validators import DataValidator
from apex.layers import (
    CausalDiscoveryLayer,
    EvolutionarySearchLayer,
    ExplainabilityLayer,
    NeuroSymbolicLayer,
    PhysicsInformedLayer,
    FoundationAgentLayer,
)
from apex.visualization.dashboard import DashboardGenerator
from apex.utils.logging_config import setup_logging


def run_pipeline(args):
    setup_logging(args.log_level)

    print("\n" + "=" * 70)
    print("  APEX — Autonomous Paradigm-fusing Explanation Engine")
    print("  UQ DATA7001 | March 2026")
    print("=" * 70 + "\n")

    config = APEXConfig("config/apex_config.yaml")

    if args.generations:
        config.evolution.n_generations = args.generations
    if args.population:
        config.evolution.population_size = args.population

    # Load data
    loader = DataLoader()
    data = loader.load_domain_data(args.domain)

    if data.empty:
        print("ERROR: No data available. See data/README.md for download instructions.")
        sys.exit(1)

    # Validate
    validator = DataValidator()
    checks = validator.validate(
        data,
        expected_columns=["molecular_weight", "logP", "hbd", "hba", "tpsa", "rotatable_bonds", "binding_affinity"],
    )
    print("\n--- Data Validation ---")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.check_name}: {c.message}")

    # Preprocess
    preprocessor = DataPreprocessor()
    data = preprocessor.fit_transform(data, target_col="binding_affinity")
    print("\n--- Data Quality Report ---")
    print(preprocessor.quality_report_dataframe().to_string(index=False))

    # Build engine
    engine = APEXEngine(config)
    engine.load_data(data, target_col="binding_affinity")

    engine.register_layer("causal_discovery", CausalDiscoveryLayer())
    engine.register_layer("evolutionary_search", EvolutionarySearchLayer())
    engine.register_layer("explainability", ExplainabilityLayer())
    engine.register_layer("neuro_symbolic", NeuroSymbolicLayer())
    engine.register_layer("physics_informed", PhysicsInformedLayer())
    engine.register_layer("foundation_agent", FoundationAgentLayer())

    # Run
    results = engine.run(n_iterations=args.iterations)

    # Summary
    print("\n" + "=" * 70)
    print("  APEX Pipeline Complete")
    print("=" * 70)
    summary = engine.summary()
    print(summary.to_string(index=False))

    # Export
    output_dir = engine.export_results("outputs")
    dashboard = DashboardGenerator()
    dashboard.export_json(engine.state, output_dir / "dashboard.json")

    print(f"\nResults exported to: {output_dir}")
    print("Dashboard JSON: outputs/dashboard.json")
    print("\nTo view the interactive dashboard:")
    print("  1. Start the API:      uvicorn api.app:app --reload")
    print("  2. Start the frontend: cd frontend && npm run dev")
    print("  3. Open:               http://localhost:5173\n")


def serve(args):
    import uvicorn
    setup_logging(args.log_level)
    print("\nStarting APEX API server...")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)


def main():
    parser = argparse.ArgumentParser(
        description="APEX — Autonomous Paradigm-fusing Explanation Engine"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the APEX pipeline")
    run_parser.add_argument("--domain", default="drug_discovery", choices=["drug_discovery", "climate", "materials"])
    run_parser.add_argument("--iterations", type=int, default=1)
    run_parser.add_argument("--generations", type=int, default=30)
    run_parser.add_argument("--population", type=int, default=20)
    run_parser.add_argument("--log-level", default="INFO")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    if args.command == "serve":
        serve(args)
    elif args.command == "run":
        run_pipeline(args)
    else:
        # Default: run pipeline
        args.domain = "drug_discovery"
        args.iterations = 1
        args.generations = 30
        args.population = 20
        args.log_level = "INFO"
        run_pipeline(args)


if __name__ == "__main__":
    main()
