"""
APEX FastAPI Backend Server
============================
Serves the APEX pipeline via REST endpoints and WebSocket for
real-time pipeline progress streaming to the React frontend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apex import __version__
from apex.core.config import APEXConfig
from apex.core.engine import APEXEngine, LayerResult, PipelineState
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

from .schemas import (
    PipelineRunRequest,
    PipelineStatusResponse,
    LayerResultResponse,
    CausalGraphResponse,
    ParetoFrontResponse,
    SHAPResponse,
    DashboardResponse,
    HealthResponse,
)

logger = logging.getLogger("apex.api")


# ══════════════════════════════════════════════════════════════════════════════
# Global state
# ══════════════════════════════════════════════════════════════════════════════

class AppState:
    def __init__(self):
        self.config: APEXConfig = APEXConfig()
        self.engine: APEXEngine | None = None
        self.pipeline_running: bool = False
        self.current_layer: str = ""
        self.results: list[LayerResult] = []
        self.ws_clients: list[WebSocket] = []


app_state = AppState()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ══════════════════════════════════════════════════════════════════════════════
# App lifecycle
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging("INFO")
    logger.info("APEX API server starting (v%s)", __version__)
    yield
    logger.info("APEX API server shutting down")


app = FastAPI(
    title="APEX Engine API",
    description="Autonomous Paradigm-fusing Explanation Engine — REST API",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket broadcast
# ══════════════════════════════════════════════════════════════════════════════

async def broadcast(message: dict):
    """Send a JSON message to all connected WebSocket clients."""
    text = json.dumps(message, cls=NumpyEncoder)
    disconnected = []
    for ws in app_state.ws_clients:
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        app_state.ws_clients.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app_state.ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(app_state.ws_clients))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        app_state.ws_clients.remove(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(app_state.ws_clients))


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version=__version__,
        domain=app_state.config.domain,
        layers_registered=list(app_state.engine._layers.keys()) if app_state.engine else [],
    )


@app.post("/pipeline/run")
async def run_pipeline(request: PipelineRunRequest):
    """Launch the full APEX pipeline (runs in background)."""
    if app_state.pipeline_running:
        raise HTTPException(400, "Pipeline is already running")

    asyncio.create_task(_run_pipeline_async(request))
    return {"status": "started", "domain": request.domain}


async def _run_pipeline_async(request: PipelineRunRequest):
    app_state.pipeline_running = True
    app_state.results = []

    try:
        config = APEXConfig()
        engine = APEXEngine(config)

        # Register layers
        engine.register_layer("causal_discovery", CausalDiscoveryLayer())
        engine.register_layer("evolutionary_search", EvolutionarySearchLayer())
        engine.register_layer("explainability", ExplainabilityLayer())
        engine.register_layer("neuro_symbolic", NeuroSymbolicLayer())
        engine.register_layer("physics_informed", PhysicsInformedLayer())
        engine.register_layer("foundation_agent", FoundationAgentLayer())

        config.evolution.population_size = request.population_size
        config.evolution.n_generations = request.n_generations

        # Load data
        loader = DataLoader()
        data = loader.load_domain_data(request.domain)

        if data.empty:
            await broadcast({"type": "error", "message": "No data available"})
            return

        preprocessor = DataPreprocessor()
        data = preprocessor.fit_transform(data, target_col="binding_affinity")

        engine.load_data(data, target_col="binding_affinity")
        app_state.engine = engine

        await broadcast({"type": "pipeline_started", "domain": request.domain})

        # Run layers sequentially with progress broadcast
        for layer_name in request.layers:
            app_state.current_layer = layer_name
            await broadcast({
                "type": "layer_started",
                "layer": layer_name,
            })

            result = await asyncio.get_event_loop().run_in_executor(
                None, engine.run_layer, layer_name
            )
            app_state.results.append(result)

            await broadcast({
                "type": "layer_completed",
                "layer": layer_name,
                "status": result.status,
                "metrics": result.metrics,
                "elapsed_seconds": result.elapsed_seconds,
            })

        # Generate dashboard data
        dashboard_gen = DashboardGenerator()
        dashboard_data = dashboard_gen.generate(engine.state)

        await broadcast({
            "type": "pipeline_completed",
            "dashboard": dashboard_data,
        })

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        await broadcast({"type": "error", "message": str(exc)})
    finally:
        app_state.pipeline_running = False
        app_state.current_layer = ""


@app.get("/pipeline/status")
async def pipeline_status():
    completed = [r.layer_name for r in app_state.results]
    all_layers = [
        "causal_discovery", "evolutionary_search", "explainability",
        "neuro_symbolic", "physics_informed", "foundation_agent",
    ]
    remaining = [l for l in all_layers if l not in completed]

    return PipelineStatusResponse(
        status="running" if app_state.pipeline_running else "idle",
        iteration=app_state.engine.state.iteration if app_state.engine else 0,
        layers_completed=completed,
        layers_remaining=remaining if app_state.pipeline_running else [],
        results=[
            LayerResultResponse(
                layer_name=r.layer_name,
                status=r.status,
                metrics=r.metrics,
                elapsed_seconds=r.elapsed_seconds,
                timestamp=r.timestamp,
            )
            for r in app_state.results
        ],
    )


@app.get("/results/causal")
async def get_causal_results():
    if not app_state.engine or not app_state.engine.state.causal_graph:
        raise HTTPException(404, "No causal results available")

    graph = app_state.engine.state.causal_graph
    validated = []
    for r in app_state.results:
        if r.layer_name == "causal_discovery" and r.artifacts:
            validated = r.artifacts.get("validated_edges", [])
            break

    return CausalGraphResponse(
        nodes=graph.get("nodes", []),
        edges=graph.get("edges", []),
        n_edges=len(graph.get("edges", [])),
        validated_edges=validated,
    )


@app.get("/results/evolution")
async def get_evolution_results():
    if not app_state.engine or not app_state.engine.state.population:
        raise HTTPException(404, "No evolution results available")

    gen_log = []
    for r in app_state.results:
        if r.layer_name == "evolutionary_search" and r.artifacts:
            gen_log = r.artifacts.get("generation_log", [])
            break

    return ParetoFrontResponse(
        pareto_front=[
            p for p in app_state.engine.state.population if p.get("rank", 99) == 0
        ],
        population_size=len(app_state.engine.state.population),
        pareto_front_size=len([
            p for p in app_state.engine.state.population if p.get("rank", 99) == 0
        ]),
        generation_log=gen_log,
    )


@app.get("/results/explainability")
async def get_xai_results():
    for r in app_state.results:
        if r.layer_name == "explainability" and r.status == "success":
            arts = r.artifacts or {}
            shap_r = arts.get("shap_result", {})
            return SHAPResponse(
                feature_ranking=shap_r.get("feature_ranking", []),
                mean_abs_shap=shap_r.get("mean_abs_shap", {}),
                expected_value=shap_r.get("expected_value", 0),
                consistency=r.metrics.get("shap_consistency", 0),
                counterfactuals=arts.get("counterfactuals", []),
            )
    raise HTTPException(404, "No explainability results available")


@app.get("/results/dashboard")
async def get_dashboard():
    if not app_state.engine:
        raise HTTPException(404, "No pipeline results available")

    gen = DashboardGenerator()
    data = gen.generate(app_state.engine.state)
    return JSONResponse(content=json.loads(json.dumps(data, cls=NumpyEncoder)))


@app.get("/data/provenance")
async def data_provenance():
    validator = DataValidator()
    table = validator.provenance_table()
    return table.to_dict(orient="records")


@app.get("/data/quality")
async def data_quality():
    """Return data quality report if preprocessing has been done."""
    return {"message": "Run pipeline first to generate quality report"}
