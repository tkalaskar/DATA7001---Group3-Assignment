# APEX — Autonomous Paradigm-fusing Explanation Engine

> **UQ DATA7001 — Introduction to Data Science | Group Project | March 2026**

A world-first unified self-improving pipeline integrating **six AI paradigms** into a single closed loop for scientific discovery.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        APEX Engine                              │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Layer 1  │──▶│ Layer 2  │──▶│ Layer 3  │──▶│ Layer 4  │    │
│  │ Causal   │   │ Evolve   │   │ XAI/MI   │   │ Symbolic │    │
│  │ NOTEARS  │   │ NSGA-III │   │ SHAP+SAE │   │ Rules    │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       │                                             │           │
│       │         ┌──────────┐   ┌──────────┐         │           │
│       │         │ Layer 5  │──▶│ Layer 6  │─────────┘           │
│       │         │ Physics  │   │ Claude   │                     │
│       └────────▶│ Kill Sig │   │ Agent    │─── Loop Back ──▶    │
│                 └──────────┘   └──────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### The Six Paradigms

| Layer | Paradigm | Technology | Role in APEX |
|-------|----------|-----------|--------------|
| L1 | **Causal AI** | NOTEARS, DoWhy | Learns structural causal model; constrains architecture search |
| L2 | **Evolutionary AI** | CMA-ES, NSGA-III | Multi-objective architecture optimisation on Pareto frontier |
| L3 | **Explainable AI** | SHAP, LIME, DiCE, Sparse AE | Feature attribution, counterfactuals, mechanistic interpretability |
| L4 | **Neuro-symbolic** | Symbolic Rule Engine | Domain rules (Lipinski, conservation laws) filter neural outputs |
| L5 | **Physics-informed** | Conservation Laws | Kill signal: eliminates physics-violating models before evaluation |
| L6 | **Foundation Agent** | Claude API | Synthesises hypotheses, proposes new rules and fitness functions |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the frontend dashboard)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (uses synthetic data if no datasets downloaded)

```bash
python run_apex.py run
```

### 3. Start the interactive dashboard

```bash
# Terminal 1 — API server
uvicorn api.app:app --reload

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** to see the APEX dashboard.

### 4. Or use Docker

```bash
docker-compose up --build
```

---

## Project Structure

```
APEX/
├── apex/                          # Core Python package
│   ├── core/
│   │   ├── config.py              # Configuration management
│   │   └── engine.py              # Pipeline orchestrator
│   ├── layers/
│   │   ├── causal_discovery.py    # L1: NOTEARS + DoWhy
│   │   ├── evolutionary_search.py # L2: NSGA-III + CMA-ES
│   │   ├── explainability.py      # L3: SHAP + SAE
│   │   ├── neuro_symbolic.py      # L4: Symbolic rules
│   │   ├── physics_informed.py    # L5: Conservation laws
│   │   └── foundation_agent.py    # L6: Claude API
│   ├── data/
│   │   ├── loaders.py             # Dataset loading
│   │   ├── preprocessors.py       # Cleaning + scaling
│   │   └── validators.py          # Quality checks + provenance
│   ├── evaluation/
│   │   ├── metrics.py             # AUC, HVI, ATE, SHAP consistency
│   │   └── validators.py          # Result quality gates
│   └── visualization/
│       ├── causal_dag.py          # DAG rendering
│       ├── pareto_frontier.py     # Pareto plots
│       ├── shap_plots.py          # SHAP waterfall + bars
│       └── dashboard.py           # Dashboard data generator
├── api/
│   ├── app.py                     # FastAPI server + WebSocket
│   └── schemas.py                 # Pydantic request/response models
├── frontend/                      # React + Tailwind dashboard
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── PipelineControl.jsx
│   │   │   ├── LayerStatus.jsx
│   │   │   ├── CausalGraph.jsx
│   │   │   ├── ParetoFrontier.jsx
│   │   │   ├── ShapExplainer.jsx
│   │   │   ├── HypothesisPanel.jsx
│   │   │   ├── MetricsGrid.jsx
│   │   │   └── ArchitectureDiagram.jsx
│   │   └── hooks/
│   └── package.json
├── notebooks/
│   ├── 01_causal_discovery.ipynb
│   ├── 02_evolutionary_search.ipynb
│   ├── 03_explainability.ipynb
│   ├── 04_neuro_symbolic.ipynb
│   ├── 05_foundation_agent.ipynb
│   └── 06_full_pipeline.ipynb
├── config/
│   ├── apex_config.yaml           # Full pipeline configuration
│   └── domain_rules.yaml          # Symbolic domain rules
├── data/                          # Place datasets here
│   ├── raw/
│   ├── processed/
│   └── README.md                  # Download instructions
├── tests/
├── run_apex.py                    # CLI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Datasets

See **[data/README.md](data/README.md)** for download instructions.

| Dataset | Domain | Records | Access |
|---------|--------|---------|--------|
| DrugBank 5.0 | Drug Discovery | 14,000+ drugs | Open Academic |
| ChEMBL 34 | Drug Discovery | 2.4M molecules | Free Download |
| STRING v12 | Drug Discovery | 67M interactions | Free Download |
| DisGeNET v7 | Drug Discovery | 1.1M associations | Open Academic |
| CMIP6 | Climate | Multi-model ensemble | Free |
| ERA5 | Climate | Hourly 1940-present | Free Academic |
| Materials Project | Materials | 154,000+ materials | Free API |
| SuperCon | Materials | 33,000 superconductors | Free Download |

> If no datasets are downloaded, APEX automatically generates **realistic synthetic drug discovery data** for development.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status and registered layers |
| `POST` | `/pipeline/run` | Launch the full APEX pipeline |
| `GET` | `/pipeline/status` | Current pipeline progress |
| `GET` | `/results/causal` | Causal graph and validated edges |
| `GET` | `/results/evolution` | Pareto front and generation log |
| `GET` | `/results/explainability` | SHAP values and counterfactuals |
| `GET` | `/results/dashboard` | Full dashboard JSON |
| `GET` | `/data/provenance` | Dataset provenance table |
| `WS` | `/ws` | Real-time pipeline progress stream |

---

## Evaluation Metrics

| Metric | Target | Layer |
|--------|--------|-------|
| AUC-ROC / R² | > 0.85 | L2 Evolutionary |
| Pareto Hypervolume (HVI) | Maximise | L2 Evolutionary |
| ATE significance (p < 0.05) | All edges | L1 Causal |
| SHAP consistency (Spearman) | > 0.90 | L3 XAI |
| Mechanistic alignment (cosine) | > 0.70 | L3 MI |
| Conformal prediction coverage | 95% | Uncertainty |

---

## Configuration

All parameters are centralized in `config/apex_config.yaml`. Key settings:

```yaml
evolution:
  population_size: 40
  n_generations: 100
  objectives: [accuracy, interpretability, complexity]

causal:
  algorithm: notears
  lambda1: 0.1
  w_threshold: 0.3

agent:
  model: claude-sonnet-4-20250514
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Team Roles (UQ DATA7001)

| Role | Responsibilities |
|------|-----------------|
| **Data Engineer** | ETL, dataset merging, data quality slide, provenance table |
| **ML / Evolutionary Specialist** | CMA-ES, NSGA-III, surrogate model, Pareto frontier |
| **XAI / Causal Specialist** | SHAP, LIME, DoWhy, NOTEARS, DiCE counterfactuals |
| **Neuro-symbolic + Presenter** | Symbolic rules, Claude agent, slides, storytelling |

---

## Ethical Considerations

- All datasets are at the molecular/material level — **no personal data**
- Bias documented: DrugBank/ChEMBL biased toward Western pharmaceutical targets
- Every prediction includes SHAP explanation, causal DAG, and uncertainty bounds
- LLM used for hypothesis synthesis only — quantitative claims grounded in deterministic pipeline
- Full reproducibility via pinned dependencies and Docker

---

## References

1. DiMasi et al. (2016). Innovation in the pharmaceutical industry. *J Health Economics*, 47, 20-33.
2. Oganov et al. (2019). Structure prediction drives materials discovery. *Nature Reviews Chemistry*, 3(5).
3. Wishart et al. (2018). DrugBank 5.0. *Nucleic Acids Research*, 46(D1).
4. Szklarczyk et al. (2023). STRING database in 2023. *Nucleic Acids Research*, 51(D1).
5. Pinero et al. (2020). DisGeNET knowledge platform. *Nucleic Acids Research*, 48(D1).
6. Zdrazil et al. (2024). ChEMBL Database in 2023. *Nucleic Acids Research*, 52(D1).

---

*Built for UQ DATA7001 — Introduction to Data Science | APEX Team | March 2026*
