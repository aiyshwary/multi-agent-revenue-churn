# Multi‑Agent Revenue & Churn Analyzer

A small, practical demo that combines deterministic Python processing with agentic reasoning (LLMs) to analyze client revenue and detect churn. The project is intentionally simple to run locally, auditable in its computations, and easy to extend.

Why this project
- Demonstrates a clear separation of responsibilities: **LLMs for planning and critique, Python for numeric work**.
- Shows real-world considerations (token budgets, chunked processing, retries, and observability) so the system behaves predictably at scale.

## Quick start
1. Prepare environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run tests:
   ```bash
   pytest -q
   ```
3. Run the pipeline (example):
   ```bash
   python run_orchestrator.py --data data/sample_input.csv
   ```

Programmatic examples
- Run the pipeline in parallel chunks:
  ```bash
  python -c "from agents.core import run_full_pipeline; print(run_full_pipeline('data/sample_input.csv', chunk_size=1000, parallel_workers=4))"
  ```
- Use the Orchestrator programmatically:
  ```bash
  python -c "from agents import Orchestrator; o=Orchestrator('data/sample_input.csv', max_workers=4); print(o.run())"
  ```

## What it does (brief)
- Normalizes input data (CSV/Excel), assigns quarters, and aggregates revenue.
- Computes per‑client quarterly revenue and churn metrics.
- Validates results deterministically and optionally via an LLM reviewer.
- Persists outputs and lightweight run metrics to `outputs/`.

## Principles
- LLMs THINK; Python COMPUTES — numeric aggregation is always done in code.
- Keep prompts small: use compressed summaries and semantic retrieval when needed.
- Make runs idempotent and observable (logs, metrics, token traces).

## Files to inspect
- `agents/` — orchestrator, tools, LLM adapters, validators, visualizers.
- `tests/` — unit tests for core behaviours and edge cases.
- `outputs/` — generated CSV/JSON and visualizations.

---

## Deployment (Docker + CI) 🔧
This repository already contains a Dockerfile, `docker-compose.yml`, a Makefile, and CI workflows. Use `make up` to run locally in Docker, and `make build` to produce a local image.

If you want a deployable HTTP endpoint or Kubernetes manifests, I can add those next.

---

If you'd like me to prepare a ZIP for submission, a suggested email body, or open a PR with a demo endpoint, tell me which and I’ll take care of it.
