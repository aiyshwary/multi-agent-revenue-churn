# Multi‑Agent Revenue & Churn Analyzer

A compact, practical pipeline that combines deterministic Python analytics with agentic reasoning (LLMs) to analyze revenue and surface churn insights. It’s designed to be easy to run locally, auditable, and straightforward to extend.

Why this repository
- Clean separation of responsibilities: `LLMs` plan and critique while `Python/pandas` performs all numeric work.
- Built to show real‑world constraints: token budgets, chunked processing for large files, retries, and basic observability.

## Quick start 
1. Prepare environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the unit tests:
   ```bash
   pytest -q
   ```
3. Run the pipeline on the sample data:
   ```bash
   python run_orchestrator.py --data data/sample_input.csv
   ```

Quick demo
- After a run, check `outputs/` for:
  - `churn_report.json` — churn details per client
  - `client_quarterly_revenue.csv` — per-client quarterly aggregates
  - `visualization/` — HTML/PNG charts you can open in a browser

How it works (short)
- Planner emits an executable plan (list or graph of nodes).
- Orchestrator executes steps (Executor), runs deterministic validators, and uses Reflection to decide retry/replan actions.
- Memory manager persists short summaries and (optionally) semantic snippets for later retrieval.

Design principles
- LLMs for high‑level reasoning; Python for all math and aggregation.
- Keep LLM prompts small — use summaries and semantic retrieval when needed.
- Make runs idempotent, auditable, and testable.

Files to look at first
- `agents/` — core implementation (orchestrator, tools, LLM client, visualizations).
- `tests/` — unit tests that demonstrate behavior and edge cases.
- `docs/` — architecture and pipeline flow diagrams.

Deployment
- Local container: `make build` / `make up` (uses `docker-compose.yml`).
- CI: `.github/workflows/ci.yml` runs the tests; `publish.yml` builds/publishes the container image.

Maintainer
- aiyshwarya aruchamy