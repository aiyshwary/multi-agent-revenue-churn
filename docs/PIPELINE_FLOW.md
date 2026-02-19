# Pipeline execution — flow & post‑run state

This page explains the runtime flow in plain language and points you to the places you’ll want to inspect when debugging or reviewing a run.

```mermaid
flowchart TD
  Start([Start]) --> Planner["PlannerAgent / LLMPlanner\n(plan)"]
  Planner --> PlanList["Plan (ordered steps / graph nodes)"]
  PlanList --> GraphRunner["GraphRunner.run\n(enforce deps, schedule ready nodes)"]
  GraphRunner --> OrchestratorStep["Orchestrator._run_single_step\n(retries, idempotency, circuit-breaker, metrics)"]
  OrchestratorStep --> Executor["ExecutorAgent.execute\n(step implementation)"]

  subgraph Steps
    L1[load_data] --> L2[assign_quarter]
    L2 --> L3[aggregate_quarter_region]
    L3 --> L4[client_quarterly]
    L4 --> L5[compute_churn]
    L5 --> L6[validate]
    L6 --> L7[reflect]
  end

  Executor --> Steps

  L3 -->|writes| FileAgg["outputs/quarterly_by_region.csv"]
  L4 -->|writes| FileClient["outputs/client_quarterly_revenue.csv"]
  L5 -->|writes| FileChurn["outputs/churn_report.json"]
  OrchestratorStep -->|records| Metrics["outputs/metrics.json / metrics snapshot"]
  Executor -->|stores| Memory["MemoryManager (memory.json & semantic_memory)"]

  L7 --> ReflectionAgent["ReflectionAgent.critique → suggestions"]
  ReflectionAgent --> Result["Orchestrator.run returns:\ncontext, reflections, suggestions, log, metrics"]
  Result --> Visualize["visualize.py → outputs/visualization/*.png/.html"]
  Visualize --> End([End])

  classDef disk fill:#f9f,stroke:#333,stroke-width:1px;
  class FileAgg,FileClient,FileChurn,Metrics disk;
```

---

## Quick debugging spots
- `agents/orchestrator.py` → `Orchestrator.run()` — entry point for a full pipeline run.
- `agents/orchestrator.py` → `GraphRunner.run()` — where dependencies are scheduled.
- `agents/orchestrator.py` → `_run_single_step()` — retry, idempotency, and circuit‑breaker logic.
- `agents/orchestrator.py` → `ExecutorAgent.execute()` — where step implementations run.
- `agents/visualize.py` → plotting helpers (verify artifact outputs).

## Useful runtime artifacts to inspect
- `context` object: evolving run state (typical keys: `df`, `agg`, `client_q`, `churn_reports`, `reflections`).
- `self.log`: per‑step status, attempt counts, and error traces.
- `self.metrics.snapshot()`: counters and timing metrics.
- Disk outputs: `outputs/churn_report.json`, `outputs/*.csv`, `outputs/metrics.json`, and `outputs/visualization/*`.
- Memory files: `outputs/memory.json` and `outputs/semantic_memory.json` (if semantic memory is enabled).

## Common questions this answers
- Who schedules steps? → `GraphRunner`.
- Where do retries and circuit‑breaker live? → `_run_single_step` / `CircuitBreaker`.
- Where is data persisted? → step handlers in `ExecutorAgent` and `visualize.py` for artifacts.

---

## Next steps
1. Add `logger.debug` statements at a few key points and run the graph‑runner test so you can step through execution. ✅
2. Export the Mermaid diagram to an SVG/PNG and add it to `docs/` for a visual reference. 🖼️

Reply with `1`, `2`, or `both` and I’ll implement them.