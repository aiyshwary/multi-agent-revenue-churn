# Architecture — how the system works (plain English)

This document explains, in straightforward terms, how the multi‑agent orchestrator is designed and why.

## Quick summary
- The system separates reasoning (LLMs) from computation (Python/pandas).
- A central Orchestrator runs a closed loop: Plan → Execute → Validate → Reflect → (retry or replan).
- All numeric work is done in deterministic Python — LLMs only suggest, critique, or replan.

## Main components
- Planner Agent — produces a structured plan (an ordered list or dependency graph). This can be LLM‑assisted or deterministic.
- Executor / Tools — the implementation of each step (load, assign quarter, aggregate, compute churn). This is the authoritative source for numeric results.
- Validator — runs deterministic checks (sums, schema constraints) and can optionally ask an LLM to review higher‑level logic.
- Reflection — inspects outputs, scores them, and recommends an action (accept, retry, replan, escalate).
- Memory Manager — keeps short‑term run state and stores compressed long‑term summaries (no raw data persisted). It also provides small semantic retrieval when helpful.
- Orchestrator — coordinates execution, enforces retry policies and idempotency, and records observability data.

## How a run proceeds (step by step)
1. Planner creates nodes describing actions, dependencies, validation checkpoints, and retry rules.
2. Orchestrator executes nodes in order (respecting dependencies) and records outputs in short‑term memory.
3. Validator checks results; Reflection scores the outcome and returns guidance.
4. Orchestrator acts on that guidance (retry a node, invoke the Planner to replan, or accept the results).

This is a verification loop—LLMs propose and critique, but deterministic checks and Python code are the source of truth.

## Memory model
- Short‑term: transient state for the current run (step outputs, context).
- Long‑term: compressed summaries stored for later context (designed to avoid storing raw tables).
- Semantic: a tiny, deterministic vector index used only to retrieve small summaries for prompts.

## Reliability & observability
- Per‑node retry policies, exponential backoff, and a circuit breaker protect runs from repeated failures.
- Idempotency markers prevent duplicate side effects on retries.
- Metrics, token logs, and step traces are written to `outputs/` so runs are auditable.

## Scaling & token strategy
- `TokenEstimator` + `LLMClient` enforce budgets.
- Planner prompts favor compressed summaries and semantic retrieval (not raw rows).
- Large files are streamed and processed in chunks; parallel workers preserve deterministic results and are covered by tests.

## Example planner node
```json
{
  "id": "agg_q1",
  "action": "aggregate_quarter_region",
  "depends_on": ["load"],
  "validation_checkpoint": "validate_quarterly",
  "retry_policy": {"max_attempts": 2}
}
```

## Failure → recovery (typical)
If a validator fails, Reflection can request a `retry` or a `replan`. The Orchestrator will follow the suggestion, re‑run affected nodes, and persist metrics and decision logs so reviewers can replay what happened.

## Tests & proof
Core behaviors (graph execution, chunking, retries, memory, semantic retrieval) are unit‑tested. See `tests/` for examples and replay scenarios.

---

