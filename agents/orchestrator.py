import time
import json
import argparse
import ast
import hashlib
import logging
from typing import List, Dict, Any, Optional
import concurrent.futures, threading

import pandas as pd

from .tools import (
    DataLoader,
    assign_quarter,
    Aggregator,
    ChurnCalculator,
    Validator,
    MemoryManager,
)
from .llm import LLMClient, SemanticMemory
from .core import CircuitBreaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")


class PlannerAgent:
    def plan(self, objective: str) -> List[Dict[str, Any]]:
        return [
            {"step": "load_data"},
            {"step": "assign_quarter"},
            {"step": "aggregate_quarter_region"},
            {"step": "client_quarterly"},
            {"step": "compute_churn"},
            {"step": "validate"},
            {"step": "reflect"},
        ]


class LLMPlanner:
    """Wrapper that asks an LLM for a plan. Falls back to the core PlannerAgent.

    Accepts a deterministic `fallback` planner (from `agents.core`) so the
    orchestrator can rely on a stable plan when the LLM is unavailable.
    """

    def __init__(self, llm: LLMClient, fallback: Any):
        self.llm = llm
        self.fallback = fallback

    def plan(self, objective: str) -> List[Dict[str, Any]]:
        try:
            prompt = f"Decompose this objective into ordered pipeline steps (return a Python list of dicts with 'id'/'action'/'tool'): {objective}"
            res = self.llm.call(prompt, max_tokens=256)
            text = res.get("text", "")
            try:
                plan = ast.literal_eval(text)
            except Exception:
                try:
                    plan = json.loads(text)
                except Exception:
                    plan = None
            if isinstance(plan, list) and all(isinstance(p, dict) and ("action" in p or "step" in p) for p in plan):
                normalized = []
                for p in plan:
                    name = p.get('id') or p.get('step') or p.get('action')
                    action = p.get('action') or p.get('step')
                    tool = p.get('tool') or p.get('tool')
                    normalized.append({"id": name, "action": action, "tool": tool})
                return normalized
        except Exception as e:
            logger.debug("LLM planner failed: %s", e)
        return self.fallback.plan(objective)


class ExecutorAgent:
    def __init__(self, memory: MemoryManager):
        self.memory = memory

    def execute(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        step_name = step["step"]
        if step_name == "load_data":
            data_path = context.get("data_path")
            if data_path and data_path.endswith('.csv'):
                df = pd.read_csv(data_path, parse_dates=['Date'])
            else:
                df = DataLoader.load_excel(context["data_path"])
            self.memory.store_step_summary("load_data", {"rows": len(df), "revenue_sum": float(df["Revenue"].sum())})
            return {"df": df}

        if step_name == "assign_quarter":
            df = context["df"]
            df_q = assign_quarter(df)
            self.memory.store_step_summary("assign_quarter", {"rows": len(df_q)})
            return {"df": df_q}

        if step_name == "aggregate_quarter_region":
            df_q = context["df"]
            agg = Aggregator.aggregate_by_quarter_region(df_q)
            self.memory.store_step_summary("aggregate_quarter_region", {"rows": len(agg), "total_revenue": float(agg["TotalRevenue"].sum())})
            agg.to_csv("outputs/quarterly_by_region.csv", index=False)
            return {"agg": agg}

        if step_name == "client_quarterly":
            df_q = context["df"]
            client_q = Aggregator.client_quarterly_revenue(df_q)
            self.memory.store_step_summary("client_quarterly", {"rows": len(client_q)})
            client_q.to_csv("outputs/client_quarterly_revenue.csv", index=False)
            return {"client_q": client_q}

        if step_name == "compute_churn":
            client_q = context["client_q"]
            quarters = [c for c in client_q.columns if c != "Client ID"]
            quarters = sorted(quarters)
            reports = []
            for i in range(len(quarters) - 1):
                r = ChurnCalculator.churn_between_quarters(client_q, quarters[i], quarters[i + 1])
                reports.append(r)
            with open("outputs/churn_report.json", "w") as f:
                json.dump(reports, f, indent=2)
            self.memory.store_step_summary("compute_churn", {"num_pairs": len(reports)})
            return {"churn_reports": reports}

        if step_name == "validate":
            df = context.get("df")
            agg = context.get("agg")
            ok, msg = Validator.totals_match(df, agg)
            self.memory.store_step_summary("validate", {"ok": ok, "message": msg})
            return {"valid": ok, "message": msg}

        if step_name == "reflect":
            churn_reports = context.get("churn_reports", [])
            reflections = []
            for r in churn_reports:
                net = r["revenue_gain"] - r["revenue_loss"]
                note = "stable"
                if r["revenue_loss"] > r["revenue_gain"] * 2 and r["revenue_loss"] > 0:
                    note = "significant_loss"
                reflections.append({"pair": f"{r['q_prev']}->{r['q_curr']}", "net_change": net, "note": note})
            self.memory.store_step_summary("reflect", {"reflections": reflections})
            return {"reflections": reflections}

        raise ValueError(f"Unknown step: {step_name}")


class ValidatorAgent:
    def validate(self, validation_output: Dict[str, Any]) -> bool:
        return validation_output.get("valid", False)


class LLMValidator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def validate(self, context_summary: str) -> str:
        prompt = f"Review the following validation output and say ACCEPT or RETRY and why:\n\n{context_summary}"
        res = self.llm.call(prompt)
        return res.get("text", "")


class ReflectionAgent:
    def critique(self, reflections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        suggestions = []
        for r in reflections:
            if r["note"] == "significant_loss":
                suggestions.append({"pair": r["pair"], "suggestion": "Investigate top lost clients and run retention offers."})
        return suggestions


class MetricsCollector:
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.timings: Dict[str, List[float]] = {}

    def increment(self, name: str, amount: int = 1):
        self.counters[name] = self.counters.get(name, 0) + amount

    def record_timing(self, name: str, value: float):
        self.timings.setdefault(name, []).append(value)

    def snapshot(self) -> Dict[str, Any]:
        return {"counters": self.counters, "timings": self.timings}


class GraphRunner:
    """Simple state-graph runner that executes nodes respecting dependencies.

    Node schema (accepted):
      {"id": "node_id", "action": "executor_step_name", "depends_on": ["node_id"], "params": {...}}

    The runner is intentionally lightweight and synchronous by default. It
    delegates actual execution to the provided `orchestrator._run_single_step` so
    retries, circuit-breaker and idempotency are consistently applied.
    """

    def __init__(self, orchestrator: "Orchestrator", max_workers: int = 1):
        self.orch = orchestrator
        self.max_workers = max_workers

    def run(self, plan: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        # preserve plan order so nodes without explicit `depends_on` run sequentially
        ordered_nids = [n["id"] for n in plan]
        nodes = {n["id"]: n.copy() for n in plan}
        deps = {nid: set(nodes[nid].get("depends_on", [])) for nid in nodes}
        executed = set()
        results = {}
        lock = threading.Lock()

        while len(executed) < len(nodes):
            # build `ready` honoring plan order: nodes without explicit deps run in sequence;
            # nodes with explicit deps may run as soon as their deps are satisfied (parallelizable).
            ready = []
            for idx, nid in enumerate(ordered_nids):
                if nid in executed:
                    continue
                node_deps = deps.get(nid, set())
                if node_deps:
                    if node_deps.issubset(executed):
                        ready.append(nid)
                    continue
                # no explicit deps -> enforce sequential order relative to earlier nodes
                prior = ordered_nids[:idx]
                if all(p in executed for p in prior):
                    ready.append(nid)
                else:
                    # preserve sequence: do not consider later sequential nodes ready
                    break

            if not ready:
                raise RuntimeError("Cyclic or unsatisfiable graph plan")

            # Execute ready nodes in parallel (bounded by max_workers). Merge outputs under a lock.
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                future_to_nid = {}
                for nid in ready:
                    node = nodes[nid]
                    step_name = node.get("action") or node.get("step")
                    step = {"step": step_name}
                    params = node.get("params") or {}
                    local_ctx = dict(context)
                    local_ctx.update(params)
                    future = ex.submit(self.orch._run_single_step, step, local_ctx)
                    future_to_nid[future] = (nid, local_ctx)

                for fut in concurrent.futures.as_completed(future_to_nid):
                    nid, _local_ctx = future_to_nid[fut]
                    out = fut.result()
                    with lock:
                        context.update(out or {})
                        results[nid] = {"status": "ok", "output": out}
                        executed.add(nid)

        return context


class Orchestrator:
    def __init__(
        self,
        data_path: str,
        use_llm: bool = False,
        llm_client: Optional[LLMClient] = None,
        planner_llm: Optional[LLMClient] = None,
        validator_llm: Optional[LLMClient] = None,
        reflection_llm: Optional[LLMClient] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        max_retries: int = 2,
        cb_fail_threshold: int = 3,
        backoff_base: float = 0.01,
        max_workers: int = 1,
    ):
        self.data_path = data_path
        self.memory = MemoryManager()
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent(self.memory)
        self.validator = ValidatorAgent()
        self.reflection = ReflectionAgent()
        self.log: List[Dict[str, Any]] = []
        self.max_retries = max_retries

        self.use_llm = use_llm
        self.llm_client = llm_client
        planner_client = planner_llm or llm_client
        validator_client = validator_llm or llm_client
        reflection_client = reflection_llm or llm_client

        self.llm_planner = LLMPlanner(planner_client, self.planner) if (use_llm and planner_client) else None
        self.llm_validator = LLMValidator(validator_client) if (use_llm and validator_client) else None
        self.llm_reflection = reflection_client if (use_llm and reflection_client) else None
        self.semantic_memory = semantic_memory or SemanticMemory()
        self.circuit_breaker = CircuitBreaker(fail_threshold=cb_fail_threshold, reset_timeout=30)
        self.circuit = self.circuit_breaker
        self.metrics = MetricsCollector()
        self._idempotency: Dict[str, bool] = {}
        self.backoff_base = backoff_base
        self.max_workers = max_workers

    def _idempotency_key(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        payload = f"{step.get('step')}|{hashlib.sha256(repr(context.get(step.get('step'))).encode()).hexdigest()}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _run_single_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step with retries, idempotency, metrics and circuit-breaker.
        Returns the executor output (dict) on success.
        """
        step_name = step.get("step") or step.get("action")
        key = self._idempotency_key({"step": step_name}, context)
        if self._idempotency.get(key):
            logger.info("Skipping idempotent step: %s", step_name)
            self.metrics.increment("skipped_steps")
            return {}

        if self.circuit_breaker.is_open(step_name):
            raise RuntimeError(f"Circuit breaker open for step {step_name}; aborting run")

        attempt = 0
        while True:
            attempt += 1
            start = time.time()
            try:
                out = self.executor.execute({"step": step_name}, context)
                duration = time.time() - start
                self.metrics.record_timing(f"step.{step_name}.duration", duration)
                self.metrics.increment(f"step.{step_name}.success")

                if step_name == "validate":
                    ok = self.validator.validate(out)
                    if not ok:
                        self.metrics.increment("validation.fail")
                        if self.use_llm and self.llm_validator:
                            advice = self.llm_validator.validate(str(out))
                            logger.info("LLM validator advice: %s", advice)
                        if attempt <= self.max_retries:
                            backoff = min(self.backoff_base * (2 ** (attempt - 1)), 2.0)
                            time.sleep(backoff)
                            continue

                self._idempotency[key] = True
                self.log.append({"step": {"step": step_name}, "status": "ok", "attempts": attempt})
                self.circuit_breaker.record_success(step_name)
                return out
            except Exception as e:
                duration = time.time() - start
                self.metrics.record_timing(f"step.{step_name}.failure_time", duration)
                self.metrics.increment(f"step.{step_name}.error")
                logger.exception("Step failed: %s", step)
                self.log.append({"step": {"step": step_name}, "status": "error", "error": str(e), "attempts": attempt})
                self.circuit_breaker.record_failure(step_name)
                if attempt >= self.max_retries:
                    raise
                backoff = min(self.backoff_base * (2 ** (attempt - 1)), 2.0)
                time.sleep(backoff)
                continue

    def run(self) -> Dict[str, Any]:
        plan = self.planner.plan("Revenue + Churn analysis")
        if self.use_llm and self.llm_planner:
            plan = self.llm_planner.plan("Revenue + Churn analysis")

        context: Dict[str, Any] = {"data_path": self.data_path}

        if isinstance(plan, list) and plan and ("id" in plan[0] or "action" in plan[0]):
            runner = GraphRunner(self, max_workers=self.max_workers)
            context = runner.run(plan, context)
        else:
            for step in plan:
                out = self._run_single_step(step, context)
                context.update(out)

        reflections = context.get("reflections", [])
        suggestions = self.reflection.critique(reflections)

        try:
            with open("outputs/metrics.json", "w") as f:
                json.dump(self.metrics.snapshot(), f, default=str)
        except Exception:
            logger.debug("Failed to persist metrics")

        result = {
            "context": context,
            "reflections": reflections,
            "suggestions": suggestions,
            "log": self.log,
            "metrics": self.metrics.snapshot(),
        }

        try:
            from .visualize import plot_churn_trends, plot_quarterly_totals, plot_top_lost_clients

            churn_reports = context.get('churn_reports') or context.get('churn_report') or []
            if churn_reports:
                plot_churn_trends(churn_reports, out_png='outputs/visualization/churn_trends.png', out_html='outputs/visualization/churn_trends.html')

            if 'agg' in context and context['agg'] is not None:
                dfagg = context['agg']
                try:
                    qdict = {q: g.to_dict(orient='records') for q, g in dfagg.groupby('Quarter')}
                    plot_quarterly_totals(qdict, out_png='outputs/visualization/quarterly_totals.png', out_html='outputs/visualization/quarterly_totals.html')
                except Exception:
                    pass
            elif result.get('quarterly'):
                plot_quarterly_totals(result['quarterly'], out_png='outputs/visualization/quarterly_totals.png', out_html='outputs/visualization/quarterly_totals.html')

            client_q = context.get('client_q')
            if client_q is not None:
                try:
                    plot_top_lost_clients(client_q, list(client_q.columns)[1], list(client_q.columns)[2] if client_q.shape[1] > 2 else list(client_q.columns)[1], out_png='outputs/visualization/top_lost_clients.png')
                except Exception:
                    try:
                        plot_top_lost_clients(client_q, '2024Q1', '2024Q2', out_png='outputs/visualization/top_lost_clients.png')
                    except Exception:
                        pass
        except Exception:
            logger.debug('Visualization generation skipped or failed')

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ar 7.xlsx", help="path to Excel file")
    args = parser.parse_args()
    o = Orchestrator(data_path=args.data)
    res = o.run()
    print(json.dumps(res, indent=2, default=str))