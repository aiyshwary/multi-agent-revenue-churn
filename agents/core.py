"""
Core agents for the Multi-Agent Orchestrator (Strategy 1)
Includes: PlannerAgent, DataExecutorAgent, ChurnAnalystAgent, ValidatorAgent, MemoryManager, ReflectionAgent
Deterministic, pandas-based implementations for revenue aggregation and churn detection.
"""
import pandas as pd
import numpy as np
import json
import zlib
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional
from .tools import DataLoader
from .llm import LLMClient, SemanticMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(' ', '').replace('-', '') for c in df.columns]
    if 'ClientID' not in df.columns and 'ClientId' in df.columns:
        df = df.rename(columns={'ClientId': 'ClientID'})
    return df


def _month_to_quarter_label(month: int) -> str:
    return f"Q{((month - 1) // 3) + 1}"


class TokenEstimator:
    @staticmethod
    def estimate(text: str) -> int:
        return max(1, len(text) // 4)


def retry_with_backoff(fn, max_retries: int = 3, backoff_factor: float = 0.1, sleep_fn=time.sleep):
    """Retry `fn()` up to `max_retries` times applying exponential backoff.

    - `fn` is a zero-argument callable.
    - raises the last exception if all retries fail.
    """
    attempts = 0
    last_exc = None
    while attempts < max_retries:
        attempts += 1
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempts >= max_retries:
                raise
            sleep_fn(backoff_factor * (2 ** (attempts - 1)))
    if last_exc:
        raise last_exc
    return None


class CircuitBreaker:
    """Simple in-process circuit breaker keyed by step id."""

    def __init__(self, fail_threshold: int = 3, reset_timeout: int = 60):
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.fail_counts: Dict[str, int] = {}
        self.opened_until: Dict[str, float] = {}

    def record_failure(self, key: str):
        from time import time
        c = self.fail_counts.get(key, 0) + 1
        self.fail_counts[key] = c
        if c >= self.fail_threshold:
            self.opened_until[key] = time() + self.reset_timeout

    def record_success(self, key: str):
        self.fail_counts.pop(key, None)
        self.opened_until.pop(key, None)

    def is_open(self, key: str) -> bool:
        from time import time
        t = self.opened_until.get(key)
        if not t:
            return False
        if time() > t:
            self.fail_counts.pop(key, None)
            self.opened_until.pop(key, None)
            return False
        return True


class MetricsCollector:
    """Very small in-memory metrics collector for counters and timings."""

    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, float] = {}

    def incr(self, name: str, amt: int = 1):
        self.counters[name] = self.counters.get(name, 0) + amt

    def gauge(self, name: str, value: float):
        self.timers[name] = float(value)


class LLMPlanner:
    """Optional, budget-aware planner stub (deterministic fallback when budget is exceeded)."""

    def __init__(self, token_budget: int = 200):
        self.token_budget = token_budget

    def plan(self, objective: str = None) -> List[Dict[str, str]]:
        tokens = TokenEstimator.estimate(objective or "")
        if tokens > self.token_budget:
            return PlannerAgent().plan()
        plan = PlannerAgent().plan()
        for s in plan:
            s["via"] = "llm"
        return plan


class LLMValidator:
    """Optional, budget-aware validator stub that respects token budgets and returns structured result."""

    def __init__(self, token_budget: int = 100):
        self.token_budget = token_budget

    def validate(self, validations: Dict[str, Any]) -> Dict[str, Any]:
        txt = json.dumps(validations)
        tokens = TokenEstimator.estimate(txt)
        if tokens > self.token_budget:
            return {"accept": False, "reason": "budget_exceeded"}
        accept = all(v.get("ok", True) for v in validations.values())
        return {"accept": accept, "reason": "llm_rule_sim"}


class MemoryManager:
    """Short-term + compressed long-term memory with idempotency, failure counters and token logging."""

    def __init__(self, compress: bool = True):
        self.short_term: List[str] = []
        self.long_term: Dict[str, bytes] = {}
        self.compress = compress
        self.token_log: List[Dict[str, Any]] = []

        self.idempotency_keys: set = set()
        self.failure_counts: Dict[str, int] = {}

    def add_short(self, note: str):
        self.short_term.append(note)
        self.token_log.append({"type": "short_add", "tokens": TokenEstimator.estimate(note)})

    def summarize_short(self, max_chars: int = 1500) -> str:
        summary = " | ".join(self.short_term[-50:])
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(' ', 1)[0]
        if self.compress:
            self.long_term['summary'] = zlib.compress(summary.encode('utf-8'))
        else:
            self.long_term['summary'] = summary.encode('utf-8')
        self.token_log.append({"type": "summarize", "tokens": TokenEstimator.estimate(summary)})
        return summary

    def retrieve_summary(self) -> str:
        raw = self.long_term.get('summary', b'')
        if not raw:
            return ''
        if self.compress:
            return zlib.decompress(raw).decode('utf-8')
        return raw.decode('utf-8')

    def save_long_term(self, path: str = "outputs/memory.json"):
        """Persist long-term memory (compressions are base64-encoded)."""
        import base64
        serializable = {}
        for k, v in self.long_term.items():
            if isinstance(v, (bytes, bytearray)):
                serializable[k] = {"_type": "bytes", "data": base64.b64encode(v).decode('ascii')}
            else:
                serializable[k] = v
        try:
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(serializable, f, indent=2)
        except Exception:
            logger.debug("failed to persist core MemoryManager long_term")

    def has_executed(self, key: str) -> bool:
        return key in self.idempotency_keys

    def mark_executed(self, key: str):
        self.idempotency_keys.add(key)

    def record_failure(self, step: str):
        self.failure_counts[step] = self.failure_counts.get(step, 0) + 1

    def reset_failures(self, step: str):
        if step in self.failure_counts:
            del self.failure_counts[step]

    def failure_count(self, step: str) -> int:
        return self.failure_counts.get(step, 0)


class MockLLM:
    """Deterministic local LLM stub for planning/validation in tests and offline runs."""
    def call(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        if "objective" in prompt.lower() or "decompose" in prompt.lower():
            plan = [
                {"id": "load", "action": "load_data", "tool": "DataExecutorAgent"},
                {"id": "quarterly", "action": "group_by_quarter", "tool": "DataExecutorAgent"},
                {"id": "client_quarterly", "action": "client_quarterly_revenue", "tool": "ChurnAnalystAgent"},
                {"id": "churn", "action": "compute_churn", "tool": "ChurnAnalystAgent"},
                {"id": "validate", "action": "validate", "tool": "ValidatorAgent"},
                {"id": "reflect", "action": "reflect", "tool": "ReflectionAgent"},
            ]
            return {"text": json.dumps(plan), "tokens": TokenEstimator.estimate(json.dumps(plan))}
        if "validate" in prompt.lower():
            return {"text": json.dumps({"decision": "accept"}), "tokens": TokenEstimator.estimate("accept")}
        return {"text": "ok", "tokens": TokenEstimator.estimate("ok")}


class PlannerAgent:
    """Creates deterministic or LLM-driven plans and can trim context to fit token budgets."""

    def plan(self, tasks: List[str] = None) -> List[Dict[str, str]]:
        steps = [
            {"id": "load", "action": "load_data", "tool": "DataExecutorAgent"},
            {"id": "quarterly", "action": "group_by_quarter", "tool": "DataExecutorAgent"},
            {"id": "client_quarterly", "action": "client_quarterly_revenue", "tool": "ChurnAnalystAgent"},
            {"id": "churn", "action": "compute_churn", "tool": "ChurnAnalystAgent"},
            {"id": "validate", "action": "validate", "tool": "ValidatorAgent"},
            {"id": "reflect", "action": "reflect", "tool": "ReflectionAgent"},
        ]
        return steps

    def plan_dynamic(self, objective: str, llm: Any = None, memory: Any = None, token_budget: int = 2000) -> List[Dict[str, str]]:
        """Use an LLM (or MockLLM) to decompose an objective into steps while respecting token budgets.
        Falls back to deterministic plan on failure.
        """
        if llm is None:
            return self.plan()

        context = ""
        if memory:
            context = memory.retrieve_summary()
        prompt = f"Objective: {objective}\nContext: {context}\nReturn a JSON array of plan steps."
        if TokenEstimator.estimate(prompt) > token_budget:
            context = memory.summarize_short(max_chars=max(200, token_budget * 4 // 3))
            prompt = f"Objective: {objective}\nContext: {context}\nReturn a JSON array of plan steps."
        resp = llm.call(prompt, max_tokens=min(512, token_budget))
        try:
            parsed = json.loads(resp["text"])
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return self.plan()


class DataExecutorAgent:
    """Deterministic data loading and aggregation using pandas. Supports chunked processing.
    Handles column normalization and basic data cleaning.
    """

    def load_data(self, path: str = None, df: pd.DataFrame = None, chunk_size: int = None) -> pd.DataFrame:
        if df is not None:
            _df = df.copy()
        else:
            if path is None:
                raise ValueError("path or df must be provided")
            if path.endswith('.csv'):
                _df = pd.read_csv(path, parse_dates=['Date'])
            else:
                from .tools import DataLoader as _DL
                _df = _DL.load_excel(path)
        _df = _normalize_columns(_df)
        if 'ClientID' not in _df.columns:
            possible = [c for c in _df.columns if c.lower().startswith('client')]
            if possible:
                _df = _df.rename(columns={possible[0]: 'ClientID'})
        _df['Date'] = pd.to_datetime(_df['Date'])
        _df['Revenue'] = pd.to_numeric(_df['Revenue'], errors='coerce').fillna(0.0)
        return _df

    def group_by_quarter(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        df = df.copy()
        df['Quarter'] = df['Date'].dt.month.apply(_month_to_quarter_label)
        quarters = {}
        for q_label, group in df.groupby('Quarter'):
            agg = (
                group.groupby(['Region', 'Country'])
                .agg(TotalRevenue=('Revenue', 'sum'), NumClients=('ClientID', 'nunique'))
                .reset_index()
            )
            quarters[q_label] = agg
        return quarters


class ChurnAnalystAgent:
    """Compute client-quarter revenue and churn between consecutive quarters."""

    def client_quarterly_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'ClientID' not in df.columns and 'Client ID' in df.columns:
            df = df.rename(columns={'Client ID': 'ClientID'})
        if 'Quarter' not in df.columns:
            if 'Date' in df.columns:
                df['Quarter'] = df['Date'].dt.month.apply(_month_to_quarter_label)
            else:
                raise ValueError("DataFrame must contain either 'Quarter' or 'Date' for client_quarterly_revenue")
        pivot = (
            df.groupby(['ClientID', 'Quarter'])['Revenue']
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        return pivot

    def compute_churn(self, pivot_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        q_cols = [c for c in pivot_df.columns if c.startswith('Q')]
        q_cols = sorted(q_cols)
        results: Dict[str, Dict[str, Any]] = {}
        for i in range(len(q_cols) - 1):
            prev = q_cols[i]
            curr = q_cols[i + 1]
            prev_vals = pivot_df[["ClientID", prev]].set_index('ClientID')[prev]
            curr_vals = pivot_df[["ClientID", curr]].set_index('ClientID')[curr]
            both = pd.concat([prev_vals, curr_vals], axis=1).fillna(0)
            prev_ser = both[prev]
            curr_ser = both[curr]
            lost_mask = (prev_ser > 0) & (curr_ser == 0)
            new_mask = (prev_ser == 0) & (curr_ser > 0)
            lost_customers = int(lost_mask.sum())
            new_customers = int(new_mask.sum())
            revenue_loss = float(prev_ser[lost_mask].sum())
            revenue_gain = float(curr_ser[new_mask].sum())
            results[f"{prev}_to_{curr}"] = {
                "prev_quarter": prev,
                "curr_quarter": curr,
                "lost_customers": lost_customers,
                "new_customers": new_customers,
                "revenue_loss": revenue_loss,
                "revenue_gain": revenue_gain,
                "total_prev_revenue": float(prev_ser.sum()),
                "total_curr_revenue": float(curr_ser.sum()),
            }
        return results


class ValidatorAgent:
    """Performs deterministic checks and triggers simple auto-corrections / retries."""

    def validate_quarterly_aggregation(self, original_df: pd.DataFrame, quarterly_groups: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        total_original = float(original_df['Revenue'].sum())
        total_agg = float(sum(g['TotalRevenue'].sum() for g in quarterly_groups.values()))
        diff = total_original - total_agg
        ok = abs(diff) < 1e-6
        return {"ok": ok, "total_original": total_original, "total_aggregated": total_agg, "difference": diff}

    def validate_churn_report(self, churn_report: Dict[str, Dict[str, Any]], pivot_df: pd.DataFrame) -> Dict[str, Any]:
        checks = {}
        for period, vals in churn_report.items():
            prev = vals['prev_quarter']
            curr = vals['curr_quarter']
            prev_ser = pivot_df.set_index('ClientID').get(prev, pd.Series(dtype=float)).fillna(0)
            curr_ser = pivot_df.set_index('ClientID').get(curr, pd.Series(dtype=float)).fillna(0)
            total_clients = int(((prev_ser > 0) | (curr_ser > 0)).sum())
            computed = vals['lost_customers'] + vals['new_customers'] + int(((prev_ser > 0) & (curr_ser > 0)).sum())
            checks[period] = {"total_clients": total_clients, "reconciles": total_clients == computed}
        all_ok = all(v['reconciles'] for v in checks.values())
        return {"ok": all_ok, "details": checks}


class ReflectionAgent:
    """Simple evaluator that suggests corrections when validators fail."""

    def reflect(self, validations: Dict[str, Any]) -> Dict[str, Any]:
        actions = []
        if not validations.get('quarterly', {}).get('ok', True):
            actions.append({'action': 'retry_aggregation', 'reason': 'sum mismatch'})
        if not validations.get('churn', {}).get('ok', True):
            actions.append({'action': 'recompute_churn', 'reason': 'reconciliation failed'})
        if not actions:
            actions.append({'action': 'accept', 'reason': 'all checks passed'})
        return {"actions": actions}



def run_full_pipeline(
    path_or_df,
    max_retries: int = 2,
    use_llm: bool = False,
    chunk_size: int = None,
    parallel_workers: int = 1,
    token_budget: int = 2000,
    backoff_base: float = 0.01,
    cb_fail_threshold: int = 3,
    sleep_fn=time.sleep,
    llm_client: Optional[LLMClient] = None,
    semantic_memory: Optional[SemanticMemory] = None,
) -> Dict[str, Any]:
    """Run the full deterministic pipeline with optional LLM and semantic memory.

    - Backwards compatible: default behavior unchanged.
    - If `use_llm` is True and an `llm_client` is provided, the planner/validator
      use the LLM subject to `token_budget`.
    - `semantic_memory` is optional; when provided, pipeline stores/retrieves
      compact summaries to/from it for context engineering.
    """
    planner_agent = PlannerAgent()
    executor = DataExecutorAgent()
    churn = ChurnAnalystAgent()
    validator = ValidatorAgent()
    llm_validator = LLMValidator(token_budget=200) if use_llm else None
    memory = MemoryManager()
    reflector = ReflectionAgent()
    metrics = MetricsCollector()
    cb = CircuitBreaker(fail_threshold=cb_fail_threshold, reset_timeout=30)

    llm_for_plan = llm_client if llm_client is not None else (MockLLM() if use_llm else None)
    if use_llm and llm_for_plan:
        plan = planner_agent.plan_dynamic("Revenue + churn analysis", llm=llm_for_plan, memory=memory, token_budget=token_budget)
    else:
        plan = planner_agent.plan()

    if semantic_memory is not None:
        semantic_memory.add("plan_summary", json.dumps([p.get('action', p.get('step')) for p in plan]), {"type": "plan"})

    memory.add_short(f"Plan: { [s.get('action', s.get('step')) for s in plan] }")

    idempotency: Dict[str, bool] = {}

    def _run_with_retry(key: str, fn, *a, **kw):
        if cb.is_open(key):
            raise RuntimeError(f"circuit open for {key}")
        attempts = 0
        while True:
            attempts += 1
            try:
                res = fn(*a, **kw)
                cb.record_success(key)
                metrics.incr(f"step.{key}.success")
                memory.long_term[f"done:{key}"] = True
                memory.save_long_term()
                return res
            except Exception as e:
                metrics.incr(f"step.{key}.failure")
                cb.record_failure(key)
                memory.record_failure(key)
                if cb.is_open(key) or attempts >= max_retries:
                    metrics.incr(f"step.{key}.failed_final")
                    raise
                sleep_fn(backoff_base * (2 ** (attempts - 1)))
                continue

    if not chunk_size:
        df = _run_with_retry("load", lambda: executor.load_data(path=path_or_df) if isinstance(path_or_df, str) else executor.load_data(df=path_or_df))
        memory.add_short(f"Loaded {len(df)} rows")

        quarters = _run_with_retry("quarterly", lambda: executor.group_by_quarter(df))

        pivot = _run_with_retry("client_quarterly", lambda: churn.client_quarterly_revenue(df))
        churn_report = _run_with_retry("compute_churn", lambda: churn.compute_churn(pivot))

    else:
        total_input_revenue = 0.0
        client_q_map = {}
        quarterly_accumulator = {}

        def _process_chunk(df_chunk: pd.DataFrame):
            nonlocal total_input_revenue
            df_norm = executor.load_data(df=df_chunk)
            df_norm['Quarter'] = df_norm['Date'].dt.month.apply(_month_to_quarter_label)
            total_input_revenue += float(df_norm['Revenue'].sum())
            for _i, row in df_norm.iterrows():
                client_q_map[(row['ClientID'], row['Quarter'])] = client_q_map.get((row['ClientID'], row['Quarter']), 0.0) + float(row['Revenue'])
                key = (row['Quarter'], row['Region'], row['Country'])
                if key not in quarterly_accumulator:
                    quarterly_accumulator[key] = {"TotalRevenue": 0.0, "clients": set()}
                quarterly_accumulator[key]["TotalRevenue"] += float(row['Revenue'])
                quarterly_accumulator[key]["clients"].add(row['ClientID'])

        # If parallel_workers > 1, process chunks concurrently and merge intermediate results.
        if parallel_workers and parallel_workers > 1:
            import concurrent.futures
            import threading

            def _process_chunk_local(df_chunk: pd.DataFrame):
                df_norm = executor.load_data(df=df_chunk)
                df_norm['Quarter'] = df_norm['Date'].dt.month.apply(_month_to_quarter_label)
                local_total = float(df_norm['Revenue'].sum())
                local_client_q = {}
                local_quarterly = {}
                for _i, row in df_norm.iterrows():
                    local_client_q[(row['ClientID'], row['Quarter'])] = local_client_q.get((row['ClientID'], row['Quarter']), 0.0) + float(row['Revenue'])
                    key = (row['Quarter'], row['Region'], row['Country'])
                    if key not in local_quarterly:
                        local_quarterly[key] = {"TotalRevenue": 0.0, "clients": set()}
                    local_quarterly[key]["TotalRevenue"] += float(row['Revenue'])
                    local_quarterly[key]["clients"].add(row['ClientID'])
                return local_total, local_client_q, local_quarterly

            merge_lock = threading.Lock()

            def _merge_results(res_tuple):
                local_total, local_client_q, local_quarterly = res_tuple
                nonlocal total_input_revenue, client_q_map, quarterly_accumulator
                with merge_lock:
                    total_input_revenue += local_total
                    for k, v in local_client_q.items():
                        client_q_map[k] = client_q_map.get(k, 0.0) + v
                    for key, vals in local_quarterly.items():
                        if key not in quarterly_accumulator:
                            quarterly_accumulator[key] = {"TotalRevenue": 0.0, "clients": set()}
                        quarterly_accumulator[key]["TotalRevenue"] += vals["TotalRevenue"]
                        quarterly_accumulator[key]["clients"].update(vals["clients"])

            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as ex:
                futures = []
                if isinstance(path_or_df, str) and path_or_df.endswith('.csv'):
                    for chunk in pd.read_csv(path_or_df, parse_dates=['Date'], chunksize=chunk_size):
                        futures.append(ex.submit(_process_chunk_local, chunk))
                else:
                    df_full = executor.load_data(path=path_or_df) if isinstance(path_or_df, str) else executor.load_data(df=path_or_df)
                    for chunk in DataLoader.chunk_dataframe(df_full, chunk_size=chunk_size):
                        futures.append(ex.submit(_process_chunk_local, chunk))

                for f in concurrent.futures.as_completed(futures):
                    _merge_results(f.result())

        else:
            if isinstance(path_or_df, str) and path_or_df.endswith('.csv'):
                for chunk in pd.read_csv(path_or_df, parse_dates=['Date'], chunksize=chunk_size):
                    _process_chunk(chunk)
            else:
                df_full = executor.load_data(path=path_or_df) if isinstance(path_or_df, str) else executor.load_data(df=path_or_df)
                for chunk in DataLoader.chunk_dataframe(df_full, chunk_size=chunk_size):
                    _process_chunk(chunk)

        quarters = {}
        for (q, region, country), vals in quarterly_accumulator.items():
            quarters.setdefault(q, []).append({
                "Region": region,
                "Country": country,
                "TotalRevenue": vals["TotalRevenue"],
                "NumClients": len(vals["clients"]),
            })
        quarters = {q: pd.DataFrame(rows) for q, rows in quarters.items()}

        client_rows = [{"ClientID": k[0], "Quarter": k[1], "Revenue": v} for k, v in client_q_map.items()]
        client_df = pd.DataFrame(client_rows) if client_rows else pd.DataFrame(columns=["ClientID", "Quarter", "Revenue"])
        pivot = churn.client_quarterly_revenue(client_df)
        churn_report = churn.compute_churn(pivot)
        df = pd.DataFrame({"Revenue": [total_input_revenue]})
        memory.add_short(f"Streamed processing; chunk_size={chunk_size}")

    df = _run_with_retry("load", lambda: executor.load_data(path=path_or_df) if isinstance(path_or_df, str) else executor.load_data(df=path_or_df))
    memory.add_short(f"Loaded {len(df)} rows")

    quarters = _run_with_retry("quarterly", lambda: executor.group_by_quarter(df))
    memory.add_short(f"Aggregated into {len(quarters)} quarters: {list(quarters.keys())}")

    pivot = _run_with_retry("client_quarterly", lambda: churn.client_quarterly_revenue(df))
    churn_report = _run_with_retry("compute_churn", lambda: churn.compute_churn(pivot))

    val_quarterly = _run_with_retry("validate_quarterly", lambda: validator.validate_quarterly_aggregation(df, quarters))
    val_churn = _run_with_retry("validate_churn", lambda: validator.validate_churn_report(churn_report, pivot))

    validations = {"quarterly": val_quarterly, "churn": val_churn}
    memory.add_short(f"Validations: {validations}")

    if use_llm:
        llm_for_val = llm_client if llm_client is not None else MockLLM()
        lv = llm_for_val.call(f"Validate: {json.dumps(validations)}")
        memory.add_short(f"LLM-validate: {lv}")
        metrics.incr("llm_validator.calls")

    reflection = reflector.reflect(validations)
    attempts = 0
    while any(a["action"].startswith("retry") for a in reflection["actions"]) and attempts < max_retries:
        attempts += 1
        metrics.incr("self_heal.attempt")
        logger.info(f"Self-healing attempt {attempts}")
        df = _run_with_retry("load", lambda: executor.load_data(df=df))
        quarters = _run_with_retry("quarterly", lambda: executor.group_by_quarter(df))
        pivot = _run_with_retry("client_quarterly", lambda: churn.client_quarterly_revenue(df))
        churn_report = _run_with_retry("compute_churn", lambda: churn.compute_churn(pivot))
        val_quarterly = _run_with_retry("validate_quarterly", lambda: validator.validate_quarterly_aggregation(df, quarters))
        val_churn = _run_with_retry("validate_churn", lambda: validator.validate_churn_report(churn_report, pivot))
        validations = {"quarterly": val_quarterly, "churn": val_churn}
        reflection = reflector.reflect(validations)
        memory.add_short(f"Self-heal attempt {attempts} validations: {validations}")

    if semantic_memory is not None:
        try:
            semantic_memory.persist('outputs/semantic_memory.json')
        except Exception:
            logger.debug("failed to persist semantic memory")

    summary = {
        "plan": plan,
        "quarterly": {k: v.to_dict(orient='records') for k, v in quarters.items()},
        "pivot": pivot.to_dict(orient='records'),
        "churn_report": churn_report,
        "validations": validations,
        "reflection": reflection,
        "memory_summary": memory.summarize_short(),
        "token_log": memory.token_log,
        "metrics": {"counters": metrics.counters, "timers": metrics.timers},
        "circuit_status": {"open": {k: cb.is_open(k) for k in cb.opened_until.keys()}},
    }

    try:
        from .visualize import plot_churn_trends, plot_quarterly_totals, plot_top_lost_clients

        cr = summary.get('churn_report')
        if isinstance(cr, dict):
            cr_list = []
            for k, v in cr.items():
                q_prev = v.get('prev_quarter') or v.get('q_prev')
                q_curr = v.get('curr_quarter') or v.get('q_curr')
                cr_list.append({
                    'q_prev': q_prev,
                    'q_curr': q_curr,
                    'revenue_loss': v.get('revenue_loss') or v.get('revenueLoss') or 0.0,
                    'revenue_gain': v.get('revenue_gain') or v.get('revenueGain') or 0.0,
                })
            cr = cr_list
        if cr:
            plot_churn_trends(cr, out_png='outputs/visualization/churn_trends.png', out_html='outputs/visualization/churn_trends.html')

        qmap = summary.get('quarterly')
        if qmap:
            plot_quarterly_totals(qmap, out_png='outputs/visualization/quarterly_totals.png', out_html='outputs/visualization/quarterly_totals.html')

        try:
            plot_top_lost_clients(pivot, list(pivot.columns)[1], list(pivot.columns)[2] if pivot.shape[1] > 2 else list(pivot.columns)[1], out_png='outputs/visualization/top_lost_clients.png')
        except Exception:
            pass
    except Exception:
        pass

    return summary

if __name__ == '__main__':
    import os
    sample = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_input.csv')
    sample = os.path.abspath(sample)
    try:
        out = run_full_pipeline(sample)
        print(json.dumps(out, indent=2)[:2000])
    except Exception as e:
        logger.warning('No sample dataset found or error running smoke test: %s', e)
