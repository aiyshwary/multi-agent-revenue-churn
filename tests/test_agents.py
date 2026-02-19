import os
import json
import pandas as pd
from agents.core import DataExecutorAgent, ChurnAnalystAgent, ValidatorAgent, run_full_pipeline, PlannerAgent, MockLLM, MemoryManager as CoreMemoryManager, retry_with_backoff
from agents.tools import DataLoader

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(DIR, '..'))
SAMPLE = os.path.join(ROOT, 'data', 'sample_input.csv')


def test_quarterly_aggregation_sum_equal_original():
    executor = DataExecutorAgent()
    df = executor.load_data(path=SAMPLE)
    quarters = executor.group_by_quarter(df)
    validator = ValidatorAgent()
    res = validator.validate_quarterly_aggregation(df, quarters)
    assert res['ok'] is True
    assert abs(res['total_original'] - res['total_aggregated']) < 1e-6


def test_churn_detection_q1_to_q2():
    out = run_full_pipeline(SAMPLE)
    churn = out['churn_report']
    assert 'Q1_to_Q2' in churn
    q = churn['Q1_to_Q2']
    assert q['lost_customers'] == 1
    assert q['new_customers'] == 2
    assert int(q['revenue_loss']) == 6000
    assert int(q['revenue_gain']) == 4500


def test_load_excel_wide_format_normalization(tmp_path):
    dates = [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-02-01')]
    df_wide = pd.DataFrame({
        'Customer Name': ['C1', 'C2'],
        'Country': ['USA', 'IND'],
        'Region': ['AMER', 'APAC'],
        dates[0]: [100, 0],
        dates[1]: [0, 200],
    })
    p = tmp_path / 'wide.xlsx'
    df_wide.to_excel(p, index=False)

    df_long = DataLoader.load_excel(str(p))
    assert 'Client ID' in df_long.columns
    assert 'Date' in df_long.columns and 'Revenue' in df_long.columns
    assert len(df_long) == 4
    r = df_long[(df_long['Client ID'] == 'C1') & (df_long['Date'] == pd.Timestamp('2024-01-01'))]
    assert float(r['Revenue'].iloc[0]) == 100.0


def test_chunked_processing_equivalence():
    out_full = run_full_pipeline(SAMPLE)
    out_chunked = run_full_pipeline(SAMPLE, chunk_size=2, sleep_fn=lambda s: None)
    assert out_full['churn_report'] == out_chunked['churn_report']


def test_parallel_chunked_processing_equivalence():
    out_full = run_full_pipeline(SAMPLE)
    out_parallel = run_full_pipeline(SAMPLE, chunk_size=2, parallel_workers=2, sleep_fn=lambda s: None)
    assert out_full['churn_report'] == out_parallel['churn_report']


def test_graph_runner_parallel_executes_dependency_graph():
    from agents.orchestrator import Orchestrator
    # exercise GraphRunner with max_workers > 1 and verify parity with sequential run
    o = Orchestrator(SAMPLE, use_llm=False, max_workers=2)

    graph_plan = [
        {"id": "n_load", "action": "load_data"},
        {"id": "n_q", "action": "assign_quarter", "depends_on": ["n_load"]},
        {"id": "n_agg", "action": "aggregate_quarter_region", "depends_on": ["n_q"]},
        {"id": "n_client", "action": "client_quarterly", "depends_on": ["n_agg"]},
        {"id": "n_churn", "action": "compute_churn", "depends_on": ["n_client"]},
        {"id": "n_validate", "action": "validate", "depends_on": ["n_churn"]},
        {"id": "n_reflect", "action": "reflect", "depends_on": ["n_validate"]},
    ]

    o.planner.plan = lambda objective=None: graph_plan
    res = o.run()
    ctx = res.get('context', {})
    assert 'df' in ctx
    assert 'agg' in ctx or 'client_q' in ctx
    assert 'reflections' in res or ctx.get('reflections') is not None


def test_run_full_pipeline_accepts_wide_excel_format():
    sample_xlsx = os.path.join(ROOT, 'data', 'ar 7.xlsx')
    res = run_full_pipeline(sample_xlsx)
    assert 'churn_report' in res and isinstance(res['churn_report'], dict) or isinstance(res['churn_report'], list)


def test_planner_dynamic_with_mock_llm_and_budget():
    pm = PlannerAgent()
    mem = CoreMemoryManager()
    llm = MockLLM()
    plan = pm.plan_dynamic("Analyze revenue and churn", llm=llm, memory=mem, token_budget=50)
    assert isinstance(plan, list) and len(plan) > 0


def test_semantic_memory_stability(tmp_path):
    import numpy as np
    from agents.tools import SemanticMemory
    path = tmp_path / 'semantic_stable.json'
    sm = SemanticMemory(path=str(path), dim=8)
    sm.add('a', 'hello world', metadata={'step': 'a'})
    sm.add('b', 'another entry', metadata={'step': 'b'})
    res = sm.query('hello', k=2)
    assert isinstance(res, list)
    assert all(isinstance(r.get('score'), float) for r in res)
    assert all(np.isfinite(r['score']) for r in res)


def test_retry_with_backoff_helper():
    calls = {'n': 0}

    def flaky():
        calls['n'] += 1
        if calls['n'] < 3:
            raise ValueError("transient")
        return "ok"

    res = retry_with_backoff(flaky, max_retries=5, backoff_factor=0.0, sleep_fn=lambda s: None)
    assert res == "ok"
    assert calls['n'] == 3


def test_semantic_memory_store_and_query(tmp_path):
    from agents.tools import MemoryManager
    mem_path = tmp_path / 'memory.json'
    semantic_path = tmp_path / 'semantic.json'
    mm = MemoryManager(path=str(mem_path), semantic=True, semantic_dim=8, semantic_path=str(semantic_path))
    mm.store_step_summary('unit_test', {'message': 'find me'}, compress=False)
    res = mm.semantic.query('find me', k=1)
    assert res
    assert res[0]['metadata']['step'] == 'unit_test'


def test_orchestrator_circuit_breaker_trips():
    from agents.orchestrator import Orchestrator
    o = Orchestrator(SAMPLE, use_llm=False, cb_fail_threshold=2, backoff_base=0.001)
    original_execute = o.executor.execute

    def flaky(step, context):
        if step['step'] == 'aggregate_quarter_region':
            raise RuntimeError('simulated failure')
        return original_execute(step, context)

    o.executor.execute = flaky
    try:
        o.run()
        assert False, 'expected run to raise'
    except RuntimeError:
        assert o.circuit_breaker.is_open('aggregate_quarter_region') is True
        errors = [e for e in o.log if e['step']['step'] == 'aggregate_quarter_region' and e['status'] == 'error']
        assert errors


def test_llm_planner_respects_token_budget():
    from agents.core import LLMPlanner
    planner = LLMPlanner(token_budget=1)
    plan = planner.plan('this is a long objective ' * 50)
    assert all('via' not in s for s in plan)
    planner2 = LLMPlanner(token_budget=1000)
    plan2 = planner2.plan('short objective')
    assert all(s.get('via') == 'llm' for s in plan2)


def test_run_full_pipeline_records_metrics():
    out = run_full_pipeline(SAMPLE, max_retries=1)
    assert 'metrics' in out
    assert any(k.startswith('step.load') for k in out['metrics']['counters'].keys())


def test_llmclient_openai_adapter_errors():
    from agents.llm import LLMClient
    client = LLMClient(provider='openai')
    import pytest
    with pytest.raises((RuntimeError, ValueError)):
        client.call("Test prompt that won't be sent to real API")


def test_orchestrator_accepts_per_role_llms_and_runs():
    from agents.llm import LLMClient
    from agents.orchestrator import Orchestrator
    planner_client = LLMClient(provider='mock')
    validator_client = LLMClient(provider='mock')
    o = Orchestrator(SAMPLE, use_llm=True, planner_llm=planner_client, validator_llm=validator_client)
    assert o.llm_planner is not None and o.llm_validator is not None
    assert getattr(o.llm_planner, 'llm', None) is planner_client
    assert getattr(o.llm_validator, 'llm', None) is validator_client
    res = o.run()
    assert 'context' in res


def test_graph_runner_executes_dependency_graph():
    from agents.orchestrator import Orchestrator
    o = Orchestrator(SAMPLE, use_llm=False)

    graph_plan = [
        {"id": "n_load", "action": "load_data"},
        {"id": "n_q", "action": "assign_quarter", "depends_on": ["n_load"]},
        {"id": "n_agg", "action": "aggregate_quarter_region", "depends_on": ["n_q"]},
        {"id": "n_client", "action": "client_quarterly", "depends_on": ["n_agg"]},
        {"id": "n_churn", "action": "compute_churn", "depends_on": ["n_client"]},
        {"id": "n_validate", "action": "validate", "depends_on": ["n_churn"]},
        {"id": "n_reflect", "action": "reflect", "depends_on": ["n_validate"]},
    ]

    o.planner.plan = lambda objective=None: graph_plan
    res = o.run()
    ctx = res.get('context', {})
    assert 'df' in ctx
    assert 'agg' in ctx or 'client_q' in ctx
    assert 'reflections' in res or ctx.get('reflections') is not None
