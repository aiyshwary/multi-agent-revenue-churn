import os
import json
import pytest
from agents.core import run_full_pipeline
from agents.llm import SemanticMemory, LLMClient
from agents.orchestrator import Orchestrator

DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(DIR, '..'))
SAMPLE = os.path.join(ROOT, 'data', 'sample_input.csv')


def test_llm_planner_mode_runs_and_returns_plan():
    out = run_full_pipeline(SAMPLE, use_llm=True)
    assert isinstance(out['plan'], list)
    assert any('churn' in (p.get('action') or p.get('step') or '') for p in out['plan'])


def test_semantic_memory_add_and_query():
    sm = SemanticMemory(dim=16)
    sm.add('a', 'top selling product region east', {'tag': 'insight'})
    sm.add('b', 'low selling product region west', {'tag': 'insight'})
    res = sm.query('selling product east', top_k=1)
    assert len(res) == 1
    assert res[0][0] == 'a'


def test_llm_client_enforces_token_budget():
    client = LLMClient(provider='mock', token_budget=10, max_response_tokens=5)
    with pytest.raises(ValueError):
        client.call('x' * 100, max_tokens=20)


def test_orchestrator_handles_transient_failure_and_retries():
    o = Orchestrator(data_path=SAMPLE)
    orig = o.executor.execute
    calls = {'n': 0}

    def flaky(step, context):
        if step.get('step') == 'load_data' and calls['n'] == 0:
            calls['n'] += 1
            raise RuntimeError('transient error')
        return orig(step, context)

    o.executor.execute = flaky
    res = o.run()
    assert calls['n'] == 1
    assert res['metrics']['counters'].get('step.load_data.success', 0) >= 1


def test_semantic_memory_persist_and_load(tmp_path):
    sm = SemanticMemory(dim=8)
    sm.add('k1', 'important note about client churn', {'client': 'C1'})
    p = tmp_path / 'sm.json'
    sm.persist(str(p))

    sm2 = SemanticMemory(dim=8)
    sm2.load(str(p))
    q = sm2.query('client churn', top_k=1)
    assert q and q[0][0] == 'k1'
