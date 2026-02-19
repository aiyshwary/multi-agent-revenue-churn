import json
import os
from jsonschema import validate
from agents.core import run_full_pipeline

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(DIR, '..'))
SAMPLE = os.path.join(ROOT, 'data', 'sample_input.csv')
SCHEMA_PATH = os.path.join(ROOT, 'schema', 'output_schema.json')


def test_run_output_conforms_to_schema():
    out = run_full_pipeline(SAMPLE)
    with open(SCHEMA_PATH, 'r') as f:
        schema = json.load(f)
    validate(instance=out, schema=schema)
