"""Simple CLI to run the Orchestrator deterministically from the command line."""
import json
import argparse
from agents import Orchestrator


def main(path: str, use_llm: bool = False, planner_provider: str = "mock", validator_provider: str = "mock", reflection_provider: str = "mock", token_budget: int = 2048, openai_model: str = None):
    planner_client = None
    validator_client = None
    reflection_client = None
    if use_llm:
        from agents.llm import LLMClient
        planner_client = LLMClient(provider=planner_provider, token_budget=token_budget, openai_model=openai_model)
        validator_client = LLMClient(provider=validator_provider, token_budget=token_budget, openai_model=openai_model)
        reflection_client = LLMClient(provider=reflection_provider, token_budget=token_budget, openai_model=openai_model)

    o = Orchestrator(data_path=path, use_llm=use_llm, planner_llm=planner_client, validator_llm=validator_client, reflection_llm=reflection_client)
    res = o.run()

    summary = {
        "suggestions": res.get("suggestions"),
        "reflections": res.get("reflections"),
        "log": [
            {"step": e["step"]["step"], "status": e.get("status"), "attempts": e.get("attempts")} for e in res.get("log", [])
        ],
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ar 7.xlsx", help="path to Excel file")
    args = parser.parse_args()
    main(args.data)
