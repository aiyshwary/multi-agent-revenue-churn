# Contributing

Thanks for taking a look — contributions, fixes, and suggestions are very welcome. Below are the simplest ways to get started and a short checklist for consistent PRs.

## Quick start (local development)
1. Create a virtual environment and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the pipeline locally:
   ```bash
   python run_orchestrator.py --data "data/sample_input.csv"
   ```
3. Run the test suite:
   ```bash
   pytest -q
   ```

## How to contribute
- Fork the repo and open a branch named `fix/` or `feature/` with a short description.
- Keep changes small and focused — one logical change per PR.
- Add tests for bug fixes and new behavior.
- Update `README.md` or `ARCHITECTURE.md` if you change public behavior or APIs.

## PR checklist
- [ ] Code builds and tests pass locally (`pytest -q`).
- [ ] New behaviour is covered by tests or a short justification is provided.
- [ ] Changes are documented (README or ARCHITECTURE where appropriate).
- [ ] Commit messages are descriptive and small (imperative tense).

## Coding & style
- Follow existing project style (type hints where useful, keep numeric logic in `agents/tools.py` and `agents/core.py`).
- Keep LLM prompts and reasoning separate from deterministic computation.

## Reporting issues
- Open an issue with a short title, steps to reproduce, expected vs actual behavior, and a small sample input if relevant.

If you’d like, I can add templates for issues and pull requests — tell me which templates to include.
