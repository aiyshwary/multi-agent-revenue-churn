import time
import math
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TokenEstimator:
    """Very small token estimator used for budgeting and chunking.
    Heuristic: ~1 token per 4 characters (keeps behaviour consistent with repo).
    """

    @staticmethod
    def estimate(text: str) -> int:
        return max(1, len(text) // 4)


class MockLLM:
    """Deterministic, local 'LLM' used for tests and offline runs.
    It recognizes simple instructions for planning and validation and returns
    structured text to simulate an LLM response.
    """

    @staticmethod
    def generate(prompt: str, max_tokens: Optional[int] = None) -> str:
        p = prompt.lower()
        if "plan" in p or "decompose" in p:
            return (
                "[{'step':'load_data'},{'step':'assign_quarter'},{'step':'aggregate_quarter_region'},{'step':'client_quarterly'},{'step':'compute_churn'},{'step':'validate'},{'step':'reflect'}]"
            )
        if "validate" in p or "validation" in p:
            if "mismatch" in p or "error" in p:
                return "RETRY: Re-normalize dates and re-run aggregation"
            return "ACCEPT"
        if "summarize" in p:
            return (prompt[:120] + ("..." if len(prompt) > 120 else ""))
        return prompt[: max_tokens or 200]


class LLMClient:
    """Lightweight adapter that supports a `mock` provider (default) and may be
    extended to include real providers (OpenAI) without changing callers.

    Important: imports for real providers should be lazy (only when used).
    """

    def __init__(self, provider: str = "mock", token_budget: int = 2048, max_response_tokens: int = 512, openai_model: Optional[str] = None):
        self.provider = provider
        self.token_budget = token_budget
        self.max_response_tokens = max_response_tokens
        self.openai_model = openai_model or "gpt-3.5-turbo"
        self.client = None

    def estimate_tokens(self, text: str) -> int:
        return TokenEstimator.estimate(text)

    def call(self, prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        max_tokens = max_tokens or self.max_response_tokens
        tokens = self.estimate_tokens(prompt) + max_tokens
        if tokens > self.token_budget:
            raise ValueError(f"Prompt+response ({tokens}) exceeds token budget ({self.token_budget})")

        if self.provider == "mock":
            text = MockLLM.generate(prompt, max_tokens=max_tokens)
            used = self.estimate_tokens(text)
            return {"text": text, "tokens_used": used}

        if self.provider == "openai":
            try:
                import openai
            except Exception as e:
                raise RuntimeError("OpenAI SDK not installed; install the 'openai' package to use provider='openai'.") from e

            api_key = openai.api_key or None
            if not api_key:
                import os

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set; cannot call OpenAI provider without credentials.")
                openai.api_key = api_key

            try:
                resp = openai.ChatCompletion.create(
                    model=getattr(self, "openai_model", "gpt-3.5-turbo"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content
            except Exception:
                resp = openai.Completion.create(model=getattr(self, "openai_model", "text-davinci-003"), prompt=prompt, max_tokens=max_tokens)
                text = resp.choices[0].text
            used = self.estimate_tokens(text)
            return {"text": text, "tokens_used": used}

        raise NotImplementedError(f"Provider '{self.provider}' is not implemented.")

    def chunk_and_call(self, texts: List[str], template: str, context_token_limit: int) -> List[Dict[str, Any]]:
        """Split `texts` into chunks where (template + chunk) fit within context_token_limit.
        Returns list of LLM responses for each chunk.
        """
        chunks: List[List[str]] = []
        cur: List[str] = []
        cur_tokens = 0
        for t in texts:
            t_tokens = self.estimate_tokens(t)
            if cur and (cur_tokens + t_tokens + self.estimate_tokens(template)) > context_token_limit:
                chunks.append(cur)
                cur = [t]
                cur_tokens = t_tokens
            else:
                cur.append(t)
                cur_tokens += t_tokens
        if cur:
            chunks.append(cur)

        responses = []
        for c in chunks:
            prompt = template + "\n\n" + "\n".join(c)
            responses.append(self.call(prompt))
        return responses


class SemanticMemory:
    """Simple deterministic in-memory semantic index.

    This is NOT a production vector DB — it's a deterministic, dependency‑free
    implementation suitable for unit tests and local runs. It supports
    add() and query() (cosine similarity) and optional persistence.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._store: Dict[str, Dict[str, Any]] = {}

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros((self.dim,), dtype=float)
        import re
        for token in re.findall(r"\w+", text.lower()):
            idx = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            h = int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)
            rng = np.random.RandomState(h)
            vec = rng.normal(size=(self.dim,)).astype(float)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec

    def add(self, key: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        vec = self._embed(text)
        self._store[key] = {"text": text, "vec": vec, "meta": metadata or {}}

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self._store:
            return []
        qv = self._embed(text)
        keys = list(self._store.keys())
        mats = np.stack([self._store[k]["vec"] for k in keys], axis=0)
        sims = mats.dot(qv)
        idx = np.argsort(-sims)[:top_k]
        results = [(keys[i], float(sims[i]), self._store[keys[i]]["meta"]) for i in idx]
        return results

    def persist(self, path: str):
        import json

        out = {k: {"text": v["text"], "vec": v["vec"].tolist(), "meta": v["meta"]} for k, v in self._store.items()}
        with open(path, "w") as f:
            json.dump(out, f)

    def load(self, path: str):
        import json

        with open(path, "r") as f:
            obj = json.load(f)
        for k, v in obj.items():
            self._store[k] = {"text": v["text"], "vec": np.array(v["vec"]), "meta": v.get("meta", {})}
