import pandas as pd
import numpy as np
import json
import os
import hashlib
from datetime import datetime
from typing import Iterator, Dict, Any, Tuple


class TokenEstimator:
    @staticmethod
    def estimate(text: str) -> int:
        return max(1, len(text) // 4)


class DataLoader:
    @staticmethod
    def load_excel(path: str) -> pd.DataFrame:
        """Deterministically load an Excel file into a DataFrame.

        - Handles two common layouts:
          1) long format (columns: Client ID, Date, Revenue, Region, Country)
          2) wide format (customer + date columns) — melts into long format
        """
        df = pd.read_excel(path, engine="openpyxl")

        date_cols = [c for c in df.columns if isinstance(c, (pd.Timestamp, datetime))]
        import re

        for c in df.columns:
            if isinstance(c, str) and re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}$", c):
                date_cols.append(c)

        if date_cols:
            id_vars = [c for c in df.columns if c not in date_cols]
            melted = df.melt(id_vars=id_vars, value_vars=date_cols, var_name="Date", value_name="Revenue")
            if 'Customer Name' in melted.columns and 'Client ID' not in melted.columns:
                melted = melted.rename(columns={'Customer Name': 'Client ID'})
            if 'Client Name' in melted.columns and 'Client ID' not in melted.columns:
                melted = melted.rename(columns={'Client Name': 'Client ID'})
            melted['Date'] = pd.to_datetime(melted['Date'])
            melted['Revenue'] = pd.to_numeric(melted['Revenue'], errors='coerce').fillna(0.0)
            cols = ['Client ID', 'Country', 'Region', 'Date', 'Revenue']
            cols_present = [c for c in cols if c in melted.columns]
            return melted[cols_present]

        return df

    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Yield DataFrame chunks (row-based) to simulate streaming/chunking."""
        n = len(df)
        for i in range(0, n, chunk_size):
            yield df.iloc[i : i + chunk_size].copy()


def assign_quarter(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["Quarter"] = df[date_col].dt.to_period("Q").astype(str)
    return df


class Aggregator:
    @staticmethod
    def aggregate_by_quarter_region(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        grouped = (
            df.groupby(["Quarter", "Region", "Country"])["Revenue"]
            .agg(TotalRevenue="sum", NumRows="count")
            .reset_index()
        )
        return grouped

    @staticmethod
    def client_quarterly_revenue(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Client ID' not in df.columns and 'ClientID' in df.columns:
            df = df.rename(columns={'ClientID': 'Client ID'})
        if 'Quarter' not in df.columns:
            if 'Date' in df.columns:
                df['Quarter'] = pd.to_datetime(df['Date']).dt.to_period('Q').astype(str)
            else:
                raise ValueError("DataFrame must contain either 'Quarter' or 'Date'")
        pivot = (
            df.groupby(["Client ID", "Quarter"])["Revenue"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        return pivot


class ChurnCalculator:
    @staticmethod
    def churn_between_quarters(client_q_rev: pd.DataFrame, q_prev: str, q_curr: str) -> Dict[str, Any]:
        prev_rev = client_q_rev[["Client ID"] + ([q_prev] if q_prev in client_q_rev.columns else [])].copy()
        curr_rev = client_q_rev[["Client ID"] + ([q_curr] if q_curr in client_q_rev.columns else [])].copy()

        prev_map = dict(zip(prev_rev["Client ID"], prev_rev[q_prev])) if q_prev in prev_rev.columns else {}
        curr_map = dict(zip(curr_rev["Client ID"], curr_rev[q_curr])) if q_curr in curr_rev.columns else {}

        prev_clients = set([cid for cid, r in prev_map.items() if r > 0])
        curr_clients = set([cid for cid, r in curr_map.items() if r > 0])

        lost = prev_clients - curr_clients
        new = curr_clients - prev_clients

        revenue_loss = sum(prev_map.get(cid, 0) for cid in lost)
        revenue_gain = sum(curr_map.get(cid, 0) for cid in new)

        return {
            "q_prev": q_prev,
            "q_curr": q_curr,
            "lost_customers": list(lost),
            "new_customers": list(new),
            "num_lost": len(lost),
            "num_new": len(new),
            "revenue_loss": float(revenue_loss),
            "revenue_gain": float(revenue_gain),
        }


class Validator:
    @staticmethod
    def totals_match(input_df: pd.DataFrame, aggregated_df: pd.DataFrame, tol: float = 1e-6) -> Tuple[bool, str]:
        input_total = float(input_df["Revenue"].sum())
        agg_total = float(aggregated_df["TotalRevenue"].sum())
        if abs(input_total - agg_total) <= tol:
            return True, "Totals match"
        return False, f"Total mismatch: input={input_total} agg={agg_total}"

    @staticmethod
    def sanity_check_churn(churn_report: Dict[str, Any]) -> Tuple[bool, str]:
        if churn_report["revenue_loss"] < 0 or churn_report["revenue_gain"] < 0:
            return False, "Negative revenue in churn report"
        return True, "Churn report sane"


class SemanticMemory:
    """Lightweight, deterministic local vector store (no external deps).
    Embeddings are simulated deterministically from text so tests remain reproducible.
    Persisted to a JSON file under `semantic_path`.
    """

    def __init__(self, path: str = "outputs/semantic_memory.json", dim: int = 64):
        self.path = path
        self.dim = dim
        self._load()

    def _load(self):
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {"entries": []}
        self.entries = data.get("entries", [])

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"entries": self.entries}, f, indent=2)

    def _embed(self, text: str):
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.RandomState(h % (2 ** 32 - 1))
        vec = rng.randn(self.dim).astype(float)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.linalg.norm(vec)
        eps = 1e-12
        if not np.isfinite(norm) or norm < eps:
            vec = np.ones((self.dim,), dtype=float) * 1e-6
            norm = np.linalg.norm(vec)
        return (vec / (norm + 1e-12)).tolist()

    def add(self, id: str, text: str, metadata: dict = None):
        vec = self._embed(text)
        self.entries.append({"id": id, "text": text, "vector": vec, "metadata": metadata or {}})
        self.save()

    def query(self, text: str, k: int = 5):
        if not self.entries:
            return []
        qv = np.asarray(self._embed(text), dtype=np.float64)
        qv = np.nan_to_num(qv, nan=0.0, posinf=0.0, neginf=0.0)
        qv_norm = np.linalg.norm(qv)
        if qv_norm > 0:
            qv = qv / (qv_norm + 1e-12)
        else:
            qv = np.zeros_like(qv)

        mat = np.array([np.nan_to_num(e["vector"], nan=0.0, posinf=0.0, neginf=0.0) for e in self.entries], dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1)
        norms = np.where(np.isfinite(norms) & (norms > 0), norms, 1.0)
        mat = mat / norms[:, None]
        try:
            sims = mat.dot(qv)
        except Exception:
            sims = np.zeros(mat.shape[0], dtype=float)
        sims = np.nan_to_num(sims, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
        idx = np.argsort(-sims)[:k]
        results = []
        for i in idx:
            score = float(sims[i]) if np.isfinite(sims[i]) else float('-inf')
            results.append({"id": self.entries[i]["id"], "score": score, "metadata": self.entries[i]["metadata"], "text": self.entries[i]["text"]})
        return results


class MemoryManager:
    """Short-term + compressed long-term memory, with an optional semantic vector store.
    - short_term: transient run data
    - long_term: compressed summaries persisted to JSON
    - semantic: optional deterministic vector store for retrieval
    - token_log: crude token accounting for instrumentation
    """

    def __init__(self, path: str = "outputs/memory.json", semantic: bool = True, semantic_dim: int = 64, semantic_path: str = None):
        self.short_term: Dict[str, Any] = {}
        self.path = path
        self.long_term: Dict[str, Any] = self._load_long_term()
        semantic_path = semantic_path or os.path.join(os.path.dirname(path) or ".", "semantic_memory.json")
        self.semantic = SemanticMemory(path=semantic_path, dim=semantic_dim) if semantic else None
        self.token_log: List[Dict[str, Any]] = []

    def _load_long_term(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_long_term(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.long_term, f, indent=2)

    def store_step_summary(self, step: str, summary: Dict[str, Any], compress: bool = True, to_semantic: bool = True):
        self.short_term[step] = summary
        if compress:
            key = f"summary:{step}"
            self.long_term[key] = self._compress(summary)
            self.save_long_term()
        if to_semantic and self.semantic:
            text = json.dumps({"step": step, "summary": summary})
            self.semantic.add(id=f"{step}:{int(datetime.utcnow().timestamp())}", text=text, metadata={"step": step})
        self.token_log.append({"type": "store_step", "tokens": TokenEstimator.estimate(str(summary))})

    def has_executed(self, key: str) -> bool:
        return bool(self.long_term.get(f"done:{key}"))

    def mark_executed(self, key: str):
        self.long_term[f"done:{key}"] = True
        self.save_long_term()

    def _compress(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        compressed = {}
        for k, v in summary.items():
            if isinstance(v, (int, float, str)):
                compressed[k] = v
            elif isinstance(v, list):
                compressed[k] = len(v)
            elif isinstance(v, dict):
                compressed[k] = {kk: (len(vv) if isinstance(vv, list) else str(type(vv))) for kk, vv in v.items()}
        return compressed
