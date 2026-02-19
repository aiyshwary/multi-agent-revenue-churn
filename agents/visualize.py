"""Visualization helpers for churn/revenue trends.

Produces static PNGs (Matplotlib) and interactive HTML (Plotly) and is
lightweight so it can be used from notebooks or CLI scripts.
"""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd

try:
    import plotly.express as px
    import plotly.io as pio
except Exception:
    px = None
    pio = None

def _ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

def plot_churn_trends(
    churn_report: List[Dict[str, Any]],
    out_png: str = "outputs/visualization/churn_trends.png",
    out_html: Optional[str] = "outputs/visualization/churn_trends.html",
    show: bool = False,
):
    """Create a grouped bar chart for revenue_loss vs revenue_gain per period.

    - `churn_report` : list of dicts matching outputs/churn_report.json
    - saves a PNG (Matplotlib) and optional interactive HTML (Plotly)
    """
    def _get_prev(r):
        return r.get('q_prev') or r.get('prev_quarter') or r.get('q_prev'.upper())

    def _get_curr(r):
        return r.get('q_curr') or r.get('curr_quarter') or r.get('q_curr'.upper())

    labels = [f"{_get_prev(r)}->{_get_curr(r)}" for r in churn_report]
    losses = [float(r.get("revenue_loss", r.get('revenueLoss', 0.0))) for r in churn_report]
    gains = [float(r.get("revenue_gain", r.get('revenueGain', 0.0))) for r in churn_report]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], losses, width, label="Revenue loss", color="#d9534f")
    ax.bar([i + width / 2 for i in x], gains, width, label="Revenue gain", color="#5cb85c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("USD")
    ax.set_title("Quarterly churn: revenue loss vs gain")
    ax.legend()
    plt.tight_layout()

    _ensure_dir(out_png)
    fig.savefig(out_png)
    if show:
        plt.show()
    plt.close(fig)

    if out_html:
        _ensure_dir(out_html)
        if px is not None:
            df = pd.DataFrame({"period": labels, "revenue_loss": losses, "revenue_gain": gains})
            df = df.melt(id_vars="period", value_vars=["revenue_loss", "revenue_gain"], var_name="type", value_name="amount")
            figp = px.bar(df, x="period", y="amount", color="type", barmode="group", title="Quarterly churn: revenue loss vs gain")
            try:
                pio.write_html(figp, out_html, include_plotlyjs="cdn")
            except Exception:
                with open(out_html, "w") as f:
                    f.write(f"<html><body><h3>Interactive plot not available — see PNG</h3><img src=\"{os.path.basename(out_png)}\"/></body></html>")
        else:
            with open(out_html, "w") as f:
                f.write(f"<html><body><h3>Interactive plot not available (plotly missing)</h3><img src=\"{os.path.basename(out_png)}\"/></body></html>")

    return out_png, out_html


def plot_quarterly_totals(
    quarterly: Dict[str, Any],
    out_png: str = "outputs/visualization/quarterly_totals.png",
    out_html: Optional[str] = "outputs/visualization/quarterly_totals.html",
    show: bool = False,
):
    """Plot total revenue per quarter (aggregating available `TotalRevenue`).

    `quarterly` can be a dict mapping quarter -> DataFrame (or list of records).
    """
    quarters = []
    totals = []
    for q, tab in sorted(quarterly.items()):
        if isinstance(tab, (list, tuple)):
            df = pd.DataFrame(tab)
        elif isinstance(tab, pd.DataFrame):
            df = tab
        else:
            df = pd.DataFrame(tab)
        total = float(df["TotalRevenue"].sum()) if "TotalRevenue" in df.columns else float(df.get("TotalRevenue", 0))
        quarters.append(q)
        totals.append(total)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(quarters, totals, color="#337ab7")
    ax.set_title("Total revenue by quarter")
    ax.set_ylabel("USD")
    plt.tight_layout()
    _ensure_dir(out_png)
    fig.savefig(out_png)
    if show:
        plt.show()
    plt.close(fig)

    if out_html and px is not None:
        df = pd.DataFrame({"quarter": quarters, "total": totals})
        figp = px.bar(df, x="quarter", y="total", title="Total revenue by quarter")
        _ensure_dir(out_html)
        pio.write_html(figp, out_html, include_plotlyjs="cdn")

    return out_png, out_html


def plot_top_lost_clients(
    pivot_df: pd.DataFrame,
    q_prev: str,
    q_curr: str,
    out_png: str = "outputs/visualization/top_lost_clients.png",
    top_n: int = 10,
    show: bool = False,
):
    """Plot the top N lost clients (by previous-quarter revenue).

    `pivot_df` is the client-quarter pivot (Client ID + Q columns).
    """
    df = pivot_df.copy()
    col_prev = q_prev
    col_curr = q_curr
    if col_prev not in df.columns or col_curr not in df.columns:
        raise ValueError("Requested quarters not present in pivot DataFrame")

    lost = df[(df[col_prev] > 0) & (df[col_curr] == 0)].copy()
    lost = lost.sort_values(col_prev, ascending=False).head(top_n)
    labels = lost["Client ID"].astype(str)
    vals = lost[col_prev].astype(float)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(labels))))
    ax.barh(labels, vals, color="#d9534f")
    ax.set_xlabel("Revenue (USD)")
    ax.set_title(f"Top {top_n} lost clients: {q_prev} -> {q_curr}")
    plt.tight_layout()
    _ensure_dir(out_png)
    fig.savefig(out_png)
    if show:
        plt.show()
    plt.close(fig)

    return out_png
