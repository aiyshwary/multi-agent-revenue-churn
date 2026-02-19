import os
import pandas as pd
from agents.visualize import plot_churn_trends, plot_quarterly_totals, plot_top_lost_clients

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(DIR, '..'))
SAMPLE = os.path.join(ROOT, 'data', 'sample_input.csv')


def test_plot_churn_trends(tmp_path):
    sample = [
        {"q_prev": "Q1", "q_curr": "Q2", "revenue_loss": 100.0, "revenue_gain": 25.0},
        {"q_prev": "Q2", "q_curr": "Q3", "revenue_loss": 50.0, "revenue_gain": 10.0},
    ]
    out_png = str(tmp_path / "churn.png")
    out_html = str(tmp_path / "churn.html")
    png, html = plot_churn_trends(sample, out_png=out_png, out_html=out_html)
    assert os.path.exists(png)
    assert os.path.exists(html)


def test_plot_quarterly_totals(tmp_path):
    q = {
        "Q1": [{"Region": "AMER", "TotalRevenue": 1000}],
        "Q2": [{"Region": "EMEA", "TotalRevenue": 500}],
    }
    out_png = str(tmp_path / "quarters.png")
    png, html = plot_quarterly_totals(q, out_png=out_png, out_html=str(tmp_path / "quarters.html"))
    assert os.path.exists(png)


def test_plot_top_lost_clients(tmp_path):
    df = pd.DataFrame({
        "Client ID": ["A", "B", "C"],
        "Q1": [100.0, 50.0, 0.0],
        "Q2": [0.0, 0.0, 0.0],
    })
    out_png = str(tmp_path / "lost.png")
    p = plot_top_lost_clients(df, "Q1", "Q2", out_png=out_png, top_n=2)
    assert os.path.exists(p)


def test_pipeline_writes_visualizations(tmp_path):
    from agents.core import run_full_pipeline
    out = run_full_pipeline(SAMPLE)
    assert os.path.exists('outputs/visualization/churn_trends.png')
    assert os.path.exists('outputs/visualization/churn_trends.html')
    assert os.path.exists('outputs/visualization/quarterly_totals.png')
    assert os.path.exists('outputs/visualization/top_lost_clients.png') or os.path.exists('outputs/visualization/top_lost_q1_q2.png')
