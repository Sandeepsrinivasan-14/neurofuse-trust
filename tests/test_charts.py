import pandas as pd
import plotly.graph_objects as go

from neurofuse import charts
from neurofuse.models import CLASSES, MODELS


def test_empty_figure_shows_the_prompt():
    fig = charts.empty(text="Nothing yet")
    assert isinstance(fig, go.Figure)
    assert fig.layout.annotations[0].text == "Nothing yet"


def test_probability_bars_has_one_bar_per_class_sorted_ascending():
    dist = {"glioma": 0.1, "meningioma": 0.2, "notumor": 0.6, "pituitary": 0.1}
    fig = charts.probability_bars(dist)
    bar = fig.data[0]
    assert len(bar.x) == len(CLASSES)
    assert list(bar.x) == sorted(bar.x)
    assert bar.y[-1] == "Notumor"


def test_load_results_is_none_when_the_csv_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(charts, "RESULTS_DIR", tmp_path)
    assert charts.load_results() is None


def test_load_results_reads_the_comparison_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(charts, "RESULTS_DIR", tmp_path)
    pd.DataFrame({"test_acc": [0.95]}, index=["hybrid"]).to_csv(tmp_path / "final_comparison.csv")
    assert charts.load_results().loc["hybrid", "test_acc"] == 0.95


def test_benchmark_bars_plots_accuracy_and_f1_for_each_model():
    df = pd.DataFrame({"test_acc": [0.95, 0.93, 0.92, 0.94], "test_macro_f1": [0.94, 0.92, 0.91, 0.93]}, index=list(MODELS))
    fig = charts.benchmark_bars(df)
    assert [t.name for t in fig.data] == ["Accuracy", "Macro-F1"]
    assert len(fig.data[0].x) == len(MODELS)
