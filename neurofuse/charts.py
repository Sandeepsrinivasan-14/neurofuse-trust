import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from .models import CKPT_DIR, CLASSES, MODELS, RESULTS_DIR

INK = "#e6ebff"
MUTED = "#8b93b8"
GRID = "rgba(139,147,184,0.14)"
FONT = "Inter, system-ui, sans-serif"

CLASS_COLORS = {
    "glioma": "#f472b6",
    "meningioma": "#a78bfa",
    "notumor": "#34d399",
    "pituitary": "#38bdf8",
}
MODEL_COLORS = {
    "hybrid": "#22d3ee",
    "effnet": "#a78bfa",
    "swin": "#f472b6",
    "coatnet": "#fbbf24",
}
CAM_SCALE = [
    [0.0, "#0b1030"], [0.25, "#3b2a8f"], [0.5, "#8b5cf6"],
    [0.7, "#ec4899"], [0.85, "#fb923c"], [1.0, "#fde68a"],
]


def _layout(fig, height=320, **kw):
    base = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=12, r=12, t=40, b=12),
        hoverlabel=dict(bgcolor="#11152b", bordercolor="#2a3160", font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(color=MUTED)),
    )
    base.update(kw)
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    return fig


def empty(height=320, text="Upload a scan to begin"):
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, font=dict(color=MUTED, size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, height)


def probability_bars(dist):
    order = sorted(dist, key=dist.get)
    vals = [dist[c] for c in order]
    fig = go.Figure(go.Bar(
        x=vals, y=[c.capitalize() for c in order], orientation="h",
        marker=dict(color=[CLASS_COLORS[c] for c in order], line=dict(width=0), cornerradius=8),
        text=[f"{v * 100:.1f}%" for v in vals], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", color=INK),
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    _layout(fig, 260, title=dict(text="Class probabilities", font=dict(size=14, color=MUTED)))
    fig.update_xaxes(range=[0, 1.18], tickformat=".0%", showgrid=True)
    fig.update_yaxes(showgrid=False)
    return fig


def cam_surface(pred, res=72):
    cam = np.asarray(Image.fromarray((pred.cam * 255).astype(np.uint8)).resize((res, res), Image.BILINEAR)) / 255.0
    floor = np.asarray(Image.fromarray((pred.base * 255).astype(np.uint8)).convert("L").resize((res, res))) / 255.0
    xs = np.arange(res)
    fig = go.Figure([
        go.Surface(
            z=np.full_like(floor, -0.02), x=xs, y=xs, surfacecolor=floor,
            colorscale="gray", showscale=False, hoverinfo="skip", opacity=1.0,
        ),
        go.Surface(
            z=cam, x=xs, y=xs, surfacecolor=cam, colorscale=CAM_SCALE, cmin=0, cmax=1,
            opacity=0.88, showscale=False,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=False, width=1)),
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.35, roughness=0.5, fresnel=0.3),
            hovertemplate="attention %{z:.2f}<extra></extra>",
        ),
    ])
    axis = dict(visible=False, showbackground=False)
    fig.update_layout(
        scene=dict(
            xaxis=axis, yaxis=dict(axis, autorange="reversed"), zaxis=dict(axis, range=[-0.05, 1.4]),
            aspectratio=dict(x=1.25, y=1.25, z=0.7),
            camera=dict(eye=dict(x=0.95, y=-0.95, z=0.75), center=dict(x=0, y=0, z=-0.12)),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return _layout(fig, 320, margin=dict(l=0, r=0, t=0, b=0))


def compare_bars(preds, calibrated=True):
    fig = go.Figure()
    for p in preds:
        dist = p.dist(calibrated)
        fig.add_bar(
            name=MODELS[p.model]["label"], x=[c.capitalize() for c in CLASSES], y=[dist[c] for c in CLASSES],
            marker=dict(color=MODEL_COLORS[p.model], cornerradius=6),
            hovertemplate="%{x}: %{y:.2%}<extra>" + MODELS[p.model]["label"] + "</extra>",
        )
    _layout(fig, 340, barmode="group", bargap=0.25, bargroupgap=0.08,
            title=dict(text="Per-model class probabilities", font=dict(size=14, color=MUTED)))
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    return fig


def load_results():
    path = RESULTS_DIR / "final_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def benchmark_bars(df):
    fig = go.Figure()
    for metric, label, color in [("test_acc", "Accuracy", "#22d3ee"), ("test_macro_f1", "Macro-F1", "#a78bfa")]:
        fig.add_bar(
            name=label, x=[MODELS[m]["label"] for m in df.index], y=df[metric],
            marker=dict(color=color, cornerradius=6),
            text=[f"{v * 100:.2f}" for v in df[metric]], textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=11),
        )
    _layout(fig, 340, barmode="group", title=dict(text="Test accuracy & macro-F1 (%)", font=dict(size=14, color=MUTED)))
    fig.update_yaxes(range=[0.9, 0.98], tickformat=".0%")
    return fig


def recall_radar(df):
    fig = go.Figure()
    cats = [c.capitalize() for c in CLASSES]
    for m in df.index:
        r = [df.loc[m, f"recall_{c}"] for c in CLASSES]
        fig.add_trace(go.Scatterpolar(
            r=r + r[:1], theta=cats + cats[:1], name=MODELS[m]["label"], fill="toself",
            line=dict(color=MODEL_COLORS[m], width=2), opacity=0.75,
            hovertemplate="%{theta}: %{r:.2%}<extra>" + MODELS[m]["label"] + "</extra>",
        ))
    _layout(fig, 360, title=dict(text="Per-class recall", font=dict(size=14, color=MUTED)))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0.84, 1.0], gridcolor=GRID, color=MUTED, showticklabels=False, showline=False),
        angularaxis=dict(gridcolor=GRID, color=INK),
    ))
    return fig


def calibration_bars(df):
    fig = go.Figure()
    for col, label, color in [("ECE_before", "Before", "#64748b"), ("ECE_after_temp_scaling", "After temp. scaling", "#34d399")]:
        fig.add_bar(name=label, x=[MODELS[m]["label"] for m in df.index], y=df[col],
                    marker=dict(color=color, cornerradius=6),
                    hovertemplate="%{x}: %{y:.4f}<extra>" + label + "</extra>")
    _layout(fig, 320, barmode="group", title=dict(text="Expected Calibration Error (lower is better)", font=dict(size=14, color=MUTED)))
    return fig


def faithfulness_bars(df):
    vals = df["faithfulness_gap"]
    fig = go.Figure(go.Bar(
        x=[MODELS[m]["label"] for m in df.index], y=vals,
        marker=dict(color=[MODEL_COLORS[m] for m in df.index], cornerradius=6),
        text=[f"{v:+.3f}" for v in vals], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11),
    ))
    _layout(fig, 320, title=dict(text="Grad-CAM faithfulness gap (positive = CAM regions matter more than random)", font=dict(size=14, color=MUTED)))
    fig.update_yaxes(range=[-0.62, 0.22])
    return fig


def training_curves():
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Validation loss", "Validation macro-F1"), horizontal_spacing=0.08)
    found = False
    for m in MODELS:
        path = CKPT_DIR / m / "history.json"
        if not path.exists():
            continue
        found = True
        hist = json.loads(path.read_text())
        ep = [h["epoch"] for h in hist]
        style = dict(color=MODEL_COLORS[m], width=3 if m == "hybrid" else 2)
        fig.add_scatter(x=ep, y=[h["val_loss"] for h in hist], name=MODELS[m]["label"], mode="lines+markers",
                        line=style, marker=dict(size=6), legendgroup=m, row=1, col=1)
        fig.add_scatter(x=ep, y=[h["val_f1"] for h in hist], name=MODELS[m]["label"], mode="lines+markers",
                        line=style, marker=dict(size=6), legendgroup=m, showlegend=False, row=1, col=2)
    if not found:
        return None
    _layout(fig, 360, margin=dict(l=12, r=12, t=48, b=12))
    fig.update_annotations(font=dict(size=14, color=MUTED))
    fig.update_xaxes(title_text="epoch", title_font=dict(color=MUTED, size=12))
    return fig
