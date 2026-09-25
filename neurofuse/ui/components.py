import base64
import json
from collections import Counter
from pathlib import Path

from ..charts import CLASS_COLORS, MODEL_COLORS
from ..models import MODELS, ROOT

UI_DIR = Path(__file__).resolve().parent
DISCLAIMER = "Research demo only — not for clinical use."

CLASS_INFO = {
    "glioma": "Glial-cell tumour, often infiltrative, typically intra-axial.",
    "meningioma": "Usually benign, extra-axial tumour arising from the meninges.",
    "notumor": "No tumour pattern detected by the model.",
    "pituitary": "Mass in the sellar region around the pituitary gland.",
}


def theme():
    import gradio as gr
    import inspect

    _THEME_KEYS = set(inspect.signature(gr.themes.Base.set).parameters)

    palette = dict(
        body_background_fill="#05070f",
        body_text_color="#e6ebff",
        body_text_color_subdued="#8b93b8",
        background_fill_primary="#0b0f22",
        background_fill_secondary="#0a0d1d",
        block_background_fill="rgba(0,0,0,0)",
        block_border_color="rgba(148,163,255,0.14)",
        block_label_background_fill="rgba(8,11,26,0.75)",
        block_label_text_color="#8b93b8",
        block_title_text_color="#8b93b8",
        border_color_primary="rgba(148,163,255,0.14)",
        border_color_accent="#22d3ee",
        input_background_fill="rgba(8,11,26,0.7)",
        input_border_color="rgba(148,163,255,0.14)",
        table_even_background_fill="rgba(255,255,255,0.02)",
        table_odd_background_fill="rgba(0,0,0,0)",
        table_border_color="rgba(148,163,255,0.14)",
        code_background_fill="rgba(34,211,238,0.1)",
        link_text_color="#67e8f9",
        color_accent_soft="rgba(34,211,238,0.15)",
        checkbox_background_color="rgba(8,11,26,0.7)",
        checkbox_background_color_selected="#22d3ee",
        checkbox_border_color="rgba(148,163,255,0.35)",
        loader_color="#22d3ee",
        shadow_drop="none",
    )
    palette.update({f"{k}_dark": v for k, v in list(palette.items())})
    palette = {k: v for k, v in palette.items() if k in _THEME_KEYS}
    return gr.themes.Base(
        primary_hue="cyan", neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(**palette)


def asset(name):
    return (UI_DIR / name).read_text(encoding="utf-8")


def _stat(value, label):
    return f'<div class="nf-stat"><span class="nf-stat-v">{value}</span><span class="nf-stat-l">{label}</span></div>'


def hero(results):
    if results is not None and "hybrid" in results.index:
        h = results.loc["hybrid"]
        stats = [
            _stat(f"{h.test_acc * 100:.1f}%", "test accuracy"),
            _stat(f"{h.test_macro_f1:.3f}", "macro-F1"),
            _stat(f"{h.ECE_after_temp_scaling:.3f}", "calibrated ECE"),
            _stat("4", "architectures"),
        ]
    else:
        stats = [_stat("4", "architectures")]
    return f"""
<section class="nf-hero">
  <div class="nf-hero-copy">
    <div class="nf-pill"><span class="nf-dot"></span>Trustworthy neuro-imaging AI</div>
    <h1 class="nf-title">NeuroFuse<span>-Trust</span></h1>
    <p class="nf-sub">A gated CNN&nbsp;+&nbsp;Transformer fusion network that classifies brain MRI into
    four classes, and shows its work with Grad-CAM attention maps, calibrated confidence
    and a live view of which branch it relied on.</p>
    <div class="nf-stats">{''.join(stats)}</div>
  </div>
  <div class="nf-hero-visual">
    <div id="nf-brain"></div>
    <div class="nf-orbit"></div>
    <div class="nf-float f1"><i style="background:#22d3ee"></i><span>CNN branch</span><b>local texture</b></div>
    <div class="nf-float f2"><i style="background:#a78bfa"></i><span>Swin branch</span><b>global context</b></div>
    <div class="nf-float f3"><i style="background:#f472b6"></i><span>Fusion gate</span><b>g = σ(·)</b></div>
  </div>
</section>
{features()}
<div class="nf-warning" role="note"><span>⚠</span><b>{DISCLAIMER}</b>
Outputs are model estimates on a public dataset and must never inform diagnosis or treatment.</div>
"""


def _ring(pct, color):
    return f"""<div class="nf-ring" style="--p:{pct:.1f};--c:{color}">
      <div class="nf-ring-in"><b>{pct:.1f}<small>%</small></b><span>confidence</span></div></div>"""


def _gate(g):
    cnn = g * 100
    return f"""
<div class="nf-gate">
  <div class="nf-gate-head"><span>Fusion gate</span><span class="nf-mono">g = {g:.3f}</span></div>
  <div class="nf-gate-bar"><div class="nf-gate-cnn" style="width:{cnn:.1f}%"></div><div class="nf-gate-tf" style="width:{100 - cnn:.1f}%"></div></div>
  <div class="nf-gate-legend"><span><i style="background:#22d3ee"></i>CNN local texture {cnn:.0f}%</span>
  <span><i style="background:#a78bfa"></i>Transformer global context {100 - cnn:.0f}%</span></div>
</div>"""


def placeholder(text="Upload an MRI slice and press Analyze"):
    return f"""<div class="nf-card nf-empty">
  <div class="nf-radar"><span></span><span></span><span></span><div class="nf-scan"></div></div>
  <p class="nf-empty-title">Awaiting scan</p><p>{text}</p>
  <div class="nf-steps"><span>① Upload</span><span>② Pick a model</span><span>③ Read the attention map</span></div>
</div>"""


def result_card(pred, calibrated=True):
    dist = pred.dist(calibrated)
    label = max(dist, key=dist.get)
    color = CLASS_COLORS[label]
    pct = dist[label] * 100
    runner = sorted(dist, key=dist.get, reverse=True)[1]
    margin = (dist[label] - dist[runner]) * 100
    info = MODELS[pred.model]
    gate = _gate(pred.gate) if pred.gate is not None else ""
    mode = f"temperature-scaled (T={pred.temperature:.2f})" if calibrated else "raw softmax"
    return f"""
<div class="nf-card nf-result nf-tilt" style="--accent:{color}">
  <div class="nf-result-top">
    {_ring(pct, color)}
    <div class="nf-result-text">
      <span class="nf-eyebrow">Predicted class</span>
      <h2 style="color:{color}">{label.capitalize()}</h2>
      <p>{CLASS_INFO[label]}</p>
      <div class="nf-chips">
        <span class="nf-chip">{info['label']}</span>
        <span class="nf-chip">{pred.latency_ms:.0f} ms</span>
        <span class="nf-chip">margin +{margin:.1f} pts vs {runner}</span>
      </div>
    </div>
  </div>
  {gate}
  <div class="nf-foot">Confidence: {mode} · {DISCLAIMER}</div>
</div>"""


def consensus_card(preds, calibrated=True):
    votes = Counter(p.label for p in preds)
    top, n = votes.most_common(1)[0]
    agree = n == len(preds)
    color = CLASS_COLORS[top]
    rows = []
    for p in preds:
        d = p.dist(calibrated)
        lab = max(d, key=d.get)
        rows.append(
            f'<div class="nf-row"><i style="background:{MODEL_COLORS[p.model]}"></i>'
            f'<span>{MODELS[p.model]["label"]}</span>'
            f'<b style="color:{CLASS_COLORS[lab]}">{lab}</b>'
            f'<span class="nf-mono">{d[lab] * 100:.1f}%</span></div>'
        )
    verdict = "Unanimous" if agree else f"{n} of {len(preds)} agree"
    return f"""
<div class="nf-card nf-consensus" style="--accent:{color}">
  <span class="nf-eyebrow">Ensemble consensus</span>
  <h2 style="color:{color}">{top.capitalize()}</h2>
  <div class="nf-badge {'ok' if agree else 'warn'}">{verdict}</div>
  <div class="nf-rows">{''.join(rows)}</div>
  <div class="nf-foot">{DISCLAIMER}</div>
</div>"""


def leakage_cards():
    path = ROOT / "data" / "splits" / "leakage_report.json"
    if not path.exists():
        return ""
    r = json.loads(path.read_text())
    s = r["sizes"]
    cards = [
        (f"{r['train_dups_dropped'] + r['test_dups_dropped']:,}", "exact duplicates removed"),
        (f"{r['train_test_overlap_removed']:,}", "train/test leaks removed"),
        (f"{s['train']:,}", "train images"),
        (f"{s['val']:,}", "validation images"),
        (f"{s['test']:,}", "held-out test images"),
    ]
    inner = "".join(f'<div class="nf-card nf-kpi"><b>{v}</b><span>{l}</span></div>' for v, l in cards)
    return f'<div class="nf-kpis">{inner}</div>'


def section(title, sub=""):
    return f'<div class="nf-section"><h3>{title}</h3><p>{sub}</p></div>'


def footer():
    return f"""
<footer class="nf-footer">
  <div><b>NeuroFuse-Trust</b> · Gated CNN+Transformer fusion for brain MRI</div>
  <div class="nf-footer-warn">{DISCLAIMER}</div>
</footer>"""


ICONS = {
    "shield": '<path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6l7-3z"/><path d="M9 12l2 2 4-4"/>',
    "eye": '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "gauge": '<path d="M12 14l4-4"/><path d="M3.3 17a9 9 0 1 1 17.4 0"/><circle cx="12" cy="14" r="1.5"/>',
    "merge": '<path d="M6 3v6a6 6 0 0 0 6 6h0a6 6 0 0 0 6-6V3"/><path d="M12 15v6"/>',
}


def _icon(name, color):
    return (f'<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


def features():
    items = [
        ("merge", "#22d3ee", "Gated fusion", "A learned gate blends CNN and Transformer features for each scan."),
        ("eye", "#a78bfa", "Explainable", "Grad-CAM heatmaps and a 3D attention surface for every prediction."),
        ("gauge", "#f472b6", "Calibrated", "Confidence is temperature-scaled on the validation set."),
        ("shield", "#34d399", "Leakage-safe", "SHA-256 dedup and a train/test overlap audit before any training."),
    ]
    cards = "".join(
        f'<div class="nf-feature nf-tilt" style="--accent:{c}"><div class="nf-feature-ic">{_icon(i, c)}</div>'
        f'<b>{t}</b><span>{d}</span></div>'
        for i, c, t, d in items
    )
    return f'<div class="nf-features">{cards}</div>'


def architecture():
    svg = (ROOT / "assets" / "architecture.svg").read_text(encoding="utf-8")
    return f'<div class="nf-arch">{svg}</div>'


def _thumb(path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def class_atlas():
    ex = ROOT / "assets" / "examples"
    cards = []
    for cls, info in CLASS_INFO.items():
        img = ex / f"{cls}_1.jpg"
        if not img.exists():
            continue
        color = CLASS_COLORS[cls]
        cards.append(
            f'<div class="nf-atlas-card nf-tilt" style="--accent:{color}">'
            f'<div class="nf-atlas-img"><img src="{_thumb(img)}" alt="{cls} sample"/></div>'
            f'<div class="nf-atlas-body"><b style="color:{color}">{"No tumor" if cls == "notumor" else cls.capitalize()}</b>'
            f'<span>{info}</span></div></div>'
        )
    return f'<div class="nf-atlas">{"".join(cards)}</div>'


def leaderboard(results):
    if results is None:
        return ""
    ranked = results.sort_values("test_macro_f1", ascending=False)
    rows = []
    for rank, (name, r) in enumerate(ranked.iterrows(), 1):
        color = MODEL_COLORS[name]
        crown = '<span class="nf-crown">★ best</span>' if rank == 1 else ""
        rows.append(f"""
<div class="nf-lb-card nf-tilt {'top' if rank == 1 else ''}" style="--accent:{color}">
  <div class="nf-lb-rank">#{rank}</div>
  <div class="nf-lb-name"><b>{MODELS[name]['label']}</b><span>{MODELS[name]['blurb']}</span>{crown}</div>
  <div class="nf-lb-metrics">
    <div><span>Accuracy</span><b>{r.test_acc * 100:.2f}%</b><i style="width:{(r.test_acc - 0.9) / 0.1 * 100:.0f}%"></i></div>
    <div><span>Macro-F1</span><b>{r.test_macro_f1:.4f}</b><i style="width:{(r.test_macro_f1 - 0.9) / 0.1 * 100:.0f}%"></i></div>
    <div><span>ECE (cal.)</span><b>{r.ECE_after_temp_scaling:.4f}</b><i style="width:{max(5, 100 - r.ECE_after_temp_scaling / 0.03 * 100):.0f}%"></i></div>
  </div>
</div>""")
    return f'<div class="nf-lb">{"".join(rows)}</div>'
