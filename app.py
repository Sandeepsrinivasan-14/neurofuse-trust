import argparse
from pathlib import Path

import gradio as gr
from PIL import Image

from neurofuse import charts
from neurofuse.inference import DEVICE, load, predict
from neurofuse.models import MODELS, ROOT
from neurofuse.ui import components as ui

EXAMPLES = sorted(str(p) for p in (ROOT / "assets" / "examples").glob("*.jpg"))
SAMPLE_CHOICES = [(Path(p).stem.replace("_", " #").replace("notumor", "no tumor").capitalize(), p) for p in EXAMPLES]
RESULTS = charts.load_results()
MODEL_CHOICES = [(f"{v['label']}  ·  {v['blurb']}", k) for k, v in MODELS.items()]

ABOUT = f"""
### How the hybrid works
Every scan passes through two pretrained backbones at the same time. **EfficientNet-B0** picks up local texture and
edges, and **Swin-Tiny** models long-range context with shifted-window attention. Each feature vector is projected to
256-d, and a small MLP learns a per-dimension **sigmoid gate**:

<code>g = σ(MLP([f_cnn ; f_tf]))</code> &nbsp;→&nbsp; <code>fused = g · f_cnn + (1 − g) · f_tf</code>

The gate is data-dependent, so every prediction reports how much it leaned on each branch.

### Trust toolkit
| Signal | What it tells you |
|---|---|
| **Grad-CAM** | Which regions pushed the logit of the predicted class (for the hybrid: the CNN branch) |
| **3D attention surface** | The same Grad-CAM map drawn as height over the scan |
| **Temperature scaling** | Confidence re-scaled with a temperature fit on the validation set |
| **Fusion gate** | The share of the decision that came from local (CNN) and from global (Transformer) features |
| **Consensus** | Whether four independently trained architectures agree |

### Data hygiene
Every image is SHA-256 hashed. Exact duplicates and train/test overlaps are removed before a stratified,
seeded (42) validation split is carved out, so none of the reported numbers are inflated by leakage.

### Limitations
Trained on a single public dataset of 2D slices. There is no external validation and no clinician-verified
localization. **{ui.DISCLAIMER}**
"""


def analyze(image, model_name, calibrated):
    if image is None:
        raise gr.Error("Upload an MRI slice first, or pick one of the examples.")
    pred = predict(image, model_name)
    return (
        ui.result_card(pred, calibrated),
        pred.overlay,
        charts.cam_surface(pred),
        charts.probability_bars(pred.dist(calibrated)),
    )


def compare(image, calibrated):
    if image is None:
        raise gr.Error("Upload an MRI slice first, or pick one of the examples.")
    preds = [predict(image, name) for name in MODELS]
    gallery = [(p.overlay, f"{MODELS[p.model]['label']} → {p.label}") for p in preds]
    return ui.consensus_card(preds, calibrated), charts.compare_bars(preds, calibrated), gallery


def build():
    with gr.Blocks(title="NeuroFuse-Trust · Brain MRI classifier") as demo:
        gr.HTML(ui.hero(RESULTS))

        with gr.Tabs():
            with gr.Tab("Diagnose", id="diagnose"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes="nf-panel"):
                        image = gr.Image(type="pil", label="MRI slice", height=300, sources=["upload", "clipboard"])
                        model = gr.Radio(MODEL_CHOICES, value="hybrid", label="Model")
                        calibrated = gr.Checkbox(value=True, label="Temperature-calibrated confidence")
                        run = gr.Button("Analyze scan", variant="primary", size="lg")
                        gr.Examples(EXAMPLES, inputs=image, label="Sample scans from the held-out test set", examples_per_page=8)
                    with gr.Column(scale=7):
                        card = gr.HTML(ui.placeholder())
                        with gr.Row():
                            overlay = gr.Image(label="Grad-CAM attention", interactive=False, height=320,
                                               elem_classes="nf-imgbox", buttons=["download", "fullscreen"])
                            surface = gr.Plot(charts.empty(320, "3D attention surface"), label="3D attention surface",
                                              elem_classes="nf-plot")
                        bars = gr.Plot(charts.empty(260, "Class probabilities"), label="Probabilities", elem_classes="nf-plot")
                outs = [card, overlay, surface, bars]
                run.click(analyze, [image, model, calibrated], outs)
                image.upload(analyze, [image, model, calibrated], outs)
                model.change(lambda i, m, c: analyze(i, m, c) if i is not None else gr.skip(), [image, model, calibrated], outs)
                calibrated.change(lambda i, m, c: analyze(i, m, c) if i is not None else gr.skip(), [image, model, calibrated], outs)

            with gr.Tab("Compare models", id="compare"):
                gr.HTML(ui.section("Four architectures, one scan",
                                   "Run every trained model on the same slice to check agreement and compare their attention maps."))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, elem_classes="nf-panel"):
                        c_image = gr.Image(type="pil", label="MRI slice", height=280, sources=["upload", "clipboard"])
                        c_cal = gr.Checkbox(value=True, label="Temperature-calibrated confidence")
                        c_run = gr.Button("Run all four models", variant="primary", size="lg")
                        c_sample = gr.Dropdown(SAMPLE_CHOICES, value=None, label="Or load a sample scan")
                    with gr.Column(scale=8):
                        c_card = gr.HTML(ui.placeholder("Run all four models to see the consensus"))
                        c_bars = gr.Plot(charts.empty(340, "Per-model probabilities"), label="Per-model probabilities", elem_classes="nf-plot")
                c_gallery = gr.Gallery(label="Grad-CAM by architecture", columns=4, rows=1, height=300, object_fit="contain")
                c_run.click(compare, [c_image, c_cal], [c_card, c_bars, c_gallery])
                c_sample.change(lambda path: Image.open(path) if path else gr.skip(), c_sample, c_image)
                image.change(lambda i: i if i is not None else gr.skip(), image, c_image)

            with gr.Tab("Benchmarks", id="bench"):
                gr.HTML(ui.section("Leaderboard", "Ranked by macro-F1 on the held-out test set."))
                gr.HTML(ui.leaderboard(RESULTS))
                gr.HTML(ui.section("Leakage-safe evaluation", "Dataset audit before any training happened."))
                gr.HTML(ui.leakage_cards())
                if RESULTS is not None:
                    gr.HTML(ui.section("Held-out test set", "2,358 images none of the models saw during training or model selection."))
                    with gr.Row():
                        gr.Plot(charts.benchmark_bars(RESULTS), elem_classes="nf-plot", show_label=False)
                        gr.Plot(charts.recall_radar(RESULTS), elem_classes="nf-plot", show_label=False)
                    with gr.Row():
                        gr.Plot(charts.calibration_bars(RESULTS), elem_classes="nf-plot", show_label=False)
                        gr.Plot(charts.faithfulness_bars(RESULTS), elem_classes="nf-plot", show_label=False)
                    curves = charts.training_curves()
                    if curves is not None:
                        gr.HTML(ui.section("Training dynamics", "Validation loss and macro-F1 per epoch. The best epoch by val macro-F1 was kept."))
                        gr.Plot(curves, elem_classes="nf-plot", show_label=False)

            with gr.Tab("About", id="about"):
                gr.HTML(ui.section("Architecture", "Two backbones, one learned gate."))
                gr.HTML(ui.architecture())
                gr.HTML(ui.section("Class atlas", "The four classes the model separates, with samples from the held-out test set."))
                gr.HTML(ui.class_atlas())
                gr.Markdown(ABOUT, elem_classes=["nf-panel", "nf-about"])

        gr.HTML(ui.footer())
    return demo


def main():
    parser = argparse.ArgumentParser(description="NeuroFuse-Trust demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    print(f"NeuroFuse-Trust running on {DEVICE}")
    for name in MODELS:
        load(name)
    print("Models loaded:", ", ".join(MODELS))
    build().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=ui.asset("style.css"),
        head=ui.asset("head.html"),
        js="() => { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); }",
        theme=ui.theme(),
        allowed_paths=[str(ROOT / "assets")],
        footer_links=["api"],
    )


if __name__ == "__main__":
    main()
