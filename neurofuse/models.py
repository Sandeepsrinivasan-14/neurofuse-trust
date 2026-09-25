from pathlib import Path

import torch
import torch.nn as nn
import timm

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"
RESULTS_DIR = ROOT / "outputs" / "results"

MODELS = {
    "hybrid": {
        "arch": "hybrid",
        "label": "NeuroFuse Hybrid",
        "blurb": "EfficientNet-B0 + Swin-Tiny, gated fusion",
    },
    "effnet": {
        "arch": "efficientnet_b0",
        "label": "EfficientNet-B0",
        "blurb": "CNN baseline",
    },
    "swin": {
        "arch": "swin_tiny_patch4_window7_224",
        "label": "Swin-Tiny",
        "blurb": "Transformer baseline",
    },
    "coatnet": {
        "arch": "coatnet_0_rw_224",
        "label": "CoAtNet-0",
        "blurb": "Off-the-shelf hybrid",
    },
}


class GatedFusionHybrid(nn.Module):
    def __init__(self, cnn_arch="efficientnet_b0", tf_arch="swin_tiny_patch4_window7_224",
                 proj_dim=256, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.cnn = timm.create_model(cnn_arch, pretrained=pretrained, num_classes=0)
        self.tf = timm.create_model(tf_arch, pretrained=pretrained, num_classes=0)
        self.proj_a = nn.Linear(self.cnn.num_features, proj_dim)
        self.proj_b = nn.Linear(self.tf.num_features, proj_dim)
        self.gate = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.head = nn.Sequential(nn.LayerNorm(proj_dim), nn.Linear(proj_dim, num_classes))
        self.last_gate = None

    def forward(self, x):
        fa = self.proj_a(self.cnn(x))
        fb = self.proj_b(self.tf(x))
        g = torch.sigmoid(self.gate(torch.cat([fa, fb], dim=1)))
        self.last_gate = g.detach()
        return self.head(g * fa + (1 - g) * fb)


def build_model(name):
    arch = MODELS[name]["arch"]
    if arch == "hybrid":
        return GatedFusionHybrid()
    return timm.create_model(arch, pretrained=False, num_classes=NUM_CLASSES)


def cam_target(name, model):
    if name == "effnet":
        return [model.conv_head], None
    if name == "hybrid":
        return [model.cnn.conv_head], None
    if name == "coatnet":
        return [model.stages[-1]], None
    if name == "swin":
        return [model.layers[-1].blocks[-1]], lambda t: t.permute(0, 3, 1, 2)
    raise ValueError(name)
