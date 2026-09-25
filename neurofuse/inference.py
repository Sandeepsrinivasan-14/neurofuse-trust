import time
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .models import CLASSES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, RESULTS_DIR, build_model, cam_target
from .weights import ensure_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_cache = {}


def _temperatures():
    path = RESULTS_DIR / "calibration.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, index_col=0)
    return df["temperature"].to_dict()


TEMPERATURES = _temperatures()


def load(name):
    if name not in _cache:
        model = build_model(name)
        state = torch.load(ensure_weights(name), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        _cache[name] = model.to(DEVICE).eval()
    return _cache[name]


@dataclass
class Prediction:
    model: str
    probs: np.ndarray
    probs_calibrated: np.ndarray
    temperature: float
    cam: np.ndarray
    overlay: np.ndarray
    base: np.ndarray
    gate: float | None
    latency_ms: float

    @property
    def label(self):
        return CLASSES[int(self.probs.argmax())]

    def dist(self, calibrated=True):
        p = self.probs_calibrated if calibrated else self.probs
        return {c: float(v) for c, v in zip(CLASSES, p)}


def predict(image: Image.Image, name: str, with_cam: bool = True) -> Prediction:
    model = load(name)
    rgb = image.convert("RGB")
    x = eval_tf(rgb).unsqueeze(0).to(DEVICE)
    base = np.asarray(rgb.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x).float()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    latency = (time.perf_counter() - t0) * 1000

    temperature = float(TEMPERATURES.get(name, 1.0))
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    probs_cal = F.softmax(logits / temperature, dim=1)[0].cpu().numpy()
    gate = float(model.last_gate.mean()) if getattr(model, "last_gate", None) is not None else None

    cam = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    if with_cam:
        layers, reshape = cam_target(name, model)
        with GradCAM(model=model, target_layers=layers, reshape_transform=reshape) as gc:
            cam = gc(input_tensor=x, targets=[ClassifierOutputTarget(int(probs.argmax()))])[0]
    overlay = show_cam_on_image(base, cam, use_rgb=True, colormap=cv2.COLORMAP_TURBO, image_weight=0.5)

    return Prediction(name, probs, probs_cal, temperature, cam, overlay, base, gate, latency)
