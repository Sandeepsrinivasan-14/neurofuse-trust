import pytest
import torch

from neurofuse.models import CLASSES, IMG_SIZE, MODELS, NUM_CLASSES, GatedFusionHybrid, build_model, cam_target


def test_class_metadata_is_consistent():
    assert NUM_CLASSES == len(CLASSES) == 4
    assert set(MODELS) == {"hybrid", "effnet", "swin", "coatnet"}
    for meta in MODELS.values():
        assert {"arch", "label", "blurb"} <= set(meta)


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_outputs_one_logit_per_class(name):
    model = build_model(name).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
    assert out.shape == (1, NUM_CLASSES)
    assert torch.isfinite(out).all()


def test_hybrid_gate_is_a_probability_per_feature():
    model = GatedFusionHybrid().eval()
    assert model.last_gate is None
    with torch.no_grad():
        model(torch.zeros(2, 3, IMG_SIZE, IMG_SIZE))
    gate = model.last_gate
    assert gate.shape == (2, 256)
    assert ((gate > 0) & (gate < 1)).all()


@pytest.mark.parametrize("name", list(MODELS))
def test_cam_target_returns_layers(name):
    layers, reshape = cam_target(name, build_model(name))
    assert len(layers) == 1
    assert (reshape is not None) == (name == "swin")  # only swin needs a token-grid reshape


def test_cam_target_rejects_unknown_model():
    with pytest.raises(ValueError):
        cam_target("resnet", object())
