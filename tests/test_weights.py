from neurofuse import weights


def test_weight_path_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(weights, "CKPT_DIR", tmp_path)
    assert weights.weight_path("hybrid") == tmp_path / "hybrid" / "best.pt"


def test_existing_weights_are_not_downloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(weights, "CKPT_DIR", tmp_path)
    existing = tmp_path / "swin" / "best.pt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"x")

    def boom(*a, **k):
        raise AssertionError("must not download when the file exists")

    monkeypatch.setattr(weights.urllib.request, "urlretrieve", boom)
    assert weights.ensure_weights("swin", quiet=True) == existing


def test_missing_weights_download_from_the_release_url(monkeypatch, tmp_path):
    monkeypatch.setattr(weights, "CKPT_DIR", tmp_path)
    seen = {}

    def fake_retrieve(url, dest, hook=None):
        seen["url"] = url
        dest.write_bytes(b"weights")

    monkeypatch.setattr(weights.urllib.request, "urlretrieve", fake_retrieve)
    path = weights.ensure_weights("effnet", quiet=True)
    assert path.read_bytes() == b"weights"
    assert seen["url"].endswith("/releases/download/v1.0.0/effnet.pt")
    assert not path.with_suffix(".part").exists()
