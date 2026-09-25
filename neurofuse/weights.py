import sys
import urllib.request

from .models import CKPT_DIR, MODELS

REPO = "Sandeepsrinivasan-14/neurofuse-trust"
TAG = "v1.0.0"
URL = "https://github.com/{repo}/releases/download/{tag}/{name}.pt"


def weight_path(name):
    return CKPT_DIR / name / "best.pt"


def ensure_weights(name, quiet=False):
    path = weight_path(name)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = URL.format(repo=REPO, tag=TAG, name=name)
    if not quiet:
        print(f"Downloading {name} weights from {url}")
    tmp = path.with_suffix(".part")

    def hook(blocks, block_size, total):
        if quiet or total <= 0:
            return
        done = min(blocks * block_size, total)
        sys.stdout.write(f"\r  {name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, hook)
    tmp.replace(path)
    if not quiet:
        print()
    return path


def main():
    names = sys.argv[1:] or list(MODELS)
    for name in names:
        ensure_weights(name)
    print("All weights ready in", CKPT_DIR)


if __name__ == "__main__":
    main()
