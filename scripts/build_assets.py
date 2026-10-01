"""One-shot builder: dataset -> features -> trained model -> visuals -> demo images.

Run from the project root:
    python3 scripts/build_assets.py
"""

import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agridoctor import CLASSES, CLASS_INFO  # noqa: E402
from agridoctor.dataset import build_dataset, generate_leaf  # noqa: E402
from agridoctor.features import extract_features  # noqa: E402
from agridoctor.model import featurize_paths, save_model, train  # noqa: E402

PER_CLASS = 120
SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    # 1. Dataset
    data_dir = ROOT / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    print("Generating synthetic dataset...")
    samples = build_dataset(data_dir, per_class=PER_CLASS, seed=SEED)
    paths = [s for s, _ in samples]
    labels = [l for _, l in samples]

    # 2. Features + train
    print("Extracting features...")
    X, y = featurize_paths(paths, labels)
    print(f"Feature matrix: {X.shape}")
    clf = train(X, y)
    print(f"Train accuracy: {clf.score(X, y):.1%}")
    save_model(clf)
    print("Model saved to models/model.pkl")

    # 3. Sample grid: healthy vs each disease (real generator output)
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    grid_rng = np.random.default_rng(2026)
    per_cls = {cls: [cv2.resize(generate_leaf(cls, grid_rng), (200, 200))
                     for _ in range(3)] for cls in CLASSES}
    grid = np.vstack([np.hstack([per_cls[cls][i] for cls in CLASSES])
                      for i in range(3)])
    # label each column strip
    labeled = np.zeros((grid.shape[0] + 40, grid.shape[1], 3), dtype=np.uint8) + 255
    labeled[40:] = grid
    for j, cls in enumerate(CLASSES):
        cv2.putText(labeled, cls, (j * 200 + 55, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 60, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(assets / "samples.png"), labeled)
    print("Wrote assets/samples.png")

    # 4. Confusion matrix on a fresh holdout (different seed)
    print("Evaluating holdout...")
    Xt, yt = [], []
    rng_test = np.random.default_rng(31337)
    for cls in CLASSES:
        for _ in range(30):
            Xt.append(extract_features(generate_leaf(cls, rng_test)))
            yt.append(cls)
    Xt = np.stack(Xt)
    pred = clf.predict(Xt)
    acc = (pred == np.array(yt)).mean()
    print(f"Holdout accuracy: {acc:.1%} (baseline: 85%)")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(yt, pred, labels=CLASSES),
                           display_labels=CLASSES).plot(ax=ax, cmap="Greens",
                                                       colorbar=False)
    ax.set_title(f"AgriDoctor holdout confusion matrix (acc {acc:.1%})")
    fig.tight_layout()
    fig.savefig(assets / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    print("Wrote assets/confusion_matrix.png")

    # 5. Bundled demo images (copies of real generated data)
    demo = ROOT / "demo_images"
    demo.mkdir(exist_ok=True)
    for f in demo.glob("*.png"):
        f.unlink()
    picks = {"healthy": 2, "leaf_spot": 1, "blight": 1, "mosaic": 1}
    for cls, count in picks.items():
        srcs = sorted((data_dir / cls).glob("*.png"))[:count]
        for i, src in enumerate(srcs, 1):
            shutil.copy(src, demo / f"demo_{cls}_{i:02d}.png")
    print(f"Wrote {len(list(demo.glob('*.png')))} demo images to demo_images/")

    print("Done.")


if __name__ == "__main__":
    main()
