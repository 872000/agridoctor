"""Command-line interface.

    python -m agridoctor.cli diagnose <image>
    python -m agridoctor.cli batch <image_dir> [--out report_dir]
    python -m agridoctor.cli train [--per-class N]   # regenerate data + retrain
"""

import argparse
import sys
from pathlib import Path

import cv2

from . import CLASSES, CLASS_INFO
from .model import MODEL_PATH, featurize_paths, load_model, save_model, train
from .treatments import get_treatment
from .annotate import annotate


def ensure_model():
    if not MODEL_PATH.exists():
        print("No trained model found — training on synthetic data first...", file=sys.stderr)
        train_command(per_class=120, seed=42)
    return load_model()


def diagnose_image(path):
    """Diagnose a single image. Returns dict with label, confidence, treatment."""
    clf, _ = ensure_model()
    classes = list(clf.classes_)  # model-native order (LabelEncoder sorts labels)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    from .features import extract_features
    X = extract_features(img).reshape(1, -1)
    proba = clf.predict_proba(X)[0]
    idx = int(proba.argmax())
    label = classes[idx]
    treatment = get_treatment(label)
    return {
        "image": str(path),
        "label": label,
        "description": CLASS_INFO[label],
        "confidence": float(proba[idx]),
        "probabilities": {c: float(p) for c, p in zip(classes, proba)},
        "treatment": treatment,
    }


def cmd_diagnose(args):
    result = diagnose_image(args.image)
    t = result["treatment"]
    print(f"Image      : {result['image']}")
    print(f"Diagnosis  : {result['label']} — {result['description']}")
    print(f"Confidence : {result['confidence']:.1%}")
    print()
    print("Class probabilities:")
    for c, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:10s} {p:.1%}")
    print()
    print(f"Treatment guidance — {t['name']}:")
    print(f"  {t['summary']}")
    print("  Organic:")
    for step in t["organic"]:
        print(f"    • {step}")
    print("  Chemical:")
    for step in t["chemical"]:
        print(f"    • {step}")
    print()
    print("Note: guidance is informational, not a substitute for a local agronomist.")


def cmd_batch(args):
    clf, _ = ensure_model()
    classes = list(clf.classes_)
    src = Path(args.dir)
    images = sorted(p for p in src.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        print(f"No images found in {src}", file=sys.stderr)
        sys.exit(1)
    out = Path(args.out)
    annotated_dir = out / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {c: 0 for c in classes}
    for p in images:
        result = diagnose_image(p)
        ann, n_boxes = annotate(cv2.imread(str(p)), result["label"], result["confidence"])
        ann_path = annotated_dir / f"annotated_{p.name}"
        cv2.imwrite(str(ann_path), ann)
        counts[result["label"]] += 1
        rows.append((p.name, result["label"], result["confidence"], n_boxes, ann_path.name))

    report = out / "report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# AgriDoctor batch diagnosis report\n\n")
        f.write(f"Images processed: {len(rows)}\n\n")
        f.write("## Summary\n\n")
        for c in classes:
            f.write(f"- {c}: {counts[c]}\n")
        f.write("\n## Results\n\n")
        f.write("| Image | Diagnosis | Confidence | Lesions boxed | Annotated |\n")
        f.write("|---|---|---|---|---|\n")
        for name, label, conf, n_boxes, ann in rows:
            f.write(f"| {name} | {label} | {conf:.1%} | {n_boxes} | annotated/{ann} |\n")
    print(f"Processed {len(rows)} images.")
    print(f"Report : {report}")
    print(f"Annotated images: {annotated_dir}")
    for c in classes:
        print(f"  {c:10s}: {counts[c]}")


def train_command(per_class=120, seed=42):
    from .dataset import build_dataset
    data_dir = Path(__file__).resolve().parent.parent / "data"
    print(f"Generating {per_class} synthetic images per class in {data_dir} ...")
    samples = build_dataset(data_dir, per_class=per_class, seed=seed)
    paths = [s for s, _ in samples]
    labels = [l for _, l in samples]
    print("Extracting features...")
    X, y = featurize_paths(paths, labels)
    print(f"Training SVM on {len(y)} samples ({X.shape[1]} features)...")
    clf = train(X, y)
    acc = clf.score(X, y)
    path = save_model(clf)
    print(f"Train accuracy: {acc:.1%}")
    print(f"Model saved to {path}")
    return clf


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agridoctor",
                                     description="AgriDoctor — crop disease detection from leaf photos")
    sub = parser.add_subparsers(dest="command", required=True)

    p_diag = sub.add_parser("diagnose", help="Diagnose a single leaf image")
    p_diag.add_argument("image", help="Path to a leaf photo (PNG/JPG)")
    p_diag.set_defaults(func=cmd_diagnose)

    p_batch = sub.add_parser("batch", help="Diagnose every image in a directory")
    p_batch.add_argument("dir", help="Directory of leaf photos")
    p_batch.add_argument("--out", default="batch_report", help="Output directory for report + annotated images")
    p_batch.set_defaults(func=cmd_batch)

    p_train = sub.add_parser("train", help="Regenerate synthetic data and retrain the model")
    p_train.add_argument("--per-class", type=int, default=120)
    p_train.set_defaults(func=lambda a: train_command(per_class=a.per_class))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
