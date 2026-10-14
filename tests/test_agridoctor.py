"""Real test suite for AgriDoctor.

Documented baseline: the SVM classifier must reach >= 85% accuracy on a
fresh synthetic holdout set (different seed from training data).
"""

import json

import cv2
import numpy as np
import pytest

from agridoctor import CLASSES, CLASS_INFO
from agridoctor.dataset import generate_leaf
from agridoctor.features import (
    N_FEATURES,
    diseased_regions,
    edge_density,
    extract_features,
    hsv_histogram,
    lbp_histogram,
    segment_leaf,
)
from agridoctor.model import featurize_paths, make_classifier, train
from agridoctor.treatments import get_treatment, load_treatments

ACCURACY_BASELINE = 0.85


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(7)


@pytest.fixture(scope="module")
def sample_images(rng):
    return {cls: generate_leaf(cls, rng) for cls in CLASSES}


# ---------- dataset ----------

def test_generate_leaf_shape_and_dtype(rng):
    for cls in CLASSES:
        img = generate_leaf(cls, rng)
        assert img.shape == (256, 256, 3)
        assert img.dtype == np.uint8


def test_classes_are_visually_distinct(rng):
    imgs = {cls: generate_leaf(cls, rng).astype(np.float32) for cls in CLASSES}
    for cls in ["leaf_spot", "blight", "mosaic"]:
        diff = np.abs(imgs[cls] - imgs["healthy"]).mean()
        assert diff > 3.0, f"{cls} too similar to healthy (mean diff {diff:.2f})"


# ---------- features ----------

def test_feature_vector_shape_and_finite(sample_images):
    for cls, img in sample_images.items():
        feats = extract_features(img)
        assert feats.shape == (N_FEATURES,), f"{cls}: {feats.shape}"
        assert np.all(np.isfinite(feats)), f"{cls}: non-finite features"


def test_hsv_histograms_normalized(sample_images):
    img = sample_images["healthy"]
    mask, _ = segment_leaf(img)
    hist = hsv_histogram(img, mask)
    assert hist.shape == (32,)
    # each of the 3 channel histograms sums to ~1
    assert abs(hist[:16].sum() - 1.0) < 1e-5
    assert abs(hist[16:24].sum() - 1.0) < 1e-5
    assert abs(hist[24:].sum() - 1.0) < 1e-5


def test_lbp_histogram_normalized(sample_images):
    img = sample_images["healthy"]
    mask, _ = segment_leaf(img)
    lbp = lbp_histogram(img, mask)
    assert lbp.shape == (16,)
    assert abs(lbp.sum() - 1.0) < 1e-5


def test_leaf_segmentation_finds_leaf():
    # Dedicated RNG: this test must not depend on shared fixture draw order.
    rng = np.random.default_rng(777)
    for cls in CLASSES:
        coverages = []
        for _ in range(8):
            img = generate_leaf(cls, rng)
            mask, contour = segment_leaf(img)
            assert contour is not None, f"{cls}: no leaf found"
            coverages.append((mask > 0).mean())
        # Never empty, never the whole frame; typically ~1/3 of the frame.
        assert min(coverages) > 0.02, f"{cls}: degenerate mask {min(coverages):.2f}"
        assert max(coverages) < 0.98, f"{cls}: mask covers everything"
        med = float(np.median(coverages))
        assert 0.10 < med < 0.80, f"{cls}: implausible median coverage {med:.2f}"


def test_edge_density_in_range(sample_images):
    for cls, img in sample_images.items():
        mask, _ = segment_leaf(img)
        e = edge_density(img, mask)
        assert 0.0 <= e <= 1.0, f"{cls}: {e}"


def test_leaf_spot_has_more_blobs_than_healthy():
    # Dedicated RNG + averaging: robust to any single random draw.
    rng = np.random.default_rng(20240501)
    h_blobs, s_blobs, s_ratios = [], [], []
    for _ in range(6):
        h_img = generate_leaf("healthy", rng)
        s_img = generate_leaf("leaf_spot", rng)
        hm, _ = segment_leaf(h_img)
        sm, _ = segment_leaf(s_img)
        hb, _, _ = diseased_regions(h_img, hm)
        sb, sr, _ = diseased_regions(s_img, sm)
        h_blobs.append(hb)
        s_blobs.append(sb)
        s_ratios.append(sr)
    assert np.mean(s_blobs) > np.mean(h_blobs)
    assert np.mean(h_blobs) == 0.0  # healthy leaves must be spotless
    assert np.mean(s_ratios) > 0.01


# ---------- classifier ----------

@pytest.fixture(scope="module")
def trained_classifier():
    rng_train = np.random.default_rng(1234)
    X, y = [], []
    for cls in CLASSES:
        for _ in range(40):
            X.append(extract_features(generate_leaf(cls, rng_train)))
            y.append(cls)
    clf = train(np.stack(X), np.array(y))
    return clf


def test_classifier_accuracy_on_fresh_holdout(trained_classifier):
    rng_test = np.random.default_rng(999)  # different seed than training
    X, y = [], []
    for cls in CLASSES:
        for _ in range(15):
            X.append(extract_features(generate_leaf(cls, rng_test)))
            y.append(cls)
    acc = trained_classifier.score(np.stack(X), np.array(y))
    assert acc >= ACCURACY_BASELINE, f"accuracy {acc:.1%} < baseline {ACCURACY_BASELINE:.0%}"


def test_predict_proba_confidence_sane(trained_classifier, sample_images):
    from agridoctor.features import extract_features
    for cls, img in sample_images.items():
        proba = trained_classifier.predict_proba(extract_features(img).reshape(1, -1))[0]
        assert abs(proba.sum() - 1.0) < 1e-5
        assert 0.0 <= proba.max() <= 1.0
        pred = trained_classifier.classes_[int(proba.argmax())]
        assert pred in CLASSES


def test_proba_argmax_agrees_with_predict(trained_classifier, sample_images):
    """Regression test: predict_proba column order must match the label mapping.

    (CalibratedClassifierCV sorts string labels alphabetically internally;
    the CLI must use clf.classes_, never an assumed order.)
    """
    from agridoctor.features import extract_features
    for cls, img in sample_images.items():
        X = extract_features(img).reshape(1, -1)
        proba = trained_classifier.predict_proba(X)[0]
        assert trained_classifier.predict(X)[0] == trained_classifier.classes_[int(proba.argmax())]


def test_featurize_paths_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    paths, labels = [], []
    for cls in CLASSES:
        p = tmp_path / f"{cls}.png"
        cv2.imwrite(str(p), generate_leaf(cls, rng))
        paths.append(str(p))
        labels.append(cls)
    X, y = featurize_paths(paths, labels)
    assert X.shape == (len(CLASSES), N_FEATURES)
    assert list(y) == labels


# ---------- treatments ----------

def test_treatment_lookup_complete_for_all_classes():
    data = load_treatments()
    for cls in CLASSES:
        assert cls in data, f"missing treatment entry for {cls}"
        entry = get_treatment(cls)
        assert entry["name"] and entry["summary"]
        assert isinstance(entry["organic"], list) and len(entry["organic"]) >= 2
        assert isinstance(entry["chemical"], list) and len(entry["chemical"]) >= 1
        assert cls in CLASS_INFO


def test_treatment_json_is_valid_json():
    from agridoctor.treatments import TREATMENTS_PATH
    with open(TREATMENTS_PATH, encoding="utf-8") as f:
        json.load(f)


def test_treatment_unknown_label_error_lists_valid_labels():
    with pytest.raises(KeyError) as excinfo:
        get_treatment("rust_typo")
    message = str(excinfo.value)
    assert "rust_typo" in message
    for cls in CLASSES:
        assert cls in message


# ---------- annotate + CLI ----------

def test_annotate_draws_boxes():
    from agridoctor.annotate import annotate
    rng = np.random.default_rng(31337)
    total_boxes = 0
    for _ in range(4):
        img = generate_leaf("leaf_spot", rng)
        out, n_boxes = annotate(img, "leaf_spot", 0.93)
        assert out.shape == img.shape
        total_boxes += n_boxes
    assert total_boxes >= 2, "expected lesion boxes on spotted leaves"


def test_cli_diagnose_end_to_end(tmp_path, trained_classifier, sample_images):
    from agridoctor import cli
    from agridoctor.model import save_model
    import agridoctor.model as model_mod

    p = tmp_path / "leaf.png"
    cv2.imwrite(str(p), sample_images["mosaic"])
    save_model(trained_classifier, tmp_path / "model.pkl")
    monkey_model = tmp_path / "model.pkl"
    orig = model_mod.MODEL_PATH
    model_mod.MODEL_PATH = monkey_model
    try:
        result = cli.diagnose_image(str(p))
    finally:
        model_mod.MODEL_PATH = orig
    assert result["label"] in CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["treatment"]["organic"]


def test_diagnose_image_missing_file_raises_clear_error(tmp_path):
    from agridoctor import cli
    with pytest.raises(FileNotFoundError, match="no such image file"):
        cli.diagnose_image(str(tmp_path / "does_not_exist.png"))


def test_diagnose_image_undecodable_file_raises_clear_error(tmp_path):
    from agridoctor import cli
    bad = tmp_path / "leaf.png"
    bad.write_bytes(b"this is not image data")
    with pytest.raises(ValueError, match="could not decode image"):
        cli.diagnose_image(str(bad))
