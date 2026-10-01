"""SVM classifier (scikit-learn) trained on the synthetic leaf feature set.

Pipeline: StandardScaler -> SVC(RBF) wrapped in CalibratedClassifierCV.
`predict_proba` gives the confidence score shown to the farmer.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import CLASSES

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"


def make_classifier():
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)),
    ])
    return CalibratedClassifierCV(base, ensemble=False, cv=3)


def train(X, y):
    clf = make_classifier()
    clf.fit(X, y)
    return clf


def save_model(clf, path=MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classes": list(clf.classes_), "model": clf}, path)
    return path


def load_model(path=MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"], bundle["classes"]


def featurize_paths(paths, labels=None):
    """Extract features for a list of image paths (OpenCV BGR read)."""
    import cv2
    from .features import extract_features
    X = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"could not read image: {p}")
        X.append(extract_features(img))
    X = np.stack(X)
    if labels is not None:
        return X, np.array(labels)
    return X
