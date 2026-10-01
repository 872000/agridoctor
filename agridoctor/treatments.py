"""Treatment guidance lookup: disease label -> organic + chemical advice."""

import json
from pathlib import Path

TREATMENTS_PATH = Path(__file__).resolve().parent.parent / "treatments.json"

_cache = None


def load_treatments(path=TREATMENTS_PATH):
    global _cache
    if _cache is None or str(path) != str(TREATMENTS_PATH):
        with open(path, encoding="utf-8") as f:
            _cache = json.load(f)
    return _cache


def get_treatment(label, path=TREATMENTS_PATH):
    """Return the treatment entry for a disease label (KeyError if unknown)."""
    data = load_treatments(path)
    return data[label]
