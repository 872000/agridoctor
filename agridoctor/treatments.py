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
    """Return the treatment entry for a disease label.

    Raises:
        KeyError: if ``label`` is unknown, naming the valid labels.
    """
    data = load_treatments(path)
    try:
        return data[label]
    except KeyError:
        valid = ", ".join(sorted(data))
        raise KeyError(
            f"unknown disease label {label!r}; valid labels: {valid}"
        ) from None
