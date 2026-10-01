"""AgriDoctor — crop disease detection from leaf photos (UN SDG 2: Zero Hunger).

Classic computer vision + scikit-learn, fully offline. No API keys, no cloud.
"""

__version__ = "1.0.0"

CLASSES = ["healthy", "leaf_spot", "blight", "mosaic"]

CLASS_INFO = {
    "healthy": "Healthy leaf — no disease detected",
    "leaf_spot": "Leaf Spot (brown spot lesions)",
    "blight": "Blight (darkened leaf edges)",
    "mosaic": "Mosaic virus (yellowing / mottled pattern)",
}
