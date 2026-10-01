"""Draw diagnosis annotations: bounding boxes around diseased blobs + label banner."""

import cv2

from .features import diseased_regions, disease_component_boxes, segment_leaf


def annotate(img_bgr, label, confidence):
    """Return (annotated_copy, n_boxes) with diseased regions boxed."""
    out = img_bgr.copy()
    mask, _ = segment_leaf(out)
    _, _, disease = diseased_regions(out, mask)
    boxes = disease_component_boxes(disease)
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)

    banner = f"{label}  ({confidence:.0%})"
    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(out, banner, (10, th + 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out, len(boxes)
