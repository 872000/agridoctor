"""Classic computer-vision feature extraction for leaf disease detection.

Feature vector (51 dimensions):
  - HSV color histogram: 16 H + 8 S + 8 V bins over the leaf region (32)
  - Edge density: Canny edge pixels / leaf pixels (1)
  - Texture: simple 8-neighbor LBP codes quantized to 16 bins (16)
  - Diseased-region analysis via contour detection: pixels far from the
      leaf's median HSV color are grouped into blobs
      (blob density + diseased area ratio) (2)
"""

import cv2
import numpy as np

N_FEATURES = 51
_LBP_BINS = 16


def segment_leaf(img_bgr):
    """Segment the leaf from the background.

    The background is estimated from the image corners (the leaf is always
    roughly centered); pixels far from the background color belong to the
    leaf. This keeps diseased tissue (spots, scorch, yellowing) inside the
    mask, unlike a pure green-range threshold.

    Returns (binary_mask, largest filled contour or None).
    """
    h, w = img_bgr.shape[:2]
    corners = np.concatenate([
        img_bgr[0:10, 0:10].reshape(-1, 3),
        img_bgr[0:10, w - 10:w].reshape(-1, 3),
        img_bgr[h - 10:h, 0:10].reshape(-1, 3),
        img_bgr[h - 10:h, w - 10:w].reshape(-1, 3),
    ]).astype(np.float32)
    bg_color = np.median(corners, axis=0)
    dist = np.sqrt(((img_bgr.astype(np.float32) - bg_color) ** 2).sum(axis=2))
    mask = (dist > 45).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(mask.shape, dtype=np.uint8), None
    biggest = max(contours, key=cv2.contourArea)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, [biggest], -1, 255, -1)
    return filled, biggest


def hsv_histogram(img_bgr, mask, h_bins=16, s_bins=8, v_bins=8):
    """Normalized HSV histograms restricted to the leaf mask."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch, bins, rng in ((0, h_bins, [0, 180]), (1, s_bins, [0, 256]), (2, v_bins, [0, 256])):
        hist = cv2.calcHist([hsv], [ch], mask, [bins], rng).ravel()
        total = hist.sum()
        feats.append(hist / total if total > 0 else np.zeros(bins))
    return np.concatenate(feats).astype(np.float32)


def edge_density(img_bgr, mask):
    """Fraction of leaf pixels that are Canny edges (texture/roughness cue)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    leaf_px = max(int((mask > 0).sum()), 1)
    edge_px = int(((edges > 0) & (mask > 0)).sum())
    return edge_px / leaf_px


def lbp_histogram(img_bgr, mask, n_bins=_LBP_BINS):
    """Simple 8-neighbor LBP: each pixel's code is quantized to n_bins."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    padded = np.pad(gray, 1, mode="edge")
    h, w = gray.shape
    code = np.zeros((h, w), dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neigh = padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
        code |= (((neigh >= gray).astype(np.uint8)) << i)
    codes = (code >> (8 - 4)).ravel()  # 256 codes -> 16 bins
    leaf = (mask > 0).ravel()
    hist, _ = np.histogram(codes[leaf], bins=n_bins, range=(0, n_bins))
    total = hist.sum()
    return (hist / total if total > 0 else np.zeros(n_bins)).astype(np.float32)


def diseased_regions(img_bgr, mask, min_area=30, threshold=0.22):
    """Find diseased blobs inside the leaf via connected-component analysis.

    The leaf's median HSV color (after a median blur that erases thin veins
    and sensor noise) is the "healthy reference". Pixels whose HSV distance
    from that reference exceeds `threshold` are diseased; 8-connected
    components of that map (filtered by `min_area`) are the blobs. Pixel
    counting (not contour polygon area) is used so ring-shaped regions such
    as blight bands are measured correctly.

    Returns (blob_density, diseased_area_ratio, binary_disease_mask).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    smooth = cv2.medianBlur(hsv, 15).astype(np.float32)
    leaf_px = smooth[mask > 0].reshape(-1, 3)
    if len(leaf_px) == 0:
        return 0.0, 0.0, np.zeros(mask.shape, dtype=np.uint8)
    ref = np.median(leaf_px, axis=0)
    d_hue = np.abs(smooth[:, :, 0] - ref[0])
    d_hue = np.minimum(d_hue, 180.0 - d_hue) / 90.0
    d_sat = np.abs(smooth[:, :, 1] - ref[1]) / 255.0
    d_val = np.abs(smooth[:, :, 2] - ref[2]) / 255.0
    dist = np.sqrt(d_hue ** 2 + d_sat ** 2 + d_val ** 2)
    # Erode the evaluation region: median blur smears background colors into
    # the leaf boundary, which would otherwise read as false disease.
    inner = cv2.erode(mask, np.ones((7, 7), np.uint8))
    disease = ((dist > threshold) & (inner > 0)).astype(np.uint8)
    disease = cv2.morphologyEx(disease, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(disease, connectivity=8)
    # label 0 is the background; keep components with >= min_area pixels
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)
             if stats[i, cv2.CC_STAT_AREA] >= min_area]
    leaf_area = max(int((mask > 0).sum()), 1)
    blob_density = len(areas) / (leaf_area / 10000.0)
    area_ratio = float(sum(areas)) / leaf_area
    return blob_density, area_ratio, (disease * 255).astype(np.uint8)


def disease_component_boxes(disease_mask, min_area=30):
    """Bounding boxes (x, y, w, h) of diseased connected components."""
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        (disease_mask > 0).astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            boxes.append((int(stats[i, cv2.CC_STAT_LEFT]),
                          int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]),
                          int(stats[i, cv2.CC_STAT_HEIGHT])))
    return boxes


def extract_features(img_bgr):
    """Full 51-dim feature vector for one BGR image."""
    mask, _ = segment_leaf(img_bgr)
    hist = hsv_histogram(img_bgr, mask)
    edge = edge_density(img_bgr, mask)
    lbp = lbp_histogram(img_bgr, mask)
    blob_density, area_ratio, _ = diseased_regions(img_bgr, mask)
    feats = np.concatenate([hist, [edge], lbp, [blob_density, area_ratio]]).astype(np.float32)
    assert feats.shape[0] == N_FEATURES, f"expected {N_FEATURES}, got {feats.shape[0]}"
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
