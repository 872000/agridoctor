"""Programmatic synthetic leaf-image dataset generation (OpenCV + NumPy).

Four classes:
    healthy    — clean green leaf with veins
    leaf_spot  — brown circular spots with yellow halos
    blight     — darkened / scorched leaf edges
    mosaic     — yellowing / mottled mosaic pattern

Everything is generated offline with seeded RNGs so the dataset is
reproducible.
"""

from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 256


def _leaf_geometry(rng):
    """Return (center, axes, angle) for a leaf-shaped ellipse."""
    cx = IMG_SIZE // 2 + int(rng.integers(-14, 15))
    cy = IMG_SIZE // 2 + int(rng.integers(-14, 15))
    ax = int(rng.integers(95, 115))
    ay = int(rng.integers(58, 72))
    angle = float(rng.integers(-25, 26))
    return (cx, cy), (ax, ay), angle


def _leaf_mask(center, axes, angle):
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    return mask


def _background(rng):
    """Soil/paper-like background with noise."""
    palettes = [
        ((60, 80, 120), (95, 120, 170)),    # soil brown (BGR)
        ((120, 140, 150), (170, 185, 190)),  # pale paper
        ((70, 110, 150), (110, 150, 190)),   # dry grass
    ]
    low, high = palettes[int(rng.integers(len(palettes)))]
    low = np.array(low, dtype=np.int16)
    high = np.array(high, dtype=np.int16)
    base = rng.integers(0, 2, size=(IMG_SIZE, IMG_SIZE, 1))
    img = np.where(base == 0, low, high).astype(np.int16)
    img = img + rng.integers(-12, 13, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _base_leaf(rng):
    """Draw a plain green leaf (veins included) on a background. Returns (img, mask)."""
    img = _background(rng)
    center, axes, angle = _leaf_geometry(rng)
    mask = _leaf_mask(center, axes, angle)

    # Green tone varies a little per leaf (H ~ 35-75 in OpenCV HSV)
    hue = int(rng.integers(35, 70))
    sat = int(rng.integers(120, 200))
    val = int(rng.integers(110, 170))
    leaf_hsv = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    leaf_hsv[:, :] = (hue, sat, val)
    leaf_bgr = cv2.cvtColor(leaf_hsv, cv2.COLOR_HSV2BGR)

    # Subtle radial shading for a 3-D-ish leaf
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    dist = np.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
    shade = 1.0 - 0.25 * np.clip(dist / (axes[0] + 1), 0, 1)
    leaf_bgr = (leaf_bgr.astype(np.float32) * shade[..., None]).astype(np.uint8)

    img = np.where(mask[..., None] == 255, leaf_bgr, img)

    # Veins: lighter green lines radiating from the midrib
    rad = np.deg2rad(angle)
    dx, dy = np.cos(rad), np.sin(rad)
    vein_col = (int(hue), int(sat * 0.5), min(255, int(val * 1.25)))
    vein_col = tuple(int(c) for c in cv2.cvtColor(
        np.uint8([[[vein_col[0], vein_col[1], vein_col[2]]]]), cv2.COLOR_HSV2BGR)[0, 0])
    x0, y0 = center
    cv2.line(img, (int(x0 - dx * axes[0]), int(y0 - dy * axes[0])),
             (int(x0 + dx * axes[0]), int(y0 + dy * axes[0])), vein_col, 3)
    for t in np.linspace(-0.75, 0.75, 7):
        px, py = x0 + dx * axes[0] * t, y0 + dy * axes[0] * t
        for sgn in (-1, 1):
            ex = int(px + (-dy) * sgn * axes[1] * 0.9)
            ey = int(py + (dx) * sgn * axes[1] * 0.9)
            cv2.line(img, (int(px), int(py)), (ex, ey), vein_col, 1)
    img = np.where(mask[..., None] == 255, img, img)  # no-op, keeps dtype clean

    # Fine sensor noise
    img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, size=img.shape), 0, 255).astype(np.uint8)
    return img, mask


def _apply_leaf_spot(img, mask, rng):
    """Brown circular lesions with yellow halos."""
    ys, xs = np.where(mask == 255)
    n = int(rng.integers(5, 14))
    for _ in range(n):
        i = int(rng.integers(len(xs)))
        cx, cy = int(xs[i]), int(ys[i])
        r = int(rng.integers(5, 13))
        # yellow halo
        cv2.circle(img, (cx, cy), r + 3, (40, 200, 220), -1)
        # brown ring
        cv2.circle(img, (cx, cy), r + 1, (30, 90, 150), -1)
        # dark center
        cv2.circle(img, (cx, cy), r, (20, 50, 100), -1)
    # re-apply mask so spots never bleed outside the leaf
    img = np.where(mask[..., None] == 255, img, img)
    return img


def _apply_blight(img, mask, rng):
    """Dark scorched edges: darken pixels close to the leaf boundary."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    edge_band = float(rng.uniform(10, 22))
    band = ((dist < edge_band) & (mask == 255)).astype(np.uint8) * 255
    # ragged edge: erode the band with noise so it looks organic
    noise = (rng.random((IMG_SIZE, IMG_SIZE)) > 0.35).astype(np.uint8) * 255
    band = cv2.bitwise_and(band, noise)
    dark = np.zeros_like(img)
    dark[:] = (25, 35, 60)  # scorched dark brown (BGR)
    alpha = (band.astype(np.float32) / 255.0 * 0.85)[..., None]
    img = (img.astype(np.float32) * (1 - alpha) + dark.astype(np.float32) * alpha).astype(np.uint8)
    return img


def _apply_mosaic(img, mask, rng):
    """Yellowing mottled patches scattered over the leaf."""
    n = int(rng.integers(10, 20))
    blotches = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for _ in range(n):
        cx = int(rng.integers(0, IMG_SIZE))
        cy = int(rng.integers(0, IMG_SIZE))
        r = int(rng.integers(18, 45))
        cv2.circle(blotches, (cx, cy), r, 1.0, -1)
    blotches = cv2.GaussianBlur(blotches, (31, 31), 0)
    blotches = (blotches * (mask == 255)).astype(np.float32)
    strength = float(rng.uniform(0.35, 0.6))
    alpha = np.clip(blotches * strength, 0, 0.75)[..., None]
    yellow = np.zeros_like(img)
    yellow[:] = (60, 220, 230)  # yellow-green (BGR)
    img = (img.astype(np.float32) * (1 - alpha) + yellow.astype(np.float32) * alpha).astype(np.uint8)
    return img


def generate_leaf(class_name, rng):
    """Generate one synthetic leaf image (BGR, IMG_SIZE x IMG_SIZE)."""
    img, mask = _base_leaf(rng)
    if class_name == "leaf_spot":
        img = _apply_leaf_spot(img, mask, rng)
    elif class_name == "blight":
        img = _apply_blight(img, mask, rng)
    elif class_name == "mosaic":
        img = _apply_mosaic(img, mask, rng)
    elif class_name != "healthy":
        raise ValueError(f"unknown class: {class_name}")
    return img


def build_dataset(out_dir, per_class=120, seed=42, img_size=IMG_SIZE):
    """Generate and save the full synthetic dataset. Returns list of (path, label)."""
    from . import CLASSES
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    samples = []
    for cls in CLASSES:
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = generate_leaf(cls, rng)
            if img.shape[0] != img_size:
                img = cv2.resize(img, (img_size, img_size))
            path = cls_dir / f"{cls}_{i:03d}.png"
            cv2.imwrite(str(path), img)
            samples.append((str(path), cls))
    return samples
