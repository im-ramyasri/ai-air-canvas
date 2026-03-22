"""
shape_recognizer.py — AI Shape Recognition

Uses geometric analysis on drawn strokes to classify shapes.
No heavy ML model needed — works with OpenCV contour math.

Shapes detected:
  Circle, Ellipse, Triangle, Rectangle, Square,
  Pentagon, Hexagon, Star, Arrow, Line, Zigzag, Free Form
"""

import cv2
import numpy as np
import math


# ─── Helpers ────────────────────────────────────────────────────────────────

def points_to_image(points, padding=20):
    """
    Renders a list of (x, y) points as a binary image (white on black).
    Returns (image, offset_x, offset_y) so callers can map back if needed.
    """
    pts = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)

    if w < 5:  w = 5
    if h < 5:  h = 5

    img = np.zeros((h + padding * 2, w + padding * 2), dtype=np.uint8)
    shifted = pts - np.array([x - padding, y - padding])

    for i in range(len(shifted) - 1):
        cv2.line(img, tuple(shifted[i]), tuple(shifted[i + 1]), 255, 3)

    # Close the shape slightly if endpoints are close
    start, end = shifted[0], shifted[-1]
    dist = np.linalg.norm(start.astype(float) - end.astype(float))
    if dist < max(w, h) * 0.25:
        cv2.line(img, tuple(start), tuple(end), 255, 3)

    return img, x - padding, y - padding


def circularity(area, perimeter):
    """1.0 = perfect circle. <0.7 = not circle."""
    if perimeter == 0:
        return 0
    return (4 * math.pi * area) / (perimeter ** 2)


def aspect_ratio(w, h):
    return w / h if h > 0 else 1.0


def stroke_direction_variance(points):
    """
    Measures how much the direction changes along the stroke.
    Low variance → straight line. High → curvy / complex.
    """
    if len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    deltas = pts[1:] - pts[:-1]
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    diffs  = np.abs(np.diff(np.unwrap(angles)))
    return float(np.mean(diffs))


def is_closed(points, threshold_ratio=0.20):
    """True if the stroke's endpoints are close together (shape is closed)."""
    pts = np.array(points, dtype=np.float32)
    _, _, w, h = cv2.boundingRect(pts.astype(np.int32))
    dist = np.linalg.norm(pts[0] - pts[-1])
    return dist < max(w, h) * threshold_ratio


def count_direction_reversals(points):
    """Count how many times the dominant direction reverses — helps detect zigzag."""
    if len(points) < 4:
        return 0
    pts   = np.array(points, dtype=np.float32)
    dx    = np.diff(pts[:, 0])
    signs = np.sign(dx[dx != 0])
    if len(signs) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(signs)) > 1))


def detect_arrow(approx, points):
    """Heuristic: arrow has ~7 vertices and a clear long-axis direction."""
    if len(approx) in (6, 7):
        pts  = np.array(points, dtype=np.float32)
        _, _, w, h = cv2.boundingRect(pts.astype(np.int32))
        ar   = aspect_ratio(w, h)
        if ar > 2.5 or ar < 0.4:
            return True
    return False


# ─── Main Recognizer ────────────────────────────────────────────────────────

def recognize_shape(points) -> str:
    """
    Classify a drawn stroke (list of (x,y) tuples) into a shape name.
    Returns a human-readable string.
    """
    if len(points) < 8:
        return ""

    pts = np.array(points, dtype=np.int32)
    _, _, w, h = cv2.boundingRect(pts)

    # Too tiny to classify
    if max(w, h) < 15:
        return "Dot"

    # ── Build binary image for contour analysis ──────────────────────────
    img, _, _ = points_to_image(points)

    # Dilate to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img    = cv2.dilate(img, kernel, iterations=2)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Unknown"

    contour   = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if area < 30:
        return "Line"

    circ     = circularity(area, perimeter)
    ar       = aspect_ratio(w, h)
    closed   = is_closed(points)
    var      = stroke_direction_variance(points)
    reversals = count_direction_reversals(points)

    # Polygon approximation at two tolerances
    eps_tight = 0.03 * perimeter
    eps_loose = 0.06 * perimeter
    approx_t  = cv2.approxPolyDP(contour, eps_tight,  True)
    approx_l  = cv2.approxPolyDP(contour, eps_loose,  True)
    verts_t   = len(approx_t)
    verts_l   = len(approx_l)

    # ── Convexity defects (star detection) ───────────────────────────────
    hull_idx    = cv2.convexHull(contour, returnPoints=False)
    is_star     = False
    if hull_idx is not None and len(hull_idx) > 3:
        try:
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is not None:
                deep = sum(1 for d in defects if d[0][3] / 256 > 15)
                if deep >= 4 and closed:
                    is_star = True
        except cv2.error:
            pass

    # ── Classification Logic ─────────────────────────────────────────────

    # Star
    if is_star:
        return "Star ⭐"

    # Zigzag (many direction reversals, low circularity, not closed)
    if reversals >= 5 and not closed and circ < 0.4:
        return "Zigzag"

    # Straight Line
    if (ar > 5 or ar < 0.2) and circ < 0.35 and not closed:
        return "Line"
    if var < 0.08 and not closed:
        return "Line"

    # Circle (high circularity + roughly equal w and h)
    if circ > 0.78 and 0.65 < ar < 1.55:
        return "Circle ⭕"

    # Ellipse (decent circularity but elongated)
    if circ > 0.60 and (ar > 1.6 or ar < 0.65):
        return "Ellipse"

    # Triangle
    if verts_l == 3:
        return "Triangle △"

    # Arrow
    if detect_arrow(approx_l, points):
        return "Arrow →"

    # Rectangle vs Square
    if verts_l == 4:
        x, y, rw, rh = cv2.boundingRect(approx_l)
        ratio = rw / rh if rh > 0 else 1
        if 0.80 <= ratio <= 1.25:
            return "Square □"
        else:
            return "Rectangle ▭"

    # Pentagon
    if verts_l == 5:
        return "Pentagon ⬠"

    # Hexagon
    if verts_l == 6:
        return "Hexagon ⬡"

    # 7+ vertices but high circularity → circle-ish
    if verts_l >= 7 and circ > 0.55:
        return "Circle ⭕"

    # Catch-all
    return "Free Form ✏"