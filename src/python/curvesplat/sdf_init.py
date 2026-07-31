"""
SDF-based curve initialization for CurveSplat.

Implements SPEC.md §2 Stage 1: given a raster image, automatically extract
contours and fit cubic Bezier curves to them.

Pipeline:
    1. Edge detection (multi-scale Sobel + non-max suppression + hysteresis)
    2. Otsu thresholding (auto-binarize if needed)
    3. JFA-based signed distance field
    4. Marching squares contour extraction
    5. Douglas-Peucker simplification
    6. Corner detection (angle-based)
    7. Least-squares cubic Bezier fitting
    8. initialize_curves_from_image() — ties it all together

All pure NumPy — no OpenCV dependency. Suitable for Phase 2 validation.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch

from curvesplat.scene import Curve, Scene


# ---------------------------------------------------------------------------
# 1. Edge detection: multi-scale Sobel + NMS + hysteresis
# ---------------------------------------------------------------------------

def _gaussian_blur_1d(signal: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian blur via convolution with a truncated kernel."""
    radius = int(math.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(signal, kernel, mode="same")


def _gaussian_blur_2d(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 2D Gaussian blur."""
    if sigma <= 0:
        return image.copy()
    # Blur rows
    blurred = np.apply_along_axis(_gaussian_blur_1d, 1, image, sigma)
    # Blur columns
    blurred = np.apply_along_axis(_gaussian_blur_1d, 0, blurred, sigma)
    return blurred


# (Deleted dead _sobel_gradients to remove scipy dependency)


def _sobel_gradients_pure(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Sobel gradients without scipy — pure numpy with padding."""
    H, W = image.shape
    padded = np.pad(image, 1, mode="reflect")

    # Sobel x: [[-1,0,1],[-2,0,2],[-1,0,1]]
    gx = (
        -padded[0:H, 0:W] + padded[0:H, 2:W+2]
        - 2*padded[1:H+1, 0:W] + 2*padded[1:H+1, 2:W+2]
        - padded[2:H+2, 0:W] + padded[2:H+2, 2:W+2]
    )

    # Sobel y: [[-1,-2,-1],[0,0,0],[1,2,1]]
    gy = (
        -padded[0:H, 0:W] - 2*padded[0:H, 1:W+1] - padded[0:H, 2:W+2]
        + padded[2:H+2, 0:W] + 2*padded[2:H+2, 1:W+1] + padded[2:H+2, 2:W+2]
    )

    return gx.astype(np.float64), gy.astype(np.float64)


def _non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Thin edges to 1px width by suppressing non-maxima along gradient direction.

    Uses 4-direction quantization (0°, 45°, 90°, 135°).
    """
    H, W = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    # Quantize direction to 4 bins (0, 1, 2, 3) → (0°, 45°, 90°, 135°)
    angle = direction % np.pi  # Map to [0, π)
    angle_idx = np.round(angle / (np.pi / 4)).astype(int) % 4

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            q = angle_idx[r, c]
            mag = magnitude[r, c]

            # Compare with neighbors along gradient direction
            if q == 0:    # 0° — compare left/right
                n1, n2 = magnitude[r, c - 1], magnitude[r, c + 1]
            elif q == 1:  # 45° — compare upper-right/lower-left
                n1, n2 = magnitude[r - 1, c + 1], magnitude[r + 1, c - 1]
            elif q == 2:  # 90° — compare up/down
                n1, n2 = magnitude[r - 1, c], magnitude[r + 1, c]
            else:         # 135° — compare upper-left/lower-right
                n1, n2 = magnitude[r - 1, c - 1], magnitude[r + 1, c + 1]

            if mag >= n1 and mag >= n2:
                suppressed[r, c] = mag

    return suppressed


def _hysteresis_threshold(
    thin_edges: np.ndarray,
    low_ratio: float = 0.1,
    high_ratio: float = 0.3,
) -> np.ndarray:
    """
    Apply double-threshold hysteresis to thin edges.

    Strong edges (above high threshold) are kept.
    Weak edges (between low and high) are kept only if connected to strong edges.
    """
    max_val = thin_edges.max()
    if max_val <= 0:
        return np.zeros_like(thin_edges, dtype=bool)

    high_thresh = max_val * high_ratio
    low_thresh = max_val * low_ratio

    strong = thin_edges >= high_thresh
    weak = (thin_edges >= low_thresh) & ~strong

    # Grow strong edges into weak edges via flood fill
    edges = strong.copy()
    changed = True
    while changed:
        changed = False
        # Dilate edges by 1px and intersect with weak
        dilated = np.zeros_like(edges)
        dilated[1:, :] |= edges[:-1, :]
        dilated[:-1, :] |= edges[1:, :]
        dilated[:, 1:] |= edges[:, :-1]
        dilated[:, :-1] |= edges[:, 1:]
        dilated[1:, 1:] |= edges[:-1, :-1]
        dilated[1:, :-1] |= edges[:-1, 1:]
        dilated[:-1, 1:] |= edges[1:, :-1]
        dilated[:-1, :-1] |= edges[1:, 1:]

        new_edges = dilated & weak & ~edges
        if new_edges.any():
            edges |= new_edges
            changed = True

    return edges


def detect_edges(
    image: np.ndarray,
    sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
    low_ratio: float = 0.1,
    high_ratio: float = 0.3,
) -> np.ndarray:
    """
    Multi-scale Sobel edge detection with Canny-style NMS and hysteresis.

    Args:
        image: Grayscale image (H, W), float in [0, 1].
        sigmas: Gaussian blur scales for multi-scale Sobel.
        low_ratio: Low hysteresis threshold as fraction of max gradient.
        high_ratio: High hysteresis threshold as fraction of max gradient.

    Returns:
        edges: Binary edge map (H, W), bool.
    """
    image = image.astype(np.float64)

    # Multi-scale Sobel: take max gradient magnitude across scales
    all_magnitudes = []
    all_directions = []

    for sigma in sigmas:
        blurred = _gaussian_blur_2d(image, sigma)
        gx, gy = _sobel_gradients_pure(blurred)
        mag = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        all_magnitudes.append(mag)
        all_directions.append(direction)

    # Take scale with max gradient at each pixel
    magnitudes_stack = np.stack(all_magnitudes, axis=0)
    best_scale = magnitudes_stack.argmax(axis=0)

    H, W = image.shape
    magnitude = np.zeros((H, W), dtype=np.float64)
    direction = np.zeros((H, W), dtype=np.float64)
    for i in range(len(sigmas)):
        mask = best_scale == i
        magnitude[mask] = all_magnitudes[i][mask]
        direction[mask] = all_directions[i][mask]

    # Non-maximum suppression (thin edges)
    thin = _non_maximum_suppression(magnitude, direction)

    # Hysteresis thresholding
    edges = _hysteresis_threshold(thin, low_ratio, high_ratio)

    return edges


# ---------------------------------------------------------------------------
# 2. Otsu thresholding
# ---------------------------------------------------------------------------

def otsu_threshold(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Otsu's automatic thresholding.

    Finds the threshold that maximizes between-class variance.

    Args:
        image: Grayscale image (H, W), values in [0, 1].

    Returns:
        binary: Binary mask (H, W), bool.
        threshold: The computed threshold value.
    """
    # Quantize to 256 bins
    quantized = (image * 255).clip(0, 255).astype(np.uint8)
    histogram = np.bincount(quantized.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()

    if total == 0:
        return np.zeros_like(image, dtype=bool), 0.5

    # Otsu's method
    best_thresh = 0
    best_variance = 0.0
    sum_total = np.sum(np.arange(256) * histogram)
    sum_bg = 0.0
    weight_bg = 0.0

    for t in range(256):
        weight_bg += histogram[t]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * histogram[t]

        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > best_variance:
            best_variance = variance
            best_thresh = t

    threshold = best_thresh / 255.0
    binary = image > threshold

    return binary, threshold


# ---------------------------------------------------------------------------
# 3. Jump Flood Algorithm — Signed Distance Field
# ---------------------------------------------------------------------------

def compute_sdf_jfa(
    edges: np.ndarray,
    interior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute a signed distance field using the Jump Flood Algorithm.

    JFA efficiently computes approximate nearest-seed Voronoi diagrams,
    giving us the distance from each pixel to the nearest edge pixel.

    Args:
        edges: Binary edge map (H, W), bool — seed pixels.
        interior: Binary interior mask (H, W), bool.
                  If None, assumes all non-edge pixels are exterior.
                  Used to assign sign (negative = inside).

    Returns:
        sdf: Signed distance field (H, W), float.
             Negative inside, positive outside, zero at edges.
    """
    H, W = edges.shape

    # Nearest seed coordinates for each pixel: (H, W, 2)
    # Initialize seeds at edge pixels; non-edge pixels get (-1, -1)
    nearest = np.full((H, W, 2), -1, dtype=np.int32)
    edge_ys, edge_xs = np.where(edges)
    nearest[edge_ys, edge_xs, 0] = edge_ys
    nearest[edge_ys, edge_xs, 1] = edge_xs

    # JFA iterations: step sizes H/2, H/4, ..., 1
    max_dim = max(H, W)
    step = 1
    while step < max_dim:
        step *= 2
    step //= 2

    while step >= 1:
        # For each of the 8 neighbors + self at distance 'step'
        for dy in [-step, 0, step]:
            for dx in [-step, 0, step]:
                if dy == 0 and dx == 0:
                    continue

                # Shifted coordinates
                src_y = np.clip(np.arange(H) + dy, 0, H - 1)
                src_x = np.clip(np.arange(W) + dx, 0, W - 1)

                # Get neighbor's nearest seed: use advanced indexing
                neighbor_nearest = nearest[np.ix_(src_y, src_x)]  # (H, W, 2)

                # Compute distances to neighbor's seed vs current seed
                yy = np.arange(H).reshape(-1, 1)
                xx = np.arange(W).reshape(1, -1)

                # Current seed distance
                has_current = nearest[:, :, 0] >= 0
                curr_dist = np.full((H, W), np.inf)
                curr_dist[has_current] = np.sqrt(
                    (yy - nearest[:, :, 0])[has_current].astype(np.float64)**2
                    + (xx - nearest[:, :, 1])[has_current].astype(np.float64)**2
                )

                # Neighbor seed distance
                has_neighbor = neighbor_nearest[:, :, 0] >= 0
                neigh_dist = np.full((H, W), np.inf)
                neigh_dist[has_neighbor] = np.sqrt(
                    (yy - neighbor_nearest[:, :, 0])[has_neighbor].astype(np.float64)**2
                    + (xx - neighbor_nearest[:, :, 1])[has_neighbor].astype(np.float64)**2
                )

                # Update where neighbor is closer
                update = neigh_dist < curr_dist
                nearest[update] = neighbor_nearest[update]

        step //= 2

    # Compute final distances
    yy = np.arange(H).reshape(-1, 1).astype(np.float64)
    xx = np.arange(W).reshape(1, -1).astype(np.float64)
    dist = np.sqrt(
        (yy - nearest[:, :, 0].astype(np.float64))**2
        + (xx - nearest[:, :, 1].astype(np.float64))**2
    )

    # Apply sign: negative inside, positive outside
    if interior is not None:
        sdf = np.where(interior, -dist, dist)
    else:
        sdf = dist

    # Zero at edge pixels
    sdf[edges] = 0.0

    return sdf


# ---------------------------------------------------------------------------
# 4. Marching squares contour extraction
# ---------------------------------------------------------------------------

# Marching squares lookup table: for each of the 16 cases, the edges to connect.
# Each edge is identified by an index 0-3:
#   0: top edge (between corners TL, TR)
#   1: right edge (between corners TR, BR)
#   2: bottom edge (between corners BL, BR)
#   3: left edge (between corners TL, BL)
# Each entry is a list of (edge_a, edge_b) pairs to connect.
_MS_TABLE = {
    0:  [],
    1:  [(2, 3)],
    2:  [(1, 2)],
    3:  [(1, 3)],
    4:  [(0, 1)],
    5:  [(0, 3), (1, 2)],  # saddle — ambiguous
    6:  [(0, 2)],
    7:  [(0, 3)],
    8:  [(0, 3)],
    9:  [(0, 2)],
    10: [(0, 1), (2, 3)],  # saddle — ambiguous
    11: [(0, 1)],
    12: [(1, 3)],
    13: [(1, 2)],
    14: [(2, 3)],
    15: [],
}


def _interpolate_edge(
    r: int, c: int,
    edge_id: int,
    sdf: np.ndarray,
    iso: float,
) -> Tuple[float, float]:
    """
    Interpolate the iso-crossing position along a cell edge.

    Returns (x, y) coordinates.
    """
    # Corner values: TL, TR, BR, BL
    tl = sdf[r, c]
    tr = sdf[r, c + 1]
    br = sdf[r + 1, c + 1]
    bl = sdf[r + 1, c]

    if edge_id == 0:  # top: TL → TR
        t = (iso - tl) / (tr - tl + 1e-12)
        return (c + t, r)
    elif edge_id == 1:  # right: TR → BR
        t = (iso - tr) / (br - tr + 1e-12)
        return (c + 1, r + t)
    elif edge_id == 2:  # bottom: BL → BR
        t = (iso - bl) / (br - bl + 1e-12)
        return (c + t, r + 1)
    else:  # edge_id == 3, left: TL → BL
        t = (iso - tl) / (bl - tl + 1e-12)
        return (c, r + t)


def extract_contours(
    sdf: np.ndarray,
    iso: float = 0.0,
    min_length: int = 5,
) -> List[np.ndarray]:
    """
    Extract iso-contours from a signed distance field using marching squares.

    Args:
        sdf: Signed distance field (H, W), float.
        iso: Iso-level to extract (default 0.0 = boundary).
        min_length: Minimum contour length in points; shorter ones are filtered.

    Returns:
        contours: List of contour arrays, each shape (M, 2) with (x, y) coords.
    """
    H, W = sdf.shape

    # Collect all line segments from marching squares
    segments = []  # List of ((x1,y1), (x2,y2))

    for r in range(H - 1):
        for c in range(W - 1):
            # Cell corners: TL(r,c), TR(r,c+1), BR(r+1,c+1), BL(r+1,c)
            tl = sdf[r, c] >= iso
            tr = sdf[r, c + 1] >= iso
            br = sdf[r + 1, c + 1] >= iso
            bl = sdf[r + 1, c] >= iso

            # Case index (4-bit)
            case = (int(tl) << 3) | (int(tr) << 2) | (int(br) << 1) | int(bl)

            # Handle saddle cases using center average
            if case == 5 or case == 10:
                center = (sdf[r, c] + sdf[r, c+1] + sdf[r+1, c] + sdf[r+1, c+1]) / 4.0
                if case == 5:
                    if center >= iso:
                        edges_to_draw = [(0, 1), (2, 3)]
                    else:
                        edges_to_draw = [(0, 3), (1, 2)]
                else:  # case 10
                    if center >= iso:
                        edges_to_draw = [(0, 3), (1, 2)]
                    else:
                        edges_to_draw = [(0, 1), (2, 3)]
            else:
                edges_to_draw = _MS_TABLE[case]

            for ea, eb in edges_to_draw:
                p1 = _interpolate_edge(r, c, ea, sdf, iso)
                p2 = _interpolate_edge(r, c, eb, sdf, iso)
                segments.append((p1, p2))

    if not segments:
        return []

    # Chain segments into contours
    return _chain_segments(segments, min_length)


def _chain_segments(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    min_length: int,
    tolerance: float = 0.01,
) -> List[np.ndarray]:
    """
    Chain individual line segments into ordered polyline contours.

    Segments sharing an endpoint (within tolerance) are connected.
    """
    if not segments:
        return []

    # Build adjacency: map rounded endpoints → segment indices
    endpoint_map: dict = {}
    seg_arr = np.array(segments)  # (N, 2, 2)

    def _key(pt: Tuple[float, float]) -> Tuple[int, int]:
        return (round(pt[0] * 100), round(pt[1] * 100))

    # For each segment, register both endpoints
    used = [False] * len(segments)
    adj: dict = {}  # seg_idx → list of connected seg_idx via shared endpoint

    for i, (p1, p2) in enumerate(segments):
        k1 = _key(p1)
        k2 = _key(p2)
        if k1 not in endpoint_map:
            endpoint_map[k1] = []
        endpoint_map[k1].append((i, 0))  # segment i, endpoint 0
        if k2 not in endpoint_map:
            endpoint_map[k2] = []
        endpoint_map[k2].append((i, 1))  # segment i, endpoint 1

    contours = []

    for start_idx in range(len(segments)):
        if used[start_idx]:
            continue

        # Start a new chain from this segment
        chain_points = deque([segments[start_idx][0], segments[start_idx][1]])
        used[start_idx] = True

        # Extend forward from the tail
        changed = True
        while changed:
            changed = False
            tail = chain_points[-1]
            k = _key(tail)
            if k in endpoint_map:
                for seg_idx, ep_idx in endpoint_map[k]:
                    if used[seg_idx]:
                        continue
                    # This segment connects at our tail
                    p1, p2 = segments[seg_idx]
                    if ep_idx == 0:
                        # p1 matches tail → extend with p2
                        chain_points.append(p2)
                    else:
                        # p2 matches tail → extend with p1
                        chain_points.append(p1)
                    used[seg_idx] = True
                    changed = True
                    break

        # Extend backward from the head
        changed = True
        while changed:
            changed = False
            head = chain_points[0]
            k = _key(head)
            if k in endpoint_map:
                for seg_idx, ep_idx in endpoint_map[k]:
                    if used[seg_idx]:
                        continue
                    p1, p2 = segments[seg_idx]
                    if ep_idx == 0:
                        chain_points.appendleft(p2)
                    else:
                        chain_points.appendleft(p1)
                    used[seg_idx] = True
                    changed = True
                    break

        if len(chain_points) >= min_length:
            contours.append(np.array(list(chain_points), dtype=np.float64))

    return contours


# ---------------------------------------------------------------------------
# 5. Douglas-Peucker simplification
# ---------------------------------------------------------------------------

def simplify_contour(
    contour: np.ndarray,
    epsilon: float = 1.5,
) -> np.ndarray:
    """
    Douglas-Peucker polyline simplification.

    Recursively removes points that are within epsilon distance of the line
    between the endpoints.

    Args:
        contour: Shape (M, 2), ordered polyline.
        epsilon: Max allowed deviation in pixels.

    Returns:
        simplified: Shape (K, 2), where K <= M.
    """
    if len(contour) <= 2:
        return contour.copy()

    # Find the point with maximum distance from the line (first → last)
    start = contour[0]
    end = contour[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        # Degenerate: all points the same, pick first and last
        distances = np.linalg.norm(contour - start, axis=1)
        max_idx = np.argmax(distances)
        max_dist = distances[max_idx]
    else:
        # Perpendicular distance from each point to the line
        line_unit = line_vec / line_len
        # Vector from start to each point
        diff = contour - start
        # Cross product magnitude = perpendicular distance
        cross = np.abs(diff[:, 0] * line_unit[1] - diff[:, 1] * line_unit[0])
        max_idx = np.argmax(cross)
        max_dist = cross[max_idx]

    if max_dist > epsilon:
        # Recurse on both halves
        left = simplify_contour(contour[:max_idx + 1], epsilon)
        right = simplify_contour(contour[max_idx:], epsilon)
        # Concatenate, avoiding duplicate at split point
        return np.vstack([left[:-1], right])
    else:
        # All points within tolerance — keep only endpoints
        return np.array([start, end])


# ---------------------------------------------------------------------------
# 6. Corner detection
# ---------------------------------------------------------------------------

def detect_corners(
    contour: np.ndarray,
    angle_threshold_deg: float = 30.0,
) -> List[int]:
    """
    Detect corner points in a simplified contour based on angle deviation.

    A point is a corner if the angle between the incoming and outgoing
    segments deviates from 180° by more than the threshold.

    Args:
        contour: Shape (K, 2), simplified polyline.
        angle_threshold_deg: Minimum angle deviation from straight (degrees).

    Returns:
        corners: List of indices into contour that are corners.
                 Always includes first and last index.
    """
    if len(contour) <= 2:
        return list(range(len(contour)))

    corners = [0]  # First point is always a corner
    threshold_rad = angle_threshold_deg * np.pi / 180.0

    for i in range(1, len(contour) - 1):
        # Incoming vector
        v_in = contour[i] - contour[i - 1]
        # Outgoing vector
        v_out = contour[i + 1] - contour[i]

        # Angle between them via atan2 of cross and dot products
        cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]
        dot = v_in[0] * v_out[0] + v_in[1] * v_out[1]
        angle = abs(math.atan2(cross, dot))

        # angle = 0 → straight; angle = π → U-turn
        # Deviation from straight = |angle|
        if angle > threshold_rad:
            corners.append(i)

    corners.append(len(contour) - 1)  # Last point is always a corner

    return corners


# ---------------------------------------------------------------------------
# 7. Cubic Bezier fitting
# ---------------------------------------------------------------------------

def fit_bezier_segment(points: np.ndarray) -> np.ndarray:
    """
    Fit a cubic Bezier curve to a sequence of points using least-squares.

    Uses chord-length parameterization and solves for the two interior
    control points P1, P2 given endpoints P0 = points[0], P3 = points[-1].

    This is a linear least-squares problem:
        P(tᵢ) = B0(tᵢ)·P0 + B1(tᵢ)·P1 + B2(tᵢ)·P2 + B3(tᵢ)·P3
    where B0, B3 are known → solve for P1, P2.

    Args:
        points: Shape (S, 2), ordered contour segment.

    Returns:
        control_points: Shape (4, 2), cubic Bezier control points [P0, P1, P2, P3].
    """
    S = len(points)
    P0 = points[0]
    P3 = points[-1]

    if S <= 2:
        # Straight line: distribute control points evenly
        P1 = P0 + (P3 - P0) / 3.0
        P2 = P0 + 2.0 * (P3 - P0) / 3.0
        return np.array([P0, P1, P2, P3])

    if S == 3:
        # Three points: use quadratic-to-cubic elevation
        # Place P1 and P2 to interpolate the middle point
        t_mid = 0.5
        # P(0.5) = 0.125*P0 + 0.375*P1 + 0.375*P2 + 0.125*P3 = points[1]
        # With P1 = P2 = Q: P(0.5) = 0.125*P0 + 0.75*Q + 0.125*P3
        # Q = (points[1] - 0.125*P0 - 0.125*P3) / 0.75
        Q = (points[1] - 0.125 * P0 - 0.125 * P3) / 0.75
        return np.array([P0, Q, Q, P3])

    # Chord-length parameterization
    diffs = np.diff(points, axis=0)
    chord_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumulative = np.concatenate([[0], np.cumsum(chord_lengths)])
    total_length = cumulative[-1]

    if total_length < 1e-12:
        P1 = P0 + (P3 - P0) / 3.0
        P2 = P0 + 2.0 * (P3 - P0) / 3.0
        return np.array([P0, P1, P2, P3])

    t = cumulative / total_length  # (S,) in [0, 1]

    # Bernstein basis functions
    omt = 1.0 - t
    B0 = omt**3
    B1 = 3.0 * omt**2 * t
    B2 = 3.0 * omt * t**2
    B3 = t**3

    # We want to minimize: Σ ‖B1(tᵢ)·P1 + B2(tᵢ)·P2 - (Qᵢ - B0(tᵢ)·P0 - B3(tᵢ)·P3)‖²
    # This is a linear system in P1 and P2 (2 unknowns × 2 dimensions)
    rhs = points - np.outer(B0, P0) - np.outer(B3, P3)  # (S, 2)

    # Build matrix A = [B1, B2] — shape (S, 2)
    A = np.column_stack([B1, B2])  # (S, 2)

    # Solve normal equations: AᵀA · [P1; P2] = Aᵀ · rhs
    ATA = A.T @ A  # (2, 2)
    ATb = A.T @ rhs  # (2, 2) — each column is x,y

    try:
        params = np.linalg.solve(ATA, ATb)  # (2, 2)
        P1 = params[0]  # (2,)
        P2 = params[1]  # (2,)
    except np.linalg.LinAlgError:
        # Fallback: uniform distribution
        P1 = P0 + (P3 - P0) / 3.0
        P2 = P0 + 2.0 * (P3 - P0) / 3.0

    return np.array([P0, P1, P2, P3])


# ---------------------------------------------------------------------------
# 8. Full pipeline: initialize_curves_from_image
# ---------------------------------------------------------------------------

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert (H,W,3) RGB image to (H,W) grayscale via luminance weights."""
    if image.ndim == 2:
        return image.astype(np.float64)
    # ITU-R 601 luminance
    return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.float64)


def _image_to_float(image) -> np.ndarray:
    """Convert image to numpy float in [0, 1]."""
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image, dtype=np.float64)

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float64) / 255.0

    # Handle (C, H, W) → (H, W, C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)

    return arr.clip(0, 1)


def _sample_color_at_curve(
    image: np.ndarray,
    control_points: np.ndarray,
) -> np.ndarray:
    """
    Sample the average color of the image along a Bezier curve.

    Returns RGB in [0, 1] as numpy array of shape (3,).
    """
    H, W = image.shape[:2]

    # Sample 10 points along the curve
    t = np.linspace(0, 1, 10)
    P0, P1, P2, P3 = control_points
    omt = 1.0 - t
    points = (
        np.outer(omt**3, P0)
        + np.outer(3 * omt**2 * t, P1)
        + np.outer(3 * omt * t**2, P2)
        + np.outer(t**3, P3)
    )  # (10, 2)

    colors = []
    for px, py in points:
        x, y = int(round(px)), int(round(py))
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        if image.ndim == 3:
            colors.append(image[y, x, :3])
        else:
            val = image[y, x]
            colors.append([val, val, val])

    avg = np.mean(colors, axis=0)
    return np.clip(avg, 0, 1)


def initialize_curves_from_image(
    image,
    canvas_size: Optional[Tuple[int, int]] = None,
    max_curves: int = 300,
    min_curves: int = 10,
    edge_sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
    simplify_epsilon: float = 1.5,
    corner_angle_threshold: float = 30.0,
    default_width: float = 3.0,
    device: str = "cpu",
    is_binary: bool = False,
) -> Scene:
    """
    Full SDF-based initialization pipeline: image → Scene with fitted curves.

    Pipeline:
        1. Convert to grayscale
        2. Edge detection (or use binary mask directly)
        3. Compute SDF via JFA
        4. Extract contours via marching squares
        5. Simplify contours (Douglas-Peucker)
        6. Detect corners and segment contours
        7. Fit cubic Bezier to each segment
        8. Package as Scene

    Args:
        image: Input image — numpy (H,W), (H,W,3) or torch Tensor.
               If is_binary=True, treated as a binary mask.
        canvas_size: (H, W) canvas dimensions. Defaults to image size.
        max_curves: Maximum number of curves to output.
        min_curves: Minimum curves — if fewer extracted, relax thresholds.
        edge_sigmas: Gaussian blur scales for multi-scale Sobel.
        simplify_epsilon: Douglas-Peucker tolerance in pixels.
        corner_angle_threshold: Corner detection angle threshold in degrees.
        default_width: Default stroke width for fitted curves.
        device: Torch device for the output Scene.
        is_binary: If True, treat input as a binary mask (skip edge detection).

    Returns:
        scene: Scene with fitted Curve objects.
    """
    # --- Convert image ---
    img_float = _image_to_float(image)
    gray = _to_grayscale(img_float)
    H, W = gray.shape

    if canvas_size is None:
        canvas_size = (H, W)

    # --- Edge detection ---
    if is_binary:
        # For binary masks, edges are at the boundary
        binary_mask = gray > 0.5

        # Find edges as boundary pixels (pixels where any neighbor differs)
        padded = np.pad(binary_mask.astype(np.float64), 1, mode="constant")
        edges = np.zeros_like(binary_mask, dtype=bool)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                shifted = padded[1+dy:H+1+dy, 1+dx:W+1+dx]
                edges |= (binary_mask != shifted.astype(bool))
        interior = binary_mask
    else:
        # Full edge detection pipeline
        edges = detect_edges(gray, sigmas=edge_sigmas)
        # Create interior mask via Otsu threshold
        interior, _ = otsu_threshold(gray)

    # --- SDF via JFA ---
    sdf = compute_sdf_jfa(edges, interior=interior)

    # --- Contour extraction ---
    contours = extract_contours(sdf, iso=0.0, min_length=5)

    if not contours:
        # Fallback: try with relaxed min_length
        contours = extract_contours(sdf, iso=0.0, min_length=2)

    # --- Process each contour into curves ---
    all_curves = []

    # Sort contours by length descending to prioritize major shapes
    contours = sorted(contours, key=len, reverse=True)

    for contour in contours:
        # Simplify
        simplified = simplify_contour(contour, epsilon=simplify_epsilon)

        if len(simplified) < 2:
            continue

        # Detect corners
        corners = detect_corners(simplified, angle_threshold_deg=corner_angle_threshold)

        # Fit Bezier to each segment between corners
        for i in range(len(corners) - 1):
            start_idx = corners[i]
            end_idx = corners[i + 1]
            segment = simplified[start_idx:end_idx + 1]

            if len(segment) < 2:
                continue

            # Fit cubic Bezier
            cp = fit_bezier_segment(segment)

            # Sample color from the original image
            color_rgb = _sample_color_at_curve(img_float, cp)

            # Convert color to pre-sigmoid (inverse sigmoid)
            color_raw = np.log(color_rgb.clip(0.01, 0.99) / (1.0 - color_rgb.clip(0.01, 0.99)))

            # Convert width to pre-softplus (inverse softplus)
            # softplus(x) = ln(1 + exp(x)), inverse: x = ln(exp(w) - 1)
            w = default_width
            width_raw = math.log(math.exp(w) - 1) if 0.5 < w < 20 else w

            # High opacity: pre-sigmoid value for ~0.95
            opacity_raw = math.log(0.95 / 0.05)  # ≈ 2.94

            curve = Curve(
                control_points=torch.tensor(cp, dtype=torch.float32, device=device),
                color=torch.tensor(color_raw, dtype=torch.float32, device=device),
                width=torch.tensor(width_raw, dtype=torch.float32, device=device),
                opacity=torch.tensor(opacity_raw, dtype=torch.float32, device=device),
                z_index=len(all_curves),
            )
            all_curves.append(curve)

            if len(all_curves) >= max_curves:
                break

        if len(all_curves) >= max_curves:
            break

    # If too few curves, retry with relaxed epsilon
    if len(all_curves) < min_curves and simplify_epsilon > 0.5:
        return initialize_curves_from_image(
            image,
            canvas_size=canvas_size,
            max_curves=max_curves,
            min_curves=min_curves,
            edge_sigmas=edge_sigmas,
            simplify_epsilon=simplify_epsilon * 0.5,
            corner_angle_threshold=corner_angle_threshold * 0.75,
            default_width=default_width,
            device=device,
            is_binary=is_binary,
        )

    # --- Build background color from image corners ---
    corner_colors = []
    for cy, cx in [(0, 0), (0, W-1), (H-1, 0), (H-1, W-1)]:
        if img_float.ndim == 3:
            corner_colors.append(img_float[cy, cx, :3])
        else:
            v = img_float[cy, cx]
            corner_colors.append([v, v, v])
    bg_rgb = np.mean(corner_colors, axis=0)
    bg_raw = np.log(np.clip(bg_rgb, 0.01, 0.99) / (1.0 - np.clip(bg_rgb, 0.01, 0.99)))

    scene = Scene(
        curves=all_curves,
        canvas_size=canvas_size,
        background=torch.tensor(bg_raw, dtype=torch.float32, device=device),
        device=device,
    )

    return scene


# ---------------------------------------------------------------------------
# Filled-region extraction (tracer-style scaffold for crisp output)
# ---------------------------------------------------------------------------

def _kmeans_quantize(image_rgb: np.ndarray, k: int, iters: int = 12,
                     max_samples: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight k-means color quantization.

    Args:
        image_rgb: (H, W, 3) float in [0, 1].
        k: number of color clusters.
        iters: Lloyd iterations.
        max_samples: subsample cap for centroid estimation (speed).

    Returns:
        labels: (H, W) int cluster index per pixel.
        centers: (k, 3) float cluster colors in [0, 1].
    """
    H, W, _ = image_rgb.shape
    pixels = image_rgb.reshape(-1, 3)
    n = pixels.shape[0]

    rng = np.random.default_rng(0)
    sample = pixels if n <= max_samples else pixels[rng.choice(n, max_samples, replace=False)]

    # k-means++ style seeding (first random, rest farthest)
    centers = sample[rng.integers(0, sample.shape[0])][None, :].copy()
    for _ in range(1, k):
        d2 = ((sample[:, None, :] - centers[None, :, :]) ** 2).sum(-1).min(1)
        total = d2.sum()
        if total <= 1e-12:
            # All remaining points coincide with existing centers — pick at random.
            centers = np.vstack([centers, sample[rng.integers(0, sample.shape[0])]])
            continue
        probs = d2 / total
        probs = probs / probs.sum()  # guard against float drift (must sum to 1)
        centers = np.vstack([centers, sample[rng.choice(sample.shape[0], p=probs)]])

    for _ in range(iters):
        d2 = ((sample[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d2.argmin(1)
        for j in range(k):
            members = sample[assign == j]
            if len(members):
                centers[j] = members.mean(0)

    # Final assignment over all pixels
    d2_all = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    labels = d2_all.argmin(1).reshape(H, W)
    return labels, centers


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Binary boundary of a mask: pixels in `mask` adjacent to a non-mask pixel."""
    b = np.zeros_like(mask, dtype=bool)
    b[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    b[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    b[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    b[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    # Image-border pixels of the mask are also boundary.
    b[0, :] |= mask[0, :]
    b[-1, :] |= mask[-1, :]
    b[:, 0] |= mask[:, 0]
    b[:, -1] |= mask[:, -1]
    return b


def _fit_closed_loop(contour: np.ndarray, simplify_epsilon: float,
                     corner_angle_deg: float) -> Optional[np.ndarray]:
    """
    Fit a closed loop of cubic Bezier segments to one (already closed) contour.

    Returns segments of shape (S, 4, 2), or None if the contour is degenerate.
    The last segment's P3 equals the first segment's P0 (exact closure).
    """
    if len(contour) < 4:
        return None

    # Ensure the polyline is explicitly closed (first point repeated at end).
    closed = contour
    if not np.allclose(contour[0], contour[-1]):
        closed = np.vstack([contour, contour[0:1]])

    simplified = simplify_contour(closed, epsilon=simplify_epsilon)
    if len(simplified) < 4:
        return None

    corners = detect_corners(simplified, angle_threshold_deg=corner_angle_deg)
    # Need at least 3 distinct corners to bound a closed area.
    if len(corners) < 3:
        corners = [0, len(simplified) // 3, 2 * len(simplified) // 3, len(simplified) - 1]

    # Cap segments per loop. The filled renderer's autograd memory scales with
    # (#boundary points)·H·W; an uncapped detailed contour (hundreds of corners)
    # OOMs at full resolution. ~MAX_SEG cubic segments render a smooth base blob.
    MAX_SEG = 48
    if len(corners) - 1 > MAX_SEG:
        keep = np.linspace(0, len(corners) - 1, MAX_SEG + 1).round().astype(int)
        corners = sorted(set(corners[k] for k in keep))

    segments = []
    for i in range(len(corners) - 1):
        seg_pts = simplified[corners[i]:corners[i + 1] + 1]
        if len(seg_pts) < 2:
            continue
        bez = fit_bezier_segment(seg_pts)  # (4, 2)
        segments.append(bez)

    if len(segments) < 2:
        return None

    segments = np.stack(segments, axis=0)  # (S, 4, 2)
    # Enforce exact closure: last P3 := first P0.
    segments[-1, 3] = segments[0, 0]
    return segments.astype(np.float32)


def initialize_filled_regions(
    image,
    n_colors: int = 12,
    min_region_frac: float = 0.002,
    simplify_epsilon: float = 1.5,
    corner_angle_deg: float = 35.0,
) -> List[dict]:
    """
    Tracer-style filled-region scaffold.

    Color-quantizes the image, then for each color cluster extracts closed
    contours (marching squares on a JFA SDF), simplifies them, and fits closed
    cubic-Bezier loops. Produces crisp flat-filled regions — the base layer that
    gives CurveSplat tracer-quality output, optionally refined by the painterly
    splat engine on top.

    Args:
        image: (H, W, 3) array (uint8 or float) or (H, W) grayscale.
        n_colors: number of color clusters (flat regions).
        min_region_frac: skip clusters/contours smaller than this fraction of
            the image area (speck filter).
        simplify_epsilon: Douglas-Peucker tolerance in pixels.
        corner_angle_deg: corner-detection angle threshold.

    Returns:
        loops: list of dicts, each:
            {'segments': (S,4,2) float32, 'color': (3,) float [0,1],
             'z_index': int, 'area': int}
        Sorted by area descending (largest region → lowest z_index, painted first).
    """
    img = _image_to_float(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[:, :, :3]
    H, W = img.shape[:2]
    min_area = max(16, int(min_region_frac * H * W))

    labels, centers = _kmeans_quantize(img, n_colors)

    raw_loops = []
    for c in range(len(centers)):
        cluster_mask = labels == c
        if cluster_mask.sum() < min_area:
            continue

        boundary = _mask_boundary(cluster_mask)
        if not boundary.any():
            continue

        sdf = compute_sdf_jfa(boundary, interior=cluster_mask)
        contours = extract_contours(sdf, iso=0.0, min_length=8)

        for contour in contours:
            # Approximate enclosed area via the shoelace formula (speck filter).
            x, y = contour[:, 0], contour[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            if area < min_area:
                continue
            segments = _fit_closed_loop(contour, simplify_epsilon, corner_angle_deg)
            if segments is None:
                continue
            # Color: mean of the actual image pixels in this cluster (faithful).
            color = img[cluster_mask].mean(0) if cluster_mask.any() else centers[c]
            raw_loops.append({
                "segments": segments,
                "color": np.clip(color, 0.0, 1.0).astype(np.float32),
                "area": float(area),
            })

    # z-order by area descending: largest region painted first (lowest z_index).
    raw_loops.sort(key=lambda d: d["area"], reverse=True)
    for z, loop in enumerate(raw_loops):
        loop["z_index"] = z
    return raw_loops
