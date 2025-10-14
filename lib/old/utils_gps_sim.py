import math
import random
from typing import List, Tuple

import numpy as np


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def simulate_distance_measurement(
    a_pos: Tuple[float, float], b_pos: Tuple[float, float], noise_std: float = 0.1
) -> float:
    """Simulate distance measurement with noise."""
    true_dist = euclidean_distance(a_pos, b_pos)
    return max(0.1, true_dist + random.gauss(0, noise_std))


def trilaterate_2d(
    anchor1: Tuple[float, float],
    dist1: float,
    anchor2: Tuple[float, float],
    dist2: float,
    anchor3: Tuple[float, float],
    dist3: float,
) -> Tuple[float, float]:
    """
    Trilateration in 2D using three known positions and their distances to unknown point.
    Returns (x, y) of unknown point.

    Args:
        anchor1, anchor2, anchor3: Known positions (x, y)
        dist1, dist2, dist3: Distances from anchors to unknown point

    Returns:
        Estimated position (x, y)
    """
    x1, y1 = anchor1
    x2, y2 = anchor2
    x3, y3 = anchor3

    # System of equations for trilateration
    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = dist1**2 - dist2**2 - x1**2 + x2**2 - y1**2 + y2**2
    D = 2 * (x3 - x2)
    E = 2 * (y3 - y2)
    F = dist2**2 - dist3**2 - x2**2 + x3**2 - y2**2 + y3**2

    denominator = A * E - B * D
    if abs(denominator) < 1e-10:
        raise ValueError(
            "Trilateration failed: degenerate geometry (anchors are collinear)"
        )

    x = (C * E - B * F) / denominator
    y = (A * F - C * D) / denominator

    return (x, y)


def multilaterate_2d(
    anchors: List[Tuple[float, float]], distances: List[float]
) -> Tuple[float, float]:
    """
    Enhanced trilateration using multiple anchors (more than 3) with least squares.

    Args:
        anchors: List of known positions [(x1, y1), (x2, y2), ...]
        distances: List of distances from anchors to unknown point

    Returns:
        Estimated position (x, y)
    """
    if len(anchors) < 3:
        raise ValueError("Need at least 3 anchors for trilateration")

    if len(anchors) == 3:
        return trilaterate_2d(
            anchors[0], distances[0], anchors[1], distances[1], anchors[2], distances[2]
        )

    # Use least squares for overdetermined system (more than 3 anchors)
    n = len(anchors)
    A = np.zeros((n - 1, 2))
    b = np.zeros(n - 1)

    x0, y0 = anchors[0]
    r0 = distances[0]

    for i in range(1, n):
        xi, yi = anchors[i]
        ri = distances[i]

        A[i - 1, 0] = 2 * (xi - x0)
        A[i - 1, 1] = 2 * (yi - y0)
        b[i - 1] = ri**2 - r0**2 - xi**2 + x0**2 - yi**2 + y0**2

    try:
        # Solve using least squares
        solution = np.linalg.lstsq(A, b, rcond=None)[0]
        return (float(solution[0]), float(solution[1]))
    except np.linalg.LinAlgError:
        # Fallback to simple trilateration with first 3 anchors
        return trilaterate_2d(
            anchors[0], distances[0], anchors[1], distances[1], anchors[2], distances[2]
        )


def is_point_in_circle(
    point: Tuple[float, float], circle_center: Tuple[float, float], radius: float
) -> bool:
    """Check if a point is inside a circular area."""
    return euclidean_distance(point, circle_center) <= radius


def calculate_position_error(
    true_pos: Tuple[float, float], estimated_pos: Tuple[float, float]
) -> float:
    """Calculate the error between true and estimated positions."""
    return euclidean_distance(true_pos, estimated_pos)


def generate_circular_trajectory(
    center: Tuple[float, float],
    radius: float,
    angle: float,
    angular_speed: float = 0.05,
) -> Tuple[float, float]:
    """Generate position on a circular trajectory."""
    new_angle = angle + angular_speed
    x = center[0] + radius * math.cos(new_angle)
    y = center[1] + radius * math.sin(new_angle)
    return (x, y), new_angle


def bound_position(
    pos: Tuple[float, float], bounds: Tuple[float, float, float, float]
) -> Tuple[float, float]:
    """Keep position within specified bounds (min_x, max_x, min_y, max_y)."""
    min_x, max_x, min_y, max_y = bounds
    x = max(min_x, min(max_x, pos[0]))
    y = max(min_y, min(max_y, pos[1]))
    return (x, y)


def calculate_dilution_of_precision(
    anchors: List[Tuple[float, float]], target_pos: Tuple[float, float]
) -> float:
    """
    Calculate Geometric Dilution of Precision (GDOP) for given anchor configuration.
    Lower values indicate better geometry.
    """
    if len(anchors) < 3:
        return float("inf")

    # Calculate the geometry matrix
    n = len(anchors)
    G = np.zeros((n, 2))

    for i, anchor in enumerate(anchors):
        dist = euclidean_distance(anchor, target_pos)
        if dist < 1e-10:
            return float("inf")  # Anchor too close to target

        G[i, 0] = (target_pos[0] - anchor[0]) / dist
        G[i, 1] = (target_pos[1] - anchor[1]) / dist

    try:
        # Calculate covariance matrix
        Q = np.linalg.inv(G.T @ G)
        gdop = math.sqrt(np.trace(Q))
        return gdop
    except np.linalg.LinAlgError:
        return float("inf")  # Singular matrix


def estimate_velocity(
    prev_pos: Tuple[float, float], curr_pos: Tuple[float, float], dt: float
) -> Tuple[float, float]:
    """Estimate velocity from position change and time difference."""
    if dt <= 0:
        return (0.0, 0.0)

    vx = (curr_pos[0] - prev_pos[0]) / dt
    vy = (curr_pos[1] - prev_pos[1]) / dt

    return (vx, vy)
