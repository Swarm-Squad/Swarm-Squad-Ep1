"""
Vectorized link-quality and jamming model, identical to SwarmState's
per-pair loops, for prediction inside the MPC-style supervisor and for tests.
"""

import numpy as np

import swarm_squad_ep1.config as config

BETA = config.ALPHA * (2**config.DELTA - 1)


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def far_field(r: np.ndarray) -> np.ndarray:
    """a_ij, the far-field reception probability."""
    return np.exp(-BETA * (r / config.R0) ** config.V)


def phi(r: np.ndarray) -> np.ndarray:
    """phi(r_ij) = g_ij * a_ij."""
    return r / np.sqrt(r**2 + config.R0**2) * far_field(r)


def penetration_depth(positions: np.ndarray, jammers) -> np.ndarray:
    """
    Penetration depth of every agent (max over jammers), 0 outside all fields.

    Args:
        positions: (n, 2) agent positions
        jammers: iterable of (x, y, r_obs)
    """
    depth = np.zeros(len(positions))
    for x, y, r_obs in jammers:
        r_jam = r_obs * config.JAMMING_RADIUS_MULTIPLIER
        d = np.linalg.norm(positions - np.array([x, y]), axis=1)
        inside = d < r_jam
        dep = 1.0 - np.maximum(0.0, (d - r_obs) / (r_jam - r_obs))
        depth = np.where(inside, np.maximum(depth, dep), depth)
    return depth


def degradation_factor(depth: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """D_final per agent; 1 for agents outside every field."""
    base = config.LOWPOWER_JAMMING_DEGRADATION
    factor = np.maximum(0.2, base + (1.0 - base) * (1.0 - depth))
    return np.where(inside, factor, 1.0)


def jammed_mask(positions: np.ndarray, jammers) -> np.ndarray:
    mask = np.zeros(len(positions), dtype=bool)
    for x, y, r_obs in jammers:
        r_jam = r_obs * config.JAMMING_RADIUS_MULTIPLIER
        mask |= np.linalg.norm(positions - np.array([x, y]), axis=1) < r_jam
    return mask


def link_quality(positions: np.ndarray, jammers=()) -> tuple:
    """
    Link-quality matrix Q and neighbor matrix a, as SwarmState computes them.

    Returns:
        (Q, a): Q_ij = phi(r_ij) D_i D_j, a_ij the far-field term
    """
    r = pairwise_distances(positions)
    n = len(positions)
    off = ~np.eye(n, dtype=bool)
    a = np.where(off, far_field(r), 0.0)
    q = np.where(off, phi(r), 0.0)
    if jammers:
        inside = jammed_mask(positions, jammers)
        factor = degradation_factor(penetration_depth(positions, jammers), inside)
        q = q * factor[:, None] * factor[None, :]
    return q, a


def jn(q: np.ndarray, a: np.ndarray) -> float:
    """J_n: mean link quality over ordered pairs with a_ij > P_T."""
    neighbors = a > config.PT
    if not neighbors.any():
        return 0.0
    return float(q[neighbors].mean())


def optimal_spacing() -> float:
    """r_alpha, the maximizer of phi."""
    r = np.linspace(1.0, 80.0, 79001)
    return float(r[np.argmax(phi(r))])
