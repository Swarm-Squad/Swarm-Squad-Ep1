"""
V2V Channel Model with LOS/NLOS classification, path loss, and fading.

Replaces the simplified distance-only communication model with a realistic
propagation model inspired by GEMV2 and 3GPP TR 37.885:

  1. Geometric LOS/NLOS classification (ray-sphere intersection)
     - PHYSICAL obstacles block LOS -> NLOS_OBSTACLE
     - LOW_JAM / HIGH_JAM zones do NOT force NLOS: they act as
       in-beam attenuators (jammer lowers SNR, not geometry).
  2. Log-distance path loss with type-dependent exponents
  3. Log-normal shadow fading (stationary AR(1))
  4. Rician (LOS) / Rayleigh (NLOS) small-scale fading
  5. Link quality mapping via sigmoid over SNR

Jamming zone degradation (D_i * D_j) is preserved by the caller and
applied on top of the channel quality returned by this model.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np


class LinkType(Enum):
    LOS = "los"
    NLOS_VEHICLE = "nlos_vehicle"
    NLOS_OBSTACLE = "nlos_obstacle"


class ObstacleKind(Enum):
    """Kind of obstacle for V2V channel purposes.

    Mirrors ``algo.base.ObstacleType`` but the channel module is kept
    independent of it to avoid circular imports.
    """

    PHYSICAL = "physical"
    LOW_JAM = "low_jam"
    HIGH_JAM = "high_jam"


@dataclass
class ObstacleSpec:
    """Geometry + semantic kind of an obstacle for V2V channel analysis."""

    center: np.ndarray
    radius: float
    kind: ObstacleKind = ObstacleKind.PHYSICAL


@dataclass
class ChannelParams:
    """Tunable channel model parameters."""

    # Transmit power (dBm) -- typical DSRC
    tx_power: float = 23.0
    # Carrier frequency (GHz)
    freq_ghz: float = 5.9
    # Reference distance (m)
    d0: float = 1.0
    # Free-space path loss at d0  (Friis: 20*log10(4*pi*d0*f/c))
    pl0: float = 47.86

    # Path loss exponents per link type
    n_los: float = 2.0
    n_nlosv: float = 2.5
    n_nloso: float = 3.5

    # Additional vehicle obstruction loss (dB) for NLOSv
    vehicle_loss_db: float = 12.0

    # Extra in-beam attenuation (dB) when the ray passes through a jamming zone
    # (LOS is preserved, but SNR is reduced). These model the jammer-induced
    # noise floor rise as equivalent signal attenuation for quality mapping.
    low_jam_extra_db: float = 6.0
    high_jam_extra_db: float = 15.0

    # Shadow fading standard deviation (dB) per link type
    sigma_los: float = 3.0
    sigma_nlosv: float = 5.0
    sigma_nloso: float = 7.0
    # Temporal correlation factor for shadow fading (0 = i.i.d., 1 = static).
    # The AR(1) recursion uses sqrt(1 - corr^2) on the innovation so the
    # unconditional variance stays sigma^2 regardless of corr.
    shadow_correlation: float = 0.8

    # Rician K-factor (linear) for LOS links
    # K=6 dB -> linear ~3.98
    rician_k_los: float = 3.98
    # NLOS uses Rayleigh (K=0)

    # Noise floor (dBm)
    noise_floor: float = -95.0
    # SNR midpoint for sigmoid quality mapping (dB)
    snr_midpoint: float = 10.0
    # Sigmoid steepness
    snr_steepness: float = 0.25

    # Effective vehicle radius for NLOSv ray intersection (meters)
    vehicle_body_radius: float = 2.0

    # Enable/disable individual model components
    enable_shadow_fading: bool = True
    enable_small_scale_fading: bool = True
    # When True, LOW_JAM/HIGH_JAM zones do NOT block LOS geometry; they
    # only add in-beam attenuation. When False, any obstacle (any kind)
    # along the ray forces NLOS_OBSTACLE (legacy behavior).
    enable_obstacle_type_awareness: bool = True


@dataclass
class LinkState:
    """Per-link state for temporal correlation."""

    link_type: LinkType = LinkType.LOS
    shadow_fading_db: float = 0.0
    small_scale_fading_db: float = 0.0
    path_loss_db: float = 0.0
    received_power_dbm: float = 0.0
    snr_db: float = 0.0
    quality: float = 1.0
    jam_attenuation_db: float = 0.0  # extra loss from in-beam jammers


class V2VChannelModel:
    """
    Geometry-aware V2V channel model.

    Usage:
        model = V2VChannelModel()
        quality_matrix = model.compute_quality_matrix(positions, obstacles)

    ``obstacles`` may be either:
      - a list of ``ObstacleSpec`` (preferred), or
      - the legacy pair ``(obstacle_centers, obstacle_radii)`` where all
        obstacles are treated as PHYSICAL blockers.
    """

    def __init__(self, params: Optional[ChannelParams] = None):
        self.params = params or ChannelParams()
        self._rng = np.random.default_rng(42)
        # Shadow fading state for temporal correlation: (i, j) -> last_value
        self._shadow_state: dict[tuple[int, int], float] = {}
        # Link state cache for visualization / debugging
        self._link_states: dict[tuple[int, int], LinkState] = {}

    def reset(self):
        """Reset fading state (e.g., on simulation reset)."""
        self._shadow_state.clear()
        self._link_states.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_quality_matrix(
        self,
        positions: np.ndarray,
        obstacles: Optional[list[ObstacleSpec]] = None,
        # Legacy positional form (kept for backward compatibility):
        obstacle_centers: Optional[list[list[float]]] = None,
        obstacle_radii: Optional[list[float]] = None,
    ) -> np.ndarray:
        """
        Compute NxN pairwise communication quality matrix.

        Returns (N, N) quality matrix with values in [0, 1]; diagonal is 1.
        """
        n = positions.shape[0]
        quality = np.eye(n)

        specs = self._normalize_obstacles(obstacles, obstacle_centers, obstacle_radii)

        for i in range(n):
            for j in range(i + 1, n):
                q = self._compute_link_quality(
                    i,
                    j,
                    positions[i],
                    positions[j],
                    positions,
                    specs,
                )
                quality[i, j] = q
                quality[j, i] = q

        return quality

    def compute_pairwise_quality(
        self,
        idx_i: int,
        idx_j: int,
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        all_positions: np.ndarray,
        obstacles: Optional[Union[list[ObstacleSpec], np.ndarray]] = None,
        obstacle_radii: Optional[np.ndarray] = None,
    ) -> float:
        """Compute quality for a single link (for use inside controller loops).

        ``obstacles`` can be either a list of ObstacleSpec or the legacy
        ``obstacle_centers`` ndarray (with ``obstacle_radii`` as second arg).
        """
        if (
            isinstance(obstacles, np.ndarray)
            or obstacles is None
            and obstacle_radii is not None
        ):
            specs = self._normalize_obstacles(None, obstacles, obstacle_radii)
        elif obstacles is None:
            specs = []
        elif obstacles and isinstance(obstacles[0], ObstacleSpec):
            specs = list(obstacles)
        else:
            # assume legacy list-of-[x,y,z] + separate radii
            specs = self._normalize_obstacles(None, obstacles, obstacle_radii)

        return self._compute_link_quality(
            idx_i,
            idx_j,
            pos_i,
            pos_j,
            all_positions,
            specs,
        )

    def get_link_states(self) -> dict[tuple[int, int], LinkState]:
        """Return cached link states for visualization."""
        return dict(self._link_states)

    def get_link_summary(
        self,
        agent_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Serialize the per-link states into plain dicts for API / logging.

        If ``agent_ids`` is provided, the (i, j) pair is rewritten as
        agent IDs; otherwise integer indices are used.
        """
        out = []
        for (i, j), ls in self._link_states.items():
            if agent_ids and i < len(agent_ids) and j < len(agent_ids):
                pair = [agent_ids[i], agent_ids[j]]
            else:
                pair = [i, j]
            out.append(
                {
                    "pair": pair,
                    "link_type": ls.link_type.value,
                    "path_loss_db": round(ls.path_loss_db, 2),
                    "shadow_fading_db": round(ls.shadow_fading_db, 2),
                    "small_scale_fading_db": round(ls.small_scale_fading_db, 2),
                    "jam_attenuation_db": round(ls.jam_attenuation_db, 2),
                    "received_power_dbm": round(ls.received_power_dbm, 2),
                    "snr_db": round(ls.snr_db, 2),
                    "quality": round(ls.quality, 4),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_obstacles(
        self,
        obstacles: Optional[list[ObstacleSpec]],
        obstacle_centers,
        obstacle_radii,
    ) -> list[ObstacleSpec]:
        if obstacles:
            return list(obstacles)
        if obstacle_centers is None or len(obstacle_centers) == 0:
            return []
        centers = np.asarray(obstacle_centers)
        radii = (
            np.asarray(obstacle_radii)
            if obstacle_radii is not None
            else np.ones(len(centers))
        )
        specs = []
        for c, r in zip(centers, radii):
            specs.append(
                ObstacleSpec(
                    center=np.asarray(c, dtype=float),
                    radius=float(r),
                    kind=ObstacleKind.PHYSICAL,
                )
            )
        return specs

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _compute_link_quality(
        self,
        idx_i: int,
        idx_j: int,
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        all_positions: np.ndarray,
        obstacles: list[ObstacleSpec],
    ) -> float:
        p = self.params
        d = float(np.linalg.norm(pos_j - pos_i))
        if d < 0.01:
            return 1.0

        # 1. Classify link + tally in-beam jammer attenuation
        link_type, jam_extra_db = self._classify_link(
            pos_i,
            pos_j,
            idx_i,
            idx_j,
            all_positions,
            obstacles,
        )

        # 2. Path loss (+ in-beam jammer attenuation)
        pl_db = self._path_loss(d, link_type) + jam_extra_db

        # 3. Shadow fading
        shadow_db = 0.0
        if p.enable_shadow_fading:
            shadow_db = self._shadow_fading(idx_i, idx_j, link_type)

        # 4. Small-scale fading
        fading_db = 0.0
        if p.enable_small_scale_fading:
            fading_db = self._small_scale_fading(link_type)

        # 5. Link budget
        prx = p.tx_power - pl_db + shadow_db + fading_db
        snr = prx - p.noise_floor

        # 6. Map to quality [0, 1] via sigmoid
        quality = 1.0 / (1.0 + np.exp(-p.snr_steepness * (snr - p.snr_midpoint)))

        # Cache state
        key = (min(idx_i, idx_j), max(idx_i, idx_j))
        self._link_states[key] = LinkState(
            link_type=link_type,
            shadow_fading_db=shadow_db,
            small_scale_fading_db=fading_db,
            path_loss_db=pl_db,
            received_power_dbm=prx,
            snr_db=snr,
            quality=float(quality),
            jam_attenuation_db=jam_extra_db,
        )

        return float(np.clip(quality, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 1. LOS / NLOS classification + jammer in-beam attenuation
    # ------------------------------------------------------------------

    def _classify_link(
        self,
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        idx_i: int,
        idx_j: int,
        all_positions: np.ndarray,
        obstacles: list[ObstacleSpec],
    ) -> tuple[LinkType, float]:
        """Determine link type and total in-beam jamming attenuation (dB)."""
        p = self.params
        origin = pos_i
        direction = pos_j - pos_i
        ray_len = float(np.linalg.norm(direction))
        if ray_len < 1e-6:
            return LinkType.LOS, 0.0
        direction = direction / ray_len

        jam_extra_db = 0.0
        blocker_found = False

        for obs in obstacles:
            if not _ray_intersects_sphere(
                origin, direction, ray_len, obs.center, obs.radius
            ):
                continue
            if p.enable_obstacle_type_awareness:
                if obs.kind == ObstacleKind.PHYSICAL:
                    blocker_found = True
                elif obs.kind == ObstacleKind.LOW_JAM:
                    jam_extra_db += p.low_jam_extra_db
                elif obs.kind == ObstacleKind.HIGH_JAM:
                    jam_extra_db += p.high_jam_extra_db
            else:
                # Legacy: every obstacle blocks LOS.
                blocker_found = True

        if blocker_found:
            return LinkType.NLOS_OBSTACLE, jam_extra_db

        # Vehicle body occlusion (only relevant when no obstacle already forced NLOSo)
        vr = p.vehicle_body_radius
        for k in range(all_positions.shape[0]):
            if k == idx_i or k == idx_j:
                continue
            if _ray_intersects_sphere(origin, direction, ray_len, all_positions[k], vr):
                return LinkType.NLOS_VEHICLE, jam_extra_db

        return LinkType.LOS, jam_extra_db

    # ------------------------------------------------------------------
    # 2. Path loss
    # ------------------------------------------------------------------

    def _path_loss(self, d: float, link_type: LinkType) -> float:
        """Log-distance path loss in dB."""
        p = self.params
        d_eff = max(d, p.d0)

        if link_type == LinkType.LOS:
            n = p.n_los
            extra = 0.0
        elif link_type == LinkType.NLOS_VEHICLE:
            n = p.n_nlosv
            extra = p.vehicle_loss_db
        else:
            n = p.n_nloso
            extra = 0.0

        return p.pl0 + 10.0 * n * np.log10(d_eff / p.d0) + extra

    # ------------------------------------------------------------------
    # 3. Shadow fading (log-normal, stationary AR(1))
    # ------------------------------------------------------------------

    def _shadow_fading(self, idx_i: int, idx_j: int, link_type: LinkType) -> float:
        """Correlated log-normal shadow fading in dB.

        Uses a stationary AR(1):

            X_t = rho * X_{t-1} + sqrt(1 - rho^2) * Z_t,  Z_t ~ N(0, sigma^2)

        so unconditional variance is exactly sigma^2 at every step.
        """
        p = self.params
        key = (min(idx_i, idx_j), max(idx_i, idx_j))

        sigma = {
            LinkType.LOS: p.sigma_los,
            LinkType.NLOS_VEHICLE: p.sigma_nlosv,
            LinkType.NLOS_OBSTACLE: p.sigma_nloso,
        }[link_type]

        innovation = float(self._rng.normal(0.0, sigma))

        prev = self._shadow_state.get(key, 0.0)
        corr = float(np.clip(p.shadow_correlation, -0.999, 0.999))
        value = corr * prev + math.sqrt(max(0.0, 1.0 - corr * corr)) * innovation

        self._shadow_state[key] = value
        return value

    # ------------------------------------------------------------------
    # 4. Small-scale fading (Rician / Rayleigh)
    # ------------------------------------------------------------------

    def _small_scale_fading(self, link_type: LinkType) -> float:
        """
        Small-scale fading gain in dB.

        LOS: Rician envelope with K-factor
        NLOS: Rayleigh envelope (K=0)
        """
        if link_type == LinkType.LOS:
            K = self.params.rician_k_los
            x = float(self._rng.normal())
            y = float(self._rng.normal())
            s = np.sqrt(2.0 * K)
            envelope = np.sqrt((x + s) ** 2 + y**2) / np.sqrt(2.0 * (K + 1.0))
        else:
            x = float(self._rng.normal(0, 1.0 / np.sqrt(2.0)))
            y = float(self._rng.normal(0, 1.0 / np.sqrt(2.0)))
            envelope = np.sqrt(x**2 + y**2)

        envelope = max(envelope, 1e-6)
        gain_db = 20.0 * np.log10(envelope)
        return float(np.clip(gain_db, -20.0, 10.0))


# ======================================================================
# Geometry utilities
# ======================================================================


def _ray_intersects_sphere(
    origin: np.ndarray,
    direction: np.ndarray,
    ray_length: float,
    sphere_center: np.ndarray,
    sphere_radius: float,
) -> bool:
    """
    Test if a ray segment intersects a sphere.

    Uses the geometric solution (fast, no sqrt needed for rejection).
    """
    oc = origin - sphere_center
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc) - sphere_radius * sphere_radius)
    discriminant = b * b - c

    if discriminant < 0:
        return False

    sqrt_disc = math.sqrt(discriminant)
    t1 = -b - sqrt_disc
    t2 = -b + sqrt_disc

    return t2 >= 0.0 and t1 <= ray_length


# ======================================================================
# Singleton
# ======================================================================

_channel_model: Optional[V2VChannelModel] = None


def get_channel_model() -> V2VChannelModel:
    """Get or create the global V2V channel model instance."""
    global _channel_model
    if _channel_model is None:
        _channel_model = V2VChannelModel()
    return _channel_model


def reset_channel_model():
    """Reset the global channel model."""
    global _channel_model
    if _channel_model is not None:
        _channel_model.reset()
    _channel_model = None
