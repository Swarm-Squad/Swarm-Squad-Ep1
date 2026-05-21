"""Smoke tests for the realistic V2V channel model."""
from __future__ import annotations

import math

import numpy as np
import pytest

from swarm_squad_ep1.algo.v2v_channel import (
    ChannelParams,
    LinkType,
    ObstacleKind,
    ObstacleSpec,
    V2VChannelModel,
)


def test_shadow_fading_stationary_variance():
    """The AR(1) shadow fading recursion must have stationary variance ~sigma^2."""
    params = ChannelParams(
        sigma_los=3.0,
        shadow_correlation=0.8,
        enable_shadow_fading=True,
        enable_small_scale_fading=False,
    )
    m = V2VChannelModel(params)
    # Seed for reproducibility
    m._rng = np.random.default_rng(1234)

    # Drive the AR(1) directly for one link pair.
    samples = []
    for _ in range(20_000):
        v = m._shadow_fading(0, 1, LinkType.LOS)
        samples.append(v)

    arr = np.asarray(samples[1000:])  # drop warmup
    sigma = float(np.std(arr))

    # Expected stationary std ~= params.sigma_los; tolerate ±15 %.
    assert 2.55 <= sigma <= 3.45, f"stationary sigma drift: {sigma}"


def test_link_type_los_without_obstacles():
    params = ChannelParams(enable_shadow_fading=False, enable_small_scale_fading=False)
    m = V2VChannelModel(params)
    positions = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    q = m.compute_quality_matrix(positions, obstacles=[])
    assert q.shape == (2, 2)
    assert q[0, 0] == pytest.approx(1.0)
    assert q[0, 1] == pytest.approx(q[1, 0])
    state = m.get_link_states().get((0, 1))
    assert state is not None
    assert state.link_type == LinkType.LOS


def test_physical_obstacle_forces_nlos_obstacle():
    params = ChannelParams(enable_shadow_fading=False, enable_small_scale_fading=False)
    m = V2VChannelModel(params)
    positions = np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    obs = [ObstacleSpec(center=np.array([20.0, 0.0, 0.0]), radius=3.0, kind=ObstacleKind.PHYSICAL)]
    m.compute_quality_matrix(positions, obstacles=obs)
    st = m.get_link_states()[(0, 1)]
    assert st.link_type == LinkType.NLOS_OBSTACLE
    assert st.jam_attenuation_db == pytest.approx(0.0, abs=1e-6)


def test_jamming_zone_preserves_los_but_adds_attenuation():
    params = ChannelParams(
        enable_shadow_fading=False,
        enable_small_scale_fading=False,
        enable_obstacle_type_awareness=True,
        high_jam_extra_db=15.0,
    )
    m = V2VChannelModel(params)
    positions = np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    obs = [ObstacleSpec(center=np.array([20.0, 0.0, 0.0]), radius=5.0, kind=ObstacleKind.HIGH_JAM)]

    m.compute_quality_matrix(positions, obstacles=obs)
    st = m.get_link_states()[(0, 1)]

    assert st.link_type == LinkType.LOS
    assert st.jam_attenuation_db == pytest.approx(15.0, rel=1e-6)


def test_jamming_degrades_quality_vs_clean():
    params = ChannelParams(enable_shadow_fading=False, enable_small_scale_fading=False)
    positions = np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]])

    m1 = V2VChannelModel(params)
    q_clean = m1.compute_quality_matrix(positions, obstacles=[])[0, 1]

    m2 = V2VChannelModel(params)
    obs = [ObstacleSpec(center=np.array([20.0, 0.0, 0.0]), radius=5.0, kind=ObstacleKind.HIGH_JAM)]
    q_jammed = m2.compute_quality_matrix(positions, obstacles=obs)[0, 1]

    assert q_jammed < q_clean, f"jamming did not degrade quality (clean={q_clean}, jammed={q_jammed})"


def test_link_summary_serialization():
    m = V2VChannelModel(ChannelParams(enable_shadow_fading=False, enable_small_scale_fading=False))
    positions = np.array([[0.0, 0.0, 0.0], [25.0, 0.0, 0.0]])
    m.compute_quality_matrix(positions, obstacles=[])
    summary = m.get_link_summary()
    assert isinstance(summary, list)
    assert summary, "expected at least one link summary"
    entry = summary[0]
    for key in ("pair", "link_type", "quality", "snr_db", "path_loss_db"):
        assert key in entry
