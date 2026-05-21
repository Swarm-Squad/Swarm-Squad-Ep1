"""Smoke tests for CryptoAuth TP/FP/FN/TN counters."""

from __future__ import annotations

import pytest

from swarm_squad_ep1.algo.crypto_auth import CryptoAuth
from swarm_squad_ep1.algo.mavlink import MAVLinkMessage, MessageType


def _mk_msg(
    sender: str, seq: int, spoofed: bool = False, t: float = 0.0
) -> MAVLinkMessage:
    return MAVLinkMessage(
        msg_type=MessageType.GLOBAL_POSITION_INT,
        sender_id=sender,
        sequence=seq,
        timestamp=t,
        payload={"lat": 0.0, "lon": 0.0, "alt": 0.0},
        is_spoofed=spoofed,
    )


def test_disabled_crypto_accepts_all_and_counts_fn_tn():
    auth = CryptoAuth()
    auth.enabled = False

    msgs = [
        _mk_msg("a1", 1, spoofed=False),
        _mk_msg("a2", 2, spoofed=True),
        _mk_msg("a1", 3, spoofed=True),
        _mk_msg("a2", 4, spoofed=False),
    ]
    accepted = auth.filter_messages(msgs)
    assert len(accepted) == 4  # nothing rejected when disabled
    assert auth.stats.tn == 2  # 2 legit accepted -> TN
    assert auth.stats.fn == 2  # 2 spoofed accepted -> FN
    assert auth.stats.tp == 0
    assert auth.stats.fp == 0


def test_enabled_crypto_rejects_unsigned_spoofs():
    auth = CryptoAuth()
    auth.enabled = True
    auth.generate_keys(["a1", "a2"])

    legit = _mk_msg("a1", 1, spoofed=False)
    legit = auth.sign_message(legit)
    spoof = _mk_msg("a1", 2, spoofed=True)  # no signature -> invalid

    accepted = auth.filter_messages([legit, spoof])
    # legit is accepted (TN), spoof is rejected (TP)
    assert len(accepted) == 1
    assert auth.stats.tn == 1
    assert auth.stats.tp == 1
    assert auth.stats.fp == 0


def test_enabled_crypto_handles_tampered_signed_spoof():
    """A spoof that forges a *wrong* signature should be rejected (TP), not FP."""
    auth = CryptoAuth()
    auth.enabled = True
    auth.generate_keys(["a1", "a2"])

    bad = _mk_msg("a1", 1, spoofed=True)
    bad.signature = b"\x00" * 32  # garbage
    accepted = auth.filter_messages([bad])
    assert accepted == []
    assert auth.stats.tp == 1
    assert auth.stats.fp == 0


def test_to_dict_detection_metrics():
    auth = CryptoAuth()
    # Synthesize counters directly to avoid signing overhead
    auth.stats.tp = 3
    auth.stats.fn = 1  # out of 4 spoofed -> detection_rate 0.75
    auth.stats.fp = 1
    auth.stats.tn = 5
    d = auth.stats.to_dict()
    # Values are rounded to 4 decimals in to_dict()
    assert d["detection_rate"] == pytest.approx(0.75, abs=1e-6)
    assert d["false_positive_rate"] == pytest.approx(1 / 6, abs=5e-4)
    assert d["precision"] == pytest.approx(0.75, abs=1e-6)
    assert d["recall"] == pytest.approx(0.75, abs=1e-6)
