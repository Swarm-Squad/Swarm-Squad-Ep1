from __future__ import annotations

import hashlib
import hmac
import time

from swarm_squad_ep1.algo.crypto_auth import (
    CryptoAuth,
    get_available_crypto_algorithms,
    list_registered_crypto_algorithms,
    register_crypto_algorithm,
    unregister_crypto_algorithm,
)
from swarm_squad_ep1.algo.custom_crypto_algorithms import clear_custom_crypto_algorithms
from swarm_squad_ep1.algo.mavlink import MAVLinkMessage, MessageType


def custom_test_sign(*, key: bytes, data: bytes, sender_id: str, crypto=None) -> bytes:
    sender = sender_id.encode("utf-8")
    return hmac.new(key, sender + data, hashlib.sha256).digest()


def custom_test_verify(
    *,
    key: bytes,
    data: bytes,
    signature: bytes,
    sender_id: str,
    crypto=None,
) -> bool:
    sender = sender_id.encode("utf-8")
    expected = hmac.new(key, sender + data, hashlib.sha256).digest()
    return hmac.compare_digest(signature, expected)


def _demo_message() -> MAVLinkMessage:
    return MAVLinkMessage(
        msg_type=MessageType.GLOBAL_POSITION_INT,
        sender_id="agent1",
        sequence=1,
        timestamp=time.time(),
        payload={"x": 1.0, "y": 2.0, "z": 3.0},
    )


def test_register_and_use_custom_crypto_algorithm():
    clear_custom_crypto_algorithms()
    try:
        created = register_crypto_algorithm(
            name="custom_hmac",
            sign_import_path="tests.test_custom_crypto_algorithms:custom_test_sign",
            verify_import_path="tests.test_custom_crypto_algorithms:custom_test_verify",
            description="Custom HMAC test algorithm",
        )
        assert created["name"] == "custom_hmac"
        assert "custom_hmac" in get_available_crypto_algorithms()
        assert any(
            item["name"] == "custom_hmac"
            for item in list_registered_crypto_algorithms()
        )

        crypto = CryptoAuth()
        crypto.enabled = True
        crypto.generate_keys(["agent1"])
        crypto.set_algorithm("custom_hmac")

        msg = _demo_message()
        signed = crypto.sign_message(msg)
        assert signed.signature is not None
        assert crypto.verify_message(signed) is True

        tampered = signed.clone()
        tampered.payload["x"] = 999.0
        assert crypto.verify_message(tampered) is False
    finally:
        unregister_crypto_algorithm("custom_hmac")
        clear_custom_crypto_algorithms()


def test_unregister_custom_crypto_algorithm():
    clear_custom_crypto_algorithms()
    register_crypto_algorithm(
        name="remove_crypto",
        sign_import_path="tests.test_custom_crypto_algorithms:custom_test_sign",
        verify_import_path="tests.test_custom_crypto_algorithms:custom_test_verify",
    )
    removed = unregister_crypto_algorithm("remove_crypto")
    assert removed is not None
    assert removed["name"] == "remove_crypto"
    assert "remove_crypto" not in get_available_crypto_algorithms()
