from __future__ import annotations

import pytest

import swarm_squad_ep1.chat.app as chat_module

pytestmark = pytest.mark.frontend_contract


def test_dashboard_shell_and_static_assets(chat_client):
    index = chat_client.get("/")
    assert index.status_code == 200
    html = index.text
    assert "/static/js/scene3d.js" in html
    assert "/static/js/chat.js" in html
    assert "/static/js/app.js" in html
    assert "Beginner Guide" not in html

    assets = [
        "/static/js/app.js",
        "/static/js/chat.js",
        "/static/js/scene3d.js",
        "/static/css/style.css",
    ]
    for path in assets:
        response = chat_client.get(path)
        assert response.status_code == 200, f"expected {path} to be served"
        assert response.text, f"expected non-empty content for {path}"


def test_app_config_bootstrap_contract(monkeypatch, chat_client):
    async def _fake_status(timeout: float = 3.0):  # pragma: no cover - async callback
        return {
            "running": False,
            "agent_count": 5,
            "boundaries": {
                "x_range": [-100.0, 100.0],
                "y_range": [-100.0, 100.0],
                "z_range": [0.0, 50.0],
                "mission_end": [20.0, 20.0, 10.0],
            },
            "timestamp": "2026-01-01T00:00:00",
        }

    monkeypatch.setattr(chat_module, "_fetch_simulation_status", _fake_status)

    response = chat_client.get("/app_config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["simulation_online"] is True
    assert "simulation" in payload
    boundaries = payload["simulation"]["boundaries"]
    assert {"x_range", "y_range", "z_range", "mission_end"} <= boundaries.keys()


def test_status_contract_includes_boundaries(chat_client):
    response = chat_client.get("/status")
    assert response.status_code == 200
    payload = response.json()
    assert "boundaries" in payload
    assert {"x_range", "y_range", "z_range", "mission_end"} <= payload[
        "boundaries"
    ].keys()


def test_frontend_runtime_config_contract_for_algorithm_selectors(chat_client):
    response = chat_client.get("/static/js/app.js")
    assert response.status_code == 200
    js = response.text
    assert "path_algorithms" in js
    assert "custom_path_algorithms" in js
    assert "crypto_algorithms" in js
    assert "crypto_algorithm_labels" in js
    assert "custom_crypto_algorithms" in js
    assert "current.path_algorithm" in js
    assert "current.crypto_algorithm" in js
