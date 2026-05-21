from __future__ import annotations

import pytest

import swarm_squad_ep1.chat.app as chat_module

pytestmark = pytest.mark.api


class _DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class _DummyAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, timeout: float = 5.0, params=None):
        if url.endswith("/simulation/state"):
            return _DummyResponse({"running": False, "agents": {}, "formation": {}})
        if url.endswith("/"):
            return _DummyResponse({"status": "online", "service": "simulation"})
        return _DummyResponse({})

    async def post(self, url: str, json=None, timeout: float = 5.0):
        return _DummyResponse({"success": True, "echo": json or {}})

    async def delete(self, url: str, timeout: float = 5.0):
        return _DummyResponse({"success": True})


class _ErrorAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, timeout: float = 5.0, params=None):
        raise RuntimeError("sim backend unavailable")


def test_tools_catalog_contract(chat_client):
    response = chat_client.get("/tools")
    assert response.status_code == 200
    payload = response.json()
    assert {"count", "tools", "categories"} <= payload.keys()
    assert payload["count"] >= 1
    assert isinstance(payload["tools"], list)
    assert "meta" in payload["categories"]
    assert "list_tools" in payload["categories"]["meta"]


def test_health_status_and_app_config_contracts(chat_client):
    health = chat_client.get("/health")
    assert health.status_code == 200
    health_payload = health.json()
    assert health_payload["chat_api"] == "online"
    assert health_payload["simulation_api"] in {"online", "offline"}
    assert health_payload["llm"] in {"ready", "unavailable"}

    status = chat_client.get("/status")
    assert status.status_code == 200
    status_payload = status.json()
    assert {"running", "agent_count", "boundaries"} <= status_payload.keys()
    assert {"x_range", "y_range", "z_range", "mission_end"} <= status_payload[
        "boundaries"
    ].keys()

    app_config = chat_client.get("/app_config")
    assert app_config.status_code == 200
    config_payload = app_config.json()
    assert {
        "beginner_mode",
        "default_preset",
        "education_presets",
        "simulation_online",
        "simulation",
    } <= config_payload.keys()


def test_education_presets_and_invalid_load(chat_client):
    presets = chat_client.get("/education/presets")
    assert presets.status_code == 200
    presets_payload = presets.json()
    assert {"presets", "default_preset"} <= presets_payload.keys()
    assert isinstance(presets_payload["presets"], dict)
    assert presets_payload["presets"], "expected at least one education preset"

    invalid = chat_client.post("/education/load_preset", json={"preset": "not_real"})
    assert invalid.status_code == 400
    invalid_payload = invalid.json()
    assert invalid_payload["success"] is False
    assert "error" in invalid_payload


def test_representative_proxy_route(monkeypatch, chat_client):
    monkeypatch.setattr(chat_module.httpx, "AsyncClient", _DummyAsyncClient)

    response = chat_client.get("/simulation/state")
    assert response.status_code == 200
    assert response.json()["running"] is False


def test_attack_metrics_proxy_fallback_when_backend_unavailable(monkeypatch, chat_client):
    monkeypatch.setattr(chat_module.httpx, "AsyncClient", _ErrorAsyncClient)

    response = chat_client.get("/simulation/attack_metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "chat_fallback"
    assert {"tp", "fp", "fn", "tn", "detection_rate"} <= payload.keys()
