from __future__ import annotations

import pytest

import swarm_squad_ep1.chat.app as chat_module

pytestmark = pytest.mark.api


class _DummyResponse:
    def __init__(self, payload, status_code: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        return self._payload


class _DummyAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(
        self,
        method: str,
        url: str,
        json=None,
        params=None,
        timeout: float = 5.0,
    ):
        if url.endswith("/simulation/state"):
            return _DummyResponse(
                {
                    "running": False,
                    "current": {
                        "formation": "communication_aware",
                        "path_algorithm": "astar",
                        "comm_model": "v2v_channel",
                        "crypto_auth_enabled": False,
                        "crypto_algorithm": "hmac_sha256",
                    },
                    "agents": {},
                    "formation": {},
                }
            )
        if url.endswith("/"):
            return _DummyResponse({"status": "online", "service": "simulation"})
        if url.endswith("/simulation/algorithm"):
            return _DummyResponse({"success": True, "echo": json or {}})
        return _DummyResponse({})

    async def get(self, url: str, timeout: float = 5.0, params=None):
        return await self.request("GET", url, params=params, timeout=timeout)

    async def post(self, url: str, json=None, timeout: float = 5.0):
        return await self.request("POST", url, json=json, timeout=timeout)

    async def delete(self, url: str, timeout: float = 5.0):
        return await self.request("DELETE", url, timeout=timeout)


class _ErrorAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(
        self,
        method: str,
        url: str,
        json=None,
        params=None,
        timeout: float = 5.0,
    ):
        raise RuntimeError("sim backend unavailable")

    async def get(self, url: str, timeout: float = 5.0, params=None):
        return await self.request("GET", url, params=params, timeout=timeout)


class _StatusPassthroughAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(
        self,
        method: str,
        url: str,
        json=None,
        params=None,
        timeout: float = 5.0,
    ):
        if url.endswith("/simulation/algorithm"):
            return _DummyResponse(
                {"detail": "invalid algorithm selection"}, status_code=422
            )
        return _DummyResponse({"detail": "upstream unavailable"}, status_code=503)


def test_tools_catalog_contract(chat_client):
    response = chat_client.get("/tools")
    assert response.status_code == 200
    payload = response.json()
    assert {"count", "tools", "categories", "registry_health"} <= payload.keys()
    assert payload["count"] >= 1
    assert isinstance(payload["tools"], list)
    assert "meta" in payload["categories"]
    assert "list_tools" in payload["categories"]["meta"]
    assert {"missing_executors", "extra_executors"} <= payload["registry_health"].keys()


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
    assert "current" in response.json()


def test_proxy_preserves_upstream_status_code(monkeypatch, chat_client):
    monkeypatch.setattr(chat_module.httpx, "AsyncClient", _StatusPassthroughAsyncClient)

    response = chat_client.post(
        "/simulation/algorithm", json={"path_algorithm": "not_real"}
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid algorithm selection"


def test_attack_metrics_proxy_fallback_when_backend_unavailable(
    monkeypatch, chat_client
):
    monkeypatch.setattr(chat_module.httpx, "AsyncClient", _ErrorAsyncClient)

    response = chat_client.get("/simulation/attack_metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "chat_fallback"
    assert payload["metric_scope"] == "spoof_detection"
    assert {
        "tp",
        "fp",
        "fn",
        "tn",
        "detection_rate",
        "duration_seconds",
        "steps",
        "avg_Jn",
        "avg_rn",
        "avg_traveled_path",
        "final_Jn",
        "final_rn",
    } <= payload.keys()


def test_fast_move_path_surfaces_failure(monkeypatch, chat_client):
    async def _failed_move(agent: str, x: float, y: float, z: float = 0.0):
        return {"success": False, "error": "blocked by spoofing filter"}

    monkeypatch.setattr(chat_module, "move_agent", _failed_move)
    response = chat_client.post("/chat", json={"message": "move agent1 to 5, 5"})
    assert response.status_code == 200
    assert "Failed to move agent1" in response.json()["response"]
