from __future__ import annotations

from urllib.parse import urlsplit

import httpx

from swarm_squad_ep1.client import SwarmSquadClient


class _ResponseAdapter:
    def __init__(self, response):
        self._response = response
        self.status_code = response.status_code
        self.text = response.text

    def json(self):
        return self._response.json()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status error: {self.status_code}",
                request=httpx.Request("GET", "http://sim.local"),
                response=httpx.Response(self.status_code),
            )


class _HttpxClientAdapter:
    def __init__(self, sim_client, timeout=10.0):
        self._sim_client = sim_client
        self._timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method: str, url: str, json=None):
        parts = urlsplit(url)
        target = parts.path or "/"
        if parts.query:
            target = f"{target}?{parts.query}"
        response = self._sim_client.request(method, target, json=json)
        return _ResponseAdapter(response)


def test_client_live_http_methods_against_sim_api(sim_client, monkeypatch):
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda timeout=10.0: _HttpxClientAdapter(sim_client, timeout=timeout),
    )

    client = SwarmSquadClient(base_url="http://sim.local")

    status = client.status()
    assert "boundaries" in status

    sim_state = client.simulation_state()
    assert "running" in sim_state

    created = client.add_jamming_zone(center=(10.0, 40.0, 10.0), radius=8.0)
    assert created["success"] is True

    zones = client.list_jamming_zones()
    assert zones["count"] >= 1

    crypto = client.set_crypto_auth(True, algorithm="hmac_sha256")
    assert crypto["success"] is True
    assert crypto["enabled"] is True

    llm = client.set_llm_assistance(True)
    assert llm["success"] is True

    started = client.start_simulation(path_algorithm="astar")
    assert started["success"] is True

    stopped = client.stop_simulation()
    assert stopped["success"] is True


def test_client_apply_preset_orchestrates_live_configuration(sim_client, monkeypatch):
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda timeout=10.0: _HttpxClientAdapter(sim_client, timeout=timeout),
    )

    client = SwarmSquadClient(base_url="http://sim.local")
    result = client.apply_preset("intro_combined", seed=1)
    assert result["success"] is True
    assert result["preset"] == "intro_combined"
    assert result["jamming_zones"] >= 1
    assert result["spoofing_zones"] >= 1
