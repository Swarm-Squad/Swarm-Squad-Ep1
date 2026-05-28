from __future__ import annotations

import hashlib
import hmac
from urllib.parse import urlsplit

import httpx

from swarm_squad_ep1.client import SwarmSquadClient


def custom_client_path(start, goal, jamming_zones, **kwargs):
    return [start, goal]


def custom_client_crypto_sign(
    *, key: bytes, data: bytes, sender_id: str, crypto=None
) -> bytes:
    return hmac.new(key, sender_id.encode("utf-8") + data, hashlib.sha256).digest()


def custom_client_crypto_verify(
    *,
    key: bytes,
    data: bytes,
    signature: bytes,
    sender_id: str,
    crypto=None,
) -> bool:
    expected = hmac.new(key, sender_id.encode("utf-8") + data, hashlib.sha256).digest()
    return hmac.compare_digest(signature, expected)


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

    def get(self, url: str, params=None):
        parts = urlsplit(url)
        target = parts.path or "/"
        query = parts.query
        if params:
            encoded = "&".join(f"{k}={v}" for k, v in params.items())
            query = f"{query}&{encoded}" if query else encoded
        if query:
            target = f"{target}?{query}"
        response = self._sim_client.get(target)
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

    added = client.add_agent(x=5.0, y=5.0, z=2.0)
    assert added["success"] is True
    added_agent_id = added["agent"]["agent_id"]
    fetched_agent = client.agent(added_agent_id)
    assert fetched_agent["agent_id"] == added_agent_id
    updated_agent = client.update_agent(
        added_agent_id,
        position=(6.0, 6.0, 2.5),
        jammed=False,
        communication_quality=0.95,
    )
    assert updated_agent["position"] == [6.0, 6.0, 2.5]

    created = client.add_jamming_zone(center=(10.0, 40.0, 10.0), radius=8.0)
    assert created["success"] is True

    zones = client.list_jamming_zones()
    assert zones["count"] >= 1
    jam_id = zones["zones"][0]["id"]
    assert client.get_jamming_zone(jam_id)["id"] == jam_id
    updated_jam = client.update_jamming_zone(
        jam_id, radius=9.5, obstacle_type="high_jam", intensity=0.8
    )
    assert updated_jam["success"] is True

    spoof = client.add_spoofing_zone(center=(20.0, 55.0, 10.0), radius=12.0)
    assert spoof["success"] is True
    spoof_id = spoof["zone"]["id"]
    assert client.get_spoofing_zone(spoof_id)["id"] == spoof_id
    updated_spoof = client.update_spoofing_zone(
        spoof_id,
        spoof_type="coordinate",
        coordinate_vector=(3.0, -2.0, 0.0),
        radius=13.0,
    )
    assert updated_spoof["success"] is True

    crypto = client.set_crypto_auth(True, algorithm="hmac_sha256")
    assert crypto["success"] is True
    assert crypto["enabled"] is True
    crypto_status = client.crypto_auth_status()
    assert "status" in crypto_status

    llm = client.set_llm_assistance(True)
    assert llm["success"] is True
    llm_targets = client.llm_targets()
    assert "targets" in llm_targets
    assert client.clear_all_llm_targets()["success"] is True

    v2v = client.set_v2v_channel(True, params={"tx_power": 20.0})
    assert v2v["success"] is True
    assert client.v2v_channel_status()["enabled"] is True

    custom = client.register_custom_algorithm(
        name="client_custom",
        import_path="tests.test_client_http:custom_client_path",
        description="Client custom path",
    )
    assert custom["success"] is True
    assert any(
        item["name"] == "client_custom"
        for item in client.list_custom_algorithms()["algorithms"]
    )
    custom_crypto = client.register_custom_crypto_algorithm(
        name="client_custom_crypto",
        sign_import_path="tests.test_client_http:custom_client_crypto_sign",
        verify_import_path="tests.test_client_http:custom_client_crypto_verify",
        description="Client custom crypto",
    )
    assert custom_crypto["success"] is True
    assert any(
        item["name"] == "client_custom_crypto"
        for item in client.list_custom_crypto_algorithms()["algorithms"]
    )

    changed_algo = client.set_algorithm(path_algorithm="client_custom")
    assert changed_algo["success"] is True
    changed_crypto = client.set_crypto_auth(True, algorithm="client_custom_crypto")
    assert changed_crypto["success"] is True

    started = client.start_simulation(
        path_algorithm="astar",
        destination=(25.0, 25.0, 8.0),
        default_obstacle_type="low_jam",
    )
    assert started["success"] is True

    step = client.simulate_step()
    assert step["success"] is True

    stopped = client.stop_simulation()
    assert stopped["success"] is True

    removed_custom = client.remove_custom_algorithm("client_custom")
    assert removed_custom["success"] is True
    removed_custom_crypto = client.remove_custom_crypto_algorithm(
        "client_custom_crypto"
    )
    assert removed_custom_crypto["success"] is True

    deleted_agent = client.remove_agent(added_agent_id)
    assert deleted_agent["success"] is True

    download_json = client.download_simulation_results(format="json")
    assert "download_timestamp" in download_json
    download_csv = client.download_simulation_results(format="csv")
    assert isinstance(download_csv, str)


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


def test_client_script_control_loop_executes_commands(sim_client, monkeypatch):
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda timeout=10.0: _HttpxClientAdapter(sim_client, timeout=timeout),
    )

    client = SwarmSquadClient(base_url="http://sim.local")
    first_agent = next(iter(client.agents()["agents"]))

    def scripted_controller(state, step_idx):
        if step_idx > 1:
            return []
        agent_state = state["agents"][first_agent]
        pos = agent_state["position"]
        return [
            {
                "agent": first_agent,
                "x": float(pos[0]) + 1.0,
                "y": float(pos[1]) + 0.5,
                "z": float(pos[2]),
            }
        ]

    trace = client.run_script_control_loop(scripted_controller, steps=3)
    assert len(trace) == 3
    assert trace[0]["commands_applied"]
    assert trace[0]["simulate_step"]["success"] is True
