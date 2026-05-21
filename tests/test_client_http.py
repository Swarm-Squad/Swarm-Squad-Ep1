from __future__ import annotations

from urllib.parse import urlsplit

import httpx

from swarm_squad_ep1.client import SwarmSquadClient


def custom_client_path(start, goal, jamming_zones, **kwargs):
    return [start, goal]


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

    added = client.add_agent(x=5.0, y=5.0, z=2.0)
    assert added["success"] is True
    added_agent_id = added["agent"]["agent_id"]
    fetched_agent = client.agent(added_agent_id)
    assert fetched_agent["agent_id"] == added_agent_id

    created = client.add_jamming_zone(center=(10.0, 40.0, 10.0), radius=8.0)
    assert created["success"] is True

    zones = client.list_jamming_zones()
    assert zones["count"] >= 1

    crypto = client.set_crypto_auth(True, algorithm="hmac_sha256")
    assert crypto["success"] is True
    assert crypto["enabled"] is True

    llm = client.set_llm_assistance(True)
    assert llm["success"] is True

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

    changed_algo = client.set_algorithm(path_algorithm="client_custom")
    assert changed_algo["success"] is True

    started = client.start_simulation(path_algorithm="astar")
    assert started["success"] is True

    step = client.simulate_step()
    assert step["success"] is True

    stopped = client.stop_simulation()
    assert stopped["success"] is True

    removed_custom = client.remove_custom_algorithm("client_custom")
    assert removed_custom["success"] is True

    deleted_agent = client.remove_agent(added_agent_id)
    assert deleted_agent["success"] is True


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
