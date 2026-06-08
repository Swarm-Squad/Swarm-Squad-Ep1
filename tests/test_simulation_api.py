from __future__ import annotations

import hashlib
import hmac

import pytest

import swarm_squad_ep1.simulation.api as sim_api
from swarm_squad_ep1.config import X_RANGE, Y_RANGE, Z_RANGE

pytestmark = pytest.mark.api


def scripted_test_path(start, goal, jamming_zones, **kwargs):
    """Simple custom path algorithm used for endpoint/registry tests."""
    mid = (start + goal) / 2
    return [start, mid, goal]


def scripted_crypto_sign(
    *, key: bytes, data: bytes, sender_id: str, crypto=None
) -> bytes:
    sender = sender_id.encode("utf-8")
    return hmac.new(key, sender + data, hashlib.sha256).digest()


def scripted_crypto_verify(
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


def _first_agent_id(sim_client) -> str:
    agents = sim_client.get("/agents").json()["agents"]
    assert agents, "expected at least one initialized agent"
    return next(iter(agents))


def test_status_and_state_contract(sim_client):
    status = sim_client.get("/status")
    assert status.status_code == 200
    payload = status.json()
    assert {"running", "agent_count", "boundaries", "timestamp"} <= payload.keys()
    assert {"x_range", "y_range", "z_range", "mission_end"} <= payload[
        "boundaries"
    ].keys()

    state = sim_client.get("/simulation/state")
    assert state.status_code == 200
    state_payload = state.json()
    assert {
        "running",
        "current",
        "formation",
        "agents",
        "jamming_zones",
        "spoofing_zones",
        "timestamp",
    } <= state_payload.keys()
    assert {
        "formation",
        "path_algorithm",
        "comm_model",
        "crypto_auth_enabled",
        "crypto_algorithm",
    } <= state_payload["current"].keys()
    assert isinstance(state_payload["agents"], dict)


def test_agent_crud_and_bounds_validation(sim_client):
    good = sim_client.post("/agents", json={"x": 0.0, "y": 0.0, "z": 0.0})
    assert good.status_code == 200
    body = good.json()
    assert body["success"] is True
    new_agent_id = body["agent"]["agent_id"]

    fetched = sim_client.get(f"/agents/{new_agent_id}")
    assert fetched.status_code == 200
    assert fetched.json()["agent_id"] == new_agent_id

    bad = sim_client.post("/agents", json={"x": X_RANGE[1] + 1.0, "y": 0.0, "z": 0.0})
    assert bad.status_code == 400

    deleted = sim_client.delete(f"/agents/{new_agent_id}")
    assert deleted.status_code == 200
    assert deleted.json()["success"] is True

    missing = sim_client.get(f"/agents/{new_agent_id}")
    assert missing.status_code == 404


def test_jamming_zone_crud(sim_client):
    created = sim_client.post(
        "/jamming_zones",
        json={
            "center": [10.0, 20.0, 5.0],
            "radius": 7.0,
            "obstacle_type": "high_jam",
        },
    )
    assert created.status_code == 200
    data = created.json()
    assert data["success"] is True
    zone_id = data["zone"]["id"]

    listed = sim_client.get("/jamming_zones")
    assert listed.status_code == 200
    assert any(zone["id"] == zone_id for zone in listed.json()["zones"])

    deleted = sim_client.delete(f"/jamming_zones/{zone_id}")
    assert deleted.status_code == 200
    assert deleted.json()["success"] is True

    cleared = sim_client.delete("/jamming_zones")
    assert cleared.status_code == 200
    assert cleared.json()["success"] is True


def test_spoofing_zone_crud(sim_client):
    created = sim_client.post(
        "/spoofing_zones",
        json={
            "center": [15.0, 25.0, 5.0],
            "radius": 10.0,
            "spoof_type": "position_falsification",
        },
    )
    assert created.status_code == 200
    payload = created.json()
    assert payload["success"] is True
    zone_id = payload["zone"]["id"]

    listed = sim_client.get("/spoofing_zones")
    assert listed.status_code == 200
    assert listed.json()["count"] >= 1
    assert any(zone["id"] == zone_id for zone in listed.json()["zones"])

    deleted = sim_client.delete(f"/spoofing_zones/{zone_id}")
    assert deleted.status_code == 200
    assert deleted.json()["success"] is True

    cleared = sim_client.delete("/spoofing_zones")
    assert cleared.status_code == 200
    assert cleared.json()["success"] is True


def test_spoofing_zone_get_and_update_endpoints(sim_client):
    created = sim_client.post(
        "/spoofing_zones",
        json={
            "center": [22.0, 35.0, 4.0],
            "radius": 9.0,
            "spoof_type": "phantom",
            "phantom_count": 2,
        },
    )
    assert created.status_code == 200
    zone_id = created.json()["zone"]["id"]

    fetched = sim_client.get(f"/spoofing_zones/{zone_id}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == zone_id

    updated = sim_client.put(
        f"/spoofing_zones/{zone_id}",
        json={
            "spoof_type": "coordinate",
            "coordinate_vector": [4.0, -2.0, 0.0],
            "radius": 12.0,
            "active": True,
        },
    )
    assert updated.status_code == 200
    payload = updated.json()["zone"]
    assert payload["spoof_type"] == "coordinate"
    assert payload["radius"] == 12.0
    assert payload["coordinate_vector"] == [4.0, -2.0, 0.0]


def test_crypto_auth_and_v2v_channel_endpoints(sim_client):
    crypto_get = sim_client.get("/simulation/crypto_auth")
    assert crypto_get.status_code == 200
    assert "enabled" in crypto_get.json()

    crypto_set = sim_client.post(
        "/simulation/crypto_auth",
        json={"enabled": True, "algorithm": "hmac_sha256"},
    )
    assert crypto_set.status_code == 200
    assert crypto_set.json()["enabled"] is True

    v2v_get = sim_client.get("/simulation/v2v_channel")
    assert v2v_get.status_code == 200
    assert {"enabled", "links", "params"} <= v2v_get.json().keys()

    v2v_set = sim_client.post("/simulation/v2v_channel", json={"enabled": True})
    assert v2v_set.status_code == 200
    assert v2v_set.json()["success"] is True


def test_simulation_start_stop_reset_algorithm_and_results(sim_client, monkeypatch):
    async def _noop_loop(destination):  # pragma: no cover - async callback shape only
        return None

    monkeypatch.setattr(sim_api, "run_simulation_loop", _noop_loop)

    started = sim_client.post(
        "/simulation/start",
        json={
            "formation": "communication_aware",
            "path_algorithm": "astar",
            "default_obstacle_type": "low_jam",
            "destination": [20.0, 30.0, 10.0],
        },
    )
    assert started.status_code == 200
    assert started.json()["success"] is True

    updated = sim_client.post(
        "/simulation/algorithm",
        json={"formation": "communication_aware", "path_algorithm": "theta_star"},
    )
    assert updated.status_code == 200
    assert updated.json()["success"] is True
    assert updated.json()["changed"]["path_algorithm"] == "theta_star"

    state = sim_client.get("/simulation/state")
    assert state.status_code == 200
    assert "running" in state.json()

    results = sim_client.get("/simulation/results")
    assert results.status_code == 200
    assert "timestamp" in results.json()

    downloaded_json = sim_client.get("/simulation/results/download?format=json")
    assert downloaded_json.status_code == 200
    assert "download_timestamp" in downloaded_json.json()

    downloaded_csv = sim_client.get("/simulation/results/download?format=csv")
    assert downloaded_csv.status_code == 200
    assert downloaded_csv.headers["content-type"].startswith("text/csv")

    stopped = sim_client.post("/simulation/stop")
    assert stopped.status_code == 200
    assert stopped.json()["success"] is True

    reset = sim_client.post("/simulation/reset")
    assert reset.status_code == 200
    assert reset.json()["success"] is True


def test_simulate_step_moves_agent_toward_llm_target(sim_client):
    agent_id = _first_agent_id(sim_client)
    before = sim_client.get(f"/agents/{agent_id}").json()["position"]

    target_x = min(float(before[0]) + 1.5, float(X_RANGE[1]))
    target_y = min(float(before[1]) + 0.5, float(Y_RANGE[1]))
    target_z = min(float(before[2]) + 0.5, float(Z_RANGE[1]))

    commanded = sim_client.post(
        "/move_agent",
        json={"agent": agent_id, "x": target_x, "y": target_y, "z": target_z},
    )
    assert commanded.status_code == 200
    assert commanded.json()["success"] is True

    stepped = sim_client.post("/simulate_step")
    assert stepped.status_code == 200
    moved = stepped.json()["moved_agents"]
    assert any(item["agent_id"] == agent_id for item in moved)


def test_simulation_config_includes_dynamic_algorithm_fields(sim_client):
    response = sim_client.get("/simulation/config")
    assert response.status_code == 200
    payload = response.json()
    assert "path_algorithms" in payload
    assert "path_algorithm_labels" in payload
    assert "custom_path_algorithms" in payload
    assert "crypto_algorithms" in payload
    assert "crypto_algorithm_labels" in payload
    assert "custom_crypto_algorithms" in payload
    assert "comm_models" in payload
    assert "current" in payload
    assert "astar" in payload["path_algorithms"]
    assert "hmac_sha256" in payload["crypto_algorithms"]
    assert {"running", "formation", "path_algorithm", "crypto_algorithm"} <= payload[
        "current"
    ].keys()


def test_custom_algorithm_registry_endpoints(sim_client):
    listed = sim_client.get("/simulation/custom_algorithms")
    assert listed.status_code == 200
    assert "algorithms" in listed.json()

    create = sim_client.post(
        "/simulation/custom_algorithms",
        json={
            "name": "scripted_midpoint",
            "import_path": "tests.test_simulation_api:scripted_test_path",
            "description": "Scripted midpoint path",
        },
    )
    assert create.status_code == 200
    create_payload = create.json()
    assert create_payload["success"] is True
    assert "scripted_midpoint" in create_payload["path_algorithms"]

    config = sim_client.get("/simulation/config").json()
    assert "scripted_midpoint" in config["path_algorithms"]
    assert (
        config["path_algorithm_labels"]["scripted_midpoint"] == "Scripted midpoint path"
    )

    update_algo = sim_client.post(
        "/simulation/algorithm", json={"path_algorithm": "scripted_midpoint"}
    )
    assert update_algo.status_code == 200
    assert update_algo.json()["success"] is True

    remove = sim_client.delete("/simulation/custom_algorithms/scripted_midpoint")
    assert remove.status_code == 200
    assert remove.json()["success"] is True
    assert "scripted_midpoint" not in remove.json()["path_algorithms"]


def test_invalid_path_algorithm_is_rejected(sim_client):
    bad_start = sim_client.post(
        "/simulation/start",
        json={
            "formation": "communication_aware",
            "path_algorithm": "not_a_real_algorithm",
        },
    )
    assert bad_start.status_code == 400
    assert "Unknown algorithm" in bad_start.json()["detail"]


def test_custom_crypto_registry_endpoints(sim_client):
    listed = sim_client.get("/simulation/custom_crypto_algorithms")
    assert listed.status_code == 200
    assert "algorithms" in listed.json()

    create = sim_client.post(
        "/simulation/custom_crypto_algorithms",
        json={
            "name": "scripted_hmac",
            "sign_import_path": "tests.test_simulation_api:scripted_crypto_sign",
            "verify_import_path": "tests.test_simulation_api:scripted_crypto_verify",
            "description": "Scripted HMAC crypto",
        },
    )
    assert create.status_code == 200
    create_payload = create.json()
    assert create_payload["success"] is True
    assert "scripted_hmac" in create_payload["crypto_algorithms"]

    config = sim_client.get("/simulation/config").json()
    assert "scripted_hmac" in config["crypto_algorithms"]
    assert config["crypto_algorithm_labels"]["scripted_hmac"] == "Scripted HMAC crypto"

    update_crypto = sim_client.post(
        "/simulation/crypto_auth",
        json={"enabled": True, "algorithm": "scripted_hmac"},
    )
    assert update_crypto.status_code == 200
    assert update_crypto.json()["success"] is True
    assert update_crypto.json()["algorithm"] == "scripted_hmac"

    remove = sim_client.delete("/simulation/custom_crypto_algorithms/scripted_hmac")
    assert remove.status_code == 200
    assert remove.json()["success"] is True
    assert "scripted_hmac" not in remove.json()["crypto_algorithms"]
