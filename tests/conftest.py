from __future__ import annotations

from collections.abc import Generator

import httpx
import pytest
from fastapi.testclient import TestClient

from swarm_squad_ep1.config import CHAT_API_URL, SIMULATION_API_URL


def _best_effort_reset_sim_state(client: TestClient) -> None:
    """Reset mutable simulation globals so tests are deterministic."""
    for method, path in (
        ("post", "/simulation/stop"),
        ("post", "/simulation/reset"),
        ("post", "/llm_targets/clear_all"),
        ("delete", "/jamming_zones"),
        ("delete", "/spoofing_zones"),
    ):
        try:
            getattr(client, method)(path)
        except Exception:
            # Keep fixture resilient even if app startup is still warming.
            pass


@pytest.fixture
def integration_urls() -> dict[str, str]:
    return {"chat": CHAT_API_URL, "simulation": SIMULATION_API_URL}


@pytest.fixture
def live_service_health(integration_urls: dict[str, str]) -> dict[str, int]:
    """Fail-fast connectivity check for integration-required tests."""
    statuses: dict[str, int] = {}
    for name, url in integration_urls.items():
        response = httpx.get(url, timeout=5.0)
        statuses[name] = response.status_code
        assert response.status_code < 500, (
            f"live service {name!r} unhealthy at {url}: {response.status_code}"
        )
    return statuses


@pytest.fixture
def sim_client() -> Generator[TestClient, None, None]:
    from swarm_squad_ep1.simulation.api import app as simulation_app

    with TestClient(simulation_app) as client:
        _best_effort_reset_sim_state(client)
        yield client
        _best_effort_reset_sim_state(client)


@pytest.fixture
def chat_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    import swarm_squad_ep1.chat.app as chat_module

    # Keep startup deterministic in tests that don't require the async background loop.
    chat_module.app.state.disable_llm_target_loop = True

    monkeypatch.setattr(
        chat_module, "test_ollama_connection", lambda verbose=False: False
    )

    try:
        with TestClient(chat_module.app) as client:
            yield client
    finally:
        chat_module.app.state.disable_llm_target_loop = False
