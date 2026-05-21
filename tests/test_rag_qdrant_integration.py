from __future__ import annotations

import numpy as np
import pytest

import swarm_squad_ep1.rag.qdrant as rag_qdrant

pytestmark = pytest.mark.integration


class _FakeEmbeddingModel:
    def encode(self, texts):
        # Deterministic vectors with correct dimensionality for Qdrant operations.
        base = np.array(
            [float(i) / rag_qdrant.VECTOR_DIM for i in range(rag_qdrant.VECTOR_DIM)],
            dtype=float,
        )
        return [base for _ in texts]


def test_rag_qdrant_end_to_end_integration(monkeypatch):
    assert rag_qdrant.test_connection() is True

    monkeypatch.setattr(rag_qdrant, "get_model", lambda: _FakeEmbeddingModel())

    rag_qdrant.clear_all()

    log_id = rag_qdrant.add_log(
        "integration message", source="user", message_type="command"
    )
    assert log_id is not None

    telemetry_id = rag_qdrant.add_telemetry(
        "agent_integration",
        (1.0, 2.0, 3.0),
        metadata={"jammed": False, "communication_quality": 0.97},
    )
    assert telemetry_id is not None

    logs = rag_qdrant.get_logs(limit=10)
    assert any(item["id"] == log_id for item in logs)

    history = rag_qdrant.get_telemetry_history("agent_integration", limit=5)
    assert history
    assert history[0]["position"] == (1.0, 2.0, 3.0)

    combined = rag_qdrant.search_all("integration", limit=5)
    assert {"telemetry", "logs"} <= combined.keys()
