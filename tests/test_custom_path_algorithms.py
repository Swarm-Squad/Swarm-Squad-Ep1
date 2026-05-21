from __future__ import annotations

import numpy as np

from swarm_squad_ep1.algo.custom_path_algorithms import clear_custom_path_algorithms
from swarm_squad_ep1.algo.path_planning import (
    PathPlanner,
    get_available_path_algorithms,
    list_registered_path_algorithms,
    register_path_algorithm,
    unregister_path_algorithm,
)


def midpoint_custom_path(start, goal, jamming_zones, **kwargs):
    midpoint = (start + goal) / 2
    return [start, midpoint, goal]


def test_register_and_use_custom_path_algorithm():
    clear_custom_path_algorithms()
    try:
        created = register_path_algorithm(
            name="midpoint_custom",
            import_path="tests.test_custom_path_algorithms:midpoint_custom_path",
            description="Midpoint demo algorithm",
        )
        assert created["name"] == "midpoint_custom"

        available = get_available_path_algorithms()
        assert "midpoint_custom" in available
        assert any(
            item["name"] == "midpoint_custom"
            for item in list_registered_path_algorithms()
        )

        planner = PathPlanner(algorithm="midpoint_custom")
        path = planner.plan_path(
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 0.0]),
            [],
            agent_id="agent_test",
        )
        assert path is not None
        assert len(path) >= 2
        assert np.allclose(path[0], np.array([0.0, 0.0, 0.0]))
        assert np.allclose(path[-1], np.array([10.0, 0.0, 0.0]))
    finally:
        unregister_path_algorithm("midpoint_custom")
        clear_custom_path_algorithms()


def test_unregister_custom_path_algorithm():
    clear_custom_path_algorithms()
    register_path_algorithm(
        name="remove_me",
        import_path="tests.test_custom_path_algorithms:midpoint_custom_path",
    )
    removed = unregister_path_algorithm("remove_me")
    assert removed is not None
    assert removed["name"] == "remove_me"
    assert "remove_me" not in get_available_path_algorithms()
