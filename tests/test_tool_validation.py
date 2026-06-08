"""Smoke tests for per-tool argument schema validation."""

from __future__ import annotations

from swarm_squad_ep1.chat.json_utils import is_valid
from swarm_squad_ep1.chat.tools import (
    TOOL_ARG_SCHEMAS,
    TOOL_CALL_SCHEMA,
    build_ollama_tools,
    get_tool_registry_health,
    validate_tool_args,
)


def test_ollama_tools_shape():
    tools = build_ollama_tools()
    assert tools, "build_ollama_tools returned nothing"
    for t in tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert "name" in fn and "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_schemas_exist_for_every_tool():
    tools = build_ollama_tools()
    names = [t["function"]["name"] for t in tools]
    for n in names:
        assert n in TOOL_ARG_SCHEMAS, f"no schema for {n}"


def test_move_agent_valid():
    ok, err = validate_tool_args("move_agent", {"agent": "agent1", "x": 1.0, "y": 2.0})
    assert ok, err


def test_move_agent_missing_required():
    ok, err = validate_tool_args("move_agent", {"agent": "agent1"})
    assert not ok
    # Error should mention one of the missing required fields
    assert "x" in err or "y" in err


def test_move_agent_wrong_type():
    ok, err = validate_tool_args("move_agent", {"agent": "a1", "x": "nope", "y": 2.0})
    assert not ok
    assert "x" in err


def test_unknown_tool_rejected():
    ok, err = validate_tool_args("not_a_tool", {})
    assert not ok
    assert "Unknown tool" in err


def test_non_dict_args():
    ok, err = validate_tool_args("get_simulation_status", "oops")  # type: ignore[arg-type]
    assert not ok


def test_no_arg_tool_accepts_empty():
    ok, err = validate_tool_args("get_simulation_status", {})
    assert ok, err


def test_list_tools_tool_registered():
    """The meta tool 'list_tools' must be registered end-to-end."""
    from swarm_squad_ep1.chat.tools import TOOL_EXECUTORS

    assert "list_tools" in TOOL_ARG_SCHEMAS
    assert "list_tools" in TOOL_EXECUTORS


def test_tool_registry_has_no_schema_executor_drift():
    health = get_tool_registry_health()
    assert health["missing_executors"] == []
    assert health["extra_executors"] == []


def test_list_tools_accepts_empty_args():
    ok, err = validate_tool_args("list_tools", {})
    assert ok, err


def test_tool_call_envelope():
    good = {"tool": "move_agent", "args": {"agent": "a1", "x": 1.0, "y": 2.0}}
    assert is_valid(good, TOOL_CALL_SCHEMA)
    bad_tool = {"tool": "nope", "args": {}}
    assert not is_valid(bad_tool, TOOL_CALL_SCHEMA)
    missing = {"tool": "move_agent"}
    assert not is_valid(missing, TOOL_CALL_SCHEMA)


def test_enum_validation_rejects_invalid_jam_type():
    ok, err = validate_tool_args(
        "add_jamming_zone",
        {"x": 1.0, "y": 2.0, "jam_type": "super_jam"},
    )
    assert not ok
    assert "enum" in err


def test_enum_validation_rejects_invalid_crypto_algorithm():
    ok, err = validate_tool_args(
        "toggle_crypto_auth",
        {"enabled": True, "algorithm": "totally_custom"},
    )
    assert not ok
    assert "enum" in err


def test_start_simulation_accepts_bi_astar():
    ok, err = validate_tool_args(
        "start_simulation",
        {"formation": "communication_aware", "path_algorithm": "bi_astar"},
    )
    assert ok, err
