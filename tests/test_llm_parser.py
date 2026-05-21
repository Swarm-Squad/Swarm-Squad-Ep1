"""Tests for the chat LLM prose/tool-call parser.

Focus: a prose answer containing unrelated JSON should NOT be misclassified
as a malformed tool call (which previously triggered a self-repair and
ultimately the dreaded 'I wasn't able to act on that' fallback).
"""

from __future__ import annotations

from swarm_squad_ep1.chat.llm import LLMAgent


def test_prose_with_unrelated_json_is_not_a_tool_call_attempt():
    agent = LLMAgent()
    content = (
        'Here are some interesting stats: {"agents": 3, "jammed": 0}. '
        "You can also move them by name."
    )
    tc, err = agent._extract_prompt_json_tool_call(content)
    assert tc is None
    # Must not signal a repair: the content is clearly prose.
    assert err is None


def test_plain_english_answer_is_not_a_tool_call_attempt():
    agent = LLMAgent()
    content = "I can list, move, and add agents. Ask me to do any of those."
    tc, err = agent._extract_prompt_json_tool_call(content)
    assert tc is None
    assert err is None


def test_valid_tool_call_json_parses():
    agent = LLMAgent()
    content = '{"tool": "get_simulation_status", "args": {}}'
    tc, err = agent._extract_prompt_json_tool_call(content)
    assert tc == {"tool": "get_simulation_status", "args": {}}
    assert err is None


def test_unknown_tool_triggers_repair_signal():
    agent = LLMAgent()
    content = '{"tool": "does_not_exist", "args": {}}'
    tc, err = agent._extract_prompt_json_tool_call(content)
    assert tc is None
    # Now the error *should* mention the unknown tool so a repair round
    # can give the model a helpful hint.
    assert err is not None
    assert "does_not_exist" in err


def test_clean_text_strips_only_pure_json_blobs():
    agent = LLMAgent()
    assert agent._as_clean_text("") == ""
    assert agent._as_clean_text('{"tool": "x"}') == ""
    assert agent._as_clean_text('hello {"x": 1}') == 'hello {"x": 1}'
