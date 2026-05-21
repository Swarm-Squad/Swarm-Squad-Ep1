"""Smoke tests for the dependency-free JSON extractor/validator."""
from __future__ import annotations

import pytest

from swarm_squad_ep1.chat.json_utils import (
    ValidationError,
    extract_first_json,
    extract_json_candidates,
    is_valid,
    validate,
)


class TestExtract:
    def test_plain_object(self):
        assert extract_first_json('{"a": 1}') == {"a": 1}

    def test_markdown_fence(self):
        text = "Sure:\n```json\n{\"tool\": \"move_agent\", \"args\": {}}\n```\n"
        obj = extract_first_json(text)
        assert obj == {"tool": "move_agent", "args": {}}

    def test_trailing_comma(self):
        obj = extract_first_json('{"a": 1, "b": [1, 2, 3,],}')
        assert obj == {"a": 1, "b": [1, 2, 3]}

    def test_embedded_in_prose(self):
        text = "Thinking... here you go: {\"x\": 42} -- done."
        assert extract_first_json(text) == {"x": 42}

    def test_multiple_objects(self):
        text = 'first {"a":1} then {"b":2}'
        cands = extract_json_candidates(text)
        # Must find both, in order.
        assert {"a": 1} in cands and {"b": 2} in cands

    def test_string_with_braces(self):
        text = '{"note": "use {braces} inside"}'
        assert extract_first_json(text) == {"note": "use {braces} inside"}

    def test_none_on_empty(self):
        assert extract_first_json("") is None
        assert extract_first_json("no json here at all") is None


class TestValidate:
    def test_basic_types(self):
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        validate({"x": 1.5}, schema)
        validate({"x": 1}, schema)
        with pytest.raises(ValidationError):
            validate({"x": "nope"}, schema)

    def test_required(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        }
        with pytest.raises(ValidationError) as info:
            validate({"a": "hi"}, schema)
        assert "b" in str(info.value)

    def test_enum(self):
        schema = {"enum": ["los", "nlos_vehicle", "nlos_obstacle"]}
        validate("los", schema)
        with pytest.raises(ValidationError):
            validate("none", schema)

    def test_array_items(self):
        schema = {"type": "array", "items": {"type": "integer"}, "minItems": 1}
        validate([1, 2, 3], schema)
        with pytest.raises(ValidationError):
            validate([], schema)
        with pytest.raises(ValidationError):
            validate([1, "x"], schema)

    def test_numeric_bounds(self):
        schema = {"type": "number", "minimum": 0, "maximum": 1}
        validate(0.5, schema)
        with pytest.raises(ValidationError):
            validate(-0.1, schema)
        with pytest.raises(ValidationError):
            validate(1.5, schema)

    def test_additional_properties_false(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": False,
        }
        with pytest.raises(ValidationError):
            validate({"a": 1, "b": 2}, schema)

    def test_is_valid_helper(self):
        assert is_valid({"a": 1}, {"type": "object"})
        assert not is_valid("x", {"type": "integer"})
