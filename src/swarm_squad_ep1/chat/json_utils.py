"""
Utilities for extracting and validating JSON produced by LLMs.

Small local models (e.g. ``llama3.2:3b``) routinely emit:
  - JSON wrapped in markdown fences
  - JSON preceded/followed by free-text reasoning
  - Multiple JSON objects in one reply
  - Trailing commas
  - Near-JSON with minor typos

This module provides:
  - ``extract_json_candidates(text)``: a balanced-brace scanner that returns
    every top-level ``{...}`` and ``[...]`` that parses, not just the first.
  - ``validate(obj, schema)``: a dependency-free JSON Schema subset validator
    covering ``type``, ``required``, ``enum``, ``minimum``/``maximum``,
    ``minItems``/``maxItems``, ``properties``, ``items``, ``oneOf``, ``anyOf``.

No external deps (avoids adding ``jsonschema`` just for this).
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_fences(text: str) -> list[str]:
    """Return contents of every ``` fenced block + the remaining text."""
    chunks: list[str] = []
    fenced = _FENCE_RE.findall(text)
    chunks.extend(fenced)
    chunks.append(_FENCE_RE.sub("", text))
    return chunks


def _fix_trailing_commas(s: str) -> str:
    """Remove trailing commas that Python's json module refuses."""
    return re.sub(r",\s*(\]|\})", r"\1", s)


def _walk_balanced(text: str) -> Iterable[str]:
    """Yield every balanced ``{...}`` or ``[...]`` substring in ``text``.

    Respects double-quoted strings (including escapes) to avoid treating
    braces inside strings as structural.
    """
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        open_ch = ch
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        j = i
        in_str = False
        esc = False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        yield text[i : j + 1]
                        break
            j += 1
        i = max(j, i + 1)


def extract_json_candidates(text: str) -> list[Any]:
    """Return every JSON object/array that can be parsed out of ``text``.

    Order matches occurrence order (fenced blocks first, then free text).
    """
    if not text:
        return []

    found: list[Any] = []
    seen_repr: set[str] = set()

    for chunk in _strip_fences(text):
        # First try the chunk as a whole.
        for candidate in [chunk.strip(), _fix_trailing_commas(chunk).strip()]:
            if candidate and candidate[0] in "{[":
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                else:
                    key = repr(obj)
                    if key not in seen_repr:
                        seen_repr.add(key)
                        found.append(obj)
                    break

        # Then scan for every balanced sub-object.
        for sub in _walk_balanced(chunk):
            for attempt in (sub, _fix_trailing_commas(sub)):
                try:
                    obj = json.loads(attempt)
                except json.JSONDecodeError:
                    continue
                key = repr(obj)
                if key in seen_repr:
                    break
                seen_repr.add(key)
                found.append(obj)
                break

    return found


def extract_first_json(text: str) -> Any | None:
    """Convenience: return the first parseable JSON in ``text`` or None."""
    cands = extract_json_candidates(text)
    return cands[0] if cands else None


# --------------------------------------------------------------------------
# Validation (tiny JSON Schema subset)
# --------------------------------------------------------------------------

_TYPE_MAP = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": type(None),
}


class ValidationError(Exception):
    pass


def _type_matches(obj, t: str) -> bool:
    if t == "null":
        return obj is None
    if t == "integer":
        return isinstance(obj, int) and not isinstance(obj, bool)
    if t == "number":
        return isinstance(obj, (int, float)) and not isinstance(obj, bool)
    if t == "boolean":
        return isinstance(obj, bool)
    if t not in _TYPE_MAP:
        return False
    return isinstance(obj, _TYPE_MAP[t])


def validate(obj: Any, schema: dict, _path: str = "$") -> None:
    """Validate ``obj`` against a subset of JSON Schema.

    Raises ``ValidationError`` on mismatch. Supports:
      type, enum, const, required, properties, additionalProperties,
      items, minItems, maxItems, minimum, maximum, oneOf, anyOf, allOf.
    """
    if not isinstance(schema, dict):
        return

    if "const" in schema and obj != schema["const"]:
        raise ValidationError(
            f"{_path}: expected const {schema['const']!r}, got {obj!r}"
        )

    if "enum" in schema and obj not in schema["enum"]:
        raise ValidationError(f"{_path}: {obj!r} not in enum {schema['enum']!r}")

    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_matches(obj, ti) for ti in types):
            raise ValidationError(
                f"{_path}: expected type {t}, got {type(obj).__name__}"
            )

    # Numeric bounds
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            raise ValidationError(f"{_path}: {obj} < minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            raise ValidationError(f"{_path}: {obj} > maximum {schema['maximum']}")

    # Arrays
    if isinstance(obj, list):
        if "minItems" in schema and len(obj) < schema["minItems"]:
            raise ValidationError(
                f"{_path}: array shorter than minItems {schema['minItems']}"
            )
        if "maxItems" in schema and len(obj) > schema["maxItems"]:
            raise ValidationError(
                f"{_path}: array longer than maxItems {schema['maxItems']}"
            )
        items = schema.get("items")
        if items:
            if isinstance(items, dict):
                for idx, elem in enumerate(obj):
                    validate(elem, items, f"{_path}[{idx}]")
            elif isinstance(items, list):
                for idx, (elem, sub) in enumerate(zip(obj, items)):
                    validate(elem, sub, f"{_path}[{idx}]")

    # Objects
    if isinstance(obj, dict):
        for key in schema.get("required", []):
            if key not in obj:
                raise ValidationError(f"{_path}: missing required property '{key}'")
        props = schema.get("properties", {})
        for k, sub in props.items():
            if k in obj:
                validate(obj[k], sub, f"{_path}.{k}")
        addl = schema.get("additionalProperties", True)
        if addl is False:
            extras = [k for k in obj.keys() if k not in props]
            if extras:
                raise ValidationError(f"{_path}: unexpected properties {extras}")
        elif isinstance(addl, dict):
            for k, v in obj.items():
                if k not in props:
                    validate(v, addl, f"{_path}.{k}")

    # Composition
    for key in ("oneOf", "anyOf"):
        if key in schema:
            errs = []
            ok = 0
            for sub in schema[key]:
                try:
                    validate(obj, sub, _path)
                    ok += 1
                except ValidationError as e:
                    errs.append(str(e))
            if key == "anyOf" and ok == 0:
                raise ValidationError(f"{_path}: failed anyOf ({'; '.join(errs)})")
            if key == "oneOf" and ok != 1:
                raise ValidationError(
                    f"{_path}: oneOf matched {ok} schemas (expected 1)"
                )
    if "allOf" in schema:
        for sub in schema["allOf"]:
            validate(obj, sub, _path)


def is_valid(obj: Any, schema: dict) -> bool:
    try:
        validate(obj, schema)
        return True
    except ValidationError:
        return False
