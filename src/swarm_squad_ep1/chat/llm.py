"""
LLM Agent with tool calling for Swarm Squad Ep1.

Robust on small local models (e.g. llama3.2:3b):

  1. Send the prompt to Ollama with its native ``tools`` array and a strict
     system message. If the model supports native tool-calling it returns
     structured ``message.tool_calls``.
  2. If not, parse every balanced JSON object in the reply and pick the
     first one that matches the tool-call schema.
  3. If no call parses, try ONE self-repair round with the validation
     error fed back to the model.
  4. Unknown/malformed output is NEVER surfaced as a "final answer" any
     more — we reply with a fixed "I wasn't able to act on that."
"""
import json
from typing import Any, Optional

from ..config import (
    LLM_MODEL,
    MISSION_END,
    SIMULATION_API_URL,
    X_RANGE,
    Y_RANGE,
    Z_RANGE,
    async_chat_with_retry,
)
from ..rag import add_log
from .json_utils import extract_json_candidates, is_valid, validate, ValidationError
from .tools import (
    TOOL_ARG_SCHEMAS,
    TOOL_CALL_SCHEMA,
    build_ollama_tools,
    execute_tool,
    get_tool_schemas_text,
    validate_tool_args,
)

MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = f"""You are an expert assistant for a 3D multi-vehicle swarm simulation.

SIMULATION CONTEXT:
- Vehicles navigate from start positions to destination {MISSION_END}
- Map bounds: X={X_RANGE}, Y={Y_RANGE}, Z={Z_RANGE}
- Vehicles use communication-aware formation control
- Jamming zones degrade communication; spoofing attacks inject fake data
- Cryptographic authentication (HMAC-SHA256, ChaCha20, AES-256-CTR) can counter spoofing

TOOL-CALL PROTOCOL (STRICT):
- If a tool is needed, output EXACTLY one JSON object and NOTHING ELSE:
  {{"tool": "tool_name", "args": {{"param1": value1}}}}
- Numeric parameters MUST be JSON numbers, not strings.
- No prose, no markdown fences, no "here is the JSON" -- just the object.
- Only ONE tool per response. Wait for the tool result before calling another.

AVAILABLE TOOLS:
{get_tool_schemas_text()}

CRITICAL RULES FOR ZONE OPERATIONS:
- To DELETE a specific zone: FIRST call list_jamming_zones or list_spoofing_zones to get the zone ID, THEN call delete_jamming_zone or delete_spoofing_zone with that ID.
- To DELETE ALL zones: call clear_all_jamming_zones or clear_all_spoofing_zones directly.
- To CREATE a zone: call add_jamming_zone or add_spoofing_zone with x, y coordinates.
- Zone IDs are auto-generated (e.g. "zone_a1b2" or "zone_1"). Never guess IDs -- always list first.
- When asked about status, ALWAYS call get_agent_status or get_simulation_status first.

META QUESTIONS:
- When the user asks "what can you do", "list tools", "list capabilities", or
  "show available commands", call the list_tools tool -- do NOT free-form a list.

FINAL ANSWERS (when no tool is needed):
- 2-4 plain sentences, no JSON, no markdown.
"""


FAILED_ANSWER = (
    "I wasn't able to act on that. Could you rephrase with specific agent "
    "IDs, zone IDs, or coordinates?"
)


class LLMAgent:
    """LLM agent using Ollama native tool calling with prompt-JSON fallback."""

    def __init__(self):
        self.model = LLM_MODEL
        self.api_url = SIMULATION_API_URL
        self._ollama_tools = build_ollama_tools()

    async def answer(self, user_query: str) -> dict:
        """Process a user message with a tool-calling loop."""
        print(f"\n{'='*60}")
        print(f"[LLM] Processing: {user_query}")

        add_log(user_query, source="user", message_type="command")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        tool_calls_made: list[dict] = []
        answer: Optional[str] = None

        for round_num in range(MAX_TOOL_ROUNDS + 1):
            tool_call, assistant_content, raw_msg = await self._request_tool_call(messages)

            # Remember the assistant turn verbatim for history.
            messages.append({
                "role": "assistant",
                "content": assistant_content or "",
                **({"tool_calls": raw_msg["tool_calls"]} if raw_msg.get("tool_calls") else {}),
            })

            if tool_call is None:
                # No tool was requested. Prefer a clean prose answer; if
                # the model returned nothing useful, auto-invoke
                # ``list_tools`` so the user still learns what's possible
                # instead of seeing a dead-end "I wasn't able to act on
                # that" message.
                clean = self._as_clean_text(assistant_content)
                if clean:
                    answer = clean
                else:
                    fallback = await execute_tool("list_tools", {})
                    tool_calls_made.append({
                        "tool": "list_tools",
                        "args": {},
                        "result_summary": f"auto-invoked ({fallback.get('count', 0)} tools)",
                    })
                    answer = self._format_tools_fallback(fallback)
                break

            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args") or {}
            print(f"[LLM] Round {round_num} tool call: {tool_name}({tool_args})")

            result = await execute_tool(tool_name, tool_args)
            tool_calls_made.append({
                "tool": tool_name,
                "args": tool_args,
                "result_summary": result.get("message", result.get("success", ""))
            })

            result_text = json.dumps(result, indent=2, default=str)
            messages.append({
                "role": "user",
                "content": (
                    f"Tool result for {tool_name}:\n```json\n{result_text}\n```\n\n"
                    "If you now have enough information, answer the user in 2-4 "
                    "plain sentences (NO JSON). Otherwise emit exactly one tool-call JSON."
                ),
            })
        else:
            answer = FAILED_ANSWER

        answer = answer or FAILED_ANSWER
        add_log(answer, source="llm", message_type="response")

        if tool_calls_made:
            tools_summary = " | ".join(f"[{tc['tool']}]" for tc in tool_calls_made)
            print(f"[LLM] Tools used: {tools_summary}")
        print(f"[LLM] Answer ready ({len(answer)} chars)")
        print(f"{'='*60}\n")

        return {"response": answer, "tool_calls": tool_calls_made}

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    async def _request_tool_call(
        self,
        messages: list[dict],
    ) -> tuple[Optional[dict], str, dict]:
        """Single round: ask the model, return (tool_call_or_None, content, raw).

        Will perform one self-repair retry on schema-violating JSON.
        """
        # 1st attempt: native tool-calling
        response = await async_chat_with_retry(
            self.model, messages=messages, tools=self._ollama_tools,
            options={"temperature": 0.0},
        )
        if not response:
            return None, FAILED_ANSWER, {}
        raw_msg = response.get("message", {}) or {}
        content = (raw_msg.get("content") or "").strip()

        # Try native tool-calling first. `err_native` describes the defect
        # (unknown tool name, bad args) so self-repair can quote it back.
        tool_call, err_native = self._extract_native_tool_call(raw_msg)
        err = err_native
        if tool_call is None:
            prompt_tc, err_prompt = self._extract_prompt_json_tool_call(content)
            if prompt_tc is not None:
                tool_call = prompt_tc
                err = None
            else:
                err = err or err_prompt

        if tool_call is None and err:
            # Self-repair: one more round asking the model to fix the JSON.
            repair_user = {
                "role": "user",
                "content": (
                    "Your previous response was not a valid tool call. "
                    f"Error: {err}. Respond with ONLY one JSON object of the "
                    "form {\"tool\": \"<name>\", \"args\": {...}} and nothing else. "
                    "If you don't know which tool to use, call "
                    '{"tool": "list_tools", "args": {}}.'
                ),
            }
            repaired = await async_chat_with_retry(
                self.model,
                messages=[*messages, {"role": "assistant", "content": content}, repair_user],
                tools=self._ollama_tools,
                format=TOOL_CALL_SCHEMA,
                options={"temperature": 0.0},
            )
            if repaired:
                raw_msg2 = repaired.get("message", {}) or {}
                content2 = (raw_msg2.get("content") or "").strip()
                tc, _ = self._extract_native_tool_call(raw_msg2)
                if tc is None:
                    tc, _ = self._extract_prompt_json_tool_call(content2)
                if tc is not None:
                    return tc, content2 or content, raw_msg2

        return tool_call, content, raw_msg

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _extract_native_tool_call(
        self, raw_msg: dict,
    ) -> tuple[Optional[dict], Optional[str]]:
        """Extract a tool call from Ollama's native ``tool_calls`` field.

        Returns ``(tool_call, error_message)``. ``error_message`` is set
        when the model DID emit a native call but its name or args are
        invalid, so the caller can trigger a self-repair round with the
        specific error.
        """
        calls = raw_msg.get("tool_calls") or []
        if not calls:
            return None, None
        first = calls[0]
        fn = first.get("function", first) or {}
        name = fn.get("name")
        args = fn.get("arguments") or fn.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not name:
            return None, "native tool call missing 'name'"
        if name not in TOOL_ARG_SCHEMAS:
            return None, f"unknown tool '{name}'"
        ok, err = validate_tool_args(name, args)
        if not ok:
            return None, err
        return {"tool": name, "args": args}, None

    def _extract_prompt_json_tool_call(
        self, content: str,
    ) -> tuple[Optional[dict], Optional[str]]:
        """Scan free-text for a tool-call JSON and schema-validate it.

        Returns (tool_call, error_for_repair).  ``error_for_repair`` is
        ``None`` when the content clearly isn't a tool-call attempt
        (plain English answer or prose with unrelated JSON), so callers
        know not to trigger a repair round.
        """
        if not content:
            return None, None

        candidates = extract_json_candidates(content)
        if not candidates:
            return None, None

        # Only consider candidates that actually LOOK like a tool-call
        # attempt (dict with a "tool" key). Random JSON blobs inside a
        # prose answer should not trigger a repair round, because the
        # model was legitimately answering, not calling a tool.
        tool_like = [
            c for c in candidates if isinstance(c, dict) and "tool" in c
        ]
        if not tool_like:
            return None, None

        last_err: Optional[str] = None
        for cand in tool_like:
            if not is_valid(cand, TOOL_CALL_SCHEMA):
                # Closest guess at what's wrong, for a self-repair prompt.
                name = cand.get("tool")
                if name and name not in TOOL_ARG_SCHEMAS:
                    last_err = f"unknown tool '{name}'"
                else:
                    last_err = last_err or f"tool-call wrapper invalid: {cand!r}"
                continue
            tc = {"tool": cand["tool"], "args": cand.get("args") or {}}
            ok, err = validate_tool_args(tc["tool"], tc["args"])
            if ok:
                return tc, None
            last_err = err

        return None, last_err or "response was JSON but not a valid tool call"

    @staticmethod
    def _format_tools_fallback(result: dict) -> str:
        """Human-friendly rendering of the list_tools catalog."""
        tools = result.get("tools") or []
        if not tools:
            return FAILED_ANSWER
        lines = [
            "I wasn't sure which tool to use, so here is what I can do:",
        ]
        for t in tools[:24]:
            name = t.get("name", "?")
            desc = (t.get("description") or "").split(".")[0]
            lines.append(f"- `{name}` — {desc}")
        lines.append("")
        lines.append("Tell me which one you want, or include the parameters directly "
                     "(e.g. 'move agent1 to 10, 20').")
        return "\n".join(lines)

    @staticmethod
    def _as_clean_text(content: str) -> str:
        """Strip any stray JSON from a model response that's mostly prose."""
        if not content:
            return ""
        # Heuristic: if the entire content is JSON, it's a malformed tool call.
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return ""
        return content


# Global instance
_agent: Optional[LLMAgent] = None


def get_agent() -> LLMAgent:
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent


async def answer_question(query: str) -> dict:
    agent = get_agent()
    return await agent.answer(query)
