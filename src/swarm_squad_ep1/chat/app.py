"""
Chat API - FastAPI app for the Swarm Squad GUI/chat interface.
"""

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from swarm_squad_ep1.chat.llm import answer_question
from swarm_squad_ep1.chat.tools import (
    TOOL_EXECUTORS,
    TOOL_SCHEMAS,
    move_agent,
)
from swarm_squad_ep1.chat.tools import (
    list_tools as list_tools_catalog,
)
from swarm_squad_ep1.config import (
    EDU_BEGINNER_MODE,
    EDU_DEFAULT_PRESET,
    ENABLE_DEBUG_TELEMETRY,
    LLM_MODEL,
    MISSION_END,
    SIMULATION_API_URL,
    X_RANGE,
    Y_RANGE,
    Z_RANGE,
    async_chat_with_retry,
    test_ollama_connection,
)
from swarm_squad_ep1.rag import get_all_telemetry, get_logs
from swarm_squad_ep1.research.scenarios import (
    build_education_scenario,
    get_education_presets,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook for chat startup/shutdown tasks."""
    await _run_chat_startup(app)
    try:
        yield
    finally:
        await _run_chat_shutdown()


# Create FastAPI app
app = FastAPI(title="Swarm Squad Ep1 Chat API", lifespan=lifespan)

# Background task for LLM target processing
_llm_target_task = None

# Last user chat interaction — shown in the LLM Context panel "Last LLM Prompt" section
# so users can confirm their message reached the model even before agents are jammed.
_last_chat_prompt: dict = {}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates/static paths from the package namespace only.
# This keeps runtime behavior consistent between editable installs and wheels.
PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_STATIC_DIR = PACKAGE_DIR / "gui" / "static"
STATIC_DIR = PACKAGE_STATIC_DIR
if not STATIC_DIR.exists():
    raise RuntimeError(f"Missing packaged GUI assets at: {STATIC_DIR}")
templates = Jinja2Templates(directory=str(STATIC_DIR))


def _default_status_payload() -> dict:
    """Stable fallback used when simulation API is unavailable."""
    return {
        "running": False,
        "agent_count": 0,
        "boundaries": {
            "x_range": list(X_RANGE),
            "y_range": list(Y_RANGE),
            "z_range": list(Z_RANGE),
            "mission_end": list(MISSION_END),
        },
    }


def _default_attack_metrics_payload(error: str | None = None) -> dict:
    """Stable fallback for attack-metrics polling when sim API is unavailable."""
    payload = {
        "crypto_enabled": False,
        "crypto_algorithm": "-",
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "detection_rate": 0.0,
        "false_positive_rate": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "active_attacks_by_type": {},
        "source": "chat_fallback",
        "timestamp": datetime.now().isoformat(),
    }
    if error:
        payload["error"] = error
    return payload


async def _fetch_simulation_status(timeout: float = 5.0) -> dict:
    """Fetch simulation status and enforce expected bootstrap keys."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{SIMULATION_API_URL}/status", timeout=timeout)
        response.raise_for_status()
        payload = response.json()

    boundaries = payload.get("boundaries")
    required_boundary_keys = {"x_range", "y_range", "z_range", "mission_end"}
    if not isinstance(boundaries, dict) or not required_boundary_keys.issubset(
        boundaries
    ):
        raise ValueError(
            "Simulation /status payload missing required boundaries contract"
        )
    return payload


def _read_proxy_payload(response: httpx.Response):
    """Decode proxied response payload, falling back to text wrappers."""
    try:
        return response.json()
    except Exception:
        text = (response.text or "").strip()
        return {"detail": text or "Upstream response was not JSON"}


async def _proxy_sim_json(
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    params: dict | None = None,
    timeout: float = 5.0,
    on_exception_payload: dict | None = None,
) -> JSONResponse:
    """Proxy a JSON endpoint to the simulation API preserving status codes."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                f"{SIMULATION_API_URL}{path}",
                json=json_body,
                params=params,
                timeout=timeout,
            )
        return JSONResponse(
            status_code=response.status_code, content=_read_proxy_payload(response)
        )
    except Exception as exc:
        payload = {"success": False, "error": str(exc)}
        if on_exception_payload:
            payload.update(on_exception_payload)
        return JSONResponse(status_code=503, content=payload)


async def llm_target_loop():
    """
    Background loop that moves agents toward their LLM targets when simulation is stopped.

    This enables the "move agent1 to 5, 5" chat commands to actually move agents
    even when the main simulation loop is not running.
    """
    while True:
        try:
            async with httpx.AsyncClient() as client:
                # Check if simulation is running
                state_response = await client.get(
                    f"{SIMULATION_API_URL}/simulation/state", timeout=2.0
                )

                if state_response.status_code == 200:
                    state_data = state_response.json()
                    sim_running = state_data.get("running", False)

                    if not sim_running:
                        # Simulation is stopped - check if any agents have LLM targets
                        agents_response = await client.get(
                            f"{SIMULATION_API_URL}/agents", timeout=2.0
                        )

                        if agents_response.status_code == 200:
                            agents_data = agents_response.json().get("agents", {})
                            has_targets = any(
                                agent.get("llm_target") is not None
                                for agent in agents_data.values()
                            )

                            if has_targets:
                                # Process one simulation step to move agents
                                await client.post(
                                    f"{SIMULATION_API_URL}/simulate_step", timeout=2.0
                                )
        except Exception:
            # Silently ignore errors - simulation API might not be ready
            pass

        await asyncio.sleep(0.1)  # 100ms update rate


def _mount_static_once(app: FastAPI) -> None:
    """Mount packaged static assets once per process."""
    already_mounted = any(
        getattr(route, "path", "") == "/static" for route in app.routes
    )
    if not already_mounted and STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


async def _run_chat_startup(app: FastAPI):
    """Run chat service startup actions."""
    global _llm_target_task

    print("[CHAT] Starting Chat API...")

    # Ensure static routes are available for the dashboard shell and assets.
    _mount_static_once(app)

    # Check LLM connectivity up front for clearer operator feedback.
    if test_ollama_connection(verbose=True):
        print("[CHAT] LLM connected")
    else:
        print("[CHAT] LLM not available - will retry on requests")

    # Start background task for user-issued target stepping unless tests disable it.
    if getattr(app.state, "disable_llm_target_loop", False):
        return
    if _llm_target_task is None or _llm_target_task.done():
        _llm_target_task = asyncio.create_task(llm_target_loop())
        print("[CHAT] LLM target processing loop started")


async def _run_chat_shutdown():
    """Stop background tasks on process shutdown."""
    global _llm_target_task
    if _llm_target_task is None:
        return
    _llm_target_task.cancel()
    with suppress(asyncio.CancelledError):
        await _llm_target_task
    _llm_target_task = None


async def _warmup_model():
    """Send a minimal request to preload the model into GPU memory."""
    try:
        # Use a generous 5-minute timeout: GPU cold-start (model pull to VRAM) can be slow,
        # especially over an HPC SSH tunnel. Once loaded, keep_alive=-1 keeps it resident.
        response = await async_chat_with_retry(
            LLM_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_retries=1,
            timeout_secs=300,
        )
        if response:
            print("[CHAT] Model preloaded into GPU")
        else:
            print("[CHAT] Model preload failed - will load on first request")
    except Exception:
        print("[CHAT] Model preload failed - will load on first request")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve main dashboard HTML."""
    try:
        return templates.TemplateResponse(request, "index.html")
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body>Error loading dashboard: {e}</body></html>",
            status_code=500,
        )


@app.get("/health")
async def health_check():
    """Check system health."""
    # Check simulation API
    sim_status = "offline"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SIMULATION_API_URL}/", timeout=2.0)
            if response.status_code == 200:
                sim_status = "online"
    except Exception:
        pass

    # Check LLM
    llm_status = "ready" if test_ollama_connection(verbose=False) else "unavailable"

    return {
        "chat_api": "online",
        "simulation_api": sim_status,
        "llm": llm_status,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/status")
async def proxy_status():
    """Proxy simulation status for frontend bootstrap contract."""
    fallback = _default_status_payload()
    try:
        payload = await _fetch_simulation_status(timeout=5.0)
        payload.setdefault("timestamp", datetime.now().isoformat())
        payload.setdefault("source", "simulation_api")
        return payload
    except Exception as exc:
        fallback.update(
            {
                "source": "chat_fallback",
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }
        )
        return fallback


@app.get("/tools")
async def list_tools_endpoint():
    """Return a compact catalog of every tool the chat agent can invoke.

    Used by the frontend to render a helpful welcome message and a
    discoverable "what can you do" panel.
    """
    catalog = await list_tools_catalog()
    tools = catalog.get("tools", [])
    available_names = {tool.get("name") for tool in tools}
    registry_health = catalog.get("registry_health", {})

    # Safety net: surface runtime registry drift right on /tools for easier debugging.
    schema_names = {t["name"] for t in TOOL_SCHEMAS}
    executor_names = set(TOOL_EXECUTORS.keys())
    runtime_missing_executors = sorted(schema_names - executor_names)
    runtime_extra_executors = sorted(executor_names - schema_names)
    if runtime_missing_executors or runtime_extra_executors:
        registry_health = {
            **registry_health,
            "missing_executors": runtime_missing_executors,
            "extra_executors": runtime_extra_executors,
        }

    # Short stable categorization for UI grouping
    categories = {
        "agents": [
            "move_agent",
            "get_agent_status",
            "add_agent",
            "remove_agent",
            "get_telemetry_history",
        ],
        "simulation": [
            "start_simulation",
            "stop_simulation",
            "reset_simulation",
            "get_simulation_status",
            "set_formation",
        ],
        "jamming": [
            "add_jamming_zone",
            "delete_jamming_zone",
            "list_jamming_zones",
            "clear_all_jamming_zones",
        ],
        "spoofing": [
            "add_spoofing_zone",
            "delete_spoofing_zone",
            "list_spoofing_zones",
            "clear_all_spoofing_zones",
        ],
        "security": ["toggle_crypto_auth", "get_protocol_stats"],
        "channel": ["toggle_v2v_channel", "get_v2v_channel_status"],
        "meta": ["list_tools"],
    }
    filtered_categories = {
        group: [name for name in names if name in available_names]
        for group, names in categories.items()
    }
    categorized = {name for names in filtered_categories.values() for name in names}
    uncategorized = sorted(
        name
        for name in available_names
        if isinstance(name, str) and name not in categorized
    )
    if uncategorized:
        filtered_categories["other"] = uncategorized

    return {
        "count": len(tools),
        "tools": tools,
        "categories": filtered_categories,
        "registry_health": registry_health,
    }


@app.get("/app_config")
async def app_config():
    """Expose frontend configuration for educational mode and debug gating."""
    fallback = _default_status_payload()
    simulation_status = fallback
    simulation_online = False
    try:
        simulation_status = await _fetch_simulation_status(timeout=3.0)
        simulation_online = True
    except Exception as exc:
        simulation_status = {
            **fallback,
            "source": "chat_fallback",
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }

    return {
        "beginner_mode": EDU_BEGINNER_MODE,
        "default_preset": EDU_DEFAULT_PRESET,
        "enable_debug_telemetry": ENABLE_DEBUG_TELEMETRY,
        "education_presets": get_education_presets(),
        "simulation_online": simulation_online,
        "simulation": simulation_status,
    }


@app.get("/education/presets")
async def education_presets():
    """Return classroom preset metadata."""
    return {
        "presets": get_education_presets(),
        "default_preset": EDU_DEFAULT_PRESET,
    }


@app.post("/education/load_preset")
async def load_education_preset(request: Request):
    """Apply a beginner preset to the simulation backend."""
    data = await request.json()
    preset = data.get("preset", EDU_DEFAULT_PRESET)
    seed = int(data.get("seed", 0))
    try:
        scenario = build_education_scenario(preset, seed=seed)
    except ValueError as exc:
        return JSONResponse(
            status_code=400, content={"success": False, "error": str(exc)}
        )

    async with httpx.AsyncClient() as client:
        await client.post(f"{SIMULATION_API_URL}/simulation/stop", timeout=5.0)
        await client.post(f"{SIMULATION_API_URL}/simulation/reset", timeout=8.0)
        await client.delete(f"{SIMULATION_API_URL}/jamming_zones", timeout=5.0)
        await client.delete(f"{SIMULATION_API_URL}/spoofing_zones", timeout=5.0)
        await client.post(
            f"{SIMULATION_API_URL}/simulation/algorithm",
            json={
                "formation": scenario.formation_type,
                "path_algorithm": scenario.path_algorithm,
            },
            timeout=5.0,
        )
        await client.post(
            f"{SIMULATION_API_URL}/simulation/crypto_auth",
            json={
                "enabled": scenario.crypto_enabled,
                "algorithm": scenario.crypto_algorithm,
            },
            timeout=5.0,
        )
        await client.post(
            f"{SIMULATION_API_URL}/simulation/llm_assistance",
            json={"enabled": scenario.llm_assistance_enabled},
            timeout=5.0,
        )

        for zone in scenario.jamming_zones:
            await client.post(
                f"{SIMULATION_API_URL}/jamming_zones",
                json={
                    "center": list(zone.center),
                    "radius": zone.radius,
                    "obstacle_type": zone.obstacle_type,
                    "active": True,
                },
                timeout=5.0,
            )

        for zone in scenario.spoofing_zones:
            await client.post(
                f"{SIMULATION_API_URL}/spoofing_zones",
                json={
                    "center": list(zone.center),
                    "radius": zone.radius,
                    "spoof_type": zone.spoof_type,
                    "phantom_count": zone.phantom_count,
                    "falsification_magnitude": zone.falsification_magnitude,
                    "coordinate_vector": list(zone.coordinate_vector),
                    "active": True,
                },
                timeout=5.0,
            )

    return {
        "success": True,
        "preset": preset,
        "seed": seed,
        "crypto_enabled": scenario.crypto_enabled,
        "llm_assistance_enabled": scenario.llm_assistance_enabled,
        "jamming_zones": len(scenario.jamming_zones),
        "spoofing_zones": len(scenario.spoofing_zones),
    }


@app.post("/chat")
async def chat(request: Request):
    """
    Main chat endpoint with MCP-style tool calling.

    The LLM has access to tools (move_agent, get_agent_status, add_spoofing_zone,
    toggle_crypto_auth, etc.) and can call them autonomously during the conversation.

    Simple move commands ("move agent1 to 5,5") are also handled via a fast regex
    path that skips the LLM for responsiveness.
    """
    global _last_chat_prompt
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return {"response": "Please enter a message."}

        print(f"\n[CHAT] Message: {user_message}")

        # Fast path: simple move commands bypass LLM for instant response
        if _is_move_command(user_message):
            result = await _handle_move_command(user_message)
            return {"response": result}

        # MCP tool-calling LLM agent handles everything else
        result = await answer_question(user_message)

        # Record for LLM Context panel so users can confirm the message reached the model
        _last_chat_prompt = {
            "agent_id": "user-chat",
            "timestamp": datetime.now().isoformat(),
            "prompt_preview": user_message[:300],
            "reasoning": (result.get("response") or "")[:300],
        }

        return result

    except Exception as e:
        print(f"[CHAT] Error: {e}")
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=500, content={"response": f"Error: {str(e)}"})


def _is_move_command(message: str) -> bool:
    """Check if message is a simple move command with coordinates."""
    msg = message.lower()
    move_keywords = ["move", "send", "relocate", "go to", "navigate"]
    has_move_verb = any(kw in msg for kw in move_keywords)
    has_agent = "agent" in msg or "vehicle" in msg
    has_target = "to" in msg
    return has_move_verb and has_agent and has_target


async def _handle_move_command(message: str) -> str:
    """Parse and execute move command with LLM fallback for complex commands."""
    import re

    msg = message.lower()

    # Extract agent ID
    agent_match = re.search(r"(?:agent|vehicle)\s*(\d+)", msg)
    if not agent_match:
        return "Could not identify which agent to move. Please specify an agent (e.g., 'agent1')."

    agent_id = f"agent{agent_match.group(1)}"

    # Try simple coordinate parsing first (fast path)
    coord_match = re.search(
        r"to\s*\(?(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)(?:[,\s]+(-?\d+\.?\d*))?\)?", msg
    )
    if coord_match:
        x = float(coord_match.group(1))
        y = float(coord_match.group(2))
        z = float(coord_match.group(3)) if coord_match.group(3) else 0.0
        result = await move_agent(agent_id, x, y, z)
        if result.get("success"):
            return result.get("message", "Move command sent.")
        return f"Failed to move {agent_id}: {result.get('error', 'unknown error')}"

    # Complex command - use LLM to parse
    print(f"[CHAT] Using LLM to parse complex move command for {agent_id}")
    parsed = await _llm_parse_move_command(message, agent_id)

    if parsed:
        x, y, z = parsed["x"], parsed["y"], parsed["z"]
        result = await move_agent(agent_id, x, y, z)
        explanation = parsed.get("explanation", "")
        if not result.get("success"):
            return f"Failed to move {agent_id}: {result.get('error', 'unknown error')}"
        response = result.get("message", "Move command sent.")
        if explanation:
            response += f" ({explanation})"
        return response

    return "Could not understand the move target. Try 'move agent1 to 5, 5' or describe the destination."


async def _llm_parse_move_command(message: str, agent_id: str) -> dict | None:
    """
    Use LLM to parse complex move commands.

    Handles references like:
    - "previous location"
    - "starting position"
    - "near agent2"
    - "center of the map"
    - "away from jamming zone"
    """
    from swarm_squad_ep1.rag import get_telemetry_history

    # Get agent history for context
    try:
        history = get_telemetry_history(agent_id, limit=20)
        trajectory = [
            (h.get("position", [0, 0, 0]), h.get("timestamp", "")) for h in history
        ]
    except Exception:
        trajectory = []

    # Get current agent positions
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SIMULATION_API_URL}/agents", timeout=5.0)
            agents_data = response.json().get("agents", {})
    except Exception:
        agents_data = {}

    # Build context for LLM
    agent_positions = {
        aid: a.get("position", [0, 0, 0]) for aid, a in agents_data.items()
    }
    current_pos = agent_positions.get(agent_id, [0, 0, 0])

    # Format trajectory
    traj_str = ""
    if trajectory:
        recent = trajectory[:5]
        traj_str = "Recent positions: " + " -> ".join(
            [f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})" for p, _ in recent]
        )
        if len(trajectory) > 0:
            oldest = trajectory[-1][0]
            traj_str += f"\nStarting position: ({oldest[0]:.1f}, {oldest[1]:.1f}, {oldest[2]:.1f})"

    # Format other agents
    other_agents = "\n".join(
        [
            f"  {aid}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})"
            for aid, p in agent_positions.items()
            if aid != agent_id
        ]
    )

    prompt = f"""Parse this vehicle movement command and extract the target coordinates.

COMMAND: "{message}"

CURRENT STATE:
- Target agent: {agent_id}
- Current position: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})
{traj_str}

OTHER AGENTS:
{other_agents if other_agents else "  (none)"}

MAP BOUNDS: X: -200 to 200, Y: -200 to 200, Z: 0 to 200
DESTINATION: (35, 150, 30)

TASK: Extract target coordinates. Handle references like:
- "previous location" = second most recent position in trajectory
- "starting position" = oldest position in trajectory
- "near agent2" = close to agent2's position
- "center" = (0, 0, 50)
- "origin" = (0, 0, 0)

Respond with ONLY valid JSON:
{{"x": 0.0, "y": 0.0, "z": 0.0, "explanation": "brief reason"}}

JSON:"""

    try:
        response = await async_chat_with_retry(
            LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )

        if response:
            content = response.get("message", {}).get("content", "")

            # Parse JSON from response
            import json

            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(line for line in lines if not line.startswith("```"))

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return {
                    "x": float(data.get("x", 0)),
                    "y": float(data.get("y", 0)),
                    "z": float(data.get("z", 0)),
                    "explanation": data.get("explanation", ""),
                }
    except Exception as e:
        print(f"[CHAT] LLM parse error: {e}")

    return None


@app.get("/data/qdrant")
async def get_qdrant_data():
    """Get recent telemetry from Qdrant."""
    try:
        data = get_all_telemetry(limit=50)
        return {"data": data, "count": len(data)}
    except Exception as e:
        return {"data": [], "error": str(e)}


@app.get("/data/postgresql")
async def get_postgresql_data():
    """Get recent logs from PostgreSQL."""
    try:
        data = get_logs(limit=50)
        return {"data": data, "count": len(data)}
    except Exception as e:
        return {"data": [], "error": str(e)}


@app.get("/agents")
async def proxy_agents():
    """Proxy to simulation API for agents."""
    return await _proxy_sim_json(
        "GET", "/agents", on_exception_payload={"agents": {}}, timeout=5.0
    )


@app.post("/agents")
async def proxy_create_agent(request: Request):
    """Create a new agent - proxy to simulation API."""
    data = await request.json()
    return await _proxy_sim_json("POST", "/agents", json_body=data, timeout=5.0)


@app.delete("/agents/{agent_id}")
async def proxy_delete_agent(agent_id: str):
    """Delete an agent - proxy to simulation API."""
    return await _proxy_sim_json("DELETE", f"/agents/{agent_id}", timeout=5.0)


@app.get("/visualization")
async def proxy_visualization(trail_length: str = "short"):
    """Get visualization data (communication links, waypoints, trails)."""
    return await _proxy_sim_json(
        "GET",
        "/visualization",
        params={"trail_length": trail_length},
        timeout=5.0,
        on_exception_payload={
            "communication_links": [],
            "waypoints": {},
            "traveled_paths": {},
        },
    )


# ============================================================================
# JAMMING ZONE PROXY ROUTES
# ============================================================================


@app.get("/jamming_zones")
async def proxy_jamming_zones():
    """Get all jamming zones from simulation API."""
    return await _proxy_sim_json(
        "GET", "/jamming_zones", timeout=5.0, on_exception_payload={"zones": []}
    )


@app.post("/jamming_zones")
async def proxy_create_jamming_zone(request: Request):
    """Create a new jamming zone."""
    data = await request.json()
    return await _proxy_sim_json("POST", "/jamming_zones", json_body=data, timeout=5.0)


@app.delete("/jamming_zones/{zone_id}")
async def proxy_delete_jamming_zone(zone_id: str):
    """Delete a jamming zone."""
    return await _proxy_sim_json("DELETE", f"/jamming_zones/{zone_id}", timeout=5.0)


# ============================================================================
# SPOOFING ZONE PROXY ROUTES
# ============================================================================


@app.get("/spoofing_zones")
async def proxy_spoofing_zones():
    """Get all spoofing zones."""
    return await _proxy_sim_json(
        "GET",
        "/spoofing_zones",
        timeout=5.0,
        on_exception_payload={"zones": [], "count": 0},
    )


@app.post("/spoofing_zones")
async def proxy_create_spoofing_zone(request: Request):
    """Create a spoofing zone."""
    data = await request.json()
    return await _proxy_sim_json("POST", "/spoofing_zones", json_body=data, timeout=5.0)


@app.delete("/spoofing_zones/{zone_id}")
async def proxy_delete_spoofing_zone(zone_id: str):
    """Delete a spoofing zone."""
    return await _proxy_sim_json("DELETE", f"/spoofing_zones/{zone_id}", timeout=5.0)


@app.delete("/spoofing_zones")
async def proxy_clear_spoofing_zones():
    """Clear all spoofing zones."""
    return await _proxy_sim_json("DELETE", "/spoofing_zones", timeout=5.0)


# ============================================================================
# MAVLINK / CRYPTO AUTH PROXY ROUTES
# ============================================================================


@app.get("/simulation/crypto_auth")
async def proxy_get_crypto_auth():
    """Get crypto auth status."""
    return await _proxy_sim_json(
        "GET",
        "/simulation/crypto_auth",
        timeout=5.0,
        on_exception_payload={"enabled": False},
    )


@app.post("/simulation/crypto_auth")
async def proxy_set_crypto_auth(request: Request):
    """Toggle crypto auth."""
    data = await request.json()
    return await _proxy_sim_json(
        "POST", "/simulation/crypto_auth", json_body=data, timeout=5.0
    )


@app.get("/protocol_stats")
async def proxy_protocol_stats():
    """Get MAVLink protocol statistics."""
    return await _proxy_sim_json(
        "GET",
        "/protocol_stats",
        timeout=5.0,
        on_exception_payload={"mavlink_enabled": False},
    )


@app.get("/simulation/attack_metrics")
async def proxy_attack_metrics():
    """Spoof-detection metrics (TP/FP/FN/TN + rates) from the sim API.

    Proxied so the dashboard (served on CHAT_API_PORT) can reach the
    endpoint hosted on SIM_API_PORT without cross-origin work.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SIMULATION_API_URL}/simulation/attack_metrics", timeout=5.0
            )
            if response.status_code >= 400:
                return _default_attack_metrics_payload(
                    f"simulation_api_status={response.status_code}"
                )
            payload = response.json()
            payload.setdefault("timestamp", datetime.now().isoformat())
            payload.setdefault("source", "simulation_api")
            return payload
    except Exception as e:
        return _default_attack_metrics_payload(str(e))


# ============================================================================
# SIMULATION CONTROL PROXY ROUTES
# ============================================================================


@app.get("/simulation/config")
async def proxy_simulation_config():
    """Get simulation configuration options."""
    return await _proxy_sim_json("GET", "/simulation/config", timeout=5.0)


@app.post("/simulation/algorithm")
async def proxy_simulation_algorithm(request: Request):
    """Update formation / path algorithm / obstacle type mid-simulation."""
    data = await request.json()
    return await _proxy_sim_json(
        "POST", "/simulation/algorithm", json_body=data, timeout=5.0
    )


@app.get("/simulation/v2v_channel")
async def proxy_v2v_channel_get():
    """Get V2V channel model status."""
    return await _proxy_sim_json("GET", "/simulation/v2v_channel", timeout=5.0)


@app.post("/simulation/v2v_channel")
async def proxy_v2v_channel_post(request: Request):
    """Toggle V2V channel model."""
    data = await request.json()
    return await _proxy_sim_json(
        "POST", "/simulation/v2v_channel", json_body=data, timeout=5.0
    )


@app.post("/simulation/start")
async def proxy_simulation_start(request: Request):
    """Start simulation."""
    data = await request.json()
    return await _proxy_sim_json(
        "POST", "/simulation/start", json_body=data, timeout=5.0
    )


@app.post("/simulation/stop")
async def proxy_simulation_stop():
    """Stop simulation."""
    return await _proxy_sim_json("POST", "/simulation/stop", timeout=5.0)


@app.post("/simulation/reset")
async def proxy_simulation_reset():
    """Reset simulation."""
    return await _proxy_sim_json("POST", "/simulation/reset", timeout=5.0)


@app.get("/simulation/state")
async def proxy_simulation_state():
    """Get simulation state."""
    return await _proxy_sim_json("GET", "/simulation/state", timeout=5.0)


# ============================================================================
# LLM ASSISTANCE PROXY ROUTES
# ============================================================================


@app.get("/simulation/llm_assistance")
async def proxy_get_llm_assistance():
    """Get LLM assistance state."""
    return await _proxy_sim_json(
        "GET",
        "/simulation/llm_assistance",
        timeout=5.0,
        on_exception_payload={"enabled": True},
    )


@app.post("/simulation/llm_assistance")
async def proxy_set_llm_assistance(request: Request):
    """Set LLM assistance state."""
    data = await request.json()
    return await _proxy_sim_json(
        "POST", "/simulation/llm_assistance", json_body=data, timeout=5.0
    )


@app.get("/llm_activity")
async def proxy_llm_activity(limit: int = 10):
    """Get recent LLM activity for chat panel."""
    return await _proxy_sim_json(
        "GET",
        "/llm_activity",
        params={"limit": limit},
        timeout=5.0,
        on_exception_payload={"activity": []},
    )


@app.get("/llm_context")
async def proxy_llm_context():
    """Get LLM context data for context panel."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SIMULATION_API_URL}/llm_context", timeout=5.0
            )
            data = _read_proxy_payload(response)

            # Inject the last user chat interaction so the "Last LLM Prompt" panel
            # is populated even when no agents are being autonomously assisted.
            if _last_chat_prompt and isinstance(data, dict):
                prompts = data.get("last_prompts") or []
                prompts.append(_last_chat_prompt)
                data["last_prompts"] = prompts

            return JSONResponse(status_code=response.status_code, content=data)
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)})


# ============================================================================
# SIMULATION RESULTS PROXY ROUTES
# ============================================================================


@app.get("/simulation/results")
async def proxy_simulation_results():
    """Get simulation results."""
    return await _proxy_sim_json("GET", "/simulation/results", timeout=5.0)


@app.get("/simulation/results/download")
async def proxy_simulation_results_download(format: str = "json"):
    """Download simulation results."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SIMULATION_API_URL}/simulation/results/download",
                params={"format": format},
                timeout=5.0,
            )
            if format == "csv":
                from fastapi.responses import PlainTextResponse

                return PlainTextResponse(
                    content=response.text,
                    status_code=response.status_code,
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": "attachment; filename=simulation_results.csv"
                    },
                )
            return JSONResponse(
                status_code=response.status_code, content=_read_proxy_payload(response)
            )
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)})


# For running standalone
if __name__ == "__main__":
    import uvicorn

    from swarm_squad_ep1.config import CHAT_API_PORT

    print("=" * 60)
    print("Starting Chat API")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=CHAT_API_PORT)
