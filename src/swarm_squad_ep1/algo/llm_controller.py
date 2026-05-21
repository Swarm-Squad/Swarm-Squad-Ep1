"""
LLM Assistance Controller for autonomous vehicle control.

When enabled, this controller monitors agent communication quality and provides
LLM-guided control when communication degrades below the perception threshold (PT).

The LLM uses historical telemetry data and knowledge of jamming zones to compute
optimal evasion vectors that help agents:
1. Restore communication quality
2. Continue progressing toward the destination
3. Avoid jamming zones
"""

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np

# Import config
try:
    from swarm_squad_ep1.config import (
        LLM_MODEL,
        LLM_TIMEOUT,
        MISSION_END,
        OLLAMA_HOST,
        PT,
        chat_with_retry,
        get_ollama_client,
    )
    from swarm_squad_ep1.rag import get_telemetry_history
except ImportError:
    # Fallback for standalone testing
    PT = 0.94
    MISSION_END = (35, 150, 30)
    LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"
    LLM_TIMEOUT = 30
    OLLAMA_HOST = "http://localhost:11434"


# ----------------------------------------------------------------------
# Guidance output contract
# ----------------------------------------------------------------------
GUIDANCE_SYSTEM_PROMPT = """You are a tactical advisor for an autonomous ground vehicle.
Your ONLY job is to output a single JSON object describing an evasion direction.
NEVER include prose, markdown, or explanations outside the JSON.
The JSON must have exactly these keys:
  direction: 3-number array [dx, dy, dz] (any unit length, will be normalized)
  speed:     number in [0.1, 1.0]
  reasoning: short string (<= 200 chars)
""".strip()

GUIDANCE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        },
        "speed": {"type": "number", "minimum": 0.1, "maximum": 1.0},
        "reasoning": {"type": "string"},
    },
    "required": ["direction", "speed", "reasoning"],
    "additionalProperties": False,
}


@dataclass
class LLMGuidance:
    """Guidance output from LLM."""

    agent_id: str
    direction: list[float]  # [dx, dy, dz] normalized direction vector
    speed: float  # Recommended speed multiplier
    reasoning: str  # LLM's explanation
    timestamp: str
    expires_at: float  # Time when this guidance expires


class LLMAssistanceController:
    """
    Controller that provides LLM-guided assistance when agent communication
    quality falls below the perception threshold (PT).

    Uses async queue pattern for non-blocking LLM requests.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the LLM assistance controller.

        Args:
            enabled: Whether LLM assistance is enabled by default
        """
        self.enabled = enabled
        self.pt_threshold = PT

        # LLM client
        self._client = None
        self._model = LLM_MODEL

        # Async request handling
        self._request_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._pending_requests: set[str] = set()
        self._worker_thread: Optional[threading.Thread] = None

        # Active guidance cache
        self._active_guidance: dict[str, LLMGuidance] = {}
        self._guidance_lifetime = 5.0  # Seconds before guidance expires

        # Rate limiting
        self._last_request_time: dict[str, float] = {}
        self._min_request_interval = 2.0  # Minimum seconds between requests per agent

        # User command blocking - agents with active user commands are blocked from auto-LLM
        self._user_blocked_agents: set[str] = set()

        # Logging
        self._log_history: list[dict] = []

        # LLM reliability counters (exposed via /llm_activity)
        self._stats = {
            "llm_calls": 0,
            "llm_parse_success": 0,
            "llm_parse_fail": 0,
            "llm_repair_attempted": 0,
            "llm_repair_success": 0,
            "llm_fallback_used": 0,
        }

        print(f"[LLMAssist] Initialized (enabled={enabled}, PT={self.pt_threshold})")

    def get_stats(self) -> dict:
        """Return a shallow copy of LLM reliability stats."""
        total = max(1, self._stats["llm_calls"])
        rep_attempts = max(1, self._stats["llm_repair_attempted"])
        return {
            **self._stats,
            "parse_success_rate": round(self._stats["llm_parse_success"] / total, 4),
            "parse_fail_rate": round(self._stats["llm_parse_fail"] / total, 4),
            "repair_success_rate": round(
                self._stats["llm_repair_success"] / rep_attempts, 4
            ),
        }

    @property
    def client(self):
        """Lazy-load Ollama client."""
        if self._client is None:
            try:
                self._client = get_ollama_client()
            except Exception as e:
                print(f"[LLMAssist] Failed to get Ollama client: {e}")
        return self._client

    def set_enabled(self, enabled: bool):
        """Enable or disable LLM assistance."""
        self.enabled = enabled
        print(f"[LLMAssist] {'Enabled' if enabled else 'Disabled'}")

        if not enabled:
            # Clear active guidance when disabled
            self._active_guidance.clear()

    def block_agent(self, agent_id: str):
        """
        Block an agent from receiving auto-LLM assistance.
        Called when user command is set for an agent.
        """
        self._user_blocked_agents.add(agent_id)
        # Also clear any active guidance for this agent
        if agent_id in self._active_guidance:
            del self._active_guidance[agent_id]
        # Remove from pending requests
        self._pending_requests.discard(agent_id)
        print(f"[LLMAssist] Blocked auto-assistance for {agent_id} (user command)")

    def unblock_agent(self, agent_id: str):
        """
        Unblock an agent, allowing auto-LLM assistance to resume.
        Called when user command completes.
        """
        self._user_blocked_agents.discard(agent_id)
        print(
            f"[LLMAssist] Unblocked auto-assistance for {agent_id} (user command completed)"
        )

    def is_blocked(self, agent_id: str) -> bool:
        """Check if an agent is blocked from auto-LLM assistance."""
        return agent_id in self._user_blocked_agents

    def check_agents_needing_assistance(self, agents: dict[str, Any]) -> list[str]:
        """
        Check which agents need LLM assistance based on communication quality.

        Args:
            agents: Dict of agent_id -> agent state

        Returns:
            List of agent IDs with comm_quality < PT (excluding blocked agents)
        """
        if not self.enabled:
            return []

        needing_assistance = []

        for agent_id, agent in agents.items():
            # Skip agents blocked by user commands
            if agent_id in self._user_blocked_agents:
                continue

            # Get communication quality
            if hasattr(agent, "communication_quality"):
                comm_quality = agent.communication_quality
            elif isinstance(agent, dict):
                comm_quality = agent.get("communication_quality", 1.0)
            else:
                continue

            # Check if below threshold
            if comm_quality < self.pt_threshold:
                needing_assistance.append(agent_id)

        return needing_assistance

    def request_guidance(
        self,
        agent_id: str,
        agent_state: Any,
        destination: tuple[float, float, float],
        jamming_zones: list[Any],
        discovered_obstacles: list[Any] = None,
    ):
        """
        Request LLM guidance for an agent (non-blocking).

        If LLM is slow/unavailable, immediately uses fallback guidance.

        Args:
            agent_id: ID of agent needing assistance
            agent_state: Current state of the agent
            destination: Mission destination coordinates
            jamming_zones: List of active jamming zones
            discovered_obstacles: List of obstacles discovered by the swarm
        """
        if not self.enabled:
            return

        # Skip agents blocked by user commands
        if agent_id in self._user_blocked_agents:
            return

        # Rate limiting
        current_time = time.time()
        last_request = self._last_request_time.get(agent_id, 0)
        if current_time - last_request < self._min_request_interval:
            return

        # Get agent position for fallback
        if hasattr(agent_state, "position"):
            position = agent_state.position
        elif isinstance(agent_state, dict):
            position = agent_state.get("position", [0, 0, 0])
        else:
            return

        # Combine jamming zones with discovered obstacles
        all_zones = []
        for zone in jamming_zones:
            if hasattr(zone, "center"):
                all_zones.append({"center": zone.center, "radius": zone.radius})
            elif isinstance(zone, dict):
                all_zones.append(
                    {
                        "center": zone.get("center", [0, 0, 0]),
                        "radius": zone.get("radius", 10),
                    }
                )

        # Add discovered obstacles (these have been found by the swarm)
        if discovered_obstacles:
            for obs in discovered_obstacles:
                if hasattr(obs, "center"):
                    all_zones.append({"center": obs.center, "radius": obs.radius})
                elif isinstance(obs, dict):
                    all_zones.append(
                        {
                            "center": obs.get("center", [0, 0, 0]),
                            "radius": obs.get("radius", 15),
                        }
                    )

        # IMMEDIATE FALLBACK: If there's no active guidance and agent is jammed,
        # provide instant deterministic guidance while waiting for LLM
        if (
            agent_id not in self._active_guidance
            or time.time() >= self._active_guidance[agent_id].expires_at
        ):
            fallback = self._fallback_guidance(
                agent_id, position, destination, all_zones
            )
            self._active_guidance[agent_id] = fallback
            print(f"[LLMAssist] INSTANT fallback for {agent_id}: {fallback.reasoning}")

        # Don't duplicate pending requests
        if agent_id in self._pending_requests:
            return

        # Start worker thread if not running
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._request_worker, daemon=True
            )
            self._worker_thread.start()

        # Queue the request
        request = {
            "agent_id": agent_id,
            "agent_state": agent_state,
            "destination": destination,
            "jamming_zones": all_zones,  # Combined zones
            "timestamp": current_time,
        }

        self._request_queue.put(request)
        self._pending_requests.add(agent_id)
        self._last_request_time[agent_id] = current_time

        print(
            f"[LLMAssist] Queued LLM request for {agent_id} (fallback active meanwhile)"
        )

    def _request_worker(self):
        """Background worker that processes LLM requests."""
        while True:
            try:
                # Get request with timeout
                request = self._request_queue.get(timeout=1.0)

                agent_id = request["agent_id"]

                try:
                    guidance = self._compute_guidance_sync(request)
                    if guidance:
                        self._result_queue.put(guidance)
                except Exception as e:
                    print(f"[LLMAssist] Error computing guidance for {agent_id}: {e}")
                finally:
                    self._pending_requests.discard(agent_id)
                    self._request_queue.task_done()

            except queue.Empty:
                # No requests, check if we should exit
                if self._request_queue.empty():
                    continue

    def _compute_guidance_sync(self, request: dict) -> Optional[LLMGuidance]:
        """
        Compute LLM guidance synchronously (runs in worker thread).

        Args:
            request: Request dict with agent info

        Returns:
            LLMGuidance object or None if failed
        """
        agent_id = request["agent_id"]
        agent_state = request["agent_state"]
        destination = request["destination"]
        jamming_zones = request["jamming_zones"]

        # Get agent position
        if hasattr(agent_state, "position"):
            position = agent_state.position
            comm_quality = getattr(agent_state, "communication_quality", 0.5)
        elif isinstance(agent_state, dict):
            position = agent_state.get("position", [0, 0, 0])
            comm_quality = agent_state.get("communication_quality", 0.5)
        else:
            return None

        # Get historical trajectory from Qdrant
        try:
            history = get_telemetry_history(agent_id, limit=10)
            trajectory = [h.get("position", [0, 0, 0]) for h in history]
        except Exception:
            trajectory = []

        # Format jamming zones
        zones_info = []
        for zone in jamming_zones:
            if hasattr(zone, "center"):
                zones_info.append(
                    {
                        "center": zone.center,
                        "radius": zone.radius,
                    }
                )
            elif isinstance(zone, dict):
                zones_info.append(
                    {
                        "center": zone.get("center", [0, 0, 0]),
                        "radius": zone.get("radius", 10),
                    }
                )

        # Build prompt
        prompt = self._build_prompt(
            agent_id=agent_id,
            position=position,
            comm_quality=comm_quality,
            destination=destination,
            jamming_zones=zones_info,
            trajectory=trajectory,
        )

        # Query LLM with explicit system/user separation + JSON-schema format.
        self._stats["llm_calls"] += 1
        try:
            if self.client is None:
                self._stats["llm_fallback_used"] += 1
                return self._fallback_guidance(
                    agent_id, position, destination, zones_info
                )

            messages = [
                {"role": "system", "content": GUIDANCE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            response = chat_with_retry(
                self.client,
                self._model,
                messages=messages,
                format=GUIDANCE_JSON_SCHEMA,
                options={"temperature": 0.0},
            )

            if response:
                content = response.get("message", {}).get("content", "") or ""
                guidance, err = self._parse_llm_response(agent_id, content)

                if guidance is not None:
                    self._stats["llm_parse_success"] += 1
                    self._log_guidance(agent_id, prompt, content, guidance)
                    return guidance

                # One self-repair round before falling back.
                self._stats["llm_repair_attempted"] += 1
                repair_msgs = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid guidance JSON. "
                            f"Error: {err or 'parse failure'}. Respond with ONLY "
                            '{"direction":[dx,dy,dz],"speed":<0.1-1.0>,"reasoning":"..."}.'
                        ),
                    },
                ]
                repaired = chat_with_retry(
                    self.client,
                    self._model,
                    messages=repair_msgs,
                    format=GUIDANCE_JSON_SCHEMA,
                    options={"temperature": 0.0},
                )
                if repaired:
                    content2 = repaired.get("message", {}).get("content", "") or ""
                    guidance2, _err2 = self._parse_llm_response(agent_id, content2)
                    if guidance2 is not None:
                        self._stats["llm_repair_success"] += 1
                        self._log_guidance(agent_id, prompt, content2, guidance2)
                        return guidance2

                self._stats["llm_parse_fail"] += 1

        except Exception as e:
            print(f"[LLMAssist] LLM query failed for {agent_id}: {e}")

        # Fallback to deterministic avoidance
        self._stats["llm_fallback_used"] += 1
        return self._fallback_guidance(agent_id, position, destination, zones_info)

    def _build_prompt(
        self,
        agent_id: str,
        position: list[float],
        comm_quality: float,
        destination: tuple[float, float, float],
        jamming_zones: list[dict],
        trajectory: list[list[float]],
    ) -> str:
        """Build the LLM prompt for guidance request with pre-computed escape directions."""
        pos_arr = np.array(position)
        dest_arr = np.array(destination)
        to_dest = dest_arr - pos_arr
        dist_to_dest = np.linalg.norm(to_dest)

        # Format trajectory
        traj_str = ""
        if trajectory:
            traj_points = [
                f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})" for p in trajectory[-5:]
            ]
            traj_str = f"\nRECENT TRAJECTORY: {' -> '.join(traj_points)}"

        # Analyze jamming zones and pre-compute escape directions
        zones_str = ""
        escape_recommendations = []

        if jamming_zones:
            zone_parts = []
            for i, z in enumerate(jamming_zones):
                c = np.array(z["center"])
                r = z["radius"]
                jamming_radius = r * 2.0  # κ_J = 2

                # Calculate relationship to this zone
                to_zone = c - pos_arr
                dist_to_zone_center = np.linalg.norm(to_zone)
                dist_to_zone_edge = dist_to_zone_center - jamming_radius

                # Check if agent is INSIDE this jamming zone
                if dist_to_zone_center < jamming_radius:
                    penetration = 1.0 - (dist_to_zone_center / jamming_radius)
                    severity = (
                        "severe"
                        if penetration > 0.5
                        else "moderate"
                        if penetration > 0.2
                        else "mild"
                    )

                    zone_parts.append(
                        f"Zone {i + 1}: center=({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}), "
                        f"radius={jamming_radius:.0f}, AGENT INSIDE ({severity}, {penetration:.0%} deep)"
                    )

                    # PRE-COMPUTE ESCAPE DIRECTION (tangent escape)
                    if dist_to_zone_center > 0.1:
                        from_zone = pos_arr - c
                        from_zone_norm = from_zone / (np.linalg.norm(from_zone) + 1e-6)
                        to_dest_norm = (
                            to_dest / (np.linalg.norm(to_dest) + 1e-6)
                            if dist_to_dest > 0.1
                            else np.array([0, 1, 0])
                        )

                        # Tangent direction (perpendicular to from_zone, toward destination)
                        cross1 = np.cross(from_zone_norm, to_dest_norm)
                        if np.linalg.norm(cross1) > 0.01:
                            tangent = np.cross(cross1, from_zone_norm)
                            tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
                            # Choose direction that points more toward destination
                            if np.dot(tangent, to_dest_norm) < 0:
                                tangent = -tangent
                        else:
                            tangent = from_zone_norm  # Parallel case, just move away

                        # Add push-out component for deep penetration
                        push_weight = min(0.5, penetration * 0.8)
                        escape_dir = (
                            1 - push_weight
                        ) * tangent + push_weight * from_zone_norm
                        escape_dir = escape_dir / (np.linalg.norm(escape_dir) + 1e-6)

                        escape_recommendations.append(
                            f"RECOMMENDED ESCAPE from Zone {i + 1}: direction=[{escape_dir[0]:.2f}, {escape_dir[1]:.2f}, {escape_dir[2]:.2f}], "
                            f"move {int(max(5, penetration * 15))} units to exit"
                        )
                else:
                    zone_parts.append(
                        f"Zone {i + 1}: center=({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}), "
                        f"radius={jamming_radius:.0f}, {-dist_to_zone_edge:.0f} units away"
                    )

            zones_str = "\nJAMMING ZONES:\n" + "\n".join(zone_parts)

        escape_str = ""
        if escape_recommendations:
            escape_str = "\n\n*** PRE-COMPUTED ESCAPE (USE THIS!) ***\n" + "\n".join(
                escape_recommendations
            )

        return f"""You are a tactical advisor for autonomous vehicle navigation.

SITUATION:
- Agent {agent_id} at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})
- Destination at ({destination[0]:.1f}, {destination[1]:.1f}, {destination[2]:.1f}) - {dist_to_dest:.1f} units away
- Communication quality: {comm_quality:.2f} (DEGRADED - need to escape jamming)
{zones_str}
{escape_str}
{traj_str}

IMPORTANT: Use the PRE-COMPUTED ESCAPE direction above! It has been calculated to:
1. Move AROUND the jamming zone (tangent escape)
2. Progress toward destination
3. Exit the jamming field efficiently

If the pre-computed escape direction is provided, USE IT DIRECTLY in your response.

Respond with ONLY valid JSON:
{{"direction": [dx, dy, dz], "speed": 0.8, "reasoning": "brief explanation"}}

JSON:"""

    def _parse_llm_response(
        self,
        agent_id: str,
        content: str,
    ) -> tuple[Optional[LLMGuidance], Optional[str]]:
        """Parse LLM response into guidance object.

        Returns ``(guidance, error_message)``. ``error_message`` is suitable
        to feed back to the model for a one-shot repair round.
        """
        try:
            # Local import so llm_controller stays independent of chat.* for tests.
            from swarm_squad_ep1.chat.json_utils import (
                ValidationError,
                extract_json_candidates,
                validate,
            )
        except Exception as e:
            print(f"[LLMAssist] json_utils import failed: {e}")
            return None, str(e)

        if not content:
            return None, "empty LLM response"

        candidates = extract_json_candidates(content)
        if not candidates:
            return None, "no JSON object found in response"

        last_err: Optional[str] = None
        for data in candidates:
            if not isinstance(data, dict):
                continue
            try:
                validate(data, GUIDANCE_JSON_SCHEMA)
            except ValidationError as e:
                last_err = str(e)
                continue

            direction = np.array(data["direction"], dtype=float)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                direction = np.array([0.0, 1.0, 0.0])

            speed = float(max(0.1, min(1.0, data["speed"])))

            return (
                LLMGuidance(
                    agent_id=agent_id,
                    direction=direction.tolist(),
                    speed=speed,
                    reasoning=str(data.get("reasoning", "")),
                    timestamp=datetime.now().isoformat(),
                    expires_at=time.time() + self._guidance_lifetime,
                ),
                None,
            )

        return None, last_err or "no candidate matched guidance schema"

    def _fallback_guidance(
        self,
        agent_id: str,
        position: list[float],
        destination: tuple[float, float, float],
        jamming_zones: list[dict],
    ) -> LLMGuidance:
        """
        Compute deterministic fallback guidance without LLM.

        Strategy: DESTINATION-BIASED exit — always maintain strong forward
        progress toward the destination while steering to the nearest zone
        boundary on the destination side. This avoids the trap where
        aggressive push-out fights the main controller's navigation.
        """
        pos = np.array(position)
        dest = np.array(destination)

        to_dest = dest - pos
        dist_to_dest = np.linalg.norm(to_dest)
        to_dest_norm = to_dest / max(dist_to_dest, 1e-6)

        best_zone = None
        best_influence = 0.0

        for zone in jamming_zones:
            zone_center = np.array(zone["center"])
            zone_radius = zone["radius"]
            dist_to_zone = np.linalg.norm(zone_center - pos)
            jamming_radius = zone_radius * 2.5
            if dist_to_zone < jamming_radius:
                influence = 1.0 - (dist_to_zone / jamming_radius)
                if influence > best_influence:
                    best_influence = influence
                    best_zone = zone

        if best_zone is not None and best_influence > 0.05:
            zone_center = np.array(best_zone["center"])
            from_zone = pos - zone_center
            dist_from_zone = np.linalg.norm(from_zone)

            if dist_from_zone > 0.1:
                from_zone_norm = from_zone / dist_from_zone

                # Lateral deflection: cross-product gives a direction tangent
                # to the zone that keeps us progressing toward the destination.
                cross = np.cross(from_zone_norm, to_dest_norm)
                if np.linalg.norm(cross) > 0.01:
                    tangent = np.cross(cross, from_zone_norm)
                    tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
                    if np.dot(tangent, to_dest_norm) < 0:
                        tangent = -tangent
                else:
                    tangent = to_dest_norm

                # Always keep destination as primary direction (≥60 %).
                # Add a small lateral nudge (≤25 %) plus a light push-out
                # (≤15 %) so the agent exits on the destination side.
                dest_w = 0.60
                tangent_w = 0.25 * best_influence
                push_w = 0.15 * best_influence
                direction = (
                    dest_w * to_dest_norm
                    + tangent_w * tangent
                    + push_w * from_zone_norm
                )
                reasoning = (
                    f"Guiding through jam zone ({best_influence:.0%} deep) "
                    f"toward destination"
                )
            else:
                direction = to_dest_norm
                reasoning = "At zone center — heading to destination"
        else:
            direction = to_dest_norm
            reasoning = "Clear path — heading to destination"

        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        speed = min(1.0, 0.85 + best_influence * 0.15)

        return LLMGuidance(
            agent_id=agent_id,
            direction=direction.tolist(),
            speed=speed,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat(),
            expires_at=time.time() + self._guidance_lifetime,
        )

    def get_guidance(self, agent_id: str) -> Optional[LLMGuidance]:
        """
        Get active guidance for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Active LLMGuidance or None if no valid guidance
        """
        # Process any pending results
        self._process_result_queue()

        # Check for active guidance
        guidance = self._active_guidance.get(agent_id)
        if guidance and time.time() < guidance.expires_at:
            return guidance

        # Guidance expired or doesn't exist
        if agent_id in self._active_guidance:
            del self._active_guidance[agent_id]

        return None

    def _process_result_queue(self):
        """Process completed guidance requests from result queue."""
        while True:
            try:
                guidance = self._result_queue.get_nowait()
                self._active_guidance[guidance.agent_id] = guidance
                print(
                    f"[LLMAssist] New guidance for {guidance.agent_id}: {guidance.reasoning}"
                )
                self._result_queue.task_done()
            except queue.Empty:
                break

    def apply_guidance(
        self,
        agent_id: str,
        base_control: np.ndarray,
        guidance: LLMGuidance,
        comm_quality: float = 1.0,
    ) -> np.ndarray:
        """
        Apply LLM guidance with ADAPTIVE weighting based on jamming severity.

        When communication quality is very low (deep in jamming), LLM gets MORE weight
        to help escape. When near the edge of jamming, path planning is more dominant.

        Control hierarchy:
        1. Human MCP Commands (highest) - blocks LLM auto-assistance
        2. Path Planning + Formation Control (adaptive weight based on comm quality)
        3. LLM Auto-Assistance (adaptive weight - higher when deeper in jamming)

        Args:
            agent_id: Agent ID
            base_control: Base control vector from path planning + formation
            guidance: LLM guidance to apply
            comm_quality: Current communication quality (0-1, lower = more jammed)

        Returns:
            Blended control vector with adaptive weighting
        """
        if guidance is None:
            return base_control

        # Convert guidance direction to control vector
        llm_control = np.array(guidance.direction) * guidance.speed

        # ADAPTIVE WEIGHTING — LLM supplements the controller, never dominates.
        # The controller already handles obstacle avoidance; the LLM adds a
        # destination-aware nudge that helps the agent progress faster.
        severity = max(0.0, min(1.0, 1.0 - comm_quality / self.pt_threshold))

        # LLM weight: 15 % at threshold edge → 35 % at zero comm quality
        llm_weight = 0.15 + severity * 0.20
        path_weight = 1.0 - llm_weight

        blended = path_weight * base_control + llm_weight * llm_control

        print(
            f"[LLMAssist] {agent_id} weights: path={path_weight:.0%}, llm={llm_weight:.0%} (comm={comm_quality:.2f})"
        )

        return blended

    def _log_guidance(
        self,
        agent_id: str,
        prompt: str,
        response: str,
        guidance: LLMGuidance,
    ):
        """Log guidance for debugging."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "prompt_preview": prompt[:200] + "...",
            "response_preview": response[:200] + "...",
            "direction": guidance.direction,
            "speed": guidance.speed,
            "reasoning": guidance.reasoning,
        }

        self._log_history.append(entry)

        # Keep history manageable
        if len(self._log_history) > 100:
            self._log_history = self._log_history[-100:]

    def get_status(self) -> dict:
        """Get controller status."""
        return {
            "enabled": self.enabled,
            "pt_threshold": self.pt_threshold,
            "active_guidance_count": len(self._active_guidance),
            "pending_requests": list(self._pending_requests),
            "log_entries": len(self._log_history),
        }

    def get_active_guidance_for_visualization(self) -> list[dict]:
        """
        Get all active guidance for 3D visualization.

        Returns:
            List of dicts with agent_id, direction, speed, reasoning for active guidance
        """
        # Process any pending results first
        self._process_result_queue()

        current_time = time.time()
        active = []
        expired_count = 0

        for agent_id, guidance in list(self._active_guidance.items()):
            if current_time < guidance.expires_at:
                active.append(
                    {
                        "agent_id": agent_id,
                        "direction": guidance.direction,
                        "speed": guidance.speed,
                        "reasoning": guidance.reasoning,
                        "timestamp": guidance.timestamp,
                        "expires_in": guidance.expires_at - current_time,
                    }
                )
            else:
                # Clean up expired guidance
                del self._active_guidance[agent_id]
                expired_count += 1

        return active

    def get_recent_activity(self, limit: int = 10) -> list[dict]:
        """
        Get recent LLM guidance activity for display in chat panel.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recent guidance log entries
        """
        return self._log_history[-limit:]

    def get_current_context(
        self,
        agents: dict = None,
        jamming_zones: list = None,
        spoofing_zones: list = None,
    ) -> dict:
        """
        Get the current context data that would be sent to LLM.

        Args:
            agents: Current agent states (optional)
            jamming_zones: Current jamming zones (optional)
            spoofing_zones: Current spoofing zones (optional)

        Returns:
            Dict with context information for display
        """
        context = {
            "enabled": self.enabled,
            "pt_threshold": self.pt_threshold,
            "agents_being_assisted": [],
            "active_guidance": [],
            "last_prompts": [],
            "jamming_zones": [],
            "spoofing_zones": [],
        }

        # Add agents needing assistance
        if agents:
            for agent_id, agent in agents.items():
                if hasattr(agent, "communication_quality"):
                    comm_quality = agent.communication_quality
                elif isinstance(agent, dict):
                    comm_quality = agent.get("communication_quality", 1.0)
                else:
                    continue

                if comm_quality < self.pt_threshold:
                    pos = (
                        agent.position
                        if hasattr(agent, "position")
                        else agent.get("position", [0, 0, 0])
                    )
                    context["agents_being_assisted"].append(
                        {
                            "agent_id": agent_id,
                            "communication_quality": float(comm_quality),
                            "position": [float(p) for p in pos],
                        }
                    )

        # Add active guidance
        context["active_guidance"] = self.get_active_guidance_for_visualization()

        # Add jamming zones info
        if jamming_zones:
            for zone in jamming_zones:
                if hasattr(zone, "center"):
                    context["jamming_zones"].append(
                        {
                            "id": zone.id if hasattr(zone, "id") else "unknown",
                            "center": zone.center,
                            "radius": zone.radius,
                        }
                    )
                elif isinstance(zone, dict):
                    context["jamming_zones"].append(
                        {
                            "id": zone.get("id", "unknown"),
                            "center": zone.get("center", [0, 0, 0]),
                            "radius": zone.get("radius", 10),
                        }
                    )

        # Add spoofing zones info
        if spoofing_zones:
            for zone in spoofing_zones:
                if hasattr(zone, "center"):
                    context["spoofing_zones"].append(
                        {
                            "id": zone.id if hasattr(zone, "id") else "unknown",
                            "center": zone.center,
                            "radius": zone.radius,
                            "spoof_type": zone.spoof_type.value
                            if hasattr(zone, "spoof_type")
                            else "unknown",
                            "active": zone.active if hasattr(zone, "active") else True,
                        }
                    )
                elif isinstance(zone, dict):
                    context["spoofing_zones"].append(
                        {
                            "id": zone.get("id", "unknown"),
                            "center": zone.get("center", [0, 0, 0]),
                            "radius": zone.get("radius", 10),
                            "spoof_type": zone.get("spoof_type", "unknown"),
                            "active": zone.get("active", True),
                        }
                    )

        # Add recent prompts (last 3)
        for entry in self._log_history[-3:]:
            context["last_prompts"].append(
                {
                    "agent_id": entry.get("agent_id"),
                    "timestamp": entry.get("timestamp"),
                    "prompt_preview": entry.get("prompt_preview", "")[:300],
                    "reasoning": entry.get("reasoning", ""),
                }
            )

        return context


# Global instance
_llm_controller: Optional[LLMAssistanceController] = None


def get_llm_controller() -> LLMAssistanceController:
    """Get or create the global LLM assistance controller."""
    global _llm_controller
    if _llm_controller is None:
        _llm_controller = LLMAssistanceController(enabled=True)
    return _llm_controller


def reset_llm_controller():
    """Reset the global LLM controller."""
    global _llm_controller
    _llm_controller = None
