"""Swarm Squad Ep1 scriptable client API."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from swarm_squad_ep1.config import SIMULATION_API_URL
from swarm_squad_ep1.research.runner import Result, run_scenario
from swarm_squad_ep1.research.scenarios import (
    Scenario,
    build_education_scenario,
    get_education_presets,
)


@dataclass
class SwarmSquadClient:
    """Stable facade for script-driven control of a running simulation backend."""

    base_url: str = SIMULATION_API_URL
    timeout: float = 10.0

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        import httpx

        url = f"{self.base_url.rstrip('/')}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(method, url, json=payload)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Live API controls (GUI/server-backed)
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def agents(self) -> dict[str, Any]:
        return self._request("GET", "/agents")

    def agent(self, agent_id: str) -> dict[str, Any]:
        return self._request("GET", f"/agents/{agent_id}")

    def add_agent(self, x: float, y: float, z: float = 0.0) -> dict[str, Any]:
        return self._request("POST", "/agents", {"x": x, "y": y, "z": z})

    def remove_agent(self, agent_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/agents/{agent_id}")

    def simulation_state(self) -> dict[str, Any]:
        return self._request("GET", "/simulation/state")

    def simulation_config(self) -> dict[str, Any]:
        return self._request("GET", "/simulation/config")

    def path_algorithms(self) -> list[str]:
        return list(self.simulation_config().get("path_algorithms", []))

    def custom_path_algorithms(self) -> list[dict[str, Any]]:
        return list(self.simulation_config().get("custom_path_algorithms", []))

    def visualization(self, trail_length: str = "short") -> dict[str, Any]:
        return self._request("GET", f"/visualization?trail_length={trail_length}")

    def start_simulation(
        self,
        formation: str = "communication_aware",
        path_algorithm: str = "astar",
        crypto_auth: bool | None = None,
        crypto_algorithm: str = "hmac_sha256",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "formation": formation,
            "path_algorithm": path_algorithm,
        }
        if crypto_auth is not None:
            payload["crypto_auth"] = bool(crypto_auth)
            payload["crypto_algorithm"] = crypto_algorithm
        return self._request("POST", "/simulation/start", payload)

    def stop_simulation(self) -> dict[str, Any]:
        return self._request("POST", "/simulation/stop")

    def reset_simulation(self) -> dict[str, Any]:
        return self._request("POST", "/simulation/reset")

    def set_algorithm(
        self,
        *,
        formation: str | None = None,
        path_algorithm: str | None = None,
        default_obstacle_type: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if formation is not None:
            payload["formation"] = formation
        if path_algorithm is not None:
            payload["path_algorithm"] = path_algorithm
        if default_obstacle_type is not None:
            payload["default_obstacle_type"] = default_obstacle_type
        if not payload:
            return {"success": False, "error": "No algorithm fields provided"}
        return self._request("POST", "/simulation/algorithm", payload)

    def simulation_results(self) -> dict[str, Any]:
        return self._request("GET", "/simulation/results")

    def simulate_step(self) -> dict[str, Any]:
        return self._request("POST", "/simulate_step")

    def move_agent(
        self, agent: str, x: float, y: float, z: float = 0.0
    ) -> dict[str, Any]:
        return self._request(
            "POST", "/move_agent", {"agent": agent, "x": x, "y": y, "z": z}
        )

    def add_jamming_zone(
        self,
        center: tuple[float, float, float],
        radius: float,
        jam_type: str = "low_jam",
        intensity: float = 1.0,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/jamming_zones",
            {
                "center": list(center),
                "radius": radius,
                "obstacle_type": jam_type,
                "intensity": intensity,
            },
        )

    def list_jamming_zones(self) -> dict[str, Any]:
        return self._request("GET", "/jamming_zones")

    def delete_jamming_zone(self, zone_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/jamming_zones/{zone_id}")

    def clear_jamming_zones(self) -> dict[str, Any]:
        return self._request("DELETE", "/jamming_zones")

    def add_spoofing_zone(
        self,
        center: tuple[float, float, float],
        radius: float,
        spoof_type: str = "phantom",
        phantom_count: int = 2,
        falsification_magnitude: float = 8.0,
        coordinate_vector: tuple[float, float, float] = (10.0, 10.0, 0.0),
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/spoofing_zones",
            {
                "center": list(center),
                "radius": radius,
                "spoof_type": spoof_type,
                "phantom_count": phantom_count,
                "falsification_magnitude": falsification_magnitude,
                "coordinate_vector": list(coordinate_vector),
            },
        )

    def list_spoofing_zones(self) -> dict[str, Any]:
        return self._request("GET", "/spoofing_zones")

    def delete_spoofing_zone(self, zone_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/spoofing_zones/{zone_id}")

    def clear_spoofing_zones(self) -> dict[str, Any]:
        return self._request("DELETE", "/spoofing_zones")

    def v2v_channel_status(self) -> dict[str, Any]:
        return self._request("GET", "/simulation/v2v_channel")

    def set_v2v_channel(
        self, enabled: bool, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"enabled": bool(enabled)}
        if params:
            payload["params"] = params
        return self._request("POST", "/simulation/v2v_channel", payload)

    def set_comm_model(self, model: str) -> dict[str, Any]:
        model = model.strip().lower()
        if model not in {"v2v_channel", "legacy"}:
            return {
                "success": False,
                "error": f"Unknown comm model '{model}'. Expected 'v2v_channel' or 'legacy'.",
            }
        return self.set_v2v_channel(enabled=model == "v2v_channel")

    def set_crypto_auth(
        self, enabled: bool, algorithm: str = "hmac_sha256"
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/simulation/crypto_auth",
            {"enabled": bool(enabled), "algorithm": algorithm},
        )

    def crypto_auth_status(self) -> dict[str, Any]:
        return self._request("GET", "/simulation/crypto_auth")

    def protocol_stats(self) -> dict[str, Any]:
        return self._request("GET", "/protocol_stats")

    def attack_metrics(self) -> dict[str, Any]:
        return self._request("GET", "/simulation/attack_metrics")

    def set_llm_assistance(self, enabled: bool) -> dict[str, Any]:
        return self._request(
            "POST", "/simulation/llm_assistance", {"enabled": bool(enabled)}
        )

    def llm_assistance_status(self) -> dict[str, Any]:
        return self._request("GET", "/simulation/llm_assistance")

    def list_custom_algorithms(self) -> dict[str, Any]:
        return self._request("GET", "/simulation/custom_algorithms")

    def register_custom_algorithm(
        self,
        *,
        name: str,
        import_path: str,
        description: str = "",
        replace: bool = False,
        mode: str = "waypoint",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/simulation/custom_algorithms",
            {
                "name": name,
                "import_path": import_path,
                "description": description,
                "replace": replace,
                "mode": mode,
            },
        )

    def remove_custom_algorithm(self, name: str) -> dict[str, Any]:
        return self._request("DELETE", f"/simulation/custom_algorithms/{name}")

    def apply_preset(self, preset: str, seed: int = 0) -> dict[str, Any]:
        """Reset live simulation and configure attack/defense state from a preset."""
        scenario = build_education_scenario(preset, seed=seed)
        self.reset_simulation()
        self.set_algorithm(
            formation=scenario.formation_type,
            path_algorithm=scenario.path_algorithm,
        )
        self.set_crypto_auth(
            scenario.crypto_enabled, algorithm=scenario.crypto_algorithm
        )
        self.set_llm_assistance(scenario.llm_assistance_enabled)

        for jam in scenario.jamming_zones:
            self.add_jamming_zone(jam.center, jam.radius, jam.obstacle_type)
        for spoof in scenario.spoofing_zones:
            self.add_spoofing_zone(spoof.center, spoof.radius, spoof.spoof_type)

        return {
            "success": True,
            "preset": preset,
            "crypto_enabled": scenario.crypto_enabled,
            "llm_assistance_enabled": scenario.llm_assistance_enabled,
            "jamming_zones": len(scenario.jamming_zones),
            "spoofing_zones": len(scenario.spoofing_zones),
        }

    # Backward-compatible alias used by older examples/tests.
    def apply_education_preset(self, preset: str, seed: int = 0) -> dict[str, Any]:
        return self.apply_preset(preset, seed=seed)

    # ------------------------------------------------------------------
    # Headless research helpers (no running servers required)
    # ------------------------------------------------------------------
    @staticmethod
    def list_presets() -> dict[str, dict]:
        return get_education_presets()

    @staticmethod
    def build_preset_scenario(preset: str, seed: int = 0) -> Scenario:
        return build_education_scenario(preset, seed=seed)

    @staticmethod
    def run_headless_scenario(
        scenario: Scenario,
        keep_trace: bool = False,
        verbose: bool = False,
    ) -> Result:
        return run_scenario(scenario, keep_trace=keep_trace, verbose=verbose)

    @staticmethod
    def run_headless_preset(
        preset: str,
        seed: int = 0,
        keep_trace: bool = False,
        verbose: bool = False,
    ) -> Result:
        scenario = build_education_scenario(preset, seed=seed)
        return run_scenario(scenario, keep_trace=keep_trace, verbose=verbose)

    # ------------------------------------------------------------------
    # Script-driven algorithm loop helper
    # ------------------------------------------------------------------
    def run_script_control_loop(
        self,
        controller: Callable[[dict[str, Any], int], Sequence[dict[str, Any]]],
        *,
        steps: int = 100,
        step_interval_s: float = 0.0,
        auto_simulate_step: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run a script-defined control loop while visualizing in the GUI.

        The controller callback receives the latest simulation state and current
        step index. It should return command dicts like:
        {"agent": "agent1", "x": 10.0, "y": 5.0, "z": 1.0}.
        """
        trace: list[dict[str, Any]] = []
        for step_idx in range(steps):
            state = self.simulation_state()
            commands = controller(state, step_idx) or []

            applied = []
            for command in commands:
                agent = command.get("agent")
                if agent is None:
                    continue
                result = self.move_agent(
                    str(agent),
                    float(command.get("x", 0.0)),
                    float(command.get("y", 0.0)),
                    float(command.get("z", 0.0)),
                )
                applied.append({"command": command, "result": result})

            step_result = self.simulate_step() if auto_simulate_step else None
            trace.append(
                {
                    "step": step_idx,
                    "commands_applied": applied,
                    "simulate_step": step_result,
                }
            )
            if step_interval_s > 0:
                time.sleep(step_interval_s)
        return trace
