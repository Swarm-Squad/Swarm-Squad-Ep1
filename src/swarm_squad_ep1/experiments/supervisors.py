"""
Supervisors that share the LLM controller's interface.

All supervisors subclass LLMController, so they use the same telemetry, the
same command parser and 32-bearing vocabulary, the same request cadence
(every llm_feedback_interval steps when none is pending and at least 2 s
after the previous request), the same severity-scaled blend, avoidance
layer, and command lifetime. Only the decision maker differs:

- none: no decisions (reactive control only)
- rule: hand-written rules over the telemetry
- mpc:  one-step look-ahead over the command vocabulary
- llm:  the Ollama-served LLM (the original controller)

With the C2 gate enabled, an agent whose penetration depth is at least
c2_depth cannot exchange telemetry or commands with the supervisor: commands
addressed to it are dropped, and the supervisor sees its last delivered
telemetry.
"""

import copy
import logging
import os
import re
import threading
import time

import numpy as np

import swarm_squad_ep1.config as config
from swarm_squad_ep1.controllers.behavior_controller import BehaviorController
from swarm_squad_ep1.controllers.controller_factory import (
    ControllerFactory,
    ControllerType,
)
from swarm_squad_ep1.controllers.formation_controller import FormationController
from swarm_squad_ep1.controllers.llm_controller import LLMController
from swarm_squad_ep1.experiments import compass, linkmodel
from swarm_squad_ep1.utils import format_swarm_state_for_llm

# Same instruction grammar as LLMController's parser
_BEARING_ALTERNATION = "|".join(
    re.escape(name) for name in sorted(compass.BEARING_NAMES, key=len, reverse=True)
)
INSTRUCTION_PATTERN = re.compile(
    r"Agent-(\d+):\s+(?:Move|Reposition)\s+(\d+)\s+units\s+"
    rf"((?i:{_BEARING_ALTERNATION}))\b"
)
LOOSE_INSTRUCTION_PATTERN = re.compile(r"Agent-(\d+):\s+(?:Move|Reposition)\s+")
HINT_PATTERN = re.compile(
    r"RECOMMENDED (EVASION|AVOIDANCE): Agent-(\d+) should move (\d+) units "
    rf"({_BEARING_ALTERNATION}) to "
)

R_ALPHA = linkmodel.optimal_spacing()


class Supervisor(LLMController):
    """Base class: C2 gating, perceived state, and decision records."""

    kind = "none"

    def __init__(
        self,
        swarm_state,
        run_dir=None,
        c2_gate=False,
        c2_depth=0.5,
        llm_model=None,
        feedback_interval=None,
    ):
        self.run_dir = run_dir
        self.c2_gate = c2_gate
        self.c2_depth = c2_depth
        self.decisions = []
        self.gamma = np.ones(swarm_state.swarm_size, dtype=bool)
        self._snapshot = None
        self._decision_lock = threading.Lock()
        super().__init__(
            swarm_state, llm_model=llm_model, llm_feedback_interval=feedback_interval
        )
        self.enabled = self.kind != "none"

    # Logging -----------------------------------------------------------------

    def _setup_file_logging(self):
        """One log file per run, in the run directory."""
        self.file_logger = logging.getLogger(f"supervisor.{id(self)}")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.propagate = False
        self.file_logger.handlers.clear()
        self.log_file_path = None
        if self.run_dir:
            self.log_file_path = os.path.join(self.run_dir, "supervisor.log")
            handler = logging.FileHandler(self.log_file_path)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.file_logger.addHandler(handler)
        else:
            self.file_logger.addHandler(logging.NullHandler())

    def close(self):
        for handler in list(self.file_logger.handlers):
            handler.close()
            self.file_logger.removeHandler(handler)

    # C2 gate and perceived state ---------------------------------------------

    def _reachability(self) -> np.ndarray:
        state = self.swarm_state
        if not self.c2_gate:
            return np.ones(state.swarm_size, dtype=bool)
        return ~(state.jamming_affected & (state.jamming_depth >= self.c2_depth))

    def _update_snapshot(self):
        state = self.swarm_state
        if self._snapshot is None:
            self._snapshot = {
                "pos": state.swarm_position.copy(),
                "q": state.communication_qualities_matrix.copy(),
                "jam": state.jamming_affected.copy(),
                "depth": state.jamming_depth.copy(),
            }
        g = self.gamma
        self._snapshot["pos"][g] = state.swarm_position[g]
        self._snapshot["q"][g, :] = state.communication_qualities_matrix[g, :]
        self._snapshot["jam"][g] = state.jamming_affected[g]
        self._snapshot["depth"][g] = state.jamming_depth[g]

    def perceived_state(self):
        """The swarm state as the supervisor sees it through the C2 links."""
        if self.gamma.all():
            return self.swarm_state
        stale = ~self.gamma
        snap = self._snapshot
        view = copy.copy(self.swarm_state)
        pos = self.swarm_state.swarm_position.copy()
        pos[stale] = snap["pos"][stale]
        q = self.swarm_state.communication_qualities_matrix.copy()
        q[stale, :] = snap["q"][stale, :]
        jam = self.swarm_state.jamming_affected.copy()
        jam[stale] = snap["jam"][stale]
        depth = self.swarm_state.jamming_depth.copy()
        depth[stale] = snap["depth"][stale]
        r = linkmodel.pairwise_distances(pos)
        off = ~np.eye(len(pos), dtype=bool)
        view.swarm_position = pos
        view.communication_qualities_matrix = q
        view.jamming_affected = jam
        view.jamming_depth = depth
        view.distances_matrix = np.where(off, r, 0.0)
        view.neighbor_agent_matrix = np.where(off, linkmodel.far_field(r), 0.0)
        return view

    def telemetry(self, view) -> str:
        """The telemetry text that the LLM receives, built from a view."""
        real = self.swarm_state
        self.swarm_state = view
        try:
            return self._condense_state_description(format_swarm_state_for_llm(view))
        finally:
            self.swarm_state = real

    # Control loop hooks ------------------------------------------------------

    def compute_control(self) -> np.ndarray:
        self.gamma = self._reachability()
        self._update_snapshot()
        return super().compute_control()

    def _request_llm_feedback_async(self):
        """Non-LLM supervisors decide synchronously, with no latency."""
        view = self.perceived_state()
        started = time.perf_counter()
        commands = self.decide(view)
        compute_s = time.perf_counter() - started
        text = compass.command_text(commands)
        with self._decision_lock:
            self.decisions.append(
                {
                    "kind": self.kind,
                    "step_request": self.step_counter,
                    "compute_s": compute_s,
                    "text": text,
                }
            )
        self.feedback_queue.put(text)

    def decide(self, view):
        """Return a list of (agent, distance, bearing) commands."""
        return []

    def _parse_llm_feedback_to_control_inputs(self):
        super()._parse_llm_feedback_to_control_inputs()
        parsed = list(dict.fromkeys(self.llm_affected_agents))
        delivered = [i for i in parsed if self.gamma[i]]
        dropped = [i for i in parsed if not self.gamma[i]]
        self.llm_affected_agents = delivered
        text = self.current_feedback or ""
        with self._decision_lock:
            if self.decisions and "step_parsed" not in self.decisions[-1]:
                self.decisions[-1].update(
                    {
                        "step_parsed": self.step_counter,
                        "n_instructions": len(LOOSE_INSTRUCTION_PATTERN.findall(text)),
                        "n_valid": len(INSTRUCTION_PATTERN.findall(text)),
                        "parsed": parsed,
                        "delivered": delivered,
                        "dropped_c2": dropped,
                    }
                )

    # Shared helpers for rule and mpc -----------------------------------------

    @staticmethod
    def known_jammers(view):
        """Jamming fields reported in the telemetry (once any agent is jammed)."""
        if view.obstacle_mode == config.ObstacleMode.LOW_POWER_JAMMING and np.any(
            view.jamming_affected
        ):
            return [tuple(o) for o in view.obstacles]
        return []

    @staticmethod
    def detected_obstacles(view, detection_radius=15.0):
        """Physical obstacles reported in the telemetry."""
        if view.obstacle_mode != config.ObstacleMode.HARD:
            return []
        found = []
        for x, y, r in view.obstacles:
            d = np.linalg.norm(view.swarm_position - np.array([x, y]), axis=1)
            if np.any(d < detection_radius + r):
                found.append((x, y, r))
        return found

    @staticmethod
    def agent_severity(view, i) -> float:
        """Same formula as LLMController._get_agent_severity, on a view."""
        if view.jamming_affected[i]:
            return float(view.jamming_depth[i])
        q = view.communication_qualities_matrix[i]
        active = q[q > 0]
        if active.size and np.any(q < config.PT):
            worst = float(active.min())
            if worst < config.PT:
                return min(1.0, (config.PT - worst) / config.PT)
        return 0.0


class RuleSupervisor(Supervisor):
    """
    Rules, in priority order:
    1. apply every evasion or avoidance hint in the telemetry;
    2. a jammed agent without a hint moves 10 units away from the field center;
    3. if J_n < P_T, then for the weakest link below P_T between two unjammed
       neighbors, the endpoint with the lower mean link quality moves 5 units
       along the link, toward its partner if the pair is farther apart than
       r_alpha and away from it otherwise.
    """

    kind = "rule"

    def decide(self, view):
        commands = {}
        text = self.telemetry(view)
        for _, agent, distance, bearing in HINT_PATTERN.findall(text):
            agent = int(agent)
            if agent not in commands:
                commands[agent] = (int(distance), bearing)

        jammers = self.known_jammers(view)
        positions = view.swarm_position
        for i in np.where(view.jamming_affected)[0]:
            if i in commands or not jammers:
                continue
            centers = np.array([j[:2] for j in jammers])
            nearest = centers[np.argmin(np.linalg.norm(centers - positions[i], axis=1))]
            commands[int(i)] = (10, compass.quantize(positions[i] - nearest))

        if view_jn(view) < config.PT:
            link = self._weakest_link_command(view, set(commands))
            if link:
                commands[link[0]] = link[1:]

        return [(i, d, b) for i, (d, b) in sorted(commands.items())]

    @staticmethod
    def _weakest_link_command(view, exclude):
        q = view.communication_qualities_matrix
        a = view.neighbor_agent_matrix
        jam = view.jamming_affected
        clean = ~jam[:, None] & ~jam[None, :]
        poor = (a > config.PT) & (q > 0) & (q < config.PT) & clean
        if not poor.any():
            return None
        i, j = np.unravel_index(np.argmin(np.where(poor, q, np.inf)), q.shape)

        def mean_quality(k):
            neighbors = a[k] > config.PT
            return q[k, neighbors].mean() if neighbors.any() else 0.0

        candidates = [int(k) for k in (i, j) if int(k) not in exclude]
        if not candidates:
            return None
        pick = min(candidates, key=lambda k: (mean_quality(k), -k))
        partner = int(j) if pick == int(i) else int(i)
        vector = view.swarm_position[partner] - view.swarm_position[pick]
        if np.linalg.norm(vector) < R_ALPHA:
            vector = -vector
        return (pick, 5, compass.quantize(vector))


class MPCSupervisor(Supervisor):
    """
    One-step look-ahead over the command vocabulary.

    Flagged agents are the jammed agents, agents within 10 units of a reported
    field, agents inside the 6-unit buffer of a detected obstacle (where the
    reactive layer slows down to follow the boundary), and, if J_n < P_T, the
    endpoints of links below P_T. For each flagged agent, in that order, the
    candidates are no command and the 32 bearings x {5, 8, 10} units. Each
    candidate is rolled forward over the command lifetime H with the blend
    weight the override would use, while the other agents follow their
    current reactive input (or their command, if already planned). The cost
    is

        -mean(J_n at H/2 and H) + progress_weight * (extra distance to p_dest)
        + collision_penalty * [path within r + 1 of a detected obstacle],

    with J_n predicted by the link and jamming model for the reported fields.
    A command is issued only if it lowers the cost by at least `deadband`
    relative to no command.
    """

    kind = "mpc"
    distances = (5, 8, 10)
    progress_weight = 0.002
    collision_penalty = 1.0
    deadband = 0.002

    def decide(self, view):
        jammers = self.known_jammers(view)
        obstacles = self.detected_obstacles(view)
        positions = view.swarm_position.copy()
        reactive, base = self._reactive_inputs(view)
        horizon = self.llm_control_lifetime
        destination = view.swarm_destination

        nominal = positions + horizon * reactive
        predicted = nominal.copy()

        flagged = self._flagged_agents(view, jammers, obstacles)
        unit = {b: compass.unit_vector(b) for b in compass.BEARING_NAMES}
        commands = []

        def cost(end, i):
            trial = predicted.copy()
            trial[i] = end
            mid = 0.5 * (positions + trial)
            j_mid = linkmodel.jn(*linkmodel.link_quality(mid, jammers))
            j_end = linkmodel.jn(*linkmodel.link_quality(trial, jammers))
            value = -0.5 * (j_mid + j_end)
            value += self.progress_weight * (
                np.linalg.norm(end - destination)
                - np.linalg.norm(nominal[i] - destination)
            )
            for x, y, r in obstacles:
                if _segment_distance(positions[i], end, np.array([x, y])) < r + 1.0:
                    value += self.collision_penalty
            return value

        for i in flagged:
            weight = min(0.95, 0.7 + self.agent_severity(view, i) * 0.25)
            start = positions[i] + (1.0 - weight) * horizon * base[i]
            idle_cost = cost(predicted[i], i)
            best_cost, best = idle_cost - self.deadband, None
            for distance in self.distances:
                for bearing in compass.BEARING_NAMES:
                    end = start + weight * distance * unit[bearing]
                    c = cost(end, i)
                    if c < best_cost:
                        best_cost, best = c, (distance, bearing, end)
            if best is not None:
                commands.append((int(i), best[0], best[1]))
                predicted[i] = best[2]
        return commands

    def _reactive_inputs(self, view):
        """Reactive input C_b, and the base the override blends with."""
        factory = self.default_controller
        state = self.swarm_state
        saved = state.swarm_position
        state.swarm_position = view.swarm_position
        try:
            formation = factory.controllers[ControllerType.FORMATION].compute_control()
            behavior = factory.controllers[ControllerType.BEHAVIOR].compute_control()
        finally:
            state.swarm_position = saved
        if state.Jn_converged:
            return 0.3 * formation + 0.7 * behavior, behavior
        return formation, formation

    def _flagged_agents(self, view, jammers, obstacles):
        n = view.swarm_size
        positions = view.swarm_position
        order = []
        jammed = np.where(view.jamming_affected)[0]
        order += sorted(jammed.tolist(), key=lambda i: -view.jamming_depth[i])
        for x, y, r in jammers:
            r_jam = r * config.JAMMING_RADIUS_MULTIPLIER
            d = np.linalg.norm(positions - np.array([x, y]), axis=1)
            order += [i for i in range(n) if d[i] < r_jam + 10.0]
        for x, y, r in obstacles:
            d = np.linalg.norm(positions - np.array([x, y]), axis=1)
            order += [i for i in range(n) if d[i] < r + 6.0]
        if view_jn(view) < config.PT:
            q = view.communication_qualities_matrix
            a = view.neighbor_agent_matrix
            poor = (a > config.PT) & (q > 0) & (q < config.PT)
            order += sorted(set(np.where(poor)[0].tolist()))
        return list(dict.fromkeys(int(i) for i in order))


class LLMSupervisor(Supervisor):
    """The LLM controller with explicit decoding options and query records."""

    kind = "llm"

    def __init__(self, swarm_state, options, seed, **kwargs):
        super().__init__(swarm_state, **kwargs)
        self.ollama_options = dict(options)
        self.base_seed = int(seed)

    def _request_llm_feedback_async(self):
        with self._decision_lock:
            index = len(self.decisions)
            self.ollama_options["seed"] = self.base_seed * 1000 + index
            self.decisions.append(
                {
                    "kind": self.kind,
                    "step_request": self.step_counter,
                    "t_request": time.time(),
                    "ollama_seed": self.ollama_options["seed"],
                }
            )
        real = self.swarm_state
        self.swarm_state = self.perceived_state()
        try:
            LLMController._request_llm_feedback_async(self)
        finally:
            self.swarm_state = real

    def _on_llm_result(self, info):
        result = info.get("result") or {}
        with self._decision_lock:
            record = self.decisions[-1]
            record.update(
                {
                    "t_start": info["t_start"],
                    "t_end": info["t_end"],
                    "latency_s": info["t_end"] - info["t_start"],
                    "step_response": self.step_counter,
                    "prompt_kind": info["prompt_kind"],
                    "error": info["error"],
                    "text": result.get("response"),
                    "done_reason": result.get("done_reason"),
                    "prompt_eval_count": result.get("prompt_eval_count"),
                    "eval_count": result.get("eval_count"),
                    "total_duration_ns": result.get("total_duration"),
                    "load_duration_ns": result.get("load_duration"),
                    "prompt_eval_duration_ns": result.get("prompt_eval_duration"),
                    "eval_duration_ns": result.get("eval_duration"),
                }
            )


class ExperimentFactory(ControllerFactory):
    """ControllerFactory whose LLM slot holds one of the supervisors above."""

    def __init__(self, swarm_state, make_supervisor):
        self._make_supervisor = make_supervisor
        self.applied_agents = []
        super().__init__(swarm_state)

    def _init_controllers(self):
        self.controllers[ControllerType.FORMATION] = FormationController(
            self.swarm_state
        )
        self.controllers[ControllerType.BEHAVIOR] = BehaviorController(self.swarm_state)
        supervisor = self._make_supervisor(self.swarm_state)
        supervisor.set_default_controller(self)
        self.controllers[ControllerType.LLM] = supervisor

    def _apply_llm_overrides(self, control_inputs, llm_control_inputs, llm_controller):
        self.applied_agents = super()._apply_llm_overrides(
            control_inputs, llm_control_inputs, llm_controller
        )
        return self.applied_agents


def view_jn(view) -> float:
    """J_n of a (possibly perceived) state."""
    return linkmodel.jn(view.communication_qualities_matrix, view.neighbor_agent_matrix)


def _segment_distance(a, b, c) -> float:
    """Distance from point c to segment ab."""
    ab = b - a
    denom = ab @ ab
    t = 0.0 if denom < 1e-12 else float(np.clip((c - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - c))


SUPERVISORS = {
    "none": Supervisor,
    "rule": RuleSupervisor,
    "mpc": MPCSupervisor,
    "llm": LLMSupervisor,
}
