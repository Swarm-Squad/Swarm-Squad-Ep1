"""
Headless scenario runner.

 Mirrors the pipeline of ``swarm_squad_ep1.simulation.api.run_simulation_loop`` but
without any FastAPI / global-state coupling:

    broadcast -> spoofing -> packet-loss -> crypto -> perceive
       -> controller.compute_commands -> (optional LLM guidance)
       -> move agents -> update quality

The ``step(...)`` function is a self-contained transition so it can be
called from unit tests and experiment loops alike.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from swarm_squad_ep1.algo.base import JammingZone, ObstacleType
from swarm_squad_ep1.algo.controller import UnifiedController
from swarm_squad_ep1.algo.crypto_auth import CryptoAuth
from swarm_squad_ep1.algo.mavlink import MAVLinkBus
from swarm_squad_ep1.algo.spoofing import SpoofingEngine, SpoofingZone, SpoofType
from swarm_squad_ep1.algo.v2v_channel import V2VChannelModel
from swarm_squad_ep1.research.scenarios import Scenario
from swarm_squad_ep1.simulation.agents import AgentState

# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class Result:
    scenario: dict
    destination_reached: bool = False
    steps_to_destination: int = 0
    total_steps: int = 0
    duration_s: float = 0.0

    # Aggregated comm-quality / formation metrics
    final_Jn: float = 0.0
    avg_Jn: float = 0.0
    avg_comm_quality: float = 0.0
    total_path_length: float = 0.0

    # Packet loss / attack metrics
    packet_loss_rate: float = 0.0
    spoof_tp: int = 0
    spoof_fp: int = 0
    spoof_fn: int = 0
    spoof_tn: int = 0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    precision: float = 0.0

    # LLM reliability
    llm_calls: int = 0
    llm_parse_success: int = 0
    llm_parse_fail: int = 0
    llm_repair_success: int = 0
    llm_fallback_used: int = 0

    # Per-step traces (optional, small by default)
    Jn_trace: list[float] = field(default_factory=list)
    comm_trace: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @property
    def llm_parse_fail_rate(self) -> float:
        n = max(1, self.llm_calls)
        return self.llm_parse_fail / n

    @property
    def llm_repair_success_rate(self) -> float:
        n = max(1, self.llm_parse_fail + self.llm_repair_success)
        return self.llm_repair_success / n


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------


def _build_agents(scenario: Scenario) -> dict[str, AgentState]:
    agents: dict[str, AgentState] = {}
    positions = scenario.agent_init_positions or [
        (i * 5.0, -40.0, 2.0) for i in range(scenario.num_agents)
    ]
    for i, pos in enumerate(positions[: scenario.num_agents]):
        aid = f"agent{i + 1}"
        state = AgentState(agent_id=aid, position=list(pos), formation_role=None)
        state._prev_pos = list(pos)
        agents[aid] = state
    return agents


def _build_jamming_zones(scenario: Scenario) -> dict[str, JammingZone]:
    zones: dict[str, JammingZone] = {}
    for i, spec in enumerate(scenario.jamming_zones):
        try:
            t = ObstacleType(spec.obstacle_type)
        except ValueError:
            t = ObstacleType.LOW_JAM
        zid = f"jam_{i + 1}"
        zones[zid] = JammingZone(
            id=zid,
            center=list(spec.center),
            radius=spec.radius,
            active=True,
            obstacle_type=t,
        )
    return zones


def _build_spoofing_zones(scenario: Scenario) -> dict[str, SpoofingZone]:
    zones: dict[str, SpoofingZone] = {}
    for i, spec in enumerate(scenario.spoofing_zones):
        try:
            t = SpoofType(spec.spoof_type)
        except ValueError:
            t = SpoofType.PHANTOM
        zid = f"spoof_{i + 1}"
        zones[zid] = SpoofingZone(
            id=zid,
            center=list(spec.center),
            radius=spec.radius,
            active=True,
            spoof_type=t,
            phantom_count=spec.phantom_count,
            falsification_magnitude=spec.falsification_magnitude,
            coordinate_vector=list(spec.coordinate_vector),
        )
    return zones


# --------------------------------------------------------------------------
# The pure step()
# --------------------------------------------------------------------------


def step(
    agents: dict[str, AgentState],
    controller: UnifiedController,
    jamming_zones: list[JammingZone],
    destination: tuple[float, float, float],
    dt: float,
    bus: Optional[MAVLinkBus] = None,
    crypto: Optional[CryptoAuth] = None,
    spoof_engine: Optional[SpoofingEngine] = None,
    spoofing_zones: Optional[list[SpoofingZone]] = None,
    llm_controller=None,
    llm_enabled: bool = False,
) -> dict:
    """Advance the simulation one tick. Mutates ``agents`` in place.

    Returns a small telemetry dict useful for metrics aggregation.
    """
    tel: dict = {
        "packets_sent": 0,
        "packets_lost": 0,
    }

    # ---------- MAVLink pipeline ----------
    perceived_positions = None
    if bus is not None:
        bus.clear_queue()
        for aid, agent in agents.items():
            msg = bus.broadcast(aid, agent.position, agent.velocity, agent.heading)
            if crypto and crypto.enabled:
                crypto.sign_message(msg)

        active_spoof = [z for z in (spoofing_zones or []) if z.active]
        if active_spoof and spoof_engine is not None:
            agent_pos = {aid: list(a.position) for aid, a in agents.items()}
            bus.set_messages(
                spoof_engine.process(bus.get_messages(), active_spoof, agent_pos)
            )

        pre_count = len(bus.get_messages())
        comm_quals = {aid: float(a.communication_quality) for aid, a in agents.items()}
        bus.apply_packet_loss(comm_quals)
        post_count = len(bus.get_messages())
        tel["packets_sent"] = pre_count
        tel["packets_lost"] = max(0, pre_count - post_count)

        if crypto is not None:
            bus.set_messages(crypto.filter_messages(bus.get_messages()))

        bus.build_perceived_state(agents)
        perceived_positions = bus.get_perceived_positions(list(agents.keys()))

    # ---------- controller ----------
    commands = controller.compute_commands(
        agents=agents,
        destination=tuple(destination),
        jamming_zones=jamming_zones,
        dt=dt,
        perceived_positions=perceived_positions,
    )

    # ---------- LLM guidance (optional) ----------
    if llm_enabled and llm_controller is not None:
        try:
            needing = llm_controller.check_agents_needing_assistance(agents)
            for aid in needing:
                agent = agents[aid]
                llm_controller.request_guidance(
                    agent_id=aid,
                    agent_state=agent,
                    destination=tuple(destination),
                    jamming_zones=jamming_zones,
                    discovered_obstacles=controller.get_discovered_obstacles()
                    if hasattr(controller, "get_discovered_obstacles")
                    else [],
                )
                guidance = llm_controller.get_guidance(aid)
                if guidance and aid in commands:
                    cmd = commands[aid]
                    if cmd.target_position:
                        current = np.array(agent.position)
                        target = np.array(cmd.target_position)
                        base = target - current
                        blended = llm_controller.apply_guidance(
                            aid,
                            base,
                            guidance,
                            agent.communication_quality,
                        )
                        mag = np.linalg.norm(blended)
                        if mag > 2.0:
                            blended = blended * (2.0 / mag)
                        new_target = np.clip(
                            current + blended,
                            controller.bounds_min,
                            controller.bounds_max,
                        )
                        cmd.target_position = new_target.tolist()
        except Exception as e:
            print(f"[research.runner] LLM step skipped: {e}")

    # ---------- apply commands ----------
    for aid, cmd in commands.items():
        if aid in agents and cmd.target_position:
            a = agents[aid]
            a.position = [float(p) for p in cmd.target_position]
            a.velocity = [float(v) for v in (cmd.velocity or [0, 0, 0])]
            a.heading = float(cmd.heading or 0.0)

    # ---------- quality update (mirrors _update_agent_jamming_status) ----------
    pairwise = controller.get_all_agent_comm_quality()
    for aid, agent in agents.items():
        max_jam = 0.0
        for z in jamming_zones:
            if z.active:
                max_jam = max(max_jam, float(z.get_jamming_level(agent.position)))
        agent.jammed = bool(max_jam > 0.1)
        if aid in pairwise:
            agent.communication_quality = float(pairwise[aid])
        else:
            agent.communication_quality = float(1.0 - max_jam * 0.8)

    return tel


# --------------------------------------------------------------------------
# Run a whole scenario
# --------------------------------------------------------------------------


def run_scenario(
    scenario: Scenario,
    keep_trace: bool = True,
    verbose: bool = False,
) -> Result:
    """Execute one scenario end-to-end and return aggregate metrics."""
    random.seed(scenario.seed)
    np.random.seed(scenario.seed)

    controller = UnifiedController(
        formation_type=scenario.formation_type,
        path_algorithm=scenario.path_algorithm,
    )
    controller._channel_model = V2VChannelModel()
    controller.use_v2v_channel = scenario.comm_model == "v2v"
    controller._channel_model._rng = np.random.default_rng(scenario.seed)

    # Use a coarser grid for headless runs (voxel 5m vs 2m cuts memory ~15x
    # and makes A* feasible for batch experiment matrices).
    if hasattr(controller, "path_planner") and controller.path_planner is not None:
        pp = controller.path_planner
        pp.voxel_size = 5.0
        pp._planner_initialized = False
        pp._planner3d = None
        pp._current_obstacles = None

    agents = _build_agents(scenario)
    jam_zones = _build_jamming_zones(scenario)
    spoof_zones = _build_spoofing_zones(scenario)

    bus = MAVLinkBus() if scenario.mavlink_enabled else None
    crypto: Optional[CryptoAuth] = None
    spoof_engine: Optional[SpoofingEngine] = None
    if scenario.mavlink_enabled:
        crypto = CryptoAuth()
        crypto.enabled = scenario.crypto_enabled
        crypto.set_algorithm(scenario.crypto_algorithm)
        crypto.generate_keys(list(agents.keys()))
        spoof_engine = SpoofingEngine()

    llm_ctl = None
    if scenario.llm_assistance_enabled:
        try:
            from swarm_squad_ep1.algo.llm_controller import LLMAssistanceController

            llm_ctl = LLMAssistanceController(enabled=True)
        except Exception as e:
            if verbose:
                print(f"[research.runner] LLM controller unavailable: {e}")
            scenario.llm_assistance_enabled = False

    t0 = time.perf_counter()
    Jn_trace: list[float] = []
    comm_trace: list[float] = []
    total_sent = 0
    total_lost = 0
    reached = False
    steps_done = 0

    for step_idx in range(scenario.max_steps):
        steps_done = step_idx + 1
        tel = step(
            agents=agents,
            controller=controller,
            jamming_zones=list(jam_zones.values()),
            destination=scenario.destination,
            dt=scenario.dt,
            bus=bus,
            crypto=crypto,
            spoof_engine=spoof_engine,
            spoofing_zones=list(spoof_zones.values()),
            llm_controller=llm_ctl,
            llm_enabled=scenario.llm_assistance_enabled,
        )
        total_sent += tel.get("packets_sent", 0)
        total_lost += tel.get("packets_lost", 0)

        if controller.Jn_history:
            Jn_trace.append(float(controller.Jn_history[-1]))
        comm = (
            np.mean([a.communication_quality for a in agents.values()])
            if agents
            else 0.0
        )
        comm_trace.append(float(comm))

        center = np.mean([a.position for a in agents.values()], axis=0)
        dist = float(np.linalg.norm(center - np.array(scenario.destination)))
        if dist < scenario.success_radius:
            reached = True
            break

    duration = time.perf_counter() - t0

    path_len = 0.0
    for aid, agent in agents.items():
        p0 = controller._agent_paths.get(aid, [])
        for i in range(1, len(p0)):
            path_len += float(np.linalg.norm(np.array(p0[i]) - np.array(p0[i - 1])))

    res = Result(
        scenario=scenario.to_dict(),
        destination_reached=reached,
        steps_to_destination=steps_done if reached else 0,
        total_steps=steps_done,
        duration_s=round(duration, 3),
        final_Jn=round(Jn_trace[-1], 4) if Jn_trace else 0.0,
        avg_Jn=round(float(np.mean(Jn_trace)), 4) if Jn_trace else 0.0,
        avg_comm_quality=round(float(np.mean(comm_trace)), 4) if comm_trace else 0.0,
        total_path_length=round(path_len, 2),
        packet_loss_rate=round(total_lost / max(1, total_sent), 4),
        Jn_trace=Jn_trace if keep_trace else [],
        comm_trace=comm_trace if keep_trace else [],
    )

    if crypto is not None:
        s = crypto.stats
        res.spoof_tp = s.tp
        res.spoof_fp = s.fp
        res.spoof_fn = s.fn
        res.spoof_tn = s.tn
        pos = s.tp + s.fn
        neg = s.fp + s.tn
        pred_pos = s.tp + s.fp
        res.detection_rate = round(s.tp / pos, 4) if pos > 0 else 0.0
        res.false_positive_rate = round(s.fp / neg, 4) if neg > 0 else 0.0
        res.precision = round(s.tp / pred_pos, 4) if pred_pos > 0 else 0.0

    if llm_ctl is not None:
        st = llm_ctl.get_stats()
        res.llm_calls = st["llm_calls"]
        res.llm_parse_success = st["llm_parse_success"]
        res.llm_parse_fail = st["llm_parse_fail"]
        res.llm_repair_success = st["llm_repair_success"]
        res.llm_fallback_used = st["llm_fallback_used"]

    if verbose:
        print(
            f"[{scenario.name}] seed={scenario.seed} reached={reached} "
            f"steps={steps_done} Jn_final={res.final_Jn} "
            f"det_rate={res.detection_rate} fpr={res.false_positive_rate}"
        )

    return res
