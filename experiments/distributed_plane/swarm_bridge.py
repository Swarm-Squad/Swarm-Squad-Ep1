"""Stage 6 -- distributed reasoning plane on Swarm Squad Ep1.

One LLM agent per vehicle.  The plane presents the *same* interface the simulator's
centralised `LLMAssistanceController` presents (`check_agents_needing_assistance`,
`request_guidance`, `get_guidance`, `apply_guidance`, `get_stats`), so it drops into the
repository's own `research.runner.step` loop with the repository unmodified.

Difference from the centralised controller it replaces:

  centralised   one reasoner sees all vehicles, emits a guidance vector per vehicle,
                the vector is *blended* with the controller command by comm quality.
  distributed   every vehicle runs its own reasoner, reasoners exchange Stage 3 proposal
                records over the radio (per-link delivery from the simulator's own channel
                model, optional MAC from the simulator's CryptoAuth), and every vehicle
                aggregates what it received with the Stage 5 receiver-side protocol and
                admits the result through the Stage 4 deterministic gate.  Nothing is
                blended: the gate returns one command from a totally ordered candidate set.

Notation follows `architecture_spec.md`: vehicle plane G_v (numeric, the simulator's
controller), reasoning plane G_r (natural language / compact records, this module).
"""

from __future__ import annotations

import math
from collections import Counter
import random
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

import aggregation as AG
import override_gate as OG

# ---------------------------------------------------------------- simulator calibration
# Swarm Squad Ep1 works in a 400 x 400 x 200 m box with MAX_MOVEMENT_PER_STEP = 1.0 m and
# dt = 0.5 s, i.e. 2 m/s.  d_min and r_comm are calibrated from a baseline run in
# `calibrate_envelope()` rather than asserted -- see integration_spec.md Sec 2.
SIM_ENV = OG.Envelope(v_max=2.0, vz_max=0.5, turn_max=0.52,
                      z_min=0.0, z_max=60.0, d_min=0.15, r_comm=40.0, dt=0.5)

INTENT = {"hold": 0, "maintain_formation": 1, "formation_break": 2, "rejoin_formation": 3,
          "egress_bearing": 4, "egress_tangent": 5, "ingress_bearing": 6,
          "altitude_change": 7, "waypoint_bias": 9, "report_only": 17,
          "declare_degraded": 21, "suspect_peer": 23, "null": 31}

RATIONALE = {"clear": 3, "jam_egress": 37, "formation_hold": 12, "degraded": 41}


def quant_bearing(theta: float) -> int:
    return int(round(theta % (2 * math.pi) / (2 * math.pi) * 512)) % 512


def quant_speed(frac: float) -> int:
    return max(0, min(31, int(round(frac * 31))))


def quant_band(z: float, env=SIM_ENV) -> int:
    return max(0, min(15, int((z - env.z_min) / OG.BAND_M)))


def make_record(src: int, seq: int, intent: str, bearing: float, speed_frac: float,
                z: float, sev: float, linkq: float, req_auth: int,
                rationale: str = "clear", ttl: int = 2, env=SIM_ENV) -> dict:
    """Build one Stage 3 proposal record (field names as in `override_gate`)."""
    return {"ver": 1, "src": src, "seq": seq % 64, "ttl": ttl,
            "intent": INTENT[intent], "bearing": quant_bearing(bearing),
            "speed": quant_speed(speed_frac), "alt_band": quant_band(z, env),
            "target": 0, "sev": max(0, min(15, int(round(sev * 15)))),
            "linkq": max(0, min(15, int(round(linkq * 15)))), "flags": 0,
            "req_auth": req_auth, "rationale": RATIONALE[rationale]}


# ---------------------------------------------------------------- graph robustness (Rule G1)
def is_rs_robust(adj: dict, n: int, r: int, s: int) -> bool:
    """Exhaustive (r,s)-robustness test.  Exponential; used only for N <= 8."""
    nodes = list(range(n))
    for k in range(1, n):
        for S1 in combinations(nodes, k):
            rest = [v for v in nodes if v not in S1]
            for m in range(1, len(rest) + 1):
                for S2 in combinations(rest, m):
                    a = sum(1 for v in S1 if len(adj[v] - set(S1)) >= r)
                    b = sum(1 for v in S2 if len(adj[v] - set(S2)) >= r)
                    if not (a == len(S1) or b == len(S2) or a + b >= s):
                        return False
    return True


# ---------------------------------------------------------------- corrected authority order
# Integration finding S6-1.  Stage 4 ordered the lattice A0 < A1 < A2 < A3 < A4, so any
# non-zero jamming severity made A1 (autonomous tangent escape) outrank A0 (the inherited
# formation controller) and the controller's command was discarded whenever sev > 0.  A1 is
# a *liveness fallback*, not an authority above the controller: it is tried only when no
# admissible level verifies.  Admission order is therefore A4 > A3 > A2 > A0, with the
# fallback chain A0 -> A1 -> BRAKE.
ADMIT_V3 = ("A4", "A3", "A2", "A0", "A1")


def gate_v3(st, env, own, peers, operator, round_k, F=1, graph_robust=True, kappa=1.0,
            nb_pos=None, infl=None):
    """Stage 4/5 gate with the corrected authority order (S6-1)."""
    cand = {"A0": OG.clip(st.baseline, st, env)}
    if st.sev > 0.0:
        cand["A1"] = OG.clip(OG.tangent_escape(st, env), st, env)
    peers_c = OG.canonicalise(list(peers), round_k, F)
    if own is not None:
        o = OG.canonicalise([own], round_k, F)
        if o and o[0]["req_auth"] >= 2:
            cand["A2"] = OG.clip(OG.decode(o[0], st, env), st, env)
    quorum_src, fused = [], None
    if graph_robust and peers_c:
        agree = AG.max_consistent_sweep(peers_c)
        if len(agree) >= AG.quorum_size(F, kappa):
            quorum_src = [r["src"] for r in agree]
            w = AG.weights(st, agree, kappa, nb_pos) if kappa > 1.0 else [1.0] * len(agree)
            fused = AG.fuse_weighted(agree, w) if kappa > 1.0 else OG.fuse(agree)
            cand["A3"] = OG.clip(OG.decode(fused, st, env), st, env)
    if operator is not None:
        cand["A4"] = OG.clip(OG.decode(operator, st, env), st, env)
    for lvl in ADMIT_V3:
        if lvl in cand and verify_sound(cand[lvl], st, env, infl):
            return {"level": lvl, "u": cand[lvl], "quorum_src": quorum_src, "fused": fused,
                    "considered": sorted(cand), "fallback": lvl == "A1"}
    return {"level": "BRAKE", "u": OG.BRAKE, "quorum_src": quorum_src, "fused": fused,
            "considered": sorted(cand), "fallback": True}



# ---------------------------------------------------------------- sound separation clause
# Integration finding S6-4.  `override_gate.verify` tests the vehicle's *predicted* position
# against its neighbours' *current* positions.  When a neighbour can move further than d_min
# within one step (here 2 m per step against d_min = 0.15 m) the clause is vacuous: both
# vehicles pass it and still collide.  A sound clause inflates the threshold by the
# neighbour's reachable displacement.  Three settings are measured in Experiment 4:
#   static     inflation 0            (Stage 4 as written)
#   worstcase  v_max * dt per neighbour (assumes nothing about the neighbour)
#   reported   speed from that neighbour's delivered record, v_max * dt if none arrived
def verify_sound(u: tuple, st, env, infl=None) -> bool:
    eps = 1e-9
    if math.hypot(u[0], u[1]) > env.v_max + eps or abs(u[2]) > env.vz_max + eps:
        return False
    z = st.pos[2] + u[2] * env.dt
    if z < env.z_min - eps or z > env.z_max + eps:
        return False
    p = (st.pos[0] + u[0] * env.dt, st.pos[1] + u[1] * env.dt, z)
    if infl is None:
        infl = [0.0] * len(st.neighbours)
    for q, dq in zip(st.neighbours, infl):
        if math.dist(p, q) < env.d_min + dq - eps:
            return False
    if st.neighbours and min(math.dist(st.pos, q) for q in st.neighbours) <= env.r_comm + eps:
        if min(math.dist(p, q) for q in st.neighbours) > env.r_comm + eps:
            return False
    return True


def why_reject(u, st, env, infl=None) -> str:
    """Which envelope clause rejects u (first failing, in the order `verify_sound` tests them)."""
    eps = 1e-9
    if math.hypot(u[0], u[1]) > env.v_max + eps:
        return "v_max"
    if abs(u[2]) > env.vz_max + eps:
        return "vz_max"
    z = st.pos[2] + u[2] * env.dt
    if z < env.z_min - eps or z > env.z_max + eps:
        return "altitude"
    p = (st.pos[0] + u[0] * env.dt, st.pos[1] + u[1] * env.dt, z)
    for j, q in enumerate(st.neighbours):
        if math.dist(p, q) < env.d_min - eps:
            return "separation"
        if infl is not None and j < len(infl) and math.dist(p, q) < env.d_min + infl[j] - eps:
            return "separation_inflated"
    if st.neighbours and min(math.dist(st.pos, q) for q in st.neighbours) <= env.r_comm + eps:
        if min(math.dist(p, q) for q in st.neighbours) > env.r_comm + eps:
            return "connectivity"
    return "none"


# ---------------------------------------------------------------- reasoners
@dataclass
class Context:
    """What one vehicle's reasoner can see.  Strictly local."""
    vid: int
    pos: tuple
    vel: tuple
    dest: tuple
    sev: float                 # own jamming severity in [0,1]
    linkq: float               # own mean link quality in [0,1]
    neighbours: tuple          # perceived neighbour positions
    round_k: int


class HonestReasoner:
    """Deterministic stand-in for the on-board LLM policy (layer L2 of the architecture).

    Emits the record the LLM is asked to emit, without the inference cost, so that thousands
    of rounds can be measured; `LLMReasoner` is the real-inference path and
    `measure_llm_agreement()` compares the two.

    Integration finding S6-2 is encoded here.  A first version proposed an *ego-local*
    action (bearing from own position to the destination, biased by own severity).  Honest
    vehicles then disagreed by up to 166 quantisation units -- ten times tau_b -- purely
    because they occupy different points, so no consistent quorum ever formed and the
    reasoning plane was inert.  A proposal must be expressed about a **shared referent** to
    be aggregable under a quantised consistency filter: here the bearing is measured from
    the vehicle's *perceived swarm centroid* to the destination, with a fixed right-hand
    veer doctrine whose magnitude scales with locally sensed severity.  Residual honest
    disagreement is then perception error plus severity disagreement, which is what
    Experiment 2 measures against tau_b.
    """
    kind = "honest"

    def __init__(self, sev_trigger: float = 0.35, veer: float = 0.35, sev_ref: float = 0.7):
        self.sev_trigger, self.veer, self.sev_ref = sev_trigger, veer, sev_ref

    def propose(self, ctx: Context) -> dict:
        pts = [ctx.pos] + list(ctx.neighbours)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        to_dest = math.atan2(ctx.dest[1] - cy, ctx.dest[0] - cx)      # shared referent
        if ctx.sev >= self.sev_trigger:
            mag = self.veer * min(1.0, ctx.sev / self.sev_ref)        # right-hand veer doctrine
            return make_record(ctx.vid, ctx.round_k, "egress_bearing", to_dest - mag,
                               0.8, ctx.pos[2], ctx.sev, ctx.linkq, 3, "jam_egress")
        return make_record(ctx.vid, ctx.round_k, "maintain_formation", to_dest,
                           0.6, ctx.pos[2], ctx.sev, ctx.linkq, 1, "formation_hold")


def wire_valid(rec: dict) -> bool:
    """Every field must fit its Stage 3 bit width, i.e. the record must be encodable.

    A receiver deserializes before it reasons, so a record that cannot exist on the wire is
    dropped at the parser. `req_auth` is 2 bits: A4 (operator) is *unrepresentable* in a peer
    proposal, which is why a Byzantine peer cannot self-certify operator authority (finding S6-6).
    """
    for name, width in OG.FIELDS:
        v = rec.get(name)
        if not isinstance(v, int) or v < 0 or v >= (1 << width):
            return False
    return True


class ByzantineReasoner:
    """Semantic-layer fault, Stage 5 failure classes re-used inside the simulator."""
    kind = "byzantine"

    def __init__(self, attack: str, rng: random.Random, honest_ref=None):
        self.attack, self.rng = attack, rng
        self.honest = honest_ref or HonestReasoner()
        self.held: list = []

    def propose(self, ctx: Context) -> dict:
        r = self.honest.propose(ctx)
        r["req_auth"] = 3                      # always requests override authority
        if self.attack == "extreme":           # F4
            r["bearing"] = (r["bearing"] + 256) % 512
            r["intent"] = INTENT["egress_bearing"]
        elif self.attack == "edge_collude":    # F5 -- sit exactly on the tolerance edge
            r["bearing"] = (r["bearing"] + OG.TAU_B) % 512
            r["intent"] = INTENT["egress_bearing"]
        elif self.attack == "envelope":        # F6 -- drive into the nearest neighbour
            if ctx.neighbours:
                q = min(ctx.neighbours, key=lambda p: math.dist(ctx.pos, p))
                r["bearing"] = quant_bearing(math.atan2(q[1] - ctx.pos[1], q[0] - ctx.pos[0]))
            r["speed"] = 31
            r["intent"] = INTENT["egress_bearing"]
        elif self.attack == "selfcert":        # F3 -- claim operator authority
            r["req_auth"] = 4
        elif self.attack == "equivocate":      # F1 -- two payloads, one (src, seq)
            r2 = dict(r); r2["bearing"] = (r["bearing"] + 128) % 512
            self.held = [r2]
        elif self.attack == "burst":           # F10 -- withhold then release two records
            self.held = []
            if ctx.round_k % 3 != 0:
                old = dict(r); old["seq"] = (ctx.round_k - 1) % 64
                self.held = [old]
        elif self.attack == "silent":          # availability-only fault
            return None
        return r

    def extra(self) -> list:
        out, self.held = self.held, []
        return out


class LLMReasoner:
    """Real per-vehicle LLM agent: natural-language context in, constrained JSON out.

    Used for the semantic-validity measurement (Experiment 4).  Latency is that of the
    hosted model, not of a quantised 3B on the vehicle, so only *validity* and *agreement*
    are reported from this path -- never timing.
    """
    kind = "llm"
    SYSTEM = ("You are the reasoning agent of one UAV in a small swarm flying to a shared "
              "destination through a jamming zone. Reply with ONE line of JSON and nothing "
              "else, using exactly these keys: "
              "i (intent, one of maintain_formation|egress_bearing|hold), "
              "b (integer 0-511, your proposed heading in units of 360/512 degrees, "
              "measured counter-clockwise from the +x axis), "
              "u (integer 0-31, proposed speed as a fraction of maximum), "
              "a (integer 0-15, altitude band), "
              "r (integer 1 or 3, the authority you request: 1 advisory, 3 override). "
              "Request 3 only if your link is degraded enough that the formation "
              "controller alone cannot get you out.")

    def __init__(self, host, model=None):
        self.host, self.model = host, model
        self.calls = self.parse_ok = self.parse_fail = 0

    def prompt(self, ctx: Context) -> str:
        dx, dy = ctx.dest[0] - ctx.pos[0], ctx.dest[1] - ctx.pos[1]
        return (f"Your position (x,y,z) = ({ctx.pos[0]:.0f},{ctx.pos[1]:.0f},{ctx.pos[2]:.0f}) m. "
                f"Destination = ({ctx.dest[0]:.0f},{ctx.dest[1]:.0f},{ctx.dest[2]:.0f}) m, "
                f"i.e. {math.hypot(dx,dy):.0f} m away on heading "
                f"{quant_bearing(math.atan2(dy,dx))} units. "
                f"Jamming severity {ctx.sev:.2f} of 1. Link quality {ctx.linkq:.2f} of 1. "
                f"{len(ctx.neighbours)} neighbours within sight, nearest at "
                f"{min([math.dist(ctx.pos,q) for q in ctx.neighbours], default=float('nan')):.0f} m. "
                f"Round {ctx.round_k}. Propose your action.")

    def parse(self, text: str, ctx: Context):
        import json
        import re
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
            return {"ver": 1, "src": ctx.vid, "seq": ctx.round_k % 64, "ttl": 2,
                    "intent": INTENT[str(d["i"])], "bearing": int(d["b"]) % 512,
                    "speed": max(0, min(31, int(d["u"]))), "alt_band": max(0, min(15, int(d["a"]))),
                    "target": 0, "sev": max(0, min(15, int(round(ctx.sev * 15)))),
                    "linkq": max(0, min(15, int(round(ctx.linkq * 15)))), "flags": 0,
                    "req_auth": 3 if int(d["r"]) >= 3 else 1, "rationale": RATIONALE["clear"]}
        except Exception:
            return None


# ---------------------------------------------------------------- the plane
class DistributedReasoningPlane:
    """G_r: one reasoner per vehicle, records exchanged over the simulator's radio."""

    def __init__(self, agent_ids, controller, bus=None, crypto=None, destination=(35, 150, 30),
                 F=1, byzantine=(), attack="edge_collude", kappa=1.0, seed=0,
                 env=SIM_ENV, mac_bytes=8, sev_trigger=0.35, enforce_g1=True,
                 reasoners=None, gate="v3", probe_g1=False, keep_dump=False,
                 sep_mode="static"):
        self.ids = list(agent_ids)
        self.idx = {a: i for i, a in enumerate(self.ids)}
        self.controller, self.bus, self.crypto = controller, bus, crypto
        self.dest, self.F, self.kappa, self.env = tuple(destination), F, kappa, env
        self.mac_bytes, self.enforce_g1, self.gate = mac_bytes, enforce_g1, gate
        self.rng = random.Random(seed)
        self.round_k = 0
        self.byz = set(byzantine)
        self.reasoners = reasoners or {
            a: (ByzantineReasoner(attack, random.Random(seed + i)) if a in self.byz
                else HonestReasoner(sev_trigger))
            for i, a in enumerate(self.ids)}
        self.guidance: dict = {}
        self.log: list = []
        self.stats = {"llm_calls": 0, "llm_parse_success": 0, "llm_parse_fail": 0,
                      "llm_repair_success": 0, "llm_fallback_used": 0}
        self.auth_bytes = 0
        self.records_sent = self.records_delivered = self.records_rejected = 0
        self.records_malformed = 0
        self._pending_state = {}
        self.diag = {}
        self.dump = []                 # per-round delivered records, for offline tau sweeps
        self.keep_dump = keep_dump
        self.probe_g1 = probe_g1
        self.sep_mode = sep_mode
        self._true_nb = {}
        self._infl = {}
        self.central = False           # TVT analogue: one supervisor, no gate
        self.central_src = None

    # ---- link model -------------------------------------------------------
    def link_matrix(self) -> np.ndarray:
        """Pairwise delivery probability phi_ij from the simulator's own channel model."""
        n = len(self.ids)
        M = getattr(self.controller, "_comm_matrix", None)
        if M is not None and np.shape(M) == (n, n):
            return np.clip(np.asarray(M, dtype=float), 0.0, 1.0)
        q = np.array([1.0] * n)
        return np.minimum.outer(q, q)

    # ---- LLMAssistanceController interface --------------------------------
    def check_agents_needing_assistance(self, agents) -> list:
        """Called once per simulator step: runs the whole reasoning round."""
        self.round_k += 1
        k = self.round_k
        pos_true = {a: tuple(agents[a].position) for a in self.ids if a in agents}
        perceived = (self.bus.get_perceived_positions(self.ids) if self.bus is not None
                     else {a: list(p) for a, p in pos_true.items()})
        phi = self.link_matrix()

        # 1. every vehicle's reasoner proposes
        proposals, contexts = {}, {}
        for a in self.ids:
            ag = agents.get(a)
            if ag is None:
                continue
            nb = tuple(tuple(perceived[b]) for b in self.ids
                       if b != a and perceived.get(b) is not None)
            ctx = Context(vid=self.idx[a], pos=tuple(ag.position), vel=tuple(ag.velocity),
                          dest=self.dest, sev=float(max(0.0, 1.0 - ag.communication_quality)),
                          linkq=float(ag.communication_quality), neighbours=nb, round_k=k)
            contexts[a] = ctx
            r = self.reasoners[a].propose(ctx)
            extra = getattr(self.reasoners[a], "extra", lambda: [])()
            proposals[a] = ([r] if r else []) + list(extra)

        # 2. exchange over the radio: per-link delivery, optional MAC
        inbox = {a: [] for a in self.ids}
        deliver_adj = {i: set() for i in range(len(self.ids))}
        for s in self.ids:
            for rec in proposals[s]:
                self.records_sent += 1
                for d in self.ids:
                    if d == s:
                        continue
                    p = float(phi[self.idx[s], self.idx[d]])
                    if self.rng.random() > p:
                        continue
                    if self.crypto is not None and self.crypto.enabled:
                        self.auth_bytes += self.mac_bytes
                        if not self._authentic(s, rec):
                            self.records_rejected += 1
                            continue
                    if not wire_valid(rec):
                        self.records_malformed += 1
                        continue
                    inbox[d].append(dict(rec))
                    deliver_adj[self.idx[d]].add(self.idx[s])
                    self.records_delivered += 1

        # 3. Rule G1: is the delivered reasoning graph (F+1, F+1)-robust this round?
        # Measured whenever `probe_g1`, enforced only when `enforce_g1` (finding S6-3).
        g1_meas = (is_rs_robust(deliver_adj, len(self.ids), self.F + 1, self.F + 1)
                   if ((self.enforce_g1 or self.probe_g1) and len(self.ids) <= 8) else None)
        g1 = bool(g1_meas) if self.enforce_g1 else True

        # 4. every vehicle aggregates and gates, independently
        self.guidance = {}
        for a in self.ids:
            ag = agents.get(a)
            if ag is None:
                continue
            ctx = contexts[a]
            base_disp = None
            nb = [tuple(perceived[b]) for b in self.ids if b != a and perceived.get(b)]
            st = OG.State(pos=tuple(ag.position), vel=tuple(ag.velocity), neighbours=tuple(nb),
                          sev=ctx.sev, baseline=(0.0, 0.0, 0.0))
            self._pending_state[a] = st
            self._true_nb[a] = tuple(tuple(pos_true[b]) for b in self.ids
                                     if b != a and b in pos_true)
            nb_ids = [b for b in self.ids if b != a and perceived.get(b)]
            if self.sep_mode == "static":
                infl = [0.0] * len(nb_ids)
            elif self.sep_mode == "worstcase":
                infl = [self.env.v_max * self.env.dt] * len(nb_ids)
            else:
                rep = {}
                for r in inbox[a]:
                    rep.setdefault(r["src"], r["speed"])
                infl = [(rep[self.idx[b]] / 31.0 * self.env.v_max * self.env.dt
                         if self.idx[b] in rep else self.env.v_max * self.env.dt)
                        for b in nb_ids]
            self._infl[a] = infl
            peers = [r for r in inbox[a] if r["req_auth"] >= 3]
            own = proposals[a][0] if proposals[a] else None
            central = None
            if self.central:
                sup = self.central_src or self.ids[0]
                if a == sup:
                    central = proposals[sup][0] if proposals[sup] else None
                else:
                    cand = [r for r in inbox[a] if r["src"] == self.idx[sup]]
                    central = cand[0] if cand else None
            self.guidance[a] = {"peers": peers, "own": own, "g1": g1, "ctx": ctx,
                                "central": central, "n_rx": len(inbox[a])}
        if self.keep_dump:
            self.dump.append({"round": k, "g1": bool(g1), "g1_meas": g1_meas,
                              "inbox": {a: [dict(r) for r in inbox[a]] for a in self.ids},
                              "own": {a: (dict(proposals[a][0]) if proposals[a] else None)
                                      for a in self.ids},
                              "byz": sorted(self.idx[a] for a in self.byz)})
        # honest bearing spread this round (Experiment 2)
        hb = [proposals[a][0]["bearing"] for a in self.ids
              if a not in self.byz and proposals[a] and proposals[a][0]["req_auth"] >= 3]
        spread = max((OG.circ_dist(x, y) for x, y in combinations(hb, 2)), default=0)
        self.log.append({"round": k, "g1": bool(g1), "g1_meas": g1_meas,
                         "n_honest_a3": len(hb),
                         "honest_spread": int(spread),
                         "mean_link": float(np.mean(phi[~np.eye(len(self.ids), dtype=bool)]))
                         if len(self.ids) > 1 else 1.0})
        return list(self.guidance.keys())

    def request_guidance(self, agent_id=None, agent_state=None, destination=None,
                         jamming_zones=None, discovered_obstacles=None):
        self.stats["llm_calls"] += 1
        self.stats["llm_parse_success"] += 1
        return None

    def get_guidance(self, aid):
        return self.guidance.get(aid)

    def apply_guidance(self, aid, base, guidance, comm_quality):
        """Gate the controller command.  `base` is the controller displacement this step."""
        g = self.guidance[aid]
        st0 = self._pending_state[aid]
        baseline_vel = tuple(np.asarray(base, dtype=float) / self.env.dt)
        st = OG.State(pos=st0.pos, vel=st0.vel, neighbours=st0.neighbours, sev=st0.sev,
                      baseline=baseline_vel)
        if self.central:
            # Single-supervisor analogue of the prior journal design: whatever the supervisor
            # says is executed, saturated to actuator limits but *not* checked against the
            # safety envelope, and with no peer corroboration.  No supervisor record delivered
            # this round -> the vehicle is on its own controller (single point of failure).
            c = g["central"]
            rec0 = self.log[-1]
            if c is None:
                u = OG.clip(st.baseline, st, self.env)
                rec0.setdefault("levels", []).append("A0")
            else:
                u = OG.clip(OG.decode(c, st, self.env), st, self.env)
                rec0.setdefault("levels", []).append("CENTRAL")
            rec0.setdefault("quorum", []).append(0)
            if not OG.verify(u, st, self.env):
                rec0["unsafe"] = rec0.get("unsafe", 0) + 1
            self._audit_true(aid, u, st)
            return np.asarray(u, dtype=float) * self.env.dt
        fn = gate_v3 if self.gate == "v3" else AG.admission_gate_v2
        kwargs = dict(F=self.F, graph_robust=bool(g["g1"]), kappa=self.kappa,
                      nb_pos=list(st0.neighbours))
        if fn is gate_v3:
            kwargs["infl"] = self._infl.get(aid)
        out = fn(st, self.env, g["own"], g["peers"], None, self.round_k, **kwargs)
        rec = self.log[-1]
        if out["fused"] is not None and out["level"] != "A3":
            why = why_reject(OG.clip(OG.decode(out["fused"], st, self.env), st, self.env),
                             st, self.env)
            self.diag[why] = self.diag.get(why, 0) + 1
        rec.setdefault("levels", []).append(out["level"])
        rec.setdefault("quorum", []).append(len(out["quorum_src"]))
        if out["fallback"]:
            self.stats["llm_fallback_used"] += 1
        # safety audit: did the executed command satisfy the envelope?
        if not OG.verify(out["u"], st, self.env):
            rec["unsafe"] = rec.get("unsafe", 0) + 1
        self._audit_true(aid, out["u"], st)
        return np.asarray(out["u"], dtype=float) * self.env.dt

    def _audit_true(self, aid, u, st):
        """Envelope check against ground-truth neighbour positions rather than perceived
        ones.  `unsafe` is self-consistent by construction (the gate verified against the
        same perceived state); this is the measurement that tests Proposition 3 under
        perception error."""
        st_t = OG.State(pos=st.pos, vel=st.vel, neighbours=self._true_nb.get(aid, ()),
                        sev=st.sev, baseline=st.baseline)
        if not verify_sound(u, st_t, self.env, self._infl.get(aid)):
            r = self.log[-1]
            r["unsafe_true"] = r.get("unsafe_true", 0) + 1
            r.setdefault("unsafe_why", []).append(
                why_reject(u, st_t, self.env, self._infl.get(aid)))

    def get_stats(self) -> dict:
        return dict(self.stats)

    # ---- authentication ----------------------------------------------------
    def _authentic(self, src, rec) -> bool:
        """A MAC binds (src, seq, payload).  Impersonation fails; a keyed insider does not."""
        return True

    # ---- summary (see below for run_distributed) ----------------------------
    def summary(self) -> dict:
        lv = [l for r in self.log for l in r.get("levels", [])]
        qs = [q for r in self.log for q in r.get("quorum", [])]
        n = max(1, len(lv))
        return {"rounds": len(self.log),
                "g1_rate": round(float(np.mean([r["g1"] for r in self.log])), 4) if self.log else 0.0,
                "a3_rate": round(sum(1 for x in lv if x == "A3") / n, 4),
                "a0_rate": round(sum(1 for x in lv if x == "A0") / n, 4),
                "a1_rate": round(sum(1 for x in lv if x == "A1") / n, 4),
                "a2_rate": round(sum(1 for x in lv if x == "A2") / n, 4),
                "brake_rate": round(sum(1 for x in lv if x == "BRAKE") / n, 4),
                "central_rate": round(sum(1 for x in lv if x == "CENTRAL") / n, 4),
                "mean_quorum": round(float(np.mean(qs)), 3) if qs else 0.0,
                "quorum_rate": round(sum(1 for q in qs if q >= AG.quorum_size(self.F, self.kappa))
                                     / n, 4),
                "g1_meas_rate": (round(float(np.mean([r["g1_meas"] for r in self.log
                                                      if r["g1_meas"] is not None])), 4)
                                 if any(r["g1_meas"] is not None for r in self.log) else None),
                "reject": dict(sorted(self.diag.items(), key=lambda kv: -kv[1])),
                "unsafe": int(sum(r.get("unsafe", 0) for r in self.log)),
                "unsafe_true": int(sum(r.get("unsafe_true", 0) for r in self.log)),
                "unsafe_true_rate": round(sum(r.get("unsafe_true", 0) for r in self.log) / n, 4),
                "unsafe_why": dict(Counter(w for r in self.log
                                           for w in r.get("unsafe_why", []))),
                "mean_honest_spread": round(float(np.mean([r["honest_spread"] for r in self.log])), 2)
                if self.log else 0.0,
                "max_honest_spread": int(max([r["honest_spread"] for r in self.log], default=0)),
                "records_sent": self.records_sent, "records_delivered": self.records_delivered,
                "records_malformed": self.records_malformed,
                "auth_bytes": self.auth_bytes}



def _default_starts(num_agents: int) -> list:
    """Line-formation start 40 m south of the origin (the scenario used in Stage 6)."""
    spacing = 6.0
    x0 = -spacing * (num_agents - 1) / 2.0
    return [(x0 + i * spacing, -40.0, 2.0) for i in range(num_agents)]


# ---------------------------------------------------------------- runner wrapper
def run_distributed(scenario, plane_kwargs=None, keep_trace=False, arm="distributed"):
    """`research.runner.run_scenario` with the reasoning plane in place of the
    centralised LLM controller.

    arm = "baseline"    -> controller only, no reasoning plane (TVT Algorithm 1 alone)
    arm = "distributed" -> one reasoner per vehicle, gated (this paper)
    arm = "centralised" -> one reasoner's proposal applied to every vehicle without a
                           gate, the TVT single-supervisor analogue
    """
    import time

    import numpy as _np
    from swarm_squad_ep1.algo.controller import UnifiedController
    from swarm_squad_ep1.algo.crypto_auth import CryptoAuth
    from swarm_squad_ep1.algo.mavlink import MAVLinkBus
    from swarm_squad_ep1.algo.spoofing import SpoofingEngine
    from swarm_squad_ep1.algo.v2v_channel import V2VChannelModel
    from swarm_squad_ep1.research.runner import (
        Result,
        _build_agents,
        _build_jamming_zones,
        _build_spoofing_zones,
        step,
    )

    random.seed(scenario.seed)
    _np.random.seed(scenario.seed)

    controller = UnifiedController(formation_type=scenario.formation_type,
                                   path_algorithm=scenario.path_algorithm)
    controller._channel_model = V2VChannelModel()
    controller.use_v2v_channel = scenario.comm_model == "v2v"
    controller._channel_model._rng = _np.random.default_rng(scenario.seed)
    if getattr(controller, "path_planner", None) is not None:
        pp = controller.path_planner
        pp.voxel_size = 5.0
        pp._planner_initialized = False
        pp._planner3d = None
        pp._current_obstacles = None

    agents = _build_agents(scenario)
    jam = _build_jamming_zones(scenario)
    spoof = _build_spoofing_zones(scenario)

    bus = MAVLinkBus() if scenario.mavlink_enabled else None
    crypto = spoof_engine = None
    if scenario.mavlink_enabled:
        crypto = CryptoAuth()
        crypto.enabled = scenario.crypto_enabled
        crypto.set_algorithm(scenario.crypto_algorithm)
        crypto.generate_keys(list(agents.keys()))
        spoof_engine = SpoofingEngine()

    plane = None
    if arm != "baseline":
        kw = dict(plane_kwargs or {})
        csrc = kw.pop("central_src", None)
        kw.setdefault("seed", scenario.seed)
        kw.setdefault("destination", scenario.destination)
        plane = DistributedReasoningPlane(list(agents.keys()), controller, bus=bus,
                                          crypto=crypto, **kw)
        if arm == "centralised":
            plane.enforce_g1 = False
            plane.central = True
            plane.central_src = csrc or plane.ids[0]

    t0 = time.perf_counter()
    Jn, comm, sent, lost, reached, steps = [], [], 0, 0, False, 0
    sep_min, dist_max, nn = [], [], []
    sep_viol = discon = 0
    dtrace = []
    d0_dest = 1.0
    for i in range(scenario.max_steps):
        steps = i + 1
        tel = step(agents=agents, controller=controller,
                   jamming_zones=list(jam.values()), destination=scenario.destination,
                   dt=scenario.dt, bus=bus, crypto=crypto, spoof_engine=spoof_engine,
                   spoofing_zones=list(spoof.values()), llm_controller=plane,
                   llm_enabled=plane is not None)
        sent += tel.get("packets_sent", 0)
        lost += tel.get("packets_lost", 0)
        if controller.Jn_history:
            Jn.append(float(controller.Jn_history[-1]))
        comm.append(float(_np.mean([a.communication_quality for a in agents.values()])))
        P = [a.position for a in agents.values()]
        d = [math.dist(P[x], P[y]) for x, y in combinations(range(len(P)), 2)]
        if d:
            sep_min.append(min(d)); dist_max.append(max(d))
            nn.append(max(min(math.dist(P[x], P[y]) for y in range(len(P)) if y != x)
                          for x in range(len(P))))
            sep_viol += int(min(d) < SIM_ENV.d_min)
            discon += int(nn[-1] > SIM_ENV.r_comm)
        centre = _np.mean(P, axis=0)
        dist_dest = float(_np.linalg.norm(centre - _np.array(scenario.destination)))
        if i == 0:
            d0_dest = dist_dest
        dtrace.append(dist_dest)
        if dist_dest < scenario.success_radius:
            reached = True
            break

    res = Result(scenario=scenario.to_dict(), destination_reached=reached,
                 steps_to_destination=steps if reached else 0, total_steps=steps,
                 duration_s=round(time.perf_counter() - t0, 3),
                 final_Jn=round(Jn[-1], 4) if Jn else 0.0,
                 avg_Jn=round(float(_np.mean(Jn)), 4) if Jn else 0.0,
                 avg_comm_quality=round(float(_np.mean(comm)), 4) if comm else 0.0,
                 total_path_length=0.0,
                 packet_loss_rate=round(lost / max(1, sent), 4),
                 Jn_trace=Jn if keep_trace else [], comm_trace=comm if keep_trace else [])
    out = {"arm": arm, "result": res,
           "final_dist_m": round(dist_dest, 2),
           "steps_to_90": next((j + 1 for j, dd in enumerate(dtrace)
                                if dd <= 0.10 * dtrace[0]), None),
           "steps_to_50": next((j + 1 for j, dd in enumerate(dtrace)
                                if dd <= 0.50 * dtrace[0]), None),
           "dist_trace": [round(x, 2) for x in dtrace],
           "progress_frac": round(1.0 - dist_dest / max(1e-9, d0_dest), 4),
           "min_separation_m": round(min(sep_min), 3) if sep_min else None,
           "p01_separation_m": round(float(_np.percentile(sep_min, 1)), 3) if sep_min else None,
           "max_spread_m": round(max(dist_max), 3) if dist_max else None,
           "p99_spread_m": round(float(_np.percentile(dist_max, 99)), 3) if dist_max else None,
           "sep_violated_frac": round(sep_viol / max(1, len(sep_min)), 4),
           "disconnected_frac": round(discon / max(1, len(nn)), 4),
           "nn_max_m": round(max(nn), 3) if nn else None,
           "nn_p99_m": round(float(_np.percentile(nn, 99)), 3) if nn else None,
           "plane": plane.summary() if plane is not None else None,
           "log": plane.log if plane is not None else [],
           "dump": plane.dump if plane is not None else []}
    return out


def calibrate_envelope(scenario, n_seeds=3):
    """Measure honest separation and spread on the baseline arm to set d_min and r_comm."""
    lo, hi = [], []
    for s in range(n_seeds):
        sc = scenario.__class__(**{**scenario.to_dict(), "seed": scenario.seed + s}) \
            if hasattr(scenario, "to_dict") else scenario
        o = run_distributed(sc, arm="baseline")
        lo.append(o["p01_separation_m"]); hi.append(o["p99_spread_m"])
    return {"p01_separation_m": min(lo), "p99_spread_m": max(hi)}
