"""Stage 4 reference implementation — deterministic admission gate (L2).

Successor to Algorithm 2 ("LLM Control Override Logic") of the IEEE TVT submission.
Pure functions only: no clocks, no RNG, no I/O, no dict-iteration-order dependence.
Commands are 3D velocity vectors, matching u_i in the TVT formulation.
"""
from dataclasses import dataclass
from itertools import combinations
import math

# ---------------------------------------------------------------- wire format (Stage 3)
FIELDS = [("ver",2),("src",3),("seq",6),("ttl",4),("intent",5),("bearing",9),("speed",5),
          ("alt_band",4),("target",3),("sev",4),("linkq",4),("flags",8),("req_auth",2),("rationale",8)]
NBITS = sum(b for _, b in FIELDS)              # 67
NBYTES = (NBITS + 7) // 8                       # 9

def pack(rec: dict) -> bytes:
    acc = 0
    for name, b in FIELDS:
        v = int(rec[name])
        if not 0 <= v < (1 << b):
            raise ValueError(f"{name}={v} does not fit in {b} bits")
        acc = (acc << b) | v
    acc <<= NBYTES * 8 - NBITS                  # reserved trailing bits = 0
    return acc.to_bytes(NBYTES, "big")

def unpack(buf: bytes) -> dict:
    if len(buf) != NBYTES:
        raise ValueError("bad length")
    acc = int.from_bytes(buf, "big") >> (NBYTES * 8 - NBITS)
    out = {}
    for name, b in reversed(FIELDS):
        out[name] = acc & ((1 << b) - 1)
        acc >>= b
    return {name: out[name] for name, _ in FIELDS}

# ---------------------------------------------------------------- envelope and state
@dataclass(frozen=True)
class Envelope:
    v_max: float = 12.0        # m/s horizontal
    vz_max: float = 3.0        # m/s vertical
    turn_max: float = 0.52     # rad per round (~30 deg)
    z_min: float = 20.0
    z_max: float = 140.0
    d_min: float = 8.0         # m separation
    r_comm: float = 150.0      # m reliable-link range (connectivity)
    dt: float = 1.0            # s round period

@dataclass(frozen=True)
class State:
    pos: tuple                 # (x, y, z)
    vel: tuple                 # (vx, vy, vz)
    neighbours: tuple          # tuple of (x, y, z), predicted one round ahead
    sev: float                 # own jamming severity in [0, 1]
    baseline: tuple            # G_i + M_i from TVT Algorithms 1 and 2 lines 1-2

BAND_M = 8.0                   # metres per altitude band

def decode(rec: dict, st: State, env: Envelope) -> tuple:
    """Quantised record -> desired velocity vector. Total function."""
    th = rec["bearing"] * 2.0 * math.pi / 512.0
    sp = rec["speed"] / 31.0 * env.v_max
    z_target = env.z_min + (rec["alt_band"] + 0.5) * BAND_M
    vz = (z_target - st.pos[2]) / env.dt
    return (sp * math.cos(th), sp * math.sin(th), vz)

def clip(u: tuple, st: State, env: Envelope) -> tuple:
    """Deterministic clipping onto the convex part of the envelope, fixed order."""
    ux, uy, uz = u
    h = math.hypot(ux, uy)                                   # 1. horizontal speed
    if h > env.v_max:
        ux, uy, h = ux * env.v_max / h, uy * env.v_max / h, env.v_max
    uz = max(-env.vz_max, min(env.vz_max, uz))               # 2. vertical rate
    z_pred = st.pos[2] + uz * env.dt                         # 3. altitude band
    if z_pred > env.z_max:
        uz = (env.z_max - st.pos[2]) / env.dt
    elif z_pred < env.z_min:
        uz = (env.z_min - st.pos[2]) / env.dt
    hv = math.hypot(st.vel[0], st.vel[1])                    # 4. turn rate
    if hv > 1e-9 and h > 1e-9:
        a0 = math.atan2(st.vel[1], st.vel[0])
        a1 = math.atan2(uy, ux)
        d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        if abs(d) > env.turn_max:
            a1 = a0 + math.copysign(env.turn_max, d)
            ux, uy = h * math.cos(a1), h * math.sin(a1)
    return (ux, uy, uz)

def verify(u: tuple, st: State, env: Envelope) -> bool:
    """Non-convex constraints: checked, never projected onto.

    Two of them, both complements of balls and therefore non-convex:
      separation   -- no predicted neighbour closer than d_min;
      connectivity -- at least one predicted neighbour within r_comm, enforced only if the
                      vehicle is currently connected (an already-isolated vehicle must be free
                      to manoeuvre back, so the constraint cannot be made unsatisfiable).
    """
    eps = 1e-9
    if math.hypot(u[0], u[1]) > env.v_max + eps: return False
    if abs(u[2]) > env.vz_max + eps: return False
    z = st.pos[2] + u[2] * env.dt
    if z < env.z_min - eps or z > env.z_max + eps: return False
    p = (st.pos[0] + u[0] * env.dt, st.pos[1] + u[1] * env.dt, z)
    for q in st.neighbours:
        if math.dist(p, q) < env.d_min - eps: return False
    if st.neighbours and min(math.dist(st.pos, q) for q in st.neighbours) <= env.r_comm + eps:
        if min(math.dist(p, q) for q in st.neighbours) > env.r_comm + eps: return False
    return True

def tangent_escape(st: State, env: Envelope) -> tuple:
    """TVT deterministic fallback, inherited unchanged: turn 90 deg off current heading."""
    hv = math.hypot(st.vel[0], st.vel[1])
    if hv < 1e-9:
        return (env.v_max * 0.5, 0.0, 0.0)
    return (-st.vel[1] / hv * env.v_max, st.vel[0] / hv * env.v_max, 0.0)

BRAKE = (0.0, 0.0, 0.0)        # liveness floor: always feasible from a feasible state

# ---------------------------------------------------------------- consistency and fusion
def circ_dist(a: int, b: int, m: int = 512) -> int:
    d = abs(a - b) % m
    return min(d, m - d)

TAU_B, TAU_U = 16, 4           # consistency tolerance: bearing units, speed units

def consistent(a: dict, b: dict, tau_b: int = None, tau_u: int = None) -> bool:
    tau_b = TAU_B if tau_b is None else tau_b
    tau_u = TAU_U if tau_u is None else tau_u
    return (a["intent"] == b["intent"]
            and circ_dist(a["bearing"], b["bearing"]) <= tau_b
            and abs(a["speed"] - b["speed"]) <= tau_u
            and abs(a["alt_band"] - b["alt_band"]) <= 1)

def largest_consistent_subset(recs: list) -> list:
    """Exhaustive, deterministic. Ties broken by lexicographically smallest src tuple."""
    best = []
    for k in range(len(recs), 0, -1):
        cands = []
        for combo in combinations(range(len(recs)), k):
            if all(consistent(recs[i], recs[j]) for i, j in combinations(combo, 2)):
                cands.append(combo)
        if cands:
            pick = min(cands, key=lambda c: tuple(recs[i]["src"] for i in c))
            best = [recs[i] for i in pick]
            break
    return best

def circular_median(vals: list, m: int = 512) -> int:
    """Deterministic circular median: minimise total circular distance over observed values."""
    return min(sorted(vals), key=lambda c: (sum(circ_dist(c, v, m) for v in vals), c))

def median(vals: list) -> int:
    s = sorted(vals)
    return s[(len(s) - 1) // 2]                # lower median: deterministic for even counts

def fuse(recs: list) -> dict:
    return {"intent": recs[0]["intent"],
            "bearing": circular_median([r["bearing"] for r in recs]),
            "speed": median([r["speed"] for r in recs]),
            "alt_band": median([r["alt_band"] for r in recs]),
            "target": median([r["target"] for r in recs]),
            "ver": recs[0]["ver"], "src": min(r["src"] for r in recs), "seq": recs[0]["seq"],
            "ttl": 0, "sev": 0, "linkq": 0, "flags": 0, "req_auth": 3, "rationale": recs[0]["rationale"]}

def canonicalise(recs: list, round_k: int, F: int) -> list:
    """Drop stale, expired and equivocating records; keep AT MOST ONE record per source.

    One-per-source is load-bearing, not hygiene: the quorum bound |c| >= 2F+1 counts distinct
    sources, so a source admitted twice in a round would hold two quorum slots (failure case F10).
    Freshest record wins; equal seq with differing payload is equivocation and drops the source.
    """
    fresh = [r for r in recs if r["ttl"] > 0 and (round_k - r["seq"]) % 64 <= 1]
    seen, equivocated = {}, set()
    for r in fresh:
        key = (r["src"], r["seq"])
        if key in seen and seen[key] != r:
            equivocated.add(r["src"])
        seen.setdefault(key, r)
    best = {}
    for (src, seq), r in seen.items():
        if src in equivocated:
            continue
        age = (round_k - seq) % 64
        cur = best.get(src)
        if cur is None or (age, -seq) < (cur[0], -cur[1]):
            best[src] = (age, seq, r)
    return sorted((v[2] for v in best.values()), key=lambda r: (r["src"], r["seq"]))


def trim(recs: list, F: int) -> list:
    """Receiver-side filter: drop the F most extreme bearings about the circular median."""
    if len(recs) <= 2 * F:
        return []
    med = circular_median([r["bearing"] for r in recs])
    ranked = sorted(recs, key=lambda r: (-circ_dist(r["bearing"], med), r["src"]))
    return sorted(ranked[F:], key=lambda r: r["src"])

# ---------------------------------------------------------------- Algorithm 3
LEVELS = ("A0", "A1", "A2", "A3", "A4")

def admission_gate(st: State, env: Envelope, own, peers, operator, round_k: int,
                   F: int = 1, graph_robust: bool = True,
                   use_trim: bool = False, quorum_k=None) -> dict:
    """Deterministic admission gate. Returns the executed command and a full audit record."""
    cand = {}                                                   # level -> velocity
    cand["A0"] = clip(st.baseline, st, env)                                     # L1 baseline
    if st.sev > 0.0:
        cand["A1"] = clip(tangent_escape(st, env), st, env)                     # TVT fallback
    peers = canonicalise(list(peers), round_k, F)
    if own is not None:
        o = canonicalise([own], round_k, F)
        if o and o[0]["req_auth"] >= 2:
            cand["A2"] = clip(decode(o[0], st, env), st, env)                   # local override
    quorum_src, fused = [], None
    if graph_robust and peers:                                                  # Rule G1
        survivors = trim(peers, F) if use_trim else peers
        agree = largest_consistent_subset(survivors)
        if len(agree) >= (2 * F + 1 if quorum_k is None else quorum_k):
            quorum_src = [r["src"] for r in agree]
            fused = fuse(agree)
            cand["A3"] = clip(decode(fused, st, env), st, env)                  # quorum override
    if operator is not None:
        cand["A4"] = clip(decode(operator, st, env), st, env)                   # operator
    for lvl in reversed(LEVELS):                                                # descend to feasible
        if lvl in cand and verify(cand[lvl], st, env):
            return {"level": lvl, "u": cand[lvl], "quorum_src": quorum_src, "fused": fused,
                    "considered": sorted(cand), "fallback": False}
    return {"level": "BRAKE", "u": BRAKE, "quorum_src": quorum_src, "fused": fused,
            "considered": sorted(cand), "fallback": True}
