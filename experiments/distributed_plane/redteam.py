"""Stage 5 — enumerated failure cases. One adversary per class, measured against the gate.

Metrics per class: safety violations (executed command outside the envelope), influence on the
fused proposal in degrees, and A3 quorum availability. Reference for influence is the honest
fused command (Stage 4 lesson: executed-command deviation saturates on envelope clipping).
"""
import os, sys, math, random, statistics
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import override_gate as OG
import aggregation as AG
from override_gate import (Envelope, State, circ_dist, circular_median, fuse, canonicalise,
                           verify, clip, decode)
from aggregation import max_consistent_sweep, admission_gate_v2, quorum_size, wmsr

MOD, DEG = 512, 360.0 / 512.0

def scenario(rng, N=7, n_byz=1, cohesion=0.9, close_nb=False):
    env = Envelope()
    pos = (rng.uniform(-200, 200), rng.uniform(-200, 200), rng.uniform(40, 120))
    vel = (rng.uniform(-12, 12), rng.uniform(-12, 12), rng.uniform(-2, 2))
    nb_pos = {s: (pos[0] + rng.uniform(-70, 70), pos[1] + rng.uniform(-70, 70),
                  pos[2] + rng.uniform(-25, 25)) for s in range(1, N)}
    if close_nb:                     # put one neighbour where separation is a live constraint
        th = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(9.0, 14.0)
        nb_pos[1] = (pos[0] + d * math.cos(th), pos[1] + d * math.sin(th), pos[2] + rng.uniform(-2, 2))
    st = State(pos, vel, tuple(nb_pos.values()), rng.choice([0.0, rng.random()]),
               (rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(-2, 2)))
    k = rng.randint(0, 63)
    base_i, base_b, base_s, base_a = rng.randint(0, 3), rng.randint(0, 511), 16, 8
    honest = []
    for s in range(1, N - n_byz):
        coh = rng.random() < cohesion
        honest.append({"ver": 1, "src": s, "seq": k, "ttl": rng.randint(2, 15),
                       "intent": base_i if coh else rng.randint(0, 3),
                       "bearing": (base_b + rng.randint(-8, 8)) % MOD if coh else rng.randint(0, 511),
                       "speed": base_s + rng.randint(-2, 2), "alt_band": base_a,
                       "target": 1, "sev": rng.randint(0, 8), "linkq": rng.randint(4, 12),
                       "flags": 0, "req_auth": 3, "rationale": rng.randint(0, 255)})
    byz_src = list(range(N - n_byz, N))
    return st, env, honest, byz_src, nb_pos, k

def honest_ref(honest, st, env, k, F):
    c = max_consistent_sweep(canonicalise(honest, k, F))
    return fuse(c) if len(c) >= 2 else None

# ---------------------------------------------------------------- adversaries
def a_equivocate(h, byz, st, env, k, rng):
    out = list(h)
    for s in byz:
        base = dict(h[0], src=s, seq=k)
        out.append(dict(base, bearing=(base["bearing"] + 128) % MOD))
        out.append(dict(base, bearing=(base["bearing"] - 128) % MOD))
    return out

def a_replay(h, byz, st, env, k, rng):
    return h + [dict(h[0], src=s, seq=(k - 5) % 64, bearing=(h[0]["bearing"] + 200) % MOD) for s in byz]

def a_selfcert(h, byz, st, env, k, rng):
    return h + [dict(h[0], src=s, seq=k, sev=15, linkq=15, req_auth=3,
                     bearing=(h[0]["bearing"] + 180) % MOD) for s in byz]

def a_extreme(h, byz, st, env, k, rng):
    med = circular_median([r["bearing"] for r in h])
    return h + [dict(h[0], src=s, seq=k, bearing=(med + 256) % MOD, speed=31) for s in byz]

def a_edge_collude(h, byz, st, env, k, rng):
    med = circular_median([r["bearing"] for r in h])
    return h + [dict(h[0], src=s, seq=k, bearing=(med + OG.TAU_B) % MOD,
                     speed=min(31, h[0]["speed"] + OG.TAU_U)) for s in byz]

def a_envelope(h, byz, st, env, k, rng):
    """Aim the whole swarm at a neighbour and at the altitude ceiling."""
    j = st.neighbours[0]
    br = int(math.atan2(j[1] - st.pos[1], j[0] - st.pos[0]) / (2 * math.pi) * MOD) % MOD
    return [dict(r, bearing=br, speed=31, alt_band=15) for r in h] + \
           [dict(h[0], src=s, seq=k, bearing=br, speed=31, alt_band=15) for s in byz]

def a_intent_flood(h, byz, st, env, k, rng):
    return h + [dict(h[0], src=s, seq=k, intent=(4 + i) % 32) for i, s in enumerate(byz)]

def a_fragment(h, byz, st, env, k, rng):
    return h + [dict(h[0], src=s, seq=k, bearing=(h[0]["bearing"] + 64 * (i + 1)) % MOD)
                for i, s in enumerate(byz)]

def a_linkq_lie(h, byz, st, env, k, rng):
    med = circular_median([r["bearing"] for r in h])
    return h + [dict(h[0], src=s, seq=k, linkq=15, sev=15,
                     bearing=(med + OG.TAU_B) % MOD) for s in byz]

def a_burst(h, byz, st, env, k, rng):
    out = []
    for s in byz:
        for q in range(k - 3, k + 1):
            out.append(dict(h[0], src=s, seq=q % 64, bearing=(h[0]["bearing"] + 96) % MOD))
    shuffled = h + out
    rng.shuffle(shuffled)
    return shuffled

def a_spoof(h, byz, st, env, k, rng):
    """Impersonate an honest source: same src, different payload."""
    victim = h[0]["src"]
    return h + [dict(h[0], src=victim, seq=k, bearing=(h[0]["bearing"] + 160) % MOD) for _ in byz]

ATTACKS = [
    ("F1 equivocation (two payloads, one seq)",        "network",  a_equivocate,   1),
    ("F2 replay of a stale record",                    "network",  a_replay,       1),
    ("F3 self-certified severity and authority",       "semantic", a_selfcert,     1),
    ("F4 single extreme proposer",                     "semantic", a_extreme,      1),
    ("F5 colluding pair on the tolerance edge",        "semantic", a_edge_collude, 2),
    ("F6 envelope-violating consensus proposal",       "semantic", a_envelope,     1),
    ("F7 intent flooding (quorum denial)",             "semantic", a_intent_flood, 2),
    ("F8 bearing fragmentation (quorum denial)",       "semantic", a_fragment,     2),
    ("F9 link-quality inflation under weighting",      "semantic", a_linkq_lie,    2),
    ("F10 withhold-then-burst with reordering",        "network",  a_burst,        1),
    ("F11 source impersonation (no MAC)",              "network",  a_spoof,        1),
]

def run(attack, n_byz, n=4000, seed=20261102, kappa=1.0, N=7, F=None,
        graph_robust=True, close_nb=False):
    rng = random.Random(seed)
    F = n_byz if F is None else F
    unsafe, infl, a3, a3_clean, brake = 0, [], 0, 0, 0
    for _ in range(n):
        st, env, honest, byz_src, nb_pos, k = scenario(rng, N=N, n_byz=n_byz, close_nb=close_nb)
        ref = honest_ref(honest, st, env, k, F)
        peers = attack(honest, byz_src, st, env, k, rng)
        out = admission_gate_v2(st, env, None, peers, None, k, F=F, kappa=kappa, nb_pos=nb_pos,
                                graph_robust=graph_robust)
        base = admission_gate_v2(st, env, None, honest, None, k, F=F, kappa=kappa, nb_pos=nb_pos,
                                 graph_robust=graph_robust)
        if out["level"] != "BRAKE" and not verify(out["u"], st, env):
            unsafe += 1
        if out["level"] == "BRAKE":
            brake += 1
        if out["level"] == "A3":
            a3 += 1
            if ref is not None and out["fused"] is not None:
                infl.append(circ_dist(ref["bearing"], out["fused"]["bearing"]) * DEG)
        if base["level"] == "A3":
            a3_clean += 1
    infl.sort()
    return {"trials": n, "N": N, "F": F, "unsafe": unsafe, "a3_rate": a3 / n, "a3_rate_no_attack": a3_clean / n,
            "brake_rate": brake / n,
            "max_influence_deg": round(infl[-1], 2) if infl else 0.0,
            "p99_influence_deg": round(infl[int(0.99 * (len(infl) - 1))], 2) if infl else 0.0,
            "n_influence_pairs": len(infl)}
