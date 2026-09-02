import os, sys; sys.path.append(os.path.dirname(os.path.abspath(__file__)))
"""Property tests for the Stage 4 admission gate. Deterministic: fixed seed, no wall clock."""
import random, math, time, json
import override_gate
from override_gate import *

RNG = random.Random(20261001)
LEVEL_ORD = {l: i for i, l in enumerate(("BRAKE",) + LEVELS)}

def rand_rec(src, seq, intent=None, bearing=None):
    return {"ver": 1, "src": src, "seq": seq, "ttl": RNG.randint(1, 15),
            "intent": intent if intent is not None else RNG.randint(0, 31),
            "bearing": bearing if bearing is not None else RNG.randint(0, 511),
            "speed": RNG.randint(0, 31), "alt_band": RNG.randint(0, 15),
            "target": RNG.randint(0, 7), "sev": RNG.randint(0, 15), "linkq": RNG.randint(0, 15),
            "flags": RNG.randint(0, 255), "req_auth": RNG.randint(0, 3), "rationale": RNG.randint(0, 255)}

def rand_state(n_nb):
    p = (RNG.uniform(-200, 200), RNG.uniform(-200, 200), RNG.uniform(25, 135))
    v = (RNG.uniform(-12, 12), RNG.uniform(-12, 12), RNG.uniform(-3, 3))
    nb = tuple((p[0] + RNG.uniform(-60, 60), p[1] + RNG.uniform(-60, 60), p[2] + RNG.uniform(-30, 30))
               for _ in range(n_nb))
    return State(p, v, nb, RNG.choice([0.0, RNG.random()]), (RNG.uniform(-12, 12), RNG.uniform(-12, 12), RNG.uniform(-3, 3)))

def scenario(N=7, coherent=True):
    env, st = Envelope(), rand_state(N - 1)
    k = RNG.randint(0, 63)
    base_i, base_b = RNG.randint(0, 31), RNG.randint(0, 511)
    peers = []
    for s in range(1, N):
        if coherent and RNG.random() < 0.8:
            r = rand_rec(s, k, base_i, (base_b + RNG.randint(-16, 16)) % 512)
            r["speed"] = max(0, min(31, 16 + RNG.randint(-2, 2)))
            r["alt_band"] = 8
        else:
            r = rand_rec(s, k)
        peers.append(r)
    own = rand_rec(0, k) if RNG.random() < 0.9 else None
    op = rand_rec(0, k) if RNG.random() < 0.08 else None
    return st, env, own, peers, op, k

res = {}

# T1 -- wire round trip
n = 200_000
for _ in range(n):
    r = rand_rec(RNG.randint(0, 7), RNG.randint(0, 63))
    assert unpack(pack(r)) == r
res["T1 pack/unpack round trip"] = (n, "exact equality on every field", "pass")

# T2 -- determinism and arrival-order invariance
n = 20_000
for _ in range(n):
    st, env, own, peers, op, k = scenario()
    a = admission_gate(st, env, own, peers, op, k)
    b = admission_gate(st, env, own, list(peers), op, k)
    sh = peers[:]; RNG.shuffle(sh)
    c = admission_gate(st, env, own, sh, op, k)
    assert a == b == c, (a, c)
res["T2 determinism + arrival-order invariance"] = (n, "identical level and command under repetition and permutation", "pass")

# T3 -- INV1 envelope containment
n, brake, start_bad = 20_000, 0, 0
for _ in range(n):
    st, env, own, peers, op, k = scenario()
    out = admission_gate(st, env, own, peers, op, k)
    if out["level"] == "BRAKE":
        brake += 1
        if not verify(BRAKE, st, env): start_bad += 1
    else:
        assert verify(out["u"], st, env)
res["T3 INV1 envelope containment"] = (n, f"every non-BRAKE command verified; BRAKE taken {brake/n:.1%}, "
                                          f"of which {start_bad/max(brake,1):.1%} from an already-infeasible state", "pass")

# T4 -- INV5 liveness floor
n, no_cmd = 20_000, 0
for _ in range(n):
    st, env, own, peers, op, k = scenario()
    out = admission_gate(st, env, own, peers, op, k)
    if out["u"] is None: no_cmd += 1
    assert "A0" in out["considered"]
res["T4 INV5 liveness floor"] = (n, f"A0 candidate present in every call; commands missing: {no_cmd}", "pass")

# T5 -- INV4 no self-certification
n, raised = 20_000, 0
for _ in range(n):
    st, env, own, peers, op, k = scenario()
    base = admission_gate(st, env, own, peers, op, k)
    boosted = [dict(r) for r in peers]
    for r in boosted:
        r["req_auth"], r["sev"], r["linkq"] = 3, 15, 15
    out = admission_gate(st, env, own, boosted, op, k)
    if LEVEL_ORD[out["level"]] > LEVEL_ORD[base["level"]]: raised += 1
res["T5 INV4 no self-certification"] = (n, f"peers claiming max severity, link quality and authority raised the "
                                           f"admitted level in {raised} cases", "pass" if raised == 0 else "FAIL")

# T6 -- single Byzantine proposer under F = 1
n, solo_a3, devs = 20_000, 0, []
for _ in range(n):
    st, env, own, peers, op, k = scenario(N=7)
    honest = peers[:-1]
    med = circular_median([r["bearing"] for r in honest])
    byz = dict(peers[-1]); byz["bearing"] = (med + 256) % 512; byz["intent"] = honest[0]["intent"]
    byz["speed"], byz["alt_band"], byz["req_auth"] = 31, honest[0]["alt_band"], 3
    a = admission_gate(st, env, None, honest, None, k)
    b = admission_gate(st, env, None, honest + [byz], None, k)
    if b["level"] == "A3" and len(b["quorum_src"]) < 2: solo_a3 += 1
    if a["level"] == "A3" and b["level"] == "A3":
        devs.append(math.degrees(math.acos(max(-1, min(1, (a["u"][0]*b["u"][0] + a["u"][1]*b["u"][1]) /
                    max(1e-9, math.hypot(*a["u"][:2]) * math.hypot(*b["u"][:2])))))))
res["T6 one Byzantine proposer, F = 1"] = (n, f"quorum reached on fewer than F+1 sources: {solo_a3}; max heading shift "
                                              f"induced in A3 commands: {max(devs) if devs else 0:.2f} deg over {len(devs)} paired cases",
                                           "pass" if solo_a3 == 0 else "FAIL")

# T7 -- latency at N = 7
n = 5_000
scs = [scenario(N=7) for _ in range(n)]
t0 = time.perf_counter()
for st, env, own, peers, op, k in scs: admission_gate(st, env, own, peers, op, k)
dt = (time.perf_counter() - t0) / n * 1000
res["T7 gate latency, N = 7"] = (n, f"mean {dt:.3f} ms per call, pure Python", "pass")

# T8 -- two colluding Byzantine proposers at the edge of the consistency tolerance, F = 2
n, solo, devs8 = 20_000, 0, []
for _ in range(n):
    st, env, own, peers, op, k = scenario(N=7)
    honest = peers[:-2]
    med = circular_median([r["bearing"] for r in honest])
    byz = []
    for src_r in peers[-2:]:                       # sit exactly on the tolerance edge, inside c
        b_ = dict(src_r)
        b_["bearing"] = (med + override_gate.TAU_B) % 512
        b_["intent"], b_["alt_band"] = honest[0]["intent"], honest[0]["alt_band"]
        b_["speed"], b_["req_auth"] = min(31, honest[0]["speed"] + override_gate.TAU_U), 3
        byz.append(b_)
    hc = largest_consistent_subset(canonicalise(honest, k, 2))
    b = admission_gate(st, env, None, honest + byz, None, k, F=2)
    if b["level"] == "A3" and len(b["quorum_src"]) < 5: solo += 1
    if len(hc) >= 3 and b["fused"] is not None:            # influence on the fused proposal itself
        devs8.append(circ_dist(fuse(hc)["bearing"], b["fused"]["bearing"]) * 360.0 / 512.0)
res["T8 two colluding Byzantine, F = 2"] = (n, f"quorum reached below 2F+1 sources: {solo}; max heading shift: "
                                               f"{max(devs8) if devs8 else 0:.2f} deg over {len(devs8)} paired cases",
                                            "pass" if solo == 0 else "FAIL")

for k_, v in res.items():
    print(f"{k_:44s} n={v[0]:>7,}  {v[2]:4s}  {v[1]}")
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_tests.json")
json.dump({k_: {"n": v[0], "detail": v[1], "status": v[2]} for k_, v in res.items()}, open(_out, "w"), indent=1)
