"""Stage 5 — Byzantine-resilient aggregation.

Replaces the exponential MaxConsistent of Stage 4 with an exact sweep, adds receiver-computed
link-quality weighting, and adds the scalar (W-MSR) path for severity and link maps.
Imports the Stage 4 primitives unchanged so the gate's propositions still apply.
"""
import os, sys, math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from override_gate import (FIELDS, pack, unpack, Envelope, State, decode, clip, verify,
                          circ_dist, consistent, circular_median, median, fuse, canonicalise,
                          largest_consistent_subset, tangent_escape, BRAKE, LEVELS)
import override_gate as OG

MOD = 512

# ------------------------------------------------------------------ exact sweep (replaces 2^n)
def max_consistent_sweep(recs, tau_b=None, tau_u=None):
    """Exact maximum consistent subset in polynomial time.

    Pairwise consistency is an L-infinity box relation: equal `intent`, and coordinatewise
    ranges within (tau_b, tau_u, 1). Axis-parallel boxes have the Helly property, so a set is
    pairwise consistent iff every coordinate's range fits the tolerance -- i.e. iff the set lies
    inside one fixed-size window. Maximum cliques are therefore windows anchored at observed
    coordinate minima, which is a sweep rather than a subset enumeration.
    Tie-break identical to Stage 4: lexicographically least tuple of `src`.
    """
    tau_b = OG.TAU_B if tau_b is None else tau_b
    tau_u = OG.TAU_U if tau_u is None else tau_u
    best, best_n = [], 0
    by_intent = {}
    for r in recs:
        by_intent.setdefault(r["intent"], []).append(r)
    for grp in by_intent.values():
        for ab in {r["bearing"] for r in grp}:                 # arc anchored at a member
            arc = [r for r in grp if (r["bearing"] - ab) % MOD <= tau_b]
            if len(arc) < best_n:
                continue
            for au in {r["speed"] for r in arc}:
                win = [r for r in arc if 0 <= r["speed"] - au <= tau_u]
                if len(win) < best_n:
                    continue
                for aa in {r["alt_band"] for r in win}:
                    cell = [r for r in win if 0 <= r["alt_band"] - aa <= 1]
                    if not cell:
                        continue
                    key = tuple(sorted(r["src"] for r in cell))
                    if len(cell) > best_n or (len(cell) == best_n and key < tuple(sorted(r["src"] for r in best))):
                        best, best_n = sorted(cell, key=lambda r: (r["src"], r["seq"])), len(cell)
    return best

# ------------------------------------------------------------------ receiver-computed weights
def link_quality(st, j_pos, r0=60.0, beta=1.0, v=2.0):
    """TVT-style link quality phi' from LOCALLY measured geometry only (INV4-W).

    No field of any received record enters this function: a sender cannot raise its own weight.
    """
    d = math.dist(st.pos, j_pos)
    return math.exp(-beta * (d / r0) ** v)

def weights(st, recs, kappa=2.0, nb_pos=None):
    """Clip locally observed link quality into [w_min, w_max] with w_max/w_min = kappa."""
    raw = []
    for r in recs:
        p = (nb_pos or {}).get(r["src"])
        raw.append(link_quality(st, p) if p is not None else 1.0)
    lo, hi = min(raw), max(raw)
    if hi - lo < 1e-12:
        return [1.0] * len(recs)
    return [1.0 + (kappa - 1.0) * (x - lo) / (hi - lo) for x in raw]   # in [1, kappa]

def weighted_circular_median(vals, w):
    """Deterministic weighted circular median over observed values; ties by least value."""
    return min(sorted(set(vals)),
               key=lambda c: (sum(wi * circ_dist(c, v) for v, wi in zip(vals, w)), c))

def weighted_median(vals, w):
    order = sorted(range(len(vals)), key=lambda i: (vals[i], i))
    tot = sum(w)
    acc = 0.0
    for i in order:
        acc += w[i]
        if acc >= tot / 2.0:
            return vals[i]
    return vals[order[-1]]

def fuse_weighted(recs, w):
    return {**fuse(recs),
            "bearing": weighted_circular_median([r["bearing"] for r in recs], w),
            "speed": weighted_median([r["speed"] for r in recs], w),
            "alt_band": weighted_median([r["alt_band"] for r in recs], w)}

def quorum_size(F, kappa=1.0):
    """Weighted breakdown condition: F*w_max < (|c|-F)*w_min  <=>  |c| > F(1+kappa)."""
    return int(math.floor(F * (1.0 + kappa))) + 1

# ------------------------------------------------------------------ scalar path (W-MSR)
def wmsr(own, vals, F):
    """One W-MSR round on a scalar. Discards up to F values above and F below `own`."""
    hi = sorted([v for v in vals if v > own], reverse=True)[:F]
    lo = sorted([v for v in vals if v < own])[:F]
    keep = list(vals)
    for v in hi + lo:
        keep.remove(v)
    keep.append(own)
    return sum(keep) / len(keep)

# ------------------------------------------------------------------ Stage 5 gate
def admission_gate_v2(st, env, own, peers, operator, round_k, F=1, graph_robust=True,
                      kappa=1.0, nb_pos=None, sweep=True):
    """Stage 4 gate with the sweep aggregator and optional receiver-weighted fusion."""
    cand = {"A0": clip(st.baseline, st, env)}
    if st.sev > 0.0:
        cand["A1"] = clip(tangent_escape(st, env), st, env)
    peers = canonicalise(list(peers), round_k, F)
    if own is not None:
        o = canonicalise([own], round_k, F)
        if o and o[0]["req_auth"] >= 2:
            cand["A2"] = clip(decode(o[0], st, env), st, env)
    quorum_src, fused = [], None
    if graph_robust and peers:
        agree = (max_consistent_sweep(peers) if sweep else largest_consistent_subset(peers))
        if len(agree) >= quorum_size(F, kappa):
            quorum_src = [r["src"] for r in agree]
            w = weights(st, agree, kappa, nb_pos) if kappa > 1.0 else [1.0] * len(agree)
            fused = fuse_weighted(agree, w) if kappa > 1.0 else fuse(agree)
            cand["A3"] = clip(decode(fused, st, env), st, env)
    if operator is not None:
        cand["A4"] = clip(decode(operator, st, env), st, env)
    for lvl in reversed(LEVELS):
        if lvl in cand and verify(cand[lvl], st, env):
            return {"level": lvl, "u": cand[lvl], "quorum_src": quorum_src, "fused": fused,
                    "considered": sorted(cand), "fallback": False}
    return {"level": "BRAKE", "u": BRAKE, "quorum_src": quorum_src, "fused": fused,
            "considered": sorted(cand), "fallback": True}
