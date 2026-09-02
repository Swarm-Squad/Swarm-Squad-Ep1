#!/usr/bin/env python3
"""Stage 6 demo: one reasoning agent per vehicle on the Swarm-Squad-Ep1 simulator.

Runs the double multi-agent loop end to end -- N vehicle agents exchanging numerical
control state through the simulator's own V2V channel, and N reasoning agents exchanging
12-byte quantised proposal records through the same lossy links, with every proposal
admitted or rejected by the deterministic override gate before it can move a vehicle.

Arms
    baseline     inherited UnifiedController only (no reasoning plane)
    distributed  one reasoning agent per vehicle, gated aggregation   [Stage 6 contribution]
    centralised  one reasoning agent for the whole swarm, ungated     [prior TVT configuration]

Examples
    python demo_stage6.py --n 5 --arm distributed --byzantine agent3 --attack edge_collude
    python demo_stage6.py --n 7 --arm baseline --no-jamming
    python demo_stage6.py --n 3 --arm centralised --central-src agent3
    python demo_stage6.py --n 5 --arm distributed --sep-mode static   # unsound clause, S6-4

Requires the simulator package on the path:
    pip install pathfinding3d python-dotenv fastapi
    git clone https://github.com/Swarm-Squad/Swarm-Squad-Ep1
    PYTHONPATH=Swarm-Squad-Ep1/src python demo_stage6.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5, help="swarm size, 3-7 (default 5)")
    p.add_argument("--arm", default="distributed",
                   choices=["baseline", "distributed", "centralised"])
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--f", type=int, default=1, dest="F", help="Byzantine budget F (default 1)")
    p.add_argument("--byzantine", nargs="*", default=[],
                   help="vehicle ids whose reasoning agent is Byzantine, e.g. agent3")
    p.add_argument("--attack", default="edge_collude",
                   help="Byzantine strategy (see swarm_bridge.ByzantineReasoner)")
    p.add_argument("--central-src", default="agent1",
                   help="which vehicle's reasoner is the supervisor in the centralised arm")
    p.add_argument("--sep-mode", default="worstcase",
                   choices=["static", "worstcase", "reported"],
                   help="separation-clause inflation (S6-4; 'static' is Stage 4 as written)")
    p.add_argument("--probe-g1", action="store_true",
                   help="measure (2F+1, F+1)-robustness of the delivered graph each round")
    p.add_argument("--enforce-g1", action="store_true",
                   help="require (2F+1, F+1)-robustness of the delivered graph (S6-3)")
    p.add_argument("--no-jamming", action="store_true")
    p.add_argument("--zone-radius", type=float, default=40.0)
    p.add_argument("--crypto", action="store_true",
                   help="enable the simulator's HMAC-SHA256 message authentication")
    p.add_argument("--json", action="store_true", help="emit the result dict as JSON")
    a = p.parse_args()

    if not 3 <= a.n <= 7:
        p.error("--n must be between 3 and 7")

    from swarm_squad_ep1.research.scenarios import JammingZoneSpec, Scenario

    import swarm_bridge as SB

    zones = ([] if a.no_jamming else
             [JammingZoneSpec(center=(10.0, 40.0, 10.0), radius=a.zone_radius,
                              obstacle_type="high_jam")])
    # Scenario defaults (destination, comm model) are the simulator's own; the Stage 6
    # experiment tables were produced with exactly this construction.
    sc = Scenario(name=f"stage6_{a.arm}_N{a.n}_s{a.seed}", seed=a.seed, num_agents=a.n,
                  agent_init_positions=SB._default_starts(a.n), max_steps=a.steps,
                  jamming_zones=zones, crypto_enabled=a.crypto)

    kw = None
    if a.arm != "baseline":
        kw = dict(F=a.F, byzantine=list(a.byzantine), attack=a.attack,
                  sep_mode=a.sep_mode, enforce_g1=a.enforce_g1, probe_g1=a.probe_g1,
                  central_src=a.central_src)
    out = SB.run_distributed(sc, arm=a.arm, plane_kwargs=kw)

    if a.json:
        keep = {k: v for k, v in out.items() if k not in ("result", "log", "dump", "dist_trace")}
        keep["result"] = {k: getattr(out["result"], k) for k in
                          ("total_steps", "destination_reached", "steps_to_destination",
                           "avg_comm_quality", "avg_Jn")}
        print(json.dumps(keep, indent=2, default=str))
        return 0

    R, P = out["result"], out["plane"]
    print(f"\narm={a.arm}  N={a.n}  seed={a.seed}  jamming={'off' if a.no_jamming else 'on'}"
          f"  byzantine={a.byzantine or 'none'}")
    print(f"  destination reached      : {R.destination_reached}"
          f"  (step {R.steps_to_destination})" if R.destination_reached else
          f"  destination reached      : False")
    print(f"  progress toward target   : {out['progress_frac']:.3f}"
          f"   (final {out['final_dist_m']} m, 50% at step {out['steps_to_50']})")
    print(f"  mean link quality        : {R.avg_comm_quality:.3f}")
    print(f"  min true separation      : {out['min_separation_m']} m"
          f"   (steps below d_min: {100 * out['sep_violated_frac']:.2f}%)")
    print(f"  steps with an isolated vehicle: {100 * out['disconnected_frac']:.2f}%")
    if P:
        print(f"  records sent/delivered   : {P['records_sent']}/{P['records_delivered']}"
              f"   auth bytes {P['auth_bytes']}")
        print(f"  authority mix            : A4 operator -/- A3 quorum {100*P['a3_rate']:.2f}%"
              f"  A2 own {100*P['a2_rate']:.2f}%  A0 controller {100*P['a0_rate']:.2f}%"
              f"  A1 escape {100*P['a1_rate']:.2f}%  brake {100*P['brake_rate']:.2f}%"
              f"  ungated {100*P['central_rate']:.2f}%")
        print(f"  envelope violations (ground truth): {P['unsafe_true']}"
              f"  rate {P['unsafe_true_rate']}  causes {P['unsafe_why']}")
        g1 = P.get("g1_meas_rate")
        print(f"  delivered graph (2F+1,F+1)-robust : "
              + (f"{100*g1:.1f}% of rounds" if g1 is not None
                 else "not probed (pass --probe-g1)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
