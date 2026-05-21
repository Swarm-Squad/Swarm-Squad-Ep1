"""
Fast smoke test for the research harness.

Runs 5 short scenarios end-to-end with no LLM so CI (or a local
dev) can verify the full pipeline without network access or heavy
dependencies. Exits non-zero if any scenario raises.
"""

from __future__ import annotations

import traceback

from swarm_squad_ep1.research.runner import run_scenario
from swarm_squad_ep1.research.scenarios import (
    baseline_scenario,
    combined_scenario,
    jamming_scenario,
    spoofing_scenario,
)

SMOKE_SCENARIOS = [
    baseline_scenario(seed=1, name="smoke_baseline"),
    jamming_scenario(jam_type="high_jam", seed=2, llm=False),
    spoofing_scenario(spoof_type="phantom", crypto=True, seed=3, llm=False),
    spoofing_scenario(
        spoof_type="position_falsification", crypto=False, seed=4, llm=False
    ),
    combined_scenario(
        jam_type="high_jam", spoof_type="phantom", crypto=True, llm=False, seed=5
    ),
]


def run_smoke(verbose: bool = True) -> int:
    """Run the smoke scenarios. Return exit code (0 = pass)."""
    failures = 0
    for sc in SMOKE_SCENARIOS:
        sc.max_steps = 120
        try:
            res = run_scenario(sc, keep_trace=False, verbose=verbose)
            if verbose:
                print(
                    f"[smoke] {sc.name} reached={res.destination_reached} "
                    f"steps={res.total_steps} Jn={res.final_Jn} "
                    f"comm_q={res.avg_comm_quality} det={res.detection_rate}"
                )
        except Exception:
            failures += 1
            print(f"[smoke] FAIL {sc.name}:")
            traceback.print_exc()
    if verbose:
        print(
            f"[smoke] {len(SMOKE_SCENARIOS) - failures}/{len(SMOKE_SCENARIOS)} passed"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_smoke())
