"""Generate `integration_spec.md` (Stage 6) with every table rendered from the saved CSVs."""

import os as _os
import pandas as pd

# Resolve inputs/outputs relative to this file so the generator runs from any cwd.
_BASE = _os.path.dirname(_os.path.abspath(__file__))
_RES = _os.path.join(_BASE, "results") if _os.path.isdir(_os.path.join(_BASE, "results")) else _BASE
_DOCS = _os.path.join(_BASE, "docs") if _os.path.isdir(_os.path.join(_BASE, "docs")) else _BASE
_r = lambda n: pd.read_csv(_os.path.join(_RES, n))

SC = _r("stage6_selfcert.csv")
_scr = SC.set_index("attack")
SELFCERT = SC.rename(columns={"records_sent": "records sent", "records_delivered": "delivered",
                              "records_malformed": "dropped malformed", "progress_frac": "progress",
                              "unsafe_true": "margin viol", "a2_rate": "A2 own-record",
                              "a3_rate": "A3 quorum"})[
    ["attack", "records sent", "delivered", "dropped malformed", "A3 quorum", "A2 own-record",
     "progress", "margin viol"]].to_markdown(index=False)
sc_mal = int(_scr.loc["selfcert", "records_malformed"])
sc_prog = float(_scr.loc["selfcert", "progress_frac"])
ec_prog = float(_scr.loc["edge_collude", "progress_frac"])

G = _r("stage6_grid.csv")
SUM = _r("stage6_arm_summary.csv")
W = _r("stage6_tau_sweep.csv")
MAC = _r("stage6_mac_cost.csv")
L = _r("stage6_llm_agreement.csv")
L2 = _r("stage6_llm_agreement_split.csv")
SEP = _r("stage6_sep_modes.csv")
DEGU = 360.0 / 512.0

NAME = {"clear_baseline": "C0  inherited controller, no jamming",
        "baseline": "C1  inherited controller, jammed",
        "centralised": "C2  single reasoner, ungated (prior TVT configuration)",
        "centralised_byz": "C3  single *Byzantine* reasoner, ungated",
        "distributed": "S6  one reasoner per vehicle, gated"}
ORDER = ["clear_baseline", "baseline", "centralised", "centralised_byz", "distributed"]


def md(df, cols=None, fmt=None):
    d = df[cols] if cols else df
    return d.to_markdown(index=False, floatfmt=fmt or ".3f")


def arm_table(N):
    d = SUM[SUM.N == N].set_index("arm").reindex(ORDER).reset_index()
    d["arm"] = d.arm.map(NAME)
    d = d.rename(columns={"arm": "configuration", "steps_to_50%": "steps to 50%",
                          "progress": "progress", "final_dist_m": "final dist (m)",
                          "comm_quality": "link quality", "min_sep_m": "min sep (m)",
                          "sep_violation_%": "sep viol (%)", "disconnected_%": "isolated (%)",
                          "envelope_violations": "margin viol",
                          "A3_quorum_%": "A3 quorum (%)", "A2_own_%": "A2 own (%)",
                          "A0_operator_%": "A0 ctrl (%)", "A1_ctrl_%": "A1 escape (%)",
                          "brake_%": "brake (%)"})
    d = d.fillna("—")
    return md(d, ["configuration", "steps to 50%", "progress", "final dist (m)", "link quality",
                  "min sep (m)", "sep viol (%)", "isolated (%)", "margin viol",
                  "A3 quorum (%)", "A2 own (%)", "A0 ctrl (%)", "A1 escape (%)", "brake (%)"],
              ".2f")


w7 = W[W.N == 7].copy()
w5 = W[W.N == 5].copy()
for w in (w5, w7):
    w["tau_deg"] = (w.tau_b * DEGU).round(2)
    w["avail_%"] = (w.quorum_avail * 100).round(2)
    w["avail_g1_%"] = (w.quorum_avail_g1 * 100).round(2)
    w["byz_in_%"] = (w.byz_in_quorum * 100).round(1)

sat5 = w5.loc[w5.quorum_avail.idxmax(), "tau_b"]
spec5 = w5[w5.tau_b == 16].iloc[0]
spec7 = w7[w7.tau_b == 16].iloc[0]
tight7 = w7[w7.tau_b == 8].iloc[0]

mono_within = (L.bearing_err_deg <= 16 * DEGU).mean() * 100
split_within = (L2.bearing_err_deg <= 16 * DEGU).mean() * 100
mono_med = L.bearing_err_deg.median()
split_med = L2.bearing_err_deg.median()

d5 = SUM[(SUM.N == 5) & (SUM.arm == "distributed")].iloc[0]
b5 = SUM[(SUM.N == 5) & (SUM.arm == "baseline")].iloc[0]
c5 = SUM[(SUM.N == 5) & (SUM.arm == "centralised")].iloc[0]
n0 = SUM[(SUM.N == 5) & (SUM.arm == "clear_baseline")].iloc[0]
hmac = MAC[MAC.algorithm == "hmac_sha256"].iloc[0]
pack = MAC[MAC.algorithm == "hmac_sha256_t64"].iloc[0]

DOC = f"""# Stage 6 — Integration on the Swarm-Squad-Ep1 simulator

**Successor to Section V ("Simulation Results") of the IEEE TVT submission.**
Reference implementation: `swarm_bridge.py`. Runnable demo: `demo_stage6.py`.
Depends on Stage 2 (`architecture_spec.md`, authority levels A0–A4, invariants INV1–INV5),
Stage 3 (`message_interface.md`, the packed proposal record), Stage 4 (`override_spec.md`,
the deterministic gate) and Stage 5 (`aggregation_spec.md`, `MaxConsistent`, `Fuse`, quorum size).

Simulator: `Swarm-Squad-Ep1` (github.com/Swarm-Squad/Swarm-Squad-Ep1), used unmodified.
The package ships a comm-aware 3-D formation controller, a distance/jamming V2V channel model,
HMAC/AEAD message authentication, a spoofing engine and a headless research harness — but no
per-vehicle reasoning agents. Stage 6 supplies those, and nothing else.

## 0. Stage criterion

The checklist asks for *"a runnable multi-LLM demo on the actual simulator, N = 3–7"*. This stage
delivers that (`demo_stage6.py`, §8) and, because the demo immediately falsified three parts of the
Stage 2–4 design, five integration findings (§6) with the corrections they forced.

## 1. Integration surface

| Simulator element | Stage 6 treatment |
|---|---|
| `algo/controller.py` `UnifiedController` | **inherited unchanged**; its command is authority level A0, the liveness floor |
| `algo/v2v_channel.py` `V2VChannelModel` | **inherited unchanged**; carries the proposal records, so record loss is the simulator's loss model, not ours |
| `algo/mavlink.py` `MAVLinkBus` | **inherited unchanged**; perceived neighbour positions come from `build_perceived_state`, so every reasoner sees a jamming-degraded world |
| `algo/crypto_auth.py` `CryptoAuth` | **inherited unchanged**; used to cost authentication of the record (§5) |
| `algo/llm_controller.py` `LLMAssistanceController` | **replaced** by `DistributedReasoningPlane`, which presents the same four-method interface (`check_agents_needing_assistance`, `request_guidance`, `get_guidance`, `apply_guidance`) |
| `research/runner.py` `step` | **reused**; `run_distributed` drives it directly |
| `research/runner.py` `run_scenario` | **not reused**: it constructs its own centralized controller internally, so there is no injection point. `run_distributed` reimplements the scenario loop around `step` and reuses the `Scenario`/`Result` dataclasses. |

Consequence of presenting the simulator's own interface rather than forking: the plane is a drop-in
for anyone who has the package installed, and the vehicle-side control law is provably untouched.

## 2. Scale calibration

The Stage 4 envelope was written for a world two orders of magnitude larger than the simulator's
(`config.py` bounds ±100 m, `MAX_MOVEMENT_PER_STEP` 2 m, `dt = 0.5 s`). Each value below was set from
measurement of the **inherited controller's own behaviour**, not chosen for convenience, because an
envelope that contradicts the controller it promises to inherit would brake continuously:

| Envelope term | Stage 4 | Stage 6 | Basis |
|---|---|---|---|
| `v_max` | 20 m/s | 2.0 m/s | `MAX_MOVEMENT_PER_STEP / dt` |
| `vz_max` | 5 m/s | 0.5 m/s | vertical share of the same cap |
| `z_min, z_max` | 40, 120 m | 0.5, 30 m | simulator altitude range; the Stage 4 floor would have rejected every command at the 2 m start altitude and brake-locked the swarm permanently |
| `d_min` | 8 m | 0.15 m | the inherited controller closes to {n0['min_sep_m']:.2f} m between vehicles in clear, unjammed flight, so any floor above that would brake a healthy swarm; 0.15 m is ≈{0.15 / n0['min_sep_m']:.0%} of it. A vehicle-scale floor is **unattainable without modifying the controller** — see §6.4 |
| `r_comm` | 150 m | 40 m | 1.15× the largest nearest-neighbour distance the honest controller produces, so the clause binds only when a vehicle genuinely strays |
| `dt` | 1.0 s | 0.5 s | simulator step |

`d_min = 0.15 m` is reported as a limitation, not a safety claim: it is the largest separation floor
compatible with the inherited controller. The reachability correction of §6.4 is what makes the clause
meaningful at that scale.

## 3. Experiment design

Scenario: line-formation start 40 m south of the origin, the simulator's default destination, one
`high_jam` zone of radius 40 m at (10, 40, 10) squarely on the transit path, realistic comm model,
`max_steps = 1200`. Five configurations, N ∈ {{3, 5, 7}}, seeds {{0, 1, 2}} — 45 runs
(`stage6_grid.csv`). One Byzantine reasoner (F = 1) with the `edge_collude` strategy in every
configuration that has a reasoning plane.

Metrics. *Mission*: steps to 50% of initial distance closed, progress fraction, final distance.
*Communication*: mean link quality (simulator's own), fraction of steps with a vehicle beyond
`r_comm` of its nearest neighbour. *Safety*: minimum **true** pairwise separation and the fraction of
steps below `d_min`, both from ground-truth positions — not from the perceived state the gate used,
which would be self-consistent by construction. *Authority*: the mix of levels actually applied.

## 4. Results

### 4.1 Configurations at N = 5

{arm_table(5)}

The five rows separate on every axis that matters:

* **The inherited controller stalls.** Jammed, it makes no net progress at any N (progress
  {b5['progress']:+.3f}, final distance {b5['final_dist_m']:.0f} m, essentially the start distance): the comm-aware
  formation term refuses to trade link quality for distance at the jamming boundary. Its perfect
  safety and connectivity numbers are vacuous — a stationary swarm violates nothing. Unjammed the
  same controller closes half the distance in {n0['steps_to_50%']:.0f} steps and finishes
  {n0['final_dist_m']:.1f} m from the destination, so the mission is feasible and the stall is caused by
  jamming alone.
* **One reasoner per vehicle, gated, recovers most of the mission**: 50% of the distance closed by
  step {d5['steps_to_50%']:.0f}, progress {d5['progress']:.3f}, and mean link quality {d5['comm_quality']:.3f}
  against {b5['comm_quality']:.3f} for the stalled baseline — the plane keeps the swarm connected while moving it.
* **A single ungated reasoner is worse on both counts**: 50% at step {c5['steps_to_50%']:.0f}, link quality
  {c5['comm_quality']:.3f}, and a vehicle isolated in {c5['disconnected_%']:.0f}% of steps against
  {d5['disconnected_%']:.0f}% for the gated plane. This is the prior TVT configuration, and it disperses the swarm.
* **The authority mix is not degenerate.** The gated plane applies the inherited controller in
  {d5['A0_operator_%']:.0f}% of vehicle-steps, a self-certified own record in {d5['A2_own_%']:.0f}%, a
  Byzantine-tolerant quorum command in {d5['A3_quorum_%']:.1f}%, the autonomous escape in
  {d5['A1_ctrl_%']:.1f}%, and brakes in {d5['brake_%']:.0f}%. Braking concentrates near the destination,
  where the swarm converges and the inflated separation margin binds.

The `margin viol` column counts commands that fail the *inflated* separation margin of §6.4 when
re-checked against ground-truth neighbour positions rather than perceived ones; the strict clause
(`d_min` itself) was never violated in any gated run — minimum true separation stayed at
{SUM[SUM.arm == 'distributed'].min_sep_m.min():.2f} m or above, and `sep viol (%)` is 0.00 throughout.
Perception error under jamming, not the gate, is what consumes the margin, and it grows with N:
{', '.join(str(int(v)) for v in SUM[SUM.arm=='distributed'].sort_values('N').envelope_violations)} margin violations at N = 3, 5, 7.

Full per-seed results for all three swarm sizes: `stage6_grid.csv`, `stage6_arm_summary.csv`.

### 4.2 Quorum availability against the consistency tolerance (`stage6_tau_sweep.csv`)

Recomputed offline from every delivered record of the N = 5 and N = 7 runs, so the same rounds are
re-scored at each tolerance. `avail` is the fraction of gate calls at which a consistent quorum of
2F+1 = 3 authority-requesting peer records existed; `byz_in` is the fraction of those quorums that
contained the Byzantine record; `max influence` is how far the fused bearing moved because it did.

N = 7:

{md(w7, ["tau_deg", "avail_%", "avail_g1_%", "byz_in_%", "max_influence_deg", "bound_deg", "n_quorum"], ".2f")}

N = 5:

{md(w5, ["tau_deg", "avail_%", "avail_g1_%", "byz_in_%", "max_influence_deg", "bound_deg", "n_quorum"], ".2f")}

Three readings, and the first is the one the checklist asked simulation to supply:

1. **Tolerance is not the binding constraint on availability.** Availability saturates by
   `τ_b ≈ {sat5 * DEGU:.0f}°`, and the specified `τ_b = 16` units ({16 * DEGU:.2f}°) already delivers
   {spec7['avail_%']:.2f}% availability against a {w7['avail_%'].max():.2f}% ceiling at N = 7 —
   {100 * spec7['avail_%'] / w7['avail_%'].max():.0f}% of everything a wider window could buy. But the
   absolute level is low, and that is the finding: what limits the quorum level is
   *simultaneous demand* — three peers requesting override authority in the same round with their
   records delivered — not the width of the consistency window.
2. **Proposition 4's influence bound holds under real dynamics.** With the Byzantine record inside
   the accepted quorum in {spec7['byz_in_%']:.0f}% of quorums at the specified tolerance, the largest
   observed deviation of the fused bearing from the honest-only fusion was
   {spec7['max_influence_deg']:.2f}° against the bound of {spec7['bound_deg']:.2f}°, and
   {spec5['max_influence_deg']:.2f}° against the same bound at N = 5. Stage 5 measured this in a
   synthetic red-team harness; it now holds with simulator-generated perception error.
3. **Tightening the tolerance buys robustness at a real availability cost.** At
   `τ_b = 8` units ({8 * DEGU:.2f}°) no Byzantine record entered any quorum at N = 5 and only
   {tight7['byz_in_%']:.1f}% did at N = 7 — but availability falls to {tight7['avail_%']:.2f}% from
   {spec7['avail_%']:.2f}%. The tolerance is therefore an availability/influence dial, and the
   influence it buys is already bounded by Proposition 4, which argues for the looser setting.

### 4.3 The global robustness rule is anti-correlated with quorum demand

`avail_g1` above requires, additionally, that the *delivered* reasoning graph be
(2F+1, F+1)-robust in that round. Across whole runs the delivered graph satisfies it often —
{W[W.N == 3].g1_meas_rate.iloc[0] * 100:.0f}% of rounds at N = 3, {W[W.N == 5].g1_meas_rate.iloc[0] * 100:.0f}%
at N = 5, {W[W.N == 7].g1_meas_rate.iloc[0] * 100:.0f}% at N = 7 — but almost never in the rounds where a
quorum exists at all: {spec5['avail_g1_%']:.2f}% against {spec5['avail_%']:.2f}% at N = 5 (the level becomes
entirely unavailable) and {spec7['avail_g1_%']:.2f}% against {spec7['avail_%']:.2f}% at N = 7, a
{100 * (1 - spec7['avail_g1_%'] / spec7['avail_%']):.0f}% reduction. The events are anti-correlated for a
mechanical reason: enough vehicles request override authority precisely when jamming is severe, which
is when the delivered graph is sparse.

This is the sparse-graph tension flagged as an open item at Stage 3, now measured. The design response
is in §6.3: the *locally checkable counting condition* (≥ 2F+1 consistent records from distinct sources)
is what Proposition 4 actually requires for one-shot filtering; (r, s)-robustness is a condition for
*asymptotic* consensus under iterated W-MSR and is not needed here. Both settings remain selectable
(`enforce_g1`), and the rows above are the cost of choosing the stronger one.

### 4.4 Separation clause: soundness against liveness (`stage6_sep_modes.csv`)

{md(SEP[["mode", "seed", "prog", "brake", "sep_viol", "min_sep", "unsafe_true"]], None, ".4f")}

Read `unsafe_true` — the ground-truth re-check — together with `sep_viol`, not instead of it. The
re-check applies *the clause that mode configured*, so the `static` rows are self-consistent by
construction (0 failures) while simultaneously spending {SEP[SEP['mode'] == 'static'].sep_viol.mean() * 100:.0f}%
of steps in physical violation: that gap **is** the unsoundness. The `worstcase` rows are re-checked
against the strictly larger inflated threshold, so their non-zero counts are margin consumption by
perception error, with true separation never below `d_min`. Comparing the two `unsafe_true` columns
directly would invert the result.

`static` is the Stage 4 clause as written. It admits commands that lead to true separations of
{SEP[SEP['mode'] == 'static'].min_sep.min():.2f} m — physical overlap — in
{SEP[SEP['mode'] == 'static'].sep_viol.mean() * 100:.0f}% of steps on average. `worstcase` inflation
eliminates them entirely ({SEP[SEP['mode'] == 'worstcase'].sep_viol.max() * 100:.2f}% of steps, minimum
separation {SEP[SEP['mode'] == 'worstcase'].min_sep.min():.2f} m) at a braking cost of
{SEP[SEP['mode'] == 'worstcase'].brake.min() * 100:.0f}–{SEP[SEP['mode'] == 'worstcase'].brake.max() * 100:.0f}%
of commands and **no measurable loss of progress**
({SEP[SEP['mode'] == 'worstcase'].prog.mean():.3f} against {SEP[SEP['mode'] == 'static'].prog.mean():.3f}).
Soundness here is cheap; see §6.4 for why the clause was unsound and why `reported` inflation — using
the neighbour's own broadcast speed — is *not* a legitimate substitute.

### 4.5 Cost of authenticating the record (`stage6_mac_cost.csv`)

Measured with the simulator's own `CryptoAuth`, all three algorithms it ships, plus the packed
Stage 3 record with a truncated MAC. Bandwidth is per vehicle, one record per neighbour at 2 Hz.

{md(MAC[["target", "algorithm", "tag_bytes", "frame_bytes", "sign_us", "verify_us", "tamper_detected", "bps_N7"]], None, ".2f")}

All three detect a single-field tamper of the payload. The two AEAD options carry the ciphertext in
the signature field, so they cost roughly twice the HMAC frame for no additional integrity property
that this design uses. The design-relevant comparison is the last row against the first: authenticating
the **packed 12-byte record** with a 64-bit truncated HMAC costs {pack['tag_bytes']:.0f} bytes and
{pack['sign_us']:.2f} µs, giving {pack['bps_N7']:.0f} bit/s per vehicle at N = 7 — against
{hmac['bps_N7']:.0f} bit/s for the simulator's JSON frame with a full 32-byte tag, a
{hmac['bps_N7'] / pack['bps_N7']:.1f}× difference. The compact record and the MAC have to be designed
together: a 32-byte tag on a 12-byte payload is a 3.7× overhead, and truncation to 64 bits is what
keeps authentication affordable on a jammed link.

### 4.6 Does a real LLM emit the record the deterministic reasoner emits?

Every run above uses `HonestReasoner`, a deterministic stand-in for the on-board policy, so that
thousands of rounds can be measured. Its validity was tested directly: {len(L)} contexts were sampled
from a real N = 5 run, stratified over sensed jamming severity, and a real LLM was given the same local
view and the doctrine in words, then asked for the record as JSON.

| prompt | JSON parsed | intent match | req_auth match | median bearing error | within `τ_b` |
|---|---|---|---|---|---|
| LLM computes the geometry | {L.parsed.mean() * 100:.1f}% | {L.intent_ok.mean() * 100:.1f}% | {L.auth_ok.mean() * 100:.1f}% | {mono_med:.1f}° | {mono_within:.0f}% |
| flight computer supplies the bearing | {L2.parsed.mean() * 100:.1f}% | {L2.intent_ok.mean() * 100:.1f}% | {L2.auth_ok.mean() * 100:.1f}% | {split_med:.1f}° | {split_within:.0f}% |

Asked to compute the swarm centroid, take an `atan2` and apply a severity-scaled veer, the model is
wrong by a median of {mono_med:.1f}° and lands inside the consistency tolerance in only
{mono_within:.0f}% of contexts — proposals that would never aggregate. Given the base bearing from the
vehicle's own flight computer and asked only for the discrete decision plus the veer, the same model
lands inside the tolerance in {split_within:.0f}% of contexts, median {split_med:.1f}° — the doctrine
bearing reproduced exactly, because what is left of it is a discrete choice. Discrete-field agreement is
{L.intent_ok.mean() * 100:.0f}% and {L2.intent_ok.mean() * 100:.0f}% respectively, with the errors
concentrated at the severity trigger. See §6.5 for the architectural rule this forces.

## 5. Figure

`stage6_results.png` — (a) mission progress by configuration at N = 5; (b) quorum availability and
Byzantine admission against `τ_b`, with the robustness-enforced curve; (c) the separation-clause
trade-off; (d) real-LLM agreement, monolithic against split-plane prompting.

## 6. Integration findings

### 6.1 (S6-1) The authority lattice was ordered wrongly for a jammed world

As specified at Stage 2, the autonomous escape level ranked *above* the inherited controller. Because
the escape candidate exists whenever sensed severity is non-zero, any jamming at all discarded the
controller's command outright: the escape level dominated every run, the reasoning plane was
effectively bypassed, and dispersion was worse than the stalled baseline.

**Correction.** The escape manoeuvre is a *liveness fallback*, not an authority. Admission now
descends operator → quorum → own record → inherited controller, with the escape tried only when none
of those verifies, and braking below it (`ADMIT_V3 = (A4, A3, A2, A0, A1)`, `gate_v3`). The original
ordering remains selectable so the difference is measurable rather than asserted.

### 6.2 (S6-2) A proposal must be about a shared referent to be aggregable

The first honest policy proposed an *ego-local* action — bearing from the vehicle's own position to
the destination. Honest vehicles then disagreed by up to 166 quantisation units, an order of magnitude
beyond `τ_b`, purely because they occupy different points in space. No consistent quorum could ever
form; the aggregation machinery was inert for a reason that had nothing to do with faults.

**Correction, and a design principle worth stating in the paper.** Under a quantised consistency
filter, a proposal must be expressed about a **shared referent**. The bearing is now measured from the
vehicle's *perceived swarm centroid* to the destination, with a fixed right-hand veer doctrine whose
magnitude scales with locally sensed severity. Residual honest disagreement is then exactly perception
error plus severity disagreement — which is what §4.2 measures against `τ_b`. This is a constraint on
the *message semantics*, not on the aggregation rule, and it applies to any quantised-consensus
messaging scheme, not just this one.

### 6.3 (S6-3) Global graph robustness is the wrong precondition for one-shot filtering

Quantified in §4.3: requiring the delivered graph to be (2F+1, F+1)-robust in the same round makes the
quorum level essentially unavailable, because demand for it and graph sparsity have the same cause.

**Correction.** Proposition 4 bounds Byzantine influence from the *counting* condition — at least
2F+1 pairwise-consistent records from distinct sources, checkable locally from the inbox. (r, s)-robustness
is a condition for asymptotic consensus under iterated W-MSR, which this design does not run: the gate
filters once per round and then defers to the controller. Robustness is measured and reported
(`g1_meas_rate`), not enforced by default.

### 6.4 (S6-4) The separation clause was unsound, and the fix follows from the message design

`Verify` tested the vehicle's *predicted* position against its neighbours' *current* positions. When a
neighbour can move further than `d_min` within one step — 2 m per step against `d_min = 0.15 m` here —
the clause is vacuous: both vehicles pass it and still collide. §4.4 measures exactly that:
{SEP[SEP['mode'] == 'static'].sep_viol.mean() * 100:.0f}% of steps below `d_min`, minimum true
separation {SEP[SEP['mode'] == 'static'].min_sep.min():.2f} m, with the gate reporting no violation.
This resolves open item 3 of `override_spec.md` §9, which anticipated the weakness but left it unmeasured.

**Correction.** The threshold is inflated by each neighbour's reachable displacement:
`dist(p_pred, q) ≥ d_min + v_max·dt` (`verify_sound`, mode `worstcase`). Cost: §4.4.

**A tempting shortcut that must be rejected.** The proposal record already carries a speed field, so a
vehicle could inflate by the neighbour's *reported* speed instead of the worst case. Measured
(`reported` rows, §4.4), that setting still admits violations and produces more margin failures than
worst-case inflation. The reason is structural, not statistical: the record's speed field is a
**proposal**, not a state report — it says what the neighbour's reasoner wants, not what the vehicle is
doing — and a Byzantine neighbour can understate it freely. Safety inflation must not depend on a
sender-supplied quantity; that is INV4 applied to the envelope rather than to the fusion weights. A
sound `reported` mode would need a separate authenticated state channel, which is a bandwidth decision
for Stage 7.

### 6.5 (S6-5) Split the planes at the semantic/numeric boundary, not at the vehicle boundary

§4.6 is a negative result with a clean architectural consequence. An LLM given a local view and a
doctrine reproduces the **discrete** fields — which intent, whether to request override authority — at
{L2.intent_ok.mean() * 100:.0f}% agreement, but cannot reliably do the **geometry**: median bearing
error {mono_med:.1f}°, only {mono_within:.0f}% of proposals inside `τ_b`, i.e. mostly unaggregable.
Supplying the base bearing from the vehicle's own state estimator moves that to {split_within:.0f}%.

**Design rule.** In the double multi-agent system, the boundary between the two planes is not merely
one-agent-per-vehicle; within each vehicle the reasoning agent owns *discrete* decisions (intent,
authority request, whom to suspect) and the vehicle agent owns *continuous* quantities (bearing, speed,
altitude). The packed record already has exactly this shape — discrete intent and authority fields
alongside quantised continuous fields — so the rule is: **the reasoner fills the discrete fields, the
flight computer fills the continuous ones.** The residual {100 - L2.intent_ok.mean() * 100:.0f}%
discrete disagreement clusters at the severity trigger, which argues for the trigger comparison also
being machine-computed, with the reasoner choosing the doctrine rather than evaluating the threshold.

### 6.6 (S6-6) The wire format decides which attacks exist

`req_auth` is a 2-bit field, so it encodes A0-A3 and **cannot express A4**. The Stage 5 failure class F3
(self-certify operator authority) therefore has no representation on the wire: `pack` raises
`req_auth=4 does not fit in 2 bits`, and a receiver that deserializes before it reasons drops the record at
the parser. This is now enforced in the plane (`wire_valid`, every field checked against its Stage 3 width),
measured at N = 5 over 600 steps with one Byzantine vehicle:

{SELFCERT}

All {sc_mal} delivered copies of the self-certifying vehicle's records are dropped as unencodable, so the
attack degrades to an availability fault: the swarm makes *more* progress ({sc_prog:.2f} of the start
distance) than under the representable edge-collusion attack ({ec_prog:.2f}), because a silenced faulty peer
is less damaging than a well-formed misleading one. Two consequences. First, an attack taxonomy written
against the reasoning layer must be re-checked against the encoding before the gate is credited with
defending it -- some classes are excluded by the format and others only by the gate, and conflating them
overstates the gate. Second, the width check belongs in the *receiver*, not only in the sender's serializer:
nothing obliges a Byzantine sender to call `pack`. Stage 3 documented this field's range incorrectly
("A1-A3 (+reserved)", implying A0 is not requestable and leaving A4 ambiguous); that descriptor is
corrected in `message_interface.md`. The document's worked example, which requests A3 in prose, was
already consistent with the encoding.

## 7. Corrections to earlier stages

1. `override_spec.md` §3 (`Verify`) — separation clause replaced by the reachability-inflated clause
   of §6.4. Appended there as a dated correction.
2. `architecture_spec.md` — authority admission order corrected per §6.1.
3. `message_interface.md` — the shared-referent requirement of §6.2 constrains the bearing field's
   semantics; the speed field is explicitly *not* usable for safety inflation (§6.4).
4. `message_interface.md` — the `req_auth` field-table range descriptor is corrected per §6.6
   (2 bits, 0 = A0 … 3 = A3, A4 not representable); the "three orders of magnitude" characterisation of
   the robustness-check budget is corrected to ~170x (5.8 ms in a 1 Hz round).
5. `aggregation_spec.md` — no change: the quorum rule and Proposition 4 survive the integration
   unchanged; only the *precondition* on graph robustness is downgraded from required to reported (§6.3).

## 8. Running the demo

```bash
pip install pathfinding3d python-dotenv fastapi
git clone https://github.com/Swarm-Squad/Swarm-Squad-Ep1
export PYTHONPATH=Swarm-Squad-Ep1/src

python demo_stage6.py --n 5 --arm distributed --byzantine agent3 --crypto --probe-g1
python demo_stage6.py --n 7 --arm centralised --central-src agent3 --byzantine agent3
python demo_stage6.py --n 3 --arm baseline --no-jamming
python demo_stage6.py --n 5 --arm distributed --sep-mode static     # the unsound clause, S6-4
python demo_stage6.py --n 5 --arm distributed --enforce-g1          # the strong precondition, S6-3
```

`swarm_bridge.py` also exposes `LLMReasoner`, which calls a real model per vehicle per round through
the same interface; it is the path used for §4.6 and is too slow for 1200-step sweeps.

## 9. What these numbers do not establish

1. **Most runs use a deterministic stand-in.** §4.6 tests it on {len(L)} contexts against a cloud
   model, not the quantised 3 B model intended to run on board, and not for {len(G)} full runs. Onboard
   latency, memory and small-model competence are unmeasured.
2. **No arm reaches the destination under jamming** within 1200 steps at the simulator's success
   radius; the comparison is on progress rate and safety, not completion.
3. **Three seeds per cell.** Differences between the two single-reasoner configurations (honest vs
   Byzantine supervisor) are within seed spread at some N and should not be read as an ordering.
4. **One jamming geometry.** A single 40 m zone on the transit path. Corridors, multiple zones and
   mobile jammers change the severity profile and therefore the demand for the quorum level.
5. **The channel is the simulator's model**, not a radio: distance-based path loss with a jamming
   term, no multipath, no MAC contention, and record loss independent per link.
6. **`d_min = 0.15 m` is not a physical separation requirement** but the largest floor compatible with
   the inherited controller (§2). A vehicle-scale floor requires modifying the controller, which this
   stage deliberately does not do.

## 10. Open items for Stage 7

1. An authenticated state channel would make `reported` inflation sound and shrink the braking cost of
   §4.4; cost it against the {pack['bps_N7']:.0f} bit/s budget of §4.5.
2. Quorum availability of a few percent (§4.2) is enough to demonstrate the mechanism but too low to
   carry a safety argument. Either the level should be reachable more often (hysteresis on the
   authority request, longer record TTL) or the paper should state plainly that the quorum level is an
   exception path, not the normal one.
3. Intent disagreement at the severity trigger (§6.5) suggests hysteresis or machine-side triggering.
4. `HonestReasoner`'s veer doctrine is fixed. Whether a real reasoner's *choice* of doctrine beats a
   fixed one is the question the double multi-agent claim ultimately rests on, and it is unanswered.
"""

open(_os.path.join(_DOCS, "integration_spec.md"), "w").write(DOC)
print("integration_spec.md", len(DOC), "chars")
