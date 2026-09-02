# Stage 6 — Integration on the Swarm-Squad-Ep1 simulator

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
| `d_min` | 8 m | 0.15 m | the inherited controller closes to 0.27 m between vehicles in clear, unjammed flight, so any floor above that would brake a healthy swarm; 0.15 m is ≈55% of it. A vehicle-scale floor is **unattainable without modifying the controller** — see §6.4 |
| `r_comm` | 150 m | 40 m | 1.15× the largest nearest-neighbour distance the honest controller produces, so the clause binds only when a vehicle genuinely strays |
| `dt` | 1.0 s | 0.5 s | simulator step |

`d_min = 0.15 m` is reported as a limitation, not a safety claim: it is the largest separation floor
compatible with the inherited controller. The reachability correction of §6.4 is what makes the clause
meaningful at that scale.

## 3. Experiment design

Scenario: line-formation start 40 m south of the origin, the simulator's default destination, one
`high_jam` zone of radius 40 m at (10, 40, 10) squarely on the transit path, realistic comm model,
`max_steps = 1200`. Five configurations, N ∈ {3, 5, 7}, seeds {0, 1, 2} — 45 runs
(`stage6_grid.csv`). One Byzantine reasoner (F = 1) with the `edge_collude` strategy in every
configuration that has a reasoning plane.

Metrics. *Mission*: steps to 50% of initial distance closed, progress fraction, final distance.
*Communication*: mean link quality (simulator's own), fraction of steps with a vehicle beyond
`r_comm` of its nearest neighbour. *Safety*: minimum **true** pairwise separation and the fraction of
steps below `d_min`, both from ground-truth positions — not from the perceived state the gate used,
which would be self-consistent by construction. *Authority*: the mix of levels actually applied.

## 4. Results

### 4.1 Configurations at N = 5

| configuration                                          | steps to 50%   |   progress |   final dist (m) |   link quality |   min sep (m) |   sep viol (%) |   isolated (%) |   margin viol | A3 quorum (%)   | A2 own (%)   | A0 ctrl (%)   | A1 escape (%)   | brake (%)   |
|:-------------------------------------------------------|:---------------|-----------:|-----------------:|---------------:|--------------:|---------------:|---------------:|--------------:|:----------------|:-------------|:--------------|:----------------|:------------|
| C0  inherited controller, no jamming                   | 321.0          |       0.98 |             2.90 |           1.00 |          0.27 |           0.00 |           0.00 |          0.00 | —               | —            | —             | —               | —           |
| C1  inherited controller, jammed                       | —              |      -0.00 |           195.26 |           0.34 |          1.99 |           0.00 |           0.00 |          0.00 | —               | —            | —             | —               | —           |
| C2  single reasoner, ungated (prior TVT configuration) | 946.667        |       0.60 |            78.64 |           0.26 |          1.36 |           0.00 |          95.36 |         13.00 | 0.0             | 0.0          | 62.993        | 0.0             | 0.0         |
| C3  single *Byzantine* reasoner, ungated               | 1051.0         |       0.51 |            95.89 |           0.20 |          1.74 |           0.00 |          95.30 |          9.00 | 0.0             | 0.0          | 69.293        | 0.0             | 0.0         |
| S6  one reasoner per vehicle, gated                    | 219.0          |       0.64 |            71.29 |           0.78 |          0.99 |           0.00 |          54.53 |        202.00 | 1.493           | 42.88        | 37.82         | 0.683           | 17.127      |

The five rows separate on every axis that matters:

* **The inherited controller stalls.** Jammed, it makes no net progress at any N (progress
  -0.000, final distance 195 m, essentially the start distance): the comm-aware
  formation term refuses to trade link quality for distance at the jamming boundary. Its perfect
  safety and connectivity numbers are vacuous — a stationary swarm violates nothing. Unjammed the
  same controller closes half the distance in 321 steps and finishes
  2.9 m from the destination, so the mission is feasible and the stall is caused by
  jamming alone.
* **One reasoner per vehicle, gated, recovers most of the mission**: 50% of the distance closed by
  step 219, progress 0.635, and mean link quality 0.775
  against 0.341 for the stalled baseline — the plane keeps the swarm connected while moving it.
* **A single ungated reasoner is worse on both counts**: 50% at step 947, link quality
  0.264, and a vehicle isolated in 95% of steps against
  55% for the gated plane. This is the prior TVT configuration, and it disperses the swarm.
* **The authority mix is not degenerate.** The gated plane applies the inherited controller in
  38% of vehicle-steps, a self-certified own record in 43%, a
  Byzantine-tolerant quorum command in 1.5%, the autonomous escape in
  0.7%, and brakes in 17%. Braking concentrates near the destination,
  where the swarm converges and the inflated separation margin binds.

The `margin viol` column counts commands that fail the *inflated* separation margin of §6.4 when
re-checked against ground-truth neighbour positions rather than perceived ones; the strict clause
(`d_min` itself) was never violated in any gated run — minimum true separation stayed at
0.45 m or above, and `sep viol (%)` is 0.00 throughout.
Perception error under jamming, not the gate, is what consumes the margin, and it grows with N:
30, 202, 1164 margin violations at N = 3, 5, 7.

Full per-seed results for all three swarm sizes: `stage6_grid.csv`, `stage6_arm_summary.csv`.

### 4.2 Quorum availability against the consistency tolerance (`stage6_tau_sweep.csv`)

Recomputed offline from every delivered record of the N = 5 and N = 7 runs, so the same rounds are
re-scored at each tolerance. `avail` is the fraction of gate calls at which a consistent quorum of
2F+1 = 3 authority-requesting peer records existed; `byz_in` is the fraction of those quorums that
contained the Byzantine record; `max influence` is how far the fused bearing moved because it did.

N = 7:

|   tau_deg |   avail_% |   avail_g1_% |   byz_in_% |   max_influence_deg |   bound_deg |   n_quorum |
|----------:|----------:|-------------:|-----------:|--------------------:|------------:|-----------:|
|      2.81 |      1.49 |         0.27 |       0.00 |                0.00 |        2.81 |     125.00 |
|      5.62 |      2.85 |         0.48 |       2.90 |                3.52 |        5.62 |     239.00 |
|      8.44 |      3.40 |         0.58 |      15.40 |                4.92 |        8.44 |     286.00 |
|     11.25 |      3.88 |         0.62 |      38.60 |                9.14 |       11.25 |     326.00 |
|     16.88 |      4.18 |         0.76 |      46.40 |                9.14 |       16.88 |     351.00 |
|     22.50 |      4.19 |         0.76 |      46.60 |                9.14 |       22.50 |     352.00 |
|     33.75 |      4.19 |         0.76 |      46.60 |                9.14 |       33.75 |     352.00 |

N = 5:

|   tau_deg |   avail_% |   avail_g1_% |   byz_in_% |   max_influence_deg |   bound_deg |   n_quorum |
|----------:|----------:|-------------:|-----------:|--------------------:|------------:|-----------:|
|      2.81 |      0.63 |         0.00 |       0.00 |                0.00 |        2.81 |      38.00 |
|      5.62 |      0.92 |         0.00 |       0.00 |                0.00 |        5.62 |      55.00 |
|      8.44 |      0.98 |         0.00 |       3.40 |                2.11 |        8.44 |      59.00 |
|     11.25 |      1.48 |         0.00 |      40.40 |                5.62 |       11.25 |      89.00 |
|     16.88 |      1.62 |         0.00 |      48.40 |                7.03 |       16.88 |      97.00 |
|     22.50 |      1.65 |         0.00 |      50.50 |                8.44 |       22.50 |      99.00 |
|     33.75 |      1.65 |         0.00 |      50.50 |                8.44 |       33.75 |      99.00 |

Three readings, and the first is the one the checklist asked simulation to supply:

1. **Tolerance is not the binding constraint on availability.** Availability saturates by
   `τ_b ≈ 22°`, and the specified `τ_b = 16` units (11.25°) already delivers
   3.88% availability against a 4.19% ceiling at N = 7 —
   93% of everything a wider window could buy. But the
   absolute level is low, and that is the finding: what limits the quorum level is
   *simultaneous demand* — three peers requesting override authority in the same round with their
   records delivered — not the width of the consistency window.
2. **Proposition 4's influence bound holds under real dynamics.** With the Byzantine record inside
   the accepted quorum in 39% of quorums at the specified tolerance, the largest
   observed deviation of the fused bearing from the honest-only fusion was
   9.14° against the bound of 11.25°, and
   5.62° against the same bound at N = 5. Stage 5 measured this in a
   synthetic red-team harness; it now holds with simulator-generated perception error.
3. **Tightening the tolerance buys robustness at a real availability cost.** At
   `τ_b = 8` units (5.62°) no Byzantine record entered any quorum at N = 5 and only
   2.9% did at N = 7 — but availability falls to 2.85% from
   3.88%. The tolerance is therefore an availability/influence dial, and the
   influence it buys is already bounded by Proposition 4, which argues for the looser setting.

### 4.3 The global robustness rule is anti-correlated with quorum demand

`avail_g1` above requires, additionally, that the *delivered* reasoning graph be
(2F+1, F+1)-robust in that round. Across whole runs the delivered graph satisfies it often —
43% of rounds at N = 3, 68%
at N = 5, 68% at N = 7 — but almost never in the rounds where a
quorum exists at all: 0.00% against 1.48% at N = 5 (the level becomes
entirely unavailable) and 0.62% against 3.88% at N = 7, a
84% reduction. The events are anti-correlated for a
mechanical reason: enough vehicles request override authority precisely when jamming is severe, which
is when the delivered graph is sparse.

This is the sparse-graph tension flagged as an open item at Stage 3, now measured. The design response
is in §6.3: the *locally checkable counting condition* (≥ 2F+1 consistent records from distinct sources)
is what Proposition 4 actually requires for one-shot filtering; (r, s)-robustness is a condition for
*asymptotic* consensus under iterated W-MSR and is not needed here. Both settings remain selectable
(`enforce_g1`), and the rows above are the cost of choosing the stronger one.

### 4.4 Separation clause: soundness against liveness (`stage6_sep_modes.csv`)

| mode      |   seed |   prog |   brake |   sep_viol |   min_sep |   unsafe_true |
|:----------|-------:|-------:|--------:|-----------:|----------:|--------------:|
| static    |      0 | 0.9755 |  0.0009 |     0.2344 |    0.0000 |             0 |
| static    |      1 | 0.8810 |  0.0020 |     0.1144 |    0.0880 |             0 |
| worstcase |      0 | 0.9771 |  0.2820 |     0.0000 |    1.0760 |            77 |
| worstcase |      1 | 0.8823 |  0.0649 |     0.0000 |    1.0170 |             5 |
| reported  |      0 | 0.9759 |  0.2249 |     0.0011 |    0.0580 |           331 |
| reported  |      1 | 0.8807 |  0.0600 |     0.0000 |    0.6260 |            11 |

Read `unsafe_true` — the ground-truth re-check — together with `sep_viol`, not instead of it. The
re-check applies *the clause that mode configured*, so the `static` rows are self-consistent by
construction (0 failures) while simultaneously spending 17%
of steps in physical violation: that gap **is** the unsoundness. The `worstcase` rows are re-checked
against the strictly larger inflated threshold, so their non-zero counts are margin consumption by
perception error, with true separation never below `d_min`. Comparing the two `unsafe_true` columns
directly would invert the result.

`static` is the Stage 4 clause as written. It admits commands that lead to true separations of
0.00 m — physical overlap — in
17% of steps on average. `worstcase` inflation
eliminates them entirely (0.00% of steps, minimum
separation 1.02 m) at a braking cost of
6–28%
of commands and **no measurable loss of progress**
(0.930 against 0.928).
Soundness here is cheap; see §6.4 for why the clause was unsound and why `reported` inflation — using
the neighbour's own broadcast speed — is *not* a legitimate substitute.

### 4.5 Cost of authenticating the record (`stage6_mac_cost.csv`)

Measured with the simulator's own `CryptoAuth`, all three algorithms it ships, plus the packed
Stage 3 record with a truncated MAC. Bandwidth is per vehicle, one record per neighbour at 2 Hz.

| target             | algorithm         |   tag_bytes |   frame_bytes |   sign_us |   verify_us | tamper_detected   |   bps_N7 |
|:-------------------|:------------------|------------:|--------------:|----------:|------------:|:------------------|---------:|
| repo JSON frame    | hmac_sha256       |          32 |           208 |     10.65 |        9.79 | True              |    19968 |
| repo JSON frame    | chacha20_poly1305 |         237 |           413 |     14.30 |       14.28 | True              |    39648 |
| repo JSON frame    | aes_256_ctr       |         257 |           433 |     23.38 |       28.25 | True              |    41568 |
| packed 12-B record | hmac_sha256_t64   |           8 |            20 |      1.96 |        1.96 | True              |     1920 |

All three detect a single-field tamper of the payload. The two AEAD options carry the ciphertext in
the signature field, so they cost roughly twice the HMAC frame for no additional integrity property
that this design uses. The design-relevant comparison is the last row against the first: authenticating
the **packed 12-byte record** with a 64-bit truncated HMAC costs 8 bytes and
1.96 µs, giving 1920 bit/s per vehicle at N = 7 — against
19968 bit/s for the simulator's JSON frame with a full 32-byte tag, a
10.4× difference. The compact record and the MAC have to be designed
together: a 32-byte tag on a 12-byte payload is a 3.7× overhead, and truncation to 64 bits is what
keeps authentication affordable on a jammed link.

### 4.6 Does a real LLM emit the record the deterministic reasoner emits?

Every run above uses `HonestReasoner`, a deterministic stand-in for the on-board policy, so that
thousands of rounds can be measured. Its validity was tested directly: 60 contexts were sampled
from a real N = 5 run, stratified over sensed jamming severity, and a real LLM was given the same local
view and the doctrine in words, then asked for the record as JSON.

| prompt | JSON parsed | intent match | req_auth match | median bearing error | within `τ_b` |
|---|---|---|---|---|---|
| LLM computes the geometry | 98.3% | 88.3% | 88.3% | 35.9° | 27% |
| flight computer supplies the bearing | 100.0% | 90.0% | 90.0% | 0.0° | 100% |

Asked to compute the swarm centroid, take an `atan2` and apply a severity-scaled veer, the model is
wrong by a median of 35.9° and lands inside the consistency tolerance in only
27% of contexts — proposals that would never aggregate. Given the base bearing from the
vehicle's own flight computer and asked only for the discrete decision plus the veer, the same model
lands inside the tolerance in 100% of contexts, median 0.0° — the doctrine
bearing reproduced exactly, because what is left of it is a discrete choice. Discrete-field agreement is
88% and 90% respectively, with the errors
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
17% of steps below `d_min`, minimum true
separation 0.00 m, with the gate reporting no violation.
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
90% agreement, but cannot reliably do the **geometry**: median bearing
error 35.9°, only 27% of proposals inside `τ_b`, i.e. mostly unaggregable.
Supplying the base bearing from the vehicle's own state estimator moves that to 100%.

**Design rule.** In the double multi-agent system, the boundary between the two planes is not merely
one-agent-per-vehicle; within each vehicle the reasoning agent owns *discrete* decisions (intent,
authority request, whom to suspect) and the vehicle agent owns *continuous* quantities (bearing, speed,
altitude). The packed record already has exactly this shape — discrete intent and authority fields
alongside quantised continuous fields — so the rule is: **the reasoner fills the discrete fields, the
flight computer fills the continuous ones.** The residual 10%
discrete disagreement clusters at the severity trigger, which argues for the trigger comparison also
being machine-computed, with the reasoner choosing the doctrine rather than evaluating the threshold.

### 6.6 (S6-6) The wire format decides which attacks exist

`req_auth` is a 2-bit field, so it encodes A0-A3 and **cannot express A4**. The Stage 5 failure class F3
(self-certify operator authority) therefore has no representation on the wire: `pack` raises
`req_auth=4 does not fit in 2 bits`, and a receiver that deserializes before it reasons drops the record at
the parser. This is now enforced in the plane (`wire_valid`, every field checked against its Stage 3 width),
measured at N = 5 over 600 steps with one Byzantine vehicle:

| attack       |   records sent |   delivered |   dropped malformed |   A3 quorum |   A2 own-record |   progress |   margin viol |
|:-------------|---------------:|------------:|--------------------:|------------:|----------------:|-----------:|--------------:|
| selfcert     |           3000 |        5253 |                1343 |           0 |           0.613 |      0.901 |             0 |
| edge_collude |           3000 |        6342 |                   0 |           0 |           0.619 |      0.829 |             0 |

All 1343 delivered copies of the self-certifying vehicle's records are dropped as unencodable, so the
attack degrades to an availability fault: the swarm makes *more* progress (0.90 of the start
distance) than under the representable edge-collusion attack (0.83), because a silenced faulty peer
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

1. **Most runs use a deterministic stand-in.** §4.6 tests it on 60 contexts against a cloud
   model, not the quantised 3 B model intended to run on board, and not for 45 full runs. Onboard
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
   §4.4; cost it against the 1920 bit/s budget of §4.5.
2. Quorum availability of a few percent (§4.2) is enough to demonstrate the mechanism but too low to
   carry a safety argument. Either the level should be reachable more often (hysteresis on the
   authority request, longer record TTL) or the paper should state plainly that the quorum level is an
   exception path, not the normal one.
3. Intent disagreement at the severity trigger (§6.5) suggests hysteresis or machine-side triggering.
4. `HonestReasoner`'s veer doctrine is fixed. Whether a real reasoner's *choice* of doctrine beats a
   fixed one is the question the double multi-agent claim ultimately rests on, and it is unanswered.
