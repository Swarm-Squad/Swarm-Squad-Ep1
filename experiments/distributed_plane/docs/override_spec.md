# Stage 4 — Deterministic override algorithm

**Successor to Algorithm 2 of the IEEE TVT submission ("LLM Control Override Logic").**
Reference implementation: `override_gate.py`. Property suite: `test_gate.py`. Paper block: `override_algorithm.tex`.
Depends on Stage 2 (`architecture_spec.md`, levels A0–A4, invariants INV1–INV5) and Stage 3
(`message_interface.md`, the 9-byte record).

## 0. A correction to the stage criterion

The checklist names Algorithm 1 as the element being succeeded. Algorithm 1 of the journal paper is the
**3D formation controller** producing `G_i` by gradient ascent on link quality; it is inherited *unchanged*
and enters this stage only as part of the baseline `C_b = G_i + M_i`. The element this stage replaces is
**Algorithm 2**, the override logic. All correspondence below is to Algorithm 2.

## 1. What changes, line by line

| TVT Algorithm 2 | Stage 4 gate | Why |
|---|---|---|
| L2 `C_b ← DefaultController(S)` | unchanged, becomes candidate `U[A0]` | baseline is now the *liveness floor*, not a blending partner |
| L5 `L_out ← RequestLLM(S,P)` | unchanged in the plane, but one LLM per vehicle | Stage 2 |
| L8 `(C_ℓ,A) ← ParseLLM(L_out)` | `Dec(·)` over the 9-byte record; unparseable is impossible | Stage 3 schema is the decode grammar |
| L10 `(abn,sev) ← CheckAbnormal3D(S)` | unchanged; `s_i` now gates *which levels exist* | severity becomes a precondition, not a gain |
| **L14–15** `w_ℓ ← w_min+(w_max−w_min)s_i`; `C_f ← w_ℓC_ℓ+(1−w_ℓ)C_b` | **deleted** | see §2 |
| — | `Canon`, `MaxConsistent`, `Fuse`, quorum test | peers now exist as command sources |
| — | `Π_cvx` then `Verify`, then descend | INV1 containment |
| L18 `ApplyControl3D(C_f)` | apply `U[a*]`, emit audit tuple `α_i` | auditability |

## 2. Why the severity blend is removed

Algorithm 2 executes `C_f[i] = w_ℓ C_ℓ[i] + (1−w_ℓ) C_b[i]` with `w_ℓ ∈ [0.3, 0.8]`. Three properties of that
line do not survive contact with a Byzantine reasoning plane:

1. **It executes a command no agent proposed.** The convex combination of a safe baseline and an unsafe
   proposal can violate constraints that both endpoints satisfy — envelope containment is not preserved by
   blending against a nonconvex feasible set (§3).
2. **`w_ℓ` is monotone in severity, and severity is exactly what an adversary manipulates.** Under Algorithm 2
   an attacker who inflates the apparent jamming severity buys authority: `w_ℓ → 0.8`. This is the
   self-certification failure INV4 forbids.
3. **It is not auditable.** After the fact, "the LLM had weight 0.63" does not identify what the LLM asked for
   or whether it should have been allowed.

The gate replaces it with a **discrete authority lattice**: a proposal is admitted **whole** or discarded, and
severity unlocks levels rather than scaling trust. TVT's behaviour is recoverable as the special case where
only `{A0, A1}` are ever admissible, which makes the journal controller the natural ablation baseline.

## 3. Safety envelope, and the honest statement about projection

`E_i` splits into a convex part and a nonconvex part, which are treated differently:

- **Convex, projected** (`Π_cvx`, applied in fixed order — horizontal speed, vertical rate, altitude band,
  turn rate): `‖u_h‖ ≤ v_max`, `|u_z| ≤ v_z^max`, `z_i + u_z Δt ∈ [z_min, z_max]`, heading change `≤ θ_max`.
  Fixed order makes `Π_cvx` a deterministic total function; each clip is idempotent and the later clips
  preserve the earlier bounds (rotation preserves norm), so one pass suffices.
- **Nonconvex, verified only** (`Verify`): inter-vehicle separation `‖p_i^+ − p_j^+‖ ≥ d_min` and connectivity.
  These sets are complements of balls; there is no deterministic metric projection worth defending, so the
  gate does not project onto them — it **rejects the candidate and descends a level**.

This distinction is why the gate is a descent over candidates rather than a single projection: a level can be
*unreachable* rather than *corrected*, and the descent terminates because `BRAKE = 0` is feasible from any
feasible state. Numbers used throughout: `v_max = 12 m/s`, `v_z^max = 3 m/s`, `θ_max = 0.52 rad ≈ 29.8°/round`,
`z ∈ [20, 140] m`, `d_min = 8 m`, `Δt = 1 s`.

## 4. Determinism

**Proposition 1 (determinism).** The gate is a pure function of `(x_i, E_i, s_i, C_b, p_i, R, o, F, G_r)`.

The implementation contains no clock, no RNG, and no dependence on set or dict iteration order: `Canon` sorts
by `(src, seq)`, `MaxConsistent` enumerates subsets in index order and breaks ties by the lexicographically
least `src` tuple, `Fuse` uses the lower median for even counts and a circular median chosen from *observed*
values with a value tie-break, and the final loop descends a fixed level order. Stochasticity is confined to
L3 (the LLM), which lies strictly above the gate — INV2 of Stage 2.

This matters operationally because the reasoning plane delivers records in arbitrary order under jamming:
T2 verifies that permuting arrival order leaves both the admitted level and the executed command bit-identical.

**Proposition 2 (containment, INV1).** Every executed command satisfies all envelope constraints.
Immediate from the descent: a level is returned only after `Verify`, and `BRAKE` terminates the descent.

**Proposition 3 (liveness floor, INV5).** A command is always defined, and `A0` is always a candidate.

## 5. The trim step is harmful here — a measured result

The gate was first written with a W-MSR-style step: discard the `F` records whose bearings are most extreme
about the circular median, then require `F+1` consistent survivors. Imported by analogy from scalar consensus,
it *creates* an attack channel: the median used as the trim reference is itself computed over the contaminated
set, so a single Byzantine record changes **which honest record** gets discarded. Measured over 20,000
adversarial draws at `N = 7`, `F = 1` (one Byzantine proposer at median + 180°, all fields maximal):

| quorum rule | A3 availability | mean shift | p99 | max shift |
|---|---|---|---|---|
| trim, then `\|c\| ≥ F+1` | 97.6 % | 0.203° | 4.22° | **59.59°** |
| no trim, `\|c\| ≥ F+1` | 98.0 % | 0.007° | 0.00° | 54.36° |
| no trim, `\|c\| ≥ 2F+1` | 93.2 % | 0.000° | 0.00° | **0.00°** |

Adopted: **no trim, `|c| ≥ 2F+1`**. Exact immunity to a single Byzantine proposer costs 4.4 points of quorum
availability. The zero is not luck — it is the breakdown point of the median: with `|c| ≥ 2F+1` consistent
records, honest sources are a strict majority *inside* `c`, so `F` of them cannot move a coordinatewise median.
`trim` remains in the module, documented as belonging to the **scalar** path (severity and link-quality maps),
which is where Stage 3 §12.1 says the graph condition applies. Note also that `|c| ≥ 2F+1` peers reproduces the
counting condition `|N_i^r| ≥ 2F+1`, hence `N ≥ 2F+2` — the Stage 2 bound, now arrived at from the attack side.

## 6. Bounded influence, and the tolerance that sets it

**Proposition 4 (bounded influence).** If `F` Byzantine records enter the quorum set `c` with `|c| ≥ 2F+1`,
they can shift the fused bearing by at most the consistency tolerance `τ_b`, and the bound is tight.

To be inside `c` a record must be pairwise consistent with every member, so it lies within `τ_b` of them; the
median then cannot leave the honest span by more than `τ_b`. T8 attains it exactly: two colluding proposers
placed on the tolerance edge at `F = 2` shift the fused bearing by **11.25°**, which is `τ_b = 16` units
(`16 × 360/512`) to the unit. `τ_b` is therefore a security parameter, not a comfort setting:

| `τ_b` | in degrees | A3 availability (`F=1`) | max influence (`F=2`) |
|---|---|---|---|
| 8 | 5.62° | 66.0 % | 5.62° |
| **16** | **11.25°** | **89.8 %** | **11.25°** |
| 24 | 16.88° | 95.2 % | 16.88° |
| 32 | 22.50° | 96.8 % | 21.80° |
| 48 | 33.75° | 96.8 % | 21.80° |

Adopted `τ_b = 16` units. Rationale: worst-case adversarial heading influence (11.25°) stays below the
per-round turn limit `θ_max` (29.8°), so a colluding pair can never dominate a single round's manoeuvre, and
availability saturates above 24 units anyway — beyond it the binding constraint is intent-class agreement,
not bearing tolerance, so wider tolerance buys influence without buying quorum.

## 7. Property suite

325,000 randomised trials, fixed seed, `test_gate.py`. All pass.

| test | trials | result |
|---|---|---|
| T1 wire round trip | 200,000 | every field exact through pack/unpack |
| T2 determinism + arrival-order invariance | 20,000 | identical level and command under repetition and permutation |
| T3 INV1 envelope containment | 20,000 | every non-BRAKE command verified; BRAKE in 0.3 %, half of those from an already-infeasible state |
| T4 INV5 liveness floor | 20,000 | `A0` candidate present in every call |
| T5 INV4 no self-certification | 20,000 | peers claiming maximal severity, link quality and authority raised the admitted level in **0** cases |
| T6 one Byzantine proposer, `F=1` | 20,000 | no sub-quorum admission; max induced heading shift **0.00°** |
| T7 gate latency, `N=7` | 5,000 | **0.07 ms** mean, pure Python |
| T8 two colluding Byzantine, `F=2` | 20,000 | no sub-quorum admission; max fused-bearing shift **11.25°** = `τ_b` |

Two measurement traps were hit and corrected while building the suite, both worth stating in the paper:
executed-command deviation saturates at `2θ_max = 59.6°` and therefore measures envelope clipping rather than
adversarial influence — influence must be read on the **fused proposal**; and an honest-only baseline is
unreachable at `F = 2, N = 7` (only four honest peers against a quorum of five), so the reference for a
collusion test must be the honest fused command, not an honest-only gate decision.

## 8. Cost and what it buys

At `N = 7` the gate costs **0.07 ms** per vehicle per round; Rule G1's robustness check costs 5.8 ms (Stage 3).
Together they are under 1 % of a 1 Hz round, i.e. the deterministic safety layer is free relative to LLM
inference, which is the entire point: nothing in the safety argument depends on the model being fast, small,
aligned, or even honest.

## 9. Open items for Stage 5

1. `MaxConsistent` is exhaustive over subsets — fine to `N = 7`, exponential beyond. A greedy or
   clique-based variant must be shown to preserve Propositions 1 and 4 before any claim above `N = 8`.
2. Records inside `c` are equally weighted. Weighting by observed link quality would improve fusion under
   asymmetric jamming but reintroduces a sender-influenced quantity — INV4 requires the weight be computed
   from *locally observed* link state only.
3. `Verify` uses one-step linear prediction of neighbour positions. A reachability-based check would be
   stronger; the cost budget in §8 leaves room for it.
4. The `intent` field is compared for exact equality. Whether semantically compatible intents
   (`transit` vs `regroup`) should be fusable is a semantics question the codebook currently forecloses.

---

## 10. Corrections from Stage 5 (added after `aggregation_spec.md`)

1. **`Verify` now implements connectivity.** §3 above described two non-convex constraints,
   separation and connectivity; the reference implementation checked only separation. Connectivity
   is now enforced — at least one predicted neighbour within `r_comm = 150 m`, and only when the
   vehicle is currently connected, so an isolated vehicle is not brake-locked. All eight properties
   still hold (T6 0.00°, T8 11.25°). Failure case F6 exercises it.
2. **`Canonicalise` keeps one record per source per round.** Proposition 4 counts distinct sources,
   but two different `seq` values from one source both fell inside the freshness window, letting a
   single fault take two quorum slots (measured 21.09° influence, above `τ_b`). Freshest record per
   source now wins. See `aggregation_spec.md` §6.
3. **`MaxConsistent` is superseded** by the exact window sweep of `aggregation_spec.md` §3, which
   returns the identical set (0 disagreements in 20,000 draws) in polynomial time. The `N ≤ 8`
   caveat in §9 item 1 is withdrawn.

---

## 11. Corrections from Stage 6 (added after `integration_spec.md`, 2026-09-02)

1. **The separation clause of §3 was unsound and has been replaced.** `Verify` compared the vehicle's
   predicted position against neighbours' *current* positions. Whenever a neighbour can travel further
   than `d_min` in one step — the normal case at simulator scale, 2 m of travel against `d_min` — two
   vehicles both pass the clause and still collide. Measured on the simulator: 11–23% of steps below
   `d_min`, minimum true separation 0.00 m, with the gate reporting no violation. The clause is now
   `dist(p_pred, q) ≥ d_min + v_max·dt` — the neighbour's reachable displacement, worst case
   (`verify_sound` in `swarm_bridge.py`). Cost: 6–28% of commands braked, no measurable loss of
   progress. This closes open item 3 of §9.
2. **The neighbour's reported speed must not be used for the inflation.** The record carries a speed
   field, so inflating by the *reported* neighbour speed looks cheaper. It is not sound: the field is a
   proposal, not a state report, and a Byzantine neighbour understates it freely. Measured, it still
   admits violations. INV4 (no sender-supplied quantity in a safety decision) applies to the envelope,
   not only to fusion weights.
3. **The authority admission order was wrong.** §5 admitted the autonomous escape above the inherited
   controller. Because the escape candidate exists whenever severity is non-zero, any jamming discarded
   the controller's command and the reasoning plane was bypassed entirely. Admission now descends
   A4 operator → A3 quorum → A2 own record → A0 inherited controller → A1 escape → brake. The escape is
   a liveness fallback, not an authority.
4. **`τ_b = 16` units (11.25°) is confirmed as the right setting** by measurement rather than argument.
   Availability saturates near 22°, and 11.25° already delivers 93% of the reachable quorum
   availability; tightening to 5.62° removes Byzantine records from quorums but costs ~27% of
   availability, and the influence it buys back is already bounded by Proposition 4. Largest measured
   fused-bearing deviation: 9.14° against the 11.25° bound.
5. **(2F+1, F+1)-robustness of the delivered graph is reported, not required.** Enforcing it in-round
   makes the quorum level essentially unavailable (0.00% against 1.48% at N = 5), because demand for
   the level and graph sparsity share a cause. Proposition 4 needs only the local counting condition.
