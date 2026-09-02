# Stage 5 — Byzantine-resilient aggregation

**Receiver-side aggregation, enumerated failure cases, and bounds**
Closes the open items of `override_spec.md` §9. Companion figure `aggregation_results.png`;
tables `aggregation_scaling.csv`, `kappa_tradeoff.csv`, `failure_cases.csv`.
Reference implementation `aggregation.py`, adversaries `redteam.py`.
Checklist row 5: "Design BFT layer: receiver-side filtering, quorum gate, compromised-agent tests."

---

## 1. What Stage 5 changes

| Stage 4 open item | Stage 5 resolution |
|---|---|
| `MaxConsistent` enumerates subsets: `O(2^n)`, unusable past `N ≈ 8` | Exact **window sweep**, `O(n^3 · n)` worst case, 0.40 ms at 50 records (§3) |
| Link-quality weighting risked violating INV4 | Weights computed from **locally observed geometry only**; quorum threshold raised to compensate (§4) |
| Trim retained but unjustified after Stage 4 removed it from the quorum path | Trim earns its place on the **scalar** path, measured 3.0× better than a mean (§5) |
| — (found in this stage) | **One record per source per round**: without it a single fault holds two quorum slots (§6) |

Three defects were found by measurement in this stage, not by inspection: the multi-slot fault (F10),
an unimplemented connectivity check in `Verify` (§7.1), and the negative result on weighting (§4.3).

---

## 2. The aggregation protocol

Executed by every vehicle `i` independently, on its own received set, once per round `k`.
No step consults a sender-supplied trust field; that is invariant INV4.

```
R0  receive     collect proposal records from N_i^r(k) (the reasoning-plane neighbourhood)
R1  authenticate  verify the message tag                      [assumption A-AUTH, §7.2]
R2  canonicalise  drop expired (ttl = 0) and stale (age > 1 round) records;
                  drop any source that equivocated in this round;
                  keep exactly ONE record per source -- the freshest              (§6)
R3  select        c <- MaxConsistentSweep(records)             exact maximum clique (§3)
R4  admit         if |c| >= floor(F(1+kappa)) + 1  AND  G_r is (F+1,F+1)-robust   (Rule G1)
                  then fuse c and offer the result as the A3 candidate
R5  gate          admission_gate: descend the authority lattice, project the convex
                  envelope, verify the non-convex envelope                  (Stage 4)
```

Scalar quantities (own severity, link-quality map) do **not** go through R3–R4. They take the
W-MSR path of §5, because they are numeric and one-dimensional; intents are discrete and do not
admit a meaningful average. This split is the correction Stage 4 forced.

---

## 3. Exact aggregation in polynomial time

**Lemma 1 (window containment).** A set of records is pairwise consistent iff it has one `intent`
and lies inside a single window of extent `(τ_b, τ_u, 1)` in `(bearing, speed, alt_band)`.

*Proof.* Consistency is a conjunction of per-coordinate interval conditions, so each record is an
axis-parallel box of fixed size and consistency is box intersection. Axis-parallel boxes are a
Helly family in any dimension: pairwise intersection implies common intersection. Hence a pairwise
consistent set has a common point, and the window centred there contains all of it. For `bearing`
the coordinate is circular; since `τ_b = 16 << 512/3`, a set with pairwise circular distance `≤ τ_b`
lies in an arc of length `τ_b`, so the same argument applies on the unrolled coordinate. ∎

**Proposition 5 (exactness and determinism).** `MaxConsistentSweep` returns a maximum consistent
subset, and returns the *same* one as the Stage 4 enumeration.

*Proof.* By Lemma 1 every maximal consistent set is a window; every window can be translated until
its lower corner meets a member in each coordinate, so enumerating windows anchored at observed
coordinate triples enumerates all maximal consistent sets. The tie-break — least tuple of `src` —
is the Stage 4 rule, so the selected set is identical. ∎

Verified rather than assumed: **0 disagreements** with the exhaustive enumeration in 20,000 random
draws (`n` = 2–10, three intent classes), comparing the returned source sets, not merely their size.

| records | subset enumeration | window sweep | speed-up |
|---|---|---|---|
| 10 | 1.06 ms | 0.034 ms | 31× |
| 14 | 13.7 ms | 0.024 ms | 575× |
| 18 | 226 ms | 0.076 ms | 2,962× |
| 22 | 3,501 ms | 0.098 ms | 35,693× |
| 30 | — (infeasible) | 0.121 ms | — |
| 50 | — (infeasible) | 0.401 ms | — |

The `N ≤ 8` restriction in Stage 4 §9 is withdrawn. At 50 records the aggregation costs 0.4 ms,
so swarm size is now limited by the *counting* condition and by inference latency, not by the
aggregator. Propositions 1 and 4 are preserved exactly because the returned set is identical.

---

## 4. Receiver-computed weights, and why they are not used on the intent path

### 4.1 INV4-W

`w_ij = clip(φ'(‖p_i − p_j‖), w_min, w_max)`, with `φ'` the TVT link-quality function evaluated on
the receiver's *own* range estimate. The `linkq` and `sev` fields of a received record are decoded
for logging and for the scalar path, and are never multipliers on that record's own influence.
A sender therefore cannot raise its own weight — the property tested as F9.

### 4.2 The weighted quorum condition

**Proposition 6.** Let `κ = w_max/w_min`. The weighted median over a quorum set `c` containing at
most `F` Byzantine sources is inside the honest range iff Byzantine weight is under half:
`F·w_max < (|c| − F)·w_min`, i.e. `|c| > F(1 + κ)`. The gate therefore requires
`|c| ≥ ⌊F(1+κ)⌋ + 1`, which reduces to Stage 4's `2F+1` at `κ = 1`.

### 4.3 Measured: weighting costs availability and buys nothing (N = 7, F = 1)

| `κ` | quorum required | A3 availability | worst observed influence |
|---|---|---|---|
| 1.0 | 3 | 97.7% | 10.55° |
| 1.5 | 3 | 97.7% | 10.55° |
| 2.0 | 4 | 91.3% | 10.55° |
| 3.0 | 5 | 59.3% | 10.55° |

Worst-case influence is **flat in `κ`**: it is set by `τ_b` and by the counting condition, not by
the weights, because the bound of Proposition 4 is attained by colluders that are *inside* the
tolerance and therefore weighted like honest neighbours. Weighting only shrinks the admissible
quorum. **Decision: `κ = 1` on the intent path** — the discrete quorum is unweighted. Link-quality
weighting is retained only where it changes an outcome, the scalar path of §5. This contradicts
the plan in `new_research.md` row 4 ("trims the worst F proposals, MSR/SAC-style") for the intent
path; the receiver-side principle survives, the weighting and the trim do not.

---

## 5. The scalar path (W-MSR)

Severity and link-quality maps are fused with one W-MSR round: discard the up-to-`F` values above
and `F` below the vehicle's own measurement, average the rest with own value. Requires
`|N_i| ≥ 2F+1` and `(F+1,F+1)`-robustness — the condition Rule G1 already checks.

Measured, F = 1, six neighbours, 20,000 draws with Byzantine severities pinned at 0 or 1:

* fused severity outside the honest range `[min, max]`: **0 cases** (the W-MSR guarantee);
* worst deviation from the honest-only fusion: **0.0467** (severity units, scale 0–1);
* worst deviation had a plain mean been used: **0.1397** — **3.0× worse**.

Trim is therefore correct where the quantity is numeric and wrong where it is an intent, which is
exactly the split Stage 4 established from the opposite direction.

---

## 6. One record per source per round

Found by measurement (F10). A withhold-then-burst adversary emits records for several sequence
numbers at once. Both `seq = k` and `seq = k−1` sit inside the freshness window, so before this fix
a *single* Byzantine source contributed two records to the quorum set and voted twice: measured
worst-case influence **21.09°**, above the `τ_b = 11.25°` bound, with the quorum condition never
violated as stated.

The hypothesis of Proposition 4 counts **distinct sources**, so canonicalisation must enforce it:
freshest record per source wins; equal `seq` with differing payload is equivocation and drops the
source entirely. After the fix, F10 measures **7.73°**, inside the bound. This is a one-line change
with a load-bearing justification, and it is now documented as such in `override_gate.py`.

---

## 7. Enumerated failure cases

Full table in `failure_cases.csv` (13 classes, 52,000 adversarial rounds). Layers follow the
three-layer threat model of the survey paper. **Zero unsafe commands were executed in any class**,
including the classes where the influence bound does not hold.

| id | class | defence | residual |
|---|---|---|---|
| F1 | equivocation | source dropped in R2 | 0 |
| F2 | stale replay | freshness window | 0 |
| F3 | self-certified severity/authority | INV4 | 2.11° (clique reselection only) |
| F4 | single extreme proposer | quorum `2F+1`, no trim, no blend | 0 |
| F5 | colluding pair on the tolerance edge | Prop. 4 | **11.25° = τ_b, tight** |
| F6 | envelope-violating consensus (all sources) | `Π_cvx` + `Verify` + descent | 0 unsafe; 14.2% of rounds fall to BRAKE |
| F7 | intent flooding | liveness only | 0 |
| F8 | bearing fragmentation | liveness only; sweep still exact | 1.41° |
| F9 | link-quality inflation | INV4-W | 9.84° |
| F10 | withhold-then-burst | one record per source (§6) | 7.73° |
| F11 | source impersonation | equivocation detection; **needs A-AUTH** | 10.55°, −3.6 pt availability |
| F12 | **actual faults exceed declared `F`** | **none** | **unbounded; 22.50° observed** |
| F13 | robustness lost to jamming | Rule G1 withdraws A3 | 0 (A3 unavailable) |

### 7.1 Correction to `override_spec.md` §3

The Stage 4 spec described `Verify` as checking two non-convex constraints, separation *and*
connectivity; the shipped implementation checked separation only. Connectivity is now implemented:
at least one predicted neighbour within `r_comm = 150 m`, enforced only when the vehicle is
currently connected, so an already-isolated vehicle is not brake-locked. All eight Stage 4
properties still hold with it enabled (T6 0.00°, T8 11.25°).

F6 is the class that exercises it. Influence is **not defined** for F6 — the adversary rewrites
every record, so there is no honest reference — which is why the table reports the safety metric
instead: 0 unsafe commands, and the gate refuses to execute at all in 14.2% of rounds.

### 7.2 Assumptions that failure cases expose

* **A-AUTH (message authentication).** F11 shows an unauthenticated adversary can impersonate an
  honest source. The gate degrades safely — the collision is read as equivocation and both records
  are dropped — but that lets one fault *silence* one honest peer, costing 3.6 points of A3
  availability. Attribution needs a MAC; per-message signing over the compact schema of Stage 3 is
  a Stage 6 item with a bandwidth cost to measure.
* **A-FBOUND (`F` is an upper bound on real faults).** F12 is the necessity proof for the swarm-size
  rule. With three faults and the gate configured for `F = 1`, the quorum set was
  Byzantine-**majority** in **90.9%** of A3 rounds and worst-case influence reached 22.50°.
  Proposition 4's hypothesis is violated, so no bound is claimed — this is the boundary of the
  design, not a leak inside it. What survives is containment: still zero unsafe commands, because
  INV1 does not depend on `F`. **Safety is unconditional; bounded influence is `F`-conditional.**
  That distinction is the paper's headline claim and should be stated in the abstract in those terms.

---

## 8. Bounds, collected

| # | Statement | Depends on |
|---|---|---|
| 1 | Determinism: identical output under repetition and arrival-order permutation | construction |
| 2 | INV1 containment: every executed command satisfies the envelope | unconditional in `F` |
| 3 | INV5 liveness: `A0` always admissible; BRAKE feasible from any feasible state | unconditional |
| 4 | Influence `≤ τ_b` on the fused heading | `b ≤ F`, `|c| ≥ 2F+1` distinct sources |
| 5 | Sweep exactness: identical selection to enumeration | Lemma 1 (Helly) |
| 6 | Weighted breakdown: honest range preserved iff `|c| > F(1+κ)` | `κ` bound on weights |

---

## 9. Open items for Stage 6

1. **A-AUTH costed.** Signing or MAC-ing the 12-byte record changes the Stage 3 bandwidth budget
   materially; a truncated MAC trades forgery probability against a per-record byte count that must
   be measured, not assumed.
2. **Runtime estimation of `F`.** Every bound in §8 is conditional on `F`, and F12 shows what
   happens when the declaration is wrong. Behaviour-based fault counting is itself attackable and
   needs its own threat model.
3. **`τ_b` under real jamming.** `τ_b = 16` was chosen against the turn-rate limit. Whether honest
   agents under degraded links stay inside it — which sets the availability half of the trade-off —
   is a simulation question for the Swarm Squad integration.
4. **Sparse `G_r` inference cost.** Still unmeasured, and still the only surviving argument for a
   sparse reasoning graph after Stage 3 retired the bandwidth argument.
