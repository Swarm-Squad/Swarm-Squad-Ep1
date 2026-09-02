# Stage 3 — Message interface and bandwidth budget

**Compact action–state schema for the reasoning plane `G_r(t)`**
Companion to `architecture_spec.md` (Stage 2). Fixes interface **I2** (peer proposal record) and the
selection rule for `G_r(t)`. Figures: `bandwidth_budget.png`. Tables: `bandwidth_budget.csv`,
`topology_robustness.csv`. Machine-readable: `proposal_record.schema.json`.

Stage 3 is complete when there is a message interface document plus a bandwidth budget against free-form
natural language. Both are below; every number in §7–§9 is computed, not estimated.

---

## 1. The problem Stage 3 actually has to solve

SQ3 asks for a representation that "retains enough natural-language interpretability for human oversight
while bounding per-round bandwidth on jammed links". These pull in opposite directions only if you assume
one representation must serve both purposes. It does not. The design here uses **three representations of
the same record**, with a canonical form and two total functions out of it:

| # | Representation | Who produces / consumes it | Where it travels |
|---|---|---|---|
| R1 | Constrained JSON, ~14 keys | emitted by L3 (the LLM), consumed by L2 | inside one vehicle, never on the air |
| R2 | **9-byte bit-packed record** | produced by L2, consumed by peer L2 | the reasoning plane `G_r(t)` — **canonical** |
| R3 | English sentence | rendered by L2/L4 from R2 | operator uplink and audit log only |

R2 is canonical: `pack(R1)` is defined for every schema-valid R1, `render(R2)` is a total function into R3,
and nothing is transmitted between vehicles except R2. Interpretability is therefore not paid for in
airtime — it is reconstructed at the operator interface from the codebook (§4), and the audit log stores
R2 plus the render, so a human always sees a sentence and the sentence is always exactly what the peers saw.

This also removes an attack surface. Free text on the peer plane is an injection channel into a neighbour's
LLM context; a 9-byte fixed-layout record with no free-text field is not. There is **no string on the
reasoning plane at all** — see §10.

## 2. Record semantics

One record is a *proposal*, never a command: "I, vehicle `src`, at round `seq`, propose action `intent` with
parameters `p`, on evidence `e`, requesting authority level `req_auth`, valid for `ttl` rounds." The receiving
vehicle's L2 gate decides what, if anything, to do with it (Stage 2 §5–§6). Nothing in the record can raise
the sender's own authority — INV4.

## 3. Field table (wire format R2)

Bit widths are fixed. Total **67 bits → 9 bytes** with 5 spare bits reserved (set to zero, reject if nonzero).

| Field | Bits | Range / units | Resolution | Purpose |
|---|---|---|---|---|
| `ver` | 2 | 0–3 | — | protocol version; reject unknown |
| `src` | 3 | 0–7 | — | proposer id (supports N ≤ 8) |
| `seq` | 6 | 0–63, wraps | 1 round | freshness / replay (§6) |
| `ttl` | 4 | 0–15 | 1 round | proposal expiry |
| `intent` | 5 | 32 classes | — | action class codebook (§4) |
| `bearing` | 9 | 0–360° | 0.703° | desired course over ground |
| `speed` | 5 | 0–1 × v_max | 3.125% | commanded speed fraction |
| `alt_band` | 4 | 16 bands | 1 band | altitude band, absolute not delta |
| `target` | 3 | 0–7 | — | referent (peer id or waypoint slot) |
| `sev` | 4 | 0–1 | 0.0625 | proposer's own jamming severity `s_i` |
| `linkq` | 4 | 0–1 | 0.0625 | proposer's local link quality |
| `flags` | 8 | 8 booleans | — | observation bits (§4) |
| `req_auth` | 2 | 0=A0 … 3=A3 | — | authority level requested; A4 (operator) is **not representable** |
| `rationale` | 8 | 256 entries | — | index into the rationale codebook (§4) |

`sev` and `linkq` are the proposer's *self-report*. They are evidence for the receiver's filter, never a
weight on the proposer's own vote — a Byzantine node that reports `sev = 1.0` gains nothing (INV4).

## 4. Codebooks

**Intent (5 bits, 32 slots).** 0 `hold`, 1 `maintain_formation`, 2 `formation_break`, 3 `rejoin_formation`,
4 `egress_bearing`, 5 `egress_tangent`, 6 `ingress_bearing`, 7 `altitude_change`, 8 `speed_change`,
9 `waypoint_bias`, 10 `loiter`, 11 `relay_reposition`, 12 `role_swap`, 13 `split_group`, 14 `merge_group`,
15 `return_to_launch`, 16 `abort_task`, 17 `report_only`, 18 `request_relay`, 19 `mark_jammer`,
20 `mark_obstacle`, 21 `declare_degraded`, 22 `declare_recovered`, 23 `suspect_peer`, 24–30 reserved,
31 `null` (no-op, used as a keepalive that still carries `sev`/`linkq`/`flags`).

Slot 23 `suspect_peer` matters for Stage 5: accusations are first-class messages, but an accusation is a
proposal like any other and is subject to the same filter. One node cannot exclude another by asserting it.

**Flags (8 bits).** b0 `jammer_detected`, b1 `link_loss`, b2 `formation_error_rising`, b3 `obstacle_near`,
b4 `position_degraded`, b5 `peer_silent`, b6 `envelope_clipped_last_round`, b7 `stagnation_detected`.

b6 is the loop-closing bit: it tells peers that this vehicle's *previous* proposal was clipped by its safety
envelope. Without it, a swarm can spend many rounds proposing manoeuvres that a neighbour's projection
silently discards.

**Rationale (8 bits, 256 slots).** A fixed table of canonical English clauses, versioned with `ver` and
identical on every vehicle, e.g. 37 = "inside jamming zone, peer unreachable, formation degrading".
Population rule: entries are added only between builds, never at runtime, and each entry is a *reason*, not
an *instruction*. The operator view renders `intent + params + rationale` into one sentence, so R3 for the
worked example in §7 is: *"V3 (sev 0.75, link 0.31): break formation and egress on 118° at 0.78 v_max,
climb to band 9, hold 3 rounds — inside jamming zone, peer unreachable, formation degrading. Requests A3."*

If the LLM's reasoning does not fit any codebook entry, L2 emits slot 0 `unclassified` and the proposal is
capped at A2 — an unexplainable proposal cannot obtain quorum authority. This is a deliberate incentive:
the swarm can only take high-impact action for reasons the operator can read.

## 5. Frame layout

```
byte:  0        1        2        3        4        5        6        7        8      [9..12]
      +--------+--------+--------+--------+--------+--------+--------+--------+--------+---------+
      | ver src seq ttl | intent bearing  | speed alt tgt    | sev linkq | flags | auth rat | MAC |
      +-----------------+-----------------+------------------+-----------+-------+----------+-----+
```
Big-endian bit packing, fields in table order, MSB first, no padding between fields, 5 trailing reserved
bits. The optional 4-byte truncated MAC authenticates *origin only*; it is not part of the Byzantine
argument, because a compromised agent holds valid keys. Recommend enabling it (13 B total, still 50× smaller
than free-form NL) to keep outsiders off the reasoning plane, and stating plainly in the paper that it does
nothing against an insider.

## 6. Freshness, replay, and staleness

- A record is **fresh** at receiver `j` iff `seq` is within `+0/−1` of `j`'s current round and `ttl > 0`.
- Records outside that window are dropped, not queued. A stale proposal is worse than no proposal: it
  describes a jamming geometry that has moved.
- Per-source duplicate suppression: at most one record per `(src, seq)` is admitted; a second one with
  differing payload is a **detected equivocation** and is logged as Stage 5 evidence (`suspect_peer`).
  Equivocation is the one Byzantine behaviour a receiver can prove locally, and the 6-bit `seq` is what
  makes it provable.
- `seq` wraps every 64 rounds; with `ttl ≤ 15` the wrap is unambiguous as long as a vehicle is silent for
  fewer than 48 consecutive rounds. Longer silence forces re-sync at A0/A1, which is the safe direction.

## 7. Bandwidth budget vs. free-form natural language

All encodings below carry **identical semantic content** — the worked example of §4. Sizes are measured, not
estimated (`bandwidth_budget.csv`).

| Encoding | Bytes | vs. free-form NL |
|---|---|---|
| Free-form NL (LLM prose, 6 sentences) | 649 | ×1 |
| Free-form NL + zlib | 383 | ×1.7 |
| Verbose JSON (descriptive keys, floats, string enums) | 499 | ×1.3 |
| Terse NL (one line, telegraphic) | 171 | ×3.8 |
| Compact JSON (2-char keys, quantized ints) | 92 | ×7.1 |
| CBOR (compact dict) | 47 | ×13.8 |
| MessagePack (compact dict) | 44 | ×14.8 |
| **Bit-packed R2 (proposed)** | **9** | **×72** |

Aggregate offered load on the reasoning plane, unicast on a complete `G_r`, one round per second:

| N | free-form NL | compact JSON | MessagePack | **bit-packed** |
|---|---|---|---|---|
| 3 | 31.2 kbit/s | 4.4 | 2.1 | **0.43** |
| 4 | 62.3 | 8.8 | 4.2 | **0.86** |
| 5 | 103.8 | 14.7 | 7.0 | **1.44** |
| 6 | 155.8 | 22.1 | 10.6 | **2.16** |
| 7 | 218.1 | 30.9 | 14.8 | **3.02** |

At N = 7 the reasoning plane costs **3.0 kbit/s** bit-packed against **218 kbit/s** free-form. Per vehicle
that is 54 B/s received on a complete graph. The headline sentence for the paper: *a seven-vehicle swarm
runs one LLM proposal round per second per vehicle inside a 3 kbit/s budget, 72× below the same content as
LLM prose.*

## 8. Selecting the reasoning graph — and one result that changes the plan

r-robustness was computed exhaustively for every candidate topology at N = 3–7 (`topology_robustness.csv`);
`F_total` is the largest F for which the graph is (F+1, F+1)-robust, the necessary and sufficient condition
for W-MSR under the F-total model, and `F_local` requires (2F+1)-robustness.

| N | topology | degree | edges | r | F_total | F_local |
|---|---|---|---|---|---|---|
| 5 | ring C(1) | 2 | 5 | 1 | 0 | 0 |
| 5 | C(1,2) ≡ K₅ | 4 | 10 | 3 | 2 | 1 |
| 6 | C(1,2) | 4 | 12 | 2 | 1 | 0 |
| 6 | K₆ | 5 | 15 | 3 | 2 | 1 |
| 7 | C(1,2) | 4 | 14 | 2 | 1 | 0 |
| 7 | K₇ | 6 | 21 | 4 | 3 | 1 |

Three consequences, one of them against the original plan:

1. **Rings are useless here.** C(1) is not even 1-robust at N ≥ 6 — the two-halves partition defeats it. A
   sparse reasoning plane cannot be a ring.
2. **At N = 5, C(1,2) *is* K₅.** Sparsification is not even definable below N = 6, so the sparse-graph story
   only exists at N = 6–7 in the stated swarm-size range.
3. **Once the schema is bit-packed, sparsifying `G_r` is a bad trade at N ≤ 7.** Going from K₇ to C(1,2) saves
   1.0 kbit/s (3.02 → 2.02) and costs Byzantine tolerance F_total 3 → 1. The bandwidth argument for sparsity
   was an argument against *verbose* messages, and compression already won it. The defensible reason to
   sparsify at these sizes is **per-vehicle inference cost** — degree sets how many records enter the LLM
   context each round, which is compute and latency, not airtime — so that is how it should be argued in the
   paper, with a measurement in Stage 7, or dropped.

**The graph is not chosen anyway — it is observed.** Under jamming, edges disappear. So `G_r(t)` is defined
as complete-by-design, degrading to whatever survives, and the robustness test runs *online*: an exhaustive
(F+1, F+1)-robustness check on the live edge set costs **5.8 ms at N = 7** (pure Python, worst case), so it
fits inside a 1 Hz round with roughly 170x of margin. This gives a clean coupling back to the
Stage 2 authority lattice:

> **Rule G1.** Level A3 (quorum override) is admissible at round *k* only if the reasoning graph observed at
> round *k* is (F+1, F+1)-robust. Otherwise the swarm degrades to A2 and below.

Rule G1 turns "we assume an r-robust graph" — an assumption every paper in the Stage 1 matrix makes and none
of them verifies at runtime — into a checked precondition. It is cheap, it is novel in this setting, and it
is the reason the Stage 1 axis A5 can be scored 2 rather than 1 for this work.

## 9. Quantization error budget

| Quantity | Step | Worst-case error | Consequence |
|---|---|---|---|
| bearing (9 b) | 0.703° | 0.352° | 0.61 m lateral at 100 m lookahead |
| speed (5 b) | 3.125% of v_max | 1.56% | below the controller's own tracking error |
| severity (4 b) | 0.0625 | 0.031 | one step ≪ the A1 trigger hysteresis |
| altitude | 1 band | ½ band | bands are the safety unit; no loss |

Every quantum is smaller than the corresponding controller tolerance, so the packing introduces no error the
closed loop can distinguish. This is worth stating explicitly in the paper because the natural reviewer
question is whether 9 bytes throws away control authority. It does not; it throws away *prose*.

## 10. What the schema deliberately cannot express

- **No free text.** No field can carry an arbitrary string, so a compromised LLM cannot use the reasoning
  plane to inject instructions into a neighbour's context. The semantic attack surface is reduced to
  "choose a wrong value from a fixed set", which is exactly the space the Stage 5 filter operates on.
- **No commands.** There is no field addressed to another vehicle's actuators. `target` names a referent,
  not a recipient of authority.
- **No self-granted authority.** `req_auth` is a request; admission is decided by the receiver. The field
  is 2 bits and encodes A0–A3 only, so a peer *cannot* claim the operator level A4 at all: a record
  carrying `req_auth = 4` is unencodable and is dropped by the receiver's deserializer (Stage 6 §6.6).
- **No unbounded parameters.** Every numeric field is range-limited by its bit width, so an out-of-envelope
  proposal is a *representable but rejectable* value, never a parser exploit.

The bit widths are a security property, not only a compression one — that framing is worth a sentence in the
paper, and it is a point none of the 22 works in the Stage 1 matrix makes.

## 11. Implementation notes (Swarm Squad)

- `proposal_record.schema.json` is the R1 contract. Use it directly as the grammar for constrained decoding
  (llama.cpp GBNF / outlines / lm-format-enforcer) so L3 cannot emit an unparseable proposal; then `pack()`
  is total by construction and the Stage 4 override algorithm never sees malformed input.
- Implement `pack`/`unpack` as pure functions with a round-trip property test: `unpack(pack(x)) == quantize(x)`
  for random valid `x`. This is the cheapest possible defence against a class of bugs that would otherwise
  look like Byzantine behaviour in the experiments.
- Log R2 bytes, not the render. The render is reproducible from the bytes; the reverse is not true.
- Bandwidth accounting in the simulator should count *offered* load per link per round, so the Stage 7 plots
  can be produced from the same counters as this table.

## 12. Open decisions

1. **MAC on or off by default.** On costs 4 B (9 → 13, still ×50) and keeps outsiders off the plane; off keeps
   the headline number cleaner. Recommend on, with the honest caveat in §5.
2. **Round period Δt.** All numbers above assume 1 Hz. A 3B-class model on the target hardware may not sustain
   1 Hz with degree-6 context; Δt is the knob that reconciles them, and every rate above scales as 1/Δt.
3. **Whether `bearing` should be absolute or relative to current heading.** Absolute is chosen here (it makes
   a stale record obviously wrong rather than subtly wrong), but relative encoding would let the field shrink
   to 7 bits.

---

## Corrections from Stage 6 (added after `integration_spec.md`, 2026-09-02)

1. **The bearing field must be about a shared referent.** Ego-local bearings (own position →
   destination) make honest records mutually inconsistent purely by geometry — up to 166 quantisation
   units of honest disagreement at N = 5, so no consistent quorum can form. The field is now defined as
   the bearing from the sender's **perceived swarm centroid** to the destination, plus a doctrine veer.
   Under any quantised consistency filter, a proposal field must denote a quantity all honest senders
   estimate of the *same* referent. See `integration_spec.md` §6.2.
2. **The speed field is a proposal, not a state report,** and must not be used in any safety margin
   (`integration_spec.md` §6.4). If a sound reported-speed inflation is wanted, it needs a separate
   authenticated state channel — costed against the 1920 bit/s per-vehicle budget of §4.5.
3. **Authentication and record size must be designed together.** Measured with the simulator's own
   `CryptoAuth`: a 64-bit truncated HMAC on the packed 12-byte record costs 8 bytes and 1.96 µs,
   1920 bit/s per vehicle at N = 7; the repo's JSON frame with a full 32-byte tag costs 19968 bit/s,
   10.4× more. The AEAD options carry ciphertext in the signature field and cost roughly twice the HMAC
   frame with no additional property this design uses. See `integration_spec.md` §4.5.

4. **The `req_auth` encoding is 0 = A0 … 3 = A3.** The field table above previously gave the range as
   "A1–A3 (+reserved)", while the schema (`proposal_record.schema.json`) and both implementations use the
   level index directly. Confirmed against `override_gate.pack/unpack`: values 0–3 round-trip, and
   `req_auth = 4` raises `req_auth=4 does not fit in 2 bits`. The §7 worked example, which requests A3,
   is unaffected — a quorum request encodes `"r": 3`.
   Consequence, now measured (Stage 6 §6.6): the F3 self-certification attack is not expressible on
   the wire. A Byzantine peer setting `req_auth = 4` produces a record that fails the width check, so
   all 1343 of its delivered copies were dropped at the parser and the attack degraded to an
   availability fault — the swarm made *more* progress (0.90 of the start distance) than under the
   representable F5 edge-collusion attack (0.83).
