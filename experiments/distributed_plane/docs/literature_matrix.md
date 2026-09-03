# Stage 1 — Literature comparison matrix

**Byzantine-resilient distributed LLM-per-vehicle swarm control**  

Scope: the 13 references of `new_research.md`, each verified against its primary source, plus 9 works added by a systematic arXiv gap search. Stage 1's completion criterion is a table stating explicitly what each prior work does *not* do; that is the `does_not_do` column of `byzantine_llm_swarm_matrix.csv` and Section 4 below.

## 1. Verification log

Every cited work resolves to a real primary source. Six arXiv identifiers returned matching titles and author lists; four classical references were resolved by canonical DOI through Crossref; the IETF draft was resolved through the datatracker API. Four corrections are needed before any of this reaches a manuscript.

| Outline ref | Verified as | Correction needed |
|---|---|---|
| [3] LLM2Swarm arXiv:2410.11387 | Strobel, Dorigo, Fritz — v3 on arXiv, first posted 2024-10-15 | none |
| [4] SAC arXiv:2605.09076 | Lee, Yun, Panagou, Karimireddy — v3, 2026-05-09 | none |
| [5] CP-WBFT arXiv:2511.10400 | Zheng, Chen, Yin, Zhang et al. — v2, 2025-11-13 | outline says arXiv 2026; the preprint is **2025**-11-13, AAAI 2026 |
| [6] PACT arXiv:2606.05304 | Huang, Wu, Zhang — 2026-06-03 | none |
| [7] AACP draft-mackay-aacp | `draft-mackay-aacp-03`, "Agent Action Compression Protocol (AACP) Version 1.4", 2026-06-17, 16 pp. | **individual submission — `stream: null`, no intended status, expires 2026-12-19.** Cite as an unadopted Internet-Draft, not a standard |
| [8] HyLaT arXiv:2605.25421 | Mou, Wang, Li, He — 2026-05-25 | author list in the outline ends "Z. Wei"; arXiv lists Yulan He as 4th author — check the full list |
| [9] Web-of-Drones arXiv:2605.03788 | Iannoli, Gigli, Sciullo, Trotta, Di Felice — 2026-05-05 | none |
| [10] W-MSR | LeBlanc, Zhang, Koutsoukos, Sundaram, IEEE JSAC 2013, doi:10.1109/jsac.2013.130413 | author order is Zhang, **Koutsoukos, Sundaram** — the outline's order matches the paper; add the DOI |
| [11] Byzantine Generals | Lamport, Shostak, Pease, ACM TOPLAS 4(3):382–401, 1982, doi:10.1145/357172.357176 | none |
| [12] PBFT | Castro & Liskov, OSDI 1999 (no DOI); extended version ACM TOCS 20(4), 2002, doi:10.1145/571637.571640 | cite the TOCS version alongside OSDI if a DOI is required |
| [13] HotStuff | Yin, Malkhi, Reiter, Golan Gueta, Abraham, PODC 2019, doi:10.1145/3293611.3331591 | none |

Also found in the IETF datatracker while checking AACP: `draft-schrock-ep-quorum` ("Multi-Party Quorum Authorization for High-Risk Agent Actions (EP-QUORUM)", rev 03, 2026-07-19) — an M-of-N quorum-authorisation profile for high-risk agent actions. It binds *humans*, not peer agents, so it is not a competitor, but it is a citable precedent that quorum gating of high-impact agent actions is being standardised.

## 2. Search method

Ten arXiv queries across Byzantine LLM-MAS, per-robot LLM swarms, agent communication protocols, resilient consensus and LLM safety gating returned 60 unique records beyond the cited set. Each abstract was screened for relevance to the nine capability axes; 37 scored 3+ (adjacent or direct competitor) and 9 were promoted into the matrix after reading the full text. The remaining screened works are listed in Appendix A. Axis scores were assigned from full text (first 30 pages) for 17 of the 18 non-classical rows. The AACP row is an IETF Internet-Draft with no evaluation section and was scored by hand from the draft text; the four pre-LLM classical rows were likewise scored by hand against the *reasoning-layer* reading of each axis — W-MSR trims numeric scalars, so it scores 2 on aggregation but 1 on 'reasoning-layer Byzantine model', and PBFT/HotStuff *require* determinism rather than measuring it, so they score 0 on determinism evaluated.

## 3. The gap, stated numerically

Four axes define the contribution: one LLM per physical vehicle (A1), receiver-side resilient aggregation (A4), a deterministic override gate on actuation (A6), and a contested channel (A8). Perfect coverage is 8 points. The best score among all 22 works is **4/8** (LLM-Enhanced Resilient UAV Swarm (dual MASS), the authors' own TVT paper). No *system* paper scores above 0 on A1 and A4 simultaneously: the only row touching both is the authors' own survey, which is a taxonomy rather than a system. The per-vehicle-LLM literature and the Byzantine-aggregation literature do not intersect anywhere in the implemented work.

- **one LLM per physical vehicle (score 2):** LLM2Swarm, Talk Less, Fly Lighter (UAV Semantic Compression) — 2 of 22
- **receiver-side resilient aggregation (score 2):** WBFT Trusted MultiLLMN, DecentLLMs, CP-WBFT (Byzantine BFT for LLM-MAS), Self-Anchored Consensus (SAC), Resilient Consensus in Agentic AI, PBFT, W-MSR / r-robustness, HotStuff — 8 of 22
- **deterministic override gate (score 2):** LLM-Enhanced Resilient UAV Swarm (dual MASS), CommandSwarm — 2 of 22
- **A1 and A4 both ≥1:** 1 works (LLM-MAS ITS Comm-Attack Survey)
- **A4 ≥1 and A6 ≥1:** 1 works (LLM-MAS ITS Comm-Attack Survey)

The only rows reaching 4/8 are the student's own TVT paper (embodiment + gate + jamming, no distributed reasoning, no BFT) and the student's own survey (partial credit everywhere because it is a taxonomy, not a system). That is the honest form of the novelty claim: *the gap is between our two papers, not somewhere in the wider literature.*

## 4. What each prior work does NOT do

### Own prior work

**LLM-Enhanced Resilient UAV Swarm (dual MASS)** (2026, [1] own)  
*Does:* A* with LLM assistance reduces mission completion time by 10.5% and path length by 11.6% while maintaining high communication quality.  
*Does not:* The paper uses one centralized supervisory LLM, not one LLM per physical vehicle, so it has no receiver-side cross-agent Byzantine filtering, quorum voting, or collusion-resistant aggregation among multiple LLM instances. It also does not model an adversarial or hallucinating LLM, define a fault bound, or evaluate output determinism/repeatability, though it does gate LLM output through a severity-dependent override against a deterministic fallback controller.

**LLM-MAS ITS Comm-Attack Survey** (2026, [2] own)  
*Does:* LLM-mediated coordination adds a first-class Layer-3 semantic/reasoning attack surface (prompt tampering, shared-memory poisoning, CORBA) that is invisible to classical Layer-1/2 vehicular defenses and requires unified cross-layer treatment.  
*Does not:* This paper is a taxonomy/survey with no implemented system, so it does not build or test a one-LLM-per-vehicle architecture, specify a receiver-side Byzantine filter or quorum rule, or define a fault bound (e.g., f<n/3). It never runs a real or simulated control loop under jamming, so no actuation-gating mechanism, bandwidth cost, or output-determinism is measured — those are named as future work/roadmap items, not delivered.

### Byzantine-resilient LLM-MAS

**IBGP** (2024-10-21, **NEW**)  
*Does:* A multi-round randomized-threshold consensus protocol (the (k,t)-protocol) provably prevents mis-coordination in IBGP under any communication attack with t<50% malicious agents, and empirically gives near-100% zero-shot robustness in MARL tasks versus degraded baselines  
*Does not:* The paper's adversary model is purely message-content corruption on a complete, always-connected graph with 1-bit signals—it never models jamming, link degradation, partitions, or sparse/r-robust topologies, and has no vehicle dynamics or real control loop. It also has no deterministic actuation-gating layer separate from the vote itself, no per-agent LLM reasoning under attack (agents are RL policies, not LLMs, in experiments), and reports no token/bandwidth cost or output-determinism metrics.

**WBFT Trusted MultiLLMN** (2025-05-08, **NEW**)  
*Does:* WBFT improves consensus security and efficiency over classical/modern BFT consensus under wireless conditions, and Trusted MultiLLMN delivers higher-quality, more credible responses than single LLMs or unweighted MultiLLMNs  
*Does not:* The paper addresses text-response consensus among LLMs, not real-time control-loop actuation; it has no deterministic safety-envelope gating, no per-vehicle embodiment, and no active jamming/adversarial channel attack model beyond generic wireless transmission-failure probability. It also lacks bandwidth/token accounting and any determinism/repeatability evaluation of LLM outputs across trials.

**DecentLLMs** (2025-07-20, **NEW**)  
*Does:* Leaderless parallel-answer-generation-plus-Byzantine-robust-evaluator-scoring (via geometric median) achieves higher accuracy (71 vs 64/50 correct) and near-constant consensus latency (~221s) regardless of Byzantine leader count, versus leader-based quorum protocols whose latency grows linearly with consecutive Byzantine leaders.  
*Does not:* The paper contains no vehicles, control loop, dynamics, or actuation of any kind — 'workers' and 'evaluators' are undifferentiated LLM instances producing text answers to benchmark questions, not one-LLM-per-vehicle agents filtering sensor/peer messages. It models no communication-channel adversary (no jamming, link degradation, or partition — the network is assumed synchronous and reliable), no deterministic override/safety-gate on outputs before use, no bandwidth/token cost accounting, and no evaluation of output determinism/repeatability across runs, so none of the receiver-side Byzantine filtering, quorum-under-jamming, or gated-actuation requirements of a real control loop are addressed.

**CP-WBFT (Byzantine BFT for LLM-MAS)** (2025-11-13, [5] cited)  
*Does:* LLM-based agents exhibit intrinsic skepticism giving greater Byzantine reliability than traditional agents, and CP-WBFT's confidence-probe weighting (esp. hidden-level HCP) achieves up to 100% round-level accuracy even at 85.7% fault rate, exceeding classical f<n/3 bound  
*Does not:* The paper never models jamming, link degradation, or network partitions, has no receiver-side Byzantine-filtering quorum protocol with a proven robustness bound, and includes zero actuators or deterministic safety-envelope gating—faults are capability-gap wrong text answers, not adversarial senders on a real-time control loop. It also never measures per-round communication cost/bandwidth or output determinism across seeds, so it gives no basis for one-LLM-per-vehicle deployment under contested comms.

**Can AI Agents Agree?** (2026-03-01, **NEW**)  
*Does:* Valid consensus is unreliable even with zero Byzantine agents (41.6% overall) and degrades further with group size and with even a single Byzantine agent; failures are dominated by liveness loss (timeouts/stalled convergence) rather than value corruption, indicating LLM-agent groups are not yet dependable at reaching agreement even in a no-stake, restricted-threat setting.  
*Does not:* The paper studies abstract scalar-value agreement among purely textual LLM agents with no physical embodiment, control loop, or actuators, so it says nothing about per-vehicle LLM instantiation, sensor/actuation gating, or deterministic safety-envelope checks on outputs before they affect a real system. Its threat model and network are both benign in ways a real deployment would not be: Byzantine agents cannot equivocate, spoof identity, or drop messages, and the communication graph is a reliable complete graph with no jamming, link degradation, or partitions modeled, so it provides no evidence on receiver-side Byzantine filtering or quorum behavior under a contested/degraded channel.

**Insider Attacks in Multi-Agent LLM Consensus Systems** (2026-05-08, **NEW**)  
*Does:* A world-model-based RL insider attacker (learning a latent surrogate transition model of benign-agent behavior, then training a DQN attack policy on it) more effectively reduces benign consensus rate and prolongs disagreement than a direct malicious-prompt LLM baseline, and an attacker that infers behavioral attributes from one observed round performs nearly as well as one given true attributes.  
*Does not:* This paper is entirely about a single insider degrading consensus among cooperative LLM agents in an abstract 1-D text environment with no vehicle dynamics, no physical actuation, no communication-channel adversary (jamming/partition), and no receiver-side Byzantine-filtering, quorum, or deterministic safety-gate mechanism to detect or neutralize the attacker — it studies attack optimization, not defense, aggregation robustness, graph-topology tolerance bounds, bandwidth cost, or output determinism, and offers no mechanism to identify or exclude the malicious agent or gate its output before it affects the shared state.

**Self-Anchored Consensus (SAC)** (2026-05-09, [4] cited)  
*Does:* Receiver-side (not self-reported) confidence scoring combined with (F+1)-robust topology lets honest LLM agents filter Byzantine neighbors and achieve non-negative accuracy gain (BFTI), unlike CP-WBFT which collapses under falsified-confidence attacks  
*Does not:* SAC has no embodiment, dynamics, or actuation layer at all — it never addresses one-LLM-per-vehicle deployment, control-loop timing, or gating LLM output through a deterministic safety envelope before actuation. It assumes a static, always-reliable communication graph with no jamming, link degradation, partition, or bandwidth accounting, and never evaluates output determinism/repeatability across seeds.

**Resilient Consensus in Agentic AI** (2026-06-12, **NEW**)  
*Does:* Prompted LLM agents fail to reach agreement even within the theoretically guaranteed regime (N>=3B+1), and classical MSR-type resilient consensus filters (global for complete graphs, local for general graphs) recover near-100% consensus, with filter benefit depending on how much robustness the topology already provides  
*Does not:* The paper has no embodiment, dynamics, actuators, or physical vehicles — it is a pure text-based scalar-agreement game with no receiver-side control-loop actuation gating and no deterministic override/safety-envelope check on agent outputs. It models no communication-channel degradation, jamming, or partition (medium is assumed fail-safe/negligible-delay), and reports no bandwidth/token cost, so a real control loop under contested wireless links with per-vehicle LLMs and quorum-gated actuation is entirely outside its scope.

### LLM-per-robot / UAV swarm

**LLM2Swarm** (2024-10-15, [3] cited)  
*Does:* Integrating LLMs into robot swarms (indirectly for controller synthesis, directly for on-device robot reasoning/collaboration) can reduce development effort, enable natural-language robot-robot and human-swarm interaction, and detect diverse anomalies without prior knowledge of their nature  
*Does not:* The paper does not model an adversarial or arbitrary Byzantine reasoning layer, define any fault-tolerance bound, or specify a receiver-side filtering/quorum mechanism for combining peer LLM outputs—its 'peer diagnosis' showcase only detects a benign sensor fault via free-text discussion, not malicious/colluding agents. It also has no deterministic actuation gate checking LLM outputs before they reach actuators, no communication-graph or bandwidth analysis, and no modeling of jamming, link degradation, or partitioning (partitioning is flagged only as a future open problem), nor any test of output determinism across runs.

**Talk Less, Fly Lighter (UAV Semantic Compression)** (2025-08-16, **NEW**)  
*Does:* LLM-driven UAV swarms can autonomously compress task instructions to reduce communication load while largely preserving semantic fidelity and task success, though aggressive compression trades off success rate  
*Does not:* The paper assumes fully cooperative, non-adversarial LLM agents with no receiver-side verification, voting, or Byzantine filtering of peer messages, and no explicit fault-tolerance bound. It also never models jamming/link degradation, never gates LLM-generated instructions through a deterministic safety envelope before actuation, and never tests output determinism/repeatability across seeds.

**Web-of-Drones MCP Agent** (2026-05-05, [9] cited)  
*Does:* General-purpose LLMs struggle at reliable swarm execution without explicit grounding; agent-enhanced execution (MCP/WoT abstraction, planning tools, runtime guardrails) substantially improves robustness; token use doesn't predict success.  
*Does not:* The paper uses one centralized LLM per mission, not one LLM per vehicle, so it never addresses cross-vehicle Byzantine reasoning faults, receiver-side filtering, or quorum consensus among distributed agents. It also does not model jamming, link degradation, or network partitions, and its only safety gating is input-schema/state validation at the device Servient, not a deterministic override envelope on control-loop actuation.

**CommandSwarm** (2026-05-08, **NEW**)  
*Does:* Compact, quantized, LoRA-adapted open LLMs embedded in a parser-validated, safety-filtered pipeline can reliably convert multilingual commands into executable swarm BTs, with LoRA lifting zero-shot syntactic validity from 0% to 72%.  
*Does not:* CommandSwarm has no notion of multiple robots each running an LLM, no receiver-side voting/Byzantine-fault filtering across agent outputs, and no fault-tolerance bound or robust-graph analysis for inter-robot communication. It also does not model a contested/jammed channel, does not evaluate output determinism across repeated runs, and its single safety gate is a deterministic XML/whitelist parser on one generated tree, not a quorum-based override mechanism across independently reasoning vehicles.

**LAUS (LLM-Centric Agentic AI for UAV Swarms)** (2026-07-05, **NEW**)  
*Does:* Manipulating only structured perception-layer observations (queue, energy, channel) before LLM input, with no model-parameter access, deterministically redirects all UAVs to attacker-chosen sensors, causing 14.2x packet-loss cost degradation across all 30 runs  
*Does not:* The paper uses a single centralized LLM scheduler, not one LLM per vehicle, and has no receiver-side cross-checking, quorum, or Byzantine filtering among peer agents — the demonstrated attack succeeds precisely because there is no redundant agent to cross-verify the tampered observation. It also does not model a contested/jamming channel or measure bandwidth, and its Safety Validator/kill-switch gate is only architecturally proposed, not exercised or shown effective against the PMA in the reported experiment.

### Agent messaging protocol

**AACP (IETF I-D)** (2026, [7] cited)  
*Does:* A coordination content layer beneath MCP/A2A: what agents say, typed and replayable.  
*Does not:* Standardises the packet, not the decision — nothing in AACP tells a receiver which of several conflicting typed packets to trust, so it cannot substitute for Byzantine filtering. It is an unadopted individual Internet-Draft with no swarm, no radio model, and no vehicle in the loop.

**HyLaT** (2026-05-25, [8] cited)  
*Does:* A hybrid dual-channel protocol — routing verbose reasoning through compact latent vectors and concise final answers/commitments through natural text — resolves the efficiency/interpretability/versatility trilemma of single-channel multi-agent communication, cutting token/time cost roughly an order of magnitude versus text baselines while matching task accuracy, and remains robust to moderate Gaussian noise on latent vectors and interoperable with pure-text agents via the retained text channel.  
*Does not:* HyLaT assumes all agents run a shared, cooperative, non-adversarial backbone model and never models faulty, hallucinating-adversarial, or colluding peers, nor any receiver-side filtering, quorum, or fault-tolerance bound on peer messages — it has no Byzantine model or robust-aggregation mechanism at all. It also has no actuators, no deterministic safety-envelope gating of outputs, no communication-graph robustness analysis, and no jamming/link-degradation/partition modeling — its only 'channel stress test' is Gaussian noise added to latent vectors offline, not a live contested network or a real control loop, so none of receiver-side Byzantine filtering, quorum voting, or deterministic override gating for vehicle control under jamming is addressed.

**PACT** (2026-06-03, [6] cited)  
*Does:* Restricting inter-agent messages to a compact action-state record (action taken, grounding state, resulting artifact) instead of free-form full-transcript communication substantially reduces token cost while preserving or improving task performance across MAS topologies and two production coding harnesses.  
*Does not:* PACT addresses only content compression of messages between cooperative, non-adversarial LLM agents in text-based QA/coding tasks — it has no notion of Byzantine or hallucinating senders, no receiver-side cross-checking, voting, or quorum mechanism, and no fault-tolerance bound (f<n/3 or similar) is defined or evaluated. It has no physical embodiment, control loop, actuation gate, or deterministic safety envelope, and does not model or test any contested/jammed/degraded communication channel, so it provides no evidence about message robustness under adversarial or unreliable network conditions for a one-LLM-per-vehicle real-time control system.

### Classical resilient consensus / BFT

**Byzantine Generals** (1982, [11] cited)  
*Does:* Interactive consistency is impossible with three generals and one traitor; n>3m is necessary and sufficient for oral messages.  
*Does not:* Assumes exact agreement on a discrete value with reliable delivery — neither holds for LLM plans on a jammed mesh, where honest agents legitimately disagree in wording and messages are dropped. Provides no approximate-agreement notion and no way to judge whether a proposal is safe rather than merely agreed.

**PBFT** (1999, [12] cited)  
*Does:* Byzantine agreement is achievable with practical overhead in asynchronous systems using 3f+1 replicas.  
*Does not:* Requires every replica to compute the identical output from identical input, which a sampled LLM violates by construction, and needs O(n^2) all-to-all messaging that a sparse jammed mesh cannot sustain. It has no concept of a plan being unsafe if all replicas agree on it.

**W-MSR / r-robustness** (2013, [10] cited)  
*Does:* Resilient asymptotic consensus is guaranteed iff the graph satisfies the stated robustness condition; connectivity/degree alone are insufficient.  
*Does not:* Operates on scalar numeric states, so it cannot filter semantically plausible natural-language plans — there is no metric space in which to trim an LLM proposal. It carries no notion of schema validity, safety envelope, actuation authority, or communication cost.

**HotStuff** (2019, [13] cited)  
*Does:* Linear communication complexity and optimistic responsiveness with a rotating leader.  
*Does not:* Re-introduces exactly the leader dependency the new architecture is trying to remove, and its liveness needs partial synchrony that a partitioned jammed mesh violates. Agreement is on opaque bytes, so nothing checks the semantic admissibility of what is being committed.

## 5. Consequences for the research plan

**5.1 Three works change the novelty framing and must be cited.** The gap search surfaced Byzantine-robust LLM coordination work the outline does not cite, all of it pre-dating or contemporaneous with SAC:

- **DecentLLMs** (2025-07-20) — Leaderless parallel-answer-generation-plus-Byzantine-robust-evaluator-scoring (via geometric median) achieves higher accuracy (71 vs 64/50 correct) and near-constant consensus latency (~221s) regardless of Byzantine leader count, versus leader-based quorum protocols whose latency grows linearly with consecutive Byzantine leaders. Threat to the claim: it already does reasoning-layer Byzantine model, receiver-side resilient aggregation. Distinguish on: one LLM per physical vehicle, embodied closed-loop control, deterministic override gate, contested channel modelled.
- **WBFT Trusted MultiLLMN** (2025-05-08) — WBFT improves consensus security and efficiency over classical/modern BFT consensus under wireless conditions, and Trusted MultiLLMN delivers higher-quality, more credible responses than single LLMs or unweighted MultiLLMNs Threat to the claim: it already does reasoning-layer Byzantine model, receiver-side resilient aggregation. Distinguish on: one LLM per physical vehicle, embodied closed-loop control, deterministic override gate.
- **Resilient Consensus in Agentic AI** (2026-06-12) — Prompted LLM agents fail to reach agreement even within the theoretically guaranteed regime (N>=3B+1), and classical MSR-type resilient consensus filters (global for complete graphs, local for general graphs) recover near-100% consensus, with filter benefit depending on how much robustness the topology already provides Threat to the claim: it already does reasoning-layer Byzantine model, receiver-side resilient aggregation. Distinguish on: one LLM per physical vehicle, embodied closed-loop control, deterministic override gate, contested channel modelled.
- **IBGP** (2024-10-21) — A multi-round randomized-threshold consensus protocol (the (k,t)-protocol) provably prevents mis-coordination in IBGP under any communication attack with t<50% malicious agents, and empirically gives near-100% zero-shot robustness in MARL tasks versus degraded baselines Threat to the claim: it already does . Distinguish on: one LLM per physical vehicle, embodied closed-loop control, deterministic override gate, contested channel modelled.
- **Insider Attacks in Multi-Agent LLM Consensus Systems** (2026-05-08) — A world-model-based RL insider attacker (learning a latent surrogate transition model of benign-agent behavior, then training a DQN attack policy on it) more effectively reduces benign consensus rate and prolongs disagreement than a direct malicious-prompt LLM baseline, and an attacker that infers behavioral attributes from one observed round performs nearly as well as one given true attributes. Threat to the claim: it already does reasoning-layer Byzantine model. Distinguish on: one LLM per physical vehicle, embodied closed-loop control, deterministic override gate, contested channel modelled.

**5.2 The strongest sentence available to you** is not "first Byzantine-resilient multi-LLM system" — that is already taken several times over. It is: *first system in which per-vehicle LLM reasoning is admitted to a physical control loop only through a receiver-side Byzantine filter and a deterministic safety envelope, evaluated under a modelled jamming channel.* Every clause of that sentence maps to an axis where the matrix shows a 0.

**5.3 Two axes are underdefended in the current plan.** A9 (determinism evaluated) is scored 2 for only one work (Web-of-Drones), and 0 for every Byzantine-LLM paper — the override-consistency metric in Section 6 of the outline is therefore a genuine contribution and should be promoted from a metric to a claim. Conversely A5 (graph robustness) is already owned by W-MSR and SAC; asserting an (F+1)-robustness result would be re-derivation unless the contribution is specifically about robustness of a *sparse jammed* graph whose edges drop out at runtime — which no row in the matrix addresses.

**5.4 Messaging design is a three-way choice, not a default.** PACT and HyLaT both score 2 on compact messaging and 0 on everything else; AACP is an unadopted draft; and *Talk Less, Fly Lighter* does UAV-specific semantic compression that the outline does not cite but that is the closest match to the SQ3 setting. Budget the bandwidth comparison against at least PACT and one UAV-native compressor.

## 6. Caveats

- Axis scores are a reading of each paper's own text by an LLM extraction pass over the first 30 pages, reviewed row by row; they are a triage instrument, not a substitute for reading the six direct competitors in full before writing related work.
- The gap search covers arXiv only. IEEE Xplore, ACM DL and Scopus are not reachable from this environment, so venue-only publications (ICRA/IROS/GLOBECOM proceedings without preprints) are not represented.
- Two 2026 works in the matrix are preprints with no venue; verify publication status before citing as peer-reviewed.

## Appendix A — screened, not promoted to the matrix

| Score | Bucket | Title | One line |
|---|---|---|---|
| 4 | agent_comm_protocol | [Capability Advertisement as a Market for Lemons: A Trust Layer for Heterogeneo](https://arxiv.org/abs/2606.03034) | Market-for-lemons model and Trust Layer for reliable capability advertising in heterogeneous LLM agent networks. |
| 4 | per_agent_llm_robot_swarm | [Online automatic code generation for robot swarms: LLMs and self-organizing hi](https://arxiv.org/abs/2510.04774) | Self-organizing swarm with online LLM code generation; global state estimation enables automated behavior synthesis. |
| 4 | per_agent_llm_robot_swarm | [CoordField: Coordination Field for Agentic UAV Task Allocation In Low-altitude](https://arxiv.org/abs/2505.00091) | LLM-based coordination field for heterogeneous UAV swarm task allocation in urban environments. |
| 4 | byzantine_llm_mas | [The Deliberative Illusion: Diagnosing Factual Attrition and Stance Homogenizat](https://arxiv.org/abs/2606.03032) | Measures factual/stance degradation in multi-LLM deliberation; adversarial robustness via information loss. |
| 4 | resilient_consensus_classical | [When Is Emergent Consensus Real? A Measured Coupling Gain and a Validity Diagn](https://arxiv.org/abs/2606.22203) | Measures coupling gain in LLM agent societies; distinguishes genuine consensus from artifacts via classical dynamics theory. |
| 4 | agent_comm_protocol | [Exploring the Topology and Memory of Consensus: How LLM Agents Agree, Fragment](https://arxiv.org/abs/2606.04197) | Multi-agent LLM consensus under topology and memory constraints; fragmentation and coordination dynamics. |
| 4 | byzantine_llm_mas | [Beyond Consensus: Trace-Level Synthesis in Mixture of Agents](https://arxiv.org/abs/2605.29116) | Trace-level LLM aggregation via perturbation diversity; recovers correctness beyond majority voting consensus. |
| 4 | resilient_consensus_classical | [HadAgent: Harness-Aware Decentralized Agentic AI Serving with Proof-of-Inferen](https://arxiv.org/abs/2604.18614) | Decentralized LLM inference via Proof-of-Inference blockchain with Byzantine-resilient node trust management. |
| 4 | resilient_consensus_classical | [TrustTrade: Human-Inspired Selective Consensus Reduces Decision Uncertainty in](https://arxiv.org/abs/2603.22567) | Multi-agent LLM consensus via selective weighting and semantic agreement; financial domain analog of Byzantine-robust aggregation. |
| 4 | resilient_consensus_classical | [MechPert: Mechanistic Consensus as an Inductive Bias for Unseen Perturbation P](https://arxiv.org/abs/2602.13791) | Multi-agent LLM consensus mechanism for robust prediction under uncertainty; agent aggregation filters spurious associations. |
| 3 | per_agent_llm_robot_swarm | [D-VLC: Decentralized Vision-Language Collaboration for Heterogeneous Embodied ](https://arxiv.org/abs/2607.29009) | Decentralized VLM reasoning per robot, asynchronous peer coordination, no Byzantine robustness or consensus aggregation. |
| 3 | per_agent_llm_robot_swarm | [Melding LLM and temporal logic for reliable human-swarm collaboration in compl](https://arxiv.org/abs/2605.07877) | Neuro-symbolic LLM planning with temporal logic constraints for human-swarm task coordination and safety gating. |
| 3 | per_agent_llm_robot_swarm | [Vision-Language Navigation for Aerial Robots: Towards the Era of Large Languag](https://arxiv.org/abs/2604.07705) | UAV natural-language navigation via LLMs/VLMs; covers multi-agent swarm control and hierarchical architectures. |
| 3 | per_agent_llm_robot_swarm | [SkySim: A ROS2-based Simulation Environment for Natural Language Control of Dr](https://arxiv.org/abs/2602.01226) | ROS2/Gazebo framework: single LLM (Gemini) generates waypoints for drone swarms, APF safety filter enforces collision/kinematic constraints. |
| 3 | uav_llm_other | [Agentic AI Meets Edge Computing in Autonomous UAV Swarms](https://arxiv.org/abs/2601.14437) | LLM-based autonomous reasoning in UAV swarms via edge computing; lacks Byzantine resilience and peer NL coordination focus. |
| 3 | per_agent_llm_robot_swarm | [PrivLLMSwarm: Privacy-Preserving LLM-Driven UAV Swarms for Secure IoT Surveill](https://arxiv.org/abs/2512.06747) | MPC-encrypted LLM inference for UAV swarm coordination; privacy via cryptography, not Byzantine resilience. |
| 3 | per_agent_llm_robot_swarm | [An LLM-based Framework for Human-Swarm Teaming Cognition in Disaster Search an](https://arxiv.org/abs/2511.04042) | LLM cognitive engine for human-UAV swarm teaming, intention-to-action task decomposition and mission planning. |
| 3 | per_agent_llm_robot_swarm | [RALLY: Role-Adaptive LLM-Driven Yoked Navigation for Agentic UAV Swarms](https://arxiv.org/abs/2507.01378) | LLM-driven UAV swarm with role-adaptive semantic coordination and MARL policy integration. |
| 3 | byzantine_llm_mas | [OEP: Poisoning Self-Evolving LLM Agents via Locally Correct but Non-Transferab](https://arxiv.org/abs/2605.18930) | Adversarial poisoning of LLM agent memory via clean but non-transferable experiences; Byzantine reasoning-layer attack. |
| 3 | llm_gating_safety_control | [From Words to Safety: Language-Conditioned Safety Filtering for Robot Navigati](https://arxiv.org/abs/2511.05889) | LLM translates natural language to safety specs; MPC filter enforces constraints on robot navigation. |
| 3 | llm_gating_safety_control | [Control Barrier Function for Aligning Large Language Models](https://arxiv.org/abs/2511.03121) | Control barrier functions as deterministic safety filter on LLM token output. |
| 3 | llm_gating_safety_control | [CBF-LLM: Safe Control for LLM Alignment](https://arxiv.org/abs/2408.15625) | Control barrier functions filter LLM token outputs to enforce safety constraints. |
| 3 | resilient_consensus_classical | [Microscopic dynamics of consensus formation in multi-agent LLM Naming Games](https://arxiv.org/abs/2608.02178) | Analyzes consensus formation in decentralized LLM-agent populations via stochastic naming games and mean-field theory. |
| 3 | agent_comm_protocol | [Semantic Consensus: Process-Aware Conflict Detection and Resolution for Enterp](https://arxiv.org/abs/2604.16339) | Multi-agent LLM coordination via semantic intent graphs and conflict detection, enterprise workflow focus. |
| 3 | resilient_consensus_classical | [Opinion Consensus Formation Among Networked Large Language Models](https://arxiv.org/abs/2601.21540) | DeGroot consensus dynamics in multi-LLM networks; opinion convergence vs. classical graph theory predictions. |
| 3 | resilient_consensus_classical | [Free-MAD: Consensus-Free Multi-Agent Debate](https://arxiv.org/abs/2509.11035) | Multi-agent LLM debate with robustness to conformity bias; single-round consensus-free aggregation. |
| 3 | agent_comm_protocol | [Agentic LLMs in the Supply Chain: Towards Autonomous Multi-Agent Consensus-See](https://arxiv.org/abs/2411.10184) | LLM agents coordinate supply chain consensus via natural-language negotiation; no Byzantine resilience or swarm robotics. |
| 3 | resilient_consensus_classical | [ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diver](https://arxiv.org/abs/2309.13007) | Multi-LLM consensus via iterative discussion and confidence-weighted voting for reasoning improvement. |
