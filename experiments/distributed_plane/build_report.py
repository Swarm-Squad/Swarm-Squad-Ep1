import os as _os
_BASE = _os.path.dirname(_os.path.abspath(__file__))
_RES = _os.path.join(_BASE, "results") if _os.path.isdir(_os.path.join(_BASE, "results")) else _BASE
_DOCS = _os.path.join(_BASE, "docs") if _os.path.isdir(_os.path.join(_BASE, "docs")) else _BASE
_SCR = _os.path.join(_RES, "screening") if _os.path.isdir(_os.path.join(_RES, "screening")) else "handoff"
import json, pandas as pd, textwrap
df = pd.read_csv(_os.path.join(_RES, "byzantine_llm_swarm_matrix.csv"))
sc = json.load(open(_os.path.join(_SCR, "screen.json"))); cl = json.load(open(_os.path.join(_SCR, "classics.json")))
ietf = json.load(open(_os.path.join(_SCR, "ietf_docs.json"))); raw = json.load(open(_os.path.join(_SCR, "arxiv_raw.json")))
AXN = {"A1":"one LLM per physical vehicle","A2":"embodied closed-loop control","A3":"reasoning-layer Byzantine model",
       "A4":"receiver-side resilient aggregation","A5":"graph robustness condition","A6":"deterministic override gate",
       "A7":"compact schema + bandwidth accounting","A8":"contested channel modelled","A9":"determinism evaluated"}
cited_meta = {r["id"]: r for r in raw["cited"]}

L = []
w = L.append
w("# Stage 1 — Literature comparison matrix\n")
w("**Byzantine-resilient distributed LLM-per-vehicle swarm control**  \n")
w("Scope: the 13 references of `new_research.md`, each verified against its primary source, plus 9 works added by a systematic arXiv gap search. "
  "Stage 1's completion criterion is a table stating explicitly what each prior work does *not* do; that is the `does_not_do` column of "
  "`byzantine_llm_swarm_matrix.csv` and Section 4 below.\n")

w("## 1. Verification log\n")
w("Every cited work resolves to a real primary source. Six arXiv identifiers returned matching titles and author lists; "
  "four classical references were resolved by canonical DOI through Crossref; the IETF draft was resolved through the datatracker API. "
  "Four corrections are needed before any of this reaches a manuscript.\n")
w("| Outline ref | Verified as | Correction needed |")
w("|---|---|---|")
rowsv = [
 ("[3] LLM2Swarm arXiv:2410.11387", "Strobel, Dorigo, Fritz — v3 on arXiv, first posted 2024-10-15", "none"),
 ("[4] SAC arXiv:2605.09076", "Lee, Yun, Panagou, Karimireddy — v3, 2026-05-09", "none"),
 ("[5] CP-WBFT arXiv:2511.10400", "Zheng, Chen, Yin, Zhang et al. — v2, 2025-11-13", "outline says arXiv 2026; the preprint is **2025**-11-13, AAAI 2026"),
 ("[6] PACT arXiv:2606.05304", "Huang, Wu, Zhang — 2026-06-03", "none"),
 ("[7] AACP draft-mackay-aacp", f"`draft-mackay-aacp-{ietf['aacp']['rev']}`, \"{ietf['aacp']['title']}\", {ietf['aacp']['time'][:10]}, {ietf['aacp']['pages']} pp.", f"**individual submission — `stream: null`, no intended status, expires {ietf['aacp']['expires'][:10]}.** Cite as an unadopted Internet-Draft, not a standard"),
 ("[8] HyLaT arXiv:2605.25421", "Mou, Wang, Li, He — 2026-05-25", "author list in the outline ends \"Z. Wei\"; arXiv lists Yulan He as 4th author — check the full list"),
 ("[9] Web-of-Drones arXiv:2605.03788", "Iannoli, Gigli, Sciullo, Trotta, Di Felice — 2026-05-05", "none"),
 ("[10] W-MSR", f"LeBlanc, Zhang, Koutsoukos, Sundaram, IEEE JSAC 2013, doi:{cl['wmsr']['doi']}", "author order is Zhang, **Koutsoukos, Sundaram** — the outline's order matches the paper; add the DOI"),
 ("[11] Byzantine Generals", f"Lamport, Shostak, Pease, ACM TOPLAS 4(3):382–401, 1982, doi:{cl['lamport']['doi']}", "none"),
 ("[12] PBFT", "Castro & Liskov, OSDI 1999 (no DOI); extended version ACM TOCS 20(4), 2002, doi:10.1145/571637.571640", "cite the TOCS version alongside OSDI if a DOI is required"),
 ("[13] HotStuff", f"Yin, Malkhi, Reiter, Golan Gueta, Abraham, PODC 2019, doi:{cl['hotstuff']['doi']}", "none"),
]
for a,b,c in rowsv: w(f"| {a} | {b} | {c} |")
w("")
w("Also found in the IETF datatracker while checking AACP: `draft-schrock-ep-quorum` "
  f"(\"{ietf['quorum_draft']['title']}\", rev {ietf['quorum_draft']['rev']}, {ietf['quorum_draft']['time'][:10]}) — "
  "an M-of-N quorum-authorisation profile for high-risk agent actions. It binds *humans*, not peer agents, so it is not a competitor, "
  "but it is a citable precedent that quorum gating of high-impact agent actions is being standardised.\n")

w("## 2. Search method\n")
w(f"Ten arXiv queries across Byzantine LLM-MAS, per-robot LLM swarms, agent communication protocols, resilient consensus and "
  f"LLM safety gating returned {len({r['id'].split('v')[0] for v in raw['harvest'].values() for r in v})} unique records beyond the cited set. "
  f"Each abstract was screened for relevance to the nine capability axes; {sum(1 for v in sc['screen'].values() if v.get('score',0)>=3)} scored 3+ "
  "(adjacent or direct competitor) and 9 were promoted into the matrix after reading the full text. "
  "The remaining screened works are listed in Appendix A. Axis scores were assigned from full text (first 30 pages) for 17 of the 18 non-classical rows. "
  "The AACP row is an IETF Internet-Draft with no evaluation section and was scored by hand from the draft text; "
  "the four pre-LLM classical rows were likewise scored by hand against the *reasoning-layer* reading of each axis — "
  "W-MSR trims numeric scalars, so it scores 2 on aggregation but 1 on 'reasoning-layer Byzantine model', "
  "and PBFT/HotStuff *require* determinism rather than measuring it, so they score 0 on determinism evaluated.\n")

w("## 3. The gap, stated numerically\n")
core = ["A1","A4","A6","A8"]
df["core"] = df[core].sum(axis=1)
best = df.sort_values("core", ascending=False).iloc[0]
w(f"Four axes define the contribution: one LLM per physical vehicle (A1), receiver-side resilient aggregation (A4), "
  f"a deterministic override gate on actuation (A6), and a contested channel (A8). Perfect coverage is 8 points. "
  f"The best score among all {len(df)} works is **{int(best['core'])}/8** ({best['short_name']}, the authors' own TVT paper). "
  "No *system* paper scores above 0 on A1 and A4 simultaneously: the only row touching both is the authors' own survey, "
  "which is a taxonomy rather than a system. The per-vehicle-LLM literature and the Byzantine-aggregation literature "
  "do not intersect anywhere in the implemented work.\n")
for a in ["A1","A4","A6"]:
    hits = df[df[a] == 2]["short_name"].tolist()
    w(f"- **{AXN[a]} (score 2):** {', '.join(hits) if hits else 'no work'} — {len(hits)} of {len(df)}")
w(f"- **A1 and A4 both ≥1:** {len(df[(df.A1>=1)&(df.A4>=1)])} works ({', '.join(df[(df.A1>=1)&(df.A4>=1)]['short_name'].tolist()) or 'none'})")
w(f"- **A4 ≥1 and A6 ≥1:** {len(df[(df.A4>=1)&(df.A6>=1)])} works ({', '.join(df[(df.A4>=1)&(df.A6>=1)]['short_name'].tolist()) or 'none'})\n")
w("The only rows reaching 4/8 are the student's own TVT paper (embodiment + gate + jamming, no distributed reasoning, no BFT) and "
  "the student's own survey (partial credit everywhere because it is a taxonomy, not a system). That is the honest form of the novelty claim: "
  "*the gap is between our two papers, not somewhere in the wider literature.*\n")

w("## 4. What each prior work does NOT do\n")
for g in ["Own prior work","Byzantine-resilient LLM-MAS","LLM-per-robot / UAV swarm","Agent messaging protocol","Classical resilient consensus / BFT"]:
    sub = df[df.group == g]
    if not len(sub): continue
    w(f"### {g}\n")
    for r in sub.itertuples():
        tag = "**NEW**" if "NEW" in r.status else r.status
        w(f"**{r.short_name}** ({r.year}, {tag})  ")
        w(f"*Does:* {r.key_claim}  ")
        w(f"*Does not:* {r.does_not_do}\n")

w("## 5. Consequences for the research plan\n")
w("**5.1 Three works change the novelty framing and must be cited.** The gap search surfaced Byzantine-robust LLM coordination work "
  "the outline does not cite, all of it pre-dating or contemporaneous with SAC:\n")
for k in ["DecentLLMs","WBFT Trusted MultiLLMN","Resilient Consensus in Agentic AI","IBGP","Insider Attacks in Multi-Agent LLM Consensus Systems"]:
    r = df[df.short_name.str.startswith(k[:22])]
    if len(r):
        r = r.iloc[0]
        w(f"- **{r.short_name}** ({r.year}) — {r.key_claim} Threat to the claim: it already does {', '.join(AXN[a] for a in ['A3','A4'] if r[a]==2)}. "
          f"Distinguish on: {', '.join(AXN[a] for a in ['A1','A2','A6','A8'] if r[a]==0)}.")
w("")
w("**5.2 The strongest sentence available to you** is not \"first Byzantine-resilient multi-LLM system\" — that is already taken several times over. "
  "It is: *first system in which per-vehicle LLM reasoning is admitted to a physical control loop only through a receiver-side Byzantine filter and a "
  "deterministic safety envelope, evaluated under a modelled jamming channel.* Every clause of that sentence maps to an axis where the matrix shows a 0.\n")
w("**5.3 Two axes are underdefended in the current plan.** A9 (determinism evaluated) is scored 2 for only one work "
  "(Web-of-Drones), and 0 for every Byzantine-LLM paper — the override-consistency metric in Section 6 of the outline is therefore a genuine "
  "contribution and should be promoted from a metric to a claim. Conversely A5 (graph robustness) is already owned by W-MSR and SAC; "
  "asserting an (F+1)-robustness result would be re-derivation unless the contribution is specifically about robustness of a *sparse jammed* graph "
  "whose edges drop out at runtime — which no row in the matrix addresses.\n")
w("**5.4 Messaging design is a three-way choice, not a default.** PACT and HyLaT both score 2 on compact messaging and 0 on everything else; "
  "AACP is an unadopted draft; and *Talk Less, Fly Lighter* does UAV-specific semantic compression that the outline does not cite but that is the "
  "closest match to the SQ3 setting. Budget the bandwidth comparison against at least PACT and one UAV-native compressor.\n")

w("## 6. Caveats\n")
w("- Axis scores are a reading of each paper's own text by an LLM extraction pass over the first 30 pages, reviewed row by row; they are a triage "
  "instrument, not a substitute for reading the six direct competitors in full before writing related work.\n"
  "- The gap search covers arXiv only. IEEE Xplore, ACM DL and Scopus are not reachable from this environment, so venue-only publications "
  "(ICRA/IROS/GLOBECOM proceedings without preprints) are not represented.\n"
  "- Two 2026 works in the matrix are preprints with no venue; verify publication status before citing as peer-reviewed.\n")

w("## Appendix A — screened, not promoted to the matrix\n")
w("| Score | Bucket | Title | One line |")
w("|---|---|---|---|")
inmat = set()
for t in df.title.tolist(): inmat.add(t[:40])
for k, v in sorted(sc["screen"].items(), key=lambda x: -x[1].get("score", 0)):
    if v.get("score", 0) < 3: continue
    c = sc["cands"][k]
    if c["title"][:40] in inmat: continue
    w(f"| {v['score']} | {v['bucket']} | [{c['title'][:78]}](https://arxiv.org/abs/{k}) | {v['one_line']} |")
w("")
open(_os.path.join(_DOCS, "literature_matrix.md"),"w").write("\n".join(L))
print("report chars:", len("\n".join(L)), "| appendix rows:", sum(1 for k,v in sc['screen'].items() if v.get('score',0)>=3 and sc['cands'][k]['title'][:40] not in inmat))
