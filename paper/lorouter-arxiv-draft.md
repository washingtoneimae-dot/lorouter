# Draft — ArXiv submission skeleton (lorouter)

Status: draft for review, assembled 2026-08-06 from REVIEW.md + FINDINGS.md
+ TECHNICAL.md (lorouter branch, github.com/washingtoneimae-dot/lorouter). Every number below
is verified by a runnable script in the repo (F-numbers refer to
FINDINGS.md). Nothing here is claimed beyond the evidence.

---

## Proposed title

**Calibrated Competence Profiles for Zero-Parameter LoRA Adapter Selection**

(Alternative: "Routing LoRA Adapter Pools by Calibrated Competence
Profiles". Leading with the representation, not the router — per review.)

## Abstract (draft)

Multi-LoRA serving systems (Punica, S-LoRA, LoRAX, vLLM, SGLang) batch
and load adapters that requests *name*; none select an adapter from query
content. We present a selection policy built on **calibrated competence
profiles**: each adapter is represented by a competence vector measured
by calibration against a boundary-dense domain corpus — a measurement,
not a learned representation — and requests are routed by cosine
similarity between a shared query profile and adapter profiles. The
selection policy has zero learned parameters and every decision reduces
to a score.

Across 5 seeds and two model sizes (135M/360M), profile routing matches a
properly-trained learned router (96.4%) and, with semantic features,
reaches 98.2% — beating raw embedding-centroid routing (96.4%), the
obvious baseline whose disambiguation limits a 2026 result independently
reports. Swapping an adapter changes routing on **0.00%** of other-domain
inputs — structural, not tuned. Scaling to 1024-adapter pools is flat at
zero profile noise, and the compounding law governing false capture does
**not** follow the independent-gate Bonferroni model: correlated adapter
variants produce far lower false capture than the worst-case bound
predicts (22.6% measured vs 100% naive at 128 variants). Selection
latency is sub-millisecond at 10,000 adapters.

Limits are stated as plainly as claims: mechanism-grade evidence (small
models, synthetic QA corpus), the 1000+-adapter scaling test uses
stand-in variants grounded in real profile shapes, and no serving-stack
integration is reported.

## Thesis (one sentence, per review)

Raw embedding similarity is the obvious approach to adapter selection;
calibrated competence profiles against a boundary-dense calibration
foundation outperform it (98.2% vs 96.4% with semantic features) and
provide isolation properties (0.00% swap collateral, structural) that
selection-by-retrieval does not characterize.

---

## Section structure

### 1. Introduction
- The serving-layer gap: multi-LoRA serving is the economically viable
  multi-tenant pattern (2026 practice), but every shipped stack
  (vLLM/SGLang/LoRAX/Ray) requires the request to name its adapter;
  "wrong adapter served" is a documented production failure mode.
- The selection problem: which adapter does this query need?
- Position: a representation, not a router. Two headline claims:
  competitive accuracy (98.2% semantic, ties learned router) and unique
  isolation (0.00% swap collateral, structural).
- Contributions list (4 items — see Results).

### 2. Background and Related Work
- Serving layers: Punica (SGMV), S-LoRA (unified paging), dLoRA
  (scheduling with switching costs), LoRAX, vLLM — none select from
  content (verified sources).
- Learned routers: EdgeLoRA's trained adapter router; LoRAMoE /
  DLP-LoRA gating (AAAI 2026); the MoLE line.
- Task-representation routing: LORAUTER (Jan 2026) — learned task
  embeddings from validation sets; LoraRetriever-style retrieval.
- The weak baseline documented by others: a 2026 Springer result reports
  embedding-cosine adapter selection "may not sufficiently disambiguate."
- The anti-learned-router wave: routing-free MoE (Apr 2026).
- Positioning: our point in the design space — calibrated measurement,
  not learned embedding; isolation guarantees none of the above
  characterize.

### 3. Method: Calibrated Competence Profiles
- Adapter profile: competence vector over the domain space, measured by
  calibration. Stand-in setting: mean claim-strength of a specialist
  classifier. Real-LoRA setting: inverse answer-conditional calibration
  loss per domain (question + that domain's answer behavior).
- **Load-bearing design constraint (F9)**: profiles must measure answer
  behavior, not question text — question-only calibration collapses
  routing 96.4% → 73.2%. (Include as a method subsection; it doubles as
  evidence of measurement discipline.)
- Query profile: shared profiler into the same domain space (lexical and
  semantic variants described).
- Selection: cosine top-k; zero learned parameters in the selection
  policy — defined precisely: the profiler is a trained classifier whose
  parameters are fixed at calibration time and independent of pool size;
  the *selection* itself is parameter-free. (Precision matters here —
  "zero learned router parameters" is about the selection layer.)
- Isolation argument (structural): a profile insertion cannot change
  other adapters' scores.

### 4. Experimental Setup (consolidated methods)
- Corpora: moat bricks 1–3 (181 → 408 → 664) and the user-built v3
  (3,010 examples, six domains, 220 hand-written boundaries, Kenyan
  context; splits 1673/807/530). Boundary-dense calibration design;
  boundary-hardness validation (TF-IDF classifier 94.0% clean / 0.0%
  boundary — boundary examples are genuinely dual-domain).
- Stand-in machinery: TF-IDF+SVD and bge-small-en-v1.5 feature paths;
  specialist classifiers as adapter stand-ins; 5 seeds; protocols per
  script (benchmark.py, semantic_embeddings.py).
- Real-LoRA protocol: SmolLM2-135M/360M, rank 8, 10 epochs, 62–300 QA
  pairs per domain (brick 2 / v3), answer-conditional profiles,
  deterministic seeds; GPU nondeterminism disclosed (pattern-level
  reproducibility).
- Metrics: routing accuracy (domain-level), false capture, swap-isolation
  flip counts, per-domain breakdowns, oracle agreement, semantic
  similarity to reference for generation quality, latency.
- Everything reproducible: script-per-claim, no external data/network for
  the synthetic suite; model downloads + GPU for the real-LoRA arm.

### 5. Results
- **R1 — Competitive routing accuracy.** Stand-in lexical 96.4% = learned
  router (F1); semantic 98.2% > centroid 96.4% (F28 — ranking flips vs
  lexical); real-LoRA 96.4% at 4 domains (F5), 95.7% at 6 domains (F42);
  stable across seeds 42/7/2026 (F26) and 135M→360M (F27).
- **R2 — Isolation under swap (the second headline).** 0.00% collateral,
  structural (suite §5); text setting 0 flips (F4); 8 adapters 0/56
  (F18); noisy pools at N=128/512: 0.00% (F35).
- **R3 — Unseen-task behavior.** Stable, interpretable routing
  (education → finance 10/14, two independent implementations agree,
  F10); quality verified at the aligned-adapter level: routing 14/14 and
  best generation output (F13, F31); the failure mode is missing
  coverage, not broken routing. Semantic embeddings remove the lexical
  exemplar artifact (F29).
- **R4 — Bonferroni compounding does not transfer to correlated adapter
  pools (own section, per review; the most novel contribution).**
  Independent-gate compounding (§7 of TECHNICAL.md) is the worst case;
  measured false capture at 128 correlated variants: 22.6% vs 100% naive
  (F34). The accuracy curve is U-shaped in pool size (variant
  multiplicity helps, extreme-N max-of-N decays, F33); zero-noise
  accuracy flat 96.74% from 8 → 1024 adapters (F32); separation min
  collapses with noise while the mean holds (F36). Design rule: budget
  false capture with the correlated-error model for variant pools, the
  independent-gate model only for distinct-domain additions.
- **R5 — Latency.** Selection policy sub-millisecond at 10,000 adapters:
  profiling 1.14 ms, cosine 17 µs → 380 µs (F30).

### 6. Discussion
- The differentiator argument (per review): raw embedding similarity is
  the obvious approach and is documented elsewhere as insufficiently
  disambiguating; calibrated competence profiles against a boundary-dense
  moat outperform it, and the isolation properties are unique to the
  representation. No claim of beating LORAUTER on their benchmarks —
  the claim is a more principled representation with unique isolation
  properties, both evidence-backed.
- The moat as the defensible asset: anyone can copy the router (it is
  simple); nobody can copy a boundary-dense, domain-labelled calibration
  foundation.
- What the scaling law means for serving practice.

### 7. Limitations (as stated in REVIEW.md — reviewers want exactly this)
- Real 1000+-adapter pool: the scaling test is stand-in, grounded in
  real profile shapes (F32–F36).
- vLLM/LoRAX integration hook: policy latency measured standalone (F30),
  not inside a serving stack.
- Generation quality on a stronger base model (the 135M margin is
  small, F31).
- Corpus: synthetic-template English, six domains; no human review pass,
  no Swahili/sheng.
- Zero-parameter scope is the selection policy; the query profiler is a
  trained classifier.

### 8. Conclusion
- The selection layer is the missing piece between multi-LoRA serving and
  multi-tenant SaaS; a measurement-based representation provides it with
  competitive accuracy, unique isolation, and sub-millisecond cost.

---

## Figures / tables plan
- F1: The serving-layer gap (request-named vs content-based selection).
- T1: Routing accuracy by strategy and feature path (lexical/semantic).
- T2: Swap-isolation matrix (structural zero; scales to 8 adapters and
  noisy 512-pools).
- F2: False capture vs pool size, measured vs naive independence
  (the F34 money plot — U-shape + the gap between curves).
- T3: Real-LoRA loss matrices (4-domain 2/4 diagonal vs 6-domain 6/6).
- F3: Latency curve 4 → 10,000 adapters.

## Submission checklist (before ArXiv)
1. Decide: 4-domain numbers (98.2% semantic) or run the 6-domain
   semantic arm first (recommended — one ~5 min run, makes R1
   consistent with the 6-domain story).
2. Humanize one corpus slice (~50 examples) to show samples in the
   appendix (currently zero human review — the corpus is machine-
   generated; showing examples is fine, claiming review is not).
3. Exact claim wording audit: "zero-parameter" (selection layer),
   "98.2%" (4 domains, brick 2), "0.00%" (structural, all runs),
   F34 (σ=0.05, N=128).
4. Endorsement/affiliation logistics for ArXiv submission (independent
   researcher; 17; needs endorsement or a co-submitter).
5. License/attribution: MIT repo; cite the parent suite's TECHNICAL.md
   lineage honestly in Related Work.
