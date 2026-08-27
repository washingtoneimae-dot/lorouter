# LOROUTER — Profile Routing as an Adapter-Selection Capability

Status: research document for the `lorouter` branch. For the complete
file map (where every file is and what it is), see README.md →
"Repository layout". Evidence tiers follow the parent suite's convention: **proven** (verified by runnable script,
reproducible), **bounded/conditional** (proven under stated conditions with
known limits), **designed** (specified, not yet tested), **open**
(hypothesis, not yet validated). Every number below is produced by a script
in this repository; the scripts are the source of truth.

---

## 1. What this angle is

The parent suite (`TECHNICAL.md`, v2 branch) established that a
profile-routed MoE — experts carrying calibrated competence vectors,
inputs profiled into the same space, cosine top-k selection — swaps experts
with zero collateral (0.00%, structural), adds moat-covered domains as
swaps, and keeps every decision traceable.

The lorouter angle is the observation that **this mechanism is not limited
to MoE experts. It is an adapter-selection capability.** The serving
ecosystem for fine-tuned models (Punica, S-LoRA, dLoRA, LoRAX, vLLM)
solved *how to batch and load* many LoRA adapters on one base model — but
every one of them requires the request to name its adapter. No serving
system decides which adapter a query needs. Lorouter is that decision
layer: it selects adapters from query content, with the same zero-learned-
parameter, fully-traceable mechanism the suite proved for experts.

**The new capability surface:** the profile vector stops being only an
internal routing artifact and becomes *adapter metadata* — a calibrated,
versionable, auditable description of what an adapter is good at, minted
from a calibration foundation (the moat). Adapter selection becomes a
data operation, not a training operation.

---

## 2. Mechanism (unchanged from the parent suite, applied to adapters)

1. **Adapter profile**: for each adapter, a competence vector over the
   known domains. In the stand-in setting: mean claim-strength of the
   adapter's specialist classifier per domain. In the real-LoRA setting:
   inverse calibration loss per domain, measured on (question + that
   domain's answer behavior) pairs — the LM-text analog of the suite's
   inverse-MSE formula. Calibration is *measurement*, not training.
2. **Query profile**: a shared profiler (TF-IDF + SVD features, §8 stack)
   maps the query into the same domain space.
3. **Selection**: cosine top-k between query profile and adapter profiles.
   Zero learned router parameters. Every decision reduces to a score.

The router's guarantees are inherited from the parent suite: swap
isolation is structural (a profile insertion cannot change other adapters'
scores), additions are swaps when the moat covers the domain (§3.2 of
possibility.md), and calibration discipline is the binding constraint (§8).

**Load-bearing design constraint (F9)**: the profile must measure the
adapter's *answer behavior*, not its fit to the question text. The first
real-LoRA implementation profiled against question-only NLL and produced
near-uniform profiles — routing collapsed to 73.2% with the code domain at
0%. Switching to (question + domain answer) calibration restored 96.4%.
This constraint is not a tuning detail: profiles that measure question
text measure the base model, not the adapter.

---

## 3. Evidence

### 3.1 Stand-in benchmark (proven, brick 2)

`experiments/benchmark.py`, 5 seeds, 56 test inputs/seed, random floor
19.3%:

| strategy | acc | note |
|---|---|---|
| centroid (task-embedding routing) | 97.9% | ahead by ~1 example/seed, small-n |
| **profile (lorouter)** | **96.4%** | zero learned params; stable across seeds |
| learned router (query→adapter map) | 96.4% | the MoE-LoRA line, simplified |
| random | 19.3% | floor |

- Profile routing ties a learned router at 96.4% with zero learned router
  parameters.
- **Swap isolation verified in the text setting**: replacing the code
  adapter with a deliberately weak specialist changed routing on 0
  other-domain inputs.

### 3.2 Real-LoRA integration (proven as mechanism, bounded as quality)

`experiments/real_lora_integration.py` — SmolLM2-135M-Instruct, rank-8
LoRA adapters, 10 epochs on QA pairs built from brick 2 (the loader's
default corpus), profiled by answer-conditional calibration loss:

- **Routing accuracy: 96.4% (54/56)** — identical to the stand-in
  benchmark, and robust across two profile metrics (question-only loss and
  answer-conditional loss both route at 96.4%).
- **Adapter differentiation: 2/4 diagonal at 62 QA pairs/domain — and
  6/6 at ~300 QA pairs/domain (F42).** At the original training volume
  the code adapter and the education adapter were lowest on their own
  domains while finance/law favored code — the base model's strong code
  priors. Re-run on the v3 corpus with ~5x the training data, every
  adapter is lowest on its own domain (medicine 0.55 vs 2.1+ elsewhere).
  The differentiation bound was data volume, not base priors (revises
  the earlier reading in FINDINGS F7/F8).
- Mechanism verdict: the full chain (real LoRA → real losses → profiles →
  cosine selection) works end-to-end.

### 3.3 Unseen-task generalization (proven stable; quality verified at
the aligned-adapter level, F13/F31)

`experiments/unseen_task_generalization.py` — education fully held out
(no education adapter, profiler never saw education):

- **Routing decisions are stable and interpretable**: education queries
  route to finance 10/14, code 2, law 2 — identical in the real-LoRA and
  the stand-in arms (two independent implementations agreeing). HELB/
  bursary/fee queries are lexically finance-adjacent; the router's choice
  is defensible.
- **Without an aligned adapter, per-query quality sits at the noise
  floor**: oracle agreement 42.9%, loss gap == random expectation — the
  earlier open result, now superseded in scope by the aligned case below.
- **With an aligned adapter present** (the moat-covered case, F13/F31):
  routing is 14/14 and the routed adapter produces the best generation
  output (0.5932 vs 0.5887 semantic similarity to reference, +0.45pp).
  The unseen-task failure mode is *missing coverage*, not broken routing —
  exactly the case the moat strategy pre-empts (possibility.md §3).

### 3.4 Adapter-space scaling and the aligned-adapter control (bounded)

The follow-up experiments this document was written to contain, all now
run:

- **Richer space (8 adapters)**: two adapters per domain, trained on
  disjoint data with different answer variants. Tests whether profiles
  stay separable and routing accuracy holds as the adapter pool grows.
- **Pool scaling to 1024 adapters** (`experiments/adapter_pool_scaling.py`,
  F32–F36): zero-noise accuracy flat at 96.74% from 8 to 1024 adapters;
  U-shaped under profile noise; Bonferroni-style compounding does NOT
  transfer to adapter pools (correlated variants); swap isolation 0.00%
  at N=128/512.
- **LORAUTER-style exemplar signal**: task embeddings derived from a
  small validation set of the unseen task, per LORAUTER (arXiv:2601.21795),
  plus the aligned-adapter control: the same unseen queries routed with an
  education adapter present — which should restore oracle-level agreement
  and confirm the mechanism's ceiling.

Results of both: see `experiments/lora_exemplar_routing.py` and
`experiments/eight_adapter_space.py`; summary tables in
`experiments/lorouter_results.xlsx` (built by
`experiments/build_lora_workbook.py`).

### 3.5 External data-moat pipeline (2026-08-27, F43–F46)

A hosted data-optimization + training pipeline (Adaption Adaptive Data +
AutoScientist on NVIDIA GPUs; Featherless for reference-answer generation)
applied to the moat corpus — the first concrete step of the coverage-side
build-out (TECHNICAL.md §9). Full record: `experiments/adaption/STATUS.md`.

- **Enhanced calibration foundation (F43)**: the v3 train split (1,673
  rows) through Adaptive Data → 1,117 enhanced QA pairs, platform grade
  D→B (+87.5%), committed as `corpus/moat_brick4_adapted.csv`. The
  `enhanced_completion` column carries model-generated, domain-grounded
  answers with Kenyan context — the training-signal upgrade over the
  synthetic templates.
- **Modernized joint-retrain counterfactual (F44)**: a 0.8B joint
  baseline (Qwen3.5-0.8B LoRA r16/α32, 1 epoch on the 1,117 enhanced
  pairs, `experiments/adaption/checkpoint/joint-baseline/`) — the thing
  the per-domain pool must beat, at a scale beyond the old 135M joint
  (which `moat_profile_addition.py` used, 184 flips at the old size).
- **External real-world domains (F45)**: agriculture (500 rows), fintech
  (200), 5G-NR/telecom (343 after a Featherless completion lift to reach
  F42 scale — `corpus/moat_telecom_domain.csv`). The 9-domain pool's new
  adapters are grounded in real external data, not synthetic generation.
- **Learned-router corpus-level preview (F46)**: LogReg on TF-IDF
  (v3 train split only) scores 96.4% on the held-out test — reproducing
  the F5 tie at corpus level vs profile routing's 95.7% (F42).
- **Eval references**: 420 held-out v3 test questions paired with
  independent Featherless reference answers (per-domain prompts, 0
  failures, ~$0.04) — the generation-quality scoring set (F31-style),
  ready for the 9-domain benchmark.

The pending step is the **9-domain real-adapter routing benchmark**
(profile vs learned vs centroid vs random on real LoRAs) — the strongest
form of the central claim.

---

## 4. The moat as the adapter-profile factory (the strategic claim)

Possibility.md §3 proved (synthetic) that a broad calibration foundation
converts additions into swaps. Lorouter makes that operational for
adapter pools: **a new adapter enters the router by calibration, not by
training the router.** The moat corpus (bricks 1–3; brick 3 is now v3 —
3,010 examples, six domains, 220 boundaries, F37–F41; plus the 2026-08-27
additions: 1,117 enhanced pairs in `moat_brick4_adapted.csv` and the
343-row telecom domain in `moat_telecom_domain.csv`) is the calibration
foundation; adapter profiles are minted
from it. The corpus growth experiment (brick 3, 25% calibration split)
resolved the p99 threshold degeneracy documented in the parent suite's
calibration trial — the calibration foundation is now large enough for
stable threshold behavior at the 99th percentile. The external pipeline
(§3.5) is how the moat grows beyond hand-written generation: hosted
data optimization plus real-world external domains.

This is the defensibility argument stated as engineering: anyone can copy
the router (it is simple, MIT); nobody can copy a boundary-dense,
domain-labelled calibration foundation — and that foundation is what makes
adapter selection cheap, auditable, and swap-safe.

---

## 5. Honest limits (consolidated)

- Real-LoRA experiments use small models (135M/360M), synthetic QA pairs,
  10 epochs, rank 8 — a mechanism test, not an adapter-quality claim.
  Generation-quality margin is small at this scale (F31). No inference
  integration (vLLM/LoRAX) exists; the selection policy's standalone
  latency is measured (F30) but not inside a serving stack.
- The text arm has both lexical (TF-IDF+SVD) and semantic
  (bge-small-en-v1.5) feature paths; the real-LoRA experiments use the
  lexical profiler — a semantic-profiler real-LoRA run is not done yet.
- Unseen-task quality is verified at the aligned-adapter level (F13,
  F31); the no-aligned-adapter case remains at the noise floor, and
  1000+-adapter scale was tested with stand-in variants grounded in real
  profile shapes (F32–F36), not with a real pool of that size.
- Adapter differentiation is bounded by base-model priors (§3.2).
- The corpus is synthetic-template English; no human review pass, no
  Swahili/sheng coverage. (Partially addressed 2026-08-27: enhanced
  completions and three real-world external domains added, F43–F46.)
- The 9-domain real-adapter routing benchmark — profile vs learned vs
  centroid vs random on real LoRAs over the 9-domain pool — is pending
  (the F43–F46 routing comparison).

---

## 6. Related work (verified sources)

- Punica (arXiv:2310.18547), S-LoRA (arXiv:2311.03285), dLoRA (OSDI
  2024), LoRAX (Predibase): the serving layer — solved, out of scope.
- LORAUTER (arXiv:2601.21795): task-representation adapter routing; the
  closest published approach. Its settings (task-level embeddings from
  validation sets, 1500+ adapters, unseen tasks with aligned adapters)
  are the benchmark this branch is building toward.
- MoLE line (Muqeeth et al. 2024; ICML 2025; HotMoE, AAAI 2026): learned
  routers over LoRA experts — the coupling profile routing removes.

---

*Evidence tiers are load-bearing: §3.1–3.2 are proven as stated, §3.3 is
stable-but-open, §3.4 is bounded by the experiments it references, §3.5's
assets are proven as delivered (corpus/checkpoint/scripts all committed)
with its routing comparison pending. The scripts are the source of truth;
rerun them before quoting these numbers elsewhere.*
