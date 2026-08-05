# possibility.md — The Adapter-Selection Capability (Use Cases, Products, Engineering Practice)

Status: vision document for the `lorouter` branch, explicitly downstream of
`LOROUTER.md` and `FINDINGS.md` (which are downstream of the parent suite's
`TECHNICAL.md`, v2 branch). Every claim is tagged by evidence tier —
**proven** (reproducible by script in this repository), **bounded**
(proven under stated conditions), **designed** (specified, not yet
tested), or **open** (hypothesis, not yet validated). Nothing below is
stated more strongly than its evidence supports; findings are cited as
F-numbers from `FINDINGS.md`.

---

## 1. Where This Capability Actually Fits

Serving systems for fine-tuned models — Punica (arXiv:2310.18547),
S-LoRA (arXiv:2311.03285), dLoRA (OSDI 2024), LoRAX, vLLM — solved *how to
batch and load* many LoRA adapters on one base model. Every one of them
requires the request to name its adapter. The lorouter capability is the
decision layer they do not provide: **which adapter does this query
need**, answered from query content, with zero learned router parameters
and fully traceable decisions (F1, F5, F24).

**Good fit — each condition tied to evidence, not aspiration:**

- **Many adapters, one base model, a known task taxonomy.** The router's
  query profiler needs a domain space to profile into; that space is the
  calibration foundation (the moat, parent suite §3). Domain-level routing
  accuracy is 96.4% in both the stand-in and real-LoRA settings (F1, F5).
- **Auditability matters.** Every decision reduces to a score — "adapter X
  won because it scored Y on domain Z" — inherited from the parent suite's
  traceability property (§10 of TECHNICAL.md).
- **The pool is swap-heavy.** Adding or replacing an adapter is a
  calibration operation, not a training operation (F24); swap isolation
  holds in the text setting (F4) and scales to 8 adapters (F18).
- **The moat covers the domains.** The aligned-adapter control shows the
  mechanism's ceiling is adapter *coverage*: with the domain's adapter
  present, routing is 14/14 (F13). The moat is what guarantees coverage.

**Poor fit — stated as plainly:**

- Single-adapter deployments (no selection problem exists).
- Open-ended task discovery — domains that must emerge from data, where
  the calibration foundation cannot pre-cover them (parent suite §6's
  open case applies unchanged).
- Latency-critical serving where even sub-millisecond profiling is
  unaffordable — routing overhead has not been measured yet (open).
- Teams unwilling to maintain calibration data: profiles are only as good
  as the calibration discipline behind them (F9 shows the metric design
  is load-bearing; parent suite §8 shows the data requirements).

---

## 2. Use Cases

### 2.1 Multi-LoRA serving selection policy (bounded)
The primary use case: a routing policy that sits in front of the serving
layer — request arrives, profiler maps it to a domain profile, cosine
top-k selects the adapters, the serving layer (Punica/S-LoRA/LoRAX/vLLM)
batches them. Verified at the mechanism level with real LoRA adapters
(96.4%, F5); the serving-layer integration itself is designed, not built.

### 2.2 Adapter marketplace / profile registry (designed)
The moat is the adapter-profile factory (F24): adapters enter the router
by calibration, so profiles are first-class, versionable, auditable
metadata. A registry of calibrated adapter profiles — priced by
calibration cost, not training cost — is the marketplace reading of this
capability, directly downstream of the parent suite's §4.2.

### 2.3 On-device adapter routing (designed)
The privacy-hybrid use case from the parent suite (§2.3), now with a
concrete selector: local profiler, local adapter pool, profile insertion
as the only update operation. Nothing leaves the device.

### 2.4 Enterprise adapter pools with audit (bounded)
Traceability inherited from the parent suite makes adapter selection
auditable per request — relevant for regulated deployments where "which
model answered this" is a compliance question.

### 2.5 Testing and auditing tooling (proven-grounded)
The benchmark and isolation harnesses (F1–F18) are themselves a product
surface: a regression suite for any profile-routed adapter pool — flip
detection, swap isolation, pool-separation checks, calibration-health
metrics (F23).

---

## 3. Evidence (what is verified, numbers and finding references)

| Experiment | Result | Findings |
|---|---|---|
| Stand-in, 5 seeds (brick 2) | profile 96.4% = learned router 96.4%; random 19.3% | F1, F2, F3 |
| Swap isolation, text setting | 0 flips on other domains | F4 |
| Real-LoRA (SmolLM2-135M, r8, 10ep) | 96.4% (54/56), robust across profile metrics | F5, F6, F9 |
| Adapter differentiation | 2/4 diagonal; base-model code priors bound it | F7, F8 |
| Unseen task (education held out) | stable routing (10/14 → finance), quality open | F10, F11 |
| Exemplar task embeddings | mislead (all → code, lexical artifact) | F12 |
| Aligned-adapter control | routing 14/14 — ceiling is coverage | F13, F14 |
| 8-adapter pool | 96.4% domain-level; separation bounded by diversity | F15, F16, F17 |
| Swap isolation, 8 adapters | 0/56 flips | F18 |
| Corpus brick 3 + calibration | 664 ex, p99 degeneracy resolved, §8 property at p95 | F19–F23 |

## 4. Products (ordered by how much of the claim is proven)

### 4.1 Adapter-selection test toolkit (strongest, proven-grounded)
Package the benchmark + isolation harness (F1–F18) as a regression-testing
product for teams running profile-routed adapter pools: routing accuracy,
swap flips, pool separation, calibration health. Mostly repackaging of
what exists and is verified.

### 4.2 Lorouter routing policy for vLLM/LoRAX (designed)
The integration product: a policy hook that turns a request into a
top-k adapter selection before the serving layer batches it. The selection
mechanism is proven; the hook, the latency budget, and the concurrency
behavior are not built or measured (open).

### 4.3 Adapter profile registry + calibration service (designed)
The moat-as-factory productized: boundary-dense calibration foundation
(§E of FINDINGS) + profile minting service (answer-conditional
calibration, F9) + versioned registry. The calibration foundation exists
(bricks 1–3); the service layer does not.

### 4.4 On-device adapter selector (designed)
Use case §2.3 productized; smallest surface, cleanest privacy story.

## 5. Engineering Practice

- **Profile minting discipline.** Profiles must measure answer behavior,
  not question text (F9). Calibration needs held-out splits never reused
  as training data, and boundary-dense calibration sets (F19–F23, parent
  suite §8).
- **Adapter lifecycle.** Calibrate → version → route → audit. The profile
  is metadata: versioned with the adapter, auditable per decision.
- **Percentile/threshold practice.** p99 over-tightens at small
  calibration n; p95 preserves recall while boundary-inclusive thresholds
  still kill false-capture (F20, F21). Choose and document the percentile
  per deployment; monitor drift.
- **Pool design.** Adapter diversity bounds profile separation (F16):
  near-duplicate adapters yield near-identical profiles and routing
  becomes profiler-dominated. Diversify adapters (data, answers, ranks)
  or accept profiler-dominated routing knowingly.
- **Monitoring.** Flip rate after swaps (F4, F18), margin distribution
  (F10's margins), per-adapter claim drift, calibration health (F23).

## 6. Honest Limits (consolidated)

- Lexical features only (TF-IDF + SVD); no semantic embeddings in the
  text arm (F-open).
- 135M model, synthetic QA pairs — a mechanism test, not an
  adapter-quality claim (F5's bound).
- Unseen-task quality is open at this scale (F11); the aligned-control
  result (F13) is the direction, not the destination.
- No serving-layer integration, no latency/memory measurement, no
  generation-quality evaluation.
- Variant-exact routing is undefined for near-duplicate adapters (F17).
- Corpus is synthetic-template English; no human review pass, no
  Swahili/sheng coverage.

## 7. Roadmap (designed/open)

1. Semantic embeddings for the text arm (replaces the lexical ceiling).
2. A real unseen task WITH aligned adapters — the LORAUTER-style claim
   (task-level routing, 1000+ adapters) tested head-to-head.
3. vLLM/LoRAX integration hook + latency measurement.
4. Generation-quality evaluation of routed adapters (does the selected
   adapter produce better answers, not just lower loss).
5. Human-reviewed corpus brick; Swahili/sheng coverage.

---

*The scripts are the source of truth. FINDINGS.md carries the numbered
evidence; LOROUTER.md carries the capability's technical documentation;
experiments/lorouter_results.xlsx carries the canonical tables and charts
(built by experiments/build_lora_workbook.py).*
