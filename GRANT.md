# Grant Proposal — lorouter: Zero-Parameter Profile Routing for Multi-LoRA Serving

> **Status:** DRAFT (2026-08-27) — ready to adapt to specific funders
> (AI4D Africa, Lacuna Fund, Mozilla Builders, Google DeepMind/Anthropic
> academic programs, KENET/ICTA research calls, local university funds).
> Replace `[FUNDER]` / `[CALL]` / `[AMOUNT]` placeholders per application.

---

## 1. Executive summary

Serving systems for fine-tuned models (Punica, S-LoRA, dLoRA, LoRAX, vLLM)
can now host thousands of LoRA adapters — but **none of them decide which
adapter a query needs**. lorouter is the missing selection layer: it routes
each query to the right adapter using **calibrated per-domain profile
vectors and cosine similarity — zero learned router parameters**. Every
decision is traceable ("adapter X won because it scored Y on domain Z"), in
contrast to learned gates whose decisions are a black-box forward pass.

The mechanism is verified end-to-end with real LoRA adapters (routing
96.4% at 4 domains, 95.7% at 6 domains with 6/6 diagonal dominance,
F5/F26/F27/F42), and pool scaling holds to 1,024 simulated adapters
(F32–F36). **This proposal funds the coverage-side build-out**: a
9-domain, real-adapter pool grounded in externally-sourced and
machine-optimized data, plus the routing benchmark that converts the
mechanism proof into a publishable, reproducible result.

## 2. Problem statement

- **The adaptation era is here, but selection is missing.** The serving
  stack loads adapters on demand; deciding *which* adapter is left to the
  application developer, who hard-codes it or falls back to one big model.
- **Learned routers are the default answer — and they're expensive.**
  The MoE-LoRA line (Muqeeth et al.; LORAUTER, arXiv:2601.21795) learns
  task representations or gates. That means: router parameters that must be
  trained, retrained when adapters change, and that make every decision a
  black box — a liability in regulated domains (finance, law, health).
- **Traceability is not a luxury; it's a requirement** for auditable
  deployments (SACCOs, telecom, government services — the actual
  deployment contexts of this project's corpus).

## 3. Approach (what lorouter does differently)

1. Each domain adapter is **calibrated**: a competence profile vector
   measured from its loss on held-out domain data (the LM-text analog of
   inverse-MSE profiles).
2. A shared, fixed-size **profiler** maps a query into the same domain
   space (TF-IDF+SVD lexical arm, bge-small semantic arm).
3. The router selects top-k adapters by **cosine similarity**.
4. **Zero learned router parameters** — parameter count is independent of
   adapter count; adding an adapter never retrains the router; swap
   isolation is provable by construction (Theorem 1, TECHNICAL.md §4.5).

Verified results (46 numbered findings, FINDINGS.md):

| Result | Finding |
|---|---|
| Profile routing ties a learned router at 96.4% — zero learned params | F5 |
| Real-LoRA routing 96.4% at 4 domains, stable across 3 seeds | F26/F27 |
| 6-domain real-LoRA: 95.7%, 6/6 diagonal dominance at ~300 pairs/domain | F42 |
| Semantic arm: 98.2%, beats embedding-centroid routing | F28 |
| Selection-policy latency sub-millisecond at 10k adapters | F30 |
| Pool scaling to 1,024 adapters: flat 96.74% at zero profile noise | F32–F36 |
| Swap isolation 0.00% at N=128/512 | F32–F36 |
| Learned-router corpus-level preview: 96.4% (reproducible) | F46 |

## 4. What this grant funds (scope, 3–4 months)

1. **9-domain real-adapter pool build** (6 existing + telecom/5G,
   agriculture, fintech): normalize the adapted corpora
   (`corpus/moat_brick4_adapted.csv` + external adaptations) into the
   validated corpus schema with split discipline; train per-domain LoRAs.
2. **The routing benchmark**: profile vs learned vs centroid vs random on
   real adapters at 9 domains — the strongest form of the paper's central
   claim, plus generation-quality scoring against 420 held-out reference
   answers (already generated).
3. **vLLM/LoRAX integration hook** (currently measured in isolation, F30;
   not yet inside a serving stack): the first in-stack latency and routing
   measurement.
4. **Publication**: arXiv submission (skeleton exists:
   `paper/lorouter-arxiv-draft.md`), open dataset and adapter release on
   Hugging Face (`washi254/lorouter-moat-corpus-v4`).

**Budget sketch** — adapt to `[FUNDER]`/`[CALL]`:

| Item | [AMOUNT] |
|---|---|
| GPU training (Adaption/Together credits, 9 LoRAs + baselines) | [~$300] |
| Data-optimization pipeline credits (Adaptive Data passes) | [~$120] |
| Human corpus review + Swahili/sheng expansion | [—] |
| Publication fees / open-access (if applicable) | [—] |
| **Total** | **[AMOUNT]** |

## 5. Why this project, why now

- **The trend is converging on this exact gap.** Adapter serving is
  mainstream; dataset-optimization startups (Adaption, ex-DeepMind/Cohere
  founders) and fine-tuning platforms (Together AI) now industrialize the
  data side. The *selection layer* is the missing piece — this project
  already has a working, verified prototype of it.
- **It's real, not a demo.** The repo contains 46 numbered findings, each
  with its reproducing script, an honest limits section, and a canonical
  benchmark that runs CPU-only in ~1 minute.
- **African context is a feature, not an edge case.** The corpus is
  Kenyan-context (M-Pesa, SACCOs, KUCCPS, HELB, Kenyan law); the deployment
  targets are African fintech and public services where auditability and
  small-model economics matter most. The applicant is a self-taught
  developer in Kenya building this without institutional compute.

## 6. Team

- **Washingtone Imae** (GitHub: washingtoneimae-dot) — project lead,
  self-taught, Kenya. Author of the lorouter evidence chain (46 findings),
  plus portfolio: Bitcoin proof-of-existence (Bit Protocol), telecom tower
  health (5G SSB), solar soiling tracker (SolDegarde), SACCO fintech
  (saccosystem2), LoRA routing research (Profile-MoE). Independent
  researcher; no institutional affiliation — which is precisely what this
  grant unlocks.

## 7. Honest limits (stated plainly, as the project always does)

- Small models (135M/360M/0.8B) — mechanism test, not an
  adapter-quality claim.
- 1,000+-adapter *real* pool remains open (verified only in simulation).
- No serving-stack integration yet; policy latency measured in isolation.
- Corpus is English, partly synthetic-derived; no human review pass yet.
- External data licenses: 5G set is CC-BY-NC (paper/demo only).

## 8. Deliverables (acceptance criteria)

1. `corpus/moat_brick5_ninedomain.csv` — normalized 9-domain corpus with
   split discipline, committed and verified (schema checks in CI-style
   script).
2. 9 per-domain LoRA adapters + joint baseline, published on Hugging Face
   with model cards.
3. Benchmark tables: profile vs learned vs centroid vs random at 9 domains
   (routing accuracy + diagonal dominance + swap isolation), canonical
   run, reproducible.
4. Generation-quality scores on the 420-question held-out reference set.
5. vLLM/LoRAX integration hook with in-stack latency numbers.
6. arXiv submission with the external data-moat addition section.

## 9. Contact & links

- Repo: https://github.com/washingtoneimae-dot/lorouter
- Findings: `FINDINGS.md` · Technical: `TECHNICAL.md` · Status:
  `experiments/adaption/STATUS.md`
- Contact: [email] washingtoneimae@gmail.com

---

*This document is a living template: `STATUS.md` tracks what is done vs
pending so the proposal can always be updated from one source of truth.*
