# lorouter

Profile-based LoRA adapter selection for multi-adapter LLM serving.

Serving systems (Punica, S-LoRA, dLoRA, LoRAX, vLLM) batch and load LoRA
adapters efficiently — but every one of them requires the request to name
its adapter. **lorouter is the selection layer they don't provide**: it
decides which adapter a query needs, from the query itself, with zero
learned router parameters.

- Each adapter carries a **calibrated profile vector**: its measured
  competence per domain (how strongly it claims inputs from each domain).
- A shared **profiler** maps a query into the same domain space.
- The router picks the top-k adapters by **cosine similarity**.
- Every decision reduces to a score: "adapter X won because it scored Y on
  domain Z" — traceable by construction, no black box.

This branch is the build-out of the multi-LoRA use case researched and
positioned in `possibility.md` (parent suite, v2 branch). The parent suite
(`TECHNICAL.md`) supplies the verified properties this router inherits:
swap isolation (0.00% collateral), boundary-example calibration (the §8
pipeline), and the data-moat addition strategy.

## Status

Research prototype. Three layers of verification, all honest about scope
(evidence: FINDINGS.md F-numbers):

1. Stand-in benchmark (brick 2 corpus machinery): profile routing ties a
   learned router (96.4% vs 96.4%) with zero learned router parameters;
   swap isolation holds in the text setting (F1–F4).
2. Real-LoRA integration (SmolLM2-135M/360M, rank 8, 10 epochs, QA pairs
   from brick 2 — the loader's default corpus): end-to-end mechanism
   verified — routing 96.4% on both model sizes and across 3 seeds
   (F5, F26, F27); adapter differentiation partial (2/4 diagonal,
   base-model code priors).
3. Unseen-task (education held out): routing stable and interpretable
   (education → finance; F10); quality verified at the aligned-adapter
   level — routing 14/14 and generation output best (F13, F31);
   LORAUTER-style exemplar embeddings mislead with lexical features (F12)
   but route sensibly with semantic embeddings (F29).
4. Semantic embeddings arm: profile routing improves to 98.2% and beats
   embedding-centroid routing — the ranking flip vs lexical features (F28).
5. Eight-adapter pool: domain accuracy holds at 96.4%; separation bounded
   by adapter diversity; swap isolation scales (F15–F18).
6. Latency spike: selection policy sub-millisecond at 10k adapters
   (F30).
7. Adapter-pool scaling to 1024 adapters (F32–F36): flat 96.74% at zero
   profile noise; U-shaped under noise; Bonferroni-style compounding does
   NOT transfer to adapter pools (correlated variants); swap isolation
   0.00% at N=128/512.
8. Six-domain real-LoRA scale-up on the v3 corpus (F42): 95.7% routing
   with 6/6 diagonal dominance — differentiation is complete at ~300 QA
   pairs/domain (revises the F7/F8 base-prior bound: it was data volume).

Full record: FINDINGS.md (42 numbered findings), REVIEW.md (significance
against 2026 trends + gap-closure status), experiments/lorouter_results.xlsx
(15 sheets, 12 charts, built by experiments/build_lora_workbook.py).

## Verified benchmark (experiments/benchmark.py, 5 seeds)

Adapter selection accuracy on clean test (56 inputs/seed; random floor
19.3%):

| strategy | acc (mean) | note |
|---|---|---|
| centroid (task-embedding routing) | 97.9% | LORAUTER-style; ahead by ~1 example/seed |
| profile (lorouter) | 96.4% | zero learned router params, stable across all 5 seeds |
| learned router (query→adapter map) | 96.4% | the MoE-LoRA line, simplified |
| random | 19.3% | floor |

- **Profile routing ties the learned router at 96.4% with zero learned
  router parameters**, while keeping every decision traceable.
- **Swap isolation verified in the text setting**: replacing the code
  adapter with a deliberately weak specialist and re-calibrating its
  profile changed routing on zero other-domain inputs (0 flips).
- The centroid gap (+1.4 pts) is one test example per seed — within
  small-n noise (56 test inputs), reported as such, not as a win.

Run it: `python3 experiments/benchmark.py` (scikit-learn + numpy, CPU-only,
~1 min). Deterministic seeds; numbers above are the canonical run.

## Repository layout (every file)

Quick index: what is this branch about → README.md · how it works →
LOROUTER.md · what could it be → possibility.md · what was proven →
FINDINGS.md · the numbers → experiments/lorouter_results.xlsx · parent-suite
theory → TECHNICAL.md (inherited from v2).

    Root documentation
      README.md            this file: status, results, limits, map
      LOROUTER.md          technical documentation of the adapter-selection
                           capability (mechanism, evidence tiers, limits)
      FINDINGS.md          all 25 numbered findings, each with its source
                           script (the complete evidence record)
      possibility.md       vision: use cases, products, engineering
                           practice, honest limits, roadmap
      REVIEW.md            internal significance review vs current trends
                           (2026 landscape, positioning, risks, priorities)
      TECHNICAL.md         parent suite's technical document (v2 branch,
                           inherited): swap isolation, addition problem,
                           calibration, prior-art acknowledgment
      performance_data.xlsx  parent suite's canonical run tables (v2,
                           inherited; regenerated by scripts/build_workbook.py)

    lorouter/              the router package (this branch's core)
      router.py            Adapter (calibrated competence vectors),
                           ProfileRouter (profiler + cosine top-k routing,
                           zero learned parameters), swap operation
      corpus.py            moat-corpus loader (split discipline preserved)
      __init__.py          package exports

    experiments/           runnable experiments (the source of truth)
      benchmark.py         stand-in: profile vs centroid vs learned vs
                           random, 5 seeds, per-domain, swap isolation
      real_lora_integration.py  SmolLM2-135M LoRA adapters (r8, 10 ep),
                           answer-conditional profiles, routing (96.4%)
      real_lora_multiseed.py    routing stability across seeds 42/7/2026
                           (96.4% each, F26)
      real_lora_360m.py        model-size check: SmolLM2-360M, 96.4% (F27)
      unseen_task_generalization.py  education held out; stability,
                           oracle agreement, loss gap (aligned-level
                           quality in F13/F31)
      lora_exemplar_routing.py  LORAUTER-style exemplar embeddings + the
                           aligned-adapter control (14/14)
      eight_adapter_space.py  8-adapter pool: scaling, profile separation,
                           swap isolation
      semantic_embeddings.py   bge-small arm: 98.2%, beats centroid (F28);
                           F12 artifact gone (F29)
      latency_spike.py     selection-policy latency 4 -> 10k adapters (F30)
      generation_quality.py    aligned-adapter generation scoring (F31)
      adapter_pool_scaling.py  pool scaling to 1024 (F32-F36); table in
                           experiments/results/pool_scaling.csv
      build_lora_workbook.py  regenerates lorouter_results.xlsx
      lorouter_results.xlsx   canonical tables + charts (15 sheets)
      results/             raw CSV outputs (pool_scaling.csv)

    scripts/               parent-suite + moat scripts (inherited from v2
                           plus this line's additions)
      shared_data.py       canonical data generator + Expert/profiler/
                           routing infrastructure (parent suite)
      addition_isolation_suite.py  parent suite §5/§6: swap isolation,
                           addition flips, gated fix, multi-seed stability
      capacity_ablation.py parent suite §6.4: 326x gate capacity test
      multi_dimension_compounding.py  parent suite §7: Bonferroni
                           compounding across simultaneous additions
      text_validation.py   parent suite §8: real-text flips + systematic
                           calibration-generation prototype
      boundary_solutions.py  parent suite §6.6: four mitigations head-to-
                           head (A/B/C/D)
      build_workbook.py    regenerates performance_data.xlsx
      moat_profile_addition.py  data-moat proof: profile insertion vs
                           joint-retrain addition (0 flips vs 184)
      build_moat_corpus.py seeded generator for the moat corpus bricks
                           (brick 3 current; 1 and 2 in git history)
      moat_calibration_trial.py  §8 replication on the corpus: clean-only
                           vs clean+boundary gate calibration, p99/p95

    corpus/                the moat calibration foundation
      moat_brick3.jsonl/.csv   current corpus: v3 — 3,010 examples,
                           6 domains (finance/law/code/education/medicine/
                           psychology), 220 boundary examples, Kenyan
                           context, splits 1673/807/530 (user-built
                           2026-08-06; bricks 1-2 + pre-v3 brick 3 in git
                           history)
      moat_brick3_crossdomain.csv  flat export of the same corpus

    assets/                media
      profile-moe-demo.mp4 parent suite's 3-minute demo video (inherited)

    .gitignore             ignores __pycache__ and venv artifacts

## Honest limits

- The real-LoRA experiments use small models (135M/360M) trained 10 epochs
  on synthetic QA pairs — a mechanism test, not an adapter-quality claim.
  Generation-quality margin is small at this scale (F31). No inference
  integration (vLLM/LoRAX) exists; the policy's standalone latency is
  measured (F30) but not inside a serving stack. The serving layer itself
  remains Punica/S-LoRA/dLoRA territory.
- The text arm now has both lexical (TF-IDF+SVD) and semantic
  (bge-small-en-v1.5) feature paths; the real-LoRA experiments use the
  lexical profiler — a semantic-profiler real-LoRA run is not done yet.
- ~135 clean calibration examples per domain (v3 brick 3: 807 calibration
  examples across 6 domains); the p99/p95 threshold sensitivity is
  documented in the parent suite's calibration trial.
- Unseen-task quality is verified at the aligned-adapter level (F13, F31),
  not at LORAUTER's 1000+-adapter scale. No head-to-head against
  LORAUTER/EdgeLoRA on their benchmarks.

## Related work (verified sources)

- Punica (arXiv:2310.18547) — multi-tenant LoRA serving, SGMV kernel
- S-LoRA (arXiv:2311.03285) — thousands of adapters, unified paging
- dLoRA (Wu et al., OSDI 2024) — dynamic request/adapter orchestration
- LoRAX (Predibase) — dynamic just-in-time adapter loading
- LORAUTER (arXiv:2601.21795) — task-representation adapter routing;
  the closest published approach to this one, with learned task embeddings
- MoLE line (Muqeeth et al. 2024; ICML 2025; HotMoE AAAI 2026) — learned
  routers over LoRA experts
