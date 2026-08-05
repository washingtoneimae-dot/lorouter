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

Research prototype. Two layers of verification, both honest about scope:

1. Stand-in benchmark (brick 2/3 corpus machinery): profile routing ties a
   learned router (96.4% vs 96.4%) with zero learned router parameters;
   swap isolation holds in the text setting.
2. Real-LoRA integration (SmolLM2-135M-Instruct, rank 8, 10 epochs, QA
   pairs built from brick 3): the full mechanism works end-to-end —
   real LoRA adapters, real calibration losses, profile router → 96.4%
   routing accuracy, matching the stand-in benchmark. Adapter
   differentiation is partial (2/4 adapters lowest on their own domain;
   finance/law adapters both favor code — the base model's code priors).
3. Unseen-task test (education held out): routing decisions are stable and
   interpretable (education → finance, 10/14, identical in the real-LoRA
   and stand-in arms), but oracle agreement is 42.9% and the loss gap
   equals random expectation — quality unverified at this scale. The
   aligned-adapter control (lora_exemplar_routing.py V3) confirms the
   mechanism's ceiling is adapter COVERAGE, not routing quality: with the
   domain's adapter present, routing is 14/14. LORAUTER-style exemplar
   task embeddings actively mislead (all → code, lexical artifact).
4. Eight-adapter pool: domain-level accuracy holds at 96.4% with 2
   adapters/domain; profile separation is bounded by adapter diversity
   (near-duplicate adapters → cosine 0.996) but routing still works; swap
   isolation scales (0/56 flips).

Full record: FINDINGS.md (25 numbered findings) and
experiments/lorouter_results.xlsx (6 sheets, charts, built by
experiments/build_lora_workbook.py).

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
      unseen_task_generalization.py  education held out; stability,
                           oracle agreement, loss gap (quality open)
      lora_exemplar_routing.py  LORAUTER-style exemplar embeddings + the
                           aligned-adapter control (14/14)
      eight_adapter_space.py  8-adapter pool: scaling, profile separation,
                           swap isolation
      build_lora_workbook.py  regenerates lorouter_results.xlsx
      lorouter_results.xlsx   canonical tables + charts (6 sheets)

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
      moat_brick3.jsonl/.csv   current brick: 664 examples, 4 domains,
                           Kenyan context, 64 boundary examples
                           (bricks 1-2 in git history)

    assets/                media
      profile-moe-demo.mp4 parent suite's 3-minute demo video (inherited)

    .gitignore             ignores __pycache__ and venv artifacts

## Honest limits

- Adapters in the stand-in benchmark are classifiers, not real LoRAs; the
  real-LoRA integration uses a 135M model trained 10 epochs on synthetic
  QA pairs — a mechanism test, not an adapter-quality claim. No inference
  integration (vLLM/LoRAX), no latency or memory measurement at serving
  scale. The serving layer itself remains Punica/S-LoRA/dLoRA territory.
- Features are lexical (TF-IDF + SVD); no semantic embeddings in the text
  arm yet.
- ~27 clean calibration examples per domain (brick 3); the p99/p95
  threshold sensitivity is documented in the parent suite's calibration
  trial.
- Unseen-task routing quality is NOT verified at this scale (loss gap
  equals random); only routing stability is. No benchmark yet against real
  LORAUTER-style routing at 1000+ adapters.

## Related work (verified sources)

- Punica (arXiv:2310.18547) — multi-tenant LoRA serving, SGMV kernel
- S-LoRA (arXiv:2311.03285) — thousands of adapters, unified paging
- dLoRA (Wu et al., OSDI 2024) — dynamic request/adapter orchestration
- LoRAX (Predibase) — dynamic just-in-time adapter loading
- LORAUTER (arXiv:2601.21795) — task-representation adapter routing;
  the closest published approach to this one, with learned task embeddings
- MoLE line (Muqeeth et al. 2024; ICML 2025; HotMoE AAAI 2026) — learned
  routers over LoRA experts
