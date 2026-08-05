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

## Layout

    lorouter/            the router package
      router.py          Adapter (calibrated competence vectors),
                         ProfileRouter (profiler + cosine top-k routing),
                         swap operation
      corpus.py          moat-corpus loader (split discipline preserved)
    experiments/
      benchmark.py       profile vs centroid vs learned vs random,
                         5 seeds, per-domain breakdown, swap-isolation test
    corpus/              moat corpus bricks (shared with v2 branch)

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
