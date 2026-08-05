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

Research prototype, first verified benchmark. Adapters are stand-in
per-domain specialist classifiers on TF-IDF + SVD features (the §8 feature
stack), not real LoRA adapters; the corpus is `corpus/moat_brick2.jsonl`
(4 domains: finance, law, code, education; Kenyan context).

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

- Adapters are stand-in classifiers, not real LoRA adapters; no inference
  integration (vLLM/LoRAX), no latency or memory measurement at serving
  scale. The serving layer itself remains Punica/S-LoRA/dLoRA territory.
- Features are lexical (TF-IDF + SVD); no semantic embeddings in the text
  arm yet.
- ~13 calibration examples per domain; the p99/p95 threshold sensitivity
  documented in the parent suite's calibration trial applies here too.
- No benchmark yet against real LORAUTER-style routing at 1000+ adapters,
  and no unseen-task generalization test. Those are the next experiments.

## Related work (verified sources)

- Punica (arXiv:2310.18547) — multi-tenant LoRA serving, SGMV kernel
- S-LoRA (arXiv:2311.03285) — thousands of adapters, unified paging
- dLoRA (Wu et al., OSDI 2024) — dynamic request/adapter orchestration
- LoRAX (Predibase) — dynamic just-in-time adapter loading
- LORAUTER (arXiv:2601.21795) — task-representation adapter routing;
  the closest published approach to this one, with learned task embeddings
- MoLE line (Muqeeth et al. 2024; ICML 2025; HotMoE AAAI 2026) — learned
  routers over LoRA experts
