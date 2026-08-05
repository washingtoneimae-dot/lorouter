# FINDINGS — lorouter, complete record

Every major finding from the lorouter line of work (branch `lorouter`,
canonical run 2026-08-05). Each finding names its source script; the
scripts are the source of truth. Evidence tiers: **proven** (reproducible
by script), **bounded** (proven under stated conditions), **open**
(hypothesis, not yet validated).

---

## A. Stand-in benchmark (`experiments/benchmark.py`, brick 2, 5 seeds)

1. **Profile routing ties a learned router with zero learned parameters.**
   96.4% vs 96.4% (learned router, the MoE-LoRA line simplified), stable
   across all 5 seeds, random floor 19.3%. The interpretability and swap
   safety of profile routing cost nothing in accuracy at this scale.
   *(proven)*
2. **Centroid routing (LORAUTER-style task embeddings) is marginally
   ahead — within noise.** 97.9% vs 96.4%; the gap is one test example
   per seed (56 test inputs). Reported as such, not as a win. *(bounded)*
3. **The selection problem is real.** Random is 19.3% on 4 domains —
   routing matters; every semantic strategy crushes it. *(proven)*
4. **Swap isolation holds in the text setting.** Swapping the code
   adapter for a deliberately weak specialist changed routing on 0
   other-domain inputs. *(proven)*

## B. Real-LoRA integration (`experiments/real_lora_integration.py`)

5. **The mechanism works end-to-end with real LoRA adapters.** Four
   domain LoRAs (SmolLM2-135M-Instruct, rank 8, 10 epochs, QA from brick
   3), profiles from answer-conditional calibration loss, cosine routing:
   **96.4% (54/56)** — identical to the stand-in benchmark. *(proven as
   mechanism; bounded as quality)*
6. **Routing is robust to the profile metric.** Question-NLL profiles and
   answer-NLL profiles both route at 96.4%. *(bounded)*
7. **Adapter differentiation is partial: 2/4 diagonal.** The finance and
   law adapters are both lowest on code — the base model's strong code
   priors make code the easiest domain for every adapter. Profiles are
   only as separable as the adapters beneath them. *(bounded)*
8. **Under-training flattens profiles but routing survives.** At 2 epochs
   the adapters barely differentiated (loss ~10 across the board), yet
   routing stayed 96.4% — the shared query profiler carries most of the
   routing signal at this scale. *(bounded)*
9. **Profile metric design is load-bearing.** Question-only loss profiles
   collapsed routing to 73.2% with code at 0% (near-uniform profiles
   after normalization); answer-conditional loss restored 96.4%. Profiles
   must measure *answer behavior*, not question text. *(proven)*

## C. Unseen-task generalization (`experiments/unseen_task_generalization.py`,
   `experiments/lora_exemplar_routing.py`)

10. **Unseen-domain routing is stable and interpretable.** Education
    queries route finance 10/14, code 2, law 2 — identical in the
    real-LoRA arm and the stand-in arm (two independent implementations
    agreeing). HELB/bursary/fee queries are lexically finance-adjacent.
    *(proven)*
11. **Unseen-task quality is NOT verified.** Oracle agreement 42.9%;
    loss gap equals random expectation (question-NLL oracle). With the
    answer-NLL oracle, ALL arms sit at the noise floor (gaps
    0.04–0.13%): the three adapters are nearly interchangeable on
    out-of-domain answer behavior at this scale. *(open)*
12. **LORAUTER-style exemplar task embeddings actively mislead.** All 14
    queries route to code — a lexical-overlap artifact: the education
    corpus contains code-flavored text ("build", "API", "implement").
    TF-IDF+SVD task representations are unreliable for unseen-task
    representation in this corpus. *(bounded)*
13. **The aligned-adapter control confirms the moat premise.** With an
    education adapter in the pool (education seen by profiler and pool),
    routing is 14/14 perfect and the loss gap is the lowest of all arms
    (+0.04%). The unseen-task failure is *missing coverage*, not broken
    routing — the exact case the moat strategy pre-empts. *(bounded)*
14. **Oracle-best varies per query** ({finance:5, law:5, code:4}): with
    three adapters, "best available" for an unseen domain is
    near-arbitrary; cross-domain competence differences are sub-noise at
    this scale. This is why the original unseen result was reported open
    rather than as a failure. *(bounded)*

## D. Eight-adapter pool (`experiments/eight_adapter_space.py`)

15. **Pool scaling holds: domain-level accuracy unchanged at 8
    adapters** (96.4%, 54/56) with two adapters per domain trained on
    disjoint data. *(bounded)*
16. **Profile separation is bounded by adapter diversity, not router
    math.** Near-duplicate adapters produce near-identical profiles (min
    pairwise cosine 0.996) — yet routing still works because query
    profiles dominate the cosine decision. *(bounded)*
17. **The router shows consistent within-domain variant preference** (code
    0A/13B, education 16A/0B, finance 0A/14B, law 13A/0B): tiny profile
    differences create consistent choices. Both variants are correct, so
    domain-level accuracy is the meaningful metric. *(bounded)*
18. **Swap isolation scales**: 0/56 routing flips on other domains with
    8 adapters in the pool. *(proven)*

## E. Corpus and calibration (`scripts/build_moat_corpus.py`,
   `scripts/moat_calibration_trial.py`)

19. **Corpus growth: 181 → 408 → 664 examples**; boundaries 17 → 48 →
    64; calibration share raised 70/15/15 → 60/25/15 (n = 32 → 80 →
    180). *(proven)*
20. **Small-n p99 degeneracy resolved by growth.** Brick 2's clean-only
    p99 threshold was 0.30 with 4.76% false-capture; brick 3's is 0.84
    with 0.00% — the degenerate-threshold failure mode from §8 is gone.
    *(proven)*
21. **The boundary set's contribution now shows cleanly at p95**:
    threshold 0.13 → 0.60, false-capture 2.90% → 0.00%, recall 100% in
    both arms — the §8 printer property, reproduced on the corpus.
    *(proven)*
22. **Addition flips drop to 0 at brick 3 size** — the jointly-retrained
    profiler no longer flips base-domain inputs; profiler stability
    improves with data volume. *(bounded)*
23. **Boundary hardness is invariant across bricks**: TF-IDF classifier
    scores 0.0% on boundary test at every brick size — the corpus's
    boundary examples are genuinely dual-domain at scale. *(proven)*

## F. Strategic

24. **The lorouter angle: profile vectors are adapter metadata.**
    Adapter selection becomes a data operation (calibration against the
    moat) instead of a training operation (learning a router) — the moat
    corpus is the adapter-profile factory, and profiles are versionable,
    auditable artifacts. *(design claim, supported by 1, 5, 13)*
25. **First direct verification of the moat premise at the routing
    level**: the aligned-adapter control (13) shows the mechanism's
    ceiling is adapter coverage, not routing quality — the exact
    trade-off the parent suite proved synthetically, now shown with real
    LoRA adapters and real text. *(bounded)*

---

*Open items, stated as plainly: unseen-task quality at scale (needs more
adapters + a real unseen task with aligned adapters), semantic embeddings
(lexical-only features throughout), serving-layer integration (vLLM/LoRAX),
latency/memory measurement, and generation-quality evaluation. The corpus
is synthetic-template English — no human review pass, no Swahili/sheng.*
