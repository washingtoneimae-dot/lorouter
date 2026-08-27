# FINDINGS — lorouter, complete record

Every major finding from the lorouter line of work (branch `lorouter`,
canonical runs 2026-08-05/06). Each finding names its source script; the
scripts are the source of truth. For the complete file map (where every
file is and what it is), see README.md → "Repository layout". Evidence
tiers: **proven** (reproducible by script), **bounded** (proven under
stated conditions), **open** (hypothesis, not yet validated).

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

## G. Gap-closure experiments (2026-08-06, canonical run)

26. **Multi-seed stability: 96.4% across seeds 42, 7, 2026** — the
    real-LoRA routing result is not a seed artifact. Diagonal dominance
    1/4 per seed (consistent weak differentiation, base-model priors).
    *(proven, `experiments/real_lora_multiseed.py`)*
27. **Model-size invariance: 96.4% on SmolLM2-360M, identical to 135M**
    (2.7x larger base, same pipeline; diagonal dominance 2/4). The
    routing result is not a small-model artifact. *(bounded,
    `experiments/real_lora_360m.py`)*
28. **Semantic embeddings improve routing and flip the centroid
    ranking.** With bge-small-en-v1.5 features: profile routing 98.2%
    (up from 96.4% with TF-IDF+SVD), tying the learned router (98.2%)
    and BEATING embedding-centroid routing (96.4%) — the reverse of the
    lexical-feature ranking. Calibrated competence profiles outperform
    raw embedding centroids once features are semantic. *(proven,
    `experiments/semantic_embeddings.py`)*
29. **The F12 artifact is a lexical-feature artifact.** With semantic
    task embeddings, education exemplars route to finance (semantically
    sensible) instead of code; per-query distribution {finance 6, code
    3, law 5} keeps the finance-leading pattern. *(proven,
    `experiments/semantic_embeddings.py`)*
30. **Selection-policy latency is sub-millisecond at any realistic pool
    size.** Query profiling 1.14 ms; cosine selection 17 µs (4 adapters)
    → 380 µs (10,000 adapters); end-to-end policy 2.78 ms at 10k.
    Adapter-switch/load costs in serving systems are ms-scale — the
    policy pays microseconds to avoid a full regenerate on wrong
    selection. Full-stack integration latency (vLLM/LoRAX hook) remains
    the unmeasured half of the spike. *(bounded,
    `experiments/latency_spike.py`)*
31. **Aligned-adapter generation quality: the routed adapter produces
    the best output.** On held-out education queries, the education
    adapter's generations score 0.5932 semantic similarity to the
    reference answer vs 0.5887 for the best incumbent (+0.45pp; random
    expectation 0.5862), and the qualitative samples show the only
    domain-plausible continuation. Margin is small at 135M scale —
    direction verified, magnitude bounded by base-model quality.
    *(bounded, `experiments/generation_quality.py`)*
    NOTE: the real-LoRA experiments run on the loader's default corpus
    (brick 2); brick 3 powers the corpus and calibration-trial line.

## H. Adapter-pool scaling (2026-08-06, canonical run)

Note: this run used the pre-v3 brick 3 (664 examples, 4 domains); the
file has since been superseded by the user's v3 (F37–F41). A scaling
re-run on v3 is a listed follow-up.

Grounded in the measured real-LoRA profile shapes (F5 loss matrix),
variant profiles = base + Gaussian noise at scale σ; 5 seeds; 92 brick-3
test queries. Full table in `experiments/results/pool_scaling.csv`.

32. **Pool-size invariance at zero profile noise**: accuracy FLAT at
    96.74% from N=8 to N=1024 identical variants — duplicating adapter
    variants costs nothing. *(proven, `experiments/adapter_pool_scaling.py`)*
33. **The accuracy curve is U-shaped under profile noise, not monotone.**
    Low variant counts: noise dominates (N=8, σ≥0.05 → 37–40%). Mid
    counts: more same-domain variants HELP — more lottery tickets for the
    true domain (N=128, σ=0.05 → 77.4%). Extreme counts: the wrong-domain
    max-of-N effect decays accuracy again (N=1024, σ=0.20 → 30.4%).
    Crossover near N=128–512 for σ≥0.05. *(proven)*
34. **Bonferroni-style compounding does NOT transfer to adapter pools.**
    Measured false capture is far below naive independence at every
    point (σ=0.05, N=128: 22.6% measured vs 100% naive) because variants
    share the domain base profile — correlated errors. v2's
    independent-gate law is the WORST case, not the adapter-pool case.
    *(proven)*
35. **Swap isolation scales**: 0.00% flips on other domains at N=128 and
    N=512 in noisy pools. *(proven)*
36. **Profile separation**: min cosine collapses with noise (≈0 at
    N=1024/σ=0.10; −0.997 at σ=0.20) while mean cosine stays 0.63–0.89 —
    the domain base profile dominates the mean; extreme-value noise
    lives in the min. *(proven)*

## I. v3 corpus (user-built, 2026-08-06, canonical run)

The corpus was grown by the user from 664 to **3,010 examples**
(`corpus/moat_brick3.jsonl`, version tag `v3-moat-brick-3-6domain-psychology`):
six clean domains (finance/law/code/education 500 each, medicine 384,
psychology 415, code 491 — the sub-500 counts are template-space
exhaustion, not curation) + **220 hand-written boundaries** across 36
cross-domain pairings (including 4- and 6-way examples). Splits
1673/807/530.

37. **Integrity at scale**: 3,010 unique ids and texts, zero
    calibration/test leakage into train, no empty rows, schema-compliant.
    *(proven, verified against the file)*
38. **Boundary hardness holds at 4.5x scale with 6 domains**: TF-IDF
    classifier 94.0% on clean test (420 examples), 0.0% on boundary test
    (110 examples). *(proven)*
39. **Calibration on v3 (p99)**: clean-only threshold 0.0448 →
    clean+boundary 0.7638; false-capture 1.79% → 0.00%; recall 97.3% →
    93.3% — the §8 property replicates. At p95 the effect is real but
    weaker (false-capture 4.02% → 1.34%) — the six-domain space is
    harder than the four-domain one. *(proven,
    `scripts/moat_calibration_trial.py`)*
40. **Escalation sensitivity improves with boundary richness**: 6 of 34
    education-involving boundaries (18%) are now flagged by the gate
    (was 1/17 = 6% at the old size) — the gate flags more genuine
    ambiguity as calibration data improves. *(proven)*
41. **Addition flips at v3 size: 7** (was 0 at 664 examples) — the
    six-domain space presses harder on the profiler; small, real, and
    the §6.3 gated fix is exactly its remedy. *(proven)*
42. **Six-domain real-LoRA scale-up: 95.7% routing with 6/6 diagonal
    dominance.** Six adapters (135M, rank 8, 10 epochs, 230–300 QA
    pairs/domain from v3): routing 95.7% (402/420; per-domain
    93.3–98.4%; random floor 16.7%) — the 4-domain 96.4% holds within
    ~1 point as the pool and domain space grow. The loss matrix is now
    fully diagonal (every adapter lowest on its own domain, e.g.
    medicine 0.55 on medicine vs 2.1+ elsewhere). **This revises the
    differentiation bound in F7/F8**: it was a data-volume bound, not a
    base-model-priors bound — at ~5x training data per adapter, the
    code-prior effect disappears. *(bounded,
    `experiments/real_lora_six_domain.py`)*

---

## J. External data-moat pipeline (2026-08-27, canonical run)

A hosted data-optimization + training pipeline (Adaption Adaptive Data +
AutoScientist on NVIDIA GPUs; Featherless for reference-answer generation)
applied to the moat corpus. Full reproducible record:
`experiments/adaption/STATUS.md` and `experiments/adaption/adaption_pipeline.py`.

43. **v3 corpus adapted externally: D→B (+87.5%)**. 1,117 enhanced QA pairs
    (platform dedup/filter from 1,673 train rows) across all 6 domains,
    committed as `corpus/moat_brick4_adapted.csv`. The `enhanced_completion`
    column carries model-generated, domain-grounded answers with Kenyan
    context — the training-signal upgrade over the synthetic templates.
    *(proven, file committed and verified)*
44. **Joint-retrain baseline at 0.8B scale trained on enhanced pairs**:
    Qwen3.5-0.8B LoRA (r16/α32), 1 epoch/23 steps on the 1,117 enhanced
    pairs, final loss 1.418 / eval loss 1.406, platform best win rate 0.4095
    (target 0.7; 3/3 iterations). This is the modernized joint-retrain
    counterfactual — successor of the `moat_profile_addition.py` joint
    model. *(proven checkpoint: `experiments/adaption/checkpoint/
    joint-baseline/`; the routing comparison is the pending step)*
45. **Three external real-world domains imported and adapted**: 5G-NR QA
    (telecom, 45 enhanced rows after platform filter), agriculture QA (500),
    personal-finance-africa (fintech, 200) — HuggingFace → Adaption, held
    under `experiments/adaption/adapted_data/`. Licenses: 5G CC-BY-NC
    (non-commercial), agriculture Apache-2.0, fintech CC-BY-4.0. *(proven;
    telecom thinness being addressed via generated completions)*
46. **Learned-router corpus-level preview: 96.4% (405/420)**. LogReg on
    TF-IDF, trained on v3 train split only, evaluated on the held-out test
    (per-domain 92.0–98.7%; random floor 17.9%) — reproduces the F5 tie at
    corpus level and sits at/above the F42 profile-routing 95.7%. The
    real-LoRA learned-router arm is part of the pending 9-domain benchmark.
    *(proven at corpus level, `experiments/adaption/learned_router_preview.py`)*

---

*Open items, stated as plainly: unseen-task quality at SCALE (the aligned-
adapter case is verified — F13, F31 — but not at LORAUTER's 1000+-adapter
scale); the vLLM/LoRAX integration hook itself (policy latency is
measured, F30, but not inside a serving stack); generation-quality at a
stronger base model (the 135M margin is small, F31); the corpus remains
synthetic-template English — no human review pass, no Swahili/sheng; and
the 9-domain pool benchmark (F43–F46's pending routing comparison).*
