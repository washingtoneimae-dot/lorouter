# fuzzy-octo-couscous

Profile-Routed Modular MoE, reproduction suite. Five CPU-only scripts and one
document that together pin down what this architecture can and cannot
guarantee. No GPU, no network, no downloads.

## What this is

Standard MoE routes inputs through a learned gate trained jointly with the
experts. This project replaces the gate with cosine similarity between
calibrated profiles: each expert gets a measured competence vector, the
profiler maps an input into the same space, and the router picks the closest
experts. The question this repo answers: what do you gain, what do you lose,
and exactly where does it break?

The short version, with the evidence in TECHNICAL.md:

- Replacing one expert's function never perturbs the others. Collateral
  change is exactly 0.00% in every run, and the reason is structural, not
  lucky: other experts' profiles cannot depend on the swapped expert.
- Adding a genuinely new domain breaks the naive approach (retrain
  everything) through catastrophic forgetting, and the breakage is measured,
  not hand-waved. A gated one-vs-rest fix recovers most of it. The remainder
  is a Bayes-error property of the data, not a capacity gap: a 326x larger
  gate does not help.
- Adding several domains at once compounds false-capture rates. Three gates
  at 1% each fire at 3.00% combined. Bonferroni correction restores the rate
  and costs real recall. Budget it before you build.
- The flip mechanism replicates on real text (TF-IDF + SVD; semantic
  embeddings were not tested, see TECHNICAL.md section 8).

## Run it

    python3 -m venv .venv
    .venv/bin/pip install scikit-learn numpy
    .venv/bin/python addition_isolation_suite.py
    .venv/bin/python capacity_ablation.py
    .venv/bin/python multi_dimension_compounding.py
    .venv/bin/python text_validation.py

Each script prints its own findings and finishes in under a minute on a
laptop. The scripts are the source of truth. The numbers in TECHNICAL.md are
this repository's canonical run: short experiments (capacity ablation,
calibration prototype) reproduce to the decimal, longer pipelines reproduce
the pattern but not always the digits. TECHNICAL.md section 12 says which is
which, and section 11 lists the honest limits.

## Layout

    TECHNICAL.md                    the actual document. read this first.
    shared_data.py                  data generator, experts, profiler, router
    addition_isolation_suite.py     swap isolation, addition flips, gated
                                    fix, multi-seed stability
    capacity_ablation.py            small gate vs 326x larger gate
    multi_dimension_compounding.py  simultaneous additions, Bonferroni
    text_validation.py              real-text flips, calibration prototype
    performance_data.xlsx           canonical run tables, including the
                                    mitigation comparison from the original
                                    repository

## Honest limits

- No semantic embedding model was available when the text tests were built.
  The real-text validation is lexical only.
- The comparison against a learned router used an undertrained baseline. No
  superiority claim rests on it.
- Multi-adapter batched serving is prior art (Punica, S-LoRA), not this
  project's contribution.
- Exact digits for the multi-stage experiments vary run to run. The
  conclusions do not. Quote the decimals only where section 12 marks them as
  stable.

## What this is not

Not a framework, not a library, not production routing, not a benchmark
against other routers. It is a measurement of one design point: what a
training-free, profile-based router can and cannot guarantee, with the
failure modes written down before someone else finds them.

## Related

The original repository, profile-moe, holds the older experiments
(versioning demo, adaptive-temperature work) that this suite references but
does not reproduce.

MIT license.
