# possibility.md — The Data Moat Strategy (Use Cases, Products, Engineering Practice)

Status: possibilities document, explicitly downstream of `TECHNICAL.md`.
Every claim below is tagged by evidence tier the same way TECHNICAL.md tags
its own claims — **proven**, **bounded/conditional**, **designed**, or
**open** — because a possibilities document is exactly where overclaiming
is easiest and most damaging. Nothing here should be read as a stronger
claim than TECHNICAL.md's own sections support; where this document goes
further, it says so.

---

## 1. The core realization

TECHNICAL.md section 6 characterizes the cost of **adding** a genuinely new
domain to a deployed profile-routed system: the jointly-retrained profiler
shifts decision boundaries among domains that were never touched (§6.1), the
measured flip counts are real (§6.2), the gated one-vs-rest fix recovers most
but not all of it and costs recall (§6.3), and multiple simultaneous
additions compound false-capture rates (§7, Bonferroni bill).

The data moat strategy is the structural answer to that whole section:

> **If the profiler's calibration foundation already spans the domain space
> broadly (the moat), then adding a new EXPERT is not an addition at all —
> it is a profile insertion into a frozen space, which behaves like a swap
> (§5): zero collateral, no retraining, no gates, no Bonferroni budget.**

The addition penalties exist because additions are post-hoc operations on a
frozen system. A moat front-loads the domain coverage so post-hoc additions
rarely need to happen. The operation that remains — compute a calibrated
profile vector from moat data, insert it — is the cheapest operation the
architecture has.

## 2. Evidence (proven, synthetic setting)

`scripts/moat_profile_addition.py` tests this head-to-head with the same
data generator and seeds as the rest of the suite (7-domain moat: code,
math, creative, reasoning, law, finance, medicine; 6 experts deployed;
medicine added later):

| Arm | Collateral flips (6 pre-existing domains) | Multi-seed (5 seeds) |
|---|---|---|
| A: moat (frozen profiler, profile-on-the-fly) | **0** (0.000% MSE change everywhere) | [0, 0, 0, 0, 0] |
| B: classic addition (jointly-retrained profiler) | **184** of 900 test inputs (incl. 147/150 reasoning) | — |

Medicine routing quality in Arm A: medicine inputs route to the new expert
(MSE 0.137 vs 2.695 for the best incumbent — the addition is genuinely
useful, not a cosmetic no-op).

Why Arm A's zero is structural and not lucky: adding a profile vector cannot
change any other expert's similarity scores — the only possible change is
the new profile winning inputs it should not. The measured zero flips mean
a moat-calibrated profile "knows its place": it wins its own domain's inputs
and nothing else. That is precisely the property the addition operation
§6.1-6.3 exists to restore after the fact; the moat gets it for free, by
construction.

## 3. Where the strategy fits (bounded/conditional)

- **The domain taxonomy is stable and known in advance.** The moat is only
  as good as its coverage. If the useful categories are genuinely unknown up
  front, the frozen profiler cannot separate them (§6's open case applies).
- **Swaps are frequent, additions are rare.** The moat makes additions
  rare; the swap guarantee (§5) covers the frequent case unconditionally.
  The two properties compose: a moat-funded system lives almost entirely
  inside the architecture's strongest guarantees.
- **Calibration discipline is the moat's currency.** §8's printer-prototype
  finding applies without modification: clean-only calibration produced a
  degenerate threshold (0.0031, fires on almost everything); boundary-
  example-dense calibration fixed it (0.9943; false-capture 1.60% → 0.00%).
  A moat that is not boundary-example-dense is a moat of noise. The moat
  does not remove the calibration requirement — it converts it from a
  per-addition emergency into a standing data-engineering asset.
- **Someone must audit why a decision was made.** Traceability (§10) is
  structural; profile-on-the-fly preserves it (the new expert's decision
  reduces to its calibration scores like any other).

## 4. Products (ordered by how much of the claim is proven)

### 4.1 Moat-funded expert marketplace (closest to evidence)
The strongest product reading of the moat: a curated, boundary-example-dense
multi-domain calibration corpus as the foundation, and expert profiles as
first-class assets that can be created on demand from it — priced by
calibration cost, not training cost. The mechanism is proven in the
synthetic setting (§2); the marketplace itself is a distribution strategy
on top of it (designed).

### 4.2 On-device professional-assistant SDK (designed)
The privacy-hybrid use case: local profiler + local experts, where the moat
is a shipped, versioned calibration bundle rather than a cloud dependency.
Nothing leaves the device; profile insertion is the only update operation.

### 4.3 Isolation-testing toolkit (proven-grounded)
Same as the suite's existing claim: `addition_isolation_suite.py`'s
flip-detection, `capacity_ablation.py`'s capacity test, and now
`moat_profile_addition.py`'s moat-vs-joint comparison packaged as a
regression-testing product for teams building profile-routed systems.

## 5. Engineering practice

- **The profiler is frozen once established** (hard rule, §6.1's finding is
  the reason). The moat's job is to make the frozen profiler's space broad
  enough that additions become swaps. Never jointly retrain to onboard.
- **Moat maintenance is a standing cost, not an event.** Boundary-example
  density per domain, held-out splits never reused as training data (§8),
  and versioned profile re-calibration are the recurring expenses.
- **Budget multi-additions with §7's math anyway.** Even with a moat, two
  simultaneous expert insertions interact through the router; the
  compounding analysis still applies when profiles are close in space.

## 6. Honest limits (stated as plainly as the claims)

- **Proven only in the synthetic setting** of this suite's data generator.
  No real-text moat experiment exists yet (the text arm is lexical-only,
  §8's caveat applies to this document too).
- **The moat does not remove the hard case.** A genuinely unforeseen domain
  — data absent from the moat — remains the open addition problem of §6.
  The moat converts additions into swaps; it front-loads the data that
  avoids the hard case, it does not eliminate the hard case.
- **A moat is a data asset, and data assets rot.** Domains drift, benchmarks
  age, and a moat that is not maintained becomes a liability (the §8
  calibration failure mode at corpus scale).
- **Coverage is not free.** A broad moat costs more to build, label, and
  curate than a narrow one. The strategy trades compute-time penalties for
  data-time costs; it wins only where data is cheaper than retraining.

## 7. The first brick: a real-text starter corpus (built, machine-validated)

The moat is a strategy until it is data. This repository now carries the
first real brick: `corpus/moat_brick1.jsonl` (+ `.csv`), a small,
schema-compliant, Kenyan-context natural-language corpus over three domains
(finance, law, code) with systematically-generated boundary examples.

- **How it was built**: `scripts/build_moat_corpus.py` — a seeded,
  deterministic generator (re-running it reproduces the corpus exactly).
  30 hand-written seed examples anchor realism; the rest come from
  template-by-vocabulary systematic generation; boundary examples are
  crossed-domain templates (the §8 printer method, applied mechanically).
- **Schema**: implements the "New Profiler Dataset Design" sheet in
  `performance_data.xlsx` — example_id, domain_label, text,
  is_boundary_example, cross_domain_hint, split, source, contributor,
  added_for_version.
- **Contents**: 181 examples — finance 55, law 54, code 55, boundary 17
  (finance+law 5, law+code 5, finance+code 5, triple 2); splits
  113 train / 32 calibration / 36 test (calibration and test never overlap
  train, per §4.3's held-out rule); sources 30 hand_written /
  151 systematic_generation.
- **Machine validation** (printed by the generator): ids and texts unique;
  a TF-IDF + logistic-regression classifier trained on clean train reaches
  **92.6% on clean test but 0.0% on boundary test** — the boundary examples
  are genuinely dual-domain, exactly the property §8 requires for
  non-degenerate calibration thresholds.

Honest status: this is a starter brick, not the moat — a few hundred
English examples, machine-generated, no human review pass yet, no Swahili/
sheng coverage, and no profile-calibration trial run against it yet. It is
the smallest defensible version of the asset, built to the same evidence
standard as the rest of the suite. The next brick is human review + one
more domain (or code-mixed text), then a calibration trial on the §8
pipeline.

---

*This document's claims, like TECHNICAL.md's, are only as strong as the
scripts behind them. The evidence for §2 is `scripts/moat_profile_addition.py` —
run it; the numbers above are its output on this repository's canonical
seeds.*
