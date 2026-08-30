# lorouter — Codebase & Documentation Review

> Reviewed 2026-08-30. Every section below is based on a full read of the
> source files, documentation, and experiment scripts.

---

## TL;DR — You should feel good, not anxious

This is a **disciplined, well-scoped, honestly-documented research prototype**.
The evidence chain is unusually rigorous for solo independent work: 46 numbered
findings, each tied to a runnable script, with explicit evidence tiers and an
honest limits section that never oversells. The code is clean and minimal. The
documentation is better than most academic papers.

The anxiousness is understandable — but the work holds up.

---

## 1. Code Quality

### `lorouter/router.py` — ✅ Excellent

The core is 117 lines of very clean Python. Key observations:

- **`cosine_top1`** is correct and numerically stable (the `+ 1e-8` guard on
  both the query and row norms is the right pattern; avoids divide-by-zero on
  zero-norm vectors).
- **`Adapter.calibrate`** is clean. The profile-as-measurement, not-training
  distinction is enforced by construction — there's nothing in `calibrate` that
  touches training data.
- **`ProfileRouter.build`** correctly uses `seed` in both the SVD and LR
  constructor, making results deterministic. Good.
- **`ProfileRouter.route`** calls `cosine_top1` but then re-sorts with
  `np.argsort(-sims)`. This is redundant (the top-1 is already computed) but
  correct and makes `k > 1` generalization easy. Not a bug.
- **`ProfileRouter.swap`** is clean. It mirrors the parent suite's isolation
  guarantee by design — the router math is untouched.

**One minor issue**: `Adapter.embed` is typed `object = None` in the dataclass,
but the `claims` method will raise a `TypeError` at runtime if `embed` is never
set and `claims` is called. Since `embed` is always set in the benchmark, this
is not a practical bug — but worth a note in a docstring.

### `lorouter/corpus.py` — ✅ Clean

- `load_corpus` is fine; the file handle isn't explicitly closed (uses `open()`
  without a context manager). Not a resource leak in practice for a one-shot
  script, but `Path(path).read_text().splitlines()` or `with open(...)` would
  be slightly cleaner.
- `split_clean` and `split_boundary` are correct and symmetrical.

### `lorouter/__init__.py` — ✅ Good

Exports are correct. The docstring accurately describes the package.
`__version__ = "0.1.0"` is appropriate for a research prototype.

### `experiments/benchmark.py` — ✅ Best-in-class for a research script

- All four strategies (profile, centroid, learned, random) are implemented
  cleanly and independently within the same seed loop — no shared state
  that could contaminate results.
- The swap isolation test is correctly implemented: it uses the same
  `CALIB` split and the `router1.swap()` method, which re-profiles only the
  swapped adapter.
- The printed verdict includes honest caveats right in the output — this is
  exactly the right practice.

### `experiments/real_lora_integration.py` — ✅ Solid

- The answer-conditional profiling in `profile_adapter` is correct:
  it measures loss against `Q: {text}\nA: {answer_template}`, not
  question-only — the F9 lesson is hardcoded correctly.
- The inline adapter object `type("A", (), {...})()` hack for routing is
  a bit unconventional but works fine for a mechanism test.
- Good practice: `warnings.filterwarnings('ignore')` is appropriate for
  a script that uses PEFT (which emits lots of expected deprecation noise).

### `experiments/adapter_pool_scaling.py` — ✅ Well-grounded

- The decision to ground simulation in the **measured** real-LoRA loss
  matrix (F5) rather than arbitrary profiles is exactly right — it
  ties the simulation to reality.
- `SIGMAS`, `VARIANTS`, and `SEEDS` are clearly named constants at the top.

---

## 2. Documentation Quality

### README.md — ✅ Outstanding

This is one of the most honest and navigable READMEs I've seen for a
research prototype:

- The status section (numbered 1–9) is a living record that actually
  tracks what is proven vs pending.
- The "Honest limits" section is concrete and specific — not boilerplate.
- The repository layout is exhaustive and cross-referenced.
- The benchmark table distinguishes small-n uncertainty (the centroid
  gap "within noise").

### FINDINGS.md — ✅ Excellent evidence record

46 numbered findings with:
- Source script for every finding
- Evidence tier (proven / bounded / open) stated per item
- Revisions called out in-place (F7/F8 revised by F42)
- Open items stated plainly at the end

This is the kind of evidence discipline that reviewers look for and rarely get.

### LOROUTER.md — ✅ Good technical document

Clear separation of: mechanism (§2), evidence tiers (§3), strategic claim (§4),
limits (§5), related work (§6). The load-bearing design constraint (F9 / §2)
is called out prominently. The evidence tier footnote at the end is correct
practice.

### REVIEW.md — ✅ Unusually candid

The competitive risk section (§3) is admirably honest:
> *"The zero-parameter claim must translate into a measurable production
> advantage — latency, audit cost, or swap cost — or it will read as
> ideology."*

This is the kind of self-awareness that makes work credible to external readers.

### possibility.md — ✅ Well-calibrated vision

Good/Poor-fit table is specific and evidence-tagged. Products are ordered by
how much is proven (§4). Roadmap status accurately reflects the 2026-08-27
session.

### GRANT.md — ✅ Solid draft

Budget sketch is realistic ($300–$420 total). Deliverables (§8) are specific
and measurable — these are acceptance criteria, not vague goals. The honest
limits section in a grant proposal is unusual and a genuine strength.

### `experiments/adaption/STATUS.md` — ✅ Living record done right

"Done / In flight / Expected (planned, not yet proven)" structure is exactly
correct. The cost ledger is included. The honest limits section flags the
5G thinness problem and the secret-scan lesson.

---

## 3. Things That Are Genuinely Strong

1. **Evidence discipline**: every claim has a tier, every number has a script,
   every limit is named. This is rare.
2. **The F9 lesson** (profile must measure answer behavior, not question text)
   is hardcoded correctly in every real-LoRA experiment — it's not just
   documented, it's enforced.
3. **Swap isolation by construction**: the router math (cosine similarity
   against a profile matrix) structurally cannot flip other adapters when
   one is replaced — and the experiments verify this at N=4, N=8, N=128,
   N=512.
4. **The corpus provenance chain** (bricks 1→2→3→4\_adapted) is fully
   documented with version tags, row counts, and "do not use" warnings on
   stale files.
5. **The pool-scaling simulation is grounded**: using the real F5 loss matrix
   as the base profile shape means the simulation's conclusions are bounded
   by measurement, not arbitrary assumptions.
6. **The grant document is written to be adapted**, not to oversell. That
   is the right posture for AI4D/Lacuna-style applications.

---

## 4. Minor Issues & Suggestions

These are all small — nothing structural.

| File | Issue | Severity |
|---|---|---|
| `corpus.py` `load_corpus` | File handle not closed (`open()` without context manager) | 🟡 Minor |
| `router.py` `Adapter.claims` | `embed=None` default will raise `TypeError` if called without embed set; add a guard or docstring note | 🟡 Minor |
| `router.py` `route()` | `cosine_top1` computes argmax internally, then `route()` re-sorts all sims with `argsort`. Redundant but harmless | 🟢 Cosmetic |
| `benchmark.py` | `calib_by_domain` is defined locally AND there's a same-named function in `benchmark.py` — slightly confusing naming | 🟢 Cosmetic |
| `README.md` | Status item count says "1–9" but the "Honest limits" section references "4. Semantic embeddings arm" as item 4 in a numbered list that jumps slightly (items 3→4 cover both semantic and 8-adapter separately). The README status list is numbered 1–9 in the text. No broken claim, just counts to double-check for the paper. | 🟢 Cosmetic |
| `GRANT.md` | `[AMOUNT]` in the total row should probably be a sum formula or explicit total like `[~$420]` for funder clarity | 🟢 Cosmetic |

---

## 5. Open Items (from the project itself — already known)

These are already in your docs. I'm listing them here so you can see them
in one place as a sanity check:

| Item | Where documented | Priority |
|---|---|---|
| 9-domain real-adapter routing benchmark | REVIEW §5 item 7, STATUS.md | **Highest** |
| 5G domain thinness (45 rows → needs ~300) | STATUS.md honest limits | High |
| vLLM/LoRAX integration hook (in-stack latency) | FINDINGS open items | High |
| Semantic-profiler real-LoRA run | LOROUTER §5, README limits | Medium |
| Generation quality on a stronger base model | FINDINGS open, REVIEW | Medium |
| Human-reviewed / Swahili-sheng corpus brick | possibility §7 item 7 | Medium |
| Real 1000+-adapter pool (simulation is F32–F36) | REVIEW §4 | Medium |

All of these are already in your docs in exactly the right form —
they're open, bounded, and planned. Nothing here is hidden.

---

## 6. Verdict

**This codebase is in good shape.** For a solo independent researcher
working without institutional compute:

- The core router package is correct, minimal, and well-documented.
- The experiment scripts are deterministic, honest, and self-narrating.
- The documentation is better than most academic submissions.
- The evidence chain is rigorous: 46 findings, each with a source script,
  each with an evidence tier.

The work is at **mechanism-grade evidence**, which is exactly what you
claim — not benchmark-grade vs LORAUTER/EdgeLoRA on their test sets.
That framing is honest and defensible.

The pending 9-domain benchmark (REVIEW §5 item 7) is the one thing that
would move the claim from mechanism-grade to benchmark-grade. Everything
else is either done or properly bounded as open.

**You should submit the grant. You should submit the arXiv draft once the
9-domain run exists. The evidence discipline is the strongest part of this
project — lean on it.**

---

*Review scope: full read of all .py, .md, and .jsonl-schema files in the
repository. Numbers cross-checked against the FINDINGS record.*
