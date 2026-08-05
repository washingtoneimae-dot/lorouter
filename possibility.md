# possibility.md — Use Cases, Products, Engineering Practice, Diagnosis

Status: vision document, explicitly downstream of `TECHNICAL.md`. Every
claim below is tagged by evidence tier the same way TECHNICAL.md tags its
own claims — **proven**, **bounded/conditional**, **designed**, or **open**
— because a possibilities document is exactly where overclaiming is easiest
and most damaging. Nothing here should be read as a stronger claim than
TECHNICAL.md's own sections support; where this document goes further, it
says so.

---

## 1. Where This Architecture Actually Fits

Not a universal MoE replacement (§3). It trades a learned gate's ability to
discover subtle routing signal at scale for a provable, calibratable
isolation guarantee and full decision traceability. That trade is worth
making in some situations and not others.

**Good fit — each condition tied to a proven or bounded property, not
aspiration:**

- **The domain taxonomy is known and stable, not something you need the
  system to discover.** Profile routing is bounded by what the calibration
  benchmark measures (§3; §11, limitation 5). If the useful categories are
  genuinely unknown in advance, a learned gate has a real advantage this
  architecture doesn't.
- **Swaps are frequent, additions are rare.** §5's swap isolation (0.00%
  collateral change, every run) is unconditional. §6's addition guarantee
  is conditional and costs real recall when pushed (§7). A deployment that
  mostly upgrades existing experts and rarely onboards genuinely new
  domains sits inside the architecture's strongest guarantee almost all the
  time — and a data moat (§3) is the structural way to keep additions rare.
- **Someone needs to audit *why* a decision was made, not just *what* it
  was.** Traceability (§10) is structural, not a logging feature bolted on
  afterward — a decision reduces to a specific calibration score by
  construction.
- **On-device or privacy-constrained deployment**, where a small profiler
  and small per-domain adapters can run locally without exporting sensitive
  queries (§2.3 — designed, not tested in this document).

**Poor fit — stated as plainly as the good-fit list:**

- Open-ended domains where the useful specialization boundaries aren't
  known ahead of time and would need to emerge from data.
- Situations where raw accuracy at scale is the only metric that matters and
  infrastructure cost is not a constraint — a well-trained learned gate has
  no fundamental ceiling here that this architecture doesn't also have, and
  the learned gate gets to discover signal this architecture must be told
  about explicitly.
- Anyone unwilling to invest in calibration data. §8 shows directly what
  happens without it: a 300-example hand-written calibration set produced a
  threshold that fires on almost everything. This architecture's guarantees
  are only as good as the calibration discipline behind them (§4; §9's
  per-domain-worst-case recommendation).

---

## 2. Use Cases

### 2.1 MoE R&D and low-risk experimentation tooling (proven-grounded)
Swap isolation plus traceability means a team building MoE systems can add,
remove, or upgrade experts and *know*, not hope, that untouched experts are
unaffected — and when something does break, trace it to the specific
calibration score responsible, the way this whole document set traces every
finding to a script and a line of reasoning. This is the single most
evidence-backed use case in this document; everything in it maps to a
**proven** result.

### 2.2 Regulated professional-domain assistants (medicine, law, engineering) (bounded)
Fields where "the system made a decision and nobody can explain why" is a
compliance problem, not just a UX complaint. For inputs that are
irreducibly ambiguous between two domains, the honest system behavior is
*not* to guess confidently — it's to flag the ambiguity for human review.
In medicine or law, "escalate instead of guess" is correct behavior, not a
limitation to apologize for. This reframes §6's hardest finding as a safety
feature specific to these fields. A data moat strengthens this use case
directly: boundary-example-dense calibration is what makes the
"genuinely ambiguous" cases detectable as ambiguous instead of confidently
miscategorized (§3.3; §7's brick is built on exactly this property).

### 2.3 Privacy-focused hybrid local/cloud deployment (designed)
Because experts are small, swappable units rather than full models, a
domain-specific behavior specialist can live entirely on a practitioner's
device, with only the profiler deciding domain locally and never sending
raw content elsewhere unless genuinely necessary. **Design requirement, not
optional extra:** an adapter is good at *behavior* — formatting a
differential diagnosis, reasoning like a structural engineer — and weak at
*storing* the facts a real practitioner needs (§3 names this trade-off
explicitly). The viable version of this use case pairs the local expert
with a local retrieval index of real reference material, not a claim that
the adapter "knows" the field. The moat fits here as a shipped, versioned
calibration bundle rather than a cloud dependency (§4.3).

### 2.4 Enterprise internal-tooling onboarding (bounded)
Swapping in an improved internal specialist (say, a better company-policy
expert) is the cheap, safe, proven operation (§5). Onboarding a genuinely
new internal domain is the rare, expensive, gated operation (§6) — and now
has a quantified cost model (§7) for budgeting it before, not after,
someone tries to add the fifth or sixth domain at once. The moat converts
most of these "rare expensive additions" into profile insertions (§3),
leaving the budget model for the genuinely uncovered remainder.

### 2.5 Diagnostic and pedagogical tooling (proven-grounded)
Because the routing mechanism is fully inspectable, the same infrastructure
that verifies this architecture's own claims (the `scripts/` suite) is
itself a legitimate product surface: a tool for teaching, auditing, or
debugging *any* profile-routed system's behavior, not just this one. See §6.

---

## 3. The Data Moat Strategy

### 3.1 The core realization
TECHNICAL.md section 6 characterizes the cost of **adding** a genuinely new
domain to a deployed profile-routed system: the jointly-retrained profiler
shifts decision boundaries among domains that were never touched (§6.1), the
measured flip counts are real (§6.2), the gated one-vs-rest fix recovers
most but not all of it and costs recall (§6.3), and multiple simultaneous
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

### 3.2 Evidence (proven, synthetic setting)
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
§6.1–6.3 exists to restore after the fact; the moat gets it for free, by
construction.

### 3.3 Where the moat fits (bounded/conditional)

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

---

## 4. Products (ordered by how much of the claim is proven)

### 4.1 MoE isolation-testing toolkit (strongest, most proven-grounded)
Package `addition_isolation_suite.py`'s flip-detection methodology,
`capacity_ablation.py`'s capacity-vs-Bayes-error test,
`multi_dimension_compounding.py`'s Bonferroni budgeting, and now
`moat_profile_addition.py`'s moat-vs-joint comparison as a standalone
regression-testing product for teams building any profile-routed or
training-free MoE system — not just this one. This is close to a direct
repackaging of what already exists and is verified; the product work is
mostly interface and generalization, not new research.

### 4.2 Moat-funded expert marketplace / calibration bank (designed; closest to evidence among the moat readings)
The strongest product reading of the moat: a curated, boundary-example-dense
multi-domain calibration corpus as the foundation, and expert profiles as
first-class assets that can be created on demand from it — priced by
calibration cost, not training cost. The mechanism is proven in the
synthetic setting (§3.2); the marketplace itself is a distribution strategy
on top of it. Directly downstream of §9's modular composition proposal,
which is explicitly **not yet validated at scale** — §9's honest caveat
applies without modification: building a comprehensive, well-populated,
boundary-example-dense calibration bank across many domains is a real
undertaking, plausibly a fit for a crowdsourced contribution model — but
that is a distribution strategy, not a substitute for validating the
underlying mechanism first on 2–3 domains. §7's brick is the first step of
that validation.

### 4.3 On-device professional-assistant SDK (designed)
The privacy-hybrid use case (§2.3) productized: local profiler, local
behavior adapters, local retrieval index, optional cloud escalation for
genuinely out-of-scope queries. The moat ships as a versioned calibration
bundle; profile insertion is the only update operation — no retraining, no
model download for a new capability. The RAG pairing is a hard requirement
of the design, stated in the SDK's own documentation, not a footnote
discovered by an unhappy developer later.

### 4.4 Isolation-bounded MoE hosting layer (designed)
**Needs real differentiation to be viable.** Multi-adapter serving at scale
is already solved well by Punica and S-LoRA (§2.2, §13); a hosting product
built on this architecture only earns its place by leading with what those
systems don't characterize — provable, budgetable isolation guarantees
under expert addition and swap — not by re-competing on serving efficiency
alone.

---

## 5. Development and Maintenance for Engineers

### 5.1 Three operation classes, three different budgets
- **Swap** (replace an expert's function, same domain slot): cheap, safe,
  day-to-day. Proven isolation (§5). No special calibration-data investment
  beyond what the new expert itself needs.
- **Profile insertion** (add an expert whose domain is covered by the moat):
  one calibration step against moat data, one array assignment. Measured
  collateral: 0.00% (§3.2). This is the moat-funded replacement for most
  additions.
- **Addition** (onboard a genuinely new domain, absent from the moat):
  rare, and should be budgeted like an infrastructure project, not a
  routine deploy. Requires: a properly-sized, boundary-example-dense
  calibration set (§8 — 300 clean examples produced a degenerate threshold;
  ~600 boundary examples against ~1,200 clean fixed it), an
  independently-trained one-vs-rest gate (§6.3), and — if adding more than
  one domain at once — an explicit false-capture budget split across all
  simultaneous additions (§7), decided before building, not discovered
  after.

### 5.2 Versioning discipline
The base profiler is frozen once established. New domains are added via
independent gates or moat profiles, never by jointly retraining the
profiler across old and new domains together (§6.1's `code#82` finding is
the concrete reason why: joint retraining can shift decision boundaries
among domains that were never touched). This is a hard rule, not a style
preference — it's the precondition the entire addition-isolation guarantee
depends on.

### 5.3 Calibration data as a first-class engineering artifact
Two different datasets, commonly conflated: **expert calibration data**
(needs real ground-truth task quality, expensive) and **gate/threshold
calibration data** (needs only domain-membership labels, cheaper, but still
needs genuine difficulty — clean examples alone produce a meaningless
threshold). Both need a held-out split that is never reused as training
data (§4's formulas assume this; §8's original failure came from violating
it). The moat makes this distinction a product surface instead of a
footnote (§4.2).

### 5.4 Pre-deployment regression testing
Before shipping any profiler or expert change, run the same flip-detection
methodology this document set uses on itself: identify inputs whose top-1
routing changes, check whether the change is contamination-driven (fixable)
or genuine ambiguity (irreducible), and confirm the change doesn't exceed
the false-capture budget set in §5.1. This is not a new tool to build —
`addition_isolation_suite.py`'s boundary-sample detection and
`moat_profile_addition.py`'s comparison harness already do exactly this,
and should be treated as a CI step, not a one-time research script.

### 5.5 Multi-addition planning
If the roadmap calls for onboarding several new domains over time, decide
the aggregate false-capture tolerance across the *whole* roadmap up front
and divide it via Bonferroni correction (§7) across the planned additions —
not per-addition in isolation. §7's finding is specific: three domains at
1% individual FPR each compound to 3% aggregate, not 1%. Discovering this
in production is a worse outcome than budgeting for it at design time. A
moat narrows this problem to the genuinely uncovered domains, but does not
delete it — profiles added simultaneously still interact through the
router.

### 5.6 Moat maintenance is a standing cost, not an event
Boundary-example density per domain, held-out splits never reused as
training data (§8), and versioned profile re-calibration are the recurring
expenses. The moat trades compute-time penalties for data-time costs; it
wins only where data is cheaper than retraining (§8).

---

## 6. Running Diagnosis

This is the use case that follows most directly from a proven property
(traceability, §10) rather than a new capability — the diagnostic tooling
largely already exists as the verification scripts used throughout this
document set.

### 6.1 Why traceability matters operationally
A learned gate's routing decision is a forward pass through jointly-
optimized weights — not decomposable into a stated reason (§3). A
profile-routed decision reduces to a specific number: "expert X won because
it scored Y on domain benchmark Z, with margin M over the second-place
expert." That's an answerable question for every single request, not just
an aggregate metric across a validation set.

### 6.2 Diagnosing a misroute: contamination vs. genuine ambiguity
When a routing decision looks wrong, the gate score tells you which failure
mode you're looking at, not just that something failed:
- **Low gate score, still misrouted** → not new-domain contamination; check
  for old-boundary shift from a joint-retraining mistake (§6.1).
- **Gate score just above threshold, misrouted** → likely genuine ambiguity
  — check the margin: a large margin (~0.3–0.4, per the measured cases)
  means the system is confidently wrong, which argues for improving the
  gate's calibration data; a narrow margin means it's a legitimate
  coin-flip case that should probably escalate to a human rather than
  resolve automatically.
- **Gate score far above threshold, still gets the "wrong" answer relative
  to a human label** → worth checking whether the human label was actually
  correct, since §8 found several of these are defensible reclassifications
  of genuinely dual-domain content, not system errors.

### 6.3 Production health metrics
Directly derived from what this document set already measures offline — the
same instrumentation, run continuously instead of once:
- **Flip rate over time**: has an expert swap or profiler update changed
  routing behavior for domains that weren't supposed to be touched? (§5's
  isolation check, as a standing monitor rather than a one-time test.)
- **τ / margin distribution**: §6.6 found 74.5% of requests sit within 2×
  of baseline temperature and only 2.7% get substantially softened — a
  shift in that distribution over time is a signal worth investigating
  before it shows up as a user-visible accuracy drop.
- **Aggregate gate false-capture rate vs. budget**: §7's compounding math
  means this needs active tracking, not a one-time calibration check — the
  moment a new domain is added, the aggregate risk changes for every domain
  already in the system.

### 6.4 The diagnostic tool is not hypothetical
Nothing in this section describes work that doesn't already exist in some
form. `addition_isolation_suite.py`, `capacity_ablation.py`,
`multi_dimension_compounding.py`, and `moat_profile_addition.py` are,
functionally, a diagnostic suite that happens to have been built to verify
this document's own claims. Turning them into a standing production tool is
an engineering and packaging task, not a research one — which is a
meaningfully lower-risk starting point than most of what's in this
document.

---

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
  train, per §5.3's held-out rule); sources 30 hand_written /
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

## 8. Honest limits (stated as plainly as the claims)

- **Proven only in the synthetic setting** of this suite's data generator
  for the moat's central claim (§3.2). The real-text arm is lexical-only,
  and the corpus brick (§7) is machine-generated; no real-text moat
  experiment exists yet.
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
- **The hosting-layer product has no differentiation without the isolation
  guarantee** (§4.4); the marketplace has no moat until the corpus is real
  (§4.2). The product list is ordered by evidence for a reason.

---

*This document's claims, like TECHNICAL.md's, are only as strong as the
scripts behind them. The evidence for §3.2 is `scripts/moat_profile_addition.py`
and for §7 is `scripts/build_moat_corpus.py` — run them; the numbers above
are their output on this repository's canonical seeds.*
