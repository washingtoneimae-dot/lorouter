# possibility.md — Use Cases, Products, Engineering Practice, Diagnosis

Status: vision document, explicitly downstream of `TECHNICAL.md`. Every claim below is tagged by evidence tier the same way TECHNICAL.md tags its own claims — **proven**, **bounded/conditional**, **designed**, or **speculative** — because a possibilities document is exactly where overclaiming is easiest and most damaging. Nothing here should be read as a stronger claim than TECHNICAL.md's own sections support; where this document goes further, it says so.

---

## 1. Where This Architecture Actually Fits

Not a universal MoE replacement (§3). It trades a learned gate's ability to discover subtle routing signal at scale for a provable, calibratable isolation guarantee and full decision traceability. That trade is worth making in some situations and not others.

**Good fit — each condition tied to a proven or bounded property, not aspiration:**

- **The domain taxonomy is known and stable, not something you need the system to discover.** Profile routing is bounded by what the calibration benchmark measures (§3, row 7; §11, limitation 5). If the useful categories are genuinely unknown in advance, a learned gate has a real advantage this architecture doesn't.
- **Swaps are frequent, additions are rare.** §5's swap isolation (0.00% collateral change, every run) is unconditional. §6's addition guarantee is conditional and costs real recall when pushed (§7). A deployment that mostly upgrades existing experts and rarely onboards genuinely new domains sits inside the architecture's strongest guarantee almost all the time.
- **Someone needs to audit *why* a decision was made, not just *what* it was.** Traceability (§10, row 2) is structural, not a logging feature bolted on afterward — a decision reduces to a specific calibration score by construction.
- **On-device or privacy-constrained deployment**, where a small profiler and small per-domain adapters can run locally without exporting sensitive queries. (§2.2 — designed, not tested in this document.)

**Poor fit — stated as plainly as the good-fit list:**

- Open-ended domains where the useful specialization boundaries aren't known ahead of time and would need to emerge from data.
- Situations where raw accuracy at scale is the only metric that matters and infrastructure cost is not a constraint — a well-trained learned gate has no fundamental ceiling here that this architecture doesn't also have, and the learned gate gets to discover signal this architecture must be told about explicitly.
- Anyone unwilling to invest in calibration data. §8 and §14 show directly what happens without it: a 300-example hand-written calibration set produced a threshold that fires on almost everything. This architecture's guarantees are only as good as the calibration discipline behind them (§4 formulas, §9's per-domain-worst-case recommendation).

---

## 2. Use Cases

### 2.1 MoE R&D and low-risk experimentation tooling (proven-grounded)
Swap isolation plus traceability means a team building MoE systems can add, remove, or upgrade experts and *know*, not hope, that untouched experts are unaffected — and when something does break, trace it to the specific calibration score responsible, the way this whole document set traces every finding to a script and a line of reasoning. This is the single most evidence-backed use case in this document; everything in it maps to a **proven** result.

### 2.2 Regulated professional-domain assistants (medicine, law, engineering)
Fields where "the system made a decision and nobody can explain why" is a compliance problem, not just a UX complaint. The addition-isolation corollary (§4.5) has a genuinely good fit here: for inputs that are irreducibly ambiguous between two domains, the honest system behavior is *not* to guess confidently — it's to flag the ambiguity for human review. In medicine or law, "escalate instead of guess" is correct behavior, not a limitation to apologize for. This reframes §6.3's hardest finding as a safety feature specific to these fields.

### 2.3 Privacy-focused hybrid local/cloud deployment
Because experts are small, swappable units rather than full models, a domain-specific behavior specialist can live entirely on a practitioner's device, with only the profiler deciding domain locally and never sending raw content elsewhere unless genuinely necessary. **Design requirement, not optional extra:** an adapter is good at *behavior* — formatting a differential diagnosis, reasoning like a structural engineer — and weak at *storing* the facts a real practitioner needs (§3, row 7 names this trade-off explicitly). The viable version of this use case pairs the local expert with a local retrieval index of real reference material, not a claim that the adapter "knows" the field.

### 2.4 Enterprise internal-tooling onboarding
Swapping in an improved internal specialist (say, a better company-policy expert) is the cheap, safe, proven operation (§5). Onboarding a genuinely new internal domain is the rare, expensive, gated operation (§6) — and now has a quantified cost model (§7) for budgeting it before, not after, someone tries to add the fifth or sixth domain at once.

### 2.5 Diagnostic and pedagogical tooling
Because the routing mechanism is fully inspectable, the same infrastructure that verifies this architecture's own claims (the `scripts/` suite) is itself a legitimate product surface: a tool for teaching, auditing, or debugging *any* profile-routed system's behavior, not just this one. See §5 below.

---

## 3. Products

Ordered by how much of the claim is already proven versus how much is still design intent.

### 3.1 MoE isolation-testing toolkit (strongest, most proven-grounded)
Package `addition_isolation_suite.py`'s flip-detection methodology, `capacity_ablation.py`'s capacity-vs-Bayes-error test, and `multi_dimension_compounding.py`'s Bonferroni budgeting as a standalone regression-testing product for teams building any profile-routed or training-free MoE system — not just this one. This is close to a direct repackaging of what already exists and is verified; the product work is mostly interface and generalization, not new research.

### 3.2 Isolation-bounded MoE hosting layer
**Needs real differentiation to be viable.** Multi-adapter serving at scale is already solved well by Punica and S-LoRA (§2.2, §13); a hosting product built on this architecture only earns its place by leading with what those systems don't characterize — provable, budgetable isolation guarantees under expert addition — not by re-competing on serving efficiency alone.

### 3.3 On-device professional-assistant SDK
The privacy-hybrid use case (§2.3) productized: local profiler, local behavior adapters, local retrieval index, optional cloud escalation for genuinely out-of-scope queries. The RAG pairing is a hard requirement of the design, stated in the SDK's own documentation, not a footnote discovered by an unhappy developer later.

### 3.4 Calibration dataset bank / marketplace (farthest out)
Directly downstream of §9's modular composition proposal, which is explicitly **not yet validated at scale**. The schema in `performance_data.xlsx`'s "New Profiler Dataset Design" sheet is a real, usable starting draft, but §9's own honest caveat applies here without modification: building a comprehensive, well-populated, boundary-example-dense calibration bank across many domains is a real undertaking, plausibly a fit for exactly the kind of crowdsourced contribution model already used for adjacent projects — but that's a distribution strategy, not a substitute for validating the underlying composition mechanism first, on 2–3 domains, before assuming it scales.

---

## 4. Development and Maintenance for Engineers

### 4.1 Two operation classes, two different budgets
- **Swap** (replace an expert's function, same domain slot): cheap, safe, day-to-day. Proven isolation (§5). No special calibration-data investment beyond what the new expert itself needs.
- **Addition** (onboard a genuinely new domain): rare, and should be budgeted like an infrastructure project, not a routine deploy. Requires: a properly-sized, boundary-example-dense calibration set (§8, §14 — 300 clean examples produced a degenerate threshold; ~600 boundary examples against ~1,200 clean fixed it), an independently-trained one-vs-rest gate (§4.3), and — if adding more than one domain at once — an explicit false-capture budget split across all simultaneous additions (§4.4, §7), decided before building, not discovered after.

### 4.2 Versioning discipline
The base profiler is frozen once established. New domains are added via independent gates, never by jointly retraining the profiler across old and new domains together (§6.1's `code#82` finding is the concrete reason why: joint retraining can shift decision boundaries among domains that were never touched). This is a hard rule, not a style preference — it's the precondition the entire addition-isolation guarantee depends on.

### 4.3 Calibration data as a first-class engineering artifact
Two different datasets, commonly conflated: **expert calibration data** (needs real ground-truth task quality, expensive) and **gate threshold calibration data** (needs only domain-membership labels, cheaper, but still needs genuine difficulty — clean examples alone produce a meaningless threshold). Both need a held-out split that is never reused as training data (§4.3's formula assumes this; §8's original failure came from violating it).

### 4.4 Pre-deployment regression testing
Before shipping any profiler or expert change, run the same flip-detection methodology this document set uses on itself: identify inputs whose top-1 routing changes, check whether the change is contamination-driven (fixable) or genuine ambiguity (irreducible, per §4.5's corollary), and confirm the change doesn't exceed the false-capture budget set in §4.1. This is not a new tool to build — `addition_isolation_suite.py`'s `find_boundary_samples()` already does exactly this, and should be treated as a CI step, not a one-time research script.

### 4.5 Multi-addition planning
If the roadmap calls for onboarding several new domains over time, decide the aggregate false-capture tolerance across the *whole* roadmap up front and divide it via Bonferroni correction (§4.4) across the planned additions — not per-addition in isolation. §7's finding is specific: three domains at 1% individual FPR each compound to 3% aggregate, not 1%. Discovering this in production is a worse outcome than budgeting for it at design time.

---

## 5. Running Diagnosis

This is the use case that follows most directly from a proven property (traceability, §10 row 2) rather than a new capability — the diagnostic tooling largely already exists as the verification scripts used throughout this document set.

### 5.1 Why traceability matters operationally
A learned gate's routing decision is a forward pass through jointly-optimized weights — not decomposable into a stated reason (§3, row 5). A profile-routed decision reduces to a specific number: "expert X won because it scored Y on domain benchmark Z, with margin M over the second-place expert." That's an answerable question for every single request, not just an aggregate metric across a validation set.

### 5.2 Diagnosing a misroute: contamination vs. genuine ambiguity
When a routing decision looks wrong, the gate score (§4.3) tells you which failure mode you're looking at, not just that something failed:
- **Low gate score, still misrouted** → not new-domain contamination; check for old-boundary shift from a joint-retraining mistake (§6.1).
- **Gate score just above threshold, misrouted** → likely genuine ambiguity (§4.5's corollary) — check the margin (§6.5): a large margin (~0.3–0.4, per the measured cases) means the system is confidently wrong, which argues for improving the gate's calibration data; a narrow margin means it's a legitimate coin-flip case that should probably escalate to a human rather than resolve automatically.
- **Gate score far above threshold, still gets the "wrong" answer relative to a human label** → worth checking whether the human label was actually correct, since §8 found several of these are defensible reclassifications of genuinely dual-domain content, not system errors.

### 5.3 Production health metrics
Directly derived from what this document set already measures offline — the same instrumentation, run continuously instead of once:
- **Flip rate over time**: has an expert swap or profiler update changed routing behavior for domains that weren't supposed to be touched? (§5's isolation check, as a standing monitor rather than a one-time test.)
- **τ / margin distribution**: §6.6 found 74.5% of requests sit within 2× of baseline temperature and only 2.7% get substantially softened — a shift in that distribution over time is a signal worth investigating before it shows up as a user-visible accuracy drop.
- **Aggregate gate false-capture rate vs. budget**: §7's compounding math means this needs active tracking, not a one-time calibration check — the moment a new domain is added, the aggregate risk changes for every domain already in the system.

### 5.4 The diagnostic tool is not hypothetical
Nothing in this section describes work that doesn't already exist in some form. `addition_isolation_suite.py`, `capacity_ablation.py`, and `multi_dimension_compounding.py` are, functionally, a diagnostic suite that happens to have been built to verify this document's own claims. Turning them into a standing production tool is an engineering and packaging task, not a research one — which is a meaningfully lower-risk starting point than most of what's in this document.
