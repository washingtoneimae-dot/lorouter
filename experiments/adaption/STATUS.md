# Adaption + Featherless pipeline — status (2026-08-27)

**One-line status:** Adaptive Data runs 4/4 done ✅ · AutoScientist training in
flight 🟢 · Featherless eval references done ✅ · 9-domain build + benchmark
not started (next).

This document is a living record: it states what is **done** (verified), what
is **in flight**, and what we **expect at the end** (planned, not yet proven).
It is honest by construction — no item moves from "expected" to "proven" until
the numbers exist.

---

## Done (verified)

1. **Adaption SDK** (`adaption` 0.9.0) authenticated; credit estimates
   (`estimate=True`) used before every paid run; dataset
   import/upload/run/download exercised end-to-end.
2. **v3 corpus adapted** → `corpus/moat_brick4_adapted.csv` (commit `16e944f`):
   1,117 enhanced QA pairs across all 6 domains, platform grade **D → B
   (+87.5%)**. The `enhanced_completion` column carries model-generated,
   domain-grounded answers (Kenyan legal/finance/code specifics).
3. **External imports** (HuggingFace → Adaption): `3gpp-5g-nr-qa` (26,926
   rows), `agriculture-qa-english-only` (22,615), `personal-finance-africa`
   (201). Import is free; adaptation is the paid step.
4. **External adaptations** (downloaded to `experiments/adaption/` assets):
   - agriculture: **500** enhanced rows, →B (+77.5%)
   - fintech: **200** enhanced rows, C→B (+14.3%)
   - 5G telecom: **45** enhanced rows, C→B (+54%) — platform filter kept 45 of
     the 500-row sample; see Honest limits.
5. **Featherless eval references**:
   `experiments/adaption/eval_references/v3_test_references.jsonl` — 420
   held-out test-split questions answered with per-domain Kenyan-context
   prompts (Qwen3-4B FinAdvisor), **0 failures, ~$0.04 total**. This is the
   reference set for the generation-quality eval (F31-style) once the trained
   adapter exists.
6. **Infra**: Netlify MCP configured; HuggingFace CLI + token (`washi254`);
   GitHub fine-grained PAT in `~/.hermes/.env` (never committed).

## In flight

1. **AutoScientist** (run `e2cdb7be`, dataset `88c24bed`): Qwen3.5-0.8B, LoRA,
   instruction format, no synthetic augmentation, max 3 iterations — training
   on the enhanced v3 pairs (prompt → `enhanced_completion`). This is the
   **modernized joint-retrain baseline** (successor of the
   `moat_profile_addition.py` joint model). Win rate 0.41 as of writing.
   Watcher: `lorouter-as-watcher` (systemd, restart-proof) → downloads best
   checkpoint.

## Expected end state (planned, not yet proven)

1. **Checkpoint**: best AutoScientist iteration (Qwen3.5-0.8B LoRA) — the
   counterfactual the router must beat at 0.8B scale.
2. **9-domain pool**: v3 (6 domains) + telecom + agriculture + fintech,
   normalized into the moat-corpus schema (domain_label, split discipline,
   boundary flags) → per-domain LoRAs trained locally on enhanced data.
3. **Routing benchmark** (F42-style on the 9-adapter pool): diagonal
   dominance + routing accuracy vs the 95.7% baseline. Results promote
   F43–F45 (drafted in this folder's README) into `FINDINGS.md` with real
   numbers.
4. **Generation-quality eval** (F31-style): adapter generations scored against
   the 420-question Featherless reference set.
5. **HuggingFace publish**: `washi254/lorouter-moat-corpus-v4` (dataset) +
   per-domain adapter repos + checkpoint.
6. **Paper**: "external data-moat addition" section in the arXiv skeleton.

## Cost ledger (free credits)

| item | credits |
|---|---|
| Adaptive Data, v3 corpus (1,673 → 1,117 rows) | 18 |
| Adaptive Data, 5G-NR QA (500-row cap) | 5 |
| Adaptive Data, Agriculture QA (500-row cap) | 5 |
| Adaptive Data, Personal Finance (201 rows) | 2 |
| AutoScientist (iterations-bounded) | TBD |
| Featherless (420 ref answers) | ~$0.04 |
| **committed** | **30 + TBD** |

## Honest limits / open risks

- **5G is thin** (45 rows): below the ~300 pairs/domain F42 showed is needed
  for full differentiation. Planned fix: fresh 5G sample + Featherless
  completions (pennies) to ~300 rows.
- **AutoScientist requires ≥1,000 rows/run**: per-domain AutoScientist runs are
  impossible without synthetic augmentation (rejected — real-data ethos); the
  pool is trained locally instead.
- External sets are not Kenyan-context (except fintech); licenses: 5G
  **CC-BY-NC** (paper/demo only), agriculture Apache-2.0, fintech CC-BY-4.0.
- **Secret-scan lesson**: Adaption-generated completions contained a
  masked-looking `sk_tes...p7dc` string; sanitized before push. Generated text
  must be scanned for secret-like patterns (`sk_`, `ghp_`, `AKIA`) before
  committing.
