# Adaption integration (2026-08-27) — data-moat upgrade via external pipeline

Frontier-lab-style dataset optimization + hosted training, applied to the
moat corpus. Full reproducible script: `experiments/adaption/adaption_pipeline.py`.

## What was done

1. **Adaptive Data on the v3 corpus** (`corpus/moat_brick4_adapted.csv`):
   1,673 clean train rows (prompt + domain context + templated completion)
   through Adaption's dataset-improvement pipeline. Platform quality grade:
   **D → B (4.0 → 7.5, +87.5%)**. Output: 1,117 rows with an
   `enhanced_completion` column — model-generated, domain-grounded answers
   with Kenyan context (e.g. law answers cite the Kenya Labour Relations Act
   2007 / ELRC; code answers include working examples).
2. **AutoScientist training**: Qwen3.5-0.8B, LoRA (r16/α32), instruction
   format, no synthetic augmentation, max 3 iterations. Trains on
   prompt → enhanced_completion. Best checkpoint: experiments/adaption/
   (auto-downloaded by watcher).
3. **External domain imports** (HuggingFace → Adaption, for the 9-domain
   pool): 3gpp-5g-nr-qa (telecom, 26,926 rows), KisanVaani
   agriculture-qa (22,615), personal-finance-africa (fintech, 201).
   Adaption dataset IDs in the pipeline script.

## Cost ledger (free credits)

| item | credits |
|---|---|
| Adaptive Data, v3 corpus (1,673 rows) | 18 |
| Adaptive Data, 5G-NR QA (500-row cap) | 5 |
| Adaptive Data, Agriculture QA (500-row cap) | 5 |
| Adaptive Data, Personal Finance (201 rows) | 2 |
| AutoScientist (iterations-bounded) | TBD |
| **total committed** | **30 + TBD** |

## Findings

- **F43. External data-moat addition via Adaptive Data**: adapting the v3
  corpus through Adaption's pipeline produced 1,117 enhanced QA pairs across
  all 6 domains with a platform-evaluated D→B (+87.5%) quality lift. The
  enhanced completions are domain-grounded (Kenyan legal/finance/code
  specifics), replacing the synthetic templates as training signal.
  *(in progress — downstream routing numbers pending)*
- **F44. Training on enhanced completions**: AutoScientist (Qwen3.5-0.8B,
  LoRA, 1,117 pairs, no augmentation) is the modernized joint-retrain
  baseline — the direct successor of the moat_profile_addition.py joint
  model (184 flips at the old size). Routing vs this baseline is the
  F42-scale comparison. *(pending run completion)*
- **F45. External real-world domains imported**: 5G-NR/telecom, agriculture,
  and Africa-fintech QA pairs imported from HuggingFace into the pipeline —
  the 9-domain pool's new adapters are grounded in real external data, not
  synthetic generation. *(pending adaptation + benchmark)*

## Honest limits

- The 5G-NR set is CC-BY-NC (non-commercial): fine for the paper/demo,
  not for a commercial product. Agriculture (Apache-2.0) and finance
  (CC-BY-4.0) are clean.
- Platform dedup/filtering reduced 1,673 → 1,117 rows during evaluation;
  per-domain balance must be re-verified before the routing benchmark.
- External sets are not Kenyan-context (except the fintech one) — they
  demonstrate the *addition strategy*, not corpus replacement.
