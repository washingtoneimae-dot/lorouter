---
base_model: togethercomputer/Qwen3.5-0.8B
library_name: peft
tags:
  - lorouter
  - lora
  - qwen
  - profile-moe
  - adapter-routing
---

# lorouter joint-retrain baseline (Qwen3.5-0.8B, LoRA)

The **modernized joint-retrain baseline** for the lorouter profile-routing
experiment: one adapter trained on all six moat domains jointly, the
counterfactual that the per-domain adapter pool must beat.

- **Base model:** `togethercomputer/Qwen3.5-0.8B` (instruct)
- **Method:** LoRA r=16, α=32, targets q/k/v/o projections, CAUSAL_LM
- **Training data:** 1,117 enhanced QA pairs from the v3 moat corpus
  (6 domains: finance/law/code/education/medicine/psychology), adapted by
  Adaption Adaptive Data (`corpus/moat_brick4_adapted.csv`); trained on
  prompt → `enhanced_completion` (model-generated, domain-grounded answers
  with Kenyan context)
- **Training:** 1 epoch, 23 steps, final loss 1.418, eval loss 1.406
- **AutoScientist best win rate: 0.4095** (platform eval metric)
- **Trained via:** Adaption AutoScientist on NVIDIA GPUs (run
  `e2cdb7be-9cf8-4463-b438-4ec4036f5d29`), 2026-08-27

## How it fits the lorouter evidence chain

This is the direct successor of the `moat_profile_addition.py` joint model
(184 addition flips at the old corpus size): a stronger, 0.8B-scale
counterfactual trained on the *enhanced* corpus. The lorouter claim is that
profile routing over a per-domain pool matches or beats this joint model
with zero learned router parameters — the F42-scale comparison on the
9-domain pool will supply the numbers.

## Load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("togethercomputer/Qwen3.5-0.8B")
tok = AutoTokenizer.from_pretrained("togethercomputer/Qwen3.5-0.8B")
model = PeftModel.from_pretrained(base, "experiments/adaption/checkpoint/joint-baseline")
```

## Honest limits

- Win rate 0.4095 is the platform's eval metric, not a routing benchmark
  number; generation-quality scoring against the Featherless reference set
  (`experiments/adaption/eval_references/`) is pending.
- One epoch on 1,117 pairs — a mechanism-test baseline, not a product model.
