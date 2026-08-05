"""
unseen_task_generalization.py

LORAUTER-style unseen-task test for lorouter: the router has never seen the
education domain (no education adapter exists, and the shared profiler was
trained on finance/law/code only). Education test queries are routed to the
best-matching EXISTING adapter.

Metrics (real-LoRA arm):
  - routing distribution: where do unseen-domain queries go?
  - oracle agreement: fraction of queries where the router picks the
    adapter with the LOWEST loss on that query (the best available choice)
  - loss gap: (router-chosen loss - oracle loss) / oracle loss, vs the
    expected gap of random selection -- routing efficiency

Stand-in arm (corpus, no model): same hold-out with the lorouter
stand-in machinery, reporting the routing distribution for comparison.

Run: python3 experiments/unseen_task_generalization.py
Requires: torch, transformers, peft (reuses the integration script's
adapter training machinery). ~5-10 min on GPU.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lorouter.router import ProfileRouter
from lorouter.corpus import load_corpus, split_clean

from experiments.real_lora_integration import (MODEL_ID, make_qa, train_adapter,
                                               profile_adapter, ANSWERS)

KNOWN = ["finance", "law", "code"]
UNSEEN = "education"
SEED = 42
MAX_LEN = 96


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print(f"UNSEEN-TASK GENERALIZATION -- {UNSEEN} held out, adapters: {KNOWN}")
    print(f"device: {device}")
    print("=" * 72)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_corpus()
    train_rows = [r for r in split_clean(rows, "train") if r["domain_label"] in KNOWN]
    calib_rows = [r for r in split_clean(rows, "calibration") if r["domain_label"] in KNOWN]
    unseen_test = [r for r in split_clean(rows, "test") if r["domain_label"] == UNSEEN]
    calib_by_domain = {d: [r["text"] for r in calib_rows if r["domain_label"] == d] for d in KNOWN}

    # ---- train 3 adapters, profile on 3 known domains
    adapters, profiles = {}, {}
    for d in KNOWN:
        qa = make_qa([r for r in train_rows if r["domain_label"] == d], d)
        print(f"training adapter_{d} on {len(qa)} QA pairs...")
        adapters[d] = train_adapter(base, tokenizer, qa, d)
        prof, _ = profile_adapter(adapters[d], tokenizer, calib_by_domain, KNOWN, device)
        profiles[d] = prof
        print(f"  profile: {dict(zip(KNOWN, [f'{p:.2f}' for p in prof]))}")

    # ---- router (profiler trained on known domains only)
    router = ProfileRouter.build(train_rows, KNOWN, seed=SEED)
    mat = np.array([profiles[d] for d in KNOWN])
    matn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)

    # ---- per-query losses on unseen-domain text (labels = unseen answer)
    loss_of = {d: [] for d in KNOWN}
    routes = []
    margins = []
    for r in unseen_test:
        q = router.query_profile(r["text"])
        sims = matn @ (q / (np.linalg.norm(q) + 1e-8))
        winner = KNOWN[int(np.argmax(sims))]
        routes.append(winner)
        margins.append(float(np.sort(sims)[-1] - np.sort(sims)[-2]))
        # loss of each adapter on (unseen query + unseen answer)
        for d in KNOWN:
            ans = ANSWERS[UNSEEN][0]
            enc = tokenizer([f"Q: {r['text']}\nA: {ans}"], max_length=MAX_LEN,
                            truncation=True, padding="max_length", return_tensors="pt").to(device)
            with torch.no_grad():
                out = adapters[d](input_ids=enc["input_ids"],
                                  attention_mask=enc["attention_mask"],
                                  labels=enc["input_ids"])
            loss_of[d].append(out.loss.item())

    L = {d: np.array(v) for d, v in loss_of.items()}
    oracle = np.min(np.stack([L[d] for d in KNOWN]), axis=0)
    chosen = np.array([L[routes[i]][i] for i in range(len(routes))])
    random_expected = np.mean(np.stack([L[d] for d in KNOWN]), axis=0)

    from collections import Counter
    dist = Counter(routes)
    print(f"\nunseen-domain queries: {len(unseen_test)}")
    print(f"routing distribution: {dict(dist)}  ({UNSEEN} closest to which known domain)")
    print(f"cosine margin (winner vs runner-up): mean {np.mean(margins):.3f}, "
          f"min {np.min(margins):.3f}, max {np.max(margins):.3f}")

    oracle_agreement = (chosen == oracle).mean()
    gap_router = ((chosen - oracle) / (oracle + 1e-4)).mean()
    gap_random = ((random_expected - oracle) / (oracle + 1e-4)).mean()
    print(f"\noracle agreement: {oracle_agreement*100:.1f}% of queries routed to the best available adapter")
    print(f"loss gap vs oracle: router {gap_router*100:+.1f}% | random expectation {gap_random*100:+.1f}%")
    eff = 1 - gap_router / (gap_random + 1e-8) if gap_random > 0 else float('nan')
    print(f"routing efficiency (1 - gap_router/gap_random): {eff*100:.0f}%")

    # ---- stand-in arm: routing distribution only (no model)
    print("\nstand-in arm (corpus machinery, no model):")
    tr = [r for r in split_clean(rows, "train") if r["domain_label"] in KNOWN]
    cal = [r for r in split_clean(rows, "calibration") if r["domain_label"] in KNOWN]
    r2 = ProfileRouter.build(tr, KNOWN, seed=SEED)
    dist2 = Counter()
    for r in unseen_test:
        q = r2.query_profile(r["text"])
        dist2[KNOWN[int(np.argmax(matn @ (q / (np.linalg.norm(q) + 1e-8))))]] += 1
    print(f"  stand-in routing distribution: {dict(dist2)}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  unseen-domain queries route to the closest EXISTING adapter "
          f"(distribution above); oracle agreement {oracle_agreement*100:.1f}%;")
    print(f"  router loss gap {gap_router*100:+.1f}% vs random expectation {gap_random*100:+.1f}%.")
    print("  Honest limits: 3 adapters, one held-out domain, 135M model, synthetic")
    print("  QA; this tests the unseen-task MECHANISM at small scale, not")
    print("  LORAUTER's 1500-adapter claims.")


if __name__ == "__main__":
    main()
