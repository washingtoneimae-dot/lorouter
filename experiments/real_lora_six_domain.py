"""
real_lora_six_domain.py

Scale-up of the real-LoRA integration to the v3 corpus (3,010 examples,
6 clean domains: finance, law, code, education, medicine, psychology).
Same protocol as real_lora_integration.py: 135M base, rank 8, 10 epochs,
answer-conditional profiles, cosine routing.

Question: does the 4-domain 96.4% routing result hold when the pool and
the domain space grow to 6 domains / 500-example domains?

Run: python3 experiments/real_lora_six_domain.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lorouter.router import ProfileRouter
from lorouter.corpus import load_corpus, split_clean
from experiments.real_lora_integration import (MODEL_ID, ANSWERS, make_qa,
                                               train_adapter, profile_adapter)

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "moat_brick3.jsonl"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print(f"REAL-LORA 6-DOMAIN -- {MODEL_ID} on v3 corpus "
          f"({CORPUS.name})")
    print(f"device: {device}")
    print("=" * 72)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_corpus(CORPUS)
    DOMAINS = sorted({r["domain_label"] for r in rows if not r["is_boundary_example"]})
    print(f"domains: {DOMAINS}")
    train_rows = split_clean(rows, "train")
    calib_rows = split_clean(rows, "calibration")
    test_rows = split_clean(rows, "test")
    per_domain = {d: sum(1 for r in train_rows if r["domain_label"] == d)
                  for d in DOMAINS}
    print(f"train QA pairs per domain: {per_domain}")
    calib_by_domain = {d: [r["text"] for r in calib_rows if r["domain_label"] == d]
                       for d in DOMAINS}

    adapters, profiles, loss_mat = {}, {}, {}
    for d in DOMAINS:
        qa = make_qa([r for r in train_rows if r["domain_label"] == d], d)
        print(f"training adapter_{d} on {len(qa)} QA pairs...")
        adapters[d] = train_adapter(base, tokenizer, qa, d)
        prof, losses = profile_adapter(adapters[d], tokenizer, calib_by_domain,
                                       DOMAINS, device)
        profiles[d] = prof
        loss_mat[d] = dict(zip(DOMAINS, [round(l, 4) for l in losses]))

    print("\nadapter loss matrix (6-domain):")
    print(f"{'adapter':12s} " + " ".join(f"{d:>12s}" for d in DOMAINS))
    for d in DOMAINS:
        print(f"{d:12s} " + " ".join(f"{loss_mat[d][dd]:>12.4f}" for dd in DOMAINS))
    diag = sum(1 for d in DOMAINS if min(loss_mat[d], key=loss_mat[d].get) == d)
    print(f"diagonal dominance: {diag}/6")

    router = ProfileRouter.build(train_rows, DOMAINS, seed=42)
    mat = np.array([profiles[d] for d in DOMAINS])
    matn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    correct = 0
    by_domain = {d: [0, 0] for d in DOMAINS}
    for r in test_rows:
        q = router.query_profile(r["text"])
        w = DOMAINS[int(np.argmax(matn @ (q / (np.linalg.norm(q) + 1e-8))))]
        ok = w == r["domain_label"]
        correct += ok
        by_domain[r["domain_label"]][0] += ok
        by_domain[r["domain_label"]][1] += 1
    acc = correct / len(test_rows)
    print(f"\nrouting accuracy (6 domains): {acc*100:.1f}% "
          f"({correct}/{len(test_rows)})")
    for d in DOMAINS:
        c, n = by_domain[d]
        print(f"  {d:12s} {c/n*100:6.1f}%  ({c}/{n})")
    print(f"\nreference: 4-domain 96.4% (brick 2, 54/56) | "
          f"random floor 1/6 = 16.7%")
    print("Honest note: 6 adapters x ~280 QA pairs x 10 epochs; GPU")
    print("nondeterminism applies (pattern claim, not exact digits).")


if __name__ == "__main__":
    main()
