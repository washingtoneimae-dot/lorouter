"""
eight_adapter_space.py

Richer adapter space: two LoRA adapters per domain (8 total), trained on
disjoint data with different answer variants. Questions:

  1. Do profiles stay separable as the pool grows? (pairwise cosine)
  2. Does routing accuracy hold at domain level? At variant level?
  3. Does swap isolation hold with 8 adapters in the pool?

Real LoRA (SmolLM2-135M-Instruct), brick 3 QA pairs.

Run: python3 experiments/eight_adapter_space.py
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
from experiments.real_lora_integration import (MODEL_ID, ANSWERS, MAX_LEN,
                                               train_adapter, profile_adapter)
from sklearn.linear_model import LogisticRegression

SEED = 42


def make_qa_variant(rows, domain, variant):
    qa = []
    for i, r in enumerate(rows):
        if i % 2 != variant:
            continue
        qa.append((f"Q: {r['text']}\nA:", f" {ANSWERS[domain][variant]}<|im_end|>"))
    return qa


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("EIGHT-ADAPTER SPACE -- 2 LoRA adapters per domain")
    print(f"device: {device}")
    print("=" * 72)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_corpus()
    DOMAINS = sorted({r["domain_label"] for r in rows if not r["is_boundary_example"]})
    train_rows = split_clean(rows, "train")
    calib_rows = split_clean(rows, "calibration")
    test_rows = split_clean(rows, "test")
    calib_by_domain = {d: [r["text"] for r in calib_rows if r["domain_label"] == d]
                       for d in DOMAINS}

    # ---- train 8 adapters (2 per domain, disjoint data, different answers)
    adapters, profiles = {}, {}
    for d in DOMAINS:
        drows = [r for r in train_rows if r["domain_label"] == d]
        for v in (0, 1):
            name = f"{d}_{'A' if v == 0 else 'B'}"
            qa = make_qa_variant(drows, d, v)
            print(f"training adapter_{name} on {len(qa)} QA pairs...")
            adapters[name] = train_adapter(base, tokenizer, qa, name, seed=SEED + v)
            prof, _ = profile_adapter(adapters[name], tokenizer, calib_by_domain,
                                      DOMAINS, device)
            profiles[name] = prof

    names = list(profiles.keys())
    mat = np.array([profiles[n] for n in names])
    matn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)

    # ---- profile separation
    sims = matn @ matn.T
    off = sims[~np.eye(len(names), dtype=bool)]
    print(f"\nprofile pairwise cosine (8 adapters): min {off.min():.3f} "
          f"| mean {off.mean():.3f} | max {off.max():.3f}")
    print("largest cross-domain similarity pairs:")
    idx = np.argsort(-off)[:3]
    pairs = [(i // (len(names) - 1), i % (len(names) - 1)) for i in idx]
    for (a, b) in pairs:
        if a >= b:
            a, b = b, a
        print(f"  {names[a]} vs {names[b]}: {sims[a][b]:.3f}")

    # ---- routing
    router = ProfileRouter.build(train_rows, DOMAINS, seed=SEED)
    dom_correct = 0
    var_pick = {d: [0, 0] for d in DOMAINS}   # which variant wins, info only
    by_domain = {d: [0, 0] for d in DOMAINS}
    for r in test_rows:
        q = router.query_profile(r["text"])
        winner = names[int(np.argmax(matn @ (q / (np.linalg.norm(q) + 1e-8))))]
        dom_correct += winner.split("_")[0] == r["domain_label"]
        by_domain[r["domain_label"]][0] += winner.split("_")[0] == r["domain_label"]
        by_domain[r["domain_label"]][1] += 1
        var_pick[winner.split("_")[0]][0 if winner.endswith("_A") else 1] += 1
    n = len(test_rows)
    print(f"\ndomain-level routing accuracy: {dom_correct/n*100:.1f}% ({dom_correct}/{n})")
    print(f"  (variant-exact matching is undefined for test rows: the two variants")
    print(f"  of a domain are both correct answers by construction)")
    print(f"within-domain variant split (all routed queries, info only): "
          f"{ {d: f'{v[0]}A/{v[1]}B' for d, v in var_pick.items()} }")
    for d in DOMAINS:
        c, t = by_domain[d]
        print(f"  {d:10s} {c/t*100:6.1f}%  ({c}/{t})")

    # ---- swap isolation with 8 adapters
    target = names[0]  # e.g. code_A
    weak = LogisticRegression(max_iter=2000, random_state=999)
    # weak stand-in replacement (classifier on 5 examples in router space)
    pos = [r["text"] for r in train_rows if r["domain_label"] == target.split("_")[0]][:5]
    neg = [r["text"] for r in train_rows if r["domain_label"] != target.split("_")[0]][:5]
    weak.fit(router.embed(pos + neg), [1] * 5 + [0] * 5)
    # replace: re-profile the swapped adapter using its claims on calibration
    from lorouter.router import Adapter
    repl = Adapter(name=target, domain=target.split("_")[0], model=weak,
                   embed=router.embed)
    repl.calibrate(calib_by_domain, DOMAINS)
    profiles2 = dict(profiles)
    profiles2[target] = repl.profile
    mat2 = np.array([profiles2[n] for n in names])
    mat2n = mat2 / (np.linalg.norm(mat2, axis=1, keepdims=True) + 1e-8)
    flips = 0
    for r in test_rows:
        q = router.query_profile(r["text"])
        w1 = names[int(np.argmax(matn @ (q / (np.linalg.norm(q) + 1e-8))))]
        w2 = names[int(np.argmax(mat2n @ (q / (np.linalg.norm(q) + 1e-8))))]
        if w1 != w2 and w1.split("_")[0] != target.split("_")[0]:
            flips += 1
    print(f"\nswap isolation (8 adapters): swapping {target} -> routing flips on other")
    print(f"domains: {flips}/{n} test inputs")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  domain-level accuracy {dom_correct/n*100:.1f}% | profile separation "
          f"min {off.min():.3f} | swap flips {flips}/{n}")
    print("  Honest limits: variants are near-duplicates (same domain data, split by")
    print("  parity, different answer wording) -- this tests POOL SCALING, not")
    print("  adapter diversity; 135M model, synthetic QA, 10 epochs.")


if __name__ == "__main__":
    main()
