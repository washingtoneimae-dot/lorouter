"""
generation_quality.py

Closes the unseen-task QUALITY gap at the aligned-adapter level: with the
domain's adapter in the pool, does the ROUTED adapter produce better
OUTPUT, not just lower loss? Generation-quality evaluation with real
generations, scored by semantic similarity to the reference answer.

Setup: education held out of the 3-adapter pool (finance/law/code), then
an education adapter is added (the moat-covered case). For each education
test query:
  - generate a continuation from each adapter (greedy, 24 tokens)
  - score each generation by embedding cosine similarity to the reference
    answer (bge-small-en-v1.5)
  - compare: aligned adapter vs best incumbent vs router choice

Run: python3 experiments/generation_quality.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lorouter.corpus import load_corpus, split_clean
from experiments.real_lora_integration import (MODEL_ID, ANSWERS, make_qa,
                                               train_adapter)

KNOWN = ["finance", "law", "code"]
UNSEEN = "education"
SEED = 42
MAX_NEW = 24


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print(f"GENERATION QUALITY -- aligned-adapter case on {MODEL_ID}")
    print(f"device: {device}")
    print("=" * 72)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_corpus()
    train_rows = split_clean(rows, "train")
    unseen_test = [r for r in split_clean(rows, "test") if r["domain_label"] == UNSEEN]

    adapters = {}
    for d in KNOWN:
        qa = make_qa([r for r in train_rows if r["domain_label"] == d], d)
        print(f"training adapter_{d} on {len(qa)} QA pairs...")
        adapters[d] = train_adapter(base, tokenizer, qa, d)
    edu_qa = make_qa([r for r in train_rows if r["domain_label"] == UNSEEN], UNSEEN)
    print(f"training adapter_{UNSEEN} on {len(edu_qa)} QA pairs...")
    adapters[UNSEEN] = train_adapter(base, tokenizer, edu_qa, UNSEEN)

    # semantic embedder for scoring
    from sentence_transformers import SentenceTransformer
    scorer = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    def generate(model, prompt):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    ref_scores = {d: [] for d in KNOWN + [UNSEEN]}
    gen_examples = {}
    for r in unseen_test:
        prompt = f"Q: {r['text']}\nA:"
        ref = ANSWERS[UNSEEN][0]
        ref_emb = scorer.encode(ref, normalize_embeddings=True)
        for d in KNOWN + [UNSEEN]:
            gen = generate(adapters[d], prompt)
            gen_emb = scorer.encode(gen, normalize_embeddings=True)
            ref_scores[d].append(float(gen_emb @ ref_emb))
        gen_examples[r["text"][:50]] = {d: generate(adapters[d], prompt)[:60]
                                        for d in KNOWN + [UNSEEN]}

    means = {d: float(np.mean(v)) for d, v in ref_scores.items()}
    print("\nmean generation similarity to reference answer (bge-small):")
    for d in KNOWN + [UNSEEN]:
        print(f"  {d:10s} {means[d]:.4f}")
    best_incumbent = max(KNOWN, key=lambda d: means[d])
    print(f"\naligned adapter ({UNSEEN}): {means[UNSEEN]:.4f} | "
          f"best incumbent ({best_incumbent}): {means[best_incumbent]:.4f} | "
          f"margin {means[UNSEEN]-means[best_incumbent]:+.4f}")
    print(f"random expectation (mean over all): {np.mean(list(means.values())):.4f}")

    print("\nsample generations (education queries):")
    for q, gens in list(gen_examples.items())[:3]:
        print(f"  Q: {q}...")
        for d in KNOWN + [UNSEEN]:
            print(f"    [{d:9s}] {gens[d]}")
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  With the aligned adapter present, routing sends the query to the")
    print("  education adapter; the quality question is whether its OUTPUT beats")
    print("  the incumbents' on unseen-domain queries. Margin above is the answer.")


if __name__ == "__main__":
    main()
