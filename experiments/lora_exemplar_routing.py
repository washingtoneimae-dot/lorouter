"""
lora_exemplar_routing.py

Follow-up to the unseen-task experiment: does task-level signal help
unseen-domain routing, and what is the mechanism's ceiling when an
aligned adapter EXISTS?

Arms (real LoRA, SmolLM2-135M-Instruct, education fully unseen):
  V1  per-query profile routing        (baseline, = unseen_task script)
  V2a LORAUTER-shape, task level:      unseen-task embedding from N
      education exemplars, cosine vs adapter task embeddings
  V2b LORAUTER-shape, per query:       query embedding, cosine vs adapter
      task embeddings
  V3  aligned-adapter control:         education adapter added to the
      pool, per-query profile routing  (the moat-covered case)

Metrics: oracle agreement (router picks the lowest-loss adapter for that
query), loss gap vs oracle (router and random expectation), routing
distribution.

Run: python3 experiments/lora_exemplar_routing.py
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

KNOWN = ["finance", "law", "code"]
UNSEEN = "education"
EXEMPLAR_N = [5, 10]
SEED = 42
MAX_LEN = 96


def losses_on(model, tokenizer, qa_pairs, device):
    """Mean NLL over ANSWER tokens only (question tokens masked with -100).
    Measures which adapter answers best, not which adapter predicts the
    question text."""
    losses = []
    pad = tokenizer.pad_token_id
    for q, a in qa_pairs:
        q_ids = tokenizer(q, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(a, add_special_tokens=False)["input_ids"]
        ids = (q_ids + a_ids)[:MAX_LEN]
        labels = [-100] * len(q_ids) + a_ids[:MAX_LEN - len(q_ids)]
        ids = ids + [pad] * (MAX_LEN - len(ids))
        labels = labels + [-100] * (MAX_LEN - len(labels))
        inp = {"input_ids": torch.tensor([ids]).to(device),
               "attention_mask": torch.tensor([[1 if t != pad else 0 for t in ids]]).to(device),
               "labels": torch.tensor([labels]).to(device)}
        with torch.no_grad():
            losses.append(model(**inp).loss.item())
    return float(np.mean(losses))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print(f"LORAUTER-STYLE EXEMPLAR ROUTING -- {UNSEEN} unseen, adapters {KNOWN}")
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
    edu_calib = [r["text"] for r in split_clean(rows, "calibration")
                 if r["domain_label"] == UNSEEN]
    unseen_test = [r for r in split_clean(rows, "test") if r["domain_label"] == UNSEEN]
    calib_by_domain = {d: [r["text"] for r in calib_rows if r["domain_label"] == d]
                       for d in KNOWN}

    adapters, profiles = {}, {}
    for d in KNOWN:
        qa = make_qa([r for r in train_rows if r["domain_label"] == d], d)
        print(f"training adapter_{d} on {len(qa)} QA pairs...")
        adapters[d] = train_adapter(base, tokenizer, qa, d)
        prof, _ = profile_adapter(adapters[d], tokenizer, calib_by_domain, KNOWN, device)
        profiles[d] = prof
    # V3: aligned adapter
    edu_qa = make_qa([r for r in split_clean(rows, "train")
                      if r["domain_label"] == UNSEEN], UNSEEN)
    print(f"training adapter_{UNSEEN} on {len(edu_qa)} QA pairs (V3 control)...")
    adapters[UNSEEN] = train_adapter(base, tokenizer, edu_qa, UNSEEN)
    calib_all = {d: [r["text"] for r in split_clean(rows, "calibration")
                     if r["domain_label"] == d] for d in KNOWN + [UNSEEN]}
    prof_all = {}
    for d in KNOWN + [UNSEEN]:
        prof_all[d], _ = profile_adapter(adapters[d], tokenizer, calib_all,
                                         KNOWN + [UNSEEN], device)

    # ---- per-query losses on the ANSWER portion (labels masked on question)
    L = {d: [] for d in KNOWN + [UNSEEN]}
    for r in unseen_test:
        ans = ANSWERS[UNSEEN][0]
        for d in KNOWN + [UNSEEN]:
            L[d].append(losses_on(adapters[d], tokenizer,
                                  [(f"Q: {r['text']}\nA:", f" {ans}")], device))
    L = {d: np.array(v) for d, v in L.items()}
    n = len(unseen_test)

    def report(name, winners):
        w = np.array(winners)
        oracle = np.min(np.stack([L[d] for d in KNOWN]), axis=0)
        chosen = np.array([L[w[i]][i] for i in range(n)])
        agree = (chosen == oracle).mean()
        gap = ((chosen - oracle) / (oracle + 1e-4)).mean()
        from collections import Counter
        print(f"{name:38s} agree {agree*100:5.1f}%  gap {gap*100:+6.2f}%  "
              f"dist {dict(Counter(w.tolist()))}")
        return agree, gap

    print(f"\nunseen queries: {n} | random expectation gap vs oracle: "
          f"{(((np.mean(np.stack([L[d] for d in KNOWN]), axis=0)) - np.min(np.stack([L[d] for d in KNOWN]), axis=0)) / (np.min(np.stack([L[d] for d in KNOWN]), axis=0) + 1e-4)).mean()*100:+.2f}%")

    # ---- V1: per-query profile routing
    router = ProfileRouter.build(train_rows, KNOWN, seed=SEED)
    mat = np.array([profiles[d] for d in KNOWN])
    matn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    v1 = []
    for r in unseen_test:
        q = router.query_profile(r["text"])
        v1.append(KNOWN[int(np.argmax(matn @ (q / (np.linalg.norm(q) + 1e-8))))])
    report("V1 per-query profile routing", v1)

    # ---- V2: LORAUTER-shape (SVD embedding space)
    emb_known = {d: np.mean(router.embed([r["text"] for r in train_rows
                                          if r["domain_label"] == d]), axis=0)
                 for d in KNOWN}
    emat = np.array([emb_known[d] for d in KNOWN])
    ematn = emat / (np.linalg.norm(emat, axis=1, keepdims=True) + 1e-8)
    for N in EXEMPLAR_N:
        task_emb = np.mean(router.embed(edu_calib[:N]), axis=0)
        task_n = task_emb / (np.linalg.norm(task_emb) + 1e-8)
        winner = KNOWN[int(np.argmax(ematn @ task_n))]
        # task-level: every query to the same adapter
        report(f"V2a task-level, exemplars={N}", [winner] * n)
        # per-query embedding routing
        v2b = []
        for r in unseen_test:
            q = router.embed([r["text"]])[0]
            qn = q / (np.linalg.norm(q) + 1e-8)
            v2b.append(KNOWN[int(np.argmax(ematn @ qn))])
        report(f"V2b per-query embeddings, exemplars={N}", v2b)

    # ---- V3: aligned-adapter control (the moat-covered case: education is
    # seen by BOTH the profiler and the adapter pool)
    train_all4 = train_rows + [r for r in split_clean(rows, "train")
                               if r["domain_label"] == UNSEEN]
    router4 = ProfileRouter.build(train_all4, KNOWN + [UNSEEN], seed=SEED)
    mat4 = np.array([prof_all[d] for d in KNOWN + [UNSEEN]])
    mat4n = mat4 / (np.linalg.norm(mat4, axis=1, keepdims=True) + 1e-8)
    v3 = []
    for r in unseen_test:
        q = router4.query_profile(r["text"])
        v3.append((KNOWN + [UNSEEN])[int(np.argmax(mat4n @ (q / (np.linalg.norm(q) + 1e-8))))])
    report("V3 aligned-adapter control", v3)
    # also: what if we route by raw competence (argmax of profile)? ceiling check
    ceil = []
    for i in range(n):
        best = min(KNOWN, key=lambda d: L[d][i])
        ceil.append(best)
    report("oracle-best adapter (ceiling)", ceil)

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  V1 vs V2: does exemplar task-level signal change unseen routing?")
    print("  V3: with an aligned adapter present (the moat-covered case), does")
    print("  oracle agreement jump? -- the claim being tested is that the")
    print("  unseen-task failure is 'no aligned adapter', not 'broken routing'.")


if __name__ == "__main__":
    main()
