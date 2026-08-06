"""
real_lora_integration.py

Real-LoRA integration test for lorouter: train four domain-specialized
LoRA adapters on a small open model (SmolLM2-135M-Instruct) using QA pairs
built from the loader's default corpus (brick 2), profile each adapter by
calibration loss, and route real test queries with the profile router.

Pipeline:
  1. Build QA pairs from the loader's default corpus (brick 2) train split
     (prompt = corpus question, answer = domain-templated response)
  2. Train one LoRA adapter per domain (rank 8, q_proj/v_proj)
  3. Profile each adapter: competence vector = inverse calibration loss
     per domain (the LM-text analog of the suite's inverse-MSE profiles)
  4. Route test queries: shared profiler (TF-IDF+SVD) -> query profile,
     cosine top-1 against adapter profiles
  5. Report: adapter differentiation (does adapter_i have lowest loss on
     domain i?), routing accuracy vs the stand-in benchmark (96.4%),
     honest limits.

This is a MECHANISM integration test: the adapters are tiny and trained on
synthetic QA pairs. What is being verified is that profile routing works
end-to-end with real LoRA adapters and real generation losses -- not that
SmolLM2-135M is a good domain specialist.

Run: python3 experiments/real_lora_integration.py
Requires: torch, transformers, peft (CPU or CUDA; downloads the model once
to the HF cache). ~5-15 min on GPU, longer on CPU.
"""
import json
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

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
RANK = 8
EPOCHS = 10
BATCH = 8
LR = 2e-4
MAX_LEN = 96
SEED = 42

ANSWERS = {
    "finance": [
        "For this finance question, review the transaction charges, daily limits and any CBK or mobile-money terms before acting.",
        "This involves money movement: check the fees, the transfer limits and the confirmation details first.",
        "Finance handling: verify the account details, the applicable charges and the refund or reversal process.",
    ],
    "law": [
        "Under Kenyan law, check the relevant Act, the parties' rights and obligations, and the proper procedure before proceeding.",
        "This is a legal matter: consult the statute, consider the parties' duties, and identify the right forum or remedy.",
        "Legal analysis: review the contract or Act in question, then determine liability and the available recourse.",
    ],
    "code": [
        "For this programming question, inspect the function, handle edge cases and errors, and test the change before committing.",
        "This is a coding task: reproduce the issue, check the input handling, and validate the fix with a test.",
        "Engineering approach: trace the data flow, cover the failure modes, and keep the change small and testable.",
    ],
    "education": [
        "For this education question, check the KUCCPS or HELB requirements, the deadlines and the documents needed for the application.",
        "This concerns schooling and admissions: confirm the entry requirements, the funding options and the application window.",
        "Education guidance: verify the course requirements, the fee structure and the registration steps at the institution.",
    ],
}


def make_qa(rows, domain):
    qa = []
    for i, r in enumerate(rows):
        ans = ANSWERS[domain][i % len(ANSWERS[domain])]
        qa.append((f"Q: {r['text']}\nA:", f" {ans}<|im_end|>"))
    return qa


def build_dataset(tokenizer, qa_pairs):
    enc = tokenizer(
        [p + a for p, a in qa_pairs],
        max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt",
    )
    enc["labels"] = enc["input_ids"].clone()
    return enc


def train_adapter(base_model, tokenizer, qa_pairs, domain, seed=SEED):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    cfg = LoraConfig(
        r=RANK, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    data = build_dataset(tokenizer, qa_pairs)
    n = len(data["input_ids"])
    steps = max(1, int(EPOCHS * n / BATCH))
    t0 = time.time()
    for step in range(steps):
        idx = torch.randperm(n)[:BATCH]
        inp = {k: v[idx].to(model.device) for k, v in data.items() if k != "labels"}
        lab = data["labels"][idx].to(model.device)
        out = model(**inp, labels=lab)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"    [{domain}] step {step}/{steps} loss {loss.item():.3f}")
    print(f"    [{domain}] trained {steps} steps in {time.time()-t0:.0f}s, final loss {loss.item():.3f}")
    return model


def profile_adapter(model, tokenizer, calib_by_domain, domains, device):
    """Competence vector: inverse mean NLL per domain on calibration QA
    pairs (question + that domain's answer template). Measuring the
    adapter against the ANSWER BEHAVIOR of each domain, not the question
    text, is what makes profiles carry signal."""
    losses = []
    for d in domains:
        ans = ANSWERS[d]
        texts = [f"Q: {t}\nA: {ans[i % len(ans)]}" for i, t in enumerate(calib_by_domain[d])]
        enc = tokenizer(texts, max_length=MAX_LEN, truncation=True,
                        padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        labels=enc["input_ids"])
        losses.append(out.loss.item())
    v = 1.0 / (np.array(losses) + 1e-4)
    return v / v.sum(), losses


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("REAL-LORA INTEGRATION -- lorouter on SmolLM2-135M-Instruct")
    print(f"device: {device} | rank {RANK} | epochs {EPOCHS} | model {MODEL_ID}")
    print("=" * 72)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"base model loaded in {time.time()-tok0:.0f}s ({sum(p.numel() for p in base.parameters())/1e6:.0f}M params)")

    rows = load_corpus()
    DOMAINS = sorted({r["domain_label"] for r in rows if not r["is_boundary_example"]})
    train_rows = split_clean(rows, "train")
    calib_rows = split_clean(rows, "calibration")
    test_rows = split_clean(rows, "test")
    calib_by_domain = {d: [r["text"] for r in calib_rows if r["domain_label"] == d] for d in DOMAINS}

    # ---- train 4 adapters
    adapters = {}
    profiles = {}
    loss_matrix = {}
    for d in DOMAINS:
        qa = make_qa([r for r in train_rows if r["domain_label"] == d], d)
        print(f"training adapter_{d} on {len(qa)} QA pairs...")
        adapters[d] = train_adapter(base, tokenizer, qa, d)
        prof, losses = profile_adapter(adapters[d], tokenizer, calib_by_domain, DOMAINS, device)
        profiles[d] = prof
        loss_matrix[d] = {dd: round(l, 3) for dd, l in zip(DOMAINS, losses)}
        print(f"  adapter_{d} profile: {dict(zip(DOMAINS, [f'{p:.2f}' for p in prof]))}")

    # ---- differentiation check
    print("\nadapter loss matrix (rows=adapter, cols=calibration domain):")
    print(f"{'adapter':10s} " + " ".join(f"{d:>12s}" for d in DOMAINS))
    for d in DOMAINS:
        print(f"{d:10s} " + " ".join(f"{loss_matrix[d][dd]:>12.4f}" for dd in DOMAINS))
    diag_win = sum(1 for d in DOMAINS
                   if min(loss_matrix[d], key=loss_matrix[d].get) == d)
    print(f"diagonal dominance (adapter lowest on own domain): {diag_win}/{len(DOMAINS)}")
    spread = {d: round(max(loss_matrix[d].values()) - min(loss_matrix[d].values()), 4)
              for d in DOMAINS}
    print(f"within-adapter loss spread (max-min): {spread}")

    # ---- route test queries
    router = ProfileRouter.build(train_rows, DOMAINS, seed=SEED)
    adps = [type("A", (), {"domain": d, "profile": profiles[d]})() for d in DOMAINS]
    correct = 0
    by_domain = {d: [0, 0] for d in DOMAINS}
    for r in test_rows:
        q = router.query_profile(r["text"])
        mat = np.array([a.profile for a in adps])
        idx = int(np.argmax(mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8) @
                            (q / (np.linalg.norm(q) + 1e-8))))
        ok = adps[idx].domain == r["domain_label"]
        correct += ok
        by_domain[r["domain_label"]][0] += ok
        by_domain[r["domain_label"]][1] += 1
    acc = correct / len(test_rows)
    print(f"\nrouting accuracy (real LoRA profiles): {acc*100:.1f}%  ({correct}/{len(test_rows)})")
    for d in DOMAINS:
        c, n = by_domain[d]
        print(f"  {d:10s} {c/n*100:6.1f}%  ({c}/{n})")
    print(f"\nstand-in benchmark (brick 2): 96.4% | random floor: 19.3%")

    # ---- verdict
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  differentiation: {diag_win}/{len(DOMAINS)} adapters lowest on own domain")
    print(f"  routing accuracy: {acc*100:.1f}%")
    print("  Honest limits: 135M model, synthetic QA pairs, 10 epochs, rank 8;")
    print("  this verifies the MECHANISM end-to-end (real LoRA + real losses +")
    print("  profile routing), not adapter quality. Generation quality was not")
    print("  evaluated; no latency or serving-layer integration measured.")


if __name__ == "__main__":
    main()
