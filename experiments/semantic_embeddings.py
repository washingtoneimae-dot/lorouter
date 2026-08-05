"""
semantic_embeddings.py

Removes the lexical ceiling: re-runs the stand-in benchmark and the
exemplar unseen-task test with semantic embeddings (bge-small-en-v1.5)
instead of TF-IDF+SVD.

Questions:
  1. Does profile routing hold its position (tie learned router) with
     semantic features?
  2. Does the F12 artifact disappear -- with semantic task embeddings,
     do education exemplars route to finance (sensible) instead of code
     (lexical artifact)?

Run: python3 experiments/semantic_embeddings.py
"""
import sys
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lorouter.corpus import load_corpus, split_clean
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
KNOWN = ["finance", "law", "code"]
UNSEEN = "education"


def main():
    print("=" * 72)
    print(f"SEMANTIC EMBEDDINGS -- {MODEL}")
    print("=" * 72)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    enc = SentenceTransformer(MODEL, device=device)

    rows = load_corpus()
    DOMAINS = sorted({r["domain_label"] for r in rows if not r["is_boundary_example"]})
    train_rows = split_clean(rows, "train")
    calib_rows = split_clean(rows, "calibration")
    test_rows = split_clean(rows, "test")
    calib_by_domain = {d: [r["text"] for r in calib_rows if r["domain_label"] == d]
                       for d in DOMAINS}

    def embed(texts):
        return enc.encode(texts, normalize_embeddings=True, batch_size=64)

    # ---- 1. stand-in benchmark with semantic features
    print("\n--- stand-in benchmark, semantic features ---")
    acc = {k: [] for k in ["profile", "centroid", "learned", "random"]}
    for seed in range(1, 6):
        emb = embed([r["text"] for r in train_rows])
        test_emb = embed([r["text"] for r in test_rows])
        test_labels = [r["domain_label"] for r in test_rows]

        # profile routing: adapter = per-domain binary LR on semantic emb
        adps = {}
        for d in DOMAINS:
            pos = [r["text"] for r in train_rows if r["domain_label"] == d]
            neg = [r["text"] for r in train_rows if r["domain_label"] != d]
            clf = LogisticRegression(max_iter=2000, random_state=seed)
            clf.fit(embed(pos + neg), [1] * len(pos) + [0] * len(neg))
            claims = {dd: clf.predict_proba(embed(calib_by_domain[dd]))[:, 1].mean()
                      for dd in DOMAINS}
            v = np.array([claims[dd] for dd in DOMAINS])
            adps[d] = v / (v.sum() + 1e-8)
        # profiler: 4-way LR on semantic emb
        prof = LogisticRegression(max_iter=2000, random_state=seed)
        prof.fit(emb, [r["domain_label"] for r in train_rows])
        mat = np.array([adps[d] for d in DOMAINS])
        matn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        qp = prof.predict_proba(test_emb)
        order = list(prof.classes_)
        qpn = qp / (np.linalg.norm(qp, axis=1, keepdims=True) + 1e-8)
        preds = np.array(DOMAINS)[np.argmax(matn @ qpn.T, axis=0)]
        acc["profile"].append((preds == np.array(test_labels)).mean())

        # centroid
        cents = {d: emb[[i for i, r in enumerate(train_rows)
                         if r["domain_label"] == d]].mean(axis=0) for d in DOMAINS}
        cn = np.array([cents[d] for d in DOMAINS])
        cn = cn / (np.linalg.norm(cn, axis=1, keepdims=True) + 1e-8)
        preds = np.array(DOMAINS)[np.argmax(cn @ test_emb.T, axis=0)]
        acc["centroid"].append((preds == np.array(test_labels)).mean())

        # learned
        lr = LogisticRegression(max_iter=2000, random_state=seed)
        lr.fit(emb, [r["domain_label"] for r in train_rows])
        acc["learned"].append(lr.score(test_emb, test_labels))

        rng = np.random.RandomState(seed)
        acc["random"].append((rng.choice(DOMAINS, len(test_labels))
                              == np.array(test_labels)).mean())

    print(f"{'strategy':10s} {'acc(mean)':>9s}  seeds")
    for k in ["profile", "centroid", "learned", "random"]:
        print(f"{k:10s} {np.mean(acc[k])*100:8.2f}%  {[f'{v*100:.1f}' for v in acc[k]]}")
    print("(TF-IDF+SVD reference: profile 96.4 | centroid 97.9 | learned 96.4 | random 19.3)")

    # ---- 2. exemplar unseen-task test (F12 retest)
    print("\n--- unseen-task exemplar routing, semantic embeddings ---")
    tr = [r for r in train_rows if r["domain_label"] in KNOWN]
    tr_emb = embed([r["text"] for r in tr])
    edu_calib = [r["text"] for r in split_clean(rows, "calibration")
                 if r["domain_label"] == UNSEEN]
    unseen_test = [r for r in test_rows if r["domain_label"] == UNSEEN]
    task_emb = embed(edu_calib[:10]).mean(axis=0)
    task_n = task_emb / (np.linalg.norm(task_emb) + 1e-8)
    cents = {d: tr_emb[[i for i, r in enumerate(tr)
                        if r["domain_label"] == d]].mean(axis=0) for d in KNOWN}
    cn = np.array([cents[d] for d in KNOWN])
    cn = cn / (np.linalg.norm(cn, axis=1, keepdims=True) + 1e-8)
    from collections import Counter
    task_winner = KNOWN[int(np.argmax(cn @ task_n))]
    print(f"task-level (10 exemplars): education -> {task_winner}  "
          f"[TF-IDF/SVD reference: code (F12 artifact)]")
    qe = embed([r["text"] for r in unseen_test])
    perq = np.array(KNOWN)[np.argmax(cn @ qe.T, axis=0)]
    print(f"per-query: {dict(Counter(perq.tolist()))}  "
          f"[TF-IDF/SVD reference: finance 10, code 2, law 2]")


if __name__ == "__main__":
    main()
