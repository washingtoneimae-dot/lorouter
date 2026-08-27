"""
learned_router_preview.py -- corpus-level learned-router vs profile-routing
preview (2026-08-27).

Learned router (the MoE-LoRA line, simplified): TF-IDF -> LogisticRegression,
trained on the v3 train split ONLY, evaluated on the held-out test split.
This is the cheap, real-data preview of the "learned router" arm of the
lorouter benchmark; the real-LoRA arm (query -> adapter, trained on adapter
losses) is part of the pending 9-domain benchmark.

Result (canonical run): test accuracy 96.4% (405/420), per-domain
92.0-98.7%, random floor 17.9%. Compare: F42 profile routing 95.7% on real
LoRAs, F5 stand-in learned router 96.4%.

Run: /home/imae/.hermes/adaption-venv/bin/python learned_router_preview.py
Requires: scikit-learn, numpy. CPU-only, ~seconds.
"""
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

CORPUS = "corpus/moat_brick3.jsonl"  # v3 corpus (same file the loader uses)

rows = [json.loads(l) for l in open(CORPUS)]
train = [r for r in rows if r["split"] == "train" and not r["is_boundary_example"]]
test = [r for r in rows if r["split"] == "test" and not r["is_boundary_example"]]
print(f"train: {len(train)} | test: {len(test)}")

X_tr = [r["text"] for r in train]
y_tr = [r["domain_label"] for r in train]
X_te = [r["text"] for r in test]
y_te = [r["domain_label"] for r in test]

clf = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000))
clf.fit(X_tr, y_tr)
acc = clf.score(X_te, y_te)

c = Counter(y_te)
rand = max(c.values()) / len(y_te)

print(f"\nlearned router (LogReg/TF-IDF) test accuracy: {acc:.4f}  ({acc*100:.1f}%)")
print(f"random floor: {rand:.4f}  ({rand*100:.1f}%)")
print(f"\nF42 profile routing reference: 95.7% (real adapters, 6 domains)")
print("\nper-domain (learned router):")
for dom in sorted(set(y_te)):
    idx = [i for i, d in enumerate(y_te) if d == dom]
    if idx:
        correct = sum(1 for i in idx if clf.predict([X_te[i]])[0] == dom)
        print(f"  {dom}: {correct}/{len(idx)} ({correct/len(idx)*100:.1f}%)")
