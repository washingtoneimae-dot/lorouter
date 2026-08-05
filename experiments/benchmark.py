"""
benchmark.py -- first verified benchmark for lorouter on the moat corpus
(corpus/moat_brick2.jsonl, 4 domains: finance, law, code, education).

Compares adapter selection strategies on clean test inputs:

  1. profile   -- lorouter: calibrated competence vectors + cosine routing
                  (zero learned router parameters)
  2. centroid  -- LORAUTER-style task-representation routing: adapter task
                  repr = mean SVD embedding of its domain's train texts,
                  query = SVD embedding, cosine route
  3. learned   -- the MoLE line simplified: a classifier maps query
                  embeddings directly to adapter ids (learned router)
  4. random    -- uniform random selection (floor)
  5. oracle    -- perfect selection (ceiling)

Plus a swap-isolation check in the text setting: replacing one adapter's
specialist model and re-calibrating its profile must not change routing
for the other domains (the suite's swap property, tested on real text).

Everything is deterministic (fixed seeds) and CPU-only.

Run: python3 experiments/benchmark.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lorouter.corpus import load_corpus, split_clean
from lorouter.router import ProfileRouter, Adapter, cosine_top1
from sklearn.linear_model import LogisticRegression

ROWS = load_corpus()
DOMAINS = sorted({r["domain_label"] for r in ROWS if not r["is_boundary_example"]})
TRAIN = split_clean(ROWS, "train")
CALIB = split_clean(ROWS, "calibration")
TEST = split_clean(ROWS, "test")


def calib_by_domain(rows, dims):
    return {d: [r["text"] for r in rows if r["domain_label"] == d] for d in dims}


def build_adapters(router, train_rows, calib_rows, dims, seed):
    """One specialist classifier per domain (stand-in for a tuned LoRA)."""
    adps = []
    for d in dims:
        pos = [r["text"] for r in train_rows if r["domain_label"] == d]
        neg = [r["text"] for r in train_rows if r["domain_label"] != d]
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(router.embed(pos + neg), [1] * len(pos) + [0] * len(neg))
        adps.append(Adapter(name=f"adapter_{d}", domain=d, model=clf,
                            embed=router.embed)
                    .calibrate(calib_by_domain(calib_rows, dims), dims))
    return adps


def main():
    print("=" * 72)
    print("LOROUTER benchmark -- adapter selection on moat brick 2")
    print(f"domains: {DOMAINS} | train {len(TRAIN)} calib {len(CALIB)} test {len(TEST)}")
    print("=" * 72)

    # ---- 5 seeds: vary adapter + profiler training
    n_seeds = 5
    acc = {k: [] for k in ["profile", "centroid", "learned", "random"]}
    for seed in range(1, n_seeds + 1):
        router = ProfileRouter.build(TRAIN, DOMAINS, seed=42 * seed)
        adapters = build_adapters(router, TRAIN, CALIB, DOMAINS, seed=100 * seed)
        emb = router.embed([r["text"] for r in TRAIN])
        test_emb = router.embed([r["text"] for r in TEST])
        test_labels = [r["domain_label"] for r in TEST]

        # 1. profile routing
        a, _ = router.accuracy(TEST, adapters)
        acc["profile"].append(a)

        # 2. centroid routing (LORAUTER-style task representations)
        centroids = {}
        for d in DOMAINS:
            idx = [i for i, r in enumerate(TRAIN) if r["domain_label"] == d]
            centroids[d] = emb[idx].mean(axis=0)
        cmat = np.array([centroids[d] for d in DOMAINS])
        cnorm = cmat / (np.linalg.norm(cmat, axis=1, keepdims=True) + 1e-8)
        qnorm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-8)
        preds = np.array(DOMAINS)[np.argmax(cnorm @ qnorm.T, axis=0)]
        acc["centroid"].append((preds == np.array(test_labels)).mean())

        # 3. learned router (query embedding -> adapter id)
        lr = LogisticRegression(max_iter=2000, random_state=seed)
        lr.fit(emb, [r["domain_label"] for r in TRAIN])
        acc["learned"].append(lr.score(test_emb, test_labels))

        # 4. random floor
        rng = np.random.RandomState(seed)
        acc["random"].append((rng.choice(DOMAINS, len(TEST)) == np.array(test_labels)).mean())

    print(f"\n{'strategy':10s} {'acc(mean)':>10s} {'acc(seeds)':>28s}")
    for k in ["profile", "centroid", "learned", "random"]:
        print(f"{k:10s} {np.mean(acc[k])*100:9.2f}%  {[f'{v*100:.1f}' for v in acc[k]]}")

    # ---- per-domain breakdown (profile routing, seed 1)
    router1 = ProfileRouter.build(TRAIN, DOMAINS, seed=42)
    adps1 = build_adapters(router1, TRAIN, CALIB, DOMAINS, seed=100)
    _, by_domain = router1.accuracy(TEST, adps1)
    print(f"\nper-domain top-1 accuracy (profile routing, seed 1):")
    for d in DOMAINS:
        c, n = by_domain[d]
        print(f"  {d:10s} {c/n*100:6.1f}%  ({c}/{n})")

    # ---- swap isolation in the text setting
    print(f"\nswap isolation (text setting, seed 1):")
    target = DOMAINS[0]
    before = {d: router1.accuracy([r for r in TEST if r['domain_label'] == d], adps1)[0]
              for d in DOMAINS}
    # deliberately weak replacement: classifier trained on 5 examples only
    weak = LogisticRegression(max_iter=2000, random_state=999)
    pos = [r["text"] for r in TRAIN if r["domain_label"] == target][:5]
    neg = [r["text"] for r in TRAIN if r["domain_label"] != target][:5]
    weak.fit(router1.embed(pos + neg), [1] * 5 + [0] * 5)
    adps_swapped = [router1.swap(a, weak, calib_by_domain(CALIB, DOMAINS))
                    if a.domain == target else a for a in adps1]
    after = {d: router1.accuracy([r for r in TEST if r['domain_label'] == d], adps_swapped)[0]
             for d in DOMAINS}
    flips = [d for d in DOMAINS if d != target and before[d] != after[d]]
    print(f"  swapped adapter: {target}")
    print(f"  routing changes on other domains: {len(flips)} {flips}")
    for d in DOMAINS:
        mark = " (swapped)" if d == target else ""
        print(f"  {d:10s} {before[d]*100:5.1f}% -> {after[d]*100:5.1f}%{mark}")

    # ---- verdict
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    p, c, l, r = (np.mean(acc[k]) for k in ["profile", "centroid", "learned", "random"])
    print(f"  profile {p*100:.1f}% | centroid {c*100:.1f}% | learned {l*100:.1f}% | random {r*100:.1f}%")
    print(f"  profile beats random by {((p-r)/r*100):.0f}% relative; "
          f"vs learned router: {p-l:+.1f} pts; vs centroid: {p-c:+.1f} pts")
    print("\n  Caveats (stated plainly): adapters are stand-in binary classifiers,")
    print("  not real LoRA adapters; features are TF-IDF+SVD (lexical); corpus is")
    print("  ~13 calibration examples per domain; no latency or serving-layer")
    print("  integration measured. This benchmarks the SELECTION mechanism only.")


if __name__ == "__main__":
    main()
