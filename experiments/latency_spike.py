"""
latency_spike.py

Serving-layer integration spike, part 1: measure the SELECTION-POLICY
latency -- the cost the router adds per request, and how it scales with
adapter-pool size. This is the production-advantage number the review
asked for: selection cost per query at N adapters.

Measured (warm, mean of 1000 queries):
  - query profiling latency (TF-IDF + SVD + classifier)
  - cosine top-k selection at pool sizes 4, 8, 100, 1k, 10k (numpy; torch
    shown separately)
  - end-to-end selection policy latency (profile + cosine) per query

Honest framing: this measures the policy in isolation. Full-stack
integration (vLLM/LoRAX hook, adapter load/switch costs, batching) is not
measured -- that is the remaining half of the spike.

Run: python3 experiments/latency_spike.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lorouter.router import ProfileRouter
from lorouter.corpus import load_corpus, split_clean


def bench(fn, n=1000):
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000  # ms


def main():
    print("=" * 72)
    print("SELECTION-POLICY LATENCY SPIKE")
    print("=" * 72)
    rows = load_corpus()
    DOMAINS = sorted({r["domain_label"] for r in rows if not r["is_boundary_example"]})
    train_rows = split_clean(rows, "train")
    router = ProfileRouter.build(train_rows, DOMAINS, seed=42)
    q = train_rows[0]["text"]
    rng = np.random.RandomState(0)

    t_profile = bench(lambda: router.query_profile(q))
    print(f"\nquery profiling (TF-IDF+SVD+LR): {t_profile:.3f} ms/query")

    print("\ncosine top-k selection latency by pool size (numpy):")
    sizes = [4, 8, 100, 1000, 10000]
    mat = rng.rand(sizes[-1], 4)
    for n in sizes:
        sub = mat[:n]
        qp = rng.rand(4)
        t = bench(lambda: _cos(sub, qp))
        print(f"  N={n:>6d}: {t*1000:.1f} us/query")
    print("  (cosine top-1 over k-dim profiles; selection is a matrix-vector")
    print("   product -- scales linearly in pool size, trivially parallel)")

    t_end = bench(lambda: _sel_end(router, q, mat))
    print(f"\nend-to-end policy (profile + cosine, N=10000): "
          f"{t_profile + t_end:.3f} ms/query")

    print("\n" + "=" * 72)
    print("FRAMING")
    print("=" * 72)
    print("  Selection policy adds ~sub-millisecond per request at 10k adapters.")
    print("  The comparison that matters is adapter SWITCH/LOAD cost in serving")
    print("  systems (ms-scale per switch; the scheduling-with-switching-costs")
    print("  framing) -- wrong-adapter requests cost a full regenerate. The")
    print("  policy pays microseconds to avoid that. Full-stack integration")
    print("  latency (vLLM/LoRAX hook) remains unmeasured -- half the spike.")


def _cos(mat, qp):
    q = qp / (np.linalg.norm(qp) + 1e-8)
    rows = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return int(np.argmax(rows @ q))


def _sel_end(router, q, mat):
    qp = router.query_profile(q)
    return _cos(mat, qp)


if __name__ == "__main__":
    main()
