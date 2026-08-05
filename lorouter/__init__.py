"""
lorouter -- profile-based LoRA adapter selection for multi-adapter serving.

The router answers one question the serving layer does not: which adapter
does this query need? Serving systems (Punica, S-LoRA, dLoRA, LoRAX, vLLM)
batch and load adapters efficiently but require the request to name its
adapter. Lorouter selects adapters from query content:

  - each adapter carries a calibrated profile vector: its measured
    competence per domain (how strongly it claims inputs from each domain)
  - a shared profiler maps a query into the same domain space
  - the router picks the top-k adapters by cosine similarity -- zero
    learned router parameters, fully traceable decisions

Adapters in this codebase are stand-ins for domain-tuned LoRA adapters:
per-domain specialist classifiers on TF-IDF + SVD features (the same
feature stack as TECHNICAL.md section 8). The isolation and calibration
properties verified in the parent suite (swap isolation, boundary-example
calibration) are the load-bearing claims; the benchmark in
experiments/benchmark.py measures routing accuracy against baselines.
"""
from .router import ProfileRouter, Adapter, cosine_top1
from .corpus import load_corpus, split_clean

__all__ = ["ProfileRouter", "Adapter", "cosine_top1", "load_corpus", "split_clean"]
__version__ = "0.1.0"
