"""
corpus.py -- loader for the moat corpus bricks (corpus/moat_brick2.jsonl).

Keeps the split discipline of the parent suite: calibration and test never
overlap train.
"""
import json
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "moat_brick2.jsonl"


def load_corpus(path=None):
    path = Path(path) if path else DEFAULT_CORPUS
    return [json.loads(l) for l in open(path)]


def split_clean(rows, split):
    """Clean (non-boundary) rows in a given split."""
    return [r for r in rows if r["split"] == split and not r["is_boundary_example"]]


def split_boundary(rows, split):
    """Boundary rows in a given split."""
    return [r for r in rows if r["split"] == split and r["is_boundary_example"]]
