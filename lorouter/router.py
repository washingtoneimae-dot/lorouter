"""
router.py -- the profile router core.

Components:
  - Adapter: a per-domain specialist (stand-in for a LoRA adapter) plus its
    calibrated profile vector. Profile[i][j] = how strongly adapter i claims
    inputs from domain j, measured on calibration data (mean predicted
    probability). Calibration is measurement, not training -- swapping an
    adapter's model and re-calibrating never touches the router math.
  - ProfileRouter: the shared query profiler + cosine top-k selection.
    Routing has zero learned parameters: it is cosine similarity between
    the query's domain profile and each adapter's competence profile.
"""
import numpy as np
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def cosine_top1(query_profile, profile_matrix):
    """Top-1 adapter by cosine similarity. THE routing formula."""
    q = query_profile / (np.linalg.norm(query_profile) + 1e-8)
    rows = profile_matrix / (np.linalg.norm(profile_matrix, axis=1, keepdims=True) + 1e-8)
    sims = rows @ q
    return int(np.argmax(sims)), sims


@dataclass
class Adapter:
    name: str                      # adapter id, e.g. "finance"
    domain: str                    # the domain it was tuned for
    model: object                  # stand-in specialist (binary classifier)
    embed: object = None           # text -> feature embedding (router's)
    profile: np.ndarray = None     # calibrated competence vector
    calibration: dict = field(default_factory=dict)

    def claims(self, X):
        """P(this input belongs to my domain), per the specialist model."""
        return self.model.predict_proba(self.embed(X))[:, 1]

    def calibrate(self, calib_by_domain, dims):
        """Profile[d] = mean claim-strength of this adapter on domain d's
        calibration examples. Measurement, not training."""
        v = np.array([self.claims(calib_by_domain[d]).mean()
                      for d in dims])
        self.calibration = dict(zip(dims, v))
        self.profile = v
        return self


class ProfileRouter:
    def __init__(self, dims, tfidf=None, svd=None, scaler=None, clf=None):
        self.dims = list(dims)
        self.tfidf = tfidf
        self.svd = svd
        self.scaler = scaler
        self.clf = clf          # 4-way domain classifier (the profiler)

    # -- construction -------------------------------------------------
    @classmethod
    def build(cls, train_rows, dims, seed=42, svd_dim=50):
        """Train the shared profiler on train text. Features: TF-IDF ->
        SVD -> scaled, the s8 stack."""
        texts = [r["text"] for r in train_rows]
        labels = [r["domain_label"] for r in train_rows]
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        X = tfidf.fit_transform(texts)
        svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
        Xs = svd.fit_transform(X)
        scaler = StandardScaler().fit(Xs)
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(scaler.transform(Xs), labels)
        return cls(dims, tfidf, svd, scaler, clf)

    def embed(self, texts):
        return self.scaler.transform(self.svd.transform(self.tfidf.transform(texts)))

    def query_profile(self, text):
        """Domain membership profile of a query (profiler softmax)."""
        p = self.clf.predict_proba(self.embed([text]))[0]
        order = list(self.clf.classes_)
        return np.array([p[order.index(d)] for d in self.dims])

    # -- routing ------------------------------------------------------
    def route(self, text, adapters, k=1):
        """Top-k adapters for a query, by cosine similarity."""
        q = self.query_profile(text)
        mat = np.array([a.profile for a in adapters])
        idx, sims = cosine_top1(q, mat)
        order = np.argsort(-sims)
        return [adapters[i] for i in order[:k]], q, sims

    def accuracy(self, test_rows, adapters, k=1):
        """Fraction of test inputs whose top-1 adapter matches the true
        domain's adapter."""
        correct = 0
        by_domain = {d: [0, 0] for d in self.dims}   # [correct, total]
        for r in test_rows:
            winners, _, _ = self.route(r["text"], adapters, k)
            ok = winners[0].domain == r["domain_label"]
            correct += ok
            by_domain[r["domain_label"]][0] += ok
            by_domain[r["domain_label"]][1] += 1
        return correct / len(test_rows), by_domain

    # -- operations ---------------------------------------------------
    def swap(self, adapter, replacement_model, calib_by_domain):
        """Replace an adapter's specialist model, re-calibrate its profile.
        The router math is untouched (mirrors the suite's swap test)."""
        new = Adapter(name=adapter.name, domain=adapter.domain,
                      model=replacement_model, embed=adapter.embed)
        new.calibrate(calib_by_domain, self.dims)
        return new
