"""
boundary_solutions.py

Solutions A, B, C are three mitigations for the boundary-flip problem
documented in TECHNICAL.md section 6. Solution D is the gated one-vs-rest
fix from TECHNICAL.md sections 4.3/6.3 -- the only one of the four that can
change WHICH expert wins top-1, rather than just reweighting the blend
around whichever expert already won. All four are benchmarked head-to-head
in the same harness below.

Run: python3 boundary_solutions.py
"""
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


def generate_data(n_train=400, n_test=150, seed=42):
    rng_train = np.random.RandomState(seed)
    rng_test = np.random.RandomState(seed + 100)
    clusters = {
        'code': ([0.0, 0.0], lambda x, y: x**2 + y),
        'math': ([5.0, 0.0], lambda x, y: np.sin(x) * y),
        'creative': ([0.0, 5.0], lambda x, y: x * np.cos(y)),
        'reasoning': ([5.0, 5.0], lambda x, y: np.sqrt(x**2 + y**2)),
    }
    train, test = {}, {}
    for name, (center, fn) in clusters.items():
        for data_dict, rng, n in [(train, rng_train, n_train), (test, rng_test, n_test)]:
            xy = rng.randn(n, 2) * 0.8 + np.array(center)
            xy += rng.randn(*xy.shape) * 0.15
            z = fn(xy[:, 0], xy[:, 1]) + rng.randn(n) * 0.1
            data_dict[name] = {'X': xy, 'y': z}
    rng_law_train = np.random.RandomState(123)
    rng_law_test = np.random.RandomState(456)
    for data_dict, rng, n in [(train, rng_law_train, n_train), (test, rng_law_test, n_test)]:
        xy = rng.randn(n, 2) * 0.7 + np.array([2.5, 2.5])
        xy += rng.randn(*xy.shape) * 0.1
        z = 1.0 / (1.0 + np.exp(-(xy[:, 0] - 2.5))) * 3 + xy[:, 1] * 0.5
        z += rng.randn(n) * 0.1
        data_dict['law'] = {'X': xy, 'y': z}
    return train, test


@dataclass
class Expert:
    name: str
    model: MLPRegressor
    profile: np.ndarray = None
    calibration_mse: dict = field(default_factory=dict)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def calibrate(self, cluster_data, profile_dims):
        mse = {}
        for name in profile_dims:
            if name in cluster_data:
                pred = self.predict(cluster_data[name]['X'])
                mse[name] = mean_squared_error(cluster_data[name]['y'], pred)
            else:
                mse[name] = float('inf')
        self.calibration_mse = mse
        skills = np.array([1.0/(mse.get(name, float('inf'))+1e-8) for name in profile_dims])
        self.profile = skills / skills.sum()


class PromptProfiler:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
        self.scaler = StandardScaler()
        self.names = None

    def fit(self, cluster_data):
        self.names = sorted(cluster_data.keys())
        X = np.vstack([cluster_data[n]['X'] for n in self.names])
        y = np.concatenate([[n]*len(cluster_data[n]['X']) for n in self.names])
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s, y)

    def predict_profile(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_s = self.scaler.transform(X)
        probs = self.model.predict_proba(X_s)
        return probs / probs.sum(axis=1, keepdims=True)


def cosine_router(input_profile, expert_profiles, k=2, temperature=0.1):
    ip_n = input_profile / (np.linalg.norm(input_profile) + 1e-8)
    ep_n = expert_profiles / (np.linalg.norm(expert_profiles, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(ep_n, ip_n)
    weights = np.exp(sims / temperature)
    weights /= weights.sum()
    top_k_idx = np.argsort(weights)[-k:][::-1]
    top_k_weights = weights[top_k_idx]
    top_k_weights /= top_k_weights.sum()
    return top_k_idx, top_k_weights, sims


# --- Solutions A, B, C: ORIGINAL, unchanged ---

def route_with_local_confidence(input_profile, experts, profiler, test_data, profile_dims, k=2, tau=0.1, n_nearby=20):
    scores = []
    for e_idx, expert in enumerate(experts):
        cluster_name = expert.name.replace('Expert_', '')
        if cluster_name in test_data:
            local_score = expert.profile[e_idx % len(profile_dims)]
            scores.append(local_score)
        else:
            scores.append(0.0)
    adjusted_profiles = np.array([e.profile for e in experts])
    for i in range(len(experts)):
        adjusted_profiles[i] *= (0.5 + 0.5 * scores[i])
    return cosine_router(input_profile, adjusted_profiles, k, tau)


def route_with_adaptive_temperature(input_profile, expert_profiles, k=2, base_tau=0.1):
    ip_n = input_profile / (np.linalg.norm(input_profile) + 1e-8)
    ep_n = expert_profiles / (np.linalg.norm(expert_profiles, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(ep_n, ip_n)
    sorted_sims = np.sort(sims)[::-1]
    top_gap = sorted_sims[0] - sorted_sims[1]
    adaptive_tau = base_tau + (1.0 - base_tau) * (1.0 - min(top_gap, 1.0))
    weights = np.exp(sims / adaptive_tau)
    weights /= weights.sum()
    top_k_idx = np.argsort(weights)[-k:][::-1]
    top_k_weights = weights[top_k_idx]
    top_k_weights /= top_k_weights.sum()
    return top_k_idx, top_k_weights, sims, adaptive_tau


def route_with_variance_penalty(input_profile, experts, variance_threshold=0.3, k=2, tau=0.1):
    expert_profiles = np.array([e.profile for e in experts])
    variances = np.ones(len(experts))
    for i, expert in enumerate(experts):
        mse_values = list(expert.calibration_mse.values())
        if len(mse_values) > 0:
            mse_arr = np.array([v for v in mse_values if v < float('inf')])
            if len(mse_arr) > 1:
                cv = np.std(mse_arr) / (np.mean(mse_arr) + 1e-8)
                variances[i] = 1.0 / (1.0 + cv)
    adjusted_profiles = expert_profiles.copy()
    for i in range(len(experts)):
        penalty = variances[i]
        adjusted_profiles[i] *= penalty
    return cosine_router(input_profile, adjusted_profiles, k, tau)


# --- Solution D: NEW. Frozen base profiler + independently-calibrated
#     one-vs-rest gate (TECHNICAL.md sections 4.3, 6.3). This is the only
#     solution that can change WHICH expert wins top-1, because it's the
#     only one that intervenes before the broken jointly-retrained profile
#     is ever computed, rather than reweighting after the fact. ---

class GatedDetector:
    """One-vs-rest detector for a single new domain, trained independently
    of the base profiler. Threshold calibrated on held-out base-domain data
    per TECHNICAL.md formula 4.3."""
    def __init__(self, new_domain_name):
        self.new_domain_name = new_domain_name
        self.model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=7)
        self.scaler = StandardScaler()
        self.threshold = None

    def fit(self, all_train_data):
        names = sorted(all_train_data.keys())
        X = np.vstack([all_train_data[n]['X'] for n in names])
        y = np.concatenate([[n]*len(all_train_data[n]['X']) for n in names])
        y_binary = (y == self.new_domain_name).astype(int)
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s, y_binary)

    def calibrate_threshold(self, base_cluster_generators, n_calib=2000, percentile=99, seed=999):
        rng = np.random.RandomState(seed)
        calib_X = []
        for center, fn in base_cluster_generators.values():
            xy = rng.randn(n_calib, 2) * 0.8 + np.array(center)
            xy += rng.randn(*xy.shape) * 0.15
            calib_X.append(xy)
        calib_X = np.vstack(calib_X)
        scores = self.model.predict_proba(self.scaler.transform(calib_X))[:, 1]
        self.threshold = np.percentile(scores, percentile)
        return self.threshold

    def score(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


def route_with_gated_detection(x, profiler_v1, gate, experts_v2, profile_dims_v1, profile_dims_v2, k=2, tau=0.1):
    """Formula: TECHNICAL.md 4.3. Base profile comes from the FROZEN v1
    profiler (never retrained), not the broken jointly-retrained v2 profiler
    every other solution here starts from."""
    base_probs = profiler_v1.predict_profile(x)[0]
    base_dict = dict(zip(profiler_v1.names, base_probs))
    law_score = gate.score(x)[0]
    gamma = law_score if law_score >= gate.threshold else 0.0
    gated = {k_: base_dict[k_] * (1 - gamma) for k_ in profile_dims_v1}
    gated[gate.new_domain_name] = gamma
    gated_profile = np.array([gated[n] for n in profile_dims_v2])
    idx_k, weights, sims = cosine_router(gated_profile, np.array([e.profile for e in experts_v2]), k, tau)
    return idx_k, weights, sims, gamma, law_score


def find_boundary_samples(test_data, experts_v1, profiler_v1, experts_v2, profiler_v2, profile_dims_v2):
    flips = []
    for cluster_name in test_data:
        if cluster_name == 'law':
            continue
        cd = test_data[cluster_name]
        for i in range(len(cd['X'])):
            x = cd['X'][i]
            y_true = cd['y'][i]
            ip_v1 = profiler_v1.predict_profile(x)[0]
            ep_v1 = np.array([e.profile for e in experts_v1])
            idx_v1, _, _ = cosine_router(ip_v1, ep_v1)
            top1_v1 = experts_v1[idx_v1[0]].name
            ip_v2 = profiler_v2.predict_profile(x)[0]
            ep_v2 = np.array([e.profile for e in experts_v2])
            idx_v2, _, _ = cosine_router(ip_v2, ep_v2)
            top1_v2 = experts_v2[idx_v2[0]].name
            if top1_v1 != top1_v2:
                flips.append((cluster_name, i, x, y_true, top1_v1, top1_v2))
    return flips


def test_solution(name, route_fn, boundary_samples, experts_v2, profiler_v1, profiler_v2,
                   profile_dims_v1, profile_dims_v2, gate=None, test_data=None):
    errors, results = [], []
    for cluster_name, idx, x, y_true, v1_top1, v2_top1 in boundary_samples:
        ip = profiler_v2.predict_profile(x)[0]
        if name == 'D: Gated One-vs-Rest':
            idx_k, weights, sims, gamma, law_score = route_fn(x, profiler_v1, gate, experts_v2,
                                                                profile_dims_v1, profile_dims_v2)
        elif name == 'B: Adaptive \u03c4':
            idx_k, weights, sims, tau = route_fn(ip, np.array([e.profile for e in experts_v2]))
        elif name == 'C: Variance Penalty':
            idx_k, weights, sims = route_fn(ip, experts_v2)
        elif name == 'A: Local Confidence':
            idx_k, weights, sims = route_fn(ip, experts_v2, profiler_v2, test_data, profile_dims_v2)
        else:
            idx_k, weights, sims = route_fn(ip, np.array([e.profile for e in experts_v2]))
        y_pred = sum(weights[j] * experts_v2[e_idx].predict(x)[0] for j, e_idx in enumerate(idx_k))
        error = abs(y_true - y_pred)
        errors.append(error)
        results.append({'cluster': cluster_name, 'sample': idx, 'x': x, 'y_true': y_true,
                         'y_pred': y_pred, 'error': error, 'top1': experts_v2[idx_k[0]].name,
                         'top1_weight': weights[0], 'correct_top1': v1_top1})
    return np.array(errors), results


def main():
    print("="*70)
    print("BOUNDARY ROUTING SOLUTIONS (A, B, C original; D new)")
    print("="*70)
    train_data, test_data = generate_data()
    profile_dims_v1 = sorted([k for k in train_data if k != 'law'])
    profile_dims_v2 = sorted(train_data.keys())
    base_cluster_generators = {
        'code': ([0.0, 0.0], None), 'math': ([5.0, 0.0], None),
        'creative': ([0.0, 5.0], None), 'reasoning': ([5.0, 5.0], None),
    }  # centers only needed for calibration sampling; functions unused there

    experts_v1, experts_v2 = [], []
    for name in profile_dims_v1:
        m = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000, early_stopping=True, random_state=42)
        m.fit(train_data[name]['X'], train_data[name]['y'])
        e = Expert(name=f"Expert_{name}", model=m)
        e.calibrate(test_data, profile_dims_v1)
        experts_v1.append(e)

    for name in profile_dims_v1:
        m = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000, early_stopping=True, random_state=42)
        m.fit(train_data[name]['X'], train_data[name]['y'])
        e = Expert(name=f"Expert_{name}", model=m)
        experts_v2.append(e)
    m = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000, early_stopping=True, random_state=99)
    m.fit(train_data['law']['X'], train_data['law']['y'])
    e = Expert(name="Expert_law", model=m)
    experts_v2.append(e)
    for e in experts_v2:
        e.calibrate(test_data, profile_dims_v2)

    profiler_v1 = PromptProfiler()
    profiler_v1.fit({k: train_data[k] for k in profile_dims_v1})
    profiler_v2 = PromptProfiler()
    profiler_v2.fit(train_data)

    gate = GatedDetector('law')
    gate.fit(train_data)
    threshold = gate.calibrate_threshold(base_cluster_generators)
    print(f"Gate threshold (99th pct, n=8000 calib points): {threshold:.4f}")

    flips = find_boundary_samples(test_data, experts_v1, profiler_v1, experts_v2, profiler_v2, profile_dims_v2)
    print(f"Found {len(flips)} boundary samples where routing changed (v1->v2)\n")

    solutions = [
        ('Baseline (broken)', lambda ip, ep: cosine_router(ip, ep)),
        ('A: Local Confidence', route_with_local_confidence),
        ('B: Adaptive \u03c4', route_with_adaptive_temperature),
        ('C: Variance Penalty', route_with_variance_penalty),
        ('D: Gated One-vs-Rest', route_with_gated_detection),
    ]
    all_results = {}
    for name, fn in solutions:
        errors, details = test_solution(name, fn, flips, experts_v2, profiler_v1, profiler_v2,
                                         profile_dims_v1, profile_dims_v2, gate=gate, test_data=test_data)
        all_results[name] = {'errors': errors, 'details': details}
        mean_err = np.mean(errors)
        print(f"{name}: mean_err={mean_err:.4f}  median={np.median(errors):.4f}  max={np.max(errors):.4f}")
        if name != 'Baseline (broken)':
            baseline_errors = all_results['Baseline (broken)']['errors']
            pct_change = ((mean_err - np.mean(baseline_errors)) / np.mean(baseline_errors)) * 100
            improved = np.sum(errors < baseline_errors)
            print(f"   vs baseline: {pct_change:+.1f}%  ({improved}/{len(errors)} samples improved)")

    print(f"\n{'='*70}\nDID TOP-1 EXPERT CHOICE ACTUALLY CHANGE (not just weight blending)?\n{'='*70}")
    any_fixed = False
    for i, flip in enumerate(flips):
        base_top1 = all_results['Baseline (broken)']['details'][i]['top1']
        correct = all_results['Baseline (broken)']['details'][i]['correct_top1']
        for name in ['A: Local Confidence', 'B: Adaptive \u03c4', 'C: Variance Penalty', 'D: Gated One-vs-Rest']:
            sol_top1 = all_results[name]['details'][i]['top1']
            if sol_top1 != base_top1:
                any_fixed = True
                tag = 'FIXED (matches correct answer)' if sol_top1 == correct else 'changed (still wrong)'
                print(f"  Sample {flip[1]} ({flip[0]}): baseline picked {base_top1}, "
                      f"{name} picked {sol_top1}  [{tag}]")
    if not any_fixed:
        print("(no solution changed which expert wins top-1)")
    print("\nExpected: A/B/C never appear above (they only reweight, never re-select).")
    print("D should appear, fixing most but not necessarily all flips -- see")
    print("TECHNICAL.md section 6.3 for which cases are provably fixable vs. genuinely ambiguous.")


if __name__ == '__main__':
    main()
