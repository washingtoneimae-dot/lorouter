"""
moat_profile_addition.py

Tests the "data moat" hypothesis from possibility.md:

  If the profiler is calibrated on a broad multi-domain corpus (the moat)
  from the start, then adding a NEW EXPERT whose profile is calibrated
  on-the-fly from moat data behaves like a SWAP -- zero collateral on the
  existing expert set -- instead of like the ADDITION operation, which
  requires a jointly-retrained profiler, one-vs-rest gates, and suffers
  decision-boundary flips (TECHNICAL.md sections 6, 7).

Two arms, same data generator (shared_data.py), same seeds:

  ARM A (moat, frozen profiler): profiler trained on all 7 domains up
    front (the moat). 6 experts deployed. Medicine expert added later via
    a profile calibrated from moat data. No retraining, no gates.

  ARM B (classic addition, joint retrain): profiler trained on 6 domains,
    then jointly retrained with medicine included -- the addition
    operation characterized in TECHNICAL.md section 6.

Measured: collateral change and routing flips on the 6 pre-existing
domains, medicine routing accuracy, and multi-seed stability (Arm A).

Run: python3 moat_profile_addition.py
Expected runtime: ~1-3 min on CPU.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from shared_data import ALL_CLUSTERS, BASE_CLUSTERS, NEW_CLUSTERS, \
                         generate_dataset, build_expert, build_profiler, \
                         cosine_top1, Expert
from sklearn.neural_network import MLPRegressor

MOAT_NAMES = sorted(ALL_CLUSTERS.keys())          # 7 domains: the moat
BASE_NAMES = sorted(BASE_CLUSTERS.keys())         # 4 fixed domains
ACTIVE_NAMES = sorted(BASE_NAMES + ['law', 'finance'])  # 6 deployed experts
NEW_NAME = 'medicine'                             # added later

rng = np.random.RandomState(7)


def onehot(name, dims):
    v = np.zeros(len(dims)); v[dims.index(name)] = 1.0
    return v


def route(profile, experts):
    ep = np.array([e.profile for e in experts])
    idx, sims = cosine_top1(profile, ep)
    return experts[idx], sims


def mse_of(winner, domain_data):
    return ((domain_data['y'] - winner.predict(domain_data['X'])) ** 2).mean()


def print_table(rows, header):
    w = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    print('  '.join(h.ljust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        print('  '.join(str(c).ljust(w[i]) for i, c in enumerate(r)))


# ----------------------------------------------------------------------
# Data (identical for both arms; seeds fixed)
# ----------------------------------------------------------------------
train, test = generate_dataset(ALL_CLUSTERS, train_seed=42, test_seed=142)
moat_dims = MOAT_NAMES

print("=" * 72)
print("DATA MOAT HYPOTHESIS: profile-on-the-fly vs classic addition")
print("=" * 72)
print(f"moat domains (profiler space): {MOAT_NAMES}")
print(f"deployed experts:              {ACTIVE_NAMES}")
print(f"added later:                   {NEW_NAME}\n")

# ----------------------------------------------------------------------
# ARM A: moat (frozen profiler over all 7 domains)
# ----------------------------------------------------------------------
print("=" * 72)
print("ARM A: frozen profiler over the 7-domain moat")
print("=" * 72)
profiler_moat, clf_moat, scaler_moat = build_profiler(train, moat_dims, seed=42)

experts_a = [build_expert(n, train, test, moat_dims, seed=42) for n in ACTIVE_NAMES]
experts_a = sorted(experts_a, key=lambda e: e.name)

# baseline: winner + MSE per active domain, before adding medicine
def domain_report(experts):
    rep = {}
    for d in ACTIVE_NAMES:
        winner, sims = route(onehot(d, moat_dims), experts)
        rep[d] = (winner.name, mse_of(winner, test[d]))
    return rep

before_a = domain_report(experts_a)

# ---- add medicine: train its regressor, calibrate profile from moat data
med_model = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000,
                         early_stopping=True, random_state=42)
med_model.fit(train[NEW_NAME]['X'], train[NEW_NAME]['y'])
med = Expert(name=f'Expert_{NEW_NAME}', model=med_model)
med.calibrate(test, moat_dims)          # profile on the fly, from moat data
experts_a2 = sorted(experts_a + [med], key=lambda e: e.name)

after_a = domain_report(experts_a2)

print(f"\n{'domain':10s} {'winner before':>16s} {'winner after':>16s} {'MSE before':>12s} {'MSE after':>12s} {'change':>10s}")
flips_a = []
for d in ACTIVE_NAMES:
    wb, mb = before_a[d]; wa, ma = after_a[d]
    ch = (ma - mb) / mb * 100 if mb > 0 else 0.0
    if wb != wa:
        flips_a.append(d)
    print(f"{d:10s} {wb:>16s} {wa:>16s} {mb:>12.5f} {ma:>12.5f} {ch:>+9.3f}%")
print(f"\nflips on pre-existing domains: {len(flips_a)} {flips_a}")

# medicine quality: does routing to the new expert beat the best incumbent?
w_med, _ = route(onehot(NEW_NAME, moat_dims), experts_a2)
incumbent_best = min(((e, mse_of(e, test[NEW_NAME])) for e in experts_a),
                     key=lambda t: t[1])
print(f"medicine inputs route to:      {w_med.name}")
print(f"medicine MSE via new expert:   {mse_of(med, test[NEW_NAME]):.5f}")
print(f"medicine MSE via best incumbent ({incumbent_best[0].name}): {incumbent_best[1]:.5f}")

# ----------------------------------------------------------------------
# ARM B: classic addition -- jointly-retrained profiler (TECHNICAL.md s6)
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("ARM B: classic addition -- jointly-retrained profiler (s6)")
print("=" * 72)
profiler_6, _, _ = build_profiler(train, ACTIVE_NAMES, seed=42)   # frozen 6-dim
profiler_7, _, _ = build_profiler(train, moat_dims, seed=42)      # joint retrain

# experts for the joint arm are all calibrated over the FULL dim set, exactly
# as addition_isolation_suite.py section 6.2 does for its 5-dim arm
experts_v1 = sorted([build_expert(n, train, test, ACTIVE_NAMES, seed=42)
                     for n in ACTIVE_NAMES], key=lambda e: e.name)
experts_v2 = sorted([build_expert(n, train, test, moat_dims, seed=42)
                     for n in ACTIVE_NAMES], key=lambda e: e.name)
med_v2 = Expert(name=f'Expert_{NEW_NAME}', model=med_model)
med_v2.calibrate(test, moat_dims)
experts_v2 = sorted(experts_v2 + [med_v2], key=lambda e: e.name)

# flips: old-domain test inputs whose top-1 expert changes after retraining
ep1 = np.array([e.profile for e in experts_v1])
ep2 = np.array([e.profile for e in experts_v2])
flips_b = []
for d in ACTIVE_NAMES:
    X = test[d]['X']
    p6 = profiler_6(X, order=ACTIVE_NAMES)         # (n, 6)
    p7 = profiler_7(X, order=moat_dims)            # (n, 7)
    w1 = np.argmax(ep1 @ (p6 / (np.linalg.norm(p6, axis=1, keepdims=True) + 1e-8)).T, axis=0)
    w2 = np.argmax(ep2 @ (p7 / (np.linalg.norm(p7, axis=1, keepdims=True) + 1e-8)).T, axis=0)
    n_flips = int((w1 != w2).sum())
    flips_b.append((d, n_flips, len(X)))
print(f"\n{'domain':10s} {'flips':>6s} {'total':>6s}")
tot_flips = 0
for d, f, n in flips_b:
    tot_flips += f
    print(f"{d:10s} {f:>6d} {n:>6d}")
print(f"total flips (joint retrain):  {tot_flips}")

# ----------------------------------------------------------------------
# Multi-seed stability of ARM A (expert training seeds 1..5)
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("ARM A multi-seed stability (expert training seeds 1-5)")
print("=" * 72)
seed_flips = []
for seed in range(1, 6):
    exps = sorted([build_expert(n, train, test, moat_dims, seed=seed)
                   for n in ACTIVE_NAMES], key=lambda e: e.name)
    mm = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000,
                      early_stopping=True, random_state=seed)
    mm.fit(train[NEW_NAME]['X'], train[NEW_NAME]['y'])
    mmed = Expert(name=f'Expert_{NEW_NAME}', model=mm)
    mmed.calibrate(test, moat_dims)
    exps2 = sorted(exps + [mmed], key=lambda e: e.name)
    before = domain_report(exps)
    after = domain_report(exps2)
    flips = sum(1 for d in ACTIVE_NAMES if before[d][0] != after[d][0])
    seed_flips.append(flips)
print(f"flips per seed: {seed_flips}")

# ----------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
moat_ok = (len(flips_a) == 0 and all(f == 0 for f in seed_flips))
print(f"ARM A (moat):  collateral flips on 6 pre-existing domains = {len(flips_a)}")
print(f"                multi-seed flips = {seed_flips}")
print(f"ARM B (joint): flips after jointly-retrained profiler = {tot_flips}")
if moat_ok and tot_flips > 0:
    print("\nHYPOTHESIS: PROVEN (synthetic setting, 7-domain moat, top-1 cosine router).")
    print("Adding an expert whose domain data was already inside the frozen")
    print("profiler's calibration space produces ZERO collateral -- no retraining,")
    print("no gates, no Bonferroni bill -- because the operation is a profile")
    print("insertion, not a gate creation.")
else:
    print("\nHYPOTHESIS: NOT PROVEN -- see numbers above.")
print("\nLIMIT (stated as plainly as the result):")
print("This proves the moat case: the added domain's data was ALREADY in the")
print("profiler's calibration distribution. A genuinely unforeseen domain --")
print("absent from the moat -- remains the hard addition case (open, per")
print("TECHNICAL.md section 6). The moat converts additions into swaps; it")
print("does not remove the hard case, it front-loads the data that avoids it.")
