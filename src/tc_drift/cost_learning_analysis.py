"""Why the agent cannot learn the cost coefficients, in isolation (William, 2026-11-17).

In plain terms: the agent's cost block is a two-number regression,
net-reward shortfall = c1 * ||u||_1 + c2 * ||u||_2^2 + noise, learned by
stochastic gradient steps. This script runs just that regression on
synthetic data with the real noise level, to compare step-size rules
(TC_DRIFT_DESIGN.md §9). It ignores the value model theta and the
bootstrapped target, so it is a best case for the agent, not a pilot of it.

    python -m src.tc_drift.cost_learning_analysis      # about 15 minutes

Assumptions:
- Trades are uniform over the 1,326 targets (exploration) from holdings near
  the gamma = 5 optimum (0.46, 0.14, 0.40) plus N(0, 0.02) noise, renormalized.
  The optimum is used here only to describe where the agent's states are; it
  is analysis, never agent input.
- Noise is white, SD 1e-3 per block in daily units (one block's reward,
  TC_DRIFT_DESIGN.md §6 pilot).
- The true coefficients are the $1B ones, c1 = kappa / B, c2 = lam / (2 B).
"""

import numpy as np

from src.tc_drift import calibration, spec

NOISE_SD = 1e-3
N_UPDATES = 500_000
N_REPLICATIONS = 200
DRIFT_TRADE = np.array([0.052, 0.00135])   # ||u||_1 and ||u||_2^2 of a 2.6% one-way transfer between two assets
SEED = 20261117


def distance_sample(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    targets = spec.target_grid()[rng.integers(0, spec.N_TARGETS, len(states))]
    u = targets - states
    return np.c_[np.abs(u).sum(axis=1), (u * u).sum(axis=1)]


def lms(features, D, c, alpha, rng, n=N_UPDATES, r=N_REPLICATIONS):
    """r independent runs of w += alpha(t) * e * x, with y = D @ c + noise."""
    w = np.zeros((r, 2))
    for t in range(n):
        i = rng.integers(0, len(D), r)
        e = D[i] @ c + rng.normal(0.0, NOISE_SD, r) - (features[i] * w).sum(axis=1)
        w += alpha(t) * e[:, None] * features[i]
    return w


def rls(D, c, rng, n=N_UPDATES, r=N_REPLICATIONS, prior=1e-3):
    """Recursive least squares: the step is preconditioned by the running
    second-moment matrix of the agent's own distance features, so it needs no
    b and no knowledge of where the agent's states are. prior * I starts the
    matrix (a small ridge)."""
    w = np.zeros((r, 2))
    A = np.repeat((prior * np.eye(2))[None], r, axis=0)    # sum of x x' so far, per replication
    for t in range(n):
        x = D[rng.integers(0, len(D), r)]
        e = x @ c + rng.normal(0.0, NOISE_SD, r) - (x * w).sum(axis=1)
        A += x[:, :, None] * x[:, None, :]
        w += np.linalg.solve(A, (e[:, None] * x)[:, :, None])[:, :, 0]
    return w


def main() -> dict:
    rng = np.random.default_rng(SEED)
    s = calibration.cost_settings()["aum_1b"]
    c = np.array([s.kappa / spec.BLOCK_LENGTH, s.lam / (2 * spec.BLOCK_LENGTH)])
    near = np.clip(np.array([0.46, 0.14, 0.40]) + rng.normal(0, 0.02, (200_000, 3)), 0, None)
    D = distance_sample(near / near.sum(axis=1, keepdims=True), rng)
    E = D.T @ D / len(D)
    b = E[0, 1] / E[0, 0]
    lattice = spec.lattice()
    E_geo = (lambda G: G.T @ G / len(G))(distance_sample(lattice[rng.integers(0, len(lattice), 200_000)], rng))
    out = {"c_true": c.tolist(), "drift_trade_cost": float(DRIFT_TRADE @ c),
           "correlation": float(np.corrcoef(D.T)[0, 1]), "eigenvalues": np.linalg.eigvalsh(E).tolist(),
           "b_near_optimum": float(b), "b_uniform_states": float(E_geo[0, 1] / E_geo[0, 0])}
    for n in (5e5, 2e6, 1e7):
        cov = NOISE_SD**2 * np.linalg.inv(n * E)
        out[f"least_squares_se_{n:.0e}"] = {"c1": float(np.sqrt(cov[0, 0])), "c2": float(np.sqrt(cov[1, 1])),
                                            "drift_trade_cost": float(np.sqrt(DRIFT_TRADE @ cov @ DRIFT_TRADE))}

    # Decorrelated and rescaled: x2 = (d2 - b d1) * k with k matching x2's second moment to d1's.
    resid = D[:, 1] - b * D[:, 0]
    k = np.sqrt(E[0, 0] / np.mean(resid**2))
    X = np.c_[D[:, 0], resid * k]
    back = lambda w: np.c_[w[:, 0] - b * k * w[:, 1], k * w[:, 1]]   # c1 d1 + c2 d2 = w1 d1 + w2 k (d2 - b d1)
    rules = {
        "constant 1e-3": (D, lambda t: 1e-3, lambda w: w),
        "constant 1e-3, decorrelated": (X, lambda t: 1e-3, back),
        "decaying 1/(E11 (t + 100))": (D, lambda t: 1 / (E[0, 0] * (t + 100)), lambda w: w),
        "decaying, decorrelated": (X, lambda t: 1 / (E[0, 0] * (t + 100)), back),
    }
    # The model-free version: b, k and the step's scale from holdings uniform on the
    # lattice (action geometry only), applied to the same near-optimum trades.
    b_u = E_geo[0, 1] / E_geo[0, 0]
    k_u = np.sqrt(E_geo[0, 0] / (E_geo[1, 1] - 2 * b_u * E_geo[0, 1] + b_u**2 * E_geo[0, 0]))
    X_u = np.c_[D[:, 0], (D[:, 1] - b_u * D[:, 0]) * k_u]
    back_u = lambda w: np.c_[w[:, 0] - b_u * k_u * w[:, 1], k_u * w[:, 1]]
    rules["decaying, decorrelated, uniform b"] = (X_u, lambda t: 1 / (E_geo[0, 0] * (t + 100)), back_u)
    E_u = X_u.T @ X_u / len(X_u)
    out["eigenvalue_ratio_uniform_b"] = float(np.divide(*np.linalg.eigvalsh(E_u)[::-1]))
    out["rules"] = {}
    rules["recursive least squares"] = None
    for name, rule in rules.items():
        if rule is None:
            w = rls(D, c, rng)
        else:
            feats, alpha, to_c = rule
            w = to_c(lms(feats, D, c, alpha, rng))
        cost = w @ DRIFT_TRADE
        out["rules"][name] = {"c1": (float(w[:, 0].mean()), float(w[:, 0].std())),
                              "c2": (float(w[:, 1].mean()), float(w[:, 1].std())),
                              "drift_trade_cost": (float(cost.mean()), float(cost.std()))}
        print(f"{name:30s} c1 {w[:, 0].mean():.2e} ({w[:, 0].std():.2e})  c2 {w[:, 1].mean():.2e} ({w[:, 1].std():.2e})"
              f"  drift trade {cost.mean():.2e} ({cost.std():.2e})", flush=True)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2))
