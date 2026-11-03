"""Feature maps for the function-approximation agent (post-hoc extension)

In plain terms: the tabular agent keeps one learned number per portfolio. The
function-approximation agent (src/fa_agent.py) instead describes every
portfolio by a short list of numbers, its features phi(w), and learns one
weight per feature, theta, so that Qhat(w) = theta . phi(w). Two portfolios
with similar features get similar Q-values, so a reward from one also teaches
the agent about its neighbours.

This file turns a matrix of portfolio weights W (one row per portfolio, the
nine columns in src/data/universe.py TICKERS order) into a feature matrix F
(one row per portfolio, one column per feature). train.py builds F once from
the action grid and hands it to the agent; the agent never sees W itself.

Leakage boundary: nothing here reads mu, Sigma, the benchmarks or any data.
A feature map is a choice about the *shape* of the Q-function, made before
training, using only the grid. test_environment.py's leakage tests cover this
file.

Post hoc: the function-approximation extension was designed after the
baseline-v1 results were seen. Label every result that uses it that way.
"""

import numpy as np

N_ASSETS = 9


def onehot_features(n_actions: int, dtype=np.float32) -> np.ndarray:
    """One feature per portfolio: row a is all zeros except a 1 in column a.

    With these features, theta . phi(a) is just theta[a], so the linear agent
    becomes the tabular agent. Used to test that src/fa_agent.py reduces to
    the tabular update, not for experiments."""
    if isinstance(n_actions, bool) or not isinstance(n_actions, (int, np.integer)) or n_actions < 1:
        raise ValueError(f"n_actions must be a positive integer, got {n_actions!r}")
    return np.eye(n_actions, dtype=dtype)


def quadratic_features(W: np.ndarray, dtype=np.float32) -> np.ndarray:
    """The 45 products w_i * w_j for i <= j, in np.triu_indices(9) order.

    This is the REFERENCE model, not the discovery model: the true expected
    reward w'mu - (gamma/2) w'Sigma w is a quadratic in w, so this feature set
    can represent it exactly. It shows how the agent does when it is handed
    the right functional form: a correctly specified reference. That is not a
    guaranteed upper bound for the generic models in a finite sample, and it
    is never the discovery result.

    No intercept and no linear terms, because the weights sum to one:
        w_i = w_i * (sum_j w_j)          (a linear term is a sum of products)
        1   = (sum_i w_i) * (sum_j w_j)  (so is a constant)
    Adding them would only duplicate columns. The 45 products span every
    quadratic in the 8 free weights (8+2 choose 2 = 45), so nothing is lost.
    """
    W = _check_weights(W)
    i, j = np.triu_indices(W.shape[1])
    return (W[:, i] * W[:, j]).astype(dtype)


def _check_weights(W: np.ndarray) -> np.ndarray:
    """Weights must be an (m, 9) matrix of long-only rows that sum to one:
    the quadratic map relies on the sum-to-one identity above."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[1] != N_ASSETS:
        raise ValueError(f"W must have shape (m, {N_ASSETS}), got {W.shape}")
    if not np.all(np.isfinite(W)) or (W < -1e-12).any():
        raise ValueError("weights must be finite and nonnegative")
    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("each row of W must sum to one")
    return W
