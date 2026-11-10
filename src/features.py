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


def bspline_basis(x: np.ndarray, n_basis: int = 5, degree: int = 3) -> np.ndarray:
    """Clamped, uniform-knot B-spline basis on [0, 1], evaluated at x.

    In plain terms: n_basis smooth "bump" functions spread across [0, 1].
    Each one is nonzero over only part of the interval, and at every x they
    are all >= 0 and add up to exactly 1, so a weighted sum of them can bend
    to follow any smooth curve in one asset's weight. "Clamped" means the
    first and last bumps equal 1 at the ends (x = 0 and x = 1), so the basis
    behaves well at the edges of the simplex, where many grid portfolios sit.
    With n_basis = 5 and degree = 3 there is one interior knot, at 0.5.

    Returns an (len(x), n_basis) array.
    """
    from scipy.interpolate import BSpline  # scipy only here; importing features stays cheap

    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (n_basis, degree)) \
            or degree < 0 or n_basis < degree + 1:
        raise ValueError(f"need integers with n_basis >= degree + 1, got n_basis={n_basis}, degree={degree}")
    x = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(x)) or x.min(initial=0.0) < 0.0 or x.max(initial=0.0) > 1.0:
        raise ValueError("x must lie in [0, 1]")
    interior = np.linspace(0.0, 1.0, n_basis - degree + 1)
    knots = np.r_[np.zeros(degree), interior, np.ones(degree)]
    return BSpline.design_matrix(x, knots, degree).toarray()


def spline_additive_pairwise_features(W: np.ndarray, n_basis: int = 5, degree: int = 3,
                                      dtype=np.float32) -> np.ndarray:
    """The MAIN model: additive splines per asset plus pairwise tensor splines.

    In plain terms: Qhat is a sum of a smooth function of each asset's weight
    and a smooth function of each pair of weights. Nothing here is built from
    the reward's true form; whether the agent gets near the best portfolio
    with this generic shape is the research question.

    What the model does and does not assume (checked numerically, 2026-11-03):
    - It is a nonparametric additive-plus-pairwise model in the nine weights.
      "Pairwise" is in the nine redundant weights: since w_9 = 1 - (w_1 + ...
      + w_8), a term in (w_1, w_9) involves all eight free weights.
    - It nests every quadratic in w (see below), so it is correctly specified
      for this reward. Its cost is size: 369 effective parameters on the 20%
      grid and 558 on the 10% grid, against 45 for the quadratic.
    - On the 20% grid it can fit ANY additive-plus-pairwise function of the
      grid (its span equals that of all pairwise cell indicators, rank 369),
      so smoothness restricts nothing there. On the 10% grid it spans 558 of
      the 1,629 dimensions of all pairwise functions, so smoothness binds.

    Columns, in order:
    - for each asset i: the n_basis 1-D basis functions at w_i (9 * n_basis);
    - for each pair i < j: every product B_k(w_i) * B_l(w_j) (36 * n_basis^2).
    That is 45 + 900 = 945 columns for the defaults, before dropping.

    Columns that are zero for every row of W are dropped. With the defaults
    these are the 36 products of the two top basis functions, which are
    nonzero only where w_i + w_j > 1, outside the simplex, so no feasible
    portfolio can excite them. The drop uses W only (the grid), never data.
    So the column set depends on the W passed in: build F once from the
    action grid and use it for the whole run.

    The design is knowingly redundant: each 1-D basis sums to 1, the tensor
    products contain the marginals, and the weights sum to 1. It is left that
    way on purpose; see LinearQAgent's l2 for why that is safe.

    Why it contains every quadratic: each 1-D basis reproduces w_i, so the
    tensor products reproduce every w_i * w_j, and on the simplex those
    products span all quadratics (see quadratic_features). So the true
    expected reward is representable, but nothing here is built from it.
    """
    # _check_weights allows rounding slightly outside [0, 1]; the basis does not.
    W = np.clip(_check_weights(W), 0.0, 1.0)
    m, n_assets = W.shape
    per_asset = bspline_basis(W.ravel(), n_basis, degree).reshape(m, n_assets, n_basis).astype(dtype)
    blocks = [per_asset.reshape(m, n_assets * n_basis)]
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            blocks.append((per_asset[:, i, :, None] * per_asset[:, j, None, :]).reshape(m, n_basis * n_basis))
    F = np.hstack(blocks)
    return F[:, F.max(axis=0) > 0]  # every entry is >= 0, so max > 0 means "not identically zero"


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
