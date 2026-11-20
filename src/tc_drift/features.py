"""Spline features for the three-asset drift agent (William).

In plain terms: the same additive-plus-pairwise cubic B-spline map as
src/features.py, for three assets instead of nine. Each portfolio is described
by smooth "bump" functions of each weight and products of bumps for each pair
of weights; the agent learns one number per column. It lives in a new file
because src/features.py hard-codes nine assets and is fingerprinted in every
function-approximation training manifest (src/train.py FA_CODE_FILES), so
editing it would flag every FA run as "training code changed". It reuses
src.features.bspline_basis by import, which leaves that file unchanged.

One difference from src/features.py, on purpose. That map drops the columns
that are zero on the rows it is given, so its column set depends on the input;
it is always called once on the whole action grid. The drift agent calls this
map on its current holdings, one state at a time, so the columns must not
depend on the input. Here a pair product B_k(w_i) * B_l(w_j) is dropped when it
is zero everywhere on the simplex: B_k is nonzero only for w above its first
knot t_k, so the product can be nonzero only if t_k + t_l < 1. With the
defaults (5 cubic basis functions, one interior knot at 0.5) that drops the
product of the two top functions for each of the 3 pairs: 15 + 75 - 3 = 87
columns (TC_DRIFT_DESIGN.md §6).

Leakage boundary: reads only the weights it is given, never data.
"""

from functools import lru_cache

import numpy as np

from src.features import bspline_basis

N_ASSETS = 3


def _check_spline(n_basis, degree):
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (n_basis, degree)) \
            or degree < 1 or n_basis < degree + 1:
        # degree 0 (step functions) is excluded: B_k(t_k) = 1 there, so the pair rule below would not hold.
        raise ValueError(f"need integers with degree >= 1 and n_basis >= degree + 1, got n_basis={n_basis}, "
                         f"degree={degree}")


@lru_cache(maxsize=None)
def _pair_mask(n_basis: int, degree: int) -> np.ndarray:
    """Boolean (n_basis * n_basis,) mask of the pair products that can be nonzero on the simplex."""
    _check_spline(n_basis, degree)
    # B_k is zero for w <= its first knot t_k = max(0, k - degree) / n_int, with
    # n_int = n_basis - degree intervals. Compared in integer knot units, so
    # linspace rounding cannot keep or drop a column by accident.
    n_int = n_basis - degree
    first = np.maximum(np.arange(n_basis) - degree, 0)
    mask = (first[:, None] + first[None, :] < n_int).ravel()
    mask.flags.writeable = False
    return mask


@lru_cache(maxsize=None)
def _piecewise(n_basis: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """The basis as polynomials between knots, fitted once from bspline_basis.

    scipy's design matrix costs about 0.5 ms a call, and the agent needs two
    calls per step. Each basis function is a polynomial of `degree` between
    consecutive distinct knots, so evaluating bspline_basis at degree + 1
    points inside each interval and solving the small Vandermonde system gives
    its exact coefficients (to rounding; the tests pin it to bspline_basis).
    Returns the interval left ends (n_int,) and coefficients
    (n_int, degree + 1, n_basis), lowest power first, in u = x - left end."""
    breaks = np.linspace(0.0, 1.0, n_basis - degree + 1)
    coef = np.empty((len(breaks) - 1, degree + 1, n_basis))
    for k, (lo, hi) in enumerate(zip(breaks[:-1], breaks[1:])):
        u = lo + (hi - lo) * (np.arange(degree + 1) + 0.5) / (degree + 1) - lo
        coef[k] = np.linalg.solve(np.vander(u, degree + 1, increasing=True), bspline_basis(u + lo, n_basis, degree))
    left = breaks[:-1].copy()
    left.flags.writeable = False
    coef.flags.writeable = False
    return left, coef


def basis_values(x: np.ndarray, n_basis: int = 5, degree: int = 3) -> np.ndarray:
    """bspline_basis(x) by the piecewise polynomials: (len(x), n_basis). x in [0, 1]."""
    left, coef = _piecewise(n_basis, degree)
    x = np.asarray(x, dtype=float).ravel()
    k = np.clip(np.searchsorted(left, x, side="right") - 1, 0, len(left) - 1)
    u = x - left[k]
    powers = u[:, None] ** np.arange(degree + 1)            # (len(x), degree + 1)
    return np.einsum("np,npb->nb", powers, coef[k])


def n_features(n_basis: int = 5, degree: int = 3) -> int:
    """Number of columns spline_features returns (87 for the defaults)."""
    return N_ASSETS * n_basis + 3 * int(_pair_mask(n_basis, degree).sum())


def spline_features(W: np.ndarray, n_basis: int = 5, degree: int = 3) -> np.ndarray:
    """(m, n) long-only weights summing to one -> (m, k) nonnegative features:
    the n_basis basis functions of each weight, then every product of basis
    functions for each pair of assets, with all-zero columns dropped.

    Columns, in order: asset 0's basis, asset 1's, asset 2's; then the pairs
    (0, 1), (0, 2), (1, 2), each as the row-major products B_k(w_i) B_l(w_j)
    that can be nonzero on the simplex. The column set depends only on n_basis
    and degree, never on W. Accepts (3,) or (m, 3); always returns (m, k)
    float64. Every row has squared norm at most 6, because each asset's basis
    sums to one."""
    W = np.atleast_2d(np.asarray(W, dtype=float))
    if W.ndim != 2 or W.shape[1] != N_ASSETS:
        raise ValueError(f"weights must have shape (3,) or (m, 3), got {np.shape(W)}")
    if not np.all(np.isfinite(W)) or (W < -1e-9).any() or (np.abs(W.sum(axis=1) - 1.0) > 1e-9).any():
        raise ValueError("weights must be finite, long-only and sum to one")
    _check_spline(n_basis, degree)
    flat = basis_values(np.clip(W, 0.0, 1.0), n_basis, degree).reshape(W.shape[0], N_ASSETS * n_basis)
    np.maximum(flat, 0.0, out=flat)  # rounding in the polynomial form can leave -1e-17
    left, right = _pair_columns(n_basis, degree)
    return np.hstack([flat, flat[:, left] * flat[:, right]])


@lru_cache(maxsize=None)
def _pair_columns(n_basis: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Column indices into the (m, 3 * n_basis) per-asset block whose products
    give the kept pair features, in the order spline_features documents."""
    k, l = np.divmod(np.flatnonzero(_pair_mask(n_basis, degree)), n_basis)
    left = np.concatenate([i * n_basis + k for i, _ in ((0, 1), (0, 2), (1, 2))])
    right = np.concatenate([j * n_basis + l for _, j in ((0, 1), (0, 2), (1, 2))])
    left.flags.writeable = False
    right.flags.writeable = False
    return left, right
