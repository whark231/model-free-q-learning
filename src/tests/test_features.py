import numpy as np
import pytest

from src.action_space.grid import generate_weight_grid
from src.fa_agent import LinearQAgent
from src.features import bspline_basis, onehot_features, quadratic_features, spline_additive_pairwise_features


def test_onehot_is_the_identity():
    np.testing.assert_array_equal(onehot_features(4), np.eye(4))
    with pytest.raises(ValueError):
        onehot_features(0)


def test_quadratic_has_45_product_columns_in_triu_order():
    W = generate_weight_grid(9, 0.25)
    F = quadratic_features(W)
    assert F.shape == (495, 45) and F.dtype == np.float32
    i, j = np.triu_indices(9)
    k = 7
    assert F[3, k] == np.float32(W[3, i[k]] * W[3, j[k]])


def test_quadratic_spans_the_linear_terms_and_the_constant():
    """No intercept or linear columns are needed: because weights sum to one,
    1 and every w_i are exact combinations of the 45 products."""
    W = generate_weight_grid(9, 0.2)
    F = quadratic_features(W, dtype=np.float64)
    for target in (np.ones(len(W)), *W.T):
        coef, *_ = np.linalg.lstsq(F, target, rcond=None)
        assert np.abs(F @ coef - target).max() < 1e-10


def test_quadratic_represents_a_mean_variance_objective_exactly():
    """w'm - (g/2) w'S w for an arbitrary m and S (not the calibrated ones) is
    a combination of the 45 columns."""
    W = generate_weight_grid(9, 0.2)
    data = np.random.default_rng(0)
    m, A = data.normal(size=9), data.normal(size=(9, 9))
    S = A @ A.T
    u = W @ m - 2.5 * np.einsum("ij,jk,ik->i", W, S, W)
    F = quadratic_features(W, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(F, u, rcond=None)
    assert np.abs(F @ coef - u).max() < 1e-9


@pytest.mark.parametrize("bad", [
    np.ones((2, 8)) / 8, np.full((2, 9), 0.2), np.array([[-0.1, 1.1] + [0.0] * 7]),
])
def test_quadratic_rejects_weights_that_are_not_long_only_and_fully_invested(bad):
    with pytest.raises(ValueError):
        quadratic_features(bad)


# ---------------------------------------------------------------------------
# Spline features (the main model)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("increment, n_rows", [(0.25, 495), (0.2, 1287)])
def test_spline_shape_is_945_columns_minus_the_36_infeasible_ones(increment, n_rows):
    """9*5 + 36*25 = 945 before dropping; the 36 dropped columns are the
    top-by-top pair products, nonzero only where w_i + w_j > 1."""
    W = generate_weight_grid(9, increment)
    F = spline_additive_pairwise_features(W)
    assert F.shape == (n_rows, 945 - 36) and F.dtype == np.float32
    # Rebuild the 945 columns by hand and check exactly which ones were dropped.
    B = bspline_basis(W.ravel()).reshape(len(W), 9, 5)
    names, cols = [], []
    for i in range(9):
        for k in range(5):
            names.append((i, k))
            cols.append(B[:, i, k])
    for i in range(9):
        for j in range(i + 1, 9):
            for k in range(5):
                for m in range(5):
                    names.append((i, j, k, m))
                    cols.append(B[:, i, k] * B[:, j, m])
    full = np.column_stack(cols)
    dropped = [n for n, c in zip(names, full.T) if not c.any()]
    assert dropped == [(i, j, 4, 4) for i in range(9) for j in range(i + 1, 9)]
    np.testing.assert_allclose(F, full[:, full.any(axis=0)], rtol=1e-6)


def test_spline_values_are_nonnegative_and_deterministic():
    W = generate_weight_grid(9, 0.25)
    F = spline_additive_pairwise_features(W)
    assert (F >= 0).all()
    np.testing.assert_array_equal(F, spline_additive_pairwise_features(W))


def test_each_1d_basis_sums_to_one_including_both_ends():
    x = np.r_[0.0, 1.0, np.random.default_rng(0).uniform(size=200)]
    B = bspline_basis(x)
    assert B.shape == (202, 5) and (B >= 0).all()
    np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-12)
    assert B[0, 0] == 1.0 and B[1, -1] == 1.0  # clamped: the end bumps equal 1 at the ends


def cox_de_boor(x, n_basis, degree):
    """Textbook Cox-de Boor recursion, written independently of scipy, on the
    same clamped uniform knots; the right end uses the last interval."""
    t = np.r_[np.zeros(degree), np.linspace(0, 1, n_basis - degree + 1), np.ones(degree)]
    B = np.zeros((len(x), len(t) - 1))
    for i in range(len(t) - 1):
        B[:, i] = (t[i] <= x) & (x < t[i + 1])
    last = np.flatnonzero(t[:-1] < t[1:])[-1]
    B[x == 1.0, :] = 0.0
    B[x == 1.0, last] = 1.0
    for d in range(1, degree + 1):
        nxt = np.zeros((len(x), len(t) - 1 - d))
        for i in range(len(t) - 1 - d):
            left = (x - t[i]) / (t[i + d] - t[i]) if t[i + d] > t[i] else 0.0
            right = (t[i + d + 1] - x) / (t[i + d + 1] - t[i + 1]) if t[i + d + 1] > t[i + 1] else 0.0
            nxt[:, i] = left * B[:, i] + right * B[:, i + 1]
        B = nxt
    return B


@pytest.mark.parametrize("n_basis, degree", [(5, 3), (4, 3), (6, 3), (4, 2), (5, 2), (3, 1), (7, 4)])
def test_basis_matches_an_independent_cox_de_boor(n_basis, degree):
    x = np.r_[0.0, 0.5, 1.0, np.random.default_rng(3).uniform(size=100)]
    np.testing.assert_allclose(bspline_basis(x, n_basis, degree), cox_de_boor(x, n_basis, degree), atol=1e-12)


def test_weights_rounded_just_outside_zero_and_one_are_accepted():
    W = np.zeros((2, 9))
    W[0, 0], W[1, :2] = 1.0, [1.0 + 1e-13, -1e-13]
    F = spline_additive_pairwise_features(W)
    assert np.all(np.isfinite(F))


def test_spline_model_contains_the_quadratic():
    """Least squares of an arbitrary mean-variance objective on the spline
    columns leaves ~0 residual: the generic model can represent the true
    shape, without being built from it."""
    W = generate_weight_grid(9, 0.2)
    data = np.random.default_rng(1)
    m, A = data.normal(size=9), data.normal(size=(9, 9))
    u = W @ m - 2.5 * np.einsum("ij,jk,ik->i", W, A @ A.T, W)
    F = spline_additive_pairwise_features(W, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(F, u, rcond=None)
    assert np.abs(F @ coef - u).max() < 1e-9 * np.abs(u).max()


def test_spline_design_is_rank_deficient_and_ridge_handles_it():
    """909 columns but rank 225 on the 25% grid (rank always in float64): the
    unpenalized normal equations F'F theta = F'y are exactly singular, so
    there is no unique least-squares theta and a direct solve fails. Adding
    l2 * I makes them invertible, gives one finite answer, and for small l2
    still fits a smooth target to near machine precision."""
    W = generate_weight_grid(9, 0.25)
    F = spline_additive_pairwise_features(W, dtype=np.float64)
    assert np.linalg.matrix_rank(F) == 225 < F.shape[1]
    G = F.T @ F
    target = W @ np.linspace(0.01, 0.09, 9) - np.einsum("ij,ij->i", W, W)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(G, F.T @ target)
    theta = np.linalg.solve(G + 1e-8 * np.eye(len(G)), F.T @ target)
    assert np.all(np.isfinite(theta))
    assert np.abs(F @ theta - target).max() < 1e-8 * np.abs(target).max()


def test_sgd_from_zero_never_moves_in_directions_the_features_cannot_tell_apart():
    """Why redundancy is safe for the agent: theta starts at 0 and every step
    adds a multiple of a feature row, so theta stays in the row space of F.
    Its component in F's null space stays at rounding level."""
    F = spline_additive_pairwise_features(generate_weight_grid(9, 0.25), dtype=np.float64)
    agent = LinearQAgent(F, alpha=0.005, epsilon0=1.0, beta=0.0, rng=np.random.default_rng(0),
                         greedy_refresh_every=100)
    noise = np.random.default_rng(1)
    for _ in range(3_000):
        agent.update(agent.select_action(), float(noise.normal()))
    _, s, Vt = np.linalg.svd(F)
    null = Vt[np.sum(s > s.max() * 1e-10):]
    assert np.abs(null @ agent.theta).max() < 1e-12 * max(np.abs(agent.theta).max(), 1.0)


@pytest.mark.parametrize("kwargs", [dict(n_basis=3, degree=3), dict(n_basis=True), dict(degree=-1), dict(n_basis=5.0)])
def test_spline_rejects_bad_basis_settings(kwargs):
    W = generate_weight_grid(9, 0.25)
    with pytest.raises(ValueError):
        spline_additive_pairwise_features(W, **kwargs)
