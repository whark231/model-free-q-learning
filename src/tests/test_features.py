import numpy as np
import pytest

from src.action_space.grid import generate_weight_grid
from src.features import onehot_features, quadratic_features


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
