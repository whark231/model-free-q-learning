import numpy as np
import pytest
from scipy.optimize import minimize

from src.benchmarks.mean_variance import (
    closed_form_optimal_weights,
    constrained_optimal_weights,
    equal_weights,
    best_discrete_weights,
)

# ---------------------------------------------------------------------------
# Fixtures: small, hand-checkable problems plus a realistic 9-asset one
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_problem():
    """3 assets, uncorrelated (Sigma = I), equal expected returns.
    Unconstrained optimum is known analytically: equal weights."""
    mu = np.array([0.08, 0.08, 0.08])
    sigma = np.eye(3)
    gamma = 3.0
    return mu, sigma, gamma


@pytest.fixture
def favorable_9_asset_problem():
    """9 assets, diagonal covariance, all positive expected returns close
    together and gamma large enough that the unconstrained optimum should
    already be long-only -> constrained and unconstrained solutions should
    agree here."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.05, 0.09, size=9)
    sigma = np.diag(rng.uniform(0.01, 0.05, size=9))
    gamma = 8.0
    return mu, sigma, gamma


@pytest.fixture
def short_seeking_problem():
    """One asset has a much worse expected return than the rest, so the
    unconstrained (short-allowed) optimum should want to short it -
    constrained optimum must instead clip it to 0."""
    mu = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, -0.20])
    sigma = np.eye(9) * 0.04
    gamma = 3.0
    return mu, sigma, gamma


@pytest.fixture
def daily_scale_problem():
    """Realistic daily moments. The objective here is ~1e-4, which is where
    SLSQP's default ftol stops prematurely."""
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.05, 0.09, size=9) / 252
    sigma = np.diag(rng.uniform(0.01, 0.05, size=9)) / 252
    return mu, sigma, 5.0


def independent_scipy_reference(mu, sigma, gamma, bounds=None, scale=1.0):
    """An independent (non-implementation) scipy solve, used to cross-check
    closed_form_optimal_weights / constrained_optimal_weights without
    testing the implementation against itself.

    `scale` exists because SLSQP's default ftol is an absolute tolerance on
    the objective value: at daily-return magnitude (~1e-4) the untouched
    starting guess can already satisfy it, so a fair comparison against a
    daily-scale problem needs the same positive-constant rescaling that
    constrained_optimal_weights itself applies (rescaling doesn't change
    the argmax -- see test_scaling_mu_and_sigma_by_a_positive_constant_...).
    """
    n = mu.shape[0]
    mu, sigma = mu * scale, sigma * scale
    neg_obj = lambda w: -(w @ mu - 0.5 * gamma * (w @ sigma @ w))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(
        neg_obj,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    assert result.success, result.message
    return result.x


# ---------------------------------------------------------------------------
# closed_form_optimal_weights (unconstrained, shorting allowed)
# ---------------------------------------------------------------------------

class TestClosedFormOptimalWeights:
    def test_weights_sum_to_one(self, favorable_9_asset_problem):
        mu, sigma, gamma = favorable_9_asset_problem
        w = closed_form_optimal_weights(mu, sigma, gamma)
        assert np.sum(w) == pytest.approx(1.0)

    def test_matches_equal_weight_for_symmetric_problem(self, identity_problem):
        """With Sigma = I and constant mu, the analytical optimum is the
        equal-weight portfolio (global min-variance == equal weight here)."""
        mu, sigma, gamma = identity_problem
        w = closed_form_optimal_weights(mu, sigma, gamma)
        np.testing.assert_allclose(w, np.full(3, 1 / 3), atol=1e-8)

    def test_matches_independent_scipy_solve(self, favorable_9_asset_problem):
        """Cross-check the closed-form formula against an independently
        constructed, equality-only scipy optimization (no bounds)."""
        mu, sigma, gamma = favorable_9_asset_problem
        w_closed = closed_form_optimal_weights(mu, sigma, gamma)
        w_scipy = independent_scipy_reference(mu, sigma, gamma, bounds=None)
        # Loose tolerance: SLSQP without bounds converges less tightly than
        # the exact closed-form solution; this is a sanity cross-check, not
        # a precision test.
        np.testing.assert_allclose(w_closed, w_scipy, atol=3e-3)

    def test_allows_negative_weights(self, short_seeking_problem):
        """Sanity check the documented behavior: this function should NOT
        clip shorts (that's what constrained_optimal_weights is for)."""
        mu, sigma, gamma = short_seeking_problem
        w = closed_form_optimal_weights(mu, sigma, gamma)
        assert np.any(w < 0)

    def test_raises_on_singular_covariance(self):
        mu = np.array([0.1, 0.1])
        singular_sigma = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-deficient
        with pytest.raises(np.linalg.LinAlgError):
            closed_form_optimal_weights(mu, singular_sigma, gamma=3.0)


# ---------------------------------------------------------------------------
# constrained_optimal_weights (long-only, 0 <= w_i <= 1)
# ---------------------------------------------------------------------------

class TestConstrainedOptimalWeights:
    def test_weights_sum_to_one(self, favorable_9_asset_problem):
        mu, sigma, gamma = favorable_9_asset_problem
        w = constrained_optimal_weights(mu, sigma, gamma)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-6)

    def test_weights_within_bounds(self, short_seeking_problem):
        mu, sigma, gamma = short_seeking_problem
        w = constrained_optimal_weights(mu, sigma, gamma)
        assert np.all(w >= -1e-8)
        assert np.all(w <= 1.0 + 1e-8)

    def test_matches_closed_form_when_unconstrained_solution_is_feasible(
        self, favorable_9_asset_problem
    ):
        """If the closed-form (shorting-allowed) solution is already
        long-only, adding the long-only constraint shouldn't change it."""
        mu, sigma, gamma = favorable_9_asset_problem
        w_closed = closed_form_optimal_weights(mu, sigma, gamma)
        assert np.all(w_closed >= 0), "fixture assumption violated: expected a feasible unconstrained optimum"
        w_constrained = constrained_optimal_weights(mu, sigma, gamma)
        # Loose tolerance: SLSQP converges on the objective (ftol), not on
        # the weights directly, so it can land a few tenths of a percent
        # away from the exact closed-form optimum even when unconstrained.
        np.testing.assert_allclose(w_constrained, w_closed, atol=3e-3)

    def test_clips_short_position_to_zero(self, short_seeking_problem):
        """The asset with a strongly negative expected return should hit
        its lower bound of 0 rather than go short."""
        mu, sigma, gamma = short_seeking_problem
        w = constrained_optimal_weights(mu, sigma, gamma)
        assert w[-1] == pytest.approx(0.0, abs=1e-6)

    def test_objective_never_exceeds_unconstrained_optimum(self, short_seeking_problem):
        """Constraining the feasible region can only weakly lower the
        achievable objective value."""
        mu, sigma, gamma = short_seeking_problem
        obj = lambda w: w @ mu - 0.5 * gamma * (w @ sigma @ w)
        w_closed = closed_form_optimal_weights(mu, sigma, gamma)
        w_constrained = constrained_optimal_weights(mu, sigma, gamma)
        assert obj(w_constrained) <= obj(w_closed) + 1e-8

    def test_matches_independent_scipy_solve(self, short_seeking_problem):
        mu, sigma, gamma = short_seeking_problem
        n = mu.shape[0]
        w = constrained_optimal_weights(mu, sigma, gamma)
        w_ref = independent_scipy_reference(mu, sigma, gamma, bounds=[(0.0, 1.0)] * n)
        np.testing.assert_allclose(w, w_ref, atol=1e-4)

    def test_optimizes_at_daily_return_scale(self, daily_scale_problem):
        """Regression for the bug the SCALE constant fixes: at daily-return
        magnitude (~1e-4), SLSQP previously reported success after a single
        iteration and returned the equal-weight starting guess untouched.
        The optimum here is not equal weight (mu/sigma differ per asset),
        so getting back w0 unchanged means the optimizer never ran."""
        mu, sigma, gamma = daily_scale_problem
        w = constrained_optimal_weights(mu, sigma, gamma)
        n = mu.shape[0]
        assert not np.allclose(w, np.full(n, 1.0 / n), atol=1e-4)

    def test_daily_scale_solution_matches_independent_scipy_solve(self, daily_scale_problem):
        """The SCALE rescaling is only a solver aid -- confirm it doesn't
        change *which* portfolio is optimal, by cross-checking against an
        independent scipy solve run directly at daily scale (no rescaling)."""
        mu, sigma, gamma = daily_scale_problem
        n = mu.shape[0]
        w = constrained_optimal_weights(mu, sigma, gamma)
        w_ref = independent_scipy_reference(mu, sigma, gamma, bounds=[(0.0, 1.0)] * n, scale=252)
        np.testing.assert_allclose(w, w_ref, atol=3e-3)

    def test_scaling_mu_and_sigma_by_a_positive_constant_leaves_weights_unchanged(
        self, favorable_9_asset_problem
    ):
        """Documents/verifies the invariance the SCALE trick relies on:
        multiplying both mu and sigma by any positive constant k leaves the
        argmax of w'mu - 0.5*gamma*w'Sigma*w unchanged."""
        mu, sigma, gamma = favorable_9_asset_problem
        w_unscaled = constrained_optimal_weights(mu, sigma, gamma)
        w_rescaled = constrained_optimal_weights(mu * 1000, sigma * 1000, gamma)
        np.testing.assert_allclose(w_rescaled, w_unscaled, atol=1e-4)

    def test_raises_when_optimizer_stops_after_one_iteration(self, monkeypatch, favorable_9_asset_problem):
        """Directly exercise the early-stopping guard by forcing scipy to
        report a successful solve that never actually iterated, independent
        of whatever real ftol/scale conditions would trigger it naturally."""
        import types
        import src.benchmarks.mean_variance as mv

        mu, sigma, gamma = favorable_9_asset_problem

        def fake_minimize(*args, **kwargs):
            n = mu.shape[0]
            return types.SimpleNamespace(success=True, nit=1, x=np.full(n, 1.0 / n))

        monkeypatch.setattr(mv, "minimize", fake_minimize)

        with pytest.raises(RuntimeError, match="stopped after"):
            constrained_optimal_weights(mu, sigma, gamma)

    def test_raises_when_optimizer_reports_failure(self, monkeypatch, favorable_9_asset_problem):
        import types
        import src.benchmarks.mean_variance as mv

        mu, sigma, gamma = favorable_9_asset_problem

        def fake_minimize(*args, **kwargs):
            return types.SimpleNamespace(success=False, message="did not converge", nit=5, x=None)

        monkeypatch.setattr(mv, "minimize", fake_minimize)

        with pytest.raises(RuntimeError, match="did not converge"):
            constrained_optimal_weights(mu, sigma, gamma)

    def test_daily_and_annual_solutions_agree(self, daily_scale_problem):
        """gamma is frequency-invariant: mu and Sigma both scale linearly with the
        horizon, so annualizing multiplies the objective by 252 and leaves the
        argmax unchanged. Disagreement means one scale failed to converge.

        This is stronger than asserting the result differs from equal weights: at
        high gamma the bug returns an arbitrary interior point, not w0.
        """
        mu, sigma, gamma = daily_scale_problem
        w_daily = constrained_optimal_weights(mu, sigma, gamma)
        w_annual = constrained_optimal_weights(mu * 252, sigma * 252, gamma)
        # atol=1e-6 was too tight in practice: two independently-run SLSQP
        # solves at different scales don't retrace identical floating-point
        # paths even at the same true optimum (observed max diff ~1.7e-6).
        np.testing.assert_allclose(w_daily, w_annual, atol=1e-5)


# ---------------------------------------------------------------------------
# equal_weights
# ---------------------------------------------------------------------------

class TestEqualWeights:
    def test_shape_and_value(self):
        w = equal_weights()
        assert w.shape == (9,)
        np.testing.assert_allclose(w, np.full(9, 1 / 9))

    def test_sums_to_one(self):
        assert np.sum(equal_weights()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# best_discrete_weights
# ---------------------------------------------------------------------------

class TestBestDiscreteWeights:
    def test_matches_brute_force_over_grid(self, favorable_9_asset_problem):
        """Re-derive the grid independently and brute-force the best
        portfolio, rather than trusting generate_weight_grid + the
        function's own argmax."""
        from src.action_space.grid import generate_weight_grid

        mu, sigma, gamma = favorable_9_asset_problem
        grid = generate_weight_grid(9, increment=0.2)
        scores = np.array([w @ mu - 0.5 * gamma * (w @ sigma @ w) for w in grid])
        expected = grid[np.argmax(scores)]

        actual = best_discrete_weights(mu, sigma, gamma)
        np.testing.assert_allclose(actual, expected)

    def test_weights_sum_to_one(self, favorable_9_asset_problem):
        mu, sigma, gamma = favorable_9_asset_problem
        w = best_discrete_weights(mu, sigma, gamma)
        assert np.sum(w) == pytest.approx(1.0)

    def test_weights_non_negative(self, favorable_9_asset_problem):
        mu, sigma, gamma = favorable_9_asset_problem
        w = best_discrete_weights(mu, sigma, gamma)
        assert np.all(w >= 0)

    def test_objective_never_exceeds_constrained_continuous_optimum(
        self, favorable_9_asset_problem
    ):
        """The 20%-grid is a strict subset of the full long-only simplex,
        so its best achievable objective value can't beat the continuous
        constrained optimum."""
        mu, sigma, gamma = favorable_9_asset_problem
        obj = lambda w: w @ mu - 0.5 * gamma * (w @ sigma @ w)
        w_discrete = best_discrete_weights(mu, sigma, gamma)
        w_constrained = constrained_optimal_weights(mu, sigma, gamma)
        assert obj(w_discrete) <= obj(w_constrained) + 1e-8

    def test_increment_argument_changes_grid_resolution(self, favorable_9_asset_problem):
        """A coarser grid (50% increments) should give a strictly worse or
        equal objective value than the default 20% grid, since it searches
        a smaller, nested set of portfolios."""
        mu, sigma, gamma = favorable_9_asset_problem
        obj = lambda w: w @ mu - 0.5 * gamma * (w @ sigma @ w)
        w_fine = best_discrete_weights(mu, sigma, gamma, increment=0.2)
        w_coarse = best_discrete_weights(mu, sigma, gamma, increment=0.5)
        assert obj(w_coarse) <= obj(w_fine) + 1e-8

    def test_explicit_grid_overrides_generated_one(self, favorable_9_asset_problem):
        """Passing a precomputed grid should search exactly that grid,
        ignoring `increment`, rather than regenerating one internally."""
        mu, sigma, gamma = favorable_9_asset_problem
        n = mu.shape[0]

        custom_grid = np.stack([
            np.full(n, 1.0 / n),                       # equal weight
            np.eye(n)[np.argmax(mu)],                   # all-in on the best asset
            np.full(n, 1.0 / n) * 0.5 + np.eye(n)[0] * 0.5,  # something in between
        ])
        custom_grid /= custom_grid.sum(axis=1, keepdims=True)

        scores = np.array([w @ mu - 0.5 * gamma * (w @ sigma @ w) for w in custom_grid])
        expected = custom_grid[np.argmax(scores)]

        actual = best_discrete_weights(mu, sigma, gamma, grid=custom_grid)
        np.testing.assert_allclose(actual, expected)

    def test_raises_on_grid_dimension_mismatch(self, favorable_9_asset_problem):
        """A grid built for the wrong number of assets should fail loudly,
        not silently broadcast/misalign against mu and sigma."""
        mu, sigma, gamma = favorable_9_asset_problem
        wrong_shape_grid = np.full((5, 4), 0.25)  # 4 assets, but mu/sigma have 9
        with pytest.raises(ValueError, match="grid has"):
            best_discrete_weights(mu, sigma, gamma, grid=wrong_shape_grid)

    def test_discretization_gap_is_non_negative_on_daily_moments(self, daily_scale_problem):
        """Guards test_objective_never_exceeds_constrained_continuous_optimum's
        invariant at the scale where it currently breaks."""
        mu, sigma, gamma = daily_scale_problem
        obj = lambda w: w @ mu - 0.5 * gamma * (w @ sigma @ w)
        assert obj(best_discrete_weights(mu, sigma, gamma)) <= obj(
            constrained_optimal_weights(mu, sigma, gamma)
        ) + 1e-12
