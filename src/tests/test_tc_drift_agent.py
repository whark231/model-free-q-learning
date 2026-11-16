"""Tests for the drift agent and its feature map (src/tc_drift/agent.py, features.py).

In plain terms: the feature map is rebuilt with scipy's B-splines and explicit
loops; the agent's greedy choice is recomputed from its parameters with
independent code; one learning step is checked by hand; and a noise-free toy
world with known costs checks that learning recovers the costs and the best
portfolio. The leakage contract is in test_tc_drift_spec.py.
"""

import numpy as np
import pytest

from src.features import bspline_basis
from src.tc_drift import agent as agent_module
from src.tc_drift import features, spec
from src.tc_drift.agent import DriftAgent

HP = {"alpha_value": 0.05, "alpha_cost": 0.05, "beta": 0.0, "target_refresh_every": 1}
TARGETS = spec.target_grid()


def make(seed=0, **overrides):
    return DriftAgent(np.random.default_rng(seed), **{**HP, **overrides})


# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------

def _loop_features(W, n_basis=5, degree=3):
    """scipy's basis and explicit loops; pair products kept when t_k + t_l < 1."""
    """scipy's basis and explicit loops; a pair product is kept when it is nonzero
    somewhere on a fine sample of the simplex (found by search, not by the knot rule)."""
    B = bspline_basis(W.ravel(), n_basis, degree).reshape(len(W), 3, n_basis)
    probe = spec.lattice(0.005)
    P = bspline_basis(probe.ravel(), n_basis, degree).reshape(len(probe), 3, n_basis)
    cols = [B[:, i, k] for i in range(3) for k in range(n_basis)]
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        cols += [B[:, i, k] * B[:, j, l] for k in range(n_basis) for l in range(n_basis)
                 if (P[:, i, k] * P[:, j, l]).max() > 1e-12]
    return np.stack(cols, axis=1)


@pytest.mark.parametrize("n_basis, degree", [(5, 3), (4, 3), (7, 3), (6, 2), (12, 3), (4, 1)])
def test_features_match_scipy_and_explicit_loops(n_basis, degree):
    W = np.vstack([spec.lattice(0.02), np.random.default_rng(0).dirichlet(np.ones(3), 500), np.eye(3)])
    F = features.spline_features(W, n_basis, degree)
    np.testing.assert_allclose(F, _loop_features(W, n_basis, degree), rtol=0, atol=1e-14)
    assert F.shape[1] == features.n_features(n_basis, degree)


def test_feature_counts_match_the_design():
    F = features.spline_features(TARGETS)
    assert F.shape == (spec.N_TARGETS, 87)
    assert np.linalg.matrix_rank(F) == 45
    fine = features.spline_features(spec.lattice(0.005))
    assert (fine >= 0).all()
    assert (fine ** 2).sum(axis=1).max() <= 6 + 1e-12
    assert (fine.max(axis=0) > 0).all()      # no kept column is zero everywhere on the simplex


def test_columns_do_not_depend_on_the_input():
    W = np.random.default_rng(1).dirichlet(np.ones(3), 50)
    batch = features.spline_features(W)
    for i in range(0, 50, 7):
        np.testing.assert_array_equal(features.spline_features(W[i]), batch[i:i + 1])


def test_features_reproduce_any_quadratic():
    # The map nests every quadratic on the simplex (TC_DRIFT_DESIGN.md §6), so a
    # random quadratic is fitted exactly. It is random, not the reward.
    rng = np.random.default_rng(2)
    A = rng.normal(size=(3, 3))
    W = spec.lattice(0.02)
    y = np.einsum("ni,ij,nj->n", W, A + A.T, W) + W @ rng.normal(size=3)
    F = features.spline_features(W)
    coef, *_ = np.linalg.lstsq(F, y, rcond=None)
    assert np.abs(F @ coef - y).max() < 1e-10


@pytest.mark.parametrize("W", [np.array([0.5, 0.5]), np.array([0.6, 0.6, -0.2]), np.array([0.5, 0.3, 0.3]),
                               np.array([np.nan, 0.5, 0.5])])
def test_features_reject_bad_weights(W):
    with pytest.raises(ValueError):
        features.spline_features(W)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_rejects_unknown_and_missing_hyperparameters():
    with pytest.raises(TypeError):
        DriftAgent(np.random.default_rng(0), **HP, gamma=5.0)
    with pytest.raises(TypeError):
        DriftAgent(np.random.default_rng(0), alpha_value=0.01)
    with pytest.raises(TypeError):
        DriftAgent(0, **HP)


@pytest.mark.parametrize("override", [
    {"alpha_value": 0.0}, {"alpha_value": -0.1}, {"alpha_cost": -0.1}, {"beta": -1.0}, {"l2": -1.0},
    {"alpha_value": np.nan}, {"alpha_value": True}, {"target_refresh_every": 0}, {"target_refresh_every": 1.5},
    {"n_basis": 2}, {"degree": 0}, {"alpha_value": 0.3, "alpha_cost": 0.05},   # 0.3 * 6 + 0.05 * 8 = 2.2 breaks the bound
])
def test_rejects_bad_hyperparameter_values(override):
    with pytest.raises(ValueError):
        make(**override)


def test_every_hyperparameter_is_accepted():
    a = make(l2=1e-3, n_basis=6, degree=3, target_refresh_every=50, beta=1e-6)
    assert set(a.hyperparameters) == DriftAgent.HYPERPARAMETERS


# ---------------------------------------------------------------------------
# Greedy choice, ties, exploration
# ---------------------------------------------------------------------------

def _independent_q(a, s):
    phi_t = features.spline_features(TARGETS, a.n_basis, a.degree)
    d1 = np.array([sum(abs(t[i] - s[i]) for i in range(3)) for t in TARGETS])
    d2 = np.array([sum((t[i] - s[i]) ** 2 for i in range(3)) for t in TARGETS])
    trades = phi_t @ a.theta - a.cost_linear * d1 - a.cost_quadratic * d2
    hold = features.spline_features(s, a.n_basis, a.degree)[0] @ a.theta
    return trades, hold


def _trained(seed=3, steps=300):
    a = make(seed)
    rng = np.random.default_rng(seed + 100)
    s = np.array([1 / 3, 1 / 3, 1 / 3])
    for _ in range(steps):
        act = a.select_action(s)
        nxt = rng.dirichlet(np.ones(3) * 20)
        a.update(s, act, float(rng.normal(0.0, 0.1) - 0.02 * (act != spec.HOLD)), nxt)
        s = nxt
    return a


def test_greedy_matches_an_independent_argmax():
    a = _trained()
    assert a.cost_linear > 0 or a.cost_quadratic > 0
    for s in np.random.default_rng(4).dirichlet(np.ones(3), 40):
        trades, hold = _independent_q(a, s)
        expected = spec.HOLD if hold >= trades.max() - 1e-12 else int(trades.argmax())
        assert a.greedy_action(s) == expected
        np.testing.assert_allclose(a.q_values(s), np.append(trades, hold), rtol=0, atol=1e-12)


def test_hold_wins_ties():
    a = make()
    for s in [TARGETS[0], TARGETS[700], np.array([0.123, 0.456, 0.421])]:
        assert a.greedy_action(s) == spec.HOLD           # every action scores 0 at the start
    b = _trained()
    trades, hold = _independent_q(b, TARGETS[700])
    assert trades[700] == pytest.approx(hold, abs=1e-14)  # trading to the target already held = HOLD
    assert b.greedy_action(TARGETS[700]) != 700


def test_full_exploration_follows_the_documented_draws():
    a = make(seed=5)
    ref = np.random.default_rng(5)
    s = TARGETS[10]
    actions = [a.select_action(s) for _ in range(20000)]
    expected = []
    for _ in range(20000):
        ref.random()
        expected.append(int(ref.integers(spec.N_ACTIONS)))
    assert actions == expected
    assert spec.HOLD in actions and max(actions) == spec.HOLD


def test_epsilon_schedule():
    a = make(beta=1e-3)
    assert a.epsilon == 1.0
    for _ in range(1000):
        a.select_action(TARGETS[0])
    assert a.epsilon == pytest.approx(np.exp(-1.0))
    greedy = make(beta=50.0)
    greedy.select_action(TARGETS[0])
    assert all(greedy.select_action(TARGETS[0]) == spec.HOLD for _ in range(100))


def test_greedy_action_changes_nothing():
    a = _trained()
    state = a.rng.bit_generator.state
    t, theta = a.epsilon, a.theta
    for s in np.random.default_rng(6).dirichlet(np.ones(3), 10):
        a.greedy_action(s)
        a.q_values(s)
    assert a.rng.bit_generator.state == state and a.epsilon == t
    np.testing.assert_array_equal(a.theta, theta)


# ---------------------------------------------------------------------------
# One learning step by hand; the frozen target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reward", [0.3, -0.3])
def test_one_trade_update_by_hand(reward):
    a = make(alpha_value=0.1, alpha_cost=0.05)
    s = np.array([0.5, 0.3, 0.2])
    action = 600
    u = TARGETS[action] - s
    d = np.array([np.abs(u).sum(), (u * u).sum()])
    a.update(s, action, reward, np.array([0.2, 0.3, 0.5]))
    # Start: every parameter 0, so Qhat = 0 and the target is the reward.
    phi = features.spline_features(TARGETS[action])[0]
    np.testing.assert_allclose(a.theta, 0.1 * reward * phi, rtol=1e-14)
    expected_cost = np.maximum(-0.05 * reward * d, 0.0)   # a reward above the estimate pushes costs down, clamped at 0
    assert (a.cost_linear, a.cost_quadratic) == pytest.approx(tuple(expected_cost), rel=1e-14)


def test_hold_update_moves_only_the_value_model():
    a = make(alpha_value=0.1, alpha_cost=0.05, l2=0.5)
    s = np.array([0.5, 0.3, 0.2])
    a.update(s, spec.HOLD, -0.4, s)
    np.testing.assert_allclose(a.theta, 0.1 * -0.4 * features.spline_features(s)[0], rtol=1e-14)
    assert a.cost_linear == 0.0 and a.cost_quadratic == 0.0


class ReferenceAgent:
    """The update written straight from TC_DRIFT_DESIGN.md section 6, with plain loops
    for the distances, to step side by side with DriftAgent."""

    def __init__(self, alpha_value, alpha_cost, l2, refresh):
        self.av, self.ac, self.l2, self.refresh = alpha_value, alpha_cost, l2, refresh
        self.phi_t = features.spline_features(TARGETS)
        self.theta = np.zeros(self.phi_t.shape[1])
        self.c = [0.0, 0.0]
        self.frozen = (self.theta.copy(), list(self.c))
        self.n = 0
        self.hold_was_max = False

    @staticmethod
    def dist(t, s):
        return sum(abs(t[i] - s[i]) for i in range(3)), sum((t[i] - s[i]) ** 2 for i in range(3))

    def q(self, s, theta, c):
        trades = []
        for a in range(spec.N_TARGETS):
            d1, d2 = self.dist(TARGETS[a], s)
            trades.append(self.phi_t[a] @ theta - c[0] * d1 - c[1] * d2)
        return np.array(trades), features.spline_features(s)[0] @ theta

    def update(self, s, a, r, s_next):
        trades, hold = self.q(s_next, *self.frozen)
        self.hold_was_max |= hold > trades.max()
        y = r + spec.DISCOUNT * max(trades.max(), hold)
        if a == spec.HOLD:
            x, d = features.spline_features(s)[0], (0.0, 0.0)
        else:
            x, d = self.phi_t[a], self.dist(TARGETS[a], s)
        e = y - (x @ self.theta - self.c[0] * d[0] - self.c[1] * d[1])
        self.theta = self.theta + self.av * (e * x - self.l2 * self.theta)
        self.c = [max(self.c[k] - self.ac * e * d[k], 0.0) for k in range(2)]   # dQ/dc_k = -d_k, then D11
        self.n += 1
        if self.n % self.refresh == 0:
            self.frozen = (self.theta.copy(), list(self.c))


def test_updates_match_a_reference_written_from_the_design():
    """Off-grid states, HOLD and trades, l2 > 0 and a frozen copy refreshed every 3
    updates, so the target's HOLD term, l2 and the frozen costs all matter."""
    a = make(seed=11, alpha_value=0.05, alpha_cost=0.05, l2=0.3, target_refresh_every=3)
    ref = ReferenceAgent(0.05, 0.05, 0.3, 3)
    rng = np.random.default_rng(12)
    s = rng.dirichlet(np.ones(3) * 3)
    for step in range(40):
        action = spec.HOLD if rng.random() < 0.4 else int(rng.integers(spec.N_TARGETS))
        s_next = rng.dirichlet(np.ones(3) * 3) if step % 5 else s.copy()   # sometimes stay put: HOLD then tops the max
        r = float(rng.normal(0.0, 1.0) - 0.5 * (action != spec.HOLD))
        a.update(s, action, r, s_next)
        ref.update(s, action, r, s_next)
        np.testing.assert_allclose(a.theta, ref.theta, rtol=1e-11, atol=1e-13)
        assert (a.cost_linear, a.cost_quadratic) == pytest.approx(tuple(ref.c), rel=1e-11, abs=1e-13)
        s = s_next
    assert ref.hold_was_max                        # the HOLD term decided at least one target
    assert ref.c[0] > 0 and ref.c[1] > 0
    assert ref.c != ref.frozen[1]                  # online and frozen costs differed at the end


def test_tie_tolerance_at_the_real_reward_scale():
    """Values are about 4e-3 and decisions differ by about 1e-8: a trade that beats
    HOLD by 1e-10 must be taken, one that beats it by rounding (1e-16) must not."""
    a = make()
    hold = 4e-3
    trades = np.full(spec.N_TARGETS, hold - 1e-6)
    for margin, expected in [(1e-10, 5), (1e-16, spec.HOLD), (0.0, spec.HOLD)]:
        trades[5] = hold + margin
        a._scores = lambda *args, t=trades.copy(): (t, hold)   # isolate the tie rule
        assert a.greedy_action(TARGETS[0]) == expected


def test_state_with_clipped_rounding_is_accepted():
    s = np.array([-0.9e-9, 0.5, 0.5 + 1.4e-9])        # passes check_holdings; the clip alone breaks the sum
    a = _trained()
    a.greedy_action(s)
    a.update(s, 3, 0.1, s)
    a.holding_values(s[None, :])


@pytest.mark.parametrize("bad", [
    {"state": np.array([0.5, 0.5])}, {"next_state": np.array([0.7, 0.7, -0.4])}, {"action": 1327},
    {"action": True}, {"action": 3.0}, {"reward_net": np.nan},
])
def test_update_rejects_bad_inputs(bad):
    args = {"state": TARGETS[0], "action": 5, "reward_net": 0.1, "next_state": TARGETS[1], **bad}
    with pytest.raises(ValueError):
        make().update(**args)


def test_holding_values_are_the_value_model():
    a = _trained()
    W = spec.lattice(0.05)
    np.testing.assert_allclose(a.holding_values(W), features.spline_features(W) @ a.theta, rtol=1e-14)


# ---------------------------------------------------------------------------
# Learning in a noise-free toy world with known costs
# ---------------------------------------------------------------------------

def test_learns_costs_and_a_near_optimal_policy_without_noise():
    """No drift and no noise: holding x for a block pays f(x) = -||x - x_star||^2
    (O(1) units so the toy runs fast), trading costs k1 ||u||_1 + k2 ||u||_2^2,
    and the next state is the post-trade portfolio. The exact optimum comes
    from the old grid solver. The agent must recover k1 and k2 to within 25%
    and lose at most 1% of what never trading loses against the optimum."""
    from src.transaction_costs import optimal_policy

    x_star = np.array([0.5, 0.2, 0.3])
    k1, k2 = 0.02, 0.05
    n = spec.N_TARGETS
    f = -((TARGETS - x_star) ** 2).sum(axis=1)
    U = TARGETS[None, :, :] - TARGETS[:, None, :]                 # U[s, a] = t_a - t_s
    cost = k1 * np.abs(U).sum(axis=2) + k2 * (U * U).sum(axis=2)

    def value(policy):                                            # policy[s] = destination index
        V = np.zeros(n)
        for _ in range(3000):
            V = f[policy] - cost[np.arange(n), policy] + spec.DISCOUNT * V[policy]
        return V

    best = value(optimal_policy(f, cost, spec.DISCOUNT))
    never = f / (1 - spec.DISCOUNT)

    a = make(seed=7, alpha_value=0.05, alpha_cost=0.05, beta=4e-5, target_refresh_every=200)
    rng = np.random.default_rng(8)
    s = TARGETS[0]
    for step in range(60_000):
        if step % 50 == 0:
            s = TARGETS[rng.integers(n)]                          # restart, as train.py will
        act = a.select_action(s)
        x = spec.post_trade_weights(s, act)
        u = x - s
        a.update(s, act, -((x - x_star) ** 2).sum() - (k1 * np.abs(u).sum() + k2 * (u * u).sum()), x)
        s = x

    assert a.cost_linear == pytest.approx(k1, rel=0.25)
    assert a.cost_quadratic == pytest.approx(k2, rel=0.25)
    greedy = np.array([a.greedy_action(t) for t in TARGETS])
    greedy = np.where(greedy == spec.HOLD, np.arange(n), greedy)
    assert (best - value(greedy)).mean() <= 0.01 * (best - never).mean()
