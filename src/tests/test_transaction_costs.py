import numpy as np

from src.transaction_costs import optimal_policy, switching_cost


def test_quadratic_reference_trade_and_no_trade():
    grid = np.array([[1, 0], [0.8, 0.2], [0.6, 0.4]])
    cost = switching_cost(grid, 2)
    np.testing.assert_array_equal(cost.diagonal(), np.zeros(3))
    np.testing.assert_allclose(cost[0], [0, 2e-4, 8e-4])
    np.testing.assert_allclose(cost, cost.T)


def test_discounted_oracle_accounts_for_the_next_holding():
    rewards = np.array([0.0, 1.0])
    cost = np.array([[0.0, 3.0], [3.0, 0.0]])
    np.testing.assert_array_equal(optimal_policy(rewards, cost, 0), [0, 1])
    np.testing.assert_array_equal(optimal_policy(rewards, cost, 0.95), [1, 1])
    np.testing.assert_array_equal(optimal_policy(rewards, np.zeros((2, 2)), 0.95), [1, 1])
