import numpy as np
from scipy.optimize import minimize
from src.action_space.grid import generate_weight_grid

def closed_Form_optimal_weights(mu: np.ndarray, sigma: np.ndarray, gamma: float) -> np.ndarray:
    """
    Derived via Lagrangian:
    A = 1' Sigma^{-1} 1
        B = 1' Sigma^{-1} mu
        lambda = (B - gamma) / A
        w* = (1/gamma) Sigma^{-1} (mu - lambda * 1)

    NOTE: allows negative weights (short positions)
    """

    n = mu.shape
    ones = np.ones(n)

    sigma_inv = np.linalg.inv(sigma)

    A = ones @ sigma_inv @ ones
    B = ones @ sigma_inv @ mu
    lam = (B - gamma) / A
    w_star = (1.0 / gamma) * (sigma_inv @ (mu - lam * ones))

    return w_star


def constrained_optimal_weights(mu: np.ndarray, sigma: np.ndarray, gamma: float) -> np.ndarray:
    """
    NOTE: Same objective function, but with long-only constraint

    New constraint 0 <= w_i <= 1
    """

    n = mu.shape[0]

    # objective helper
    neg_obj = lambda w: -(w @ mu - 0.5 * gamma * (w @ sigma @ w))


    # start with equal weight portfolio
    w0 = np.full(n, 1.0 / n)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        neg_obj,
        w0,
        method = "SLSQP", # sequential least squares programming
        bounds = bounds,
        constraints = constraints,
    )

    if not result.success:
        raise RuntimeError(f"Optimizer failed: {result.message}")

    return result.x

# Equal weight benchmark
def equal_weights(n_assets: int = 9):
    return np.full(n_assets, 1.0 / n_assets)

# Discrete action-space
def best_discrete_weights(mu: np.ndarray, sigma: np.ndarray, gamma: float) -> np.ndarray:
    grid = generate_weight_grid()
    mean_var_scores = grid @ mu - 0.5 * gamma * ((grid @ sigma) * grid).sum(axis=1)
    return grid[np.argmax(mean_var_scores)]
