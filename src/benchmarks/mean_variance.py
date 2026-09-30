import numpy as np
from scipy.optimize import minimize
from src.action_space.grid import generate_weight_grid

def closed_form_optimal_weights(mu: np.ndarray, sigma: np.ndarray, gamma: float) -> np.ndarray:
    """
    Derived via Lagrangian:
    A = 1' Sigma^{-1} 1
        B = 1' Sigma^{-1} mu
        lambda = (B - gamma) / A
        w* = (1/gamma) Sigma^{-1} (mu - lambda * 1)

    NOTE: allows negative weights (short positions)
    """

    n = mu.shape[0]
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

    SCALE = 252  # trading days/year -- gamma is frequency-invariant, so scaling
                 # mu and sigma by a positive constant leaves the argmax unchanged
                 # and only rescales the objective to a magnitude ftol handles well
    mu_scaled = mu * SCALE
    sigma_scaled = sigma * SCALE

    n = mu.shape[0]

    # objective helper
    neg_obj = lambda w: -(w @ mu_scaled - 0.5 * gamma * (w @ sigma_scaled @ w))


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
        options={"ftol": 1e-12, "maxiter": 1000}
    )

    if not result.success:
        raise RuntimeError(f"Optimizer failed: {result.message}")

    # SLSQP can report success after a single iteration when ftol is large
    # relative to the objective's scale -- at daily frequency that meant
    # silently returning the starting guess. This turns that into a loud failure.
    if result.nit <= 1:
        raise RuntimeError(
            f"Optimizer stopped after {result.nit} iteration(s); ftol is likely "
            "too large relative to the objective scale."
        )

    return result.x

# Equal weight benchmark
def equal_weights(n_assets: int = 9):
    return np.full(n_assets, 1.0 / n_assets)

# Discrete action-space
def best_discrete_weights(mu: np.ndarray, sigma: np.ndarray, gamma: float, increment: float = 0.2, grid: np.ndarray | None = None) -> np.ndarray:
    if grid is None:
        grid = generate_weight_grid(mu.shape[0], increment=increment)

    if grid.shape[1] != mu.shape[0]:
        raise ValueError(f"grid has {grid.shape[1]} assets but mu has {mu.shape[0]}")
    
    mean_var_scores = grid @ mu - 0.5 * gamma * ((grid @ sigma) * grid).sum(axis=1)
    return grid[np.argmax(mean_var_scores)]
