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

    # Check the KKT conditions directly rather than using iteration count as a
    # proxy for "did not converge" -- nit<=1 is both a false alarm (a symmetric
    # problem whose optimum IS the equal-weight start correctly stops in one
    # iteration) and a missed failure (a wrong interior point can take several
    # iterations to reach). At the long-only optimum, held assets (0 < w_i)
    # share one gradient value (the budget constraint's multiplier) and no
    # asset at its lower bound can exceed it.
    grad = mu_scaled - gamma * sigma_scaled @ result.x
    held = result.x > 1e-8
    lam = grad[held].mean()
    residual = max(np.abs(grad[held] - lam).max(), (grad[~held] - lam).max(initial=0.0))
    # Relative, not absolute: residual scales linearly with whatever
    # magnitude of mu/sigma the caller passes in (confirmed: scaling inputs
    # by 1000x scales the raw residual by ~1000x too), so an absolute
    # threshold breaks the scale-invariance this function otherwise has.
    grad_scale = max(np.abs(grad).max(), 1e-12)
    if residual / grad_scale > 1e-4:
        raise RuntimeError(f"Optimizer result fails the optimality check (relative KKT residual {residual / grad_scale:.1e})")

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
    return grid[np.argmax(mean_var_scores)].copy()
