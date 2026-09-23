import numpy as np
from itertools import combinations

def generate_weight_grid(n_assets: int = 9, increment: float = 0.2) -> np.ndarray:
    """
    Generate every long-only, fully invested portfolio

    Returns: weights -> shape: (C(n_assets - 1 + total_units, n_assets - 1) x n_assets)
    """

    portfolios = []

    total_units = round(1.0 / increment) # <- investment units

    for divider_positions in combinations(range(n_assets - 1 + total_units), n_assets - 1):

        # Defines boundaries for assets in portfolio in the list with weights
        boundaries = [-1] + list(divider_positions) + [n_assets - 1 + total_units]

        # All possible distribution of weights based on boundaries
        units = [boundaries[i + 1] - boundaries[i] - 1 for i in range(n_assets)]
        portfolios.append(units)

    weights = np.array(portfolios) * increment
    return weights