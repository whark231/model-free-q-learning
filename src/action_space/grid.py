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

    # Divide by the integer unit count rather than multiplying by the increment.
    # 0.2 has no exact binary representation, so 3 * 0.2 gives 0.6000000000000001
    # while 3 / 5 gives exactly 0.6. Matters wherever a weight is compared to a
    # literal or used as a dict key, e.g. mapping holdings back to an action
    # index in the transaction-cost extension.
    weights = np.array(portfolios) / total_units
    return weights