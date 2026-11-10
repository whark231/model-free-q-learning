import numpy as np
from itertools import combinations

def generate_weight_grid(n_assets: int = 9, increment: float = 0.2) -> np.ndarray:
    """
    Generate every long-only, fully invested portfolio

    Returns: weights -> shape: (C(n_assets - 1 + total_units, n_assets - 1) x n_assets)
    """

    portfolios = []

    # The grid only exists when the increment divides 1 exactly. Rounding 1/0.3
    # to 3 would silently build a one-third grid instead (AUDIT_2026-11-06.md L1).
    if not np.isfinite(increment) or increment <= 0:
        raise ValueError(f"increment must be positive and finite, got {increment!r}")
    units_exact = 1.0 / increment
    total_units = round(units_exact) # <- investment units
    if total_units < 1 or abs(units_exact - total_units) > 1e-9 * total_units:
        raise ValueError(f"1/increment must be a whole number of units; {increment!r} gives {units_exact!r}")

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