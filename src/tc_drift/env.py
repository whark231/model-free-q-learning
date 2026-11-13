"""Three-asset market with weight drift (Albert). Phase 1: signatures only.

In plain terms: draws daily excess returns for SPY, IEF and GLD from the
calibrated sub-block of mu and Sigma, and plays out one 63-day block from a
starting portfolio. In the drift mode the shares are fixed after the trade, so
the weights move with each day's returns; in the no-drift mode the weights are
reset every day (the baseline's convention) and the next state is the
post-trade portfolio.

The interface splits drawing returns from scoring them (docs/methods/
TC_DRIFT_DESIGN.md §10, conflict C3), so the oracle can score 5,151 starting
portfolios on the SAME fixed paths (common random numbers) and the training
loop, the evaluation and the oracle all use one scoring function.
"""

from pathlib import Path

import numpy as np

from src.environment import PROCESSED_DIR
from src.tc_drift.spec import BLOCK_LENGTH, Market


def load_market(processed_dir: Path = PROCESSED_DIR) -> Market:
    """The (3,) mu and (3, 3) Sigma sub-block, in spec.ASSETS order, from
    src.environment.load_calibration()."""
    raise NotImplementedError("Phase 2 (Albert)")


def draw_returns(market: Market, rng: np.random.Generator, m: int,
                 block_length: int = BLOCK_LENGTH) -> np.ndarray:
    """m independent blocks of daily excess returns, shape (m, block_length, 3).
    Draws the same numbers whatever portfolio is later scored on them."""
    raise NotImplementedError("Phase 2 (Albert)")


def block_outcome(w0: np.ndarray, excess_returns: np.ndarray, gamma: float,
                  drift: bool) -> tuple[np.ndarray, np.ndarray]:
    """Score post-trade weights w0 (m, 3) on returns (m, B, 3).

    Returns (reward (m,), w_end (m, 3)). reward is the realized mean minus
    (gamma / 2) times the realized variance (ddof = 1) of the B daily
    portfolio excess returns, each day using that day's weights. With
    drift=False the weights stay w0 every day, reward equals
    src.rewards.episodic_mean_variance_reward row by row, and w_end = w0.
    """
    raise NotImplementedError("Phase 2 (Albert)")


def simulate_block(w0: np.ndarray, rng: np.random.Generator, B: int, drift: bool, *,
                   market: Market, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """draw_returns then block_outcome: (reward (m,), w_end (m, 3)). The form
    the Phase 1 prompt proposed, plus the market and gamma it needs."""
    raise NotImplementedError("Phase 2 (Albert)")
