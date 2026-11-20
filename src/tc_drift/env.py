"""Three-asset market with weight drift (Albert).

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

Assumptions (TC_DRIFT_DESIGN.md §3 and C2):
- Drift uses excess returns only. Adding the risk-free rate would move the
  end-of-block weights by at most 7.1e-6 (§12), and the assignment defines the
  environment over excess returns.
- Returns are Gaussian, drawn by src.environment.ReturnSimulator, so from the
  same generator they are the same numbers the baseline would draw.

Environment and benchmark only: the agent must not import this module
(src/tests/test_tc_drift_spec.py allowlist).
"""

from pathlib import Path

import numpy as np

from src.environment import MIN_EPISODE_LENGTH, PROCESSED_DIR, ReturnSimulator, load_calibration
from src.tc_drift.spec import ASSET_INDEX, BLOCK_LENGTH, N_ASSETS, Market, check_holdings


def load_market(processed_dir: Path = PROCESSED_DIR) -> Market:
    """The (3,) mu and (3, 3) Sigma sub-block, in spec.ASSETS order, from
    src.environment.load_calibration().

    load_calibration checks the ticker labels against TICKERS, and
    spec.ASSET_INDEX holds SPY, IEF and GLD's positions in TICKERS, so the
    sub-block cannot pick up the wrong assets. Both arrays are read-only copies,
    so no caller can change the market in place.
    """
    mu9, sigma9 = load_calibration(processed_dir)
    idx = list(ASSET_INDEX)
    mu = mu9[idx].copy()
    sigma = sigma9[np.ix_(idx, idx)].copy()
    mu.flags.writeable = False
    sigma.flags.writeable = False
    return Market(mu=mu, sigma=sigma)


def draw_returns(market: Market, rng: np.random.Generator, m: int,
                 block_length: int = BLOCK_LENGTH) -> np.ndarray:
    """m independent blocks of daily excess returns, shape (m, block_length, 3).
    Draws the same numbers whatever portfolio is later scored on them.

    Uses ReturnSimulator.sample_episodes, so from the same generator state the
    draws equal the baseline's m episodes of block_length days. It validates mu,
    Sigma (symmetric, positive definite), m, block_length (at least 2, the
    ddof = 1 variance needs two days) and the generator type.
    """
    return ReturnSimulator(market.mu, market.sigma, rng).sample_episodes(m, block_length)


def block_outcome(w0: np.ndarray, excess_returns: np.ndarray, gamma: float,
                  drift: bool) -> tuple[np.ndarray, np.ndarray]:
    """Score post-trade weights w0 (m, 3) on returns (m, B, 3).

    Returns (reward (m,), w_end (m, 3)). reward is the realized mean minus
    (gamma / 2) times the realized variance (ddof = 1) of the B daily
    portfolio excess returns, each day using that day's weights. With
    drift=False the weights stay w0 every day, reward equals
    src.rewards.episodic_mean_variance_reward row by row, and w_end = w0.

    w0 may also be a single portfolio, shape (3,) or (1, 3), scored on all m
    paths (the oracle scores each lattice point on the same M paths).

    Memory: with drift on, the temporaries take about 10 KB per path at
    B = 63. Score the oracle's lattice one point at a time (about 11 MB per
    call at M = 2,000) or in chunks of at most about 1e5 rows; tiling all
    5,151 points times 2,000 paths into one call would need about 80 GB.

    Drift: with shares fixed, the value of asset i at the start of day t is
    w0_i * prod_{s<t} (1 + e_{s,i}), so day t's weights are those values
    renormalized, the portfolio's excess return is sum_i w_{t,i} e_{t,i}, and
    w_end is the renormalized value after day B. The weights stay long-only
    and sum to one because every gross return 1 + e is required to be positive.
    """
    e = _check_returns(excess_returns)
    m, n_days, _ = e.shape
    w = check_holdings(w0)
    if w.shape[0] == 1 and m != 1:
        w = np.broadcast_to(w, (m, N_ASSETS))
    elif w.shape[0] != m:
        raise ValueError(f"w0 has {w.shape[0]} rows but excess_returns has {m} paths")
    if isinstance(gamma, bool) or not np.isfinite(gamma) or gamma < 0:
        raise ValueError(f"gamma must be finite and nonnegative, got {gamma!r}")
    if not isinstance(drift, (bool, np.bool_)):
        raise TypeError(f"drift must be True or False, got {type(drift).__name__}")

    if drift:
        if (e <= -1.0).any():
            raise ValueError("a daily excess return of -100% or below makes the drifted weights undefined")
        growth = np.cumprod(1.0 + e, axis=1)                        # value after day t per unit held
        value = np.concatenate([np.ones((m, 1, N_ASSETS)), growth], axis=1) * w[:, None, :]
        weights = value / value.sum(axis=2, keepdims=True)          # (m, B + 1, 3), day 1 .. after day B
        portfolio = np.einsum("mti,mti->mt", weights[:, :n_days], e)
        w_end = weights[:, n_days]
    else:
        portfolio = np.einsum("mti,mi->mt", e, w)
        w_end = np.array(w, dtype=float, copy=True)

    reward = portfolio.mean(axis=1) - 0.5 * gamma * portfolio.var(axis=1, ddof=1)
    return reward, w_end


def simulate_block(w0: np.ndarray, rng: np.random.Generator, B: int, drift: bool, *,
                   market: Market, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """draw_returns then block_outcome: (reward (m,), w_end (m, 3)). The form
    the Phase 1 prompt proposed, plus the market and gamma it needs. One block
    of B days is drawn per row of w0."""
    m = check_holdings(w0).shape[0]
    return block_outcome(w0, draw_returns(market, rng, m, B), gamma, drift)


def _check_returns(excess_returns) -> np.ndarray:
    e = np.asarray(excess_returns, dtype=float)
    if e.ndim != 3 or e.shape[2] != N_ASSETS:
        raise ValueError(f"excess_returns must have shape (m, B, {N_ASSETS}), got {e.shape}")
    if e.shape[0] < 1 or e.shape[1] < MIN_EPISODE_LENGTH:
        raise ValueError(f"need at least one path of at least {MIN_EPISODE_LENGTH} days, got {e.shape}")
    if not np.isfinite(e).all():
        raise ValueError("excess_returns must be finite")
    return e
