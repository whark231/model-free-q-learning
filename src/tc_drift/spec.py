"""Frozen shared interface for the transaction-cost-with-drift extension.

In plain terms: this file is the contract between William's code (oracle,
agent, features) and Albert's code (environment, training, evaluation). It
fixes the constants, the action numbering, the cost formula, the random-stream
layout and the method names, so each person can build against it without
waiting for the other. Change it only through a pull request both approve
(docs/methods/TC_DRIFT_DESIGN.md §10).

Post hoc: designed after baseline-v1 and the first transaction-cost sweep were
seen. Label every result that uses it that way.

Leakage boundary: the agent may import the action and lattice helpers here
(the names in AGENT_SAFE_NAMES). It must not use the cost function, the cost
calibration, the settings or market classes, or GAMMA_HEADLINE: the agent
never receives mu, Sigma, kappa, lambda or gamma (enforced by
src/tests/test_tc_drift_spec.py, which catches accidental leaks, not
deliberate ones). Nothing here reads data.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from src.action_space.grid import generate_weight_grid
from src.data.universe import TICKERS

# ---------------------------------------------------------------------------
# Universe, actions, lattice, timing, seeds
# ---------------------------------------------------------------------------

# The three ETFs the nine-asset long-only optimum holds (SPY and GLD only at
# gamma = 3), chosen because three assets keep the oracle computable
# (TC_DRIFT_DESIGN.md C1: a benchmark-informed restriction, disclosed). They
# are also the assignment's "original three ETFs". Positions in the nine-asset
# TICKERS order, so the sub-block of mu and Sigma is mu[ASSET_INDEX] and
# Sigma[np.ix_(ASSET_INDEX, ASSET_INDEX)].
ASSETS = ("SPY", "IEF", "GLD")
ASSET_INDEX = tuple(TICKERS.index(t) for t in ASSETS)  # (0, 3, 6)
N_ASSETS = len(ASSETS)

TARGET_INCREMENT = 0.02  # trade targets: every long-only 2% portfolio
N_TARGETS = 1326         # C(52, 2), checked against target_grid() in the tests
HOLD = N_TARGETS         # the last action index: do not trade, let the weights drift
N_ACTIONS = N_TARGETS + 1

LATTICE_INCREMENT = 0.01  # the oracle's state lattice; every 2% target is a lattice point
N_LATTICE = 5151          # C(102, 2)

BLOCK_LENGTH = 63   # trading days per block (one decision per block)
DISCOUNT = 0.95     # rho, per block (about 0.81 per year at 4 blocks a year)
GAMMA_HEADLINE = 5.0  # environment and benchmarks only; the agent must not read it

# Random streams (TC_DRIFT_DESIGN.md §10). Reported runs: run i uses
# src.environment.SeedPlan(MASTER_SEED, n_runs).run_streams(i) -> (returns_rng,
# agent_rng), with the training start states drawn from agent_rng.spawn(1)[0];
# evaluation paths come from SeedPlan(...).evaluation_rng(). The oracle's fixed
# paths come from their own generator, np.random.default_rng(ORACLE_PATH_SEED),
# never from a training or evaluation stream. Tuning uses TUNING_MASTER_SEED in
# the same layout, never the reported seed.
MASTER_SEED = 20261016         # the project's reported seed (src/config.py)
TUNING_MASTER_SEED = 20261112  # TC-drift hyperparameter tuning only
ORACLE_PATH_SEED = (20261016, 3)  # SeedSequence entropy for the oracle's fixed paths

TRADING_DAYS = 252
BPS = 1e4

# Reference trades that pin kappa and lambda: (||u||_1, ||u||_2^2) for a
# transfer of 20 and of 40 percentage points between two ETFs.
REFERENCE_TRADES = {"20pt": (0.4, 0.08), "40pt": (0.8, 0.32)}

_SUM_TOL = 1e-9


def _units(increment: float) -> int:
    units = round(1.0 / increment)
    if units < 1 or abs(units * increment - 1.0) > 1e-9:
        raise ValueError(f"1/increment must be a whole number, got {increment!r}")
    return units


@lru_cache(maxsize=None)
def _grid(increment: float) -> np.ndarray:
    base = generate_weight_grid(N_ASSETS, increment)
    base.flags.writeable = False
    view = base.view()  # its base is read-only, so nobody can switch it back to writeable
    view.flags.writeable = False
    return view


def target_grid() -> np.ndarray:
    """The 1,326 trade targets, shape (N_TARGETS, 3), in generate_weight_grid
    order. Action a < HOLD means "trade to target_grid()[a]"."""
    return _grid(TARGET_INCREMENT)


def lattice(increment: float = LATTICE_INCREMENT) -> np.ndarray:
    """Every long-only portfolio on a grid of `increment`, shape (n, 3). The
    oracle's states; the 0.5% lattice is the refinement check."""
    return _grid(increment)


@lru_cache(maxsize=None)
def _lattice_table(increment: float) -> np.ndarray:
    """table[u1, u2] = lattice index of the point with units (u1, u2, N - u1 - u2)."""
    n = _units(increment)
    keys = np.rint(lattice(increment) * n).astype(np.int64)
    table = np.full((n + 1, n + 1), -1, dtype=np.int64)
    table[keys[:, 0], keys[:, 1]] = np.arange(len(keys))
    table.flags.writeable = False
    return table


def check_holdings(weights: np.ndarray) -> np.ndarray:
    """Holdings as an (m, 3) float array of finite, long-only rows summing to
    one (within 1e-9). Tiny negatives from rounding (> -1e-9) are clipped."""
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    if w.ndim != 2 or w.shape[1] != N_ASSETS:
        raise ValueError(f"holdings must have shape (3,) or (m, 3), got {np.shape(weights)}")
    if not np.all(np.isfinite(w)) or (w < -_SUM_TOL).any() or (np.abs(w.sum(axis=1) - 1.0) > _SUM_TOL).any():
        raise ValueError("holdings must be finite, long-only and sum to one")
    return np.clip(w, 0.0, None)


def nearest_lattice_index(weights: np.ndarray, increment: float = LATTICE_INCREMENT):
    """Index of the nearest lattice point (Euclidean) for each row of weights.

    Rounds each weight down to the lattice and hands the leftover units to the
    largest remainders. On the simplex lattice this gives the Euclidean-nearest
    point (checked against brute force in the tests); ties go to the lower
    asset index. Accepts (3,) or (m, 3); returns an int or an (m,) int64 array.
    """
    single = np.ndim(weights) == 1
    w = check_holdings(weights)
    n = _units(increment)
    scaled = w * n
    base = np.floor(scaled + 1e-12).astype(np.int64)
    leftover = n - base.sum(axis=1)                       # 0, 1 or 2 units for three assets
    order = np.argsort(-(scaled - base), axis=1, kind="stable")
    rank = np.argsort(order, axis=1, kind="stable")       # 0 = largest remainder
    base += rank < leftover[:, None]
    out = _lattice_table(increment)[base[:, 0], base[:, 1]]
    return int(out[0]) if single else out


def target_lattice_index(increment: float = LATTICE_INCREMENT) -> np.ndarray:
    """Lattice index of each 2% target, shape (N_TARGETS,)."""
    return nearest_lattice_index(target_grid(), increment)


def _check_actions(action) -> np.ndarray:
    a = np.asarray(action)
    if a.dtype == bool or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"actions must be integers, got dtype {a.dtype}")
    if ((a < 0) | (a > HOLD)).any():
        raise ValueError(f"actions must lie in [0, {HOLD}]")
    return a


def is_hold(action):
    return _check_actions(action) == HOLD


def post_trade_weights(state: np.ndarray, action) -> np.ndarray:
    """Weights right after the decision: the target for a trade, the current
    (drifted) holdings for HOLD. state (3,) with an int action, or (m, 3) with
    an (m,) array of actions. The cost is paid pro rata from every holding,
    so the post-trade weights are exactly the target."""
    single = np.ndim(state) == 1
    s = check_holdings(state)
    a = np.atleast_1d(_check_actions(action))
    out = np.where((a == HOLD)[:, None], s, target_grid()[np.where(a == HOLD, 0, a)])
    return out[0] if single else out


# ---------------------------------------------------------------------------
# Costs (environment and benchmarks only; never passed to the agent)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TCSettings:
    """One cost setting. kappa and lam are fractions of wealth (1 bp = 1e-4):
    cost(s, a) = kappa * ||a - s||_1 + (lam / 2) * ||a - s||_2^2."""

    kappa: float
    lam: float
    gamma: float = GAMMA_HEADLINE

    def __post_init__(self):
        for name in ("kappa", "lam"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma!r}")


@dataclass(frozen=True)
class Market:
    """The three-asset sub-block of the calibrated daily moments. Built by
    env.load_market(); used by the environment and the oracle only."""

    mu: np.ndarray
    sigma: np.ndarray


def trade_cost(state: np.ndarray, action, settings: TCSettings):
    """Cost of the decision as a fraction of wealth, measured from the DRIFTED
    pre-trade holdings. ||.||_1 counts both legs of a transfer. HOLD costs 0.
    Shapes as in post_trade_weights."""
    single = np.ndim(state) == 1
    s = check_holdings(state)
    a = np.atleast_1d(_check_actions(action))
    u = post_trade_weights(s, a) - s
    cost = settings.kappa * np.abs(u).sum(axis=1) + 0.5 * settings.lam * (u * u).sum(axis=1)
    cost = np.where(a == HOLD, 0.0, cost)
    return float(cost[0]) if single else cost


def coefficients_from_reference_trades(cost_20pt_bps: float, cost_40pt_bps: float) -> tuple[float, float]:
    """(kappa, lam) that make a 20-point transfer cost cost_20pt_bps and a
    40-point transfer cost cost_40pt_bps (total, in bp of wealth).

    Two equations, kappa * L1 + (lam / 2) * L2sq = cost, one per reference
    trade, give lam / 2 = (c40 - 2 c20) / 0.16 and kappa = (4 c20 - c40) / 0.8.
    Both are nonnegative exactly when 2 * c20 <= c40 <= 4 * c20: kappa alone
    doubles the cost of a trade twice the size, lam alone quadruples it."""
    (l1_a, l2_a), (l1_b, l2_b) = REFERENCE_TRADES["20pt"], REFERENCE_TRADES["40pt"]
    c_a, c_b = cost_20pt_bps / BPS, cost_40pt_bps / BPS
    if min(c_a, c_b) < 0 or not (2 * c_a - 1e-15 <= c_b <= 4 * c_a + 1e-15):
        raise ValueError("need 0 <= 2 * cost_20pt <= cost_40pt <= 4 * cost_20pt")
    half_lam = (c_b - c_a * l1_b / l1_a) / (l2_b - l2_a * l1_b / l1_a)
    kappa = (c_a - half_lam * l2_a) / l1_a
    return max(kappa, 0.0), max(2 * half_lam, 0.0)


# ---------------------------------------------------------------------------
# Reward, tie-breaking and reporting conventions
# ---------------------------------------------------------------------------

# Ties: when HOLD and a trade have the same value (on a lattice state that is
# itself a 2% target, HOLD and "trade to it" both cost 0 and lead to the same
# weights), HOLD wins, in the oracle and in the agent. Metrics count a trade
# whose post-trade weights equal the pre-trade holdings as a hold.


def net_reward(block_reward, cost, block_length: int = BLOCK_LENGTH):
    """Daily-unit reward the agent sees: the block's realized mean-variance
    reward minus the trading cost spread over the block's days."""
    return block_reward - np.asarray(cost) / block_length


def annualized_bps(value, discount: float = DISCOUNT, horizon: int | None = None):
    """Discounted sum of per-block daily-unit rewards -> bps per year.
    Infinite horizon: (1 - rho) * value is the equivalent constant daily
    reward. Finite horizon H: divide by sum_{t<H} rho^t instead."""
    if not 0 <= discount < 1:
        raise ValueError(f"discount must be in [0, 1), got {discount!r}")
    if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer))
                                or horizon < 1):
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    weight = (1 - discount) if horizon is None else 1.0 / sum(discount**t for t in range(horizon))
    return np.asarray(value) * weight * TRADING_DAYS * BPS


# What the agent and its feature map may import from this module.
AGENT_SAFE_NAMES = frozenset({
    "ASSETS", "N_ASSETS", "TARGET_INCREMENT", "N_TARGETS", "HOLD", "N_ACTIONS", "LATTICE_INCREMENT",
    "N_LATTICE", "BLOCK_LENGTH", "DISCOUNT", "target_grid", "lattice", "nearest_lattice_index",
    "target_lattice_index", "check_holdings", "is_hold", "post_trade_weights", "TCAgent",
})


# ---------------------------------------------------------------------------
# The agent interface (William implements; Albert's train/evaluate call it)
# ---------------------------------------------------------------------------

class TCAgent(Protocol):
    """What train.py and evaluate.py may call. state is the current (drifted)
    holdings, (3,), summing to one; actions are ints in [0, HOLD]. The agent is
    built from an rng and its own hyperparameters only; it reads the target
    grid from target_grid()."""

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy choice; the agent keeps its own exploration schedule."""

    def update(self, state: np.ndarray, action: int, reward_net: float, next_state: np.ndarray) -> None:
        """One learning step from (s, a, r_net, s'), r_net from net_reward()."""

    def greedy_action(self, state: np.ndarray) -> int:
        """The current greedy action (HOLD wins ties); never changes the agent's state."""
