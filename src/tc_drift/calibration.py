"""Realistic trading-cost levels for the drift extension (decided D9, 2026-11-12).

In plain terms: what does it cost to trade SPY, IEF and GLD in practice? Two
parts. The bid-ask spread is about 1 bp for all three (issuer and ETF.com
figures, TC_DRIFT_DESIGN.md §8), so half a bp per dollar traded. Market impact
follows the practitioners' square-root rule: cost per dollar is about the
daily volatility times the square root of (trade / daily dollar volume), so
it grows with the size of the portfolio. Daily dollar volume and volatility
come from our own committed Yahoo files (data/raw/prices), over the 12 months
ending on the last day of the calibration sample (data/processed), so the
volume window matches the period mu and Sigma were estimated on. The cost levels are portfolio sizes: zero cost (the control),
$100M, $1B (headline) and $10B.

The environment's cost model is kappa * ||u||_1 + (lam / 2) * ||u||_2^2. Its
kappa and lam are fitted by least squares on relative error to the
square-root cost of transfers between two ETFs of 0.5% to 40% of the
portfolio, averaged over the three pairs; more of the points are small,
because drift trades are a few percent. A fit through only the 20- and
40-point reference trades would overcharge drift-sized trades by about 1.6 to
2.5 times.

Environment and benchmark only. The agent never imports this module
(src/tests/test_tc_drift_spec.py allowlist).

Assumptions, stated in the design doc: impact coefficient 1; a 0.5 bp
half-spread for all three (conservative for SPY, whose quoted spread is well
under 1 bp); impact uses the last 12 months' volatility, while the simulator's
Sigma is full-sample; and the same cost for every pair. GLD is the costliest
to trade (highest recent volatility), and drift is mostly SPY against GLD, so
averaging the pairs undercharges the typical drift trade by roughly a fifth.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.tc_drift.spec import ASSETS, GAMMA_HEADLINE, TCSettings, coefficients_from_reference_trades

REPO = Path(__file__).resolve().parents[2]
RAW_PRICES = REPO / "data" / "raw" / "prices"
SAMPLE_FILE = REPO / "data" / "processed" / "excess_returns.csv"

HALF_SPREAD = 0.5e-4        # half of a 1 bp quoted spread, per dollar traded
IMPACT_COEFFICIENT = 1.0    # Y in Y * sigma * sqrt(trade / ADV)
LIQUIDITY_WINDOW_DAYS = 365
TRADE_SIZES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.40)  # transfer between two ETFs, fraction of the portfolio
AUM_LEVELS = {"zero": 0.0, "aum_100m": 1e8, "aum_1b": 1e9, "aum_10b": 1e10}
HEADLINE_LEVEL = "aum_1b"


def sample_end(sample_file: Path = SAMPLE_FILE) -> pd.Timestamp:
    """Last date of the calibration sample (the processed excess returns)."""
    return pd.to_datetime(pd.read_csv(sample_file, usecols=[0]).iloc[:, 0]).max()


def liquidity(prices_dir: Path = RAW_PRICES, window_days: int = LIQUIDITY_WINDOW_DAYS,
              end: pd.Timestamp | None = None) -> dict:
    """{ticker: (mean daily dollar volume, daily return volatility)} over the
    window_days ending at `end` (default: the calibration sample's last day).
    Dollar volume is Close x Volume (unadjusted close, the price the shares
    traded at); volatility uses Adj Close."""
    end = sample_end() if end is None else pd.Timestamp(end)
    out = {}
    for ticker in ASSETS:
        frame = pd.read_csv(Path(prices_dir) / f"{ticker}.csv", parse_dates=["Date"]).set_index("Date")
        recent = (frame.index > end - pd.Timedelta(days=window_days)) & (frame.index <= end)
        dollar_volume = (frame["Close"] * frame["Volume"])[recent].mean()
        volatility = frame["Adj Close"].pct_change()[recent].std()
        out[ticker] = (float(dollar_volume), float(volatility))
    return out


def square_root_cost(transfer: float, aum: float, liq: dict) -> float:
    """Cost, as a fraction of wealth, of moving `transfer` of the portfolio
    from one ETF to another, averaged over the three pairs. Each leg pays the
    half-spread plus Y * sigma * sqrt(transfer * aum / ADV) per dollar."""
    def leg(ticker):
        adv, sigma = liq[ticker]
        return transfer * (HALF_SPREAD + IMPACT_COEFFICIENT * sigma * np.sqrt(transfer * aum / adv))
    return float(np.mean([leg(a) + leg(b) for a, b in itertools.combinations(ASSETS, 2)]))


def fit_cost_coefficients(aum: float, liq: dict) -> tuple[float, float]:
    """(kappa, lam), both >= 0, minimizing the relative error of
    2 * kappa * q + lam * q^2 (the model's cost of a transfer q, since
    ||u||_1 = 2q and ||u||_2^2 = 2q^2) against square_root_cost over TRADE_SIZES."""
    q = np.asarray(TRADE_SIZES)
    target = np.array([square_root_cost(x, aum, liq) for x in q])
    design = np.column_stack([2 * q, q**2]) / target[:, None]
    (kappa, lam), _ = nnls(design, np.ones_like(q))
    return float(kappa), float(lam)


def mechanism_settings(gamma: float = GAMMA_HEADLINE, prices_dir: Path = RAW_PRICES) -> dict:
    """The $1B cost split into its two mechanisms, for the oracle-only
    illustration (§8): "kappa_only" and "lambda_only" charge the same
    20-point transfer as the $1B model, with c40 = 2 c20 (linear) and
    c40 = 4 c20 (quadratic)."""
    kappa, lam = fit_cost_coefficients(AUM_LEVELS[HEADLINE_LEVEL], liquidity(prices_dir))
    c20_bps = (2 * kappa * 0.2 + lam * 0.04) * 1e4
    out = {}
    for name, ratio in (("kappa_only", 2.0), ("lambda_only", 4.0)):
        k, l = coefficients_from_reference_trades(c20_bps, ratio * c20_bps)
        out[name] = TCSettings(kappa=k, lam=l, gamma=gamma)
    return out


def cost_settings(gamma: float = GAMMA_HEADLINE, prices_dir: Path = RAW_PRICES) -> dict:
    """{level name: TCSettings}. "zero" has no costs at all (the control)."""
    liq = liquidity(prices_dir)
    out = {}
    for name, aum in AUM_LEVELS.items():
        kappa, lam = (0.0, 0.0) if aum == 0 else fit_cost_coefficients(aum, liq)
        out[name] = TCSettings(kappa=kappa, lam=lam, gamma=gamma)
    return out
