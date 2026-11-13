"""Tests for the realistic cost levels (src/tc_drift/calibration.py, decided D9).

In plain terms: the cost levels come from bid-ask spreads plus the square-root
impact rule on our own volume data. These tests re-derive one cost by hand,
check that the fitted two-term model tracks the square-root cost across trade
sizes, and check the levels behave sensibly as the portfolio grows.
"""

import numpy as np
import pytest

from src.tc_drift import calibration as cal
from src.tc_drift import spec

LIQ = cal.liquidity()


def test_liquidity_is_read_for_the_three_etfs_and_is_plausible():
    assert set(LIQ) == set(spec.ASSETS)
    for adv, sigma in LIQ.values():
        assert adv > 1e8 and 0.001 < sigma < 0.05  # dollars per day; daily volatility


def test_square_root_cost_matches_a_hand_calculation():
    """One transfer of 10% at $1B, re-derived pair by pair."""
    q, aum = 0.10, 1e9
    def leg(t):
        adv, sigma = LIQ[t]
        return q * (0.5e-4 + sigma * np.sqrt(q * aum / adv))
    pairs = [("SPY", "IEF"), ("SPY", "GLD"), ("IEF", "GLD")]
    expected = sum(leg(a) + leg(b) for a, b in pairs) / 3
    assert cal.square_root_cost(q, aum, LIQ) == pytest.approx(expected, rel=1e-12)


def test_with_no_impact_the_fit_is_the_half_spread_alone():
    kappa, lam = cal.fit_cost_coefficients(1e-6, LIQ)  # a tiny portfolio: impact is negligible
    assert kappa == pytest.approx(cal.HALF_SPREAD, rel=1e-3) and lam == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("level", ["aum_100m", "aum_1b", "aum_10b"])
def test_the_fitted_model_tracks_the_square_root_cost_within_30_percent(level):
    aum = cal.AUM_LEVELS[level]
    kappa, lam = cal.fit_cost_coefficients(aum, LIQ)
    for q in cal.TRADE_SIZES:
        model = 2 * kappa * q + lam * q**2
        assert model == pytest.approx(cal.square_root_cost(q, aum, LIQ), rel=0.30), q


def test_the_fitted_model_agrees_with_spec_trade_cost():
    """The fit's cost formula is the environment's: a 10-point transfer priced both ways."""
    kappa, lam = cal.fit_cost_coefficients(1e9, LIQ)
    s = np.array([0.6, 0.2, 0.2])
    a = int(np.flatnonzero(np.all(np.isclose(spec.target_grid(), [0.5, 0.3, 0.2]), axis=1))[0])
    assert spec.trade_cost(s, a, spec.TCSettings(kappa=kappa, lam=lam)) == pytest.approx(2 * kappa * 0.1 + lam * 0.01)


def test_costs_rise_with_portfolio_size_and_zero_is_free():
    levels = cal.cost_settings()
    assert levels["zero"].kappa == levels["zero"].lam == 0.0
    sizes = ["aum_100m", "aum_1b", "aum_10b"]
    costs = [2 * levels[n].kappa * 0.2 + levels[n].lam * 0.04 for n in sizes]
    assert costs == sorted(costs) and costs[0] > 0
    assert cal.HEADLINE_LEVEL in levels


# The table in TC_DRIFT_DESIGN.md §8, to the digits the doc quotes, so the doc cannot drift from the code.
DOC_TABLE = {  # level: (kappa bp, lam, 20-point bp, 40-point bp, undo-one-block-of-drift bp)
    "aum_100m": (1.63, 4.38e-3, 2.40, 8.32, 0.115),
    "aum_1b": (3.90, 1.43e-2, 7.27, 25.97, 0.302),
    "aum_10b": (11.04, 4.57e-2, 22.69, 81.93, 0.892),
}


@pytest.mark.parametrize("level", list(DOC_TABLE))
def test_the_design_doc_table_matches_the_code(level):
    s = cal.cost_settings()[level]
    kappa_bp, lam, c20, c40, undo = DOC_TABLE[level]
    cost = lambda q: (2 * s.kappa * q + s.lam * q * q) * 1e4  # noqa: E731
    assert s.kappa * 1e4 == pytest.approx(kappa_bp, abs=0.006)
    assert s.lam == pytest.approx(lam, rel=0.005)
    assert (cost(0.2), cost(0.4), cost(0.0262)) == pytest.approx((c20, c40, undo), abs=0.006)


def test_the_volume_window_ends_with_the_calibration_sample():
    import pandas as pd
    assert cal.sample_end() == pd.Timestamp("2026-07-31")


def test_the_mechanism_settings_charge_the_headline_20_point_transfer_one_way_each():
    h = cal.cost_settings()[cal.HEADLINE_LEVEL]
    c20 = 2 * h.kappa * 0.2 + h.lam * 0.04
    m = cal.mechanism_settings()
    assert m["kappa_only"].lam == pytest.approx(0.0, abs=1e-15) and m["lambda_only"].kappa == pytest.approx(0.0, abs=1e-15)
    for s in m.values():
        assert 2 * s.kappa * 0.2 + s.lam * 0.04 == pytest.approx(c20, rel=1e-12)
