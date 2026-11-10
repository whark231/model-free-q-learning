"""Checks on the cleaned Part I dataset.

These run against the committed data/processed outputs, so they are fast and
offline. They exist to catch a cleaning regression: if someone changes clean.py
and the sample silently shifts, these must fail.

Design rule, learned the hard way: a test that compares two processed files
against each other proves nothing, because clean.py wrote both from the same
line of code. An earlier version of this file passed cleanly while the excess
returns had no risk-free rate subtracted at all. Every check below therefore
either re-derives its target from data/raw, or pins a literal value from
DATA_CLEANING.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.universe import TICKERS

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

# The sample documented in DATA_CLEANING.md §3 and quoted throughout the report.
# Hardcoded on purpose: these are published numbers, so a change to them should
# break the build and force the documentation to be updated with it.
EXPECTED_ROWS = 5152
EXPECTED_START = pd.Timestamp("2006-02-07")
EXPECTED_END = pd.Timestamp("2026-07-31")

# Well-known trading days, used as date-alignment anchors. A one-day shift
# anywhere in the pipeline moves these off their dates and fails.
# These are SPY TOTAL returns (adjusted close to adjusted close), not excess.
KNOWN_EVENT_SPY_TOTAL_RETURNS = {
    "2008-09-29": -0.078362,  # TARP rejected by the House
    "2008-10-13": +0.145198,  # coordinated central-bank action
    "2018-02-05": -0.041822,  # "Volmageddon"
    "2020-03-16": -0.109424,  # COVID crash
    "2020-03-24": +0.090604,  # CARES Act rally
}

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "excess_returns.csv").exists(),
    reason="processed data not built; run python -m src.data.clean",
)


@pytest.fixture(scope="module")
def excess():
    return pd.read_csv(PROCESSED / "excess_returns.csv", index_col=0, parse_dates=True)


@pytest.fixture(scope="module")
def adj_close():
    return pd.read_csv(PROCESSED / "adj_close.csv", index_col=0, parse_dates=True)


@pytest.fixture(scope="module")
def log():
    return json.loads((PROCESSED / "CLEANING_LOG.json").read_text())


@pytest.fixture(scope="module")
def raw_risk_free():
    """Re-parse the French daily file from data/raw with an independent reader.

    Deliberately does NOT reuse clean.py's load_risk_free. This is the only way
    to catch the percent-vs-decimal conversion being dropped, the header offset
    being wrong, or the copyright footer leaking into the data.
    """
    path = RAW / "risk_free" / "F-F_Research_Data_Factors_daily.csv"
    values = {}
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5 and parts[0].isdigit() and len(parts[0]) == 8:
            values[pd.Timestamp(parts[0])] = float(parts[4]) / 100.0
    series = pd.Series(values, name="RF").sort_index()
    assert len(series) > 20000, "independent RF parse found too few rows"
    return series


@pytest.fixture(scope="module")
def raw_adj_close():
    """Rebuild the adjusted-close panel straight from the nine raw ticker files."""
    frames = {}
    for ticker in TICKERS:
        df = pd.read_csv(
            RAW / "prices" / f"{ticker}.csv", index_col="Date", parse_dates=True
        ).sort_index()
        frames[ticker] = df["Adj Close"]
    return pd.DataFrame(frames)[TICKERS]


# --- The documented sample, pinned to literals -------------------------------

def test_sample_matches_the_documented_shape(excess):
    """DATA_CLEANING.md §3. If this fails, the docs are now wrong too."""
    assert len(excess) == EXPECTED_ROWS
    assert excess.index.min() == EXPECTED_START
    assert excess.index.max() == EXPECTED_END


def test_columns_are_the_nine_tickers_in_canonical_order(excess):
    """Weight vectors from the optimizer are positional, so column order must
    match src.data.universe.TICKERS or w'r pairs weights with the wrong assets."""
    assert list(excess.columns) == TICKERS


def test_dates_are_sorted_and_unique(excess):
    assert excess.index.is_monotonic_increasing
    assert not excess.index.has_duplicates


def test_no_missing_or_infinite_values(excess):
    assert not excess.isna().any().any()
    assert np.isfinite(excess.to_numpy()).all()


# --- Re-derivation from data/raw ---------------------------------------------

def test_risk_free_was_actually_subtracted(excess, raw_risk_free):
    """The regression this suite previously missed entirely.

    Rebuilds excess returns from the raw ticker files and the independently
    parsed French file. Catches RF being skipped, left in percent, misaligned,
    or silently zeroed.
    """
    prices = {}
    for ticker in TICKERS:
        df = pd.read_csv(
            RAW / "prices" / f"{ticker}.csv", index_col="Date", parse_dates=True
        ).sort_index()
        prices[ticker] = df["Adj Close"]
    panel = pd.DataFrame(prices)[TICKERS].dropna(how="any")

    returns = panel.pct_change(fill_method=None).dropna(how="any")
    rebuilt = returns.loc[excess.index].sub(raw_risk_free.loc[excess.index], axis=0)

    np.testing.assert_allclose(
        rebuilt.to_numpy(), excess[TICKERS].to_numpy(), atol=1e-12
    )


def test_risk_free_is_nonzero_and_plausible(raw_risk_free, excess):
    """Guards the percent/decimal conversion from both sides.

    A daily RF above 0.1% means the /100 was missed. An all-zero RF means it was
    never applied. Note the French daily rate is quantized to 1bp, and 57% of
    this sample sits in the post-2008 and post-COVID zero-rate periods where it
    rounds to exactly 0, so a simple "mostly nonzero" check would be wrong. The
    cumulative rate is the meaningful test: ~35% over 20.5 years is an average
    short rate near 1.7%.
    """
    in_sample = raw_risk_free.loc[excess.index]
    assert in_sample.min() >= 0.0
    assert in_sample.max() < 0.001
    assert (in_sample > 0).sum() > 1000, "risk-free rate is essentially all zero"
    assert 0.10 < in_sample.sum() < 1.00, (
        f"cumulative risk-free rate of {in_sample.sum():.4f} over the sample is "
        "outside any plausible range; check the percent/decimal conversion"
    )


def test_adjusted_closes_match_the_raw_files(adj_close, raw_adj_close):
    """The published price file must be the raw data, not a transformed copy."""
    overlap = adj_close.index
    np.testing.assert_allclose(
        adj_close[TICKERS].to_numpy(),
        raw_adj_close.loc[overlap, TICKERS].to_numpy(),
        rtol=1e-12,
    )


# --- Date alignment ----------------------------------------------------------

def test_known_event_returns_land_on_the_right_dates(excess, raw_risk_free):
    """Anchors the calendar against a one-day shift, which is the failure mode
    that would introduce look-ahead bias without changing any row count."""
    for date_string, expected_total_return in KNOWN_EVENT_SPY_TOTAL_RETURNS.items():
        date = pd.Timestamp(date_string)
        assert date in excess.index, f"{date_string} missing from the sample"
        actual = excess.loc[date, "SPY"] + raw_risk_free.loc[date]
        assert actual == pytest.approx(expected_total_return, abs=1e-5), (
            f"SPY total return on {date_string} is {actual:.6f}, expected "
            f"{expected_total_return:.6f} - the calendar may be shifted"
        )


def test_vendor_gap_date_absent_from_the_raw_panel(raw_adj_close):
    """2026-09-22 has prices for SPY and GLD only, so the aligned panel must
    drop it. Checked against the raw panel because the date falls outside the
    final sample, where its absence would prove nothing."""
    aligned = raw_adj_close.dropna(how="any")
    assert pd.Timestamp("2026-09-22") in raw_adj_close.index
    assert pd.Timestamp("2026-09-22") not in aligned.index


def test_no_weekend_dates(excess):
    assert (excess.index.dayofweek < 5).all()


# --- Internal consistency of the published outputs ---------------------------

def test_adj_close_reproduces_returns_simple(adj_close):
    """The published price file must regenerate the published return file.

    This is the reproducibility claim DATA_CLEANING.md makes, and it failed
    before the sample-restriction fix because adj_close had lost its base price.
    """
    returns = pd.read_csv(
        PROCESSED / "returns_simple.csv", index_col=0, parse_dates=True
    )
    rebuilt = adj_close.pct_change(fill_method=None).dropna(how="any")
    assert rebuilt.index.equals(returns.index)
    np.testing.assert_allclose(rebuilt.to_numpy(), returns[TICKERS].to_numpy(), atol=1e-12)


def test_all_published_files_cover_the_same_sample(excess):
    """A reader in Parts II-VI must not be able to pick up a file that silently
    spans a different window."""
    for name in ["returns_simple", "rf_daily", "excess_returns"]:
        frame = pd.read_csv(PROCESSED / f"{name}.csv", index_col=0, parse_dates=True)
        assert frame.index.min() == EXPECTED_START, name
        assert frame.index.max() == EXPECTED_END, name
        assert len(frame) == EXPECTED_ROWS, name


def test_covariance_file_matches_the_excess_returns(excess):
    saved = pd.read_csv(PROCESSED / "covariance_daily.csv", index_col=0)
    np.testing.assert_allclose(saved.to_numpy(), excess.cov().to_numpy(), rtol=1e-10)


def test_correlation_file_matches_the_excess_returns(excess):
    saved = pd.read_csv(PROCESSED / "correlation_matrix.csv", index_col=0)
    np.testing.assert_allclose(saved.to_numpy(), excess.corr().to_numpy(), rtol=1e-10)


def test_summary_statistics_match_the_excess_returns(excess):
    saved = pd.read_csv(PROCESSED / "summary_statistics.csv", index_col=0)
    np.testing.assert_allclose(
        saved["mean_annualized"].to_numpy(),
        (excess.mean() * 252).to_numpy(),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        saved["vol_annualized"].to_numpy(),
        (excess.std(ddof=1) * np.sqrt(252)).to_numpy(),
        rtol=1e-10,
    )


def test_cleaning_log_agrees_with_the_documented_sample(log):
    assert log["excess_returns"]["rows"] == EXPECTED_ROWS
    assert log["excess_returns"]["sample_start"] == str(EXPECTED_START.date())
    assert log["excess_returns"]["sample_end"] == str(EXPECTED_END.date())


# --- Properties Part II depends on -------------------------------------------

def test_covariance_is_positive_definite(excess):
    """Part II inverts Sigma in closed form; a non-PD estimate would break it."""
    assert np.linalg.eigvalsh(excess.cov().to_numpy()).min() > 0


def test_covariance_is_well_conditioned(excess):
    """Guards against a future calibration window too short to estimate 45
    covariance parameters stably."""
    eigenvalues = np.linalg.eigvalsh(excess.cov().to_numpy())
    assert eigenvalues.max() / eigenvalues.min() < 1e4


def test_returns_are_economically_plausible(excess):
    """A units or alignment error would show up as absurd daily moves."""
    assert excess.abs().max().max() < 0.5
    annualized_vol = excess.std(ddof=1) * np.sqrt(252)
    assert annualized_vol.between(0.01, 1.0).all()


def test_adjustment_and_split_conventions_were_verified(log):
    """Both halves of the Part I verification requirement must have run."""
    assert log["adjustment_verification"]["worst_max_abs_difference_in_sample"] < 1e-3
    splits = log["split_verification"]["splits_found"]
    assert splits, "no split found to verify the split convention against"
    for check in splits.values():
        assert abs(check["close_ratio"] - 1.0) < 0.2, (
            "Close jumped by roughly the split ratio, so it is NOT split-adjusted"
        )


def test_extend_rescales_appended_adjusted_prices_onto_the_existing_basis():
    """A fresh pull restates history after a distribution; the return across
    the join must equal the fresh series' return (audit F-M9)."""
    import pandas as pd
    from src.data.download_raw import rescale_appended
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    fresh = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0], "Adj Close": [98.0, 99.0, 100.0, 101.0]}, index=dates)
    existing = pd.DataFrame({"Close": [100.0, 101.0], "Adj Close": [99.0, 100.0]}, index=dates[:2])
    appended, factor = rescale_appended(existing, fresh, fresh.loc[dates[2:]])
    assert factor == pytest.approx(100.0 / 99.0)
    joined = pd.concat([existing, appended])["Adj Close"]
    # Returns from the join onward follow the fresh series; earlier ones keep the old basis.
    assert joined.pct_change().iloc[2:].tolist() == pytest.approx(fresh["Adj Close"].pct_change().iloc[2:].tolist())
    assert appended["Close"].tolist() == [102.0, 103.0]          # unadjusted prices untouched
    assert rescale_appended(existing, fresh, fresh.iloc[:0])[1] is None


def test_the_sample_end_is_pinned_to_the_committed_calibration():
    """A data refresh must not move mu and Sigma silently (audit F-M8)."""
    from src.data.universe import DEFAULT_SAMPLE_END
    from src.environment import PROCESSED_DIR
    import pandas as pd
    assert DEFAULT_SAMPLE_END == "2026-07-31"
    excess = pd.read_csv(PROCESSED_DIR / "excess_returns.csv", index_col=0, parse_dates=True)
    assert str(excess.index.max().date()) == DEFAULT_SAMPLE_END
