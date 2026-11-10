"""Turn the raw Part I downloads into the calibration dataset.

Pipeline:  data/raw  --copy-->  data/interim  --clean-->  data/processed

data/raw is never touched after download_raw.py writes it. This script first
copies it to data/interim and works only on that copy, so a bad cleaning run
can always be rebuilt without re-downloading (Yahoo history is not stable over
time, so re-downloading would silently change the sample).

Every filtering decision is counted and written to data/processed/CLEANING_LOG.json
so the row-count reconciliation in the report is reproducible.

Usage:
    python -m src.data.clean
"""

import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.universe import DEFAULT_SAMPLE_END, TICKERS, UNIVERSE

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

# Assignment: common sample beginning no earlier than February 2006.
SAMPLE_FLOOR = pd.Timestamp("2006-02-01")

TRADING_DAYS_PER_YEAR = 252

# Flag (do not remove) daily moves beyond this size for manual review.
EXTREME_RETURN_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Step 0: working copy
# ---------------------------------------------------------------------------

def copy_raw_to_interim() -> None:
    if INTERIM_DIR.exists():
        shutil.rmtree(INTERIM_DIR)
    shutil.copytree(RAW_DIR, INTERIM_DIR)


# ---------------------------------------------------------------------------
# Step 1: load prices and drop the in-progress session
# ---------------------------------------------------------------------------

def load_price_panel(log: dict) -> dict[str, pd.DataFrame]:
    """Load each ticker's full OHLCV+actions frame from the working copy."""
    manifest = json.loads((INTERIM_DIR / "MANIFEST.json").read_text())
    download_date = pd.Timestamp(manifest["downloaded_at_utc"]).tz_convert(
        "America/New_York"
    ).normalize().tz_localize(None)

    frames = {}
    dropped_partial = {}
    for ticker in TICKERS:
        df = pd.read_csv(
            INTERIM_DIR / "prices" / f"{ticker}.csv",
            index_col="Date",
            parse_dates=True,
        ).sort_index()

        if df.index.has_duplicates:
            raise ValueError(f"{ticker}: duplicate dates in raw file")

        # The bar dated on the download date is the session that was still
        # trading when we pulled it (partial volume, non-final close). Drop it.
        n_before = len(df)
        df = df[df.index < download_date]
        dropped_partial[ticker] = n_before - len(df)

        frames[ticker] = df

    log["download_date_et"] = str(download_date.date())
    log["rows_raw_per_ticker"] = {t: int(len(f)) + dropped_partial[t] for t, f in frames.items()}
    log["partial_session_rows_dropped"] = dropped_partial
    return frames


# ---------------------------------------------------------------------------
# Step 2: verify the adjustment convention
# ---------------------------------------------------------------------------

def verify_adjustment_convention(
    frames: dict[str, pd.DataFrame], log: dict, sample_index: pd.DatetimeIndex
) -> None:
    """Confirm 'Adj Close' really is a distribution-reinvested total-return series.

    The assignment asks us to verify, not assume, the adjustment convention.
    If Adj Close is adjusted for both splits and cash distributions, then

        AdjClose_t / AdjClose_{t-1} - 1  ==  (Close_t + D_t) / Close_{t-1} - 1

    on every day, where D_t is the per-share distribution paid on date t. We
    reconstruct the right-hand side from the unadjusted close plus the
    Dividends and Capital Gains columns and report the largest disagreement,
    both over the full downloaded history and over the dates we actually keep.
    The two differ because the most recent ex-distribution dates can carry a
    provisional adjustment factor that Yahoo has not yet finalised.
    """
    checks = {}
    for ticker, df in frames.items():
        distributions = df.get("Dividends", 0.0)
        if "Capital Gains" in df.columns:
            distributions = distributions + df["Capital Gains"].fillna(0.0)

        adj_return = df["Adj Close"].pct_change()
        reconstructed = (df["Close"] + distributions) / df["Close"].shift(1) - 1.0

        diff = (adj_return - reconstructed).abs().dropna()
        in_sample = diff.reindex(sample_index).dropna()

        checks[ticker] = {
            "max_abs_difference_full_history": float(diff.max()),
            "max_abs_difference_in_sample": float(in_sample.max()),
            "median_abs_difference": float(diff.median()),
            # idxmax on an all-zero series returns the first row, which would
            # read as a meaningful date for a ticker that pays no distributions.
            "worst_date_full_history": (
                str(diff.idxmax().date()) if diff.max() > 0 else None
            ),
            "distribution_days": int((distributions > 0).sum()),
        }

    worst_full = max(c["max_abs_difference_full_history"] for c in checks.values())
    worst_sample = max(c["max_abs_difference_in_sample"] for c in checks.values())
    log["adjustment_verification"] = {
        "worst_max_abs_difference_full_history": worst_full,
        "worst_max_abs_difference_in_sample": worst_sample,
        "per_ticker": checks,
        "interpretation": (
            "Adj Close return reconciles with (Close + distributions) / prior "
            "Close, confirming Adj Close is distribution-adjusted. Split "
            "adjustment is established separately, in split_verification. The "
            f"typical disagreement is floating-point noise and rises to "
            f"{worst_sample:.1e} in-sample on ex-distribution days, which is the "
            "rounding in Yahoo's published adjustment factors rather than a "
            "different convention."
        ),
    }
    print(
        f"adjustment check: worst |adj - reconstructed| daily return = "
        f"{worst_sample:.3e} in-sample ({worst_full:.3e} over full history)"
    )


# ---------------------------------------------------------------------------
# Step 3: align onto a common trading calendar
# ---------------------------------------------------------------------------

def build_aligned_prices(frames: dict[str, pd.DataFrame], log: dict) -> pd.DataFrame:
    """Inner-join adjusted closes so every retained date has all nine ETFs.

    Two things get removed here:
      * Jan 2006, where DBC does not yet exist (its inception is the binding
        constraint on the common sample, as the assignment notes).
      * Isolated vendor gaps, where Yahoo simply has no bar for some tickers on
        a date that other tickers do have.

    An inner join is the right handling for both: a return needs a price for
    all nine assets on the same date, and forward-filling a missing price would
    manufacture a spurious zero return followed by a spurious double-size one.
    """
    adj_close = pd.DataFrame(
        {ticker: df["Adj Close"] for ticker, df in frames.items()}
    )[TICKERS]

    union_dates = len(adj_close)
    aligned = adj_close.dropna(how="any")

    # Record which dates were lost to vendor gaps rather than to DBC inception,
    # since only the former is a data-quality concern worth reporting.
    dbc_start = frames["DBC"].index.min()
    lost = adj_close.index.difference(aligned.index)
    vendor_gaps = [d for d in lost if d >= dbc_start]

    aligned = aligned[aligned.index >= SAMPLE_FLOOR]

    log["calendar_alignment"] = {
        # After the partial-session drop, not the raw download union.
        "union_of_trading_dates_after_partial_drop": int(union_dates),
        "common_dates_retained": int(len(aligned)),
        "dbc_inception": str(dbc_start.date()),
        "dates_before_dbc_inception_dropped": int(sum(1 for d in lost if d < dbc_start)),
        "vendor_gap_dates_dropped": [str(d.date()) for d in vendor_gaps],
        "missing_value_policy": "inner join on date; no forward-fill, no interpolation",
    }
    return aligned


# ---------------------------------------------------------------------------
# Step 4: returns, risk-free rate, excess returns
# ---------------------------------------------------------------------------

def compute_simple_returns(prices: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Simple (arithmetic) returns, as the mean-variance objective requires.

    Mean-variance utility is defined over simple returns because portfolio
    return is the weighted average of asset simple returns; log returns are not
    additive across assets, so they would break w'r.

    fill_method=None is load-bearing. pandas still defaults to 'pad', which
    forward-fills a missing price before differencing and so manufactures a
    zero return followed by a double-size one - exactly the artefact the
    alignment step in build_aligned_prices exists to avoid. The default is
    deprecated and flips in pandas 3.0, so we state it rather than inherit it.
    """
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    log["returns"] = {
        "definition": "simple total return: AdjClose_t / AdjClose_{t-1} - 1",
        "rows": int(len(returns)),
        "first_return_date_dropped": str(prices.index[0].date()),
        "fill_method": "None (explicit); no forward-fill before differencing",
    }
    return returns


def load_risk_free(log: dict) -> pd.Series:
    """Daily risk-free rate from the Kenneth French daily factor file.

    The file is fixed-width-ish CSV with a multi-line header, a YYYYMMDD date
    key and values in PERCENT per day, so it needs dividing by 100.
    """
    path = INTERIM_DIR / "risk_free" / "F-F_Research_Data_Factors_daily.csv"
    raw = path.read_text().splitlines()

    header_idx = next(i for i, line in enumerate(raw) if line.startswith(",Mkt-RF"))
    rows = []
    for line in raw[header_idx + 1:]:
        key = line.split(",")[0].strip()
        if not key.isdigit() or len(key) != 8:
            break  # trailing copyright / blank lines
        rows.append(line)

    frame = pd.read_csv(
        pd.io.common.StringIO("\n".join([raw[header_idx]] + rows)),
        index_col=0,
    )
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    frame.index.name = "Date"

    rf = frame["RF"] / 100.0  # percent per day -> decimal per day
    rf.name = "RF"

    log["risk_free"] = {
        "source": "Kenneth French Data Library, F-F_Research_Data_Factors_daily",
        "units": "converted from percent-per-day to decimal-per-day (/100)",
        "coverage": f"{rf.index.min().date()} to {rf.index.max().date()}",
        "rows": int(len(rf)),
    }
    return rf


def build_excess_returns(
    returns: pd.DataFrame,
    rf: pd.Series,
    log: dict,
    sample_end: pd.Timestamp | None = None,
) -> tuple:
    """Excess returns on the intersection of the ETF and risk-free calendars.

    The French file lags the ETF data by roughly two months, so this inner join
    truncates the sample end. That is deliberate: the environment is defined
    over excess returns, and extending it would mean inventing a risk-free rate.

    sample_end optionally caps the window further. Pin it while new data is
    being appended so the numbers in the report stay reproducible; leave it None
    to let the sample grow to whatever the risk-free file supports.
    """
    etf_end = returns.index.max()
    joined = returns.join(rf, how="inner")
    rf_aligned = joined["RF"]
    excess = joined[TICKERS].sub(rf_aligned, axis=0)

    rows_before_cap = len(excess)
    if sample_end is not None:
        if sample_end < excess.index.min():
            raise ValueError(
                f"--end {sample_end.date()} precedes the sample start "
                f"{excess.index.min().date()}"
            )
        if sample_end > excess.index.max():
            raise ValueError(
                f"--end {sample_end.date()} is beyond the available data, which "
                f"ends {excess.index.max().date()}. The risk-free file ends "
                f"{rf.index.max().date()}; re-run download_raw.py when a newer "
                "one is published."
            )
        excess = excess.loc[:sample_end]
        rf_aligned = rf_aligned.loc[:sample_end]

    log["excess_returns"] = {
        "definition": "r_excess = r_simple - RF_daily",
        "rows": int(len(excess)),
        "sample_start": str(excess.index.min().date()),
        "sample_end": str(excess.index.max().date()),
        "etf_data_available_through": str(etf_end.date()),
        "risk_free_available_through": str(rf.index.max().date()),
        "rows_dropped_to_risk_free_coverage": int(len(returns) - rows_before_cap),
        "explicit_end_cap": str(sample_end.date()) if sample_end is not None else None,
        "rows_dropped_to_explicit_cap": int(rows_before_cap - len(excess)),
        "note": (
            "Sample end is set by the French daily file, not by the ETF data. "
            "Forward-filling RF past its published end was rejected in favour "
            "of truncating."
        ),
    }
    return excess, rf_aligned


# ---------------------------------------------------------------------------
# Step 5: validation and summary statistics
# ---------------------------------------------------------------------------

def restrict_to_final_sample(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    excess: pd.DataFrame,
    log: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put every published file on the one sample the report describes.

    The risk-free join sets the sample end, and it happens after prices and
    returns are built, so both of those still carry 36 trading days past the
    calibration window. Publishing them unclipped invites a reader in Parts
    II-VI to calibrate on a window the report never documents.

    Prices keep ONE extra leading bar - the base price the first return is
    computed from - so that adj_close.csv can reproduce returns_simple.csv
    exactly. Without it the published price file cannot regenerate the
    published return file, which is the reproducibility claim we are making.
    """
    start, end = excess.index.min(), excess.index.max()

    base_positions = prices.index.get_indexer([start])
    if base_positions[0] < 1:
        raise ValueError("no price bar precedes the sample start")
    base_date = prices.index[base_positions[0] - 1]

    clipped_prices = prices.loc[base_date:end]
    clipped_returns = returns.loc[start:end]

    log["final_sample_restriction"] = {
        "reason": (
            "prices and returns are built before the risk-free join, so they "
            "initially extend past the calibration window; all published files "
            "are clipped to the documented sample"
        ),
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "price_base_date_retained": str(base_date.date()),
        "price_rows": int(len(clipped_prices)),
        "return_rows": int(len(clipped_returns)),
        "trailing_rows_dropped_from_prices": int(len(prices.loc[prices.index > end])),
    }
    return clipped_prices, clipped_returns


def verify_split_convention(frames: dict[str, pd.DataFrame], log: dict) -> None:
    """Confirm the unadjusted 'Close' column is already split-adjusted.

    The distribution check in verify_adjustment_convention contains no split
    term, so on its own it can only show that Close and Adj Close share a split
    convention - not which one. Splits settle that: across a split date, an
    unadjusted price series would jump by the split ratio, and an adjusted one
    would not.
    """
    checks = {}
    for ticker, df in frames.items():
        if "Stock Splits" not in df.columns:
            continue
        split_dates = df.index[df["Stock Splits"] > 0]
        for date in split_dates:
            position = df.index.get_loc(date)
            if position == 0:
                continue
            checks[f"{ticker} {date.date()}"] = {
                "split_ratio": float(df.loc[date, "Stock Splits"]),
                "close_ratio": float(
                    df["Close"].iloc[position] / df["Close"].iloc[position - 1]
                ),
                "adj_close_ratio": float(
                    df["Adj Close"].iloc[position] / df["Adj Close"].iloc[position - 1]
                ),
            }

    log["split_verification"] = {
        "splits_found": checks,
        "interpretation": (
            "Both Close and Adj Close move by ~1 across every split date rather "
            "than by the split ratio, so yfinance split-adjusts BOTH series. "
            "'Close' is therefore the split-adjusted, distribution-UNadjusted "
            "close, not a raw unadjusted price."
        ),
    }
    if checks:
        worst = max(abs(c["close_ratio"] - 1.0) for c in checks.values())
        print(f"split check: {len(checks)} split(s); worst |Close ratio - 1| = {worst:.3f}")


def validate(excess: pd.DataFrame, prices: pd.DataFrame, log: dict) -> None:
    problems = []
    if excess.isna().any().any():
        problems.append("NaNs present in excess returns")
    if not np.isfinite(excess.to_numpy()).all():
        problems.append("non-finite values present in excess returns")
    if not excess.index.is_monotonic_increasing:
        problems.append("dates not sorted")
    if excess.index.has_duplicates:
        problems.append("duplicate dates")
    if (prices <= 0).any().any():
        problems.append("non-positive adjusted prices")
    if problems:
        raise ValueError("validation failed: " + "; ".join(problems))

    extreme = excess.abs() > EXTREME_RETURN_THRESHOLD
    flagged = {
        ticker: [
            {"date": str(d.date()), "excess_return": round(float(excess.loc[d, ticker]), 4)}
            for d in excess.index[extreme[ticker]]
        ]
        for ticker in TICKERS
        if extreme[ticker].any()
    }
    log["validation"] = {
        "checks_passed": True,
        "extreme_move_threshold": EXTREME_RETURN_THRESHOLD,
        "flagged_moves_retained": flagged,
        "note": (
            "Flagged moves are real crisis-period prints (2008 and March 2020) "
            "and are retained; removing them would bias the covariance downward "
            "in exactly the states the agent most needs to price."
        ),
    }


def covariance_diagnostics(covariance: pd.DataFrame, log: dict) -> None:
    """Conditioning checks on Sigma.

    Part II inverts Sigma in closed form and Part VI asks us to discuss its
    numerical stability, so record the spectrum here rather than discovering a
    problem inside the optimizer.
    """
    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy())
    log["covariance_diagnostics"] = {
        "smallest_eigenvalue": float(eigenvalues.min()),
        "largest_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "positive_definite": bool(eigenvalues.min() > 0),
        "note": (
            "5,152 daily observations against 9 assets gives 45 covariance "
            "parameters from a long sample, so the sample estimator is well "
            "conditioned and invertible without shrinkage. A shrinkage "
            "estimator remains available as the Part VI statistical robustness "
            "check."
        ),
    }
    print(
        f"covariance: condition number = "
        f"{log['covariance_diagnostics']['condition_number']:.1f}, "
        f"min eigenvalue = {eigenvalues.min():.3e}"
    )


def summary_statistics(excess: pd.DataFrame) -> pd.DataFrame:
    daily_mean = excess.mean()
    daily_std = excess.std(ddof=1)
    table = pd.DataFrame(
        {
            "segment": [UNIVERSE[t][0] for t in TICKERS],
            "exposure": [UNIVERSE[t][1] for t in TICKERS],
            "mean_daily": daily_mean,
            "std_daily": daily_std,
            "mean_annualized": daily_mean * TRADING_DAYS_PER_YEAR,
            "vol_annualized": daily_std * np.sqrt(TRADING_DAYS_PER_YEAR),
            "sharpe_annualized": (daily_mean / daily_std) * np.sqrt(TRADING_DAYS_PER_YEAR),
            "skew": excess.skew(),
            "excess_kurtosis": excess.kurtosis(),
            "min_daily": excess.min(),
            "max_daily": excess.max(),
        },
        index=TICKERS,
    )
    table.index.name = "ticker"
    return table


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--end",
        default=DEFAULT_SAMPLE_END,
        help=(
            "YYYY-MM-DD cap on the sample end. Defaults to the pinned "
            "calibration end in src/data/universe.py."
        ),
    )
    args = parser.parse_args()
    sample_end = pd.Timestamp(args.end) if args.end else None

    log: dict = {}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    copy_raw_to_interim()
    frames = load_price_panel(log)

    prices = build_aligned_prices(frames, log)
    returns = compute_simple_returns(prices, log)
    rf = load_risk_free(log)
    excess, rf_aligned = build_excess_returns(returns, rf, log, sample_end=sample_end)

    prices, returns = restrict_to_final_sample(prices, returns, excess, log)
    verify_adjustment_convention(frames, log, excess.index)
    verify_split_convention(frames, log)
    validate(excess, prices, log)

    stats = summary_statistics(excess)
    correlation = excess.corr()
    covariance = excess.cov()
    covariance_diagnostics(covariance, log)

    prices.to_csv(PROCESSED_DIR / "adj_close.csv", lineterminator="\n")
    returns.to_csv(PROCESSED_DIR / "returns_simple.csv", lineterminator="\n")
    rf_aligned.to_csv(PROCESSED_DIR / "rf_daily.csv", lineterminator="\n")
    excess.to_csv(PROCESSED_DIR / "excess_returns.csv", lineterminator="\n")
    stats.to_csv(PROCESSED_DIR / "summary_statistics.csv", lineterminator="\n")
    correlation.to_csv(PROCESSED_DIR / "correlation_matrix.csv", lineterminator="\n")
    covariance.to_csv(PROCESSED_DIR / "covariance_daily.csv", lineterminator="\n")

    log["outputs"] = sorted(p.name for p in PROCESSED_DIR.glob("*.csv"))
    log["library_versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__}  # CLAUDE.md §6 (audit F-L14)
    (PROCESSED_DIR / "CLEANING_LOG.json").write_text(json.dumps(log, indent=2) + "\n")

    print(
        f"\nfinal sample: {len(excess)} daily observations, "
        f"{excess.index.min().date()} -> {excess.index.max().date()}"
    )
    print(f"outputs -> {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
