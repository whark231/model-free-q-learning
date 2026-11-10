"""Download the raw Part I inputs: nine ETF price histories + the daily risk-free rate.

Writes only to data/raw/. Everything downstream works on a copy (see clean.py).

Two modes:

  --mode full     Fresh pull of the whole history, overwriting
                  data/raw. Use this to start over.

  --mode extend   Keep every existing raw row byte-for-byte and append only
                  dates after the current end. Use this to add new data as it
                  becomes available, without disturbing the calibration the
                  report already describes.

Extend mode exists because Yahoo restates adjusted history as distributions
accrue, so a full re-pull silently changes μ and Σ for dates we have already
published. Extend mode also RE-CHECKS the overlapping dates and reports any
that Yahoo has since revised, rather than applying the revisions silently. A
revision is information, not an error; it is surfaced so the team can decide.

The risk-free file is always replaced in full, because the French file is
append-only upstream and is the binding constraint on the sample end.

The assignment requires recording the download date and verifying the
adjustment convention, so every file is logged in data/raw/MANIFEST.json with a
SHA-256 digest, the row count, the date span and the library version that
produced it.

--mode is required, so the raw data is never overwritten by accident
(audit F-M8). In extend mode the appended Adj Close rows are rescaled onto the
existing rows' adjustment basis at the last shared date, so no return crosses
two bases (F-M9); the factor is recorded in the manifest.

Usage:
    python3 -m src.data.download_raw --mode full    # full pull, overwrites data/raw
    python3 -m src.data.download_raw --mode extend  # append new dates only
"""

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from src.data.universe import FF_DAILY_URL, REQUEST_START, TICKERS

RAW_DIR = Path("data/raw")
PRICE_DIR = RAW_DIR / "prices"
RF_DIR = RAW_DIR / "risk_free"
MANIFEST_PATH = RAW_DIR / "MANIFEST.json"

# With auto_adjust=True, yfinance overwrites Close with the fully adjusted
# series and drops "Adj Close". We keep both columns instead: "Adj Close" is
# what the assignment asks us to build returns from, and having Close beside it
# is what lets clean.py verify the adjustment convention rather than trust it.
#
# Note Close is NOT a raw unadjusted price - yfinance split-adjusts it too. The
# difference between the two columns is distributions only. clean.py checks
# both halves of that claim (verify_adjustment_convention, verify_split_convention).
AUTO_ADJUST = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rescale_appended(existing: pd.DataFrame, fresh: pd.DataFrame, appended: pd.DataFrame,
                     column: str = "Adj Close") -> tuple[pd.DataFrame, float | None]:
    """Put appended adjusted prices on the existing rows' adjustment basis.

    A fresh Yahoo pull restates every earlier adjusted price after a
    distribution, so its new rows sit on a different basis from rows kept from
    an older pull. Multiplying the appended rows by existing / fresh on the
    last shared date makes the return across the join equal the fresh series'
    return (audit F-M9). Unadjusted columns are left alone."""
    shared = existing.index.intersection(fresh.index)
    if appended.empty or column not in appended or shared.empty:
        return appended, None
    last = shared.max()
    factor = float(existing.loc[last, column] / fresh.loc[last, column])
    out = appended.copy()
    out[column] = out[column] * factor
    return out, factor


def _fetch_history(ticker: str) -> pd.DataFrame:
    """Full daily history for one ticker, exactly as yfinance returns it."""
    df = yf.Ticker(ticker).history(
        start=REQUEST_START,
        interval="1d",
        auto_adjust=AUTO_ADJUST,
        actions=True,  # keeps Dividends / Stock Splits for the audit trail
    )
    if df.empty:
        raise RuntimeError(f"{ticker}: empty response from Yahoo Finance")

    # Yahoo returns a tz-aware index (America/New_York). Drop the tz so the
    # saved index is a plain trading date; the time component is always
    # midnight and carries no information for daily bars.
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "Date"
    return df


def _compare_overlap(
    existing: pd.DataFrame, fresh: pd.DataFrame, tolerance: float = 1e-6
) -> list[dict]:
    """Dates where Yahoo has revised a value we already hold.

    Restatements are expected on recent dates as distributions settle. They are
    reported rather than applied, so the sample the report describes never moves
    underneath us without someone deciding that it should.
    """
    shared = existing.index.intersection(fresh.index)
    columns = [c for c in ("Close", "Adj Close") if c in existing and c in fresh]

    revisions = []
    for column in columns:
        difference = (fresh.loc[shared, column] - existing.loc[shared, column]).abs()
        relative = difference / existing.loc[shared, column].abs().replace(0, pd.NA)
        changed = relative[relative > tolerance].dropna()
        for date, size in changed.items():
            revisions.append(
                {
                    "date": str(date.date()),
                    "column": column,
                    "old": float(existing.loc[date, column]),
                    "new": float(fresh.loc[date, column]),
                    "relative_change": float(size),
                }
            )
    return sorted(revisions, key=lambda r: r["date"])


def download_prices(mode: str = "full") -> tuple[dict, dict]:
    """One CSV per ticker. Returns (manifest entries, extend report)."""
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    entries = {}
    extend_report: dict = {}

    for ticker in TICKERS:
        fresh = _fetch_history(ticker)
        out = PRICE_DIR / f"{ticker}.csv"

        if mode == "extend" and out.exists():
            existing = pd.read_csv(out, index_col="Date", parse_dates=True).sort_index()
            new_dates = fresh.index.difference(existing.index)
            appended, factor = rescale_appended(existing, fresh, fresh.loc[new_dates])

            # Existing rows are kept exactly as they are; only genuinely new
            # dates are added, and they are re-sorted into place.
            combined = pd.concat([existing, appended]).sort_index()
            combined.index.name = "Date"

            extend_report[ticker] = {
                "rows_before": int(len(existing)),
                "rows_appended": int(len(appended)),
                "appended_dates": [str(d.date()) for d in new_dates],
                "adj_close_rescale_factor": factor,
                "upstream_revisions_detected": _compare_overlap(existing, fresh),
            }
            df = combined
        else:
            df = fresh

        df.to_csv(out, lineterminator="\n")  # LF on every machine, so the digest matches a checkout
        entries[ticker] = {
            "file": str(out),
            "rows": int(len(df)),
            "first_date": str(df.index[0].date()),
            "last_date": str(df.index[-1].date()),
            "columns": list(df.columns),
            "sha256": _sha256(out),
        }
        suffix = ""
        if ticker in extend_report:
            suffix = f"  (+{extend_report[ticker]['rows_appended']} appended)"
        print(
            f"{ticker:<5} {len(df):>5} rows  {entries[ticker]['first_date']}"
            f" -> {entries[ticker]['last_date']}{suffix}"
        )

    return entries, extend_report


def download_risk_free() -> dict:
    """Kenneth French daily factors.

    The monthly F-F factor file already in the course folder is NOT usable here:
    our returns are daily, and the assignment requires means, covariances, risk
    aversion and rewards to share one time frequency.
    """
    RF_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(FF_DAILY_URL, timeout=60)
    response.raise_for_status()

    zip_path = RF_DIR / "F-F_Research_Data_Factors_daily_CSV.zip"
    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        member = zf.namelist()[0]
        extracted = RF_DIR / member
        extracted.write_bytes(zf.read(member))

    last_date = _french_last_date(extracted)
    print(f"RF    {extracted.name} ({extracted.stat().st_size:,} bytes), ends {last_date}")
    return {
        "url": FF_DAILY_URL,
        "zip_file": str(zip_path),
        "csv_file": str(extracted),
        "zip_sha256": _sha256(zip_path),
        "csv_sha256": _sha256(extracted),
        "last_date": last_date,
        "note": (
            "The French daily file publishes with roughly a two-month lag and "
            "therefore sets the sample end, not the ETF data. Re-run this "
            "script when a new month posts."
        ),
    }


def _french_last_date(path: Path) -> str:
    """Last dated row in the French file, so the manifest records the real
    coverage rather than leaving the sample end to be discovered downstream."""
    last = None
    for line in path.read_text().splitlines():
        key = line.split(",")[0].strip()
        if key.isdigit() and len(key) == 8:
            last = key
    if last is None:
        raise RuntimeError("no dated rows found in the French daily file")
    return f"{last[:4]}-{last[4:6]}-{last[6:]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("full", "extend"),
        required=True,
        help="full: fresh pull, overwrite. extend: keep existing rows, append new dates.",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.now(timezone.utc)

    prices, extend_report = download_prices(mode=args.mode)
    risk_free = download_risk_free()

    manifest = {
        "downloaded_at_utc": downloaded_at.isoformat(timespec="seconds"),
        "mode": args.mode,
        "requested_start": REQUEST_START,
        "price_source": "Yahoo Finance via yfinance",
        "adjustment_convention": (
            "auto_adjust=False. 'Adj Close' is adjusted for both splits and "
            "cash distributions, and is what returns are built from. 'Close' is "
            "split-adjusted but NOT distribution-adjusted, so the two columns "
            "differ by distributions only. Verified in clean.py by "
            "verify_adjustment_convention (distributions) and "
            "verify_split_convention (splits); results in "
            "data/processed/CLEANING_LOG.json."
        ),
        "versions": {
            "yfinance": yf.__version__,
            "pandas": pd.__version__,
            "requests": requests.__version__,
        },
        "prices": prices,
        "risk_free": risk_free,
    }

    if MANIFEST_PATH.exists():
        previous = json.loads(MANIFEST_PATH.read_text())
        # Carry forward the correction history so the audit trail survives a
        # re-download.
        if "manifest_corrections" in previous:
            manifest["manifest_corrections"] = previous["manifest_corrections"]

    if extend_report:
        manifest["extend_report"] = extend_report
        total_appended = sum(r["rows_appended"] for r in extend_report.values())
        revised = {
            ticker: report["upstream_revisions_detected"]
            for ticker, report in extend_report.items()
            if report["upstream_revisions_detected"]
        }
        print(f"\nappended {total_appended} row(s) across {len(extend_report)} tickers")
        if revised:
            print(
                f"WARNING: Yahoo has revised {sum(len(v) for v in revised.values())} "
                f"value(s) on dates already held, across {len(revised)} ticker(s). "
                "Existing rows were KEPT as-is. See manifest extend_report."
            )
            for ticker, revisions in revised.items():
                for revision in revisions[:3]:
                    print(
                        f"  {ticker} {revision['date']} {revision['column']}: "
                        f"{revision['old']:.6f} -> {revision['new']:.6f}"
                    )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
