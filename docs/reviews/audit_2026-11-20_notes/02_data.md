# 02 Data and calibration (working notes)

**TL;DR / In plain terms:** the data pipeline is reproducible from committed raw files and I rebuilt its outputs with separate code. Every number the docs quote for Part I matched.

Sources: `docs/methods/DATA_CLEANING.md`, `src/data/{universe,download_raw,clean}.py`, `data/raw/MANIFEST.json`, `data/processed/*`.

## Facts and where they live
- Download: 2026-09-24 (`data/raw/MANIFEST.json: downloaded_at_utc` = 2026-09-24T15:03:51+00:00), Yahoo via yfinance 1.7.0, `auto_adjust=False`, returns from `Adj Close`.
- Risk-free: Kenneth French daily factors file (`data/raw/risk_free/`), percent per day ÷ 100; ends 2026-07-31, which sets the sample end.
- Sample: 5,152 daily observations, 2006-02-07 to 2026-07-31 (`data/processed/excess_returns.csv`). Start set by DBC inception 2006-02-06.
- Reconciliation (DATA_CLEANING §3): 5,214 raw union dates → drop in-progress session (1) → pre-DBC (23) → vendor gap 2026-09-22 (1) → differencing (1) → restrict to RF coverage (36) → 5,152.
- Simple returns; inner join; nothing imputed; extreme moves kept (EEM +22.77% 2008-10-13, +20.93% 2008-10-28).
- Adjustment verified two ways (§5): distributions identity (median abs disagreement about 1e-7, worst in-sample 6.0e-4), EEM 3:1 split 2008-07-24 (ratio 0.9630 in both columns).
- Annualized stats (`data/processed/summary_statistics.csv`): SPY 10.7% / 19.3% / 0.55; EFA 6.1 / 21.4 / 0.29; EEM 7.5 / 28.0 / 0.27; IEF 1.7 / 6.8 / 0.25; LQD 2.6 / 8.6 / 0.30; TIP 1.7 / 6.2 / 0.28; GLD 9.2 / 18.4 / 0.50; VNQ 8.9 / 29.0 / 0.31; DBC 2.3 / 19.3 / 0.12. Excess kurtosis: LQD 58.9 (2008), EEM 17.8, VNQ 17.2, SPY 15.0.
- Correlations (`correlation_matrix.csv`): SPY–EFA 0.88, SPY–EEM 0.82, IEF–TIP 0.76, SPY–IEF −0.29, SPY–GLD 0.07.
- Σ: condition number 268.07, smallest eigenvalue 3.10e-6 (`covariance_daily.csv`; `CLEANING_LOG.json: covariance_diagnostics`).
- γ justification: DATA_CLEANING §8.4 (Elminejad et al. 2022 range 2 to 7; Barsky et al. 1997 median about 7, from a secondary source, primary table not checked; Mehra–Prescott cap 10). Decided 2026-10-16 before any real run. A data-derived candidate (2.9) was rejected to avoid using the dataset.
- French RF is quantized (three values); bias about 6 bp/yr; argmax unaffected under 1'w = 1 (checked to 3.6e-9 in the doc).

## Independent check I ran (2026-11-20, scratch script, not committed)
Rebuilt excess returns from `data/raw/prices/*.csv` and the French CSV with separate pandas code:
- 5,152 rows, 2006-02-07 to 2026-07-31; columns in `TICKERS` order; max abs difference from `excess_returns.csv` 1.0e-16.
- Covariance (ddof = 1) differs from `covariance_daily.csv` by at most 9.3e-17; condition number 268.07; min eigenvalue 3.10e-6.
- Annualized means and vols match `summary_statistics.csv` to the printed digits.

## Gaps against the assignment
- None found for I-1 to I-11. The report needs the Part I table (period, n, annualized stats, correlation matrix); both tables exist as CSV, but no figure or formatted table exists for the report yet.
- Not done (optional under Part VI): alternative calibration window, shrinkage covariance, fat-tailed returns. The report should discuss them as limitations (LQD kurtosis 58.9 against a Gaussian simulator).
