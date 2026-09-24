# Part I — Data Construction and Cleaning

**Owner:** Albert · **Status:** complete, reviewed, pending William's sign-off
**Download date:** 2026-09-24 (America/New_York)
**Final sample:** 5,152 daily observations, 2006-02-07 to 2026-07-31

---

## TL;DR / In plain terms

We downloaded 20 years of daily prices for the nine ETFs the assignment
specifies, lined them up so every date has all nine, turned them into daily
returns, subtracted the risk-free rate, and checked the result several different
ways. What comes out is a 5,152-row table of daily excess returns. Everything
later in the project (the theoretical optimum, the simulated environment, the
Q-learning agent) is built from that table and its average and covariance.

**Key concepts to understand**

- **Excess return.** Return above the risk-free rate. If SPY returned 0.5% on a
  day when T-bills paid 0.01%, the excess return is 0.49%. We model excess
  returns because that is what the assignment specifies, and because it makes
  "put 100% of your money somewhere" a sensible constraint.
- **Adjusted close.** A price series rewritten so that dividends are treated as
  reinvested. Using raw prices would make every dividend look like a loss.
- **μ and Σ.** The average return of each asset (μ, a list of 9 numbers) and how
  they move together (Σ, a 9×9 grid). These two objects *are* the simulated
  world the agent trains in.
- **Inner join.** Keep only dates present in every series. Used here instead of
  filling gaps, because a made-up price creates a made-up return.

**Key takeaways**

1. The sample runs 2006-02-07 to 2026-07-31. The start is set by DBC's launch,
   the end by the risk-free rate file, not by the ETF data.
2. The covariance matrix is well behaved (condition number 268), so Part II can
   invert it directly without shrinkage.
3. The correlations look like real markets, which is the main evidence nothing
   is scrambled: stocks move together, Treasuries move opposite to stocks, gold
   is roughly independent.
4. Two decisions are still William's: the value of γ, and the reward formula.

---

## 1. How to reproduce

```bash
python3 -m src.data.download_raw   # writes data/raw/   (network required)
python3 -m src.data.clean          # writes data/interim/ and data/processed/
python3 -m pytest src/tests -q     # 55 tests
```

`download_raw.py` is the only script that reaches the network and the only one
that writes to `data/raw/`. Running it again overwrites the raw sample, which is
not a neutral action (see §7.1).

### Directory layout

| Path | Contents | In git? |
|---|---|---|
| `data/raw/prices/<TICKER>.csv` | Untouched yfinance output per ETF: OHLCV, Adj Close, Dividends, Stock Splits, Capital Gains | Yes |
| `data/raw/risk_free/` | Kenneth French daily factor zip and extracted CSV | Yes |
| `data/raw/MANIFEST.json` | Download timestamp, library versions, per-file SHA-256, row counts, date spans, adjustment convention | Yes |
| `data/interim/` | Working copy of `data/raw`, rebuilt on every clean run | No, regenerable |
| `data/processed/` | Cleaned, aligned, analysis-ready outputs | Yes |

`data/raw` is committed deliberately. Yahoo restates adjusted-close history as
new distributions accrue, so re-downloading in December would produce different
μ and Σ than the ones in our report. The committed bytes are the reproducibility
anchor. `data/interim` exists so a botched cleaning run can be rebuilt without
re-downloading.

### Processed outputs

All published files cover the same 5,152-day sample, 2006-02-07 to 2026-07-31.
The one exception is `adj_close.csv`, which carries one extra leading row
(2006-02-06) so that it can regenerate `returns_simple.csv` exactly. A test
enforces that.

| File | Rows | Contents |
|---|---|---|
| `adj_close.csv` | 5,153 | Aligned adjusted closes, plus the base price |
| `returns_simple.csv` | 5,152 | Simple total returns |
| `rf_daily.csv` | 5,152 | Daily risk-free rate, decimal |
| `excess_returns.csv` | 5,152 | **The modeling dataset.** μ and Σ come from here |
| `summary_statistics.csv` | 9 | Per-asset annualized stats (Part I requirement) |
| `correlation_matrix.csv` | 9×9 | Correlation matrix (Part I requirement) |
| `covariance_daily.csv` | 9×9 | Daily covariance, the Σ the environment uses |
| `CLEANING_LOG.json` | — | Machine-readable record of every count and decision below |

---

## 2. Universe

Prescribed by the assignment, not chosen by us:

| Segment | Tickers |
|---|---|
| Equity | SPY, EFA, EEM |
| Fixed income | IEF, LQD, TIP |
| Real assets | GLD, VNQ, DBC |

Ticker order is fixed in `src/data/universe.py` and is the column order
everywhere downstream, so weight vectors from the benchmark optimizer and from
the Q-learning agent stay comparable without re-indexing. A test enforces it.

---

## 3. Sample reconciliation

Every row removed, and why:

| Step | Rows | Removed | Reason |
|---|---|---|---|
| Raw download, union of all trading dates | 5,214 | — | 2006-01-03 to 2026-09-24 |
| Drop in-progress session | 5,213 | 1 | 2026-09-24 was still trading when downloaded (§4.1) |
| Drop dates before DBC inception | 5,190 | 23 | DBC's first trade is 2006-02-06 (§4.2) |
| Drop vendor gap dates | 5,189 | 1 | 2026-09-22 (§4.3) |
| Differencing to returns | 5,188 | 1 | First date has no prior price |
| Restrict to risk-free coverage | **5,152** | 36 | French file ends 2026-07-31 (§4.4) |

Final: **5,152 daily observations, 2006-02-07 to 2026-07-31**, about 20.5 years.

---

## 4. Cleaning decisions and assumptions

### 4.1 The final bar was dropped as incomplete

The download ran mid-session on 2026-09-24. Yahoo returns a bar for the
in-progress day with partial volume and a non-final close: SPY showed 9.1M
shares against a trailing-250-day median of 62.6M, and all nine tickers showed
11% to 41% of normal volume. Any bar dated on the download date is dropped.

> **Assumption:** a same-day bar is never a settled observation.

This drop currently removes zero observations from the final sample, because
2026-09-24 already falls past the risk-free cutoff in §4.4. It becomes
load-bearing the moment the French file catches up, or if anyone re-downloads
mid-session.

### 4.2 DBC inception sets the sample start

DBC's first trading day is 2006-02-06, the latest inception in the universe,
exactly as the assignment anticipates. January 2006 is dropped for all nine
ETFs rather than running an unbalanced panel.

> **Assumption:** a balanced panel matters more than 23 extra days of
> eight-asset history. An unbalanced panel would estimate μ and Σ over different
> samples per asset, breaking the internal consistency of the covariance matrix.

### 4.3 Dates are inner-joined, nothing is imputed

Dates are kept only if all nine ETFs have a price. One date, **2026-09-22**, is
present for SPY and GLD but absent for the other seven. It was an ordinary
Tuesday, so this is a gap in Yahoo's data rather than a market holiday.

Forward-filling was rejected. Carrying a stale price forward manufactures a
zero return followed by a double-size one, which corrupts the second moment the
whole project depends on.

> **Assumption:** a portfolio return needs all nine prices on the same date, so
> a date missing any asset carries no usable information.

**Known limitation of this choice.** Dropping a date removes the fake zero but
not the multi-day return that follows it. The return dated the day after a gap
spans two calendar days while being treated as one. Measured on this panel, the
SPY return recorded for 2026-09-23 is −0.736% where the true one-day figure is
−0.720%. Both affected dates fall past the risk-free cutoff, so no such return
is in the current sample, and `CLEANING_LOG.json` records the gap dates so the
situation is auditable if the sample end moves.

### 4.4 The sample ends where the risk-free rate ends

The ETF data runs to 2026-09-23, but the Kenneth French daily factor file
currently ends **2026-07-31**. The sample is truncated to the French coverage,
costing 36 trading days.

> **Assumption:** truncating beats extrapolating. Forward-filling RF for two
> months would fabricate the quantity that defines the excess returns the
> environment is built on. Losing 36 of 5,188 observations (0.7%) has no
> material effect on μ or Σ.

The monthly `F-F_Research_Data_Factors.csv` in the course folder is not usable
here. It is monthly, and the assignment requires returns, moments, risk aversion
and rewards to share one time frequency. `download_raw.py` fetches the daily
file.

### 4.5 Returns are simple, not log

Returns are `AdjClose_t / AdjClose_{t-1} - 1`, computed with `fill_method=None`
so pandas cannot forward-fill a missing price before differencing.

> **Assumption:** mean-variance utility is defined over simple returns, because
> portfolio return is the weighted average of asset simple returns. Log returns
> are not additive across assets, so `w'r` would not be the portfolio log return
> and the Part II objective would no longer match the data.

### 4.6 The environment is defined over excess returns

`r_excess = r_simple − RF_daily`. The French RF is published in **percent per
day** and is divided by 100.

> **Assumption:** the assignment specifies an excess-return process, which also
> makes the full-investment constraint `1'w = 1` coherent: the investor
> allocates wealth across nine risky sleeves funded at the risk-free rate.

### 4.7 Extreme moves are kept

Two observations exceed ±20%: EEM on 2008-10-13 (+22.77%) and 2008-10-28
(+20.93%). Both are genuine crisis prints and are retained.

> **Assumption:** these are data, not errors. Deleting them would bias the
> covariance downward in precisely the states a portfolio agent most needs to
> price, and the Part VI fat-tail robustness check only means something if the
> baseline keeps the tails.

### 4.8 Annualization uses 252 trading days

Summary statistics are annualized as `mean × 252` and `vol × √252`. These are
for reporting only. All modeling runs at daily frequency.

---

## 5. Verification of the adjustment convention

The assignment asks us to verify the adjustment convention rather than assume
it. There are two separate things to establish, and they need two separate
tests.

### 5.1 Distribution adjustment

Prices are downloaded with `auto_adjust=False`, which keeps the unadjusted
`Close` beside `Adj Close`. If `Adj Close` treats distributions as reinvested:

```
AdjClose_t / AdjClose_{t-1} − 1  ==  (Close_t + D_t) / Close_{t-1} − 1
```

where `D_t` is the per-share distribution (Dividends plus Capital Gains). The
right side is reconstructed independently and compared:

- **Median absolute disagreement: about 1×10⁻⁷**, which is floating-point noise.
- **Worst in-sample disagreement: 6.0×10⁻⁴**, on ex-distribution days. This is
  the rounding in Yahoo's published adjustment factors.
- Worst over the full downloaded history is 8.6×10⁻³, all from **VNQ on
  2026-09-23**, an ex-dividend date one day before download where Yahoo had not
  finalized the adjustment factor. It falls outside the retained sample. The
  most recent few days of any Yahoo pull are provisional.

GLD shows exactly zero disagreement across the whole history. It pays no
distributions, so adjusted and unadjusted closes are identical, which is the
control case confirming the test measures what it claims.

### 5.2 Split adjustment

The identity in §5.1 contains no split term, so on its own it shows only that
the two columns share a split convention, not which one. Splits settle it. The
sample contains one split, EEM's 3:1 on 2008-07-24:

| | Ratio across the split date |
|---|---|
| `Close` | 0.9630 |
| `Adj Close` | 0.9630 |

An unadjusted price series would jump by roughly 3.0 here. Both are near 1.0, so
yfinance split-adjusts both columns.

**Conclusion:** `Adj Close` is split- and distribution-adjusted. `Close` is
split-adjusted but not distribution-adjusted. The two differ by distributions
only. An earlier version of `MANIFEST.json` described `Close` as the unadjusted
close, which was wrong; the file records the correction.

Per-ticker figures are in `CLEANING_LOG.json` under `adjustment_verification`
and `split_verification`.

---

## 6. What the data looks like

Annualized over the 5,152-day excess-return sample:

| Ticker | Mean | Vol | Sharpe | Excess kurtosis |
|---|---|---|---|---|
| SPY | 10.7% | 19.3% | 0.55 | 15.0 |
| EFA | 6.1% | 21.4% | 0.29 | 13.9 |
| EEM | 7.5% | 28.0% | 0.27 | 17.8 |
| IEF | 1.7% | 6.8% | 0.25 | 2.7 |
| LQD | 2.6% | 8.6% | 0.30 | 58.9 |
| TIP | 1.7% | 6.2% | 0.28 | 10.2 |
| GLD | 9.2% | 18.4% | 0.50 | 6.5 |
| VNQ | 8.9% | 29.0% | 0.31 | 17.2 |
| DBC | 2.3% | 19.3% | 0.12 | 3.0 |

The correlation structure is economically sensible, which is the main evidence
the pipeline is not scrambled: the equity bloc is tightly coupled (SPY–EFA 0.88,
SPY–EEM 0.82), the two Treasury sleeves track each other (IEF–TIP 0.76),
equities and Treasuries are negatively correlated (SPY–IEF −0.29), and gold is
nearly uncorrelated with equities (SPY–GLD 0.07).

**Covariance conditioning.** Part II inverts Σ in closed form, so its spectrum
was checked up front: condition number **268**, smallest eigenvalue 3.1×10⁻⁶,
positive definite. 5,152 observations for 45 covariance parameters is a
comfortable ratio, so the sample estimator needs no shrinkage to be invertible.
Shrinkage stays available as the Part VI statistical robustness check.

**Fat tails.** LQD's excess kurtosis of 58.9 is driven by the 2008 credit
crisis, not by COVID: its two largest moves are 2008-09-30 (+9.76%) and
2008-09-29 (−9.12%), and removing March 2020 entirely *raises* kurtosis to 62.8.
This bears on the Part VI Gaussian-versus-fat-tailed check, because the
calibrated environment simulates Gaussian returns by construction, so the
simulated world has thinner tails than the data μ and Σ were estimated from.
That is a property of the assignment's design, and belongs in the report's
limitations.

---

## 7. Known limitations

### 7.1 Re-downloading changes the sample
Yahoo restates adjusted history. `download_raw.py` will not reproduce
`data/raw` byte-for-byte on a later date. The committed raw files and their
SHA-256 digests in `MANIFEST.json` are the anchor. Prefer re-running `clean.py`
over `download_raw.py`.

### 7.2 Survivorship and single-vendor risk
All nine ETFs survived the full period, and all prices come from one vendor with
no cross-check against a second source.

### 7.3 One calibration window
μ and Σ use the full 2006 to 2026 sample, which spans the GFC, the 2020 crash
and the 2022 rate shock, and the environment then treats these as stationary.
The alternative-calibration-window check in Part VI is where to probe this.

### 7.4 The French daily rate is quantized
The French daily RF is constant within each month and rounded to 1bp, taking
only three distinct values across the sample (0, 1e-4, 2e-4). Cumulating it
gives 35.06% over the sample against 33.80% from the monthly file, a bias of
about 6bp per year.

This is immaterial under our constraint and it is worth knowing why. With
`1'w = 1`, adding a constant to every element of μ adds a constant to the
objective on the whole feasible set, so the argmax does not move. Verified:
`max|w(μ) − w(μ + bias)| = 3.6e-09`. It would become material if the budget
constraint were relaxed, which is the bounded-leverage bonus extension.

### 7.5 Version pinning
`requirements.txt` does not pin versions. The environment used was numpy 2.3.5,
scipy 1.16.3, pandas 2.3.3, yfinance 1.7.0. `MANIFEST.json` records yfinance,
pandas and requests but not numpy or scipy, both of which affect Part II
(`np.linalg.eigvalsh`, SLSQP).

---

## 8. Handoff to Part II

### 8.1 γ is frequency-invariant, verified rather than assumed

Because returns are i.i.d. in the calibrated environment, μ and Σ both scale
linearly with the horizon, so annualizing multiplies the whole objective by 252
and leaves its **argmax unchanged**:

```
w'μ_a − (γ/2)·w'Σ_a·w  =  252·[ w'μ_d − (γ/2)·w'Σ_d·w ]
```

The same γ means the same thing at daily and annual frequency, so no rescaling
is needed when feeding daily moments to the Part II optimizer. Confirmed on the
calibrated moments: closed-form solutions agree to 4×10⁻¹⁵, long-only solutions
to 2×10⁻⁸.

This holds only while the reward is defined over **per-period** returns. An
episodic reward computed on an episode's *cumulative* return over B days is a
B-period objective, and γ would not carry over.

### 8.2 Blocker: the long-only optimizer fails silently on daily moments

Reported in full in `REVIEW_NOTES_FOR_WILLIAM.md`, issue 1. In short,
`constrained_optimal_weights` uses SLSQP at its default `ftol=1e-6`, which at
daily scale is comparable to the entire objective, so the optimizer stops
prematurely and returns a wrong portfolio while reporting success. At γ = 5 it
gives up 58.6% of the achievable objective. Part II cannot produce a meaningful
continuous benchmark until this is fixed.

### 8.3 Sanity check of the calibrated environment

With the tolerance corrected, the long-only continuous optimum behaves as theory
predicts, shifting from risk assets to Treasuries as γ rises:

| γ | Ann. return | Ann. vol | Holdings | Discretization gap |
|---|---|---|---|---|
| 1 | 10.2% | 14.9% | SPY 0.70, GLD 0.30 | 0.36% |
| 3 | 10.0% | 13.9% | SPY 0.55, GLD 0.45 | 0.35% |
| 5 | 8.9% | 11.9% | SPY 0.47, GLD 0.39, IEF 0.14 | 1.08% |
| 8 | 6.8% | 8.7% | IEF 0.39, SPY 0.36, GLD 0.25 | 1.62% |
| 15 | 5.1% | 6.6% | IEF 0.53, SPY 0.26, GLD 0.14, TIP 0.06 | 9.99% |

The final column is how much of the continuous objective is lost by restricting
to the 1,287-action 20% grid. It is small at low γ, where the optimum is
concentrated enough for a coarse grid to represent it, and reaches 10% at
γ = 15, where a 6% position cannot be expressed in 20% steps.

These portfolios are concentrated: at γ ≤ 3 the optimum holds two of nine
assets. That is the standard corner-solution behaviour of mean-variance
optimization, which treats noisy estimates of μ as certain, and it is the
concrete form the Part VI estimation-error discussion should take.

### 8.4 The one decision still open

**What value of γ, and on what grounds?** The units question is settled in §8.1.
The level is not, and the assignment requires a clearly stated risk-aversion
parameter.

γ should be justified on external grounds, for example the standard empirical
range for this class of problem, or matching whatever the original three-asset
version of this problem used. It should **not** be chosen by inspecting the
table in §8.3 and picking the value whose answer looks best. Part VI explicitly
asks about benchmark leakage, and selecting the headline economic parameter to
produce a non-degenerate optimum is a researcher-degrees-of-freedom problem a
grader can name.

Once γ is fixed on outside grounds, report the resulting concentration and
discretization gap as findings.

---

## 9. Review history

An independent agent review on 2026-09-24 verified the sample reconciliation,
the RF parsing, the adjustment convention, the γ invariance claim, the
covariance conditioning and the summary statistics, and reproduced every
processed file byte-for-byte. It found no look-ahead bias, no leakage of μ or Σ
into the action space, and no date-alignment errors.

It also found five defects, all since fixed:

| Finding | Fix |
|---|---|
| Tests were self-referential; excess returns with no RF subtracted passed all 14 | Tests rewritten to re-derive from `data/raw` and pin literal sample values; verified by mutation testing |
| `adj_close.csv` could not reproduce `returns_simple.csv`; sample truncated at the wrong end | `restrict_to_final_sample` clips all files to one sample and keeps the price base row |
| `pct_change()` forward-filled by default | `fill_method=None` stated explicitly |
| `MANIFEST.json` called `Close` unadjusted; it is split-adjusted | Corrected, and split adjustment now verified separately (§5.2) |
| §6 attributed LQD's kurtosis to March 2020; it is 2008 | Corrected |
