# Report Skeleton: Model-Free Q-Learning for Mean-Variance Portfolio Choice

**By:** Claude, for Albert and William · **First written:** 2026-11-20 · **Updated:** 2026-11-23 (transaction-cost results filled, per-run takeaways, figure inventory, slide outline) · **Status:** skeleton; no report text written yet
**Built from:** `docs/reviews/AUDIT_2026-11-20.md` and its notes folder, `reports/registry.json`, every `reports/*/report.html` and `data.json`, and `results/tc-drift/runs/`. Every number below names the committed file it comes from. The 2026-11-23 additions were read from those files on that date, at commit `2169c7e`.

---

## TL;DR / In plain terms

This is the outline of the 10 to 15 page report, section by section, in the order the assignment requires. Each section lists what it has to claim, the exact numbers with the file they come from, the figures to use, the caveats that must be printed, and a line for the slide deck. Section 8 lists every reusable figure in the repo and how to get it out. Section 9 is a 14-slide outline. A writer or a slide builder can work from this file without re-checking the repo.

**Key concepts to understand**

- **Three portfolios, two gaps.** The continuous optimum w* is the best long-only portfolio with any weights. The best grid portfolio is the best one the agent is allowed to pick. The learned portfolio is what the agent picked. Discretization gap = best grid vs w*. Learning gap = learned vs best grid. Total gap = their sum.
- **Certainty-equivalent loss in bps per year.** Every gap is 252 × 10,000 × (difference in daily utility U). It reads as "the investor gives up this many basis points of sure return per year".
- **Exact vs sampled evaluation.** Gaps are computed from the known μ and Σ, since a 10,000-episode sample has a standard error of about 40 bps/yr, larger than the gaps. The fresh evaluation draws are reported beside them.
- **Pre-registered vs post hoc.** Only the four `baseline-v1` experiments had settings and convergence rules fixed before any result. Every other run is labelled post hoc wherever it appears. Several post hoc runs had an expected direction written down before they ran; those are "registered expectations" and are graded met or not met.
- **Where the agent heads vs when it trades (transaction costs).** With trading costs the agent's shortfall against the oracle splits into the portfolio it aims for and its decisions about when to trade. The two are reported separately (§6.5).

**Key takeaways**

- Headline answer to the research question: at γ = 5 on the 20% grid, the median seed ends 14.1 bps/yr behind the best grid portfolio and 19.8 bps/yr behind the continuous optimum, out of 530.5 bps/yr available; 3 of 10 seeds pick the best grid portfolio exactly. Equal weight is 310.9 bps/yr behind.
- What prevented exact recovery: reward noise (about 3,900 bps/yr per 20-day episode) against a 10.5 bps/yr difference between the best and second-best grid portfolio.
- Finer grids shrink the discretization gap and make learning harder. On the 10% grid the two cancel at γ = 5 (paired change −0.6, SE 3.7); at γ = 3 and 8 the finer grid only adds learning error (+18.0, SE 7.5; +4.6, SE 1.6). The coarser 25% grid has the smaller median total gap at γ = 5 (14.3 against 19.8).
- Function approximation beats the like-for-like table on the 20% grid at γ = 5 and 3, and loses to the table at every γ on the 10% grid (post hoc, negative result).
- Transaction costs (post hoc): the agent loses to the oracle at every cost level, by 13.1, 13.1, 10.0 and 2.8 bps/yr at zero, $100M, $1B and $10B. At the three lower levels the gap is mostly where it heads (the wrong target portfolio). At $10B it is mostly timing: the agent's timing captures 6.2 of the oracle's 7.8 bps/yr lead over always rebalancing, and it beats always rebalancing by 5.0. 8 of 12 registered expectations are met and 4 partly met.

Unless stated, numbers are bps/yr of certainty-equivalent loss, 10 seeds, master seed 20261016.

---

## 0. Map: assignment requirement → report section

| Assignment item | Report section |
|---|---|
| Central question; the four separations (continuous vs grid, learning vs discretization error, calibration vs fresh evaluation, statistical vs economic convergence) | §1, answered in §5 and §7 |
| Part I: data, sample, returns, alignment, adjustment, frequency, summary table, correlation matrix, fixed moments, no leakage | §2 |
| Part II: γ, closed form derived or verified, numerical long-only solve, identical constraints, two benchmarks | §3 |
| Part III: state, action counts, no optimum in the action set, reward, update, ε-greedy, "report and justify" list | §4 |
| Part IV: training, fresh evaluation, seeds, convergence defined first, observations and episodes, share of seeds, effect of action-set size | §5.1, §5.2 |
| Part V: weights chart and distances, performance table, gaps and their distribution, learning curve, regret, two resolutions | §5.3 to §5.5 |
| Part VI: two robustness checks, six discussion topics | §6.1 to §6.3 |
| Bonus extensions (FA, transaction costs) | §6.4, §6.5 |
| "what the agent learned, how close it came to the best attainable action, and what prevented exact recovery" | §5.6, §7 |
| Code and README reproducibility | Appendix A |

Rubric buckets (20% each; `audit_2026-11-20_notes/assignment_text.txt` lines 222 to 228): Data construction and calibrated environment → §2. Theoretical benchmark and feasible-set design → §3. Q-learning design and implementation → §4. Convergence analysis and benchmark comparison → §5. Economic interpretation, robustness, and insight → §6, §7. The assignment's report structure (lines 201 to 209) starts with "Introduction and central research question" and ends with "Conclusion", which §1 and §7 match.

Page budget (suggested): §1 one, §2 one and a half, §3 one and a half, §4 two, §5 three and a half, §6 four, §7 half. Appendix outside the count. §6.5 now has results, so it needs about one page of the four; take it from §6.2 and §6.4 by keeping each to a table and a paragraph.

Other parts of this file: §8 figure inventory (IDs F1 to F37 and the figures still to build, N1 to N7), §9 slide outline, then open decisions and the issues found while updating.

---

## 1. Introduction and central research question

**Purpose.** State the question and how the report separates the four things the assignment asks to separate.

**Claims.**
- Question (quote): "Can a model-free learning algorithm recover a near-optimal portfolio allocation when it observes realized rewards but does not know expected returns or covariances?"
- Setup in two sentences: nine ETFs, a Gaussian simulator calibrated once, a stateless Q-learning agent choosing among 1,287 long-only portfolios.
- Preview the answer with the three headline numbers in the takeaways above.
- Say which results were pre-registered and which were added later.

**Numbers.** 14.1 and 19.8 median, 530.5, 310.9: `results/baseline_g5_grid20/evaluation/summary.json` (`learned.ce_loss_vs_best_discrete_bps.median`, `learned.ce_loss_vs_continuous_bps.median`, `benchmarks.*.exact.objective_daily`).

**Figures.** None in the report. Slides: N7 (gap ladder, to build) or F8.

**Caveats.** None here beyond the pre-registered / post hoc sentence.

**Slide takeaway.** A model-free agent gets within about 20 bps/yr of the optimum out of 530; reward noise stops it getting the rest.

---

## 2. ETF data, sample construction, and calibration

**Purpose.** Show the environment is built from clean, documented data at one frequency. (Bucket: data and environment.)

**Claims.**
- Nine ETFs (SPY, EFA, EEM, IEF, LQD, TIP, GLD, VNQ, DBC), Yahoo Adjusted Close, downloaded 2026-09-24 with yfinance 1.7.0, raw files committed and never re-downloaded.
- Risk-free rate from the Kenneth French daily file; environment defined on excess returns.
- Sample 2006-02-07 to 2026-07-31, 5,152 daily observations. Start set by DBC's first trading day (2006-02-06); end set by the risk-free file.
- Simple returns throughout; no log returns anywhere. Dates aligned by inner join; nothing imputed; row count reconciled 5,214 → 5,152 (drops of 1, 23, 1, 1, 36).
- Adjustment convention verified two ways (distribution identity; EEM 3-for-1 split on 2008-07-24).
- Everything daily: μ, Σ, γ, rewards. Annualization (×252, ×√252) for reporting only.
- Simulator: r = μ + Lz, L the Cholesky factor of Σ, moments fixed. Training, agent randomness and evaluation use separate random streams from one master seed.
- The agent's code cannot import the data loader, benchmarks or data files (tests enforce it).

**Numbers and sources.**
- Table 1 (annualized mean / volatility / Sharpe): SPY 10.7% / 19.3% / 0.55; EFA 6.1 / 21.4 / 0.29; EEM 7.5 / 28.0 / 0.27; IEF 1.7 / 6.8 / 0.25; LQD 2.6 / 8.6 / 0.30; TIP 1.7 / 6.2 / 0.28; GLD 9.2 / 18.4 / 0.50; VNQ 8.9 / 29.0 / 0.31; DBC 2.3 / 19.3 / 0.12. Source `data/processed/summary_statistics.csv`. The same means and volatilities, with the standard error of each mean (4.27, 4.73, 6.20, 1.51, 1.91, 1.37, 4.08, 6.42, 4.27 percent a year), are in `reports/full_run_2026-10-22/data.json` (`calibration.mean_annual_pct`, `.vol_annual_pct`, `.se_mean_annual_pct`); the standard errors feed §6.3's estimation-error row.
- Table 2 (correlation matrix): `data/processed/correlation_matrix.csv`. Quote SPY–EFA 0.88, SPY–EEM 0.82, IEF–TIP 0.76, SPY–IEF −0.29, SPY–GLD 0.07.
- Σ condition number 268.07, smallest eigenvalue 3.10e-6: `data/processed/CLEANING_LOG.json` (`covariance_diagnostics`), `covariance_daily.csv`.
- Download date: `data/raw/MANIFEST.json`. Row reconciliation and adjustment checks: `docs/methods/DATA_CLEANING.md` §3, §5.

**Figures and tables.** Tables 1 and 2 must be formatted from the two CSV files (no figure exists yet). Optional: correlation heatmap (N1, to build).

**Caveats and disclosures.**
- Excess kurtosis is large (LQD 58.9, EEM 17.8, VNQ 17.2, SPY 15.0) and the simulator is Gaussian. Forward reference to §6.3.
- The French risk-free series takes few distinct values; effect on the optimum checked to 3.6e-9 (`DATA_CLEANING.md`).
- Extreme days kept (EEM +22.77% on 2008-10-13).

**Slide takeaway.** 5,152 daily observations, 2006 to 2026, one frequency, raw data frozen; the agent only ever sees simulated returns.

---

## 3. Theoretical benchmark and feasible portfolio set

**Purpose.** Define the objective, derive the closed form, solve the constrained problem the agent faces, and report both benchmarks. (Bucket: benchmark and feasible set.) Audit finding H2 (no methods prose for Part II) is still open on 2026-11-23: there is no `docs/methods/BENCHMARKS.md`, so this section is written from the `src/benchmarks/mean_variance.py` docstring and the tests.

**Claims.**
- Objective U(w) = w'μ − (γ/2)w'Σw, daily, γ = 5 baseline (3 and 8 as robustness). Justification of γ = 5: literature range 2 to 7 (Elminejad et al. 2022), household survey median near 7 (Barsky et al. 1997, taken from a secondary source), cap of 10 (Mehra and Prescott). Fixed 2026-10-16 before any run; a data-derived value (2.9) was rejected so the dataset did not pick its own γ.
- Budget-only closed form: w = (1/γ)Σ⁻¹(μ − η1) with η chosen so 1'w = 1. Give the three-line Lagrangian derivation. State how it was verified: the tests compare it with an independent scipy solve, check 1'w = 1, and check that the long-only solver returns the same answer whenever the closed form happens to be long-only (`src/tests/test_mean_variance.py`).
- The closed form shorts four ETFs heavily, so it is not the agent's benchmark. The agent's feasible set is long-only and fully invested, and the benchmark is solved numerically under the same constraints.
- Two benchmarks for every specification: continuous long-only optimum, and best grid portfolio by full enumeration.
- The grid is built from the asset count and the step only; the optimum is not inserted.

**Numbers and sources.**
- Table 3, long-only optimum and best grid portfolio (source for each γ: `results/{baseline_g5_grid20,robust_g3_grid20,robust_g8_grid20}/evaluation/summary.json`, `benchmarks.continuous` and `benchmarks.best_discrete`; `weights.csv`):

| γ | w* (SPY, IEF, GLD; rest 0) | Return | Volatility | U(w*) bps/yr | Best 20% grid (SPY, IEF, GLD) |
|---|---|---|---|---|---|
| 3 | 0.5502, 0, 0.4498 | 10.00% | 13.89% | 710.8 | 0.6, 0, 0.4 |
| 5 | 0.4688, 0.1363, 0.3948 | 8.86% | 11.92% | 530.5 | 0.4, 0.2, 0.4 |
| 8 | 0.3562, 0.3899, 0.2539 | 6.79% | 8.66% | 379.2 | 0.4, 0.4, 0.2 |

- Best 20% grid at γ = 5: return 8.28%, volatility 11.01%, U = 2.0824e-4 daily. Equal weight: 5.62%, 11.71%, U = 8.714e-5 daily. Source `results/baseline_g5_grid20/evaluation/performance.csv`.
- Discretization gap by grid step (25% / 20% / 10%): γ = 5: 11.74 / 5.74 / 1.29. γ = 3: 2.51 / 2.47 / 2.47. γ = 8: 22.60 / 6.13 / 6.13. Sources: `summary.json: discretization_gap.ce_loss_bps` in `results/resolution_g5_grid25/`, the three 20% folders, and `results/grid10-robustness/*/`; the same values are `summary.disc_gap_bps` in `reports/{baseline-v1,grid10-robustness}/data.json`. The 25% values at γ = 3 and 8 are the audit's calculation only; no committed run has them, so leave them out or mark them.
- Budget-only closed form at γ = 5: weights (1.82, −0.88, −0.19, 0.27, −0.02, −0.27, 0.69, −0.13, −0.28) in ticker order, U = 925.9 bps/yr. **No committed source.** Audit calculation only (`docs/reviews/AUDIT_2026-11-20.md` H3, still open on 2026-11-23: `src/benchmarks/` holds `mean_variance.py` only). Do not print until a committed script produces it.

**Figures and tables.** Table 3. No existing figure; the weights chart in §5 (F1) shows both benchmarks. Optional: weights of w* by γ (N3, to build).

**Caveats and disclosures.**
- At most three of nine ETFs have positive weight in w* at any γ (two at γ = 3). The grid's best portfolio shares that support.
- Bounded shorting (bonus) was not attempted.
- 5% grid (3,108,105 portfolios) is counted, not trained.

**Slide takeaway.** Same constraints for theory and agent; at γ = 5 the best grid portfolio is 5.7 bps/yr below the continuous optimum.

---

## 4. Q-learning design and implementation

**Purpose.** Specify the agent completely and justify each setting. (Bucket: Q-learning design.) This section carries the audit's largest rubric risk (H1, still open on 2026-11-23: no "why these settings" text exists in any methods doc). Each row below needs a reason beside its value.

**Claims.**
- State: none (stateless). Each episode the agent picks one portfolio and holds it for B days.
- Actions: every long-only portfolio with weights in multiples of 20%: 1,287. Comparison grids: 25% (495) and 10% (43,758).
- Reward: realized mean of the portfolio's daily excess return over the episode minus (γ/2) times its realized variance (ddof = 1). Its expectation equals U(w) exactly, so the reward and the benchmark objective agree.
- Update: Q(a) ← Q(a) + α[r − Q(a)], the stateless form of the Q-learning update with discount ρ = 0.
- Exploration: ε-greedy, ε_t = exp(−βt).
- Training and evaluation draws are separate streams; the agent sees rewards and its own action indices only.

**Settings table (values from `src/config.py`; reasons to be written, drafts given).**

| Setting | Value | Reason to give |
|---|---|---|
| Learning rate | α = 1/N(a), N(a) = visits to a | Makes Q(a) the sample mean of a's rewards, the unbiased estimate in a stationary problem; satisfies the usual step-size conditions. `tabular-const` (§6.4) shows what a constant α costs: mean learning gap 43.5 against 17.6 at γ = 5 |
| Discount | ρ = 0 | No state carries over between episodes; the assignment's "clean bandit-like benchmark" |
| Initial Q | 0 | No prior information; with 1/N the first visit overwrites it |
| Returns per episode | B = 20 days | About one trading month; long enough for a variance estimate, short enough for 13M updates. `b63` tests the choice (§6.2) |
| Episodes | 13,000,000 per seed (260M simulated days) | **Disclose:** sized from a pilot of about 13M episodes that came within roughly 2% of the best grid portfolio's value (`docs/reviews/REVIEW_HISTORY.md` §4). The pilot was scored against the benchmark by us; the agent never saw it. `n26m` doubles it (§6.2) |
| Exploration | ε₀ = 1, β ≈ 7.08e-7 so ε = 1% at the halfway episode | First half explores (about 1.41M random picks, about 1,097 per portfolio), second half mostly exploits. `eps-decay` tests 3% (§6.2) |
| Seeds | 10, master seed 20261016 | Enough to report a distribution; cost about 3.5 hours for the baseline set |
| Tie-breaking | Random among maxima when acting; lowest index when reporting the final portfolio | Random ties avoid favouring low-index portfolios during learning; a fixed rule makes the reported portfolio reproducible |
| Evaluation | 10,000 fresh episodes, exploration off | Separate stream, never used for updates |

**Numbers and sources.** `src/config.py`; every `results/*/manifest.json` records the settings; `docs/methods/EXPERIMENTS.md` §1; the same settings are in the "Run settings" table of `reports/baseline-v1/report.html` (section 2 of that page). Reward SD at B = 20 for the best grid portfolio: 0.0015478 daily = 3,900 bps/yr, 7.43 times its utility (`reports/full_run_2026-10-22/data.json: calibration.reward_sd_best_daily`, `.u_best_daily`). The audit's separate calculation gives 3,909 bps/yr and 7.45 times U(w*) (`audit_2026-11-20_notes/03_environment_benchmarks.md`); the two differ by 0.2% and no committed file holds 3,909, so print "about 3,900".

**Figures.** Optional: ε schedule (N2, to build). None existing.

**Caveats and disclosures.**
- Settings fixed in commit `07bfb64` (2026-10-16); first full run 2026-10-22.
- An illustrative demo (γ = 5, B = 20, 2M episodes) ran before the settings were fixed (`docs/methods/EVALUATION.md` §6).
- Benchmark information entered one design choice (the episode budget). It did not enter the action set or the agent.
- Outputs count simulated days; episodes = days / B.

**Slide takeaway.** One table value per portfolio, sample-average updates, exploration decaying to 1% by halfway; every setting fixed before the first real run.

---

## 5. Convergence and benchmark comparison

**Purpose.** Report what was learned, how fast, how reliably, and how far from each benchmark. (Bucket: convergence and comparison.) Everything in this section is the pre-registered run `baseline-v1` (`reports/baseline-v1/report.html`; the original hand-built page is `reports/full_run_2026-10-22/report.html`).

**Run takeaways (`reports/registry.json`, `baseline-v1` notes).** All four experiments trained 10 seeds with no crashes. On every grid and γ the agent learns to hold SPY and GLD, often with one bond slice. Exact recovery of the best grid portfolio happens in 3 to 5 seeds of 10. Every seed settles on its final portfolio in the first third of training; the pre-registered rule reports later dates because of brief excursions. Caveat from the registry: the tag `baseline-v1` points at commit `f70296a`, which has later behaviour-equivalent edits; training ran at `80a6e9c`, recorded in every manifest.

### 5.1 Convergence definition (fixed before results)

**Claims.**
- Primary rule (`greedy_stable_through_end`): the greedy portfolio stops changing and stays unchanged to the end of training, for at least K = 10% of episodes (1.3M). Uses the agent's own Q-table only.
- Also pre-registered: `greedy_stable`, `dq_below_tol_for_good` (largest Q change below 1e-5 per 10,000-episode window), `near_optimal_for_good` (within 25 bps/yr of the best grid portfolio from then on; uses the benchmark, evaluation only).
- Statistical vs economic convergence: the first three are statistical, the fourth is economic.

**Caveats.** `settled` and `near_optimal_every_episode` were added 2026-11-09 and the Q-fit measures 2026-11-04. Label them post hoc and descriptive. The 25 bps line is 3.5%, 4.8% and 6.7% of the best grid value at γ = 3, 5, 8 and contains 2, 4 and 2 of the 1,287 portfolios (`reports/full_run_2026-10-22/data.json: robust.g{3,5,8}.n_within_near`).

### 5.2 How often and how fast

**Table 4 (source `results/INDEX.md`; each row's `evaluation/summary.json: convergence.*`; the same values are `summary.primary`, `.settled`, `.near_ckpt` in `reports/baseline-v1/data.json`).**

| Experiment | Primary rule met | Median days to meet it | Settled, median days (post hoc) | Near-optimal for good | Median days |
|---|---|---|---|---|---|
| γ = 5, 20% grid | 9 of 10 | 185.1M | 47.1M | 7 of 10 | 25.8M |
| γ = 5, 25% grid | 10 of 10 | 101.2M | 18.5M | 9 of 10 | 9.6M |
| γ = 3, 20% grid | 8 of 10 | 192.6M | 26.4M | 6 of 10 | 22.2M |
| γ = 8, 20% grid | 9 of 10 | 153.2M | 27.9M | 7 of 10 | 25.8M |

Episodes = days / 20. The primary rule's count includes the K waiting window.

**Figures.** F12 (when each convergence measure is reached, 20% vs 25%), F14 (each seed's greedy portfolio over time).

**Caveats.** Non-converged seeds: 1 (γ = 5), 2 (γ = 3), 1 (γ = 8) on the 20% grid. Their weights and gaps are still reported in §5.3 and §5.4 (say so). Near-optimal read at checkpoints (25.8M days) is about six times faster than read at every episode (155.7M days at γ = 5; `reports/baseline-v1/data.json: summary.near_episode.median`).

### 5.3 Learned weights against the benchmarks

**Claims.** At γ = 5 every seed holds SPY (40% to 60%) and GLD (20% to 40%); 8 of 10 add a bond ETF (IEF in 6, LQD in 2, TIP in 1; seed 8 holds two) and seeds 1 and 6 hold none; 5 of 10 hold exactly the optimum's three assets (`results/baseline_g5_grid20/evaluation/weights.csv`); 3 of 10 are the best grid portfolio (0.4, 0.2, 0.4); the 10 seeds end on 7 distinct portfolios (`reports/baseline-v1/data.json: summary.distinct_final_portfolios`). Report per seed, never the average across seeds (a blend is not a portfolio any seed chose and can beat the grid).

**Numbers.** L1 distance to w*: median 0.331. Euclidean: median 0.216. Source `summary.json: learned.l1_to_continuous`, `learned.euclidean_to_continuous`. Absolute differences by asset: `weights.csv` (columns `absdiff_vs_continuous_run_*`).

**Figures.** F1, `results/baseline_g5_grid20/evaluation/weights_grouped_bar.png` (nine weights; continuous, best grid, equal weight, learned). Do not use F13 as is: its "learned" bar is the average of the ten seeds' final portfolios (its own caption says so), which the claim above rules out.

**Caveats.** "Average learned weights over a steady-state evaluation sample" is read as the final greedy portfolio with exploration off, so the average equals that portfolio. The chart's "median seed" is the 6th of 10 by gap.

### 5.4 Performance and gaps

**Table 5 (γ = 5; source `results/baseline_g5_grid20/evaluation/performance.csv`; the learned row from `summary.json: learned.{mean_annual,vol_annual,objective_daily}.median`, which is no single seed's pair. The median seed by gap, run 1, has 10.07% / 14.12%, U 508.9, total gap 21.6).**

| Portfolio | Return | Volatility | U (bps/yr) | Loss vs w* |
|---|---|---|---|---|
| Continuous optimum | 8.86% | 11.92% | 530.5 | 0 |
| Best grid | 8.28% | 11.01% | 524.8 | 5.7 |
| Learned, median across seeds (each column's median taken separately; range) | 8.28% | 11.11% | 510.7 | 19.8 (5.7 to 53.1) |
| Equal weight | 5.62% | 11.71% | 219.6 | 310.9 |

**Table 6 (gaps; source each `evaluation/summary.json`, `per_seed.csv`; cross-checked on 2026-11-23 against `reports/baseline-v1/data.json: summary.{disc_gap_bps,learning_gap,total_gap,exact_best}`).**

| Experiment | Discretization gap | Learning gap, mean / median (range) | Total gap, mean / median | Exactly best grid |
|---|---|---|---|---|
| γ = 5, 20% | 5.74 | 17.6 / 14.1 (0 to 47.3) | 23.3 / 19.8 | 3 of 10 |
| γ = 5, 25% | 11.74 | 5.1 / 2.5 (0 to 30.7) | 16.9 / 14.3 | 5 of 10 |
| γ = 3, 20% | 2.47 | 30.6 / 10.0 (0 to 81.1) | 33.0 / 12.5 | 5 of 10 |
| γ = 8, 20% | 6.13 | 13.0 / 5.7 (0 to 53.1) | 19.2 / 11.9 | 5 of 10 |

**Figures.** F2, `results/baseline_g5_grid20/evaluation/ce_loss_by_seed.png` (distribution across seeds). Alternatives with all four experiments on one chart: F18 (decomposition) and F19 (every seed's total gap).

**Caveats.** Gaps use exact μ and Σ. The fresh 10,000-episode sample gives the realized columns in `performance.csv` with a standard error near 1.6e-5 daily (about 40 bps/yr), too coarse to rank portfolios 10 bps apart. Say this in one sentence so the reader knows why both exist.

### 5.5 Learning curve, regret and resolution

**Figures (all exist).**
- F3, `results/baseline_g5_grid20/evaluation/learning_curve.png`
- F4, `results/baseline_g5_grid20/evaluation/cumulative_regret.png` (against the best grid portfolio)
- F5, `results/comparison_g5_grid20_vs_grid25/convergence_by_resolution.png`
- F6, for the 10% grid: `results/grid10-robustness/comparison_g{3,5,8}_grid20_vs_grid10/convergence_by_resolution.png` (post hoc; used in §6.1)
- F7 and F9 are the HTML versions of the 20% vs 25% comparison, with a median line and a band for the middle half of seeds.

**Claims.** Coarser grid: larger discretization gap (11.74 vs 5.74), smaller learning gap (median 2.5 vs 14.1), faster and more reliable convergence (10 of 10 at 101.2M days vs 9 of 10 at 185.1M). Total gap median 14.3 vs 19.8. Median training regret per episode: 65.5 bps/yr on the 20% grid and 58.1 on the 25% grid (`reports/baseline-v1/data.json: summary.regret_avg.median`).

### 5.6 What prevented exact recovery

**Claims.** The second-best grid portfolio is 10.5 bps/yr behind the best at γ = 5 (20.0 at γ = 3, 11.5 at γ = 8), while one episode's reward has a standard deviation near 3,900 bps/yr. Telling the top two apart at two standard errors takes about 1.1 million tries of each on the 20% grid (7.4 million on the 25% grid, where the top two are 5.1 bps/yr apart), and exploration gives each portfolio about 1,097 random visits before the agent narrows to a few. Q-fit supports this: the table ranks portfolios well overall (median Spearman 0.821) and poorly at the very top (top-10 recall 0.35).

**Numbers.** Second-best gaps 20.02, 10.53, 11.46 and the visit counts 1,097,318 and 7,362,199: `reports/full_run_2026-10-22/data.json` (`robust.g{3,5,8}.gap_best_second_bps`, `grids.g20.visits_each_to_separate_2se`, `grids.g25.visits_each_to_separate_2se`, `grids.g25.gap_best_second_bps`). Spearman and recall: `summary.json: q_fit.all` (post hoc measure); `reports/baseline-v1/data.json: summary.q_spearman_all`, `.q_top10_recall_all`.

**Figures.** F16 (how often the best portfolio was tried, in seeds that missed it). Slides: N5 (noise against gap, to build).

**Caveat.** The 2026-11-20 version of this skeleton gave "about 139,000 visits", which is (3,909 / 10.5)², the count for one standard error of a single mean. The committed figure is the two-standard-error count for a difference of two means, eight times larger. Use the committed 1.1 million and say it is an order of magnitude.

**Slide takeaways.**
- 9 of 10 seeds converge by the pre-set rule; median seed is 14 bps/yr from the best available portfolio and 20 from the optimum.
- Coarser grid: easier to learn, worse ceiling. The total barely moves.
- The top portfolios differ by about 10 bps/yr; one episode's noise is about 3,900.

---

## 6. Robustness, research challenges, and limitations

**Purpose.** The two required checks, the further runs, the six discussion topics, and the two extensions. (Bucket: interpretation, robustness, insight.) Everything in §6.2, §6.4 and §6.5 is post hoc; the 10% grid in §6.1 is too.

### 6.1 Action-space design (required check 1)

**Claims.** Three resolutions at γ = 5: 25% (495), 20% (1,287), 10% (43,758). The 10% grid was run with the `all-three` settings (B = 63, 26M episodes, ε = 3% at halfway), so compare it with `all-three`, not with the baseline.

**Numbers (source `results/grid10-robustness/INDEX.md`, `results/all-three/INDEX.md`, their `summary.json`; `reports/{grid10-robustness,all-three}/data.json`).**

| γ | Disc. gap 20% → 10% | Mean learning gap 20% → 10% | Exactly best 20% → 10% | Total gap paired change (SE) | Primary rule 20% → 10% |
|---|---|---|---|---|---|
| 5 | 5.7 → 1.3 | 7.9 → 11.7 | 5 → 0 of 10 | −0.6 (3.7) | 10 → 0 of 10 |
| 3 | 2.5 → 2.5 | 4.0 → 22.0 | 8 → 4 | +18.0 (7.5) | 10 → 0 |
| 8 | 6.1 → 6.1 | 1.1 → 5.8 | 9 → 1 | +4.6 (1.6) | 10 → 0 |

**Run takeaways (`reports/registry.json`, `grid10-robustness` notes).**
- The finer grid lowers the discretization gap only at γ = 5; at γ = 3 and 8 the best 10% portfolio is the best 20% portfolio.
- At γ = 5 the 4.4 bps/yr the finer grid offers does not show up in what the agent learns (paired change −0.6, SE 3.7).
- The grid's best portfolio was tried 81 to 4,114 times per seed at γ = 5 (median 116), close to the 85 random tries each portfolio gets from ε. Each seed's final portfolio was tried 19M to 22M times.
- The greedy portfolio changed a median of 188k, 185k and 155k times per seed at γ = 5, 3 and 8, against about 2,500 to 3,000 in `all-three`.
- Q-fit on the 10% grid: median Spearman 0.543, 0.406, 0.725 and top-10 recall 10% at γ = 5, 3, 8 (`reports/grid10-robustness/data.json: summary.q_spearman_all`, `.q_top10_recall_all`).
- Registered directions (2026-10-27) held for the learning gap, the discretization gap and convergence. The expected exactly-best share near zero held at γ = 5 and 8 and failed at γ = 3 (4 of 10).

**Figures.** F6 (convergence by resolution, three γ), F27 (per-seed greedy loss on the 10% grid, which shows the late switching).

**Caveats.** On the 10% grid no seed of 30 meets the primary convergence rule; each portfolio gets about 85 random tries. Weights for those seeds are final greedy portfolios, not converged ones. Near-optimal-for-good is still met by 10, 5 and 10 of 10 seeds. Post hoc (registered 2026-10-27). The run did not test a larger budget on the 10% grid, so it cannot say how much more training would close the gap. The `grid10-robustness` HTML report does not line the run up against `all-three` (its comparison section matches on γ and grid); the paired numbers above come from the `comparison_g{3,5,8}_grid20_vs_grid10/` folders.

### 6.2 Economic and statistical assumptions (required check 2, and further runs)

**Claims.**
- Risk aversion γ = 3 and 8 (pre-registered, part of `baseline-v1`): Table 6 rows. The optimum moves from equity and gold toward bonds as γ rises; the agent follows.
- Post hoc tabular variants (registered 2026-10-27, run 10-28 and 10-29): longer episode, more episodes, slower ε decay, and all three.

**Numbers (mean learning gap; change vs baseline with SE; change and SE computed in `docs/reviews/audit_2026-11-20_notes/10_reports_results_inventory.md:60-63` from each run's `per_seed.csv` as the SE of the seed-by-seed difference, an approximation because these runs are not paired draw for draw (the unpaired SEs are slightly different and change no conclusion below); means cross-checked against `reports/<run>/data.json: summary.learning_gap.mean`).**

| Run | Change | γ = 5 | γ = 3 | γ = 8 |
|---|---|---|---|---|
| baseline | | 17.6 | 30.6 | 13.0 |
| `b63` | B = 63 | 5.1 (−12.5, SE 6.2) | 10.0 (−20.6, 10.4) | 9.1 (−3.9, 6.6) |
| `n26m` | 26M episodes | 8.3 (−9.3, 5.3) | 16.2 (−14.3, 14.7) | 5.0 (−8.0, 7.2) |
| `eps-decay` | ε = 3% at halfway | 16.1 (−1.5, 7.2) | 4.0 (−26.6, 11.1) | 11.6 (−1.4, 9.2) |
| `all-three` | all of the above | 7.9 (−9.7, 5.9) | 4.0 (−26.6, 11.7) | 1.1 (−11.9, 5.4) |

**Per-run takeaways (each from that run's notes in `reports/registry.json` and its `reports/<run>/data.json`).**

| Run | Headline | Second fact | Caveat to print |
|---|---|---|---|
| `b63` | γ = 5: median gap 14.1 → 0.0, exactly best 3 → 6 of 10. Worst seed improves at every γ: 47.3 → 15.9, 81.1 → 20.0, 53.1 → 29.5 | Primary rule met earlier in episodes (γ = 5 median 9.25M → 6.90M) and later in simulated days (185M → 435M), since an episode is 63 days | Three times the simulated days per episode, so not a like-for-like data budget. Matches the registered direction on the mean and worst seed |
| `n26m` | Mean gap falls at every γ (17.6 → 8.3, 30.6 → 16.2, 13.0 → 5.0); seeds above 25 bps/yr: 0, 1, 1 against 3, 4, 3 | Medians move both ways: 14.1 → 10.5, 10.0 → 20.0, 5.7 → 0.0. Exactly best 3, 4, 7 of 10 against 3, 5, 5 | Registered direction partly met. Doubling the budget gives about 2,200 random tries per portfolio against the 0.5M to 1.1M needed to separate the top two |
| `eps-decay` | γ = 5 and 8 about unchanged (16.1 and 11.6); γ = 3 improves, 8 of 10 exactly best against 5, mean 30.6 → 4.0 | Median training regret rises: 65.5 → 77.3 (γ = 5), 77.0 → 99.2 (γ = 8). Primary rule met by 6 of 10 at γ = 5 and 8, against 9 | 8 of 10 against 5 of 10 on 10 seeds is not a significant difference; report the γ = 3 result as a possible effect |
| `all-three` | Median gap 14.1 → 6.2, 10.0 → 0.0, 5.7 → 0.0; exactly best 5, 8, 9 of 10 against 3, 5, 5 | All 30 seeds meet the primary rule. Q-fit improves: median Spearman 0.971, 0.946, 0.988 against 0.821, 0.710, 0.920 | Three settings changed at once, so it cannot say which did the work. K doubled, so convergence shares are not comparable with the baseline's |

**Figures.** No existing chart puts the four variants side by side: each of these reports compares its run with `baseline-v1` only (F26). Build N4 (bar chart of mean learning gap by run and γ) from the table above, or print the table.

**Caveats.** Post hoc: chosen after seeing the baseline. Runs with a different B or ε consume the random streams differently, so these differences share a starting seed but are not paired draw for draw. Most single differences are within two standard errors of zero. The ones beyond two standard errors: `b63` at γ = 5 (−12.5, SE 6.2, just over 2), `eps-decay` at γ = 3 (−26.6, 11.1), `all-three` at γ = 3 (−26.6, 11.7) and γ = 8 (−11.9, 5.4). `b63` uses three times the simulated days per episode, so it is not a like-for-like budget. B = 252 was not run.

**Open decision.** Which of these get full treatment (audit M8). Recommendation: the first table plus two sentences each.

### 6.3 Research challenges and limitations (the six required topics)

No written text exists for any of these. Facts to build on:

| Topic | Facts available | Source |
|---|---|---|
| Estimation error in μ and Σ | μ, Σ estimated once from 5,152 days and treated as truth; the agent's result is conditional on them. The standard error of each annualized mean is 1.4 to 6.4 percentage points (SPY 10.67 ± 4.27), larger than most differences between assets. No alternative window or shrinkage run was done | `DATA_CLEANING.md`; `reports/full_run_2026-10-22/data.json: calibration.se_mean_annual_pct`; say "not done" |
| Conditioning and numerical stability of Σ | Condition number 268.07, smallest eigenvalue 3.10e-6; Cholesky is stable; the budget-only closed form's large short positions show sensitivity to Σ⁻¹ | `CLEANING_LOG.json`; §3 |
| Curse of dimensionality | 495 → 1,287 → 43,758 → 3,108,105 portfolios; random tries per portfolio fall from about 1,097 (20%) to 85 (10%, with double the budget); 0 of 30 seeds converge on the 10% grid. Splines did not rescue the large grid (§6.4). The first transaction-cost design had 1.66M (holding, target) pairs and learned nothing (§6.5) | §6.1, §6.4, §6.5 |
| Discretization error and benchmark leakage | Gap 11.74 / 5.74 / 1.29 at γ = 5; grid built without benchmark input; tests enforce import allowlists; disclosed human uses of the benchmark (episode budget; TC design) | §3, §4, §6.5 |
| Reward and objective consistency | E[reward] = U(w) with ddof = 1; headline gaps on exact moments. In the TC extension the reward is net of the trading cost divided by the block length, and the oracle maximizes the same discounted objective | §4; `test_rewards.py`; `TC_DRIFT_DESIGN.md` §3 |
| Stationary simulated environment | iid Gaussian, fixed moments, no regimes, no fat tails (LQD kurtosis 58.9 in the data); 260M simulated days is about a million years of daily data; 1/N averaging would fail if moments drifted | §2 |

### 6.4 Extension: function approximation (post hoc)

**Claims.** Replace the table with Q̂(a) = θ·φ(w_a): quadratic features (45 columns) or splines (909 columns, rank 369 on the 20% grid). Constant step size, tuned on a separate master seed (20261103) at γ = 5 only. The like-for-like tabular comparison is `tabular-const` (same constant-α rule), since 1/N is unavailable to FA.

**Numbers (mean learning gap; source `results/{tabular-const,fa-quadratic,fa-spline}/INDEX.md`, `results/fa-spline-grid10-eps{3,10}_2026-11-11/`, `docs/methods/EXPERIMENTS.md` §8.7, §8.8; cross-checked against `reports/<run>/data.json: summary.learning_gap.mean`).**

| Run | γ = 5 | γ = 3 | γ = 8 |
|---|---|---|---|
| baseline (1/N table) | 17.6 | 30.6 | 13.0 |
| `tabular-const` | 43.5 | 107.2 | 43.3 |
| `fa-quadratic` | 22.6 | 8.0 | 58.7 |
| `fa-spline` | 15.7 | 45.4 | 54.6 |
| 10% grid, table (`grid10-robustness`) | 11.7 | 22.0 | 5.8 |
| 10% grid, `fa-spline-grid10-eps3` | 78.8 | 140.6 | 59.6 |
| 10% grid, `fa-spline-grid10-eps10` | 81.4 | 155.6 | 60.2 |

Paired differences (SE): `fa-spline` − `tabular-const` −27.8 (11.9), −61.8 (24.2), +11.4 (5.1). `fa-spline` − baseline −1.8 (9.5), +14.8 (17.3), +41.6 (6.0). FA 10% − table 10%: +67.1 (9.1), +118.6 (12.2), +53.8 (9.8).

**Second table: how well each agent ranks the whole grid (median over seeds, γ = 5 / 3 / 8; source `reports/<run>/data.json: summary.q_spearman_all`, `.q_top10_recall_all`, `.exact_best`, `.primary.n`; post hoc measures).**

| Run | Spearman | Top-10 recall | Exactly best (of 10) | Primary rule met (of 10) |
|---|---|---|---|---|
| baseline | 0.821 / 0.710 / 0.920 | 35% / 20% / 20% | 3 / 5 / 5 | 9 / 8 / 9 |
| `tabular-const` | 0.813 / 0.703 / 0.917 | 25% / 15% / 20% | 0 / 2 / 0 | 10 / 8 / 10 |
| `fa-quadratic` | 0.915 / 0.854 / 0.951 | 55% / 80% / 20% | 0 / 6 / 0 | 2 / 0 / 9 |
| `fa-spline` | 0.950 / 0.945 / 0.957 | 45% / 65% / 20% | 5 / 3 / 0 | 10 / 10 / 9 |
| 10% grid, table | 0.543 / 0.406 / 0.725 | 10% / 10% / 10% | 0 / 4 / 1 | 0 / 0 / 0 |
| 10% grid, FA ε 3% | 0.976 / 0.963 / 0.983 | 0% / 0% / 10% | 0 / 0 / 0 | 0 / 0 / 0 |
| 10% grid, FA ε 10% | 0.975 / 0.964 / 0.985 | 0% / 0% / 0% | 0 / 0 / 0 | 0 / 0 / 0 |

**Per-run takeaways (`reports/registry.json` notes).**
- `tabular-const`: a constant α = 3.2e-4 in place of 1/N raises the mean gap at every γ (paired +25.9, SE 11.5; +76.6, 24.6; +30.2, 8.0). Portfolios tried only during exploration end near 0.3 of their true value, which suggests (inferred, not tested; registry `tabular-const`) that rivals rarely overtake the greedy portfolio once ε fades. At γ = 5 all 10 seeds held their final portfolio for the last 1.3M episodes, none on the best portfolio, and they ended on 9 different portfolios, so under a constant α the primary rule measures stability and says little about accuracy. Registered expectations: larger gap met; fewer seeds converging not met.
- `fa-quadratic` (the correctly specified reference model, told the reward is quadratic): against `tabular-const` the gap falls at γ = 5 and 3 (−20.9, SE 9.8; −99.2, 23.7) and rises at γ = 8 (+15.4, 6.5). Each γ's 10 seeds end on only 2 portfolios. At γ = 8 all 10 seeds miss the best portfolio by 55.7 or 70.8.
- `fa-spline` (the discovery model): median gap 5.3 at γ = 5 against 42.0 for `tabular-const` and 14.1 for the baseline, with 5 of 10 exactly best. At γ = 8 it is 51.6, the same as `tabular-const` (51.5) and far behind the baseline (5.7). Registered expectation (lower Spearman and recall than `fa-quadratic`) not met (recall was lower at γ = 5 and 3 and tied at γ = 8; the registry grades it "Not met").
- `fa-spline-grid10-eps3`: FA ranks the 43,758 portfolios far better than the table and picks worse at every γ; worse on 10, 10 and 9 of 10 seeds. Every seed under-weights SPY (picks hold 20% to 40% against 50%, 60%, 40% in the best portfolio). Registered: higher Spearman met; at least 6 of 10 seeds converging failed (0 of 10).
- `fa-spline-grid10-eps10`: 1.5 times the random episodes did not shrink the gap at any γ (paired against ε 3%: +2.6, SE 11.8; +15.0, 14.8; +0.6, 10.1) and raised training regret by about 25 to 28 bps/yr per episode. The registered smaller gap at γ = 5 is not shown.

**Figures.** F23 (learning gap per seed for baseline, `tabular-const`, `fa-quadratic`, `fa-spline`), F24 (the same on the 10% grid for the table and the two FA runs), F28. PNGs of each experiment's weights and curves are in its `evaluation/` folder.

**Caveats and disclosures.**
- Post hoc (registered 2026-11-05 and 2026-11-11). Step size tuned on separate seeds, at γ = 5 only; the 10% grid step (2.778e-5) was noise-matched, not re-tuned.
- FA beats the like-for-like table at γ = 5 and 3 and loses at γ = 8; on the 10% grid it loses to the table at every γ. This contradicts the motivation for the 10% grid run (`EXPERIMENTS.md` §8.8); no direction was registered for the gap. Report it as a negative result.
- `fa-spline` and `tabular-const` use separately tuned learning rates, so that comparison mixes features and learning rate. Against `grid10-robustness` the FA runs change the learning rule and the representation at once.
- The ΔQ rule says nothing under a constant α.
- Explanation for γ = 8 ("directions that have not finished learning": 36 of 45 quadratic and 351 of 369 spline directions) rests on calculations with no committed script (audit H4, still open on 2026-11-23: no surrogate, mean-field or shrinkage script is tracked). Word it as a hypothesis unless William commits them.
- William's mean-field calculation predicted gaps of about 33.6, 0 and 1.3 bps/yr on the 10% grid at ε = 3% and 4.4, 0 and 1.3 at ε = 10%; the realized gaps are 78.8, 140.6, 59.6 and 81.4, 155.6, 60.2. The calculation does not describe this agent's errors (`reports/registry.json`, both `fa-spline-grid10` caveats).
- On the 10% grid FA ranks portfolios well overall (median Spearman 0.976) and picks badly at the top (top-10 recall 0.00).

### 6.5 Extension: transaction costs (post hoc)

**First design (negative result).** One paragraph, pending decision D-G. State = previous grid target, quadratic cost, 1.66M (holding, target) pairs, 57.1% ever tried. At zero cost the learned policy scores 56.6 against a grid oracle of 524.8 and a random-portfolio average of 56.7. Learned policies trade in 94.8 to 100 of 100 blocks. Source `docs/methods/TRANSACTION_COSTS.md` ("Outcome" table), `results/tc-quadratic-cost/*.json`, `reports/tc-quadratic-cost/report.html`. The state space was too large for the data, and with no drift there was nothing to rebalance, which motivated the redesign. If D-G goes to "paragraph plus table", the four-row table is:

| Cost per 20-point transfer | Grid oracle | Ignore costs | Myopic | Learned (median) | Learned gap to oracle, median (range) |
|---|---|---|---|---|---|
| 0 bp | 524.8 | 524.8 | 524.8 | 56.6 | 468.2 (436.4 to 485.2) |
| 1 bp | 520.9 | 520.8 | 515.9 | −16.8 | 537.7 (482.4 to 589.8) |
| 2 bp | 517.1 | 516.8 | 512.6 | −90.9 | 608.0 (572.5 to 658.7) |
| 5 bp | 507.6 | 504.9 | 479.2 | −296.8 | 804.3 (744.9 to 844.3) |

Do not reuse the text of `reports/tc-quadratic-cost/report.html`: two of its sentences ("Differences between the four cost levels isolate the configured cost", "trading behavior may be concentrated near the starting allocation") do not hold for these policies (`TRANSACTION_COSTS.md`, "What the report says that should change").

**Redesign with drift: design.**
- Three ETFs (SPY, IEF, GLD), γ = 5, 63-day blocks during which holdings drift with returns. Action: one of 1,326 targets on a 2% grid, or HOLD. Cost = κ‖trade‖₁ + (λ/2)‖trade‖₂² as a fraction of wealth. Discount 0.95 per block (explain: about 0.81 per year, half the weight within 3.4 years).
- Cost levels from own liquidity data (half-spread 0.5 bp of a 1 bp quoted spread; square-root impact): zero; $100M (κ 1.63 bp, λ 4.4e-3); $1B headline (3.90 bp, 1.43e-2); $10B (11.04 bp, 4.57e-2). Source `src/tc_drift/calibration.py`, `docs/methods/TC_DRIFT_DESIGN.md` §8; the κ and λ used are in each `results/tc-drift/runs/<level>/summary.json` (`kappa`, `lam`). Undoing one block's drift costs 0.115, 0.302 and 0.892 bp at the three levels, against about 0.33 bp per block (1.33 bps/yr) of utility lost by holding the drifted portfolio for that block (design §8), so $1B sits near break-even.
- Numerical oracle (value iteration on a 1% lattice, 2,000 paths). In its own model its lead over always rebalancing is 0.008 / 0.214 / 1.319 / 6.939 bps/yr at zero / $100M / $1B / $10B, with hold fraction 0.156 / 0.272 / 0.308 / 0.443. Source `results/tc-drift/oracle/summary.json` (`settings.<level>.lead_over_always_rebalance_bps`, `.oracle.hold_fraction`).
- Agent: linear Q with 87 spline columns plus two cost features whose coefficients it learns from net rewards; recursive least squares on the cost block after a burn-in (block 301,030); reported policy uses θ averaged over the final 10% of training. It never sees μ, Σ, κ, λ or the oracle.
- Tuning: α = 3e-4 chosen over 1e-3 on tuning seed 20261112 at $1B, by realized net reward (505.4 vs 496.1, paired difference 9.31, SE 6.32, 3 seeds). Source `results/tc-drift/tuning/summary.json` (`choice`).
- Reference policies, all scored on the same paths: the oracle; always rebalance to the best no-cost target (0.46, 0.14, 0.40); trade to it once, then hold; myopic (the oracle solved with ρ = 0 when choosing, scored at 0.95).

**The oracle's lead over always rebalancing has three committed values. Say which one is printed.**

| Source | Sample | zero | $100M | $1B | $10B |
|---|---|---|---|---|---|
| `results/tc-drift/oracle/summary.json` | In the oracle's own model (in-sample; V's level is biased upward, the lead need not be) | 0.008 | 0.214 | 1.319 | 6.939 |
| `results/tc-drift/checks/oracle_replay_2026-11-17.json` (`levels.<level>.lead_replay`) | Replay, m = 1,000 paths, tuning-seed stream | −0.03 (SE 0.08) | −0.17 (0.43) | 1.30 (0.70) | 7.10 (1.26) |
| `results/tc-drift/runs/<level>/summary.json` (`oracle_minus_always_rebalance`) | Replay, m = 4,000 paths, the final evaluation set | 0.04 (0.04) | 0.42 (0.22) | 1.78 (0.36) | 7.79 (0.64) |

The three agree within sampling error. Print the third row beside the agent's results, since those are the paths the agent is scored on. The oracle's hold fraction differs the same way: 0.156 / 0.272 / 0.308 / 0.443 in-model and 0.153 / 0.283 / 0.320 / 0.453 on the final paths (`references.oracle.hold_fraction`).

**Economic point.** The whole prize is small. The oracle beats "always rebalance" by 1.8 bps/yr at $1B and 7.8 at $10B. The question for the agent is whether it learns when not to trade.

#### Final results (filled 2026-11-23)

40 runs, 4 cost levels × 10 seeds, 4M blocks each, master seed 20261016, run 2026-11-20 at commit `1787f0d` with a clean tree (`provenance.commit`, `provenance.src_dirty: false` in every `summary.json`). Scored on m = 4,000 shared starts and paths, H = 90 blocks. Source for every cell: `results/tc-drift/runs/<level>/summary.json` and `per_seed.csv`, levels `zero`, `aum_100m`, `aum_1b`, `aum_10b`; the table is also in `results/tc-drift/runs/README.md` and `reports/tc-drift/report.html`. Post hoc relative to `baseline-v1`.

**Table TC-1. Values of each policy (bps/yr, expected discounted mean-variance utility).** References: `references.<policy>.mean_value_bps`. Agent: mean over the 10 seeds of `per_seed[].agent.mean_value_bps`, computed for this skeleton on 2026-11-23; it equals the oracle's value plus the agent − oracle mean.

| Policy | zero | $100M | $1B | $10B |
|---|---|---|---|---|
| Oracle | 531.52 | 530.16 | 528.09 | 523.22 |
| Myopic | 531.52 | 530.18 | 527.96 | 520.95 |
| Always rebalance | 531.48 | 529.75 | 526.31 | 515.44 |
| Agent (mean of 10 seeds) | 518.37 | 517.07 | 518.11 | 520.45 |
| Trade once, then hold | 508.49 | 507.22 | 504.56 | 496.11 |

**Table TC-2. The agent against the references (mean over 10 seeds; `across_seeds.<field>.mean` unless noted).**

| | zero | $100M | $1B | $10B |
|---|---|---|---|---|
| Agent − oracle (across-seed SD) | −13.14 (4.58) | −13.09 (5.52) | −9.99 (4.32) | −2.77 (1.70) |
| Range across seeds (`.min`, `.max`) | −20.3 to −5.8 | −20.6 to −4.8 | −19.6 to −2.6 | −5.9 to −0.5 |
| Path SE of the seed-averaged gap (`results/tc-drift/runs/README.md`) | ±1.03 | ±1.02 | ±0.83 | ±0.35 |
| Agent − always rebalance | −13.11 | −12.67 | −8.21 | +5.01 |
| Agent − myopic | −13.14 | −13.10 | −9.86 | −0.50 |
| Agent − trade once, then hold, lowest seed | 2.68 | 2.30 | 3.93 | 21.19 |
| Split, when it trades: agent − own target | 0.00 | 0.08 | 0.83 | 6.16 |
| Split, where it heads: own target − always rebalance | −13.11 | −12.75 | −9.04 | −1.14 |
| Favourite target's utility loss (uses μ, Σ; diagnostic) | 12.91 | 12.95 | 9.75 | 1.79 |
| Agent hold fraction (seed range) | 0.35 (0.03 to 0.73) | 0.25 (0.00 to 0.71) | 0.18 (0.01 to 0.70) | 0.17 (0.06 to 0.21) |
| Oracle hold fraction | 0.15 | 0.28 | 0.32 | 0.45 |
| Agent turnover per block (oracle; always rebalance 2.98%) | 2.06% (2.88%) | 2.28% (2.20%) | 2.36% (2.08%) | 2.11% (1.55%) |
| Converged, operative rule of 2026-11-20 (of 10) | 5 | 7 | 7 | 4 |
| Converged, original and exact-action rules | 0 | 0 | 0 | 0 |
| Tail average − last iterate | −0.35 | 0.42 | 0.20 | 0.13 |
| D10: mean paired SE vs target | 1.10 vs 0.01, n/a | 1.10 vs 0.14, n/a | 0.94 vs 0.59, missed | 0.43 vs 2.60, met |

Per seed (`per_seed.csv`): agent − oracle is negative on 10 of 10 seeds at every level. Agent − always rebalance is negative on 10 of 10 seeds at zero, $100M and $1B, and positive on 10 of 10 at $10B (1.86 to 7.26). At $10B the agent beats myopic on 5 of 10 seeds. Seeds meeting the operative convergence rule first hold it from episode 17,355 to 39,160 of about 44,400 (`converged_from_episode`).

**What the results say (claims for the text).**
- The agent loses to the oracle at every cost level and on every seed. Say this first.
- At zero cost, $100M and $1B nearly all of the gap is where the agent heads. Its favourite target is 10 to 13 bps/yr of utility worse than the best target, while its timing adds 0.0 to 0.8. Every seed's favourite target at those levels holds 18% to 32% IEF against 14% in the best target (`favourite_target` in `per_seed.csv`); at zero cost every seed holds 22% to 30%.
- At those three levels the agent also loses to always rebalancing, a rule that needs no learning once the target is known. The comparison is unequal in one respect that must be stated: always rebalance is given the best target, which is computed from μ and Σ, and the agent has to find its target from rewards.
- At $10B the agent's timing is worth +6.2 bps/yr against rebalancing to its own target, its target split (own target − always rebalance) is −1.1, and it beats always rebalancing by 5.0. It is still 2.8 behind the oracle and 0.5 behind myopic.
- The agent does not get there by holding more: at $10B it holds in 17% of blocks against the oracle's 45%, and its turnover (2.11% a block) is above the oracle's (1.55%) and below always rebalancing's (2.98%). The timing value at $10B therefore comes from trading less per trade or choosing which drift to undo, and no committed analysis says which. Print it as an open point.
- The quadratic cost coefficient is learned and the linear one is not. c2 is within a factor of 2 of the truth on 9 of 10 seeds at $100M and 10 of 10 at $1B and $10B. c1 is zero (clamped) or below 0.2 of the truth on all 10 seeds at $10B, at or below 0.15 on 6 of 10 at $1B, and between 2.8 and 14.9 times too high on five seeds at $100M (`c1_over_true`, `c2_over_true`).
- Seeds differ by more than the path noise. The across-seed SD of the gap is 4.3 to 5.5 at the three lower levels against a path SE near 1, so quote the SD.
- The pilot understated the target error: 2 to 6 bps/yr on three tuning seeds at $1B against 9.75 on the final ten (`results/tc-drift/runs/README.md`).
- Why the target error is much smaller at $10B (target split −1.1 against −13.1, −12.8 and −9.0; favourite-target utility loss 1.8 against 12.9, 13.0 and 9.8) has not been investigated (`results/tc-drift/runs/README.md`, "Not yet done"). Do not offer a cause in the report.

**Table TC-3. Registered expectations (registered 2026-11-20 in `TC_DRIFT_DESIGN.md` §7 before any final run; verdicts from `results/tc-drift/runs/README.md`, clause by clause in `reports/tc-drift/data.json: expectations`). 8 met, 4 partly met, 0 not met.**

| # | Registered expectation | Field | Result | Verdict |
|---|---|---|---|---|
| E1 | Agent − oracle mean between −1 and −8 at each level; negative on at least 8 of 10 seeds | `agent_minus_oracle_mean`; per seed in `per_seed.csv` | −13.1, −13.1, −10.0, −2.8; negative on 10 of 10 at every level. Mean outside the band at three levels, on the worse side | Partly met |
| E2 | Favourite-target error is most of the gap at zero, $100M, $1B | the two `split_*` means | Own target − always rebalance −13.1, −12.8, −9.0 against agent − own target 0.0, 0.1, 0.8 | Met |
| E3 | Agent loses to always-rebalance at $100M and $1B; relatively better at $10B than at $1B | `agent_minus_always_rebalance_mean` | −12.7, −8.2; +5.0 at $10B | Met |
| E4 | Agent − own target rises from $1B to $10B and is positive at $10B; within ±1 at zero and $100M | `split_agent_minus_own_target_mean` | 0.8 → 6.2; 0.0 and 0.1 | Met |
| E5 | Agent beats trade-once-then-hold at every level and seed | `agent_minus_trade_once_then_hold` (per seed) | Lowest seed 2.68, 2.30, 3.93, 21.19 | Met |
| E6 | Agent loses to myopic at every level | `agent_minus_myopic_mean` | −13.1, −13.1, −9.9, −0.5 | Met |
| E7 | Hold fraction varies across seeds; mean higher at $10B than zero; at zero cost the agent holds more than the oracle | `agent_hold_fraction` | Spread 0.01 to 0.70 at $1B; at zero 0.35 against the oracle's 0.15; mean at $10B (0.17) is lower than at zero (0.35), the opposite of the prediction | Partly met |
| E8 | c2 within ×2 on at least 7 of 10 seeds at $1B and $10B; c1 off by more than ×2 on at least 3 of 10 at each cost level | `c1_over_true`, `c2_over_true` | c2: 10 and 10 of 10. c1 off on 10, 8 and 10 of 10 at $100M, $1B, $10B | Met |
| E9 | Fewer than 5 of 10 seeds converge per level under the 2026-11-20 rule; none under the original and exact-action rules on the last iterate | `d12_converged_count` | Operative rule 5, 7, 7, 4; original and exact-action 0 at every level. "Fewer than 5" holds only at $10B | Partly met |
| E10 | Tail-averaged and last-iterate policies differ by under 1 bps/yr in the mean | `agent_minus_last_iterate_mean` | −0.35, 0.42, 0.20, 0.13 | Met |
| E11 | Precision target missed at $1B (SE 0.5 to 0.8 against about 0.43), met at $10B, not applicable at zero and $100M | `d10` | Missed at $1B, met at $10B, n/a at the other two, as predicted. The $1B SE is 0.94, outside 0.5 to 0.8, and the target is 0.59 (a third of the 1.78 lead) | Partly met |
| E12 | Across-seed SD of agent − oracle at $1B at least 0.5 | `sd_across_seeds` | 4.32 | Met |

Under the design's "what would change the story" (§7): E1 failed away from zero, so the agent learned less than the pilot suggested. At $10B E4 holds while the favourite-target error is small (−1.1), so there the agent learned both when to trade and roughly where to head.

**Figures.** F29 (where the gap comes from, by cost level) is the one chart to print. F31 (favourite targets against the best target) and F32 (hold fraction, agent against oracle) support it. F30 (every seed against the references) and F33 (learned costs over training) are for the appendix or backup slides.

**Disclosures for the TC extension (print all).**
- Post hoc relative to `baseline-v1`. The 40 runs came from commit `1787f0d`, clean tree.
- The three-ETF universe is the support of the nine-asset optimum, so benchmark information was used in the design.
- The cost levels were chosen with the oracle in view (`TC_DRIFT_DESIGN.md` §8 and §14 risk 5), although κ and λ come from our own liquidity data.
- The cost features build in the cost family's shape; the agent learns two coefficients, clamped at zero from below.
- Learning rule, tail averaging and the convergence rule were chosen with pilot benchmark diagnostics in view. The convergence rule changed three times before any final run (frozen 2026-11-20); all four verdicts are reported.
- The convergence check reads the policy at checkpoints only, so a switch that reverts between two checkpoints is not seen; the baseline's rule tracked every change (`TC_DRIFT_DESIGN.md` §7).
- E1 to E12 lean on the $1B tuning pilot (same agent, different seeds).
- Tuned at $1B only. A first set of six pilot runs was discarded.
- Oracle check 3 was restated after the fact; check 5 is a two-standard-error bound at the cost levels.
- One cost for every asset pair undercharges the typical SPY-vs-GLD drift trade by about a fifth.
- Oracle summary and tuning records were produced from uncommitted code (`src_dirty: true`; audit M1). The final agent runs were not.
- Drift is applied on excess returns (effect bounded at 7.1e-6).
- The always-rebalance and trade-once references use the best no-cost target, which is computed from μ and Σ.
- The precision target (D10) is missed at $1B: the paired SE of one seed's gap is 0.94 against a target of 0.59.

**Slide takeaways.**
- Two required checks done: grid resolution and risk aversion. Longer episodes and more data help; the finest grid does not.
- Function approximation beats the like-for-like table on the small grid at two of three γ and loses to the table on the large grid at every γ. It is reported as a negative result.
- Transaction costs: the agent trails the oracle by 13, 13, 10 and 3 bps/yr as costs rise. Its target is the problem at low cost; at $10B its timing is worth 6 bps/yr and it beats always rebalancing by 5.

---

## 7. Conclusion

**Purpose.** Answer the question in the assignment's words: what the agent learned, how close it came to the best attainable action, what prevented exact recovery.

**Claims.**
- Learned: SPY and GLD on every seed and a bond ETF on most, roughly the right mix, from rewards alone.
- How close: median 14.1 bps/yr from the best attainable portfolio and 19.8 from the continuous optimum at γ = 5, against 310.9 for equal weight; exact recovery in 3 of 10 seeds (5 of 10 at γ = 3 and 8).
- What prevented it: top grid portfolios about 10 bps/yr apart against per-episode noise near 3,900.
- Trade-off: a finer grid removes discretization error and adds learning error.
- Extensions, both post hoc: splines rank a large action set well and still pick badly at the top; with trading costs the agent learns the quadratic cost and, at the highest cost level, when to trade, and it misses the target portfolio by 9 to 13 bps/yr at the lower levels.
- Limits: known stationary Gaussian world; μ and Σ taken as truth; a data budget no investor has.

**Caveat.** Keep the pre-registered results (baseline, 25% grid, γ = 3 and 8) as the headline; post hoc runs support, they do not replace.

**Slide takeaway.** Model-free learning recovers a near-optimal portfolio in a clean simulated world, at a data cost that makes the benchmark's value clear.

---

## 8. Figure inventory

**TL;DR / In plain terms.** The repo already holds 37 reusable figures in three forms. PNG files sit next to the results. The transaction-cost pages carry finished vector charts inside the HTML. The 12 run-report pages draw their charts in the browser when the page is opened, so those have to be captured from an open page. Seven more figures the report wants do not exist and are listed at the end.

**Key concepts to understand**
- **Inline SVG.** The chart's drawing instructions are written into the HTML file. Copy the `<svg>…</svg>` block into a `.svg` file, or rerun the function that made it.
- **Drawn in the browser.** `reports/<run_id>/report.html` (11 pages) and `reports/full_run_2026-10-22/report.html` contain no chart markup (0 `<svg>` tags in the saved files). One script builds each chart as an SVG node inside `<div class="chart" id="…">` from the page's embedded data. To extract one: open the page, pick the experiment or family with the page's toggle, then save the node from the browser's inspector or take a screenshot. The numbers behind every chart are in the sibling `data.json`.
- **Family toggle.** Section 5 of a pipeline report shows one (γ, grid) family at a time. State which family a captured chart shows.

**Key takeaways**
- For the main results, the PNGs (F1 to F6) cover every chart Part V asks for.
- For slides, the HTML charts F7, F8, F12, F16, F23, F24 and the TC charts F29, F31, F32 carry the story better than the PNGs.
- Nothing was exported or changed for this inventory.

**How each source is produced**

| Source | Form | Rebuild |
|---|---|---|
| `results/**/evaluation/*.png`, `results/**/convergence_by_resolution.png` | PNG, committed | `python -m src.evaluate` (`src/evaluate.py`: `_plot_weights`, `_plot_learning`, `_plot_regret`, `_plot_loss_by_seed`, `_plot_comparison`) |
| `reports/full_run_2026-10-22/report.html` | Drawn in the browser from `data.json` | `python reports/full_run_2026-10-22/build_data.py`, then `build_report.py` (inlines `data.json` into `template.html`) |
| `reports/<run_id>/report.html`, 11 runs | Drawn in the browser from embedded data | `python -m src.reporting.build <run_id>` (`src/reporting/template.html`) |
| `reports/tc-drift/report.html` | 5 inline matplotlib SVGs | `python -m src.tc_drift.report` (`src/tc_drift/report.py`: `fig_split`, `fig_vs_references`, `fig_targets`, `fig_hold`, `fig_costs`) |
| `reports/tc-quadratic-cost/report.html` | 1 inline matplotlib SVG | `python -m src.tc_sweep_report` |

### 8.1 PNG files (main results)

| ID | File | Shows | Takeaway | Report | Slide |
|---|---|---|---|---|---|
| F1 | `results/baseline_g5_grid20/evaluation/weights_grouped_bar.png` | Nine asset weights for w*, best grid, equal weight and the median seed | The learned portfolio holds the same three assets as the optimum | §5.3 | 6 |
| F2 | `results/baseline_g5_grid20/evaluation/ce_loss_by_seed.png` | Each seed's CE loss, with the 25 bps near-optimal line | 3 of 10 seeds are exactly best; the worst is 47.3 behind the best grid portfolio | §5.4 | 6 (backup) |
| F3 | `results/baseline_g5_grid20/evaluation/learning_curve.png` | Moving-average reward over training against the two benchmark levels | Reward climbs as exploration fades and ends just under the best grid portfolio | §5.5 | 5 |
| F4 | `results/baseline_g5_grid20/evaluation/cumulative_regret.png` | Cumulative regret against the best grid portfolio | Regret is front-loaded in the exploring half | §5.5 | backup |
| F5 | `results/comparison_g5_grid20_vs_grid25/convergence_by_resolution.png` | Greedy loss over training, 20% against 25% grid | The coarser grid converges faster and more reliably | §5.5 | 8 |
| F6 | `results/grid10-robustness/comparison_g{3,5,8}_grid20_vs_grid10/convergence_by_resolution.png` (3 files) | The same, 20% against 10% grid at the `all-three` settings | The 10% grid never settles within the budget | §6.1 | 8 |

The same four evaluation PNGs exist for every other experiment (for example `results/robust_g3_grid20/evaluation/`, `results/fa-spline/baseline_g5_grid20_FAspline/evaluation/`); use them for γ = 3 and 8 or for an appendix. Ignore `results/smoke*`, which are test runs.

### 8.2 `reports/full_run_2026-10-22/report.html` (original baseline page; drawn in the browser)

| ID | Section and figure | Shows | Takeaway | Report | Slide |
|---|---|---|---|---|---|
| F7 | §3 "20% vs. 25%: head to head", Figure 3.1 "Where the shortfall comes from" | Total loss split into discretization and median learning gap, both grids | 25% grid: 11.7 + 2.5. 20% grid: 5.7 + 14.1 | §5.5 | 8 |
| F8 | §3, Figure 3.2 "Every seed's total gap to the continuous optimum" | One dot per seed, both grids, with each grid's floor | Seeds on the floor found the exact best grid portfolio | §5.4 | 6 |
| F9 | §3, Figure 3.3 "How fast each grid closes the learning gap" | Median greedy loss with a band for the middle half of seeds | The 25% grid closes its gap earlier | §5.5 | 8 (alternative to F5) |
| F10 | §3, Figure 3.4 "Learning curve" | Median reward as a CE return, both grids | Includes exploration, so it sits below the best portfolio until ε is small | §5.5 | backup |
| F11 | §3, Figure 3.5 "Average regret so far" | Running average regret, both grids | Early exploration dominates | §5.5 | backup |
| F12 | §3, Figure 3.6 "When each convergence measure is reached" | One dot per seed for each of five measures | "Settled" comes in the first third of training; the pre-registered rule comes later because of brief excursions | §5.2 | 5 |
| F13 | §3, Figure 3.7 and §9, Figure 9.1 "Portfolio weights" | Nine weights, benchmarks and learned | Same content as F1, but "learned" is the average of the ten seeds' final portfolios. Do not print as is (§5.3) | none | none |
| F14 | §4 "The ten seeds on the 20% grid", Figure 4.1 "Each seed's greedy portfolio over time" (§5, Figure 5.1 for the 25% grid) | Ten small panels, one per seed, with settled and rule-met lines | Seeds settle early and then make brief excursions | §5.2 or appendix | backup |
| F15 | §4, Figure 4.4 "Brief excursions after settling" (§5, Figure 5.4) | How bad each temporary portfolio was | Median excursion lasts 3 to 4 episodes; the longest 9,645 on the 20% grid | appendix | none |
| F16 | §6 "Why the agent misses the best portfolio", Figure 6.1 "How often the best portfolio was tried, in seeds that missed it" | Tries of the best portfolio per seed on a log scale, against the 1.1M (20%) and 7.4M (25%) needed | The best portfolio was tried about a thousand times; separating it needs about a million | §5.6 | 7 |
| F17 | §9 "Findings as we would present them", Figure 9.2 "Distribution of the learning gap across seeds" | One dot per seed for all four baseline experiments | The spread across seeds is as large as the gap | §5.4 | backup |

Figures 4.2, 4.3 and 5.2, 5.3 on that page (per-seed learning curves and share of episodes on the final portfolio) duplicate F3 and add little; they are not numbered here.

### 8.3 Pipeline reports `reports/<run_id>/report.html` (11 runs, one template; drawn in the browser)

Every page has the same figures: 3.1 "Where each experiment's shortfall comes from", 3.2 "Every seed's total gap", 4.1 "Each seed's greedy portfolio over time", 4.2 "Learning curve, every seed", 4.3 "Share of episodes on the final portfolio", 4.4 "Brief excursions after settling", 5.1 "Every family across runs" (a table), 5.2 "Median greedy loss over training, by run", 5.3 "Learning gap per seed, by run", 5.4 "Total gap, by run", 5.5 "Learning curve, by run", 5.6 "When each run settles and converges". The rows below pick the instances worth reusing.

| ID | Report, section and figure | Shows | Takeaway | Report | Slide |
|---|---|---|---|---|---|
| F18 | `baseline-v1`, §3 "Results by experiment", Figure 3.1 | Gap decomposition for all four pre-registered experiments | Learning gap exceeds the discretization gap on the 20% grid at every γ | §5.4 | 6 (alternative to F8) |
| F19 | `baseline-v1`, §3, Figure 3.2 | Every seed's total gap, four experiments | Shows γ = 3 and 8 beside the baseline in one chart | §5.4, §6.2 | 9 |
| F20 | `baseline-v1`, §2 "Run settings" (table) | Every setting read from the manifests | Source for the §4 settings table | §4 | 4 |
| F21 | `all-three`, §5 "Comparison with earlier runs", Figure 5.3, family γ = 5 / 20% | Learning gap per seed, baseline against `all-three` | Median 14.1 → 6.2; worst seed 47.3 → 29.4 | §6.2 | 9 |
| F22 | `all-three`, §5, Figure 5.2 (x-axis set to share of budget) | Median greedy loss over training, baseline against `all-three` | Longer episodes and more data settle on a better portfolio | §6.2 | backup |
| F23 | `fa-spline`, §5, Figure 5.3, one capture per γ family | Learning gap per seed for `baseline-v1`, `tabular-const`, `fa-quadratic`, `fa-spline` | FA beats the constant-α table at γ = 5 and 3 and loses at γ = 8 | §6.4 | 10 |
| F24 | `fa-spline-grid10-eps10`, §5, Figure 5.3, family γ = 5 / 10% | Learning gap per seed for `grid10-robustness`, FA ε 3%, FA ε 10% | On the 10% grid FA is 67 to 70 bps/yr worse than the table; more exploration does not help | §6.4 | 10 |
| F25 | `fa-spline-grid10-eps10`, §5, Figure 5.6 | When each run settles and converges on the 10% grid | No FA seed meets the primary rule; settled points sit at the end of training | §6.4 | backup |
| F26 | `b63`, `n26m`, `eps-decay`, each §5, Figures 5.3 and 5.4 | Each single-change run against the baseline | One chart per run only; no page overlays all four variants (see N4) | §6.2 | backup |
| F27 | `grid10-robustness`, §4 "Each experiment, seed by seed", Figures 4.1 and 4.4 | Per-seed greedy loss and excursions on the 10% grid | The greedy portfolio keeps switching (a median of 188k changes per seed at γ = 5) | §6.1 | 8 (backup) |
| F28 | `tabular-const`, §5, Figure 5.3 | Learning gap per seed, 1/N against constant α | A constant step size costs 26 to 77 bps/yr | §6.4, §4 | backup |

`reports/fa-quadratic/report.html` shows the same three-way FA comparison as F23 from the other side; use one of the two.

### 8.4 `reports/tc-drift/report.html` (inline SVG)

| ID | Section heading (function) | Shows | Takeaway | Report | Slide |
|---|---|---|---|---|---|
| F29 | "Where the gap to the oracle comes from" (`fig_split`) | Per cost level: where it heads, when it trades, the oracle's lead, and every seed's agent − oracle | The target error is the gap at low cost; at $10B the timing term is +6.2 | §6.5 | 12 |
| F30 | "Every seed against the references" (`fig_vs_references`) | Three panels: agent − oracle, agent − always rebalance, agent − myopic, one dot per seed | At $10B every seed beats always rebalancing; at lower costs none does | §6.5 or appendix | 12 (backup) |
| F31 | "Where the agent heads, and how often it trades", left chart (`fig_targets`) | Each seed's favourite target (SPY against GLD weight) with the best target starred | Favourite targets sit off the best target at low cost and close to it at $10B | §6.5 | 12 |
| F32 | Same heading, right chart (`fig_hold`) | Share of blocks with no trade, each seed against the oracle | The agent's hold fraction is scattered at low cost and well below the oracle's at $10B | §6.5 | 13 (backup) |
| F33 | "Learned costs" (`fig_costs`) | c1 and c2 over training as a ratio to the truth, per seed and cost level | c2 converges near the truth; c1 is clamped at zero or far too high | §6.5 or appendix | backup |
| F34 | Header KPI row and "Registered expectations (E1 to E12)" (HTML tables) | Four headline numbers; each expectation with its clauses and verdict | 8 met, 4 partly met, 0 not met | §6.5 (Table TC-3) | 13 |
| F35 | "Convergence (D12) and evaluation precision (D10)" (HTML table and per-seed tables) | Converged counts under four rule versions; SE against target; every seed's row | 5, 7, 7, 4 of 10 converge under the operative rule | appendix | none |

### 8.5 `reports/tc-quadratic-cost/report.html` (inline SVG)

| ID | Section heading | Shows | Takeaway | Report | Slide |
|---|---|---|---|---|---|
| F36 | "Performance against baseline-v1" (chart and table) | Expected net CE by cost level: learned, baseline, grid oracle | The learned line sits 468 to 804 bps/yr under the oracle (median seed), starting at zero cost | §6.5 first paragraph, only if D-G chooses to show it | 11 (backup) |
| F37 | "Trading and training behavior" (table) | Turnover, trades, pair coverage 57.1%, stability 0 of 10 | The agent traded in every block and visited 57% of pairs | §6.5 first paragraph | none |

### 8.6 Figures that do not exist yet

| ID | Figure | Data source | Report | Slide |
|---|---|---|---|---|
| N1 | Correlation heatmap of the nine ETFs | `data/processed/correlation_matrix.csv` | §2 | 2 |
| N2 | ε schedule over training with the halfway point marked | `src/config.py` (ε₀ = 1, β ≈ 7.08e-7) | §4 | 4 |
| N3 | Optimal and best-grid weights by γ (3, 5, 8) | Table 3 | §3 | 3 |
| N4 | Mean learning gap by run and γ for baseline, `b63`, `n26m`, `eps-decay`, `all-three` | §6.2 table | §6.2 | 9 |
| N5 | Reward noise against the gap between the top two portfolios | `reports/full_run_2026-10-22/data.json: calibration` | §5.6 | 7 |
| N6 | Schematic of one transaction-cost block: drift, then hold or trade to a target | `TC_DRIFT_DESIGN.md` §3 | §6.5 | 11 |
| N7 | Gap ladder: equal weight, learned, best grid, optimum on one axis | Table 5 | §1 or §5.4 | 1 or 14 |

---

## 9. Slide outline

**TL;DR / In plain terms.** Fourteen slides that follow the report's order. Each lists its title, the points it makes, the numbers it cites and the figure it uses. Slides 2 to 4 are setup, 5 to 8 are the pre-registered results, 9 to 12 are post hoc, 13 and 14 close.

**Key concepts to understand**
- **One claim per slide.** The title states the claim; the figure shows it.
- **Label post hoc on the slide.** Slides 9 to 12 carry "post hoc" in the title or a corner tag, as the report does.

**Key takeaways**
- Slides 5, 6, 7, 8, 10, 12 and 13 use a figure that already exists. Slides 1, 2, 3, 4, 9, 11 and 14 need a new figure (N1 to N7) or fall back on a table or an existing chart named in their row.
- Backup slides: F2, F4, F14, F25, F30, F33, Table TC-3 in full.

| # | Title | Points | Numbers | Figure |
|---|---|---|---|---|
| 1 | Can an agent that only sees rewards find a near-optimal portfolio? | The question in the assignment's words. Nine ETFs, a simulator calibrated once, 1,287 long-only portfolios. The answer in one line | 19.8 bps/yr behind the optimum out of 530.5 (median seed, γ = 5); equal weight 310.9 behind | N7 (to build) or none |
| 2 | The environment: 20 years of daily data, frozen | Nine ETFs, 2006-02-07 to 2026-07-31, 5,152 days, excess returns, one frequency. The agent sees simulated returns only | SPY 10.7% / 19.3%; GLD 9.2% / 18.4%; IEF 1.7% / 6.8%; SPY–IEF −0.29, SPY–GLD 0.07; condition number 268 | N1 (to build) or Table 1 |
| 3 | Two benchmarks under the agent's own constraints | Closed form (no committed script yet, H3) shorts six ETFs, four heavily, so the benchmark is the long-only optimum. The best grid portfolio is the second benchmark | w* = 0.47 SPY, 0.14 IEF, 0.39 GLD; U = 530.5; best 20% grid (0.4, 0.2, 0.4), 5.7 below | Table 3 or N3 (to build) |
| 4 | The agent: one value per portfolio, averaged rewards | Stateless, ε-greedy, α = 1/N, ρ = 0. B = 20, 13M episodes, 10 seeds. Settings and convergence rule fixed 2026-10-16, before the first run on 2026-10-22 | 1,287 actions; ε = 1% at halfway; about 1,097 random tries per portfolio; 260M simulated days per seed | F20 as a table, or N2 (to build) |
| 5 | Convergence: 9 of 10 seeds by the pre-set rule | Rule defined first: the greedy portfolio unchanged through the end for at least 10% of episodes. Statistical and economic versions reported side by side | 9 of 10 at a median of 185.1M days; settled (post hoc) at 47.1M; near-optimal for good 7 of 10 at 25.8M | F12, or F3 |
| 6 | What it learned: SPY, GLD and usually a bond, 14 bps/yr from the best available | Every seed holds SPY and GLD; 8 of 10 add a bond ETF. 3 of 10 pick the best grid portfolio exactly | Learning gap median 14.1 (0 to 47.3); total 19.8; L1 distance to w* 0.331 | F1 and F8 |
| 7 | Why not exact: the top portfolios are 10 bps/yr apart and the noise is 3,900 | One episode's reward SD against the gap between the top two. The best portfolio got about a thousand tries and needed about a million | 10.5 bps/yr; 3,900 bps/yr; 1.1M tries to separate at two SE; Spearman 0.821, top-10 recall 35% | F16, or N5 (to build) |
| 8 | A finer grid trades discretization error for learning error | 25%, 20%, 10% grids. At γ = 5 the total gap barely moves, at γ = 3 and 8 it gets worse; on the 10% grid no seed converges (10% grid is post hoc) | Disc. gap 11.74 / 5.74 / 1.29; median learning gap 2.5 (25%) vs 14.1 (20%); 10% vs 20% at `all-three` settings: paired change −0.6 (SE 3.7) at γ = 5; 0 of 30 seeds meet the rule | F7 and F6 (γ = 5) |
| 9 | Robustness: risk aversion, episode length, budget, exploration (post hoc except γ) | The agent follows the optimum from equity toward bonds as γ rises. Longer episodes and more data cut the gap; slower ε decay mostly does not | γ = 3 / 8 median gap 10.0 / 5.7. Mean gap at γ = 5: baseline 17.6, `b63` 5.1, `n26m` 8.3, `eps-decay` 16.1, `all-three` 7.9 | N4 (to build) or the §6.2 table; F19 for γ |
| 10 | Function approximation: better ranking, worse top pick on the large grid (post hoc) | On the 20% grid splines beat the like-for-like table at γ = 5 and 3 and lose at γ = 8. On the 10% grid they lose at every γ. Reported as a negative result | 20% grid, γ = 5: `tabular-const` 43.5, `fa-spline` 15.7, baseline 17.6. 10% grid: table 11.7, FA 78.8 (paired +67.1, SE 9.1); Spearman 0.976 vs 0.543; top-10 recall 0% | F23 (γ = 5) and F24 |
| 11 | Transaction costs: a redesign after a first design that learned nothing (post hoc) | First design: 1.66M state-action pairs, scored like a random portfolio at zero cost. Redesign: three ETFs, drift, HOLD or trade to a 2% target, costs from real spreads and volume at $100M, $1B, $10B, a numerical oracle | First design 56.6 vs oracle 524.8. Oracle's lead over always rebalancing on the final paths: 0.04, 0.42, 1.78, 7.79 | N6 (to build); F36 as backup |
| 12 | With costs the agent loses to the oracle everywhere, and learns timing only at $10B (post hoc) | Gap splits into where it heads and when it trades. Low cost: the target is wrong and it loses to always rebalancing. $10B: timing is worth 6.2 and it beats always rebalancing | Agent − oracle −13.1, −13.1, −10.0, −2.8 (SD across seeds 4.6, 5.5, 4.3, 1.7). Agent − always rebalance −13.1, −12.7, −8.2, +5.0. Target split −13.1, −12.8, −9.0, −1.1 | F29 and F31 |
| 13 | What we predicted and what we got; what the results rest on | Expectations registered before the TC runs: 8 met, 4 partly met. The gap was larger than predicted. Disclosures: post hoc runs, benchmark information in design choices, Gaussian stationary world, μ and Σ taken as truth | E1: predicted −1 to −8, got −13.1 to −2.8. c2 within ×2 on 10 of 10 seeds at $1B and $10B; c1 off on 8 to 10 of 10. LQD kurtosis 58.9; 260M simulated days is about a million years | F34 (table), F32 as backup |
| 14 | Conclusion | It learns the right assets and a near-optimal mix from rewards alone. Noise, far larger than the differences among the top portfolios, prevents exact recovery. Finer grids and richer models move the error around without removing it at this budget | 14.1 and 19.8 (median, γ = 5); 3 of 10 exact; 310.9 for equal weight | N7 (to build) or F8 |

---

## Appendix A. Reproducibility (outside the page count)

- Commands: `README.md` (data rebuild, `python -m src.run_experiments`, `python -m src.evaluate`, `python -m src.reporting.build`, TC commands). TC report: `python -m src.tc_drift.report`.
- Seeds: reported runs 20261016; FA tuning 20261103; TC tuning 20261112.
- Versions: Python 3.13, numpy 2.3.5, scipy 1.16.3, pandas 2.3.3, matplotlib 3.10.6, yfinance 1.7.0.
- Tests: 758 passing on 2026-11-20 (Mac). Windows: 672 passing on 2026-11-19. Not rerun for the 2026-11-23 update, which changed no code.
- Run registry: `reports/registry.json` (11 runs, each with its git ref). `reports/tc-drift/` and `reports/tc-quadratic-cost/` are outside the registry.
- TC final runs: commit `1787f0d`, clean tree, Albert's Mac (arm64, Python 3.13.9, numpy 2.3.5), BLAS pinned to one thread, 2026-11-20 15:27 to 17:41, about 20 minutes a run (`results/tc-drift/runs/README.md`). Every seed's gap was recomputed from `values.npz` and matched `per_seed.csv` within 1e-13 (same file).
- Known reproducibility gaps on 2026-11-23: closed-form table has no script (H3); three FA analyses have no script (H4); TC oracle tables are git-ignored and take 20 minutes to 2.5 hours to rebuild (M2); the oracle summary and tuning records carry `src_dirty: true` (M1).

---

## Open decisions and open audit findings

**Audit high findings (`docs/reviews/AUDIT_2026-11-20.md` §2), status checked 2026-11-23 at commit `2169c7e`.** All four are flagged and none is fixed.

| # | Finding | Status on 2026-11-23 | What the report does meanwhile |
|---|---|---|---|
| H1 | No written justification for B, the episode count, the seed count, the ε target and K | Open. No "why these settings" text in `docs/methods/EXPERIMENTS.md` | §4's settings table carries draft reasons; the 13M-episode budget's benchmark-scored pilot is disclosed there |
| H2 | No methods prose for Part II or the reward | Open. No `docs/methods/BENCHMARKS.md` | §3 is written from the code docstring and tests |
| H3 | The budget-only closed-form portfolio is in no committed output | Open. `src/benchmarks/` holds `mean_variance.py` only | §3 does not print the closed-form weights |
| H4 | Three FA analyses have no committed script | Open. No such script is tracked | §6.4 words the γ = 8 explanation as a hypothesis |

Audit blocker B2 (the 40 TC runs) is closed: the runs are in `results/tc-drift/runs/`. B1 (the report and slides) is open.

**Decisions (prospectus §6 and the audit).**

| Decision | Owner | Effect on the report | Recommendation |
|---|---|---|---|
| D-G: how the first TC design is reported | Both (William's input needed) | §6.5 first paragraph; whether old items M7 to M11 need fixing | One paragraph and the four-row table printed in §6.5 above; no HTML report text reused |
| Report scope: which post hoc runs get full treatment (audit M8) | Both | Length of §6.2 and §6.4 | `grid10-robustness`, γ = 3 and 8, and `all-three` in full; `b63`, `n26m`, `eps-decay` in one table row and two sentences each |
| Report scope: how much of the TC extension | Both | Length of §6.5 | Tables TC-1 to TC-3, F29, and the disclosures; F30 to F33 in an appendix |
| Final run list (candidates in `AUDIT_2026-11-06.md` §6) | Both | Whether any new run appears | Run nothing new |
| Commit the FA analysis scripts or word as hypothesis (H4) | William | Wording in §6.4 | Hypothesis wording unless committed |
| Closed-form table script (H3) | William | Whether §3 can print the budget-only portfolio | Add the script |
| Commit oracle tables and per-seed TC logs (M2) | Both | Appendix A | Commit them |
| Where the "why these settings" text lives (H1) | Albert | §4 | Write in `EXPERIMENTS.md`, copy into §4 |
| Investigate why the favourite-target error is smaller at $10B | Both | One sentence in §6.5 | Leave as "not investigated" unless there is time; do not guess a cause |

The TC go/no-go that the 2026-11-20 skeleton listed for 2026-11-24 no longer applies: the runs finished on 2026-11-20 and were merged in PR #47.

---

## Issues found while updating (2026-11-23)

Nothing below was fixed. Only this file was edited.

**Source disagreements (both values cited where they are used).**

1. Oracle's lead over always rebalancing: three committed values per level from three samples (§6.5 table). The 2026-11-20 skeleton quoted the in-model and m = 1,000 values (1.3 at $1B); the final paths give 1.78.
2. Oracle hold fraction: 0.156 / 0.272 / 0.308 / 0.443 in `results/tc-drift/oracle/summary.json` against 0.153 / 0.283 / 0.320 / 0.453 in `results/tc-drift/runs/<level>/summary.json`. In-model against replay.
3. E11's target: `TC_DRIFT_DESIGN.md` §7 predicted "about 0.43" from the 1.30 lead; `d10.target_se` at $1B is 0.593 from the 1.78 lead. `results/tc-drift/runs/README.md` notes it.
4. Reward SD at B = 20: 3,909 bps/yr in the audit notes (no committed file) against 3,900 from `reports/full_run_2026-10-22/data.json`. §4 prints "about 3,900".
5. Visits needed to separate the top two portfolios: "about 139,000" in the 2026-11-20 skeleton (one standard error, single mean) against 1,097,318 in `reports/full_run_2026-10-22/data.json` (two standard errors, difference of means). §5.6 now uses the committed figure.
6. F13 (weights chart in the original HTML report) averages the ten seeds' portfolios; §5.3 says never to report the average. Use F1.

**Stale text in other docs.**

7. `docs/methods/TC_DRIFT_DESIGN.md:3` (status line) says "The 40 agent runs have not started". They ran on 2026-11-20. Shared file; flag to William.
8. `docs/PROJECT_PROSPECTUS.md:5` says the 40 runs "are ready to launch" and "What remains is the TC agent runs and their analysis". William's file; flag, do not edit.
9. `README.md:128` describes `results/tc-drift/runs/<level>/` as existing "once the agent runs finish", and no line of `README.md` mentions `reports/tc-drift/` or `python -m src.tc_drift.report`.
10. `reports/README.md`, "Reports outside this pipeline", lists `reports/tc-quadratic-cost/` only. `reports/tc-drift/` is also outside the registry and is not mentioned.
11. `docs/notes/NEXT_STEPS_FOR_ALBERT.md:21` still lists the TC runs as a to-do with "Go/no-go Nov 24".
12. `CLAUDE.md` §4's doc table has no rows for `docs/reviews/AUDIT_2026-11-20.md`, `docs/report/REPORT_SKELETON.md` or `results/tc-drift/runs/README.md` (`docs/README.md` has the first two).
13. `HANDOFF.md` (untracked, dated 2026-10-09) is six weeks out of date, and Claude's memory index points to it (audit M6, unchanged).
14. `reports/tc-quadratic-cost/report.html` still carries the two sentences `TRANSACTION_COSTS.md` says should change (old M8). Depends on D-G.

**Could not fill.**

15. The agent row of Table TC-1 is a mean computed for this skeleton from `per_seed[].agent.mean_value_bps`; no committed file stores that mean as one field.
16. Why the agent's timing is worth 6.2 bps/yr at $10B while it holds less often than the oracle, and why the target error shrinks at $10B, have no committed analysis.
17. The rendered appearance of the browser-drawn charts (F7 to F28) was not checked; their titles, captions and data were read from the HTML source and `data.json`.
18. The test suite was not rerun for this update.
