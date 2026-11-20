# Report Skeleton: Model-Free Q-Learning for Mean-Variance Portfolio Choice

**By:** Claude, for Albert and William · **Date:** 2026-11-20 · **Status:** skeleton; no report text written yet
**Built from:** `docs/reviews/AUDIT_2026-11-20.md` and its notes folder. Every number below was tied to the committed file named beside it on 2026-11-20.

---

## TL;DR / In plain terms

This is the outline of the 10 to 15 page report, section by section, in the order the assignment requires. Each section lists what it has to claim, the exact numbers with the file they come from, the figures that already exist, the caveats that must be printed, and one line for the slide deck. A writer can work from this file without re-checking the repo. The transaction-cost results section is a placeholder until the 40 runs finish.

**Key concepts to understand**

- **Three portfolios, two gaps.** The continuous optimum w* is the best long-only portfolio with any weights. The best grid portfolio is the best one the agent is allowed to pick. The learned portfolio is what the agent picked. Discretization gap = best grid vs w*. Learning gap = learned vs best grid. Total gap = their sum.
- **Certainty-equivalent loss in bps per year.** Every gap is 252 × 10,000 × (difference in daily utility U). It reads as "the investor gives up this many basis points of sure return per year".
- **Exact vs sampled evaluation.** Gaps are computed from the known μ and Σ, since a 10,000-episode sample has a standard error of about 40 bps/yr, larger than the gaps. The fresh evaluation draws are reported beside them.
- **Pre-registered vs post hoc.** Only the four `baseline-v1` experiments had settings and convergence rules fixed before any result. Every other run is labelled post hoc wherever it appears.

**Key takeaways**

- Headline answer to the research question: at γ = 5 on the 20% grid, the median seed ends 14.1 bps/yr behind the best grid portfolio and 19.8 bps/yr behind the continuous optimum, out of 530.5 bps/yr available; 3 of 10 seeds pick the best grid portfolio exactly. Equal weight is 310.9 bps/yr behind.
- What prevented exact recovery: reward noise (about 3,909 bps/yr per 20-day episode) against a 10.5 bps/yr difference between the best and second-best grid portfolio.
- Finer grids shrink the discretization gap and make learning harder; the two roughly cancel.

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

Rubric buckets (20% each): Data and environment → §2. Benchmark and feasible set → §3. Q-learning design → §4. Convergence and comparison → §5. Interpretation, robustness, insight → §6, §7.

Page budget (suggested): §1 one, §2 one and a half, §3 one and a half, §4 two, §5 three and a half, §6 four, §7 half. Appendix outside the count.

---

## 1. Introduction and central research question

**Purpose.** State the question and how the report separates the four things the assignment asks to separate.

**Claims.**
- Question (quote): "Can a model-free learning algorithm recover a near-optimal portfolio allocation when it observes realized rewards but does not know expected returns or covariances?"
- Setup in two sentences: nine ETFs, a Gaussian simulator calibrated once, a stateless Q-learning agent choosing among 1,287 long-only portfolios.
- Preview the answer with the three headline numbers in the takeaways above.
- Say which results were pre-registered and which were added later.

**Numbers.** 14.1 and 19.8 median, 530.5, 310.9: `results/baseline_g5_grid20/evaluation/summary.json` (`learned.ce_loss_vs_best_discrete_bps.median`, `learned.ce_loss_vs_continuous_bps.median`, `benchmarks.*.exact.objective_daily`).

**Figures.** None.

**Caveats.** None here beyond the pre-registered / post hoc sentence.

**Slide takeaway.** A model-free agent gets within about 20 bps/yr of the optimum out of 530; noise, not the method, stops it getting the rest.

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
- Table 1 (annualized mean / volatility / Sharpe): SPY 10.7% / 19.3% / 0.55; EFA 6.1 / 21.4 / 0.29; EEM 7.5 / 28.0 / 0.27; IEF 1.7 / 6.8 / 0.25; LQD 2.6 / 8.6 / 0.30; TIP 1.7 / 6.2 / 0.28; GLD 9.2 / 18.4 / 0.50; VNQ 8.9 / 29.0 / 0.31; DBC 2.3 / 19.3 / 0.12. Source `data/processed/summary_statistics.csv`.
- Table 2 (correlation matrix): `data/processed/correlation_matrix.csv`. Quote SPY–EFA 0.88, SPY–EEM 0.82, IEF–TIP 0.76, SPY–IEF −0.29, SPY–GLD 0.07.
- Σ condition number 268.07, smallest eigenvalue 3.10e-6: `data/processed/CLEANING_LOG.json` (`covariance_diagnostics`), `covariance_daily.csv`.
- Download date: `data/raw/MANIFEST.json`. Row reconciliation and adjustment checks: `docs/methods/DATA_CLEANING.md` §3, §5.

**Figures and tables.** Tables 1 and 2 must be formatted from the two CSV files (no figure exists yet). Optional: correlation heatmap (to build).

**Caveats and disclosures.**
- Excess kurtosis is large (LQD 58.9, EEM 17.8, VNQ 17.2, SPY 15.0) and the simulator is Gaussian. Forward reference to §6.3.
- The French risk-free series takes few distinct values; effect on the optimum checked to 3.6e-9 (`DATA_CLEANING.md`).
- Extreme days kept (EEM +22.77% on 2008-10-13).

**Slide takeaway.** 5,152 daily observations, 2006 to 2026, one frequency, raw data frozen; the agent only ever sees simulated returns.

---

## 3. Theoretical benchmark and feasible portfolio set

**Purpose.** Define the objective, derive the closed form, solve the constrained problem the agent faces, and report both benchmarks. (Bucket: benchmark and feasible set.)

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
- Discretization gap by grid step (25% / 20% / 10%): γ = 5: 11.74 / 5.74 / 1.29. γ = 3: 2.51 / 2.47 / 2.47. γ = 8: 22.60 / 6.13 / 6.13. Sources: `summary.json: discretization_gap.ce_loss_bps` in `results/resolution_g5_grid25/`, the three 20% folders, and `results/grid10-robustness/*/`. The 25% values at γ = 3 and 8 are the audit's calculation only; no committed run has them, so leave them out or mark them.
- Budget-only closed form at γ = 5: weights (1.82, −0.88, −0.19, 0.27, −0.02, −0.27, 0.69, −0.13, −0.29) in ticker order, U = 925.9 bps/yr. **No committed source.** Audit calculation only (`docs/reviews/AUDIT_2026-11-20.md` H3). Do not print until a committed script produces it.

**Figures and tables.** Table 3. No existing figure; the weights chart in §5 shows both benchmarks.

**Caveats and disclosures.**
- Only three of nine ETFs have positive weight in w* at any γ. The grid's best portfolio shares that support.
- Bounded shorting (bonus) was not attempted.
- 5% grid (3,108,105 portfolios) is counted, not trained.

**Slide takeaway.** Same constraints for theory and agent; at γ = 5 the best grid portfolio is 5.7 bps/yr below the continuous optimum.

---

## 4. Q-learning design and implementation

**Purpose.** Specify the agent completely and justify each setting. (Bucket: Q-learning design.) This section carries the audit's largest rubric risk (H1): each row below needs a reason, not only a value.

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
| Learning rate | α = 1/N(a), N(a) = visits to a | Makes Q(a) the sample mean of a's rewards, the unbiased estimate in a stationary problem; satisfies the usual step-size conditions |
| Discount | ρ = 0 | No state carries over between episodes; the assignment's "clean bandit-like benchmark" |
| Initial Q | 0 | No prior information; with 1/N the first visit overwrites it |
| Returns per episode | B = 20 days | About one trading month; long enough for a variance estimate, short enough for 13M updates. `b63` tests the choice (§6.2) |
| Episodes | 13,000,000 per seed (260M simulated days) | **Disclose:** sized from a pilot of about 13M episodes that came within roughly 2% of the best grid portfolio's value (`docs/reviews/REVIEW_HISTORY.md` §4). The pilot was scored against the benchmark by us; the agent never saw it. `n26m` doubles it (§6.2) |
| Exploration | ε₀ = 1, β ≈ 7.08e-7 so ε = 1% at the halfway episode | First half explores (about 1.41M random picks, about 1,097 per portfolio), second half mostly exploits. `eps-decay` tests 3% (§6.2) |
| Seeds | 10, master seed 20261016 | Enough to report a distribution; cost about 3.5 hours for the baseline set |
| Tie-breaking | Random among maxima when acting; lowest index when reporting the final portfolio | Random ties avoid favouring low-index portfolios during learning; a fixed rule makes the reported portfolio reproducible |
| Evaluation | 10,000 fresh episodes, exploration off | Separate stream, never used for updates |

**Numbers and sources.** `src/config.py`; every `results/*/manifest.json` records the settings; `docs/methods/EXPERIMENTS.md` §1. Reward SD about 3,909 bps/yr at B = 20, 7.45 times U(w*): audit calculation (`audit_2026-11-20_notes/03_environment_benchmarks.md`); it also follows from `performance.csv` (`realized_objective_se` 1.56e-5 × √10,000).

**Figures.** Optional: ε schedule (to build). None existing.

**Caveats and disclosures.**
- Settings fixed in commit `07bfb64` (2026-10-16); first full run 2026-10-22.
- An illustrative demo (γ = 5, B = 20, 2M episodes) ran before the settings were fixed (`docs/methods/EVALUATION.md` §6).
- Benchmark information entered one design choice (the episode budget). It did not enter the action set or the agent.
- Outputs count simulated days; episodes = days / B.

**Slide takeaway.** One table value per portfolio, sample-average updates, exploration decaying to 1% by halfway; every setting fixed before the first real run.

---

## 5. Convergence and benchmark comparison

**Purpose.** Report what was learned, how fast, how reliably, and how far from each benchmark. (Bucket: convergence and comparison.)

### 5.1 Convergence definition (fixed before results)

**Claims.**
- Primary rule (`greedy_stable_through_end`): the greedy portfolio stops changing and stays unchanged to the end of training, for at least K = 10% of episodes (1.3M). Uses the agent's own Q-table only.
- Also pre-registered: `greedy_stable`, `dq_below_tol_for_good` (largest Q change below 1e-5 per 10,000-episode window), `near_optimal_for_good` (within 25 bps/yr of the best grid portfolio from then on; uses the benchmark, evaluation only).
- Statistical vs economic convergence: the first three are statistical, the fourth is economic.

**Caveats.** `settled` and `near_optimal_every_episode` were added 2026-11-09 and the Q-fit measures 2026-11-04. Label them post hoc and descriptive. The 25 bps line is 3.5%, 4.8% and 6.7% of the best grid value at γ = 3, 5, 8 and contains 2, 4 and 2 of the 1,287 portfolios.

### 5.2 How often and how fast

**Table 4 (source `results/INDEX.md`; each row's `evaluation/summary.json: convergence.*`).**

| Experiment | Primary rule met | Median days to meet it | Settled, median days (post hoc) | Near-optimal for good | Median days |
|---|---|---|---|---|---|
| γ = 5, 20% grid | 9 of 10 | 185.1M | 47.1M | 7 of 10 | 25.8M |
| γ = 5, 25% grid | 10 of 10 | 101.2M | 18.5M | 9 of 10 | 9.6M |
| γ = 3, 20% grid | 8 of 10 | 192.6M | 26.4M | 6 of 10 | 22.2M |
| γ = 8, 20% grid | 9 of 10 | 153.2M | 27.9M | 7 of 10 | 25.8M |

Episodes = days / 20. The primary rule's count includes the K waiting window.

**Caveats.** Non-converged seeds: 1 (γ = 5), 2 (γ = 3), 1 (γ = 8) on the 20% grid. Their weights and gaps are still reported in §5.3 and §5.4 (say so). Near-optimal read at checkpoints (25.8M days) is about six times faster than read at every episode (155.7M days at γ = 5).

### 5.3 Learned weights against the benchmarks

**Claims.** At γ = 5 every seed's portfolio holds SPY and GLD plus one bond-like ETF; 3 of 10 are the best grid portfolio (0.4, 0.2, 0.4). Report per seed, never the average across seeds (a blend is not a portfolio any seed chose and can beat the grid).

**Numbers.** L1 distance to w*: median 0.331. Euclidean: median 0.216. Source `summary.json: learned.l1_to_continuous`, `learned.euclidean_to_continuous`. Absolute differences by asset: `weights.csv` (columns `absdiff_vs_continuous_run_*`).

**Figures.** `results/baseline_g5_grid20/evaluation/weights_grouped_bar.png` (nine weights; continuous, best grid, equal weight, learned).

**Caveats.** "Average learned weights over a steady-state evaluation sample" is read as the final greedy portfolio with exploration off, so the average equals that portfolio. The chart's "median seed" is the 6th of 10 by gap.

### 5.4 Performance and gaps

**Table 5 (γ = 5; source `results/baseline_g5_grid20/evaluation/performance.csv`).**

| Portfolio | Return | Volatility | U (bps/yr) | Loss vs w* |
|---|---|---|---|---|
| Continuous optimum | 8.86% | 11.92% | 530.5 | 0 |
| Best grid | 8.28% | 11.01% | 524.8 | 5.7 |
| Learned (median seed; range) | 8.28% median | 11.11% median | | 19.8 (5.7 to 53.1) |
| Equal weight | 5.62% | 11.71% | 219.6 | 310.9 |

**Table 6 (gaps; source each `evaluation/summary.json`, `per_seed.csv`).**

| Experiment | Discretization gap | Learning gap, mean / median (range) | Total gap, mean / median | Exactly best grid |
|---|---|---|---|---|
| γ = 5, 20% | 5.74 | 17.6 / 14.1 (0 to 47.3) | 23.3 / 19.8 | 3 of 10 |
| γ = 5, 25% | 11.74 | 5.1 / 2.5 (0 to 30.7) | 16.9 / 14.3 | 5 of 10 |
| γ = 3, 20% | 2.47 | 30.6 / 10.0 (0 to 81.1) | 33.0 / 12.5 | 5 of 10 |
| γ = 8, 20% | 6.13 | 13.0 / 5.7 (0 to 53.1) | 19.2 / 11.9 | 5 of 10 |

**Figures.** `results/baseline_g5_grid20/evaluation/ce_loss_by_seed.png` (distribution across seeds).

**Caveats.** Gaps use exact μ and Σ. The fresh 10,000-episode sample gives the realized columns in `performance.csv` with a standard error near 1.6e-5 daily (about 40 bps/yr), too coarse to rank portfolios 10 bps apart. Say this in one sentence so the reader knows why both exist.

### 5.5 Learning curve, regret and resolution

**Figures (all exist).**
- `results/baseline_g5_grid20/evaluation/learning_curve.png`
- `results/baseline_g5_grid20/evaluation/cumulative_regret.png` (against the best grid portfolio)
- `results/comparison_g5_grid20_vs_grid25/convergence_by_resolution.png`
- For the 10% grid: `results/grid10-robustness/comparison_g{3,5,8}_grid20_vs_grid10/convergence_by_resolution.png` (post hoc; used in §6.1)

**Claims.** Coarser grid: larger discretization gap (11.74 vs 5.74), smaller learning gap (median 2.5 vs 14.1), faster and more reliable convergence (10 of 10 at 101.2M days vs 9 of 10 at 185.1M). Total gap median 14.3 vs 19.8.

### 5.6 What prevented exact recovery

**Claims.** The second-best grid portfolio is 10.5 bps/yr behind the best at γ = 5 (20.0 at γ = 3, 11.5 at γ = 8), while one episode's reward has a standard deviation near 3,909 bps/yr. Separating the two by sample averages needs on the order of (3,909 / 10.5)² ≈ 139,000 visits to each, and exploration gives each portfolio about 1,097 random visits before the agent narrows to a few. Q-fit supports this: the table ranks portfolios well overall (median Spearman 0.821) and poorly at the very top (top-10 recall 0.35).

**Numbers.** Second-best gaps: audit calculation, tied to `REVIEW_HISTORY.md` §8. Spearman and recall: `summary.json: q_fit.all` (post hoc measure).

**Caveat.** The 139,000 figure is a back-of-envelope order of magnitude, not a result. Label it.

**Slide takeaways.**
- 9 of 10 seeds converge by the pre-set rule; median seed is 14 bps/yr from the best available portfolio and 20 from the optimum.
- Coarser grid: easier to learn, worse ceiling. The total barely moves.
- The top portfolios differ by about 10 bps/yr; one episode's noise is about 3,900.

---

## 6. Robustness, research challenges, and limitations

**Purpose.** The two required checks, the further runs, the six discussion topics, and the two extensions. (Bucket: interpretation, robustness, insight.) Everything in §6.2, §6.4 and §6.5 is post hoc; the 10% grid in §6.1 is too.

### 6.1 Action-space design (required check 1)

**Claims.** Three resolutions at γ = 5: 25% (495), 20% (1,287), 10% (43,758). The 10% grid was run with the `all-three` settings (B = 63, 26M episodes, ε = 3% at halfway), so compare it with `all-three`, not with the baseline.

**Numbers (source `results/grid10-robustness/INDEX.md`, `results/all-three/INDEX.md`, their `summary.json`).**

| γ | Disc. gap 20% → 10% | Mean learning gap 20% → 10% | Exactly best 20% → 10% | Total gap paired change (SE) | Primary rule 20% → 10% |
|---|---|---|---|---|---|
| 5 | 5.7 → 1.3 | 7.9 → 11.7 | 5 → 0 of 10 | −0.6 (3.7) | 10 → 0 of 10 |
| 3 | 2.5 → 2.5 | 4.0 → 22.0 | 8 → 4 | +18.0 (7.5) | 10 → 0 |
| 8 | 6.1 → 6.1 | 1.1 → 5.8 | 9 → 1 | +4.6 (1.6) | 10 → 0 |

**Caveats.** On the 10% grid no seed of 30 meets the primary convergence rule; each portfolio gets about 85 random tries. Weights for those seeds are final greedy portfolios, not converged ones. Near-optimal-for-good is still met by 10, 5 and 10 of 10 seeds. Post hoc (registered 2026-10-27).

### 6.2 Economic and statistical assumptions (required check 2, and further runs)

**Claims.**
- Risk aversion γ = 3 and 8 (pre-registered, part of `baseline-v1`): Table 6 rows. The optimum moves from equity and gold toward bonds as γ rises; the agent follows.
- Post hoc tabular variants (registered 2026-10-27, run 10-28 and 10-29): longer episode, more episodes, slower ε decay, and all three.

**Numbers (mean learning gap; change vs baseline with SE; source `results/{b63,n26m,eps-decay,all-three}/INDEX.md` and `per_seed.csv`).**

| Run | Change | γ = 5 | γ = 3 | γ = 8 |
|---|---|---|---|---|
| baseline | | 17.6 | 30.6 | 13.0 |
| `b63` | B = 63 | 5.1 (−12.5, SE 6.2) | 10.0 (−20.6, 10.4) | 9.1 (−3.9, 6.6) |
| `n26m` | 26M episodes | 8.3 (−9.3, 5.3) | 16.2 (−14.3, 14.7) | 5.0 (−8.0, 7.2) |
| `eps-decay` | ε = 3% at halfway | 16.1 (−1.5, 7.2) | 4.0 (−26.6, 11.1) | 11.6 (−1.4, 9.2) |
| `all-three` | all of the above | 7.9 (−9.7, 5.9) | 4.0 (−26.6, 11.7) | 1.1 (−11.9, 5.4) |

**Caveats.** Post hoc: chosen after seeing the baseline. Runs with a different B or ε consume the random streams differently, so these differences share a starting seed but are not paired draw for draw. Most single differences are within two standard errors of zero; say which are not. `b63` uses three times the simulated days per episode, so it is not a like-for-like budget. B = 252 was not run.

**Open decision.** Which of these get full treatment (audit M8). Recommendation: this one table plus two sentences each.

### 6.3 Research challenges and limitations (the six required topics)

No written text exists for any of these. Facts to build on:

| Topic | Facts available | Source |
|---|---|---|
| Estimation error in μ and Σ | μ, Σ estimated once from 5,152 days and treated as truth; the agent's result is conditional on them. No alternative window or shrinkage run was done | `DATA_CLEANING.md`; say "not done" |
| Conditioning and numerical stability of Σ | Condition number 268.07, smallest eigenvalue 3.10e-6; Cholesky is stable; the budget-only closed form's large short positions show sensitivity to Σ⁻¹ | `CLEANING_LOG.json`; §3 |
| Curse of dimensionality | 495 → 1,287 → 43,758 → 3,108,105 portfolios; random tries per portfolio fall from about 1,097 (20%) to 85 (10%, with double the budget); 0 of 30 seeds converge on the 10% grid | §6.1 |
| Discretization error and benchmark leakage | Gap 11.74 / 5.74 / 1.29 at γ = 5; grid built without benchmark input; tests enforce import allowlists; disclosed human uses of the benchmark (episode budget; TC design) | §3, §4, §6.5 |
| Reward and objective consistency | E[reward] = U(w) with ddof = 1; headline gaps on exact moments | §4; `test_rewards.py` |
| Stationary simulated environment | iid Gaussian, fixed moments, no regimes, no fat tails (LQD kurtosis 58.9 in the data); 260M simulated days is about a million years of daily data; 1/N averaging would fail if moments drifted | §2 |

### 6.4 Extension: function approximation (post hoc)

**Claims.** Replace the table with Q̂(a) = θ·φ(w_a): quadratic features (45 columns) or splines (909 columns, rank 369 on the 20% grid). Constant step size, tuned on a separate master seed (20261103) at γ = 5 only. The like-for-like tabular comparison is `tabular-const` (same constant-α rule), since 1/N is unavailable to FA.

**Numbers (mean learning gap; source `results/{tabular-const,fa-quadratic,fa-spline}/INDEX.md`, `results/fa-spline-grid10-eps{3,10}_2026-11-11/`, `docs/methods/EXPERIMENTS.md` §8.7, §8.8).**

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

**Figures.** HTML reports `reports/{tabular-const,fa-quadratic,fa-spline,fa-spline-grid10-eps3,fa-spline-grid10-eps10}/`; PNGs in each experiment's `evaluation/` folder.

**Caveats and disclosures.**
- Post hoc (registered 2026-11-05 and 2026-11-11). Step size tuned on separate seeds, at γ = 5 only; the 10% grid step (2.778e-5) was noise-matched, not re-tuned.
- FA beats the like-for-like table at γ = 5 and 3 and loses at γ = 8; on the 10% grid it loses to the table at every γ. This contradicts the registered expectation that FA would help most on the large grid. Report it as a negative result.
- The ΔQ rule says nothing under a constant α.
- Explanation for γ = 8 ("directions that have not finished learning": 36 of 45 quadratic and 351 of 369 spline directions) rests on calculations with no committed script (audit H4). Word it as a hypothesis unless William commits them.
- On the 10% grid FA ranks portfolios well overall (median Spearman 0.976) and picks badly at the top (top-10 recall 0.00).

### 6.5 Extension: transaction costs

**First design (negative result).** One paragraph, pending decision D-G. State = previous grid target, quadratic cost, 1.66M (holding, target) pairs, 57.1% ever tried. At zero cost the learned policy scores 56.6 against a grid oracle of 524.8 and a random-portfolio average of 56.7. Source `docs/methods/TRANSACTION_COSTS.md`, `results/tc-quadratic-cost/*.json`. Lesson: the state space was too large for the data, which motivated the redesign.

**Redesign with drift: design (can be written now).**
- Three ETFs (SPY, IEF, GLD), γ = 5, 63-day blocks during which holdings drift with returns. Action: one of 1,326 targets on a 2% grid, or HOLD. Cost = κ‖trade‖₁ + (λ/2)‖trade‖₂² as a fraction of wealth. Discount 0.95 per block (explain: about 0.81 per year, half the weight within 3.4 years).
- Cost levels from own liquidity data (spread 0.5 bp; square-root impact): zero; $100M (κ 1.63 bp, λ 4.4e-3); $1B headline (3.90 bp, 1.43e-2); $10B (11.04 bp, 4.57e-2). Source `src/tc_drift/calibration.py`, `docs/methods/TC_DRIFT_DESIGN.md` §8.
- Numerical oracle (value iteration on a 1% lattice, 2,000 paths). Its lead over always rebalancing: 0.008 / 0.214 / 1.319 / 6.939 bps/yr at zero / $100M / $1B / $10B; hold fraction 0.156 / 0.272 / 0.308 / 0.443. Source `results/tc-drift/oracle/summary.json` (`settings.<level>.lead_over_always_rebalance_bps`, `.oracle.hold_fraction`).
- Replay check on fresh paths (1,000 starts, 90 blocks): lead 1.30 (SE 0.70) at $1B and 7.10 (1.26) at $10B. Source `results/tc-drift/checks/oracle_replay_2026-11-17.json`.
- Agent: linear Q with 87 spline columns plus two cost features whose coefficients it learns from net rewards; recursive least squares on the cost block after a burn-in; reported policy uses θ averaged over the final 10% of training.
- Tuning: α = 3e-4 chosen over 1e-3 on tuning seed 20261112 at $1B, by realized net reward (505.4 vs 496.1, paired difference 9.31, SE 6.32, 3 seeds). Source `results/tc-drift/tuning/summary.json`.

**Economic point to make now.** The whole prize is small: the oracle beats "always rebalance" by 1.3 bps/yr at $1B. The interesting question is whether the agent learns when not to trade.

#### TO FILL AFTER THE TC RUNS

Source for every cell: `results/tc-drift/runs/<level>/summary.json` and `per_seed.csv`, levels `zero`, `aum_100m`, `aum_1b`, `aum_10b`; 10 seeds, m = 4,000 starts, H = 90 blocks. These files do not exist on 2026-11-20.

| # | Registered expectation (2026-11-20, before any run) | Field | Result | Met? |
|---|---|---|---|---|
| E1 | Agent − oracle mean between −1 and −8 at each level; negative on at least 8 of 10 seeds | `agent_minus_oracle_mean`; per seed in `per_seed.csv` | TO FILL | |
| E2 | Favourite-target error is most of the gap at zero, $100M, $1B | the two `split_*` means | TO FILL | |
| E3 | Agent loses to always-rebalance at $100M and $1B; relatively better at $10B than at $1B | `agent_minus_always_rebalance_mean` | TO FILL | |
| E4 | Agent − own target rises from $1B to $10B and is positive at $10B; within ±1 at zero and $100M | `split_agent_minus_own_target_mean` | TO FILL | |
| E5 | Agent beats trade-once-then-hold at every level and seed | `agent_minus_trade_once_then_hold` (per seed) | TO FILL | |
| E6 | Agent loses to myopic at every level | `agent_minus_myopic_mean` | TO FILL | |
| E7 | Hold fraction varies across seeds; mean higher at $10B than zero; at zero cost the agent holds more than the oracle | `agent_hold_fraction` | TO FILL | |
| E8 | Quadratic cost coefficient learned better than the linear one (c2 within ×2 on at least 7 of 10 seeds at $1B and $10B; c1 off by more than ×2 on at least 3 of 10 at each cost level) | `c1_over_true`, `c2_over_true` | TO FILL | |
| E9 | Fewer than 5 of 10 seeds converge per level under the 2026-11-20 rule; none under the original and exact-action rules on the last iterate | `d12_converged_count` | TO FILL | |
| E10 | Tail-averaged and last-iterate policies differ by under 1 bps/yr in the mean | `agent_minus_last_iterate_mean` | TO FILL | |
| E11 | Precision target missed at $1B (SE 0.5 to 0.8 against about 0.43), met at $10B, not applicable at zero and $100M | `d10` | TO FILL | |
| E12 | Across-seed SD of agent − oracle at $1B at least 0.5 | `sd_across_seeds` | TO FILL | |

Also to fill: one table of agent, oracle, always-rebalance, trade-once-then-hold and myopic values per level; hold fraction and turnover for agent vs oracle; a per-seed gap chart; the commit hash the 40 runs came from.

**Disclosures for the TC extension (print all).**
- The three-ETF universe is the support of the nine-asset optimum: benchmark information used in the design.
- The cost features build in the cost family's shape; the agent learns two coefficients, clamped at zero from below.
- Learning rule, tail averaging and the convergence rule were chosen with pilot benchmark diagnostics in view. The convergence rule changed three times before any final run (frozen 2026-11-20); all four verdicts are reported.
- E1 to E12 lean on the $1B tuning pilot (same agent, different seeds).
- Tuned at $1B only. A first set of six pilot runs was discarded.
- Oracle check 3 was restated after the fact; check 5 is a two-standard-error bound at the cost levels.
- One cost for every asset pair undercharges the typical SPY-vs-GLD drift trade by about a fifth.
- Oracle summary and tuning records were produced from uncommitted code (`src_dirty: true`).
- Drift is applied on excess returns (effect bounded at 7.1e-6).

**Slide takeaways.**
- Two required checks done: grid resolution and risk aversion. Longer episodes and more data help; the finest grid does not.
- Function approximation matches the table on the small grid and fails on the large one: a negative result, reported as one.
- Transaction costs: TO FILL. Design line: the oracle's edge from trading less is 1.3 bps/yr at $1B, so the bar is low and the signal is faint.

---

## 7. Conclusion

**Purpose.** Answer the question in the assignment's words: what the agent learned, how close it came to the best attainable action, what prevented exact recovery.

**Claims.**
- Learned: the right three assets and roughly the right mix at every γ, from rewards alone.
- How close: median 14.1 bps/yr from the best attainable portfolio and 19.8 from the continuous optimum at γ = 5, against 310.9 for equal weight; exact recovery in 3 of 10 seeds (5 of 10 at γ = 3 and 8).
- What prevented it: top grid portfolios about 10 bps/yr apart against per-episode noise near 3,900.
- Trade-off: a finer grid removes discretization error and adds learning error.
- Limits: known stationary Gaussian world; μ and Σ taken as truth; a data budget no investor has.

**Caveat.** Keep the pre-registered results (baseline, 25% grid, γ = 3 and 8) as the headline; post hoc runs support, they do not replace.

**Slide takeaway.** Model-free learning recovers a near-optimal portfolio in a clean simulated world, at a data cost that makes the benchmark's value clear.

---

## Appendix A. Reproducibility (outside the page count)

- Commands: `README.md` (data rebuild, `python -m src.run_experiments`, `python -m src.evaluate`, `python -m src.reporting.build`, TC commands).
- Seeds: reported runs 20261016; FA tuning 20261103; TC tuning 20261112.
- Versions: Python 3.13, numpy 2.3.5, scipy 1.16.3, pandas 2.3.3, matplotlib 3.10.6, yfinance 1.7.0.
- Tests: 758 passing on 2026-11-20 (Mac). Windows: 672 passing on 2026-11-19.
- Run registry: `reports/registry.json` (11 runs, each with its git ref).
- Known reproducibility gaps on 2026-11-20: closed-form table has no script (H3); three FA analyses have no script (H4); TC oracle tables are git-ignored and take 20 minutes to 2.5 hours to rebuild (M2).

---

## Open decisions that affect the report (prospectus §6 and the audit)

| Decision | Owner | Effect on the report | Recommendation |
|---|---|---|---|
| D-G: how the first TC design is reported | Both | §6.5 first paragraph; whether old items M7 to M11 need fixing | One paragraph and the four-row table |
| Which post hoc runs get full treatment | Both | Length of §6.2 | Table in §6.2 plus two sentences each |
| Final run list (candidates in `AUDIT_2026-11-06.md` §6) | Both | Whether any new run appears | Run nothing new except the TC runs |
| TC go/no-go (2026-11-24) | William | Whether §6.5's placeholder is filled or the extension is reported as design plus oracle only | Decide on the 24th; the design half of §6.5 stands either way |
| Commit the FA analysis scripts or word as hypothesis (H4) | William | Wording in §6.4 | Hypothesis wording unless committed by Nov 24 |
| Closed-form table script (H3) | William | Whether §3 can print the budget-only portfolio | Add the script |
| Commit oracle tables and per-seed TC logs (M2) | Both | Appendix A | Commit them |
| Where the "why these settings" text lives (H1) | Albert | §4 | Write in `EXPERIMENTS.md`, copy into §4 |
