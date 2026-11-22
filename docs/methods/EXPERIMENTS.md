# Running the Experiments

**Owner:** Albert · **Status:** settings fixed 2026-10-16; first full-scale run done 2026-10-22 (tag `baseline-v1`, `results/`); post-hoc runs registered 2026-10-27, all five run 2026-10-28 to 2026-10-29 (§7); constant-learning-rate runs `tabular-const`, `fa-quadratic` and `fa-spline` registered 2026-11-05 and run 2026-11-06, outcome in §8.7; `baseline-v1`, `n26m`, `all-three`, `grid10-robustness` and `fa-spline` re-evaluated 2026-11-09 to add the total gap, the post-hoc settled measure, Q fit and provenance, with every earlier number unchanged (`AUDIT_2026-11-06.md` §2.1); the transaction-cost runs are run separately by `src/transaction_costs.py` (`TRANSACTION_COSTS.md`); FA on the 10% grid at two exploration levels (`fa-spline-grid10-eps3`, `fa-spline-grid10-eps10`) registered 2026-11-11 before training (§8.8); later runs follow `reports/README.md`
**Settings:** `src/config.py` · **Runner:** `src/run_experiments.py` · **Tests:** `src/tests/test_config.py`

---

## TL;DR / In plain terms

One command trains every `baseline-v1` experiment, evaluates each one, compares
the two grid sizes and writes a results index. The baseline settings live in
`src/config.py` and were fixed before any real run, so no baseline number can
be accused of being tuned to its outcome. The later runs (§7 and §8) were chosen
after seeing those results and are labelled post hoc. A full run takes roughly 3.5 hours on
Albert's Mac and can be stopped and restarted.

**Key concepts to understand**

- **Baseline vs. robustness.** γ = 5 on the 20% grid is the headline experiment.
  The other three each change one thing (the grid, or γ to 3 or 8), so any
  difference in results can be traced to that one change.
- **Fixed in advance.** Every setting below was chosen before results existed.
  Changing one later needs a written reason, never "it looked better."

**Key takeaways**

- Part IV (row 7 of the prospectus) has its convergence rule and settings fixed,
  and the full-scale run (`baseline-v1`) is done, with post-hoc runs in §7 and §8.
- Results land in `results/`. The small outputs are committed; the training
  logs stay local (see §4).

---

## 1. The settings

| Setting | Value | Status |
|---|---|---|
| Risk aversion γ | 3, 5, 8; baseline 5 | Team decision, 2026-10-16 |
| Episode length B | 20 trading days | Team decision |
| Convergence rule | Greedy action unchanged for K = 10% of training episodes. Primary reading: the final greedy action held for at least K episodes through the end of training (`greedy_stable_through_end`); "first time any action held for K" is reported as secondary | Team decision; strict reading confirmed 2026-10-16 |
| Seeds | 10, master seed 20261016 | Albert's proposal, adopted |
| Training budget | 13,000,000 episodes per seed (260 million simulated days) | Proposal, adopted |
| Exploration | ε₀ = 1, β set so ε = 1% halfway through training (β ≈ 7.08e-7) | Proposal, adopted |
| Learning rate | Sample average, α = 1/N(a); discount ρ = 0; Q starts at 0 | Settled earlier |
| \|ΔQ\| tolerance | 1e-5 per 10,000-episode window, for good | Proposal, adopted |
| Near-optimal line | 25 bps/yr of certainty-equivalent loss vs. best on the grid | Proposal, adopted |
| Evaluation sample | 10,000 episodes from the evaluation stream | Proposal, adopted |
| Grids | 20% (1,287 portfolios) baseline; 25% (495) comparison | Proposal, adopted |

With 13 million episodes, K = 1,300,000 episodes (26 million simulated days).

## 2. The experiments

| Folder | γ | Grid | Answers |
|---|---|---|---|
| `baseline_g5_grid20` | 5 | 20% | Part V weights, performance and convergence; Part IV counts |
| `resolution_g5_grid25` | 5 | 25% | Part V's second action-space resolution; Part VI action-space check |
| `robust_g3_grid20` | 3 | 20% | Part VI risk-aversion check |
| `robust_g8_grid20` | 8 | 20% | Part VI risk-aversion check |
| `comparison_g5_grid20_vs_grid25` | 5 | both | Convergence by resolution chart and table |

## 3. How to run

From the repo root, with the environment in `README.md`:

```bash
python -m pytest src/tests -q                   # everything passes first
python -m src.run_experiments --dry-run         # prints the plan; writes nothing
python -m src.run_experiments --episodes 200000 --runs 2 --results-dir results/smoke   # about a minute
python -m src.run_experiments                   # full scale, about 3.5 hours
```

- To run one experiment: `--only baseline_g5_grid20`.
- A results folder holds one set of settings. The first call writes
  `settings.json` there, and any later call with different settings is refused
  without touching anything, so a smoke test can never overwrite a full run.
  Use a separate `--results-dir` for each set of settings.
- If a run is interrupted, re-run the same command. Experiments that finished
  training with the same settings are reused. If the runner reports an
  experiment as "partly trained", delete that one experiment's folder and run
  again. If it reports "different settings", nothing is wrong with the folder:
  you passed different settings, so use another `--results-dir`.
- Timing: about 24 µs per episode on the 20% grid, so about 5 minutes per seed
  and 50 minutes per experiment.

## 4. Where results go, and how to look at them

```
results/
  INDEX.md                         one table: every experiment's key numbers, with links
  index.json                       the same, machine-readable, with settings and git commit
  baseline_g5_grid20/
    manifest.json                  config, seeds, data/code/grid fingerprints
    run_000.npz … run_009.npz      training logs, one per seed
    evaluation/
      summary.json                 every number in EVALUATION.md §2
      weights.csv  performance.csv  per_seed.csv
      weights_grouped_bar.png  learning_curve.png  cumulative_regret.png  ce_loss_by_seed.png
  …
  comparison_g5_grid20_vs_grid25/
    comparison.json  convergence_by_resolution.png  eval_25%/  eval_20%/
```

Start with `results/INDEX.md`, then open an experiment's `evaluation/` folder.
Definitions of every number are in `EVALUATION.md`.

**Sharing.** Adopted 2026-10-19 (`.gitignore`, results block). Git tracks
`INDEX.md`, `index.json`, `settings.json`, each experiment's `manifest.json`
and every `evaluation/` folder, well under 1 MB per experiment. The `.npz`
training logs (about 2.5 MB per seed, so about 100 MB for all four
experiments) stay local at any folder depth; the same settings and seeds
regenerate them. Anything under `results/smoke*/` stays local entirely, so
put every reduced-settings run there. Commit results with
`git add results/`, never `git add -A`, and only after a full-scale run
(`INDEX.md` says "Full-scale settings").

**Later runs and their reports.** Every run after `baseline-v1` trains into its own `--results-dir results/<run_id>`, is registered in `reports/registry.json`, extracted once with `python -m src.reporting.extract <run_id>` on the machine that ran it, and gets a report from `python -m src.reporting.build <run_id>` that compares it with every earlier run. The checklist, including writing the expected direction here before training, is in `reports/README.md`.

## 5. Before reading results

- Run the smoke test to check that the pipeline runs. Its numbers come from
  a tiny budget and say nothing about the agent.
- Record in the report the date, the git commit (written into `INDEX.md`) and
  that every setting came from `src/config.py` unchanged.
- The near-optimal line, K and the |ΔQ| tolerance were fixed before results.
  Report results against them as they are.
- Disclose in the report that an illustrative demo (γ = 5, B = 20, 2 million
  episodes, `EVALUATION.md` §6) ran before the settings were fixed.

## 6. Design notes

- **Shared random streams.** Every experiment uses the same master seed, so
  seed i sees identical simulated returns in all four experiments, and all are
  scored on the same evaluation sample. Exploration draws match only between
  experiments on the same grid, because the random action is drawn from the
  grid's size (`RUN_ISSUES_2026-10-22.md` Finding 4). Differences
  between experiments therefore come from the setting that changed, not from
  luck.
- **The near-optimal line across γ.** 25 bps/yr is 3.5%, 4.8% and 6.7% of the
  best grid portfolio's value at γ = 3, 5 and 8, so it is a tighter bar at
  γ = 3 than at γ = 8 (`EVALUATION.md` §7).

## 7. Post-hoc runs (registered 2026-10-27, run 2026-10-28 to 2026-10-29)

**TL;DR / In plain terms:** after the baseline run we planned five more runs
that change the episode length, the training budget, the exploration schedule
or the grid. They were chosen after seeing results, so the report must label
them post hoc. Each one's settings, expected result, command and run time are
written here before it runs, so the expectation cannot be fitted to the
outcome afterwards.

**Key concepts to understand**

- **Post hoc.** A run designed after looking at results can be cherry-picked
  without anyone meaning to. Writing down the expected direction first, and
  reporting the run whichever way it comes out, is the standard defence.
- **One variant per results folder.** Each run trains all three γ on its own
  settings into its own `results/<run_id>/`, and the folder's `settings.json`
  refuses anything else, so a variant can never mix with the baseline.
- **Random tries per portfolio.** With ε_t = exp(−βt), the expected number of
  random (exploration) episodes over N episodes is (1 − e^(−βN))/β. Divided by
  the number of portfolios, it says how often an untried portfolio gets
  sampled. The baseline gives about 1,100 per portfolio, while telling the top
  two portfolios apart needs about 1.1 million (`REVIEW_HISTORY.md` §7.1).

**Key takeaways**

- Running without `--run` is unchanged and reproduces `baseline-v1`:
  `test_baseline_settings_still_match_the_committed_baseline_v1_run` checks the
  settings and every training config against the files that run wrote.
- The five runs are numbered in the order they were planned: run 1 `b63`
  (§7.3), run 2 `n26m` (§7.4), run 3 `eps-decay` (§7.5), run 4 `all-three`
  (§7.6) and run 5 `grid10-robustness` (§7.7). Each was built on its own
  branch (`run/<run_id>`) and all five were merged into `dev` on 2026-10-27.
- All five have run. Results are in `results/<run_id>/` and reports in
  `reports/<run_id>/report.html`. Runs 1 to 4 are checked against their
  expectations in `REVIEW_HISTORY.md` §8; run 5 in §7.7 below.
- `RunVariant.expected` in `src/config.py` is the one-line summary printed in
  `INDEX.md`; the section below is the full pre-registered expectation.

### 7.1 How a variant is defined and run

A `RunVariant` in `src/config.py` names what changes (B, training episodes,
ε at halfway, grid) and keeps every other setting at the baseline's. Its
experiments are `baseline_g5_<grid>_<suffix>`, `robust_g3_<grid>_<suffix>` and
`robust_g8_<grid>_<suffix>`. The runner adds `post_hoc_run` and
`epsilon_at_halfway` to `settings.json` and writes the registration date and
expected direction into `INDEX.md`.

```bash
python -m src.run_experiments --run <run_id> --dry-run                      # the plan, nothing written
python -m src.run_experiments --run <run_id> --episodes 200000 --runs 2 --results-dir results/smoke_<run_id>
python -m src.run_experiments --run <run_id> --results-dir results/<run_id>   # full scale
```

Assumptions that apply to every variant:

- **Same seeds.** Master seed 20261016, 10 seeds, as in the baseline. Seed i
  starts from the same random streams, but a different B or ε uses them
  differently, so trajectories are not paired draw for draw with the baseline.
- **ε is defined at halfway of the variant's own budget.** With 26M episodes,
  "1% at halfway" means ε = 1% at episode 13M, so β halves.
- **K stays 10% of the training episodes.** With 26M episodes K is 2.6M, so
  "converged" is a stricter test in absolute episodes than in the baseline.
- **The discretization gap does not depend on B, N or ε.** It is computed
  from μ and Σ on the grid. Only the 10% grid run can change it.
- **No resolution comparison.** Each variant uses one grid, so the runner's
  grid-comparison folder is not written.

### 7.2 Run-time basis

Measured on Albert's Mac: the baseline took 281 to 323 s per seed (mean
about 313 s, 24 µs per episode) at B = 20 on the 20% grid with 13M episodes, so about
2.5 hours for 30 seeds (3 γ × 10). Scaling rules used below:

- **Episode length.** B = 63 measured 43.8 µs vs 38.2 µs at B = 20 in
  William's timing, about 0.13 µs per extra simulated day. B = 63 adds about
  5.6 µs per episode, so +15% (William's measured ratio) to +24% (the same
  5.6 µs added to the baseline's 23.6 µs). Estimates below use the range.
- **Training budget.** Time is proportional to the number of episodes.
- **ε.** No effect on speed worth estimating.
- **10% grid.** 91 µs per episode with the current runner and 51.6 µs for the
  bare loop (`REVIEW_HISTORY.md` §4). Timed on Albert's Mac on 2026-10-27,
  the full Q-table copy after every episode (§5.3) is only about 6 µs of that
  difference; `train.py`'s per-episode `np.argmax` over 43,758 entries is
  about 36 µs, and the agent's own greedy scan about 42 µs.

Evaluation adds a few minutes per run on the 20% grid. Estimates are ±20%.

### 7.3 Run `b63`: episode length B = 63 (post hoc)

**TL;DR / In plain terms:** each training episode scores a portfolio on 63
simulated days (about a quarter of a trading year) instead of 20. A longer batch gives a less noisy score for
every portfolio at once, so the agent should be better at telling the top
few apart. Nothing else changes.

**Key concepts to understand**

- **Why B matters.** One episode's reward is the sample mean minus γ/2 times
  the sample variance of 20 daily returns. Its standard deviation falls
  roughly with √B, so B = 63 cuts the noise of every reward by about √(63/20) ≈ 1.8
  without needing any extra episodes.
- **Why B does not touch the benchmark.** μ and Σ are daily, and the reward is
  a daily mean-variance score at any B, so the best grid portfolio and the
  discretization gap are the same as in the baseline.

**Key takeaways**

- This is the lever William's surrogate study found strongest (`REVIEW_HISTORY.md`
  §7.2). It is the Part VI "episode lengths" robustness check the assignment
  lists.
- William proposed B = 63 and B = 252 (`REVIEW_NOTES_FOR_WILLIAM.md`, "To
  decide with Albert"). Albert first registered B = 60 and changed it to
  B = 63 on 2026-10-27, before anything ran, to match William's proposal and
  the B = 63 surrogate estimate. B = 252 is not registered here.

| Setting | Baseline (`baseline-v1`) | This run |
|---|---|---|
| Episode length B | 20 | **63** |
| Training episodes per seed | 13,000,000 | 13,000,000 (819M simulated days) |
| ε at halfway | 1% (β ≈ 7.08e-7) | 1% (same β) |
| Grid | 20% (1,287) | 20% (1,287) |
| K (convergence) | 1,300,000 episodes | 1,300,000 episodes |
| γ | 3, 5, 8 | 3, 5, 8 |
| Seeds | 10, master seed 20261016 | same |

**Expected direction, written before running.** At every γ the learning gap
against the best grid portfolio shrinks. The surrogate at γ = 5 with B = 63
gave a mean gap of 7.6 vs 9.3 bps/yr and an unchanged exactly-best share
(44%), so with 10 seeds the change in exactly-best share is expected to be
within noise. Training regret about unchanged (59 vs 62 in the surrogate).
Discretization gap identical to the baseline (2.5, 5.7, 6.1 bps/yr at γ = 3,
5, 8).

**Command.**

```bash
python -m src.run_experiments --run b63 --dry-run
python -m src.run_experiments --run b63 --results-dir results/b63
```

**Estimated time.** 345 to 375 s per seed, so about 3 hours for 30 seeds
(3 γ × 10), plus a few minutes of evaluation.

### 7.4 Run `n26m`: 26 million training episodes per seed (post hoc)

**TL;DR / In plain terms:** the agent trains for twice as many episodes,
26 million per seed instead of 13 million. The exploration schedule is
stretched to match, so ε still reaches 1% halfway through, now at episode
13 million. Every portfolio gets about twice as many random tries.

**Key concepts to understand**

- **Stretched schedule.** β is set from the budget: β = ln(100)/(N/2). At
  N = 26M, β halves (≈ 3.54e-7), so the shape of ε over the fraction of
  training is identical to the baseline and the share of random episodes is
  the same 10.9%.
- **Random tries per portfolio.** (1 − e^(−βN))/β / 1,287 is 1,097 in the
  baseline and 2,193 here. Separating the top two portfolios needs about
  1.1 million visits each (`REVIEW_HISTORY.md` §7.1), so doubling random
  tries alone is still about 500 times short. Any gain has to come from the
  longer exploitation phase revisiting the top contenders.
- **Convergence threshold.** K is 10% of the budget, so it doubles to
  2,600,000 episodes. The convergence share is therefore measured against a
  stricter absolute bar than the baseline's.

**Key takeaways**

- This is the Part VI "different training budgets" robustness check the
  assignment lists.
- Albert confirmed on 2026-10-27 that "double up to 26M" means training
  episodes, with B left at 20 (B = 63 is run `b63`).

| Setting | Baseline (`baseline-v1`) | This run |
|---|---|---|
| Training episodes per seed | 13,000,000 | **26,000,000** (520M simulated days) |
| Episode length B | 20 | 20 |
| ε at halfway | 1% (β ≈ 7.08e-7) | 1% of the new budget, at episode 13M (β ≈ 3.54e-7) |
| Grid | 20% (1,287) | 20% (1,287) |
| K (convergence) | 1,300,000 episodes | **2,600,000 episodes** |
| γ | 3, 5, 8 | 3, 5, 8 |
| Seeds | 10, master seed 20261016 | same |

**Expected direction, written before running.** A slightly smaller learning
gap and a slightly higher exactly-best share at every γ, likely within noise
for 10 seeds, because the extra uniform tries are far short of what
separating the top portfolios needs. Training regret per episode (bps/yr)
about unchanged, since the exploration share is the same. Discretization gap
identical to the baseline.

**Command.**

```bash
python -m src.run_experiments --run n26m --dry-run
python -m src.run_experiments --run n26m --results-dir results/n26m
```

**Estimated time.** About 600 s per seed, so about 5 hours for 30 seeds,
plus a few minutes of evaluation. Training logs are about 5 MB per seed
(twice the checkpoints).

### 7.5 Run `eps-decay`: slower exploration decay, ε = 3% at halfway (post hoc)

**TL;DR / In plain terms:** the agent keeps exploring a little longer. ε
falls to 3% halfway through training instead of 1%, which means about 31%
more random episodes in total. Everything else is the baseline.

**Key concepts to understand**

- **The schedule.** ε_t = exp(−βt) with β = ln(1/0.03)/(N/2) ≈ 5.39e-7
  (baseline 7.08e-7). ε ends training at 0.09% instead of 0.01%.
- **Random tries per portfolio.** 1,439 instead of 1,097. These are about
  760 and 1,000 times fewer than the 1.1 million visits needed to separate the top two
  portfolios (`REVIEW_HISTORY.md` §7.1).
- **Regret is the price.** Every random episode is mostly spent on a poor
  portfolio, so more exploration raises the cumulative regret even if it
  does not improve the final choice.

**Key takeaways**

- William's surrogate study (`REVIEW_HISTORY.md` §7.2, 100 seeds at γ = 5)
  found ε = 3% at halfway gave a 9.5 bps/yr mean learning gap vs 9.3, an
  exactly-best share of 36% vs 44% (±5 points of sampling error), and
  training regret of 77 vs 62 bps/yr. William's decision of 2026-10-26 was to
  drop slower decay. Albert chose to run 3% anyway on 2026-10-27 as the
  mildest tested change. Report it as a post-hoc check of that surrogate
  result.

| Setting | Baseline (`baseline-v1`) | This run |
|---|---|---|
| ε at halfway | 1% (β ≈ 7.08e-7) | **3%** (β ≈ 5.39e-7) |
| Episode length B | 20 | 20 |
| Training episodes per seed | 13,000,000 | 13,000,000 |
| Grid | 20% (1,287) | 20% (1,287) |
| K (convergence) | 1,300,000 episodes | 1,300,000 episodes |
| γ | 3, 5, 8 | 3, 5, 8 |
| Seeds | 10, master seed 20261016 | same |

**Expected direction, written before running.** Learning gap and
exactly-best share unchanged within noise at every γ. Training regret higher,
about 15 bps/yr at γ = 5 per the surrogate. Discretization gap identical to
the baseline.

**Command.**

```bash
python -m src.run_experiments --run eps-decay --dry-run
python -m src.run_experiments --run eps-decay --results-dir results/eps-decay
```

**Estimated time.** About 300 s per seed, so about 2.5 hours for 30 seeds,
plus a few minutes of evaluation.

### 7.6 Run `all-three`: B = 63, 26M episodes and ε = 3% at halfway together (post hoc)

**TL;DR / In plain terms:** this run applies all three changes at once:
63-day episodes, 26 million episodes per seed, and ε falling to 3% at
halfway (episode 13 million). It shows how much the changes give together,
compared with runs `b63`, `n26m` and `eps-decay`, which each change one thing.

**Key concepts to understand**

- **Combined versus one at a time.** The single-change runs say which lever
  moves the result. This run says how far the levers go together. Its
  difference from the baseline cannot be split between the three changes
  without the single-change runs.
- **Exploration.** β = ln(1/0.03)/13M ≈ 2.70e-7. Random tries per portfolio
  are 2,878, against 1,097 in the baseline.

**Key takeaways**

- Run 5 (`grid10-robustness`) uses exactly these settings on the 10% grid, so
  the difference between run 5 and this run is the grid alone.
- K doubles to 2,600,000 episodes with the budget.

| Setting | Baseline (`baseline-v1`) | This run |
|---|---|---|
| Episode length B | 20 | **63** |
| Training episodes per seed | 13,000,000 | **26,000,000** (1.64B simulated days) |
| ε at halfway | 1% (β ≈ 7.08e-7) | **3%** of the new budget, at episode 13M (β ≈ 2.70e-7) |
| Grid | 20% (1,287) | 20% (1,287) |
| K (convergence) | 1,300,000 episodes | **2,600,000 episodes** |
| γ | 3, 5, 8 | 3, 5, 8 |
| Seeds | 10, master seed 20261016 | same |

**Expected direction, written before running.** The smallest learning gap
of runs 1 to 4 at every γ, mostly from the longer episodes; at γ = 5 a mean
gap below `b63`'s. Exactly-best share at or above the baseline's. Training
regret slightly above `b63`'s because of the slower ε decay. Discretization
gap identical to the baseline.

**Command.**

```bash
python -m src.run_experiments --run all-three --dry-run
python -m src.run_experiments --run all-three --results-dir results/all-three
```

**Estimated time.** 690 to 745 s per seed, so about 6 hours for 30 seeds,
plus a few minutes of evaluation.

### 7.7 Run `grid10-robustness`: the 10% grid at run `all-three`'s settings (post hoc)

**TL;DR / In plain terms:** the agent picks from a finer menu of 43,758
portfolios (weights in steps of 10%) instead of 1,287. Every other setting
matches run `all-three` (B = 63, 26M episodes, ε = 3% at halfway), so the
difference between the two runs is the grid alone. This is the Part VI
action-space robustness check. A finer menu contains a better portfolio, but
the agent has 34 times as many portfolios to try with the same budget.

**Key concepts to understand**

- **Two gaps move in opposite directions.** The discretization gap (best grid
  portfolio vs the continuous optimum) can only shrink, because the 20% grid
  is a subset of the 10% grid. The learning gap (learned vs best grid
  portfolio) should grow, because the budget is spread over far more
  portfolios and more of them sit close to the best.
- **Near-ties.** On the 10% grid many portfolios are almost as good as the
  best, so "exactly the best" stops being a meaningful target. Computed from
  μ and Σ for this section only (the agent never sees them):

  | γ | Discretization gap, 20% → 10% (bps/yr) | 2nd best behind best, 10% grid (bps/yr) | Portfolios within 25 bps, 20% → 10% |
  |---|---|---|---|
  | 3 | 2.5 → 2.5 (same best portfolio) | 0.04 | 2 → 5 |
  | 5 | 5.7 → 1.3 | 4.45 | 4 → 25 |
  | 8 | 6.1 → 6.1 (same best portfolio) | 1.25 | 2 → 36 |

  At γ = 3 and 8 the finer grid does not find a better portfolio at all, so
  the check can only show what the extra choices cost the learner.
- **Discussion of ε decay for this run.** The expected number of random
  (exploration) episodes is (1 − e^(−βN))/β, and dividing by the number of
  portfolios gives the random tries each one gets:

  | Schedule | Budget | Random episodes | Random tries per portfolio, 10% grid (20% grid) |
  |---|---|---|---|
  | ε 1% at halfway (baseline) | 13M | 1.41M | 32 (1,097) |
  | ε 1% at halfway | 26M | 2.82M | 65 (2,193) |
  | **ε 3% at halfway (this run)** | **26M** | **3.70M** | **85** (2,878) |
  | ε 10% at halfway | 26M | 5.59M | 128 (4,343) |

  `test_random_tries_per_portfolio_quoted_in_experiments_md` re-derives these
  by summing the schedule episode by episode. With 85 tries, one portfolio's
  average reward still has a standard error of roughly 370/√85 ≈ 40 times the
  10.5 bps/yr gap that separates the top two on the 20% grid (`REVIEW_HISTORY.md`
  §7.1, B = 20; B = 63 cuts that by about 1.8). Random tries therefore only
  screen out clearly bad portfolios; the choice among the near-best is made
  by exploitation, as on the 20% grid. A slower decay (10% at halfway) would
  add 43 tries per portfolio at a cost of about 1.9M more random episodes,
  most of them spent on poor portfolios, and William's surrogate study on the
  20% grid found no gain beyond noise from that change (§7.2 of
  `REVIEW_HISTORY.md`). We keep 3% so that runs `all-three` and
  `grid10-robustness` differ only in the grid. Choosing ε for the 10% grid by
  how this run turns out would be tuning on results, so it is not planned.

**Key takeaways**

- Report this against run `all-three`. Against the baseline it changes four
  things at once.
- Seed i of this run and seed i of `all-three` draw the same simulated
  returns, because the returns stream does not depend on the actions taken.
  The comparison is therefore paired seed by seed, and `evaluate.py` can
  compare the two directly since their training configs differ only in the
  grid:
  `python -m src.evaluate --compare results/all-three/baseline_g5_grid20_B63_N26M_eps3 results/grid10-robustness/baseline_g5_grid10_B63_N26M_eps3 --labels 20% 10% --out results/grid10-robustness/comparison_g5_grid20_vs_grid10 --eval-episodes 10000 --converge-k 2600000 --dq-tol 1e-5 --near-optimal-bps 25`.
- William proposed running the 10% grid at baseline settings (B = 20, 13M)
  so it changes one thing from `baseline-v1` (`REVIEW_NOTES_FOR_WILLIAM.md`).
  Albert chose `all-three`'s settings on 2026-10-27 so the comparison is with
  the configuration expected to learn best.

| Setting | Baseline (`baseline-v1`) | Run `all-three` | This run |
|---|---|---|---|
| Grid | 20% (1,287) | 20% (1,287) | **10% (43,758)** |
| Episode length B | 20 | 63 | 63 |
| Training episodes per seed | 13,000,000 | 26,000,000 | 26,000,000 |
| ε at halfway | 1% (β ≈ 7.08e-7) | 3% (β ≈ 2.70e-7) | 3% (β ≈ 2.70e-7) |
| K (convergence) | 1,300,000 | 2,600,000 | 2,600,000 |
| γ | 3, 5, 8 | 3, 5, 8 | 3, 5, 8 |
| Seeds | 10, master seed 20261016 | same | same |

**Expected direction, written before running.** At every γ, a larger
learning gap against the best grid portfolio than run `all-three`'s, an
exactly-best share near zero, and fewer seeds meeting the primary
convergence rule. Discretization gap 1.3 instead of 5.7 bps/yr at γ = 5 and
unchanged at γ = 3 and 8. The loss against the continuous optimum (the sum of
the two gaps) has no confident direction: at γ = 5 the 4.4 bps/yr the grid
gains may or may not outweigh the larger learning gap.

**Code changes for this run.**

- `src/train.py`: `MAX_SNAPSHOT_VALUES` raised from 100M to 120M. This run
  logs 2,600 checkpoints × 43,758 portfolios = 113.8M values per array. The
  alternative, logging every 20,000 episodes, would change the |ΔQ| window
  compared with run `all-three`.
- `src/train.py`: Albert's half of the Q-table copy fix
  (`REVIEW_HISTORY.md` §5.3). When `update()` returns the new Q(action),
  `train.py` updates its own copy of the table at one entry and checks it
  against `agent.q_values` at every checkpoint. `TabularQAgent.update()`
  returns nothing today, so until William's half lands the runner copies the
  full table every episode, as before. Tests show identical logs either way,
  with both a stand-in agent and `TabularQAgent`.

**Command.**

```bash
python -m src.run_experiments --run grid10-robustness --dry-run
python -m src.run_experiments --run grid10-robustness --results-dir results/grid10-robustness
```

**Estimated time and resources.** Per episode at the 10% grid: 91 µs with the
current runner, plus 15% or 5.6 µs for B = 63, so about 97 to 105 µs.

- As the code stands: about 2,500 to 2,700 s per seed, so 21 to 23 hours for
  30 seeds.
- With William's half of the copy fix, about 6 µs per episode less: 20 to
  22 hours. A first estimate of 12 to 15 hours assumed the copy was the whole
  91 vs 51.6 µs gap; the independent review timed it and found it is not.
- Replacing `train.py`'s per-episode `np.argmax` with exact incremental
  tracking of the logged greedy action would save about 36 µs more (about
  14 to 16 hours in total). Not done; it is a proposed follow-up.
- Memory: about 910 MB per seed for the logged arrays, on a 16 GB machine.
- Disk: the training logs are not committed. Scaling the baseline's
  compression gives about 160 MB per seed, so about 5 GB for 30 seeds; allow
  up to about 14 GB in case the larger Q snapshots compress worse.
  Evaluation reads one seed at a time.

Run it in three parts if an overnight run is not possible:
`--only baseline_g5_grid10_B63_N26M_eps3` and so on, into the same results
folder. Finished experiments are reused.

**How it actually ran (2026-10-29).** The three γ trained at the same time,
as three `--only` processes into `results/grid10-robustness`, followed by one
plain `--run grid10-robustness` pass that reused all three and wrote the
index. Seeds are deterministic, so this gives the same results as running
them one after another. Training took 2,677 to 3,075 s per seed and about
7.7 hours in all, at commit `e13a9a6` (same `src/` as `all-three`'s `81814e6`).
William's `update()` fix was not in. The training logs take 12 GB.

#### Outcome, written after the run (2026-10-29)

**TL;DR / In plain terms:** on the finer grid the agent misses the grid's
best portfolio by more at every γ. At γ = 5 that roughly cancels the better
portfolio the finer grid offers, so the total loss is about unchanged. At
γ = 3 and 8 the finer grid has no better portfolio, so the total loss rises.
The full numbers and caveats are in the `grid10-robustness` entry of
`reports/registry.json` and in `reports/grid10-robustness/report.html`.

| γ | Discretization gap, 20% → 10% (bps/yr) | Mean learning gap, `all-three` → this run | Exactly best (of 10) | Total loss vs continuous, mean (paired difference, SE) | Primary rule met |
|---|---|---|---|---|---|
| 5 | 5.7 → 1.3 | 7.9 → 11.7 | 5 → 0 | 13.6 → 13.0 (−0.6, 3.7) | 10 → 0 |
| 3 | 2.5 → 2.5 | 4.0 → 22.0 | 8 → 4 | 6.5 → 24.5 (+18.0, 7.5) | 10 → 0 |
| 8 | 6.1 → 6.1 | 1.1 → 5.8 | 9 → 1 | 7.3 → 11.9 (+4.6, 1.6) | 10 → 0 |

The paired comparisons are in
`results/grid10-robustness/comparison_g{3,5,8}_grid20_vs_grid10/`, made with
the command above, extended to γ = 3 and 8.

**Key concepts to understand**

- **Too few tries per portfolio.** Each portfolio gets about 85 random tries.
  At γ = 5 the grid's best portfolio was tried a median of 116 times per seed,
  and each seed's final pick 19M to 22M times. A portfolio whose early
  estimate comes out low is rarely chosen again.
- **Why no seed met the convergence rule.** The greedy portfolio changed a
  median of 155k to 188k times per seed, against about 2,500 to 3,000 in
  `all-three`. With 43,758 noisy estimates, a poor portfolio regularly looks
  best by chance for a few hundred episodes. Over 99% of these detours went
  to a portfolio more than 25 bps/yr from the best.

**Against the expectation written before the run**

- Larger learning gap than `all-three` at every γ: held.
- Discretization gap 1.3 at γ = 5 and unchanged at γ = 3 and 8: held exactly.
- Fewer seeds meeting the primary rule: held (none at any γ).
- Exactly-best share near zero: held at γ = 5 (0 of 10) and γ = 8 (1 of 10),
  failed at γ = 3 (4 of 10). At γ = 3 only 4 rivals lie within 25 bps/yr of
  the best, and the seeds split: four on the best, five more than 25 bps/yr
  away.
- Total loss: no direction was predicted. It is about unchanged at γ = 5 and
  worse at γ = 3 and 8.

**Key takeaways**

- This is the Part VI action-space check. Report it against `all-three`,
  labelled post hoc.
- With this budget a finer grid costs tabular Q-learning more in learning
  than it gains in discretization. The run did not test a larger budget on
  the 10% grid, so it cannot say how much more training would close the gap.

---

## 8. Constant-learning-rate runs (registered 2026-11-05, run 2026-11-06, post hoc)

**TL;DR / In plain terms:** three more runs, all with a constant learning rate α instead of the baseline's α = 1/N(a). The professor agreed to a constant α for function approximation (FA) and a separately tuned constant α for the tabular comparison.
- `tabular-const` is the baseline agent with its own tuned constant α.
- `fa-quadratic` is the FA agent handed the true functional form: the reference model.
- `fa-spline` is the FA agent with generic smooth features: the main model.

Every α (and the spline l2) was chosen on separate tuning seeds before these runs (§8.1). These runs were designed after the baseline results were seen, so the report labels them post hoc.

**Key concepts to understand**

- **Why compare with `tabular-const`, not only `baseline-v1`.** `fa-quadratic` and `tabular-const` have the same α (3.2e-4), seeds and settings. They differ mainly in how a reward is shared across portfolios: one-hot features (tabular) or 45 quadratic features. With one-hot features the FA agent gives the tabular Q-values (`src/tests/test_fa_agent.py`), so this pair isolates what the features do.
  - One mechanical difference remains: FA refreshes its greedy portfolio every 100 updates, tabular every episode. Both break ties at random when acting, and both log the lowest-index argmax for the convergence rules. (Corrected 2026-11-06: an earlier version said tabular breaks ties to the lowest index when acting; `src/agent.py` `greedy_action` breaks them at random.)
  - Against `baseline-v1`, the learning rate also differs.
- **A constant α never averages the noise away.** Each Q-value keeps a noise floor of about σ·√(α/2) for tabular. With σ ≈ 1.55e-3 per episode at B = 20, α = 3.2e-4 gives about 2.0e-5 per day, or 49 bps/yr. The best grid portfolio leads the second by 20.0, 10.5 and 11.5 bps/yr at γ = 3, 5 and 8.
- **How each expectation is read.**
  - "Learning gap" means the mean over the 10 seeds, as in `REVIEW_HISTORY.md` §8.
  - "Q-value fit" means the median over seeds of the Spearman correlation and the top-10 recall over all portfolios (`q_fit_all`, `EVALUATION.md` §2).
  - The fit metrics are in each experiment's `evaluation/summary.json`, `INDEX.md` and, since 2026-11-09, the HTML reports (a report extracted before then shows them as n/a until re-extracted).
- **Tuned at γ = 5 only.** The same α and l2 are used at γ = 3 and 8.

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V. These runs go in the FA extension and in the Part III discussion of the learning-rate rule ("whether it changes with action visits").
- The quadratic is the correctly specified reference. Only `fa-spline` can speak to whether the agent finds the structure without being told it, and no such claim is made before its results.
- Evaluate all three with the Q-value fit metrics (`EVALUATION.md` §2), and compare with `baseline-v1` re-evaluated the same way.

### 8.1 Tuning (done 2026-11-04 to 2026-11-05)

`python -m src.tune_fa --all --jobs 4` on master seed 20261103 (never the reported 20261016). The procedure:
- each candidate α (8 fractions of each model's stability bound) trained in full on 3 seeds;
- each seed's final portfolio scored by realized reward on 200,000 fresh evaluation episodes;
- for each model, the candidate with the highest mean kept;
- if the runner-up was within one seed-paired standard error, the top 3 trained on 3 more seeds and decided on all 6.

The rule was fixed before any full-scale result. Details are in `FA_INTEGRATION_NOTES.md` §5. Records and the scored table are in `results/tuning/` (`fa_tuning.json`, `run.log`, `extend.log`).

| Model | Chosen α | Fraction of bound | l2 | Decided on |
|---|---|---|---|---|
| Tabular (constant α) | 3.2e-4 | 1.6e-4 of 2 | — | 3 seeds, runner-up outside one SE |
| Quadratic (reference) | 3.2e-4 | 1.6e-4 of 2 | 0 | 3 seeds, runner-up outside one SE |
| Spline (main) | 7.11e-6 | 1.6e-4 of 2/45 | 0 | 6 seeds: a close call; l2 = 1e-4 led on seeds 0 to 2, l2 = 0 won over all 6 |

Two notes on the tuning:
- **A bug in the close-call check.** The spline's first summary called its runner-up "not close" because of a rounding difference of 1.2e-14. That was a bug in the check (fixed in commit `945a55e`), so the extra seeds ran as the rule requires.
- **The rule checks only the runner-up.** For the quadratic, the third-ranked candidate (fraction 8e-4) is within one seed SE of the best, but it is not the runner-up, so the rule did not extend it. The rule was applied as fixed rather than widened after seeing the table.

The 9 extra-seed records (spline seeds 3 to 5 of the three extended candidates) name commit `945a55e-dirty`, because the registration edits in this section's commit were uncommitted while they ran. None of those edits touch training or tuning: `src/tune_fa.py` reads only constants from `src/config.py`, and their values did not change.

The tuning numbers are tuning-seed scores. Each model's best is the maximum of 8 to 16 candidates, so it is optimistic, and none of these numbers is reported as a result.

### 8.2 Settings shared by the three runs

| Setting | `baseline-v1` | These runs |
|---|---|---|
| Learning rate | α = 1/N(a) | constant, tuned (§8.1) |
| Agent | tabular | tabular (`tabular-const`) or linear FA (`fa-quadratic`, `fa-spline`) |
| Greedy portfolio | recomputed every episode | FA: refreshed every 100 updates (`FA_INTEGRATION_NOTES.md` §3) |
| Episode length B | 20 | 20 |
| Training episodes per seed | 13,000,000 | 13,000,000 |
| ε at halfway | 1% | 1% |
| Grid | 20% (1,287) | 20% (1,287) |
| γ | 3, 5, 8 | 3, 5, 8 |
| Seeds | 10, master seed 20261016 | same |
| K (convergence) | 1,300,000 episodes | same |

### 8.3 Run `tabular-const` (post hoc)

The baseline agent with α = 3.2e-4 for every update. Experiments: `{baseline_g5, robust_g3, robust_g8}_grid20_TABconst`.

**Expected direction, written before running.**
- **Learning gap:** a larger mean than `baseline-v1`'s at every γ.
- **Primary convergence rule:** fewer seeds meet it than in `baseline-v1` at every γ (9, 8 and 9 of 10 at γ = 5, 3 and 8). The Q-value noise floor (about 49 bps/yr) is far above the gaps between the top portfolios, so the greedy portfolio keeps switching among near-ties.
- **Discretization gap:** identical to `baseline-v1`'s (2.5, 5.7 and 6.1 bps/yr at γ = 3, 5 and 8).

### 8.4 Run `fa-quadratic` (post hoc, reference model)

The FA agent with the 45 quadratic features w_i·w_j, α = 3.2e-4, l2 = 0 and greedy refresh every 100 updates. Experiments: `{baseline_g5, robust_g3, robust_g8}_grid20_FAquad`.

**Expected direction, written before running.**
- **Against `tabular-const`** (same α, seeds and settings), at every γ:
  - a smaller mean learning gap;
  - a higher Q-value fit: median Spearman correlation and top-10 recall over all portfolios.

  Every reward updates all 45 parameters, so the agent averages over far more data per parameter than per tabular cell. §4 of `FA_INTEGRATION_NOTES.md` puts its Q̂ noise at about 17 bps/yr at this α, and its slowest direction learns in about 5.5M of the 13M episodes.
- **Against `baseline-v1`:** no direction is predicted.
- **Discretization gap:** identical to `baseline-v1`'s.

### 8.5 Run `fa-spline` (post hoc, main model)

The FA agent with additive plus pairwise cubic B-spline features, α = 7.11e-6, l2 = 0 and greedy refresh every 100 updates. The features are 909 columns, of rank 369 on this grid. They are never told the reward is quadratic, though they can represent it (`src/features.py`). Experiments: `{baseline_g5, robust_g3, robust_g8}_grid20_FAspline`.

**Expected direction, written before running.**
- **Against `fa-quadratic`,** at every γ: a lower Q-value fit (median Spearman correlation and top-10 recall over all portfolios). `FA_INTEGRATION_NOTES.md` §4 puts the spline's Q̂ noise at about 39 bps/yr at this α. Its median direction needs about 3.8e8 episodes to learn, far more than 13M, so most of the spline's directions stay near their starting value of 0.
- **Learning gap against `fa-quadratic`:** no direction is predicted. The slow-learning argument points to a larger gap, since picking the best portfolio needs only the top of the landscape right. On the tuning seeds at γ = 5, though, the spline's final portfolios earned more than the quadratic's: 523 bps/yr over 6 seeds against 507 over 3. Those are tuning scores, each the best of its model's candidates, so they are optimistic. The evidence points both ways, so no direction is registered. (An earlier draft predicted a larger gap; it was changed before anything ran, after the independent review pointed out the tuning evidence.)
- **Against `tabular-const` and `baseline-v1`:** no direction is predicted.
- **Discretization gap:** identical to `baseline-v1`'s.

### 8.6 Commands and run time

```powershell
python -m src.run_experiments --run tabular-const --dry-run
python -m src.run_experiments --run tabular-const --results-dir results/tabular-const
python -m src.run_experiments --run fa-quadratic --results-dir results/fa-quadratic
python -m src.run_experiments --run fa-spline --results-dir results/fa-spline
```

These are the commands the runs were made with. `run_experiments` now sets UTF-8 output itself, so `PYTHONIOENCODING` is no longer needed on Windows (fixed 2026-11-09). In a fresh clone the folders above hold committed manifests without the training logs, so `run_experiments` refuses them; to reproduce a run, train into a new folder, for example `--results-dir results/repro_fa-quadratic`, and compare with the committed summaries. With the logs present, the same commands only re-run the evaluation.

The three are independent, so they can run as three processes at once. Before starting, time a short run of each with `--episodes 200000 --runs 1 --results-dir results/smoke_<run_id>`.

**Estimated time** comes from the tuning runs on William's machine, 4 at a time, per 13M-episode seed:
- tabular about 30 min;
- quadratic about 22 min;
- spline about 17 min.

The tabular figure is about 6× the 5 min per seed `baseline-v1` took alone on Albert's Mac. It is unexplained, so use the timing check above. For 30 seeds each, that is up to about 15, 11 and 8.5 hours sequentially. Evaluation adds a few minutes per experiment.

**Actual time** (from each manifest; all trained at commit `bc311a3`; the manifests do not record whether other jobs ran at the same time): `tabular-const` 9.9 to 15.2 min per seed (6.2 hours) and `fa-quadratic` 7.9 to 9.2 min (4.1 hours), both on William's machine; `fa-spline` 4.7 to 5.2 min (2.4 hours) on Albert's. All three came in under the tuning-based estimates; the tuning ran 4 jobs at once, which may explain the difference, but that is not measured.

### 8.7 Outcome, written after the runs (2026-11-06)

**TL;DR / In plain terms:** a constant learning rate made the tabular agent clearly worse than `baseline-v1`, as expected. Both FA models beat `tabular-const` at γ = 5 and 3 and ranked the whole reward landscape better at every γ (Spearman; top-10 recall tied at γ = 8), but neither beat `baseline-v1` reliably. At γ = 8 every FA seed and 6 of 10 `tabular-const` seeds picked a portfolio with too little SPY. `src/config.py` registers three expectation strings; split into five checks below, one held in full and four failed in whole or in part.

All numbers are from each experiment's `evaluation/summary.json`. Learning gaps are in bps/yr. Seeds are paired across runs (same master seed and returns stream), and the SE is of the paired mean over 10 seeds.

| γ | Run | Mean gap | Median (range) | Exactly best | Primary rule | Near-optimal | Spearman (all) | Top-10 recall |
|---|---|---|---|---|---|---|---|---|
| 5 | `baseline-v1` | 17.6 | 14.1 (0.0 to 47.3) | 3 | 9 | 7 | 0.821 | 35% |
| 5 | `tabular-const` | 43.5 | 42.0 (10.5 to 109.9) | 0 | 10 | 4 | 0.813 | 25% |
| 5 | `fa-quadratic` | 22.6 | 22.6 (15.9 to 29.4) | 0 | 2 | 5 | 0.915 | 55% |
| 5 | `fa-spline` | 15.7 | 5.3 (0.0 to 78.5) | 5 | 10 | 8 | 0.950 | 45% |
| 3 | `baseline-v1` | 30.6 | 10.0 (0.0 to 81.1) | 5 | 8 | 6 | 0.710 | 20% |
| 3 | `tabular-const` | 107.2 | 102.6 (0.0 to 205.2) | 2 | 8 | 2 | 0.703 | 15% |
| 3 | `fa-quadratic` | 8.0 | 0.0 (0.0 to 20.0) | 6 | 0 | 10 | 0.854 | 80% |
| 3 | `fa-spline` | 45.4 | 62.3 (0.0 to 68.3) | 3 | 10 | 3 | 0.945 | 65% |
| 8 | `baseline-v1` | 13.0 | 5.7 (0.0 to 53.1) | 5 | 9 | 7 | 0.920 | 20% |
| 8 | `tabular-const` | 43.3 | 51.5 (11.5 to 61.2) | 0 | 10 | 1 | 0.917 | 20% |
| 8 | `fa-quadratic` | 58.7 | 55.7 (55.7 to 70.8) | 0 | 9 | 0 | 0.951 | 20% |
| 8 | `fa-spline` | 54.6 | 51.6 (50.0 to 61.2) | 0 | 9 | 0 | 0.957 | 20% |

Counts are seeds out of 10. "Near-optimal" is the `near_optimal_for_good` rule (within 25 bps/yr from some point on). The discretization gap is identical to `baseline-v1`'s in every run, as expected.

Paired mean learning-gap differences (SE):

| Pair | γ = 5 | γ = 3 | γ = 8 |
|---|---|---|---|
| `tabular-const` − `baseline-v1` | +25.9 (11.5) | +76.6 (24.6) | +30.2 (8.0) |
| `fa-quadratic` − `tabular-const` | −20.9 (9.8) | −99.2 (23.7) | +15.4 (6.5) |
| `fa-spline` − `tabular-const` | −27.8 (11.9) | −61.8 (24.2) | +11.4 (5.1) |
| `fa-spline` − `fa-quadratic` | −6.9 (8.3) | +37.4 (10.2) | −4.1 (2.0) |
| `fa-quadratic` − `baseline-v1` | +5.0 (5.2) | −22.6 (12.8) | +45.7 (5.6) |
| `fa-spline` − `baseline-v1` | −1.8 (9.5) | +14.8 (17.3) | +41.6 (6.0) |

**Registered expectations against the results:**

| Run | Expectation (§8.3 to §8.5) | Outcome |
|---|---|---|
| `tabular-const` | Larger mean learning gap than `baseline-v1` at every γ | Met |
| `tabular-const` | Fewer seeds meeting the primary rule at every γ | Not met at any γ (10, 8, 10 against 9, 8, 9) |
| `fa-quadratic` | Smaller mean learning gap than `tabular-const` at every γ | Met at γ = 5 and 3; not at γ = 8 |
| `fa-quadratic` | Higher median Spearman and top-10 recall than `tabular-const` at every γ | Spearman met at all three; recall met at γ = 5 and 3, tied at γ = 8 |
| `fa-spline` | Lower median Spearman and top-10 recall than `fa-quadratic` at every γ | Not met: Spearman higher at every γ; recall lower at γ = 5 and 3, tied at γ = 8 |

**What the results say, and what they do not.**
- **Under a constant α the primary rule tracks stability and says little about accuracy.** `tabular-const` met it in 28 of 30 seeds while ending on the best portfolio in 2. The registered reason for predicting the opposite (a noise floor that keeps the greedy portfolio switching) did not hold. What is measured is shrinkage toward the starting value 0:
  - with a constant α and Q starting at 0, a portfolio updated N times holds about (1 − (1 − α)^N)·U of its true value U;
  - most portfolios are tried only during exploration, a median of about 1,100 times, which gives 0.296·U;
  - in `run_000`, regressing final Q on U over portfolios with fewer than 5,000 visits gives slopes of 0.291, 0.291 and 0.296 at γ = 5, 3 and 8, while the greedy portfolio sits at 1.06 to 1.14 of its U in `run_000` (0.69 to 1.14 across all 30 seeds).

  Every rival is shrunk about 70% toward 0. The inferred consequence, not tested by a counterfactual run, is that no rival can overtake the greedy portfolio once ε fades, so the agent locks in on whatever was greedy then; noise still decides which portfolio that is. The same shrinkage dominates `tabular-const`'s Q-fit and optimism numbers.
- **§8.4's speed estimate assumed uniform exploration for the whole budget.** "The slowest direction learns in about 5.5M of the 13M episodes" treats every episode as a uniform draw. Uniform exploration actually totals about 1/β ≈ 1.41M episodes, and at that count 36 of the 45 quadratic directions and 351 of the 369 spline directions have α·λ·1.41M < 1, so they never finish learning from θ = 0 (independent review, 2026-11-06).
- **The ΔQ rule is uninformative under a constant α.** Every seed of all three runs meets it in the first 10,000-episode window, because the largest single update in a window is 1e-6 to 5e-6 (`tabular-const` `run_000`), under the 1e-5 tolerance from the start (corrected 2026-11-10; an earlier draft said 5e-7, the size of a typical update).
- **γ = 8 is a systematic miss.** The best grid portfolio there is 40% SPY, 40% IEF, 20% GLD. All 20 FA seeds and 6 of 10 `tabular-const` seeds hold 20% SPY, against 1 of 10 in `baseline-v1`. `fa-quadratic` picks the same portfolio on 8 of 10 seeds, which points to a bias shared by every seed rather than noise. The suspected cause is the unfinished directions above. Solving the average (mean-field) update from θ = 0 over 1.41M uniform episodes predicts the spline's γ = 8 pick at a 50.0 bps/yr gap, against actual picks 50.0 to 61.2 off, and predicts the best portfolio at γ = 3 and 5 (reviewer's calculation, not re-checked, no committed script). For the quadratic, this exploration-only calculation is far off, so it explains the quadratic's miss only in part. Also untested: α was tuned at γ = 5 only.
- **No discovery claim.** `fa-spline` ranks the landscape at least as well as the correctly specified quadratic on Spearman, but its picks are not better than `baseline-v1`'s at any γ in a way the SEs support (−1.8 ± 9.5 at γ = 5). That is evidence the generic features can represent the landscape. It is no evidence that the agent found the optimum.
- **`baseline-v1`'s Q fit** (re-evaluated 2026-11-09, `AUDIT_2026-11-06.md` H3): every FA run ranks the landscape better than `baseline-v1` on Spearman at every γ; on top-10 recall `fa-quadratic` and `fa-spline` are higher at γ = 5 and 3 and tie at γ = 8. `tabular-const` is just below `baseline-v1` on Spearman at every γ.

### 8.8 Runs `fa-spline-grid10-eps3` and `fa-spline-grid10-eps10` (post hoc, registered 2026-11-11 before training)

**TL;DR / In plain terms:** the spline FA agent on the 10% grid (43,758 portfolios), where the tabular agent never met the primary convergence rule. Two runs with the same learning rate, differing only in how much the agent explores. `eps3` keeps `grid10-robustness`'s exploration, so it pairs with it seed by seed and asks whether shared weights beat a table on a large grid. `eps10` explores more (ε = 10% at halfway) and asks whether the FA agent's weak spot is too few random portfolios to learn from. The first registration (2026-11-10, William) used one run with `fa-spline`'s α unchanged; Albert replaced it with these two on 2026-11-11, before any training.

**Key concepts to understand**

- **Why FA should help on a large grid.** The table learns each of the 43,758 values separately. Uniform exploration gives each one about 85 tries (3.70M uniform episodes / 43,758). The spline model shares its 909 weights, so every reward moves the value of every portfolio.
- **Why exploration matters more for FA, not less.** A linear model learns along directions of its weights, and a direction is learned only from rewards of portfolios whose features vary along it. Greedy episodes repeat one portfolio and teach one direction; the variety comes from the ε-random episodes. A direction with eigenvalue λ of E[φφᵀ] over the uniform grid is roughly finished once α·λ·(uniform episodes) > 1.
- **How many directions finish.** Counting eigenvalues above 1e-12 of the largest (558 directions, the count the earlier registration used), α·λ·N < 1 for 504 at ε = 3% and 496 at ε = 10%. The count depends on the cutoff (486 directions at 1e-10, 378 at 1e-8), and it says nothing about which directions decide the ranking, so it is a rough guide only. On the 20% grid, unfinished directions are the suspected cause of every FA seed holding too little SPY at γ = 8 (§8.7).

**Settings.**
- **From `grid10-robustness`.** 10% grid, B = 63, 26M episodes, master seed 20261016 and 10 seeds per γ, at γ = 5, 3 and 8.
- **From `fa-spline`.** Additive plus pairwise cubic B-spline features (909 columns), l2 = 0, greedy refresh every 100 updates.
- **α = 2.778e-5, noise-matched, not re-tuned** (`config.ALPHA_SPLINE_GRID10`). `fa-spline`'s tuned 7.11e-6 (§8.1, tuned on the 20% grid at B = 20 and γ = 5) times 63/20, because the reward is a mean over B days and its noise variance falls as 1/B, times 1.240, the ratio of the mean ‖φ‖² over the grid (28.41 on the 20% grid, 22.92 on the 10% grid). This keeps the per-update noise the tuning chose on average over the grid. Most updates are greedy, and the best portfolios have ‖φ‖² of about 30 on both grids, so greedy updates get about 24% more noise than in the tuning (independent review, 2026-11-11); correcting for that would need μ and Σ. It uses the grid's features only, no μ or Σ. It is 6.3e-4 of the stability bound 2 / max‖φ‖² = 2/45 (max‖φ‖² = 45 on both grids). Re-tuning on this grid would take about 2 days and changes to `tune_fa.py`; it is not done, so the α is a reasoned choice and not the best one for this grid.
- **ε at halfway.** `eps3`: 3% (β = 2.697e-7, 3.70M uniform-equivalent episodes), as `grid10-robustness`. `eps10`: 10% (β = 1.771e-7, 5.59M).
- **Experiments.** `{baseline_g5, robust_g3, robust_g8}_grid10_B63_N26M_{eps3, eps10}_FAspline`.

**Disclosure.** William's mean-field calculation (`NEXT_STEPS_FOR_ALBERT.md`, 2026-11-10) predicts learning gaps of about 33.6 / 0 / 1.3 bps/yr at γ = 5 / 3 / 8 with ε at 3% and 4.4 / 0 / 1.3 with ε at 10%. It uses the true μ and Σ. Both levels are run, so neither was chosen by it. Two predictions below were informed by it: the γ = 5 direction for `eps10`, and `eps3`'s learning gap, where the first registration predicted the γ = 8 bias would persist and this one gives no direction.

**Expected direction, written before running.** A direction holds when the paired mean difference over the 10 seeds has the predicted sign and exceeds 2 SE; with the right sign but under 2 SE it is "not shown"; with the wrong sign beyond 2 SE it failed. Counts of seeds are read directly.

`fa-spline-grid10-eps3` against `grid10-robustness` (paired by seed):
- **Q-value fit.** Median Spearman correlation over all 43,758 portfolios of at least 0.9 at every γ (`grid10-robustness`: 0.543, 0.406 and 0.725; `fa-spline` on the 20% grid: 0.950, 0.945 and 0.957).
- **Primary rule.** At least 6 of 10 seeds meet it at every γ (`grid10-robustness`: 0 of 10 at each). A constant α settles the greedy portfolio, so this measures stability and says little about accuracy (§8.7). Neighbouring portfolios are closer in value on the finer grid, so it can fail.
- **Learning gap.** No direction at any γ (`grid10-robustness` means 11.7, 22.0 and 5.8 bps/yr at γ = 5, 3 and 8). For FA: shared information across portfolios. Against: most directions stay unfinished. At γ = 8 the 20% grid bias (seeds holding 20% SPY where the best holds 40%) may carry over; the larger α and 2.6× the uniform episodes of `fa-spline` push against it.
- **Discretization gap.** Identical (1.3, 2.5 and 6.1 bps/yr), since the grid, μ and Σ are the same.
- **No direction** for top-10 recall, the exactly-best share or training regret.

`fa-spline-grid10-eps10` against `fa-spline-grid10-eps3` (paired by seed; only ε differs):
- **Learning gap.** Smaller mean at γ = 5. No direction at γ = 3 and 8.
- **Training regret.** Higher at γ = 3 and 8, since about 1.9M more episodes are random (about +30 bps/yr per episode on average over the run from random episodes at γ = 5). No direction at γ = 5, where a smaller gap would make the greedy episodes cheaper and could offset that (independent review, 2026-11-11).
- **Q-value fit.** Higher median Spearman correlation at every γ, from more varied portfolios.
- Against `grid10-robustness`, descriptive only: it differs in learning rule, representation and exploration.

**What these runs cannot separate.** Against `grid10-robustness`, both change the learning rule (constant α against 1/N) and the representation (features against a table) at once. A tabular constant-α run on the 10% grid would separate them; it is not planned.

**Commands.** The six experiments in parallel, two folders dated by the training start, then one pass per folder that reuses them and writes the index. One BLAS thread per process is as fast as four here and avoids contention.

```bash
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for eps in eps3 eps10; do for g in baseline_g5 robust_g3 robust_g8; do
  d=results/fa-spline-grid10-${eps}_2026-11-11; mkdir -p $d
  caffeinate -i python -m src.run_experiments --run fa-spline-grid10-$eps --results-dir $d \
      --only ${g}_grid10_B63_N26M_${eps}_FAspline > $d/train_${g}.log 2>&1 &
  sleep 5   # staggered, so the first process writes settings.json before the others read it
done; done; wait
for eps in eps3 eps10; do python -m src.run_experiments --run fa-spline-grid10-$eps --results-dir results/fa-spline-grid10-${eps}_2026-11-11; done
```

**Estimated time and space (Albert's Mac, 10 cores, 16 GB).**
- **Time.** A 100,000-episode timing run (2026-11-11, one seed, `baseline_g5`, `eps3`) took 7.3 s of training, about 32 minutes per 26M-episode seed and 5.3 hours per experiment. With six processes sharing the machine, expect about 6 to 8 hours. William's 42 s per 100,000 episodes (2026-11-10) was on another machine with other jobs running.
- **Disk.** About 455 to 480 MB per seed (2,600 snapshots × 43,758 float32 values, which barely compress), so 4.5 to 4.8 GB per experiment and about 28 GB for both runs. 40 GB was free at registration.
- **Memory.** About 1.1 to 1.2 GB per process, peaking near 1.4 GB at each checkpoint (a float64 copy of the feature matrix), about 8.5 GB for six.
- **Caveat for the report.** The runs will show "Training code changed" against earlier runs for `src/action_space/grid.py`, `src/environment.py` and `src/train.py`. Since `fa-spline`, those edits are LF writes, the increment guard and provenance fields (`AUDIT_2026-11-06.md` §2.2 and §8); `environment.py` now resolves the processed-data path relative to the file instead of the working directory. Against `grid10-robustness`, `train.py` also gains the FA agent, which is the change under test. `grid_sha256` matches.

#### Outcome for `fa-spline-grid10-eps3`, written after the run (2026-11-11)

**TL;DR / In plain terms:** the FA agent learned the shape of the value landscape much better than the table did, but it picked worse portfolios at every γ, and on no seed was its final pick held for the required 2.6M episodes. It under-weights SPY on every seed. `fa-spline-grid10-eps10` (training) tests whether more exploration fixes this.

Paired by seed against `grid10-robustness` (10 seeds per γ; numbers from each experiment's `evaluation/per_seed.csv`):

| | γ = 5 | γ = 3 | γ = 8 |
|---|---|---|---|
| Mean learning gap, FA / table (bps/yr) | 78.8 / 11.7 | 140.6 / 22.0 | 59.6 / 5.8 |
| Paired difference (SE) | +67.1 (9.1) | +118.6 (12.2) | +53.8 (9.8) |
| Median Spearman, FA / table | 0.976 / 0.543 | 0.963 / 0.406 | 0.983 / 0.725 |
| Primary rule met, FA / table | 0 / 0 of 10 | 0 / 0 | 0 / 0 |
| SPY in FA picks (best grid portfolio) | 20% to 30% (50%) | 20% to 40% (60%) | 20% to 30% (40%) |
| Training regret per episode, FA / table (bps/yr) | 122.7 / 78.8 | 160.0 / 91.3 | 126.5 / 88.1 |

Registered expectations:
- **Spearman of at least 0.9 at every γ: held** (lowest seed 0.916).
- **At least 6 of 10 seeds meeting the primary rule at every γ: failed** (0 of 10 at every γ). Unlike `fa-spline` and `tabular-const` on the 20% grid, a constant α did not settle the greedy portfolio here; the post-hoc settled point has a median of 1,557M to 1,601M of 1,638M simulated days.
- **Discretization gap identical: held** (1.3, 2.5 and 6.1 bps/yr).
- **No direction** was registered for the learning gap, top-10 recall, the exactly-best share or regret. The gap and regret came out worse for FA beyond 2 SE at every γ. Median top-10 recall was lower at γ = 5 and 3 (0% against 10%) and tied at γ = 8 (10%; means 12% against 9%). FA ended on the exactly-best portfolio on 0 seeds at every γ, against 0, 4 and 1 for the table, so it is clearly behind only at γ = 3.

This run shows that a good overall ranking (Spearman) does not imply a good pick, because the pick depends on the small differences among the best portfolios. The SPY under-weight on every seed points to a bias shared across seeds rather than noise. On the 20% grid `fa-spline` under-weighted SPY on every seed only at γ = 8 (7 of 10 at γ = 3, 1 of 10 at γ = 5). The tabular `grid10-robustness` run also under-weights SPY, on 8, 4 and 6 of 10 seeds by one grid step (10 points), so the direction is not specific to the features; FA's under-weight is larger (10 to 30 points) and on every seed. The suspected cause, unfinished weight directions, is not tested by this run. William's mean-field calculation (true μ and Σ) predicted gaps of about 33.6, 0 and 1.3 bps/yr; the realized gaps are larger at every γ, so it understates the problem.

#### Outcome for `fa-spline-grid10-eps10`, written after the run (2026-11-11)

**TL;DR / In plain terms:** raising ε at halfway from 3% to 10% (1.5 times the random episodes, 5.59M uniform-equivalent against 3.70M) did not help. The learning gap is within noise of ε = 3% at every γ, training regret rose by about 25 to 28 bps/yr per episode, and the greedy portfolio still never settled. At the same α, this much extra exploration did not close FA's gap to the table on this grid.

Paired by seed against `fa-spline-grid10-eps3` (10 seeds per γ; numbers from each experiment's `evaluation/per_seed.csv`):

| | γ = 5 | γ = 3 | γ = 8 |
|---|---|---|---|
| Mean learning gap, ε 10% / ε 3% / table (bps/yr) | 81.4 / 78.8 / 11.7 | 155.6 / 140.6 / 22.0 | 60.2 / 59.6 / 5.8 |
| ε 10% minus ε 3% (SE) | +2.6 (11.8) | +15.0 (14.8) | +0.6 (10.1) |
| Median Spearman, ε 10% / ε 3% | 0.975 / 0.976 | 0.964 / 0.963 | 0.985 / 0.983 |
| Paired Spearman difference (SE) | +0.0068 (0.0049) | +0.0065 (0.0066) | +0.0027 (0.0011) |
| Training regret per episode, ε 10% / ε 3% (bps/yr) | 147.8 / 122.7 | 188.2 / 160.0 | 151.2 / 126.5 |
| Paired regret difference (SE) | +25.1 (3.9) | +28.1 (4.5) | +24.8 (3.4) |
| Primary rule met, ε 10% | 0 of 10 | 0 of 10 | 0 of 10 |

Registered expectations (against `fa-spline-grid10-eps3`):
- **Smaller learning gap at γ = 5: not shown.** The point estimate has the wrong sign (+2.6) and is 0.2 SE from zero. The gaps at γ = 3 and 8 (no direction registered) are also within noise.
- **Higher training regret at γ = 3 and 8: held** (6.2 and 7.3 SE; higher on every seed). At γ = 5, where no direction was registered, regret is also higher by 6.4 SE.
- **Higher Spearman at every γ: held at γ = 8 only** (2.5 SE, 8 of 10 seeds higher). Not shown at γ = 5 (1.4 SE) or γ = 3 (1.0 SE, 5 of 10 seeds higher). Graded on the paired mean, as the 2 SE rule requires; at γ = 5 the median moved the other way (0.975 against 0.976).

Against `grid10-robustness` (descriptive only), the learning gap is larger by +69.7 (SE 4.7), +133.6 (12.8) and +54.4 (6.4) bps/yr, on all 10 seeds at every γ. Every seed still under-weights SPY (picks hold 20% or 30% against the best portfolio's 50%, 60% and 40%), no seed ends on the exactly-best portfolio, median top-10 recall is 0% at every γ, and the discretization gap is identical. No seed holds any portfolio for 2.6M episodes, against 4, 3 and 3 seeds at ε = 3%. The ΔQ rule is met in the first window by 6 or 7 of 10 seeds and later by the rest (up to 749M days), where at ε = 3% every seed met it in the first window. The late seeds are the same seeds at the same days at every γ (runs 0, 5 and 9, plus run 8 at γ = 5 and 8), which points to the exploration stream the γ's share rather than to anything γ-specific.

This run does not settle why FA trails the table. By the α·λ·N count, ε = 10% finishes only about 8 more weight directions at every cutoff (504 to 496 of 558 unfinished at 1e-12, 432 to 424 at 1e-10, 324 to 316 at 1e-8), so it is a weak test of the unfinished-directions explanation, and a null result does not rule that explanation out. William's mean-field calculation, the quantitative form of that explanation, predicted the γ = 5 gap would fall from 33.6 to 4.4 bps/yr (−29.2). The realized change is +2.6 (SE 11.8), about 2.7 SE from that prediction, so the run is evidence against the mean-field version. The realized gaps (81.4, 155.6 and 60.2) are also far above its levels (4.4, 0 and 1.3). A tabular constant-α run on the 10% grid would separate the learning-rule effect from the representation effect; it is not planned.
