# Running the Experiments

**Owner:** Albert · **Status:** settings fixed 2026-10-16; first full-scale run done 2026-10-22 (tag `baseline-v1`, `results/`); post-hoc runs registered 2026-10-27, not yet run (§7); later runs follow `reports/README.md`
**Settings:** `src/config.py` · **Runner:** `src/run_experiments.py` · **Tests:** `src/tests/test_config.py`

---

## TL;DR / In plain terms

One command trains every experiment the report needs, evaluates each one, compares
the two grid sizes and writes a results index. All settings live in
`src/config.py` and were fixed before any real run, so no number in the report can
be accused of being tuned to its outcome. A full run takes roughly 3.5 hours on
Albert's Mac and can be stopped and restarted.

**Key concepts to understand**

- **Baseline vs. robustness.** γ = 5 on the 20% grid is the headline experiment.
  The other three each change one thing (the grid, or γ to 3 or 8), so any
  difference in results can be traced to that one change.
- **Fixed in advance.** Every setting below was chosen before results existed.
  Changing one later needs a written reason, never "it looked better."

**Key takeaways**

- Part IV (row 7 of the prospectus) has its convergence rule and settings fixed;
  the next step is the full-scale run.
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
  seed i sees identical simulated returns and exploration draws in all four
  experiments, and all are scored on the same evaluation sample. Differences
  between experiments therefore come from the setting that changed, not from
  luck.
- **The near-optimal line across γ.** 25 bps/yr is 3.5%, 4.8% and 6.7% of the
  best grid portfolio's value at γ = 3, 5 and 8, so it is a tighter bar at
  γ = 3 than at γ = 8 (`EVALUATION.md` §7).

## 7. Post-hoc runs (registered 2026-10-27, not yet run)

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
- Each run lives on its own branch (`run/b60`, `run/n26m`, `run/eps-decay`,
  `run/all-three`, `run/grid10-robustness`) holding only its own entry in
  `src/config.py: POST_HOC_RUNS`, its tests and its section below.

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

Measured on Albert's Mac: the baseline took 281 to 323 s per seed (about
23.6 µs per episode) at B = 20 on the 20% grid with 13M episodes, so about
2.5 hours for 30 seeds (3 γ × 10). Scaling rules used below:

- **Episode length.** B = 63 measured 43.8 µs vs 38.2 µs at B = 20 in
  William's timing, about 0.13 µs per extra simulated day. B = 60 adds about
  5.2 µs per episode, so +14% to +22% depending on whether it scales with the
  total or adds a fixed amount. Estimates below use the range.
- **Training budget.** Time is proportional to the number of episodes.
- **ε.** No effect on speed worth estimating.
- **10% grid.** 91 µs per episode with the current runner and 51.6 µs for the
  bare loop (`REVIEW_HISTORY.md` §4). The difference is the full Q-table copy
  after every episode (§5.3).

Evaluation adds a few minutes per run on the 20% grid. Estimates are ±20%.

### 7.3 Run `b60`: episode length B = 60 (post hoc)

**TL;DR / In plain terms:** each training episode scores a portfolio on 60
simulated days instead of 20. A longer batch gives a less noisy score for
every portfolio at once, so the agent should be better at telling the top
few apart. Nothing else changes.

**Key concepts to understand**

- **Why B matters.** One episode's reward is the sample mean minus γ/2 times
  the sample variance of 20 daily returns. Its standard deviation falls
  roughly with √B, so B = 60 cuts the noise of every reward by about √3 ≈ 1.7
  without needing any extra episodes.
- **Why B does not touch the benchmark.** μ and Σ are daily, and the reward is
  a daily mean-variance score at any B, so the best grid portfolio and the
  discretization gap are the same as in the baseline.

**Key takeaways**

- This is the lever William's surrogate study found strongest (`REVIEW_HISTORY.md`
  §7.2). It is the Part VI "episode lengths" robustness check the assignment
  lists.
- William proposed B = 63 and B = 252 (`REVIEW_NOTES_FOR_WILLIAM.md`, "To
  decide with Albert"). Albert chose B = 60 (about a quarter, a round multiple
  of the baseline's 20). B = 252 is not registered here.

| Setting | Baseline (`baseline-v1`) | This run |
|---|---|---|
| Episode length B | 20 | **60** |
| Training episodes per seed | 13,000,000 | 13,000,000 (780M simulated days) |
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
python -m src.run_experiments --run b60 --dry-run
python -m src.run_experiments --run b60 --results-dir results/b60
```

**Estimated time.** 340 to 370 s per seed, so about 3 hours for 30 seeds
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
  episodes, with B left at 20 (B = 60 is run `b60`).

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
- **Random tries per portfolio.** 1,439 instead of 1,097. Both are about 800
  times fewer than the 1.1 million visits needed to separate the top two
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
  result, not as a tuning attempt.

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

### 7.6 Run `all-three`: B = 60, 26M episodes and ε = 3% at halfway together (post hoc)

**TL;DR / In plain terms:** this run applies all three changes at once:
60-day episodes, 26 million episodes per seed, and ε falling to 3% at
halfway (episode 13 million). It shows how much the changes give together,
compared with runs `b60`, `n26m` and `eps-decay`, which each change one thing.

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
| Episode length B | 20 | **60** |
| Training episodes per seed | 13,000,000 | **26,000,000** (1.56B simulated days) |
| ε at halfway | 1% (β ≈ 7.08e-7) | **3%** of the new budget, at episode 13M (β ≈ 2.70e-7) |
| Grid | 20% (1,287) | 20% (1,287) |
| K (convergence) | 1,300,000 episodes | **2,600,000 episodes** |
| γ | 3, 5, 8 | 3, 5, 8 |
| Seeds | 10, master seed 20261016 | same |

**Expected direction, written before running.** The smallest learning gap
of runs 1 to 4 at every γ, mostly from the longer episodes; at γ = 5 a mean
gap below `b60`'s. Exactly-best share at or above the baseline's. Training
regret slightly above `b60`'s because of the slower ε decay. Discretization
gap identical to the baseline.

**Command.**

```bash
python -m src.run_experiments --run all-three --dry-run
python -m src.run_experiments --run all-three --results-dir results/all-three
```

**Estimated time.** 680 to 730 s per seed, so about 6 hours for 30 seeds,
plus a few minutes of evaluation.
