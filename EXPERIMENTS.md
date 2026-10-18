# Running the Experiments

**Owner:** Albert · **Status:** settings fixed 2026-10-16; no full-scale run yet
**Settings:** `src/config.py` · **Runner:** `src/run_experiments.py` · **Tests:** `src/tests/test_config.py` (19)

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
- Results land in `results/`, which git ignores by default (see §4).

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

**Sharing.** `results/` is in `.gitignore`, so nothing here reaches GitHub yet.
The evaluation outputs are small (well under 1 MB per experiment). The training
logs are larger (about 2.5 MB per seed measured at full scale, so about 25 MB
per experiment and 100 MB for all four) and can always be regenerated from the
settings and seeds. The recommended option is to commit `INDEX.md`, `index.json`,
the manifests and the `evaluation/` folders, and keep the `.npz` logs local. That
needs a team decision and a small `.gitignore` change.

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
