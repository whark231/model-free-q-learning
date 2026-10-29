# Next Steps

**From:** William
**Date:** 2026-10-29
**Status:** the baseline and four post-hoc runs (`b63`, `n26m`, `eps-decay`, `all-three`) are done, reported and merged into `dev`. Next: the evaluation fixes from `RUN_ISSUES`, the 10% grid run, and the transaction-cost extension, which starts Oct 30.
**Update 2026-10-29 (Albert):** item 2 is done. `grid10-robustness` ran without the `update()` fix and is in PR #22, with its outcome in `EXPERIMENTS.md` §7.7. Item 4 is done in the same PR. The note in `reports/README.md` about raising `--log-every` is updated too.

---

## TL;DR / In plain terms

Thanks for setting up and running the post-hoc runs and the reporting system. Every variant lowered the mean learning gap and the worst seed at every γ. Longer episodes (`b63`) gave the most for the least regret. The required parts are nearly complete: what's left for Part VI is the 10% grid. After that, the remaining work is the two extensions and the report.

**Key concepts to understand**

- **Better on average, not always better.** Every variant beat the baseline's mean gap and worst seed in all 12 comparisons (4 runs × 3 γ). Medians and exact-best shares did not all improve: `n26m` at γ = 3 got worse on both. With 10 seeds, part of the mean improvement is the baseline's single bad seeds not recurring.
- **Days vs. episodes.** Every variant takes more simulated days to converge, because it sees more data. Counted in episodes, `b63` is faster than the baseline, and `n26m` and `eps-decay` are slower, because ε stays high for longer.

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V. The post-hoc runs go in Part VI, labelled post hoc. The full table and the check against each run's registered expectation are in `REVIEW_HISTORY.md` §8.
- The `RUN_ISSUES` fixes are still not in the code, so the tables still lack the total-gap column. That needs fixing before anyone writes Parts IV to VI.

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Data pipeline, μ and Σ | `src/data/`, `data/processed/` | `DATA_CLEANING.md`, 20 tests |
| Simulator | `src/environment.py` | `ENVIRONMENT.md`, 69 tests |
| Benchmarks, grid, reward | `src/benchmarks/`, `src/action_space/`, `src/rewards.py` | 60 tests |
| Agent, runner, evaluation, batch runner, post-hoc variants | `src/agent.py`, `src/train.py`, `src/evaluate.py`, `src/config.py`, `src/run_experiments.py` | `EVALUATION.md`, `EXPERIMENTS.md` |
| Multi-run reports | `src/reporting/`, `reports/` | `reports/README.md`, 18 tests |
| Baseline run, 2026-10-22, with run log kept | `results/`, `reports/baseline-v1/` | `RUN_ISSUES_2026-10-22.md`; every number re-derived |
| Post-hoc runs `b63`, `n26m`, `eps-decay`, `all-three` | `results/<run_id>/`, `reports/<run_id>/` | `REVIEW_HISTORY.md` §8 |
| Your half of the Q-table-copy fix | `src/train.py` | Bit-identical logs test |

283 tests pass.

## Your next steps

1. **`RUN_ISSUES` §4 steps 1 to 6, still open.** `per_seed.csv` and `INDEX.md` still have no total-gap column, and there is no post-hoc "settled" measure or checkpoint label yet. Since the post-hoc runs are in, the re-run in step 4 should regenerate their evaluations too. I'll review the PR.
2. **Run `grid10-robustness`** (`EXPERIMENTS.md` §7.7), the Part VI action-space check, about 21 hours. Before it starts, I'll add the `update()` return value on my side, which saves about 6 µs per episode. The default log interval now fits: 2,600 checkpoints × 43,758 Q-values is 113.8M, under the raised 120M guard in `src/train.py`, so the "raise `--log-every`" note under "10% grid" in `reports/README.md` is out of date. It needs about 910 MB of memory per seed and up to 14 GB of disk for the logs (`EXPERIMENTS.md` §7.7). Then extract and build its report.
3. **Windows crash, still on `dev`.** `python -m src.run_experiments --run <id> --dry-run` stops with `UnicodeEncodeError` on "γ" in the status print (same cause as `REVIEW_HISTORY.md` §1 item D, but in `print`, not a file write). Today every Windows run needs `$env:PYTHONIOENCODING = "utf-8"`. Fix: call `sys.stdout.reconfigure(encoding="utf-8")` at the start of `main()`.
4. **`EXPERIMENTS.md` status.** Line 3 and the §7 heading still say the post-hoc runs are "not yet run". Four have run.
5. **Your report sections** (prospectus §4): Data/Calibration, and convergence analysis and visualizations. `RUN_ISSUES` Findings 4, 5, 8 and 9 become sentences in Parts IV to VI.

## To decide together

**Transaction costs, Week 7 (Oct 30 to Nov 5), paired.** Two open prospectus decisions first: a fixed cost or a sweep, and whether holdings are the exact prior weights or bucketed.

**Which post-hoc runs go in the report.** The assignment needs one action-space check (the 10% grid) and one economic or statistical check (γ = 3 and 8, done). `b63` and `n26m` cover its listed "different training budgets, episode lengths". My suggestion: `eps-decay` short, as mostly a negative result (no gain at γ = 5 and 8, higher regret), and `all-three` as the combined check the 10% grid is compared against.
