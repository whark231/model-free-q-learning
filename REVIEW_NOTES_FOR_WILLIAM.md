# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-11-05
**Status:** the baseline and five post-hoc runs are done. The function-approximation (FA) code, the Q-value fit metrics and the step-size tuning are done. Three constant-learning-rate runs are registered but not run: `tabular-const`, `fa-quadratic` and `fa-spline`. Closed items are in `REVIEW_HISTORY.md`. The FA hand-off to Albert, with every edit to his files, is in `FA_INTEGRATION_NOTES.md`.

---

## TL;DR / In plain terms

The professor agreed to a constant learning rate α for FA, and a separately tuned constant α for the tabular comparison. Tuning on separate seeds picked 3.2e-4 for tabular and the quadratic, and 7.11e-6 with l2 = 0 for the splines. The three runs are registered with their expected results. They start once Albert has reviewed the PR and agreed to those expectations. No FA result exists yet.

**Key concepts to understand**

- **The model ladder.** Tabular (the baseline) → splines, the main model (generic smooth features, never told the reward is quadratic) → quadratic, the correctly specified reference. The quadratic is never reported as the discovery result.
- **Why `tabular-const` exists.** It has the same α, seeds and settings as `fa-quadratic`, so the pair differs only in the features. Against `baseline-v1` (α = 1/N), the learning rate also differs.
- **Constant α keeps a noise floor.** For tabular at α = 3.2e-4 it is about 49 bps/yr per Q-value. The best grid portfolio leads the second by only 10.5 bps/yr at γ = 5 (`EXPERIMENTS.md` §8).

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V. Everything FA and constant-α is labelled post hoc.
- On the tuning seeds, tabular with α = 1/N scored 48 bps/yr above the best constant α (seed SE 28). These are tuning numbers, not results, but they are worth a sentence in the Part III learning-rate discussion.

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Benchmarks, grid, reward, tabular agent | `src/benchmarks/`, `src/action_space/`, `src/rewards.py`, `src/agent.py` | `REVIEW_HISTORY.md` §1 |
| Baseline and five post-hoc runs | `results/`, `reports/` | `REVIEW_HISTORY.md` §8, `EXPERIMENTS.md` §7 |
| FA agent, quadratic and spline features, FA training path | `src/fa_agent.py`, `src/features.py`, `src/train.py` | `FA_INTEGRATION_NOTES.md` §1; independent reviews applied |
| Q-value fit metrics (correlation, top-10 recall, optimism) | `src/evaluate.py` | `EVALUATION.md` §2; `b63` and `eps-decay` re-evaluated |
| Step-size tuning | `src/tune_fa.py`, `results/tuning/` | `FA_INTEGRATION_NOTES.md` §5, `EXPERIMENTS.md` §8.1; 108 runs; close-call rule fixed before any result |
| `tabular-const`, `fa-quadratic`, `fa-spline` registered | `src/config.py`, `EXPERIMENTS.md` §8 | Values tested against `fa_tuning.json`; committed post-hoc folders still match |

405 tests pass.

## Your next steps

1. **Finish `tabular-const`** (see the to-do below), then commit `results/tabular-const/`, `reports/tabular-const/` and `reports/registry.json` on `run/tabular-const` and open a PR to `dev`.
2. **Run `fa-quadratic` and `fa-spline`**, one at a time on our own machines, each on its branch (`run/fa-quadratic`, `run/fa-spline`, made from `dev` after PR #23). `tabular-const` took 10.5 to 13.4 min per seed. After each run: tag the commit, register it with `compare_with` (`reports/README.md`), extract and build on the machine that ran it. Merge the runs into `dev` in order (`tabular-const`, then `fa-quadratic`, then `fa-spline`), so each report can compare against the ones before it. Expect a small `registry.json` conflict each time; keep every entry.
3. **Transaction-cost extension, with Albert.** First settle the two open prospectus decisions: a fixed cost or a sweep, and how finely to represent holdings. `feature/transaction-costs` is ready.
4. **Review Albert's evaluation PR** for `RUN_ISSUES` Findings 1 to 3 when he opens it.
5. **`update()` return value for the tabular agent** (`REVIEW_HISTORY.md` §5.3). Low priority; it only speeds up 10% grid tabular runs.
6. **Your report sections** (prospectus §4): Theory/Benchmark, Q-learning (including the constant-α vs 1/N discussion), and the FA extension.

**`tabular-const` report notes (to do).** The run finished on 2026-11-06 (30 seeds, trained at tag `tabular-const`, commit `bc311a3`). It is registered in `reports/registry.json`, extracted and built (`reports/tabular-const/`), but its `notes` are empty. To do:
- Read the report and check it against the expectations in `EXPERIMENTS.md` §8.3: a larger mean learning gap than `baseline-v1` at every γ, and fewer seeds meeting the primary rule at every γ (baseline-v1: 9, 8 and 9).
- Fill in `tldr`, `takeaways` and `caveats`, then run `python -m src.reporting.build tabular-const`.
- Record the outcome under §8.3, as was done for §7.7.
- The Q-fit numbers are in each `evaluation/summary.json` under `q_fit`, not in the report.
- Put this in `caveats`: the report flags the calibration and `grid.py`, `environment.py` and `rewards.py` as changed against `baseline-v1`. That is false: the content is byte-identical once Windows line endings are normalized. `agent.py` and `train.py` really changed, but not in tabular behaviour (an `epsilon` property; the FA path and logging).

## To decide with Albert

**Which post-hoc runs go in the report.** `grid10-robustness` is the action-space check and γ = 3 and 8 the economic check. `b63` and `n26m` cover "different training budgets, episode lengths". `eps-decay` should stay short, as mostly a negative result. `all-three` is the combined check `grid10-robustness` is compared against. The three new runs belong in the FA extension.

**A 10% grid FA run.** `fa-spline-grid10` was planned in `FA_INTEGRATION_NOTES.md` §2. Decide after the 20% grid results.
