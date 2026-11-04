# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-11-03
**Status:** the baseline, five post-hoc runs (including `grid10-robustness`) and the function-approximation (FA) code are done. FA is built and tested but not yet tuned or run. Everything closed is in `REVIEW_HISTORY.md`; the FA hand-off to Albert is in `FA_INTEGRATION_NOTES.md`.

---

## TL;DR / In plain terms

The tabular agent is finished, and the FA extension now has an agent, two feature models and a training path. Nothing has been trained with FA at full scale, so there are no FA results yet. The next step is tuning the step size α on separate seeds, and a design calculation says a constant α may be too noisy or too slow at 13M episodes, so the tuning is a real test, not a formality.

**Key concepts to understand**

- **Function approximation.** Instead of one Q-value per portfolio, the agent learns a parameter vector θ and scores portfolio a as θ · F[a], where F[a] are features computed from its weights. A reward for one portfolio then also moves the estimates of similar portfolios.
- **The model ladder.** Tabular (the baseline) → splines, the main model (additive plus pairwise in the nine weights, never told the reward is quadratic) → quadratic, a correctly specified reference (handed the true functional form). The quadratic is never reported as the discovery result.
- **Constant α trades noise for speed.** A large α learns fast but never stops reacting to reward noise; a small α averages the noise but learns slowly. `FA_INTEGRATION_NOTES.md` §4 puts numbers on it.

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V; post-hoc runs and FA are labelled post hoc.
- FA runs reuse the baseline's settings and seeds (ρ = 0, B = 20, same ε schedule, 13M episodes, master seed 20261016), so each FA seed is paired with the matching baseline seed. If B or ε change for FA, the tabular baseline must be rerun at those settings too.

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Benchmarks, grid, reward | `src/benchmarks/`, `src/action_space/`, `src/rewards.py` | 60 tests; review issues closed (`REVIEW_HISTORY.md` §1) |
| Tabular agent, with `epsilon` property | `src/agent.py` | 24 tests; unchanged by the FA work |
| γ justification | `DATA_CLEANING.md` §8.4 | Sources listed there |
| Baseline run, 2026-10-22 | `results/`, `reports/baseline-v1/` | `RUN_ISSUES_2026-10-22.md`; every number re-derived |
| Post-hoc runs `b63`, `n26m`, `eps-decay`, `all-three`, `grid10-robustness` | `results/<run_id>/`, `reports/<run_id>/` | `REVIEW_HISTORY.md` §8 (the first four); each expected direction registered before running |
| FA agent `LinearQAgent` (post hoc) | `src/fa_agent.py` | 25 tests in `test_fa_agent.py`; independent review applied |
| FA features: quadratic (reference) and additive + pairwise splines (main) | `src/features.py` | 26 tests in `test_features.py`; spline basis checked against an independent Cox–de Boor; independent review applied |
| FA training path, `TrainConfig` FA fields | `src/train.py` (Albert's file, changes listed in `FA_INTEGRATION_NOTES.md` §1) | 21 tests in `test_train.py`, 1 in `test_config.py`; tabular records unchanged and all 19 committed folders still "finished"; independent review applied |
| FA timing benchmark | `src/bench_fa.py` | `FA_INTEGRATION_NOTES.md` §3 |
| Leakage test covers both FA modules | `src/tests/test_environment.py` | |

360 tests pass.

### What the FA work changed, in one place

- **New files:** `src/features.py`, `src/fa_agent.py`, `src/bench_fa.py`, `src/tests/test_fa_agent.py`, `src/tests/test_features.py`, `FA_INTEGRATION_NOTES.md`.
- **Albert's files, edited with his permission and documented for him:** `src/train.py` (FA path, config fields, `config_record`), `src/run_experiments.py` (compares configs through `config_record`), `src/tests/test_config.py`, `src/tests/test_train.py`.
- **Not changed:** `src/agent.py`, `src/evaluate.py`, `src/config.py`, the reporting code, every committed result.

### Design decisions made along the way

| Decision | Why |
|---|---|
| `update()` returns `None` and the agent sets `updates_all_actions = True` | One update moves every Q̂; `train.py` treats a returned value as "only Q(action) changed" |
| Separate `train_one_run_fa`; tabular loop untouched | Scoring every portfolio each episode would dominate FA run time; keeps tabular behaviour identical |
| Greedy cached and refreshed inside `update()` on a fixed count, every 100 updates for runs | Reads never change behaviour; refresh 100 cuts the spline model on the 10% grid from about 25 hours to 32 minutes per 13M episodes (agent only) |
| Greedy scored in float32, Q-values reported in float64 | Avoids a 330 MB copy per refresh on the 10% spline grid; exact up to float32 rounding |
| Quadratic: 45 products, no intercept or linear terms | The weights sum to one, so those are already in the span |
| Splines: 5 cubic B-splines per asset, pair tensor products, 36 infeasible columns dropped; left redundant | 909 columns, rank 369 (20%) and 558 (10%). SGD from θ = 0 never moves in the redundant directions; l2 damps poorly informed ones |
| Spline model described accurately | It nests every quadratic. On the 20% grid it fits any additive-plus-pairwise function (smoothness restricts nothing there); smoothness binds only on the 10% grid |
| Tabular manifests omit the FA fields (`config_record`) | So committed `baseline-v1` and post-hoc folders still count as "finished" |
| FA runs need `n_episodes` and `log_every` to be multiples of the refresh interval | Found in review: otherwise a run can end on a stale greedy (6 of 16 small runs did), so the logged and reported portfolios disagree and the report extractor crashes |

## Your next steps

1. **Transaction-cost extension, with Albert.** It adds a state (current holdings) and a turnover penalty. Settle the two open prospectus decisions first: a fixed cost or a sweep, and how finely to represent holdings. `feature/transaction-costs` is ready.
2. **FA: write and run the tuning script** (`FA_INTEGRATION_NOTES.md` §5). Tuning master seed 20261103 (3 seeds, never the reported ones), γ = 5, full 13M episodes, α at 8 fractions of the stability bound from 0.5 down to 6.4e-6, then a spline l2 sweep. Choose by the realized reward of the final greedy portfolio on 200k common evaluation episodes, so no μ or Σ enters the choice. About 20 compute-hours.
3. **FA: raise the constant-α risk with the professor if tuning confirms it.** The design calculation says noise falls to the 5 to 18 bps/yr level of the candidate gaps only where learning takes about the whole budget (quadratic) or far longer (splines). A decaying step is the standard fix but departs from the suggested constant α.
4. **FA: after tuning, with Albert** (`FA_INTEGRATION_NOTES.md` §2): FA fields in `config.py`'s `RunVariant` and the runs registered with expected directions; agent fields in `batch_settings` and the report extractor; an `EXPERIMENTS.md` section. Then run `fa-spline` and `fa-quadratic` on the 20% grid, and later the 10% grid.
5. **FA evaluation metrics** (`FA_INTEGRATION_NOTES.md` §6): fitted-vs-true Q correlation, top-10 recall, optimism of the learned portfolio, computed after training for both FA and tabular. Proposed for `evaluate.py`; not implemented.
6. **`update()` return value for the tabular agent** (`REVIEW_HISTORY.md` §5.3). Lower priority now that `grid10-robustness` has run; it only saves time on future 10% grid tabular runs.
7. **Review Albert's evaluation PR** for `RUN_ISSUES` Findings 1 to 3 when he opens it.
8. **Your report sections** (prospectus §4): Theory/Benchmark, Q-learning, and the FA extension.

## To decide with Albert

**Which post-hoc runs go in the report.** `grid10-robustness` is the action-space check and γ = 3 and 8 the economic check. `b63` and `n26m` cover the assignment's "different training budgets, episode lengths". I'd keep `eps-decay` and `all-three` short: the first as mostly a negative result (no gain at γ = 5 and 8, higher regret), the second as the combined check `grid10-robustness` is compared against.

**Reviewing the FA changes to his files.** `FA_INTEGRATION_NOTES.md` §1 lists every edit to `train.py`, `run_experiments.py` and the two test files.
