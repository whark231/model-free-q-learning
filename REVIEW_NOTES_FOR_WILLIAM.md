# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-11-09
**Status:** the required work (Parts I to VI) is done and independently re-checked. The three constant-learning-rate runs (`tabular-const`, `fa-quadratic`, `fa-spline`) are run and reported, with Q fit in every report. Every non-TC code issue from both audits is closed, Albert's on `dev` and William's on `feature/william-audit-fixes` (not yet merged). What stays open is listed below. Closed items are in `REVIEW_HISTORY.md`.

---

## TL;DR / In plain terms

The code is in good shape: two audits found no correctness bug in the core or the FA code, and every code issue they raised outside transaction costs is fixed. What is left is decisions and writing. The transaction-cost agent scores like a random portfolio even with free trading, so the TC extension needs a design decision before any more TC runs; then pick the final runs and write the report, including the FA expectations that failed.

**Key concepts to understand**

- **Shrinkage under a constant α.** With α constant and Q starting at 0, a portfolio updated N times holds about (1 − (1 − α)^N) of its true value. Portfolios tried only during exploration (about 1,100 times) end at about 0.3 of their value, measured at 0.29 to 0.30 in `tabular-const`'s logs. The inferred consequence (not tested by a counterfactual run) is that the greedy portfolio can then never be overtaken, which would explain why `tabular-const` "converged" in 28 of 30 seeds while finding the best portfolio in 2.
- **Unfinished FA directions.** Uniform exploration lasts about 1/β ≈ 1.41M episodes, not the 13M our §8.4 estimate assumed. At that count 36 of the 45 quadratic directions and 351 of the 369 spline directions barely move from θ = 0. This is the suspected cause of the γ = 8 miss (every FA seed holds 20% SPY where the best portfolio holds 40%); it is not tested yet.

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V.
- For the FA section: both FA models beat `tabular-const` at γ = 5 and 3 (paired −20.9 ± 9.8 and −99.2 ± 23.7 bps/yr for the quadratic) and rank the landscape better at every γ on Spearman, but neither beats `baseline-v1` reliably. `fa-spline` ranked the landscape better than the correctly specified quadratic, against our prediction. No "discovery" claim.

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Benchmarks, grid, reward, tabular agent | `src/benchmarks/`, `src/action_space/`, `src/rewards.py`, `src/agent.py` | `REVIEW_HISTORY.md` §1; re-derived in both audits |
| Baseline and five tabular post-hoc runs | `results/`, `reports/` | `REVIEW_HISTORY.md` §8, `EXPERIMENTS.md` §7 |
| FA agent, features, training path, Q-value fit metrics | `src/fa_agent.py`, `src/features.py`, `src/train.py`, `src/evaluate.py` | `FA_INTEGRATION_NOTES.md`; no bug found in either audit |
| Step-size tuning | `src/tune_fa.py`, `results/tuning/` | `EXPERIMENTS.md` §8.1 |
| `tabular-const`, `fa-quadratic`, `fa-spline` run and reported | `results/<run>/`, `reports/<run>/` | `EXPERIMENTS.md` §8.7 |
| Two whole-project audits | `AUDIT_2026-11-06.md` | Independent reviewers; §4 and §8.4 list what was verified |
| Albert's audit fixes, 2026-11-09 | `dev` | `AUDIT_2026-11-06.md` §2.1, §8.3 |
| Every item Albert left for William, 2026-11-09: the rounding test, L1, L3, the `.gitattributes` refresh, re-evaluation and re-extraction of `b63`, `eps-decay`, `tabular-const`, `fa-quadratic` (L8, F-M3), the prospectus status (F-M11) | `feature/william-audit-fixes` | `AUDIT_2026-11-06.md` §2.2, `REVIEW_HISTORY.md` §9; no reported value changed |
| `tune_fa --summarize` byte-stable (F-L12, FA part); every non-TC output written with LF (a bug found while closing F-M3) | `feature/william-audit-fixes` | Tests fail without each fix; `NEXT_STEPS_FOR_ALBERT.md` explains the LF bug for Albert |

442 tests pass on 2026-11-09 (Windows, Python 3.10).

## Still open

| Item | What | Waits on |
|---|---|---|
| Commit and merge | `feature/william-audit-fixes` is uncommitted; then a PR into `dev` | William |
| F-M1 | Merge `dev` into `main` (GitHub's default branch, about 50 commits behind) through one PR | William and Albert agreeing |
| H4, M12, F-L10 | Report text: the failed FA expectations, the tuning disclosure, caveats on weights where no seed converged | The report draft |
| F-L9 | Episodes column in `performance.csv` (partly fixed: `INDEX.md` gives days = episodes × B) | Low priority; or say it in the report |
| L7 | Whether `all-three`'s report should compare with its single-change runs | A decision with Albert |
| F-L13 | Tag `grid10-robustness` exists only on Albert's Mac | Albert |
| TC | H1, H2, M7 to M9, M11, plus M1 and F-L12 for `tc_sweep_report.py` | The TC redesign |

## Your next steps

1. **Commit `feature/william-audit-fixes`, push it and open a PR into `dev`.** Run the full suite first; Albert's `test_every_registered_report_builds_from_committed_data` fails on a broken registry or a stale report.
2. **Decide the TC redesign with Albert** (`AUDIT_2026-11-06.md` H1, H2, §6.2). The reviewer's suggestion splits the problem: learn each target's reward by sample average, as `baseline-v1` does, and plan over the known costs with the existing Bellman solver. Its one-seed test reached 0.8 bps/yr of the oracle at the 2 bp anchor. Decide whether that counts as model-free enough, whether to add drift, and which cost levels matter. Until then, do not run `discount.json` or `learning-rate.json`.
3. **Choose the final runs** from `AUDIT_2026-11-06.md` §6. The strongest FA candidate is FA on the 10% grid (C1), where tabular struggled most; a γ = 8 check (C2) would test the unfinished-directions explanation. Register each with an expected direction before it runs.
   - A run trained from the new branch will show "Training code changed: src/action_space/grid.py, src/train.py" against every earlier run. The flag is honest but no algorithm changed: the edits are LF writes and the increment guard, and `grid_sha256` still matches. Say so in each new run's caveats.
4. **Write the FA section** using `EXPERIMENTS.md` §8.7, including the failed expectations, the measured shrinkage and the tuning disclosure (tuned by simulating the same environment, on separate seeds, at γ = 5 only; `AUDIT_2026-11-06.md` M12).
5. **`update()` return value for the tabular agent** (`REVIEW_HISTORY.md` §5.3). Low priority; speeds up 10% grid tabular runs only, so it matters if C1 includes a tabular 10% comparison.
6. **Your report sections** (prospectus §4): Theory/Benchmark, Q-learning (including the constant-α versus 1/N discussion), and the FA extension.

## To decide with Albert

**TC redesign** (step 2 above).

**Final runs.** `AUDIT_2026-11-06.md` §6. R0 (the Q-fit re-evaluations) is done; next a TC option, then C1 to C3, P1 and P2 as time allows.

**Merging `dev` into `main`** (F-M1).

**`all-three`'s comparison set** (L7).

**Which post-hoc runs go in the report.** `grid10-robustness` is the action-space check and γ = 3 and 8 the economic check. `b63` and `n26m` cover "different training budgets, episode lengths". `eps-decay` stays short, as mostly a negative result. `all-three` is the combined check `grid10-robustness` is compared against. The three constant-α runs go in the FA extension and in the Part III learning-rate discussion.
