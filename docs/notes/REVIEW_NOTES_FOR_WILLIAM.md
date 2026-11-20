# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-11-09, updated 2026-11-20
**Status:** open items only. Closed items are in `REVIEW_HISTORY.md` and `AUDIT_2026-11-06.md` §2.

---

## TL;DR / In plain terms

What is left is the transaction-cost agent runs and the report. The TC redesign (`TC_DRIFT_DESIGN.md`) is built, checked and tuned, and its run driver is written; the 40 runs come next. Nothing of the report is drafted yet, and it includes the FA expectations that failed.

**Key concepts to understand**

- **Shrinkage under a constant α.** With α constant and Q starting at 0, a portfolio updated N times holds about (1 − (1 − α)^N) of its true value. Portfolios tried only during exploration (about 1,100 times) end at about 0.3 of their value, measured at 0.29 to 0.30 in `tabular-const`'s logs. The inferred consequence (not tested by a counterfactual run) is that the greedy portfolio can then never be overtaken, which would explain why `tabular-const` "converged" in 28 of 30 seeds while finding the best portfolio in 2.
- **Unfinished FA directions.** Uniform exploration lasts about 1/β ≈ 1.41M episodes, not the 13M our §8.4 estimate assumed. At that count 36 of the 45 quadratic directions and 351 of the 369 spline directions barely move from θ = 0. This is the suspected cause of the γ = 8 miss (every FA seed holds 20% SPY where the best portfolio holds 40%); it is not tested yet.

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V.
- For the FA section: both FA models beat `tabular-const` at γ = 5 and 3 (paired −20.9 ± 9.8 and −99.2 ± 23.7 bps/yr for the quadratic) and rank the landscape better at every γ on Spearman, but neither beats `baseline-v1` reliably. `fa-spline` ranked the landscape better than the correctly specified quadratic, against our prediction. No "discovery" claim.

---

## Still open

Updated 2026-11-20. Items closed since 2026-11-19 (TC-1, TC-2, TC-3, TC-5, TC-6, TC-7, TC-9, check 5, D-B, D-F, the Part IV counts, Albert's 2026-11-17 questions) are in `REVIEW_HISTORY.md` §10. The whole open list for both of us is `AUDIT_2026-11-06.md` §9.6.

| Item | What | Waits on |
|---|---|---|
| TC runs | Albert runs the 40 final agent runs after his review of `run.py` (decided 2026-11-20), then scores each level (`TC_DRIFT_DESIGN.md` §10; his notes have the steps). Go/no-go Nov 24 | Albert |
| TC review | Albert reviews `run.py` and every edit to his files listed in `NEXT_STEPS_FOR_ALBERT.md` (2026-11-20 section), before any level is scored | Albert |
| TC-4, D-H | The report is not started. The report schedule is still to agree with Albert | Both |
| TC-8, D-G | How the first TC design is reported. M7, M8, M9, M11 and the TC halves of M1 and F-L12 wait on it | Both |
| TC D7 | The pointer in `TRANSACTION_COSTS.md` (Albert's file); the audit's is in | Albert |
| M12, F-L10 | Report text: the FA tuning disclosure; caveats on weights where no seed converged | The report draft |
| TC report text | The two oracle directions not met (trade-once does not approach the oracle as costs grow; myopic holds less than the oracle under λ only); the oracle's in-sample V is about 5 bps/yr high at M = 2,000; the disclosures in design §14 risk 5 (TC-13 and the tail-average change); c1 not resolved (pilot 0.41, 3.19 and 0 of true, against c2 at 0.94, 0.49, 1.01), the exploration limit §9 predicted; D12 tracks the target the agent trades to and says nothing about when it holds (§9) | The TC section |
| F-L9 | Episodes column in `performance.csv` (partly fixed: `INDEX.md` gives days = episodes × B) | Low priority; or say it in the report |
| L7 | Whether `all-three`'s report should compare with its single-change runs | A decision with Albert |
| F-L13 | Tag `grid10-robustness` exists only on Albert's Mac | Albert |

## Your next steps

1. **TC agent runs: Albert's** (above). Answer his review questions on `run.py` and `agent.py` quickly, since the runs wait on them.
2. **Final non-TC runs.** C2, C3, P1 and P2 (`AUDIT_2026-11-06.md` §6) are not run; the 2026-11-19 review recommends none, except P2 if one of us has a free day (§9.4 D-H). Register any run with an expected direction before it starts.
   - A run trained from the current `dev` will show "Training code changed: src/action_space/grid.py, src/train.py" against every earlier run. The flag is honest but no algorithm changed: the edits are LF writes and the increment guard, and `grid_sha256` still matches. Say so in each new run's caveats.
3. **Write the FA section** from `AUDIT_2026-11-06.md` §2.3 (the experiment, the failed expectations and what can be claimed, all re-checked) and `EXPERIMENTS.md` §8.7 and §8.8, including the measured shrinkage and the tuning disclosure (tuned by simulating the same environment, on separate seeds, at γ = 5 only; `AUDIT_2026-11-06.md` M12).
4. **`update()` return value for the tabular agent** (`REVIEW_HISTORY.md` §5.3). Low priority; speeds up 10% grid tabular runs only, and C1 ran without a tabular 10% comparison, so probably not needed.
5. **Your report sections** (prospectus §4): Theory/Benchmark, Q-learning (including the constant-α versus 1/N discussion), and the FA extension.

## To decide with Albert

- **The report schedule** (D-H) and **how the first TC design is reported** (D-G).
- **Final runs** (above).
- **`all-three`'s comparison set** (L7).

**Which post-hoc runs go in the report.** `grid10-robustness` is the action-space check and γ = 3 and 8 the economic check. `b63` and `n26m` cover "different training budgets, episode lengths". `eps-decay` stays short, as mostly a negative result. `all-three` is the combined check `grid10-robustness` is compared against. The three constant-α runs go in the FA extension and in the Part III learning-rate discussion.
