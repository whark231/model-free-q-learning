# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-11-09
**Status:** open items only. Closed items are in `REVIEW_HISTORY.md` and `AUDIT_2026-11-06.md` §2.

---

## TL;DR / In plain terms

What is left is decisions and writing. The transaction-cost agent scores like a random portfolio even with free trading, so the TC extension needs a design decision before any more TC runs; then pick the final runs and write the report, including the FA expectations that failed.

**Key concepts to understand**

- **Shrinkage under a constant α.** With α constant and Q starting at 0, a portfolio updated N times holds about (1 − (1 − α)^N) of its true value. Portfolios tried only during exploration (about 1,100 times) end at about 0.3 of their value, measured at 0.29 to 0.30 in `tabular-const`'s logs. The inferred consequence (not tested by a counterfactual run) is that the greedy portfolio can then never be overtaken, which would explain why `tabular-const` "converged" in 28 of 30 seeds while finding the best portfolio in 2.
- **Unfinished FA directions.** Uniform exploration lasts about 1/β ≈ 1.41M episodes, not the 13M our §8.4 estimate assumed. At that count 36 of the 45 quadratic directions and 351 of the 369 spline directions barely move from θ = 0. This is the suspected cause of the γ = 8 miss (every FA seed holds 20% SPY where the best portfolio holds 40%); it is not tested yet.

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V.
- For the FA section: both FA models beat `tabular-const` at γ = 5 and 3 (paired −20.9 ± 9.8 and −99.2 ± 23.7 bps/yr for the quadratic) and rank the landscape better at every γ on Spearman, but neither beats `baseline-v1` reliably. `fa-spline` ranked the landscape better than the correctly specified quadratic, against our prediction. No "discovery" claim.

---

## Still open

| Item | What | Waits on |
|---|---|---|
| M12, F-L10 | Report text: the tuning disclosure, caveats on weights where no seed converged | The report draft |
| F-L9 | Episodes column in `performance.csv` (partly fixed: `INDEX.md` gives days = episodes × B) | Low priority; or say it in the report |
| L7 | Whether `all-three`'s report should compare with its single-change runs | A decision with Albert |
| F-L13 | Tag `grid10-robustness` exists only on Albert's Mac | Albert |
| TC | H1, H2, M7 to M9, M11, plus F-L12 and Albert's M1 print fix (`sys.stdout.reconfigure`) for `tc_sweep_report.py` | The TC redesign |

## Your next steps

1. **Decide the TC redesign with Albert** (`AUDIT_2026-11-06.md` H1, H2, §6.2). The reviewer's suggestion splits the problem: learn each target's reward by sample average, as `baseline-v1` does, and plan over the known costs with the existing Bellman solver. Its one-seed test reached 0.8 bps/yr of the oracle at the 2 bp anchor. Decide whether that counts as model-free enough, whether to add drift, and which cost levels matter. Until then, do not run `discount.json` or `learning-rate.json`.
2. **Choose the final runs** from `AUDIT_2026-11-06.md` §6. The strongest FA candidate is FA on the 10% grid (C1), where tabular struggled most; a γ = 8 check (C2) would test the unfinished-directions explanation. Register each with an expected direction before it runs.
   - A run trained from the current `dev` will show "Training code changed: src/action_space/grid.py, src/train.py" against every earlier run. The flag is honest but no algorithm changed: the edits are LF writes and the increment guard, and `grid_sha256` still matches. Say so in each new run's caveats.
3. **Write the FA section** from `AUDIT_2026-11-06.md` §2.3 (the experiment, the failed expectations and what can be claimed, all re-checked) and `EXPERIMENTS.md` §8.7, including the measured shrinkage and the tuning disclosure (tuned by simulating the same environment, on separate seeds, at γ = 5 only; `AUDIT_2026-11-06.md` M12).
4. **`update()` return value for the tabular agent** (`REVIEW_HISTORY.md` §5.3). Low priority; speeds up 10% grid tabular runs only, so it matters if C1 includes a tabular 10% comparison.
5. **Your report sections** (prospectus §4): Theory/Benchmark, Q-learning (including the constant-α versus 1/N discussion), and the FA extension.

## To decide with Albert

**TC redesign** (step 1 above).

**Final runs.** `AUDIT_2026-11-06.md` §6. A TC option first, then C1 to C3, P1 and P2 as time allows.

**`all-three`'s comparison set** (L7).

**Which post-hoc runs go in the report.** `grid10-robustness` is the action-space check and γ = 3 and 8 the economic check. `b63` and `n26m` cover "different training budgets, episode lengths". `eps-decay` stays short, as mostly a negative result. `all-three` is the combined check `grid10-robustness` is compared against. The three constant-α runs go in the FA extension and in the Part III learning-rate discussion.
