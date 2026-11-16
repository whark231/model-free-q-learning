# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-11-09
**Status:** open items only. Closed items are in `REVIEW_HISTORY.md` and `AUDIT_2026-11-06.md` §2.

---

## TL;DR / In plain terms

What is left is the transaction-cost rebuild and the report. The TC redesign is decided (`TC_DRIFT_DESIGN.md`): three ETFs with drift, realistic costs, an oracle benchmark and a spline agent. Phase 2 builds it in parallel with Albert. Then write the report, including the FA expectations that failed.

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
| TC | The redesign replaces the old design, which stays as a negative result. Phase 2: your oracle, agent and features (`TC_DRIFT_DESIGN.md` §10) | Gate Nov 17 |
| TC D10 | Evaluation size: 90 blocks per path, 1,000 starts | Albert's approval |

## Your next steps

1. **Build your Phase 2 TC files** (`TC_DRIFT_DESIGN.md` §5, §6, §9, §10):
   - `src/tc_drift/oracle.py`: built (`feature/tc-drift-oracle`); checks 1 and 2 pass, so the Nov 17 gate is met. Tolerance for checks 3 to 5: 0.5 bps/yr. Preliminary at $1B: check 4 passes (0.000 bps/yr); **check 3 fails at M = 2,000** (doubling M lowers the mean V by 3.1 bps/yr). Decide: more paths, or restate check 3 on the oracle's policy (`TC_DRIFT_DESIGN.md` §5). Check 5's form is still a proposal and needs Albert's `evaluate.py`.
   - `src/tc_drift/features.py` and `src/tc_drift/agent.py`: built (`feature/tc-drift-agent`, which includes the oracle branch), reviewed, all tests pass (`TC_DRIFT_DESIGN.md` §6, "Implemented").
   - **The tuning design is blocked on a decision.** The $1B pilot (§6) shows a constant α cannot learn the cost coefficients at the real noise level: after 500,000 blocks they are off by factors of 3 to 10. Pick from the §9 options (smaller separate α_cost, decaying steps, decorrelated cost features, longer budget), then pilot the D12 rule before fixing the grid.
2. **Choose the final runs** from `AUDIT_2026-11-06.md` §6. FA on the 10% grid (C1) is handed to Albert: it's registered on `run/fa-spline-grid10`, and he picks ε and runs it (`NEXT_STEPS_FOR_ALBERT.md`, 2026-11-10). A γ = 8 check (C2) would test the unfinished-directions explanation. Register each with an expected direction before it runs.
   - A run trained from the current `dev` will show "Training code changed: src/action_space/grid.py, src/train.py" against every earlier run. The flag is honest but no algorithm changed: the edits are LF writes and the increment guard, and `grid_sha256` still matches. Say so in each new run's caveats.
3. **Write the FA section** from `AUDIT_2026-11-06.md` §2.3 (the experiment, the failed expectations and what can be claimed, all re-checked) and `EXPERIMENTS.md` §8.7, including the measured shrinkage and the tuning disclosure (tuned by simulating the same environment, on separate seeds, at γ = 5 only; `AUDIT_2026-11-06.md` M12).
4. **`update()` return value for the tabular agent** (`REVIEW_HISTORY.md` §5.3). Low priority; speeds up 10% grid tabular runs only, so it matters if C1 includes a tabular 10% comparison.
5. **Your report sections** (prospectus §4): Theory/Benchmark, Q-learning (including the constant-α versus 1/N discussion), and the FA extension.

## To decide with Albert

**TC evaluation size** (D10): waiting for Albert.

**Final runs.** `AUDIT_2026-11-06.md` §6. A TC option first, then C1 to C3, P1 and P2 as time allows.

**`all-three`'s comparison set** (L7).

**Which post-hoc runs go in the report.** `grid10-robustness` is the action-space check and γ = 3 and 8 the economic check. `b63` and `n26m` cover "different training budgets, episode lengths". `eps-decay` stays short, as mostly a negative result. `all-three` is the combined check `grid10-robustness` is compared against. The three constant-α runs go in the FA extension and in the Part III learning-rate discussion.
