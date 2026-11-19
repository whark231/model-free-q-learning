# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-11-09, updated 2026-11-18
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
| TC | The redesign replaces the old design, which stays as a negative result. Phase 2: oracle, agent and features built; oracle run done; agent tuning not designed yet (`TC_DRIFT_DESIGN.md` §9) | Your §9 choice; Albert's `train.py` and `evaluate.py` |
| TC D10 | Evaluation size: 90 blocks per path, 1,000 starts; the oracle's $1B lead (1.3 bps/yr) puts m between about 1,300 and 15,000 (`TC_DRIFT_DESIGN.md` §8) | Albert measures it with `evaluate.py` |
| TC tuning | Which §9 option the agent uses. Recommended: recursive least squares on the cost block, after a constant-α burn-in or with a forgetting factor (§9). Not yet tested in the agent | Your choice, then a $1B pilot |
| TC exploration | Even at the least-squares floor, a drift trade's cost is uncertain by about two thirds after 500,000 exploring trades; the current β explores only about 50,000 blocks (§9) | Part of the tuning pilot |
| TC check 5 | Paired form decided 2026-11-17; measured by Albert 2026-11-17, all four levels inside 0.5 bps/yr, resolved only at zero cost (see below) | Your OK on the finite-horizon model side |
| TC D7 | One-line pointers to `TC_DRIFT_DESIGN.md` in `TRANSACTION_COSTS.md` and `AUDIT_2026-11-06.md` §6.2. Due when the design merged; not added, because both files are on Claude's do-not-touch list | Albert adds them, or you lift the restriction |
| TC C12 | Prospectus §4 (TC ownership) and §5 week 9 were out of date (`TC_DRIFT_DESIGN.md` C12). Albert updated them, with §1, §6 and §7, on 2026-11-17 | Your review; it is your file |
| TC report | Two registered oracle directions not met (trade-once does not approach the oracle as costs grow; myopic holds less than the oracle under λ only). The oracle's in-sample V is about 5 bps/yr high at M = 2,000 | The TC section |

## Your next steps

1. **Build your Phase 2 TC files** (`TC_DRIFT_DESIGN.md` §5, §6, §9, §10):
   - `src/tc_drift/oracle.py`: merged. Checks 1 to 4 pass; the Nov 17 gate is met. Check 3 restated on the policy (decided 2026-11-17, post hoc) and scored in an independent model: mean difference at most 0.003 bps/yr at every setting (0.014 at one start), but uninformative at zero cost and $100M, where the oracle's own lead over always rebalancing is under the tolerance. M stays at 2,000. The V-based failure was a shared level shift in r̂, not maximization bias (`TC_DRIFT_DESIGN.md` §5).
   - **Oracle run on all six settings done** (`feature/tc-drift-oracle-runs`, `results/tc-drift/oracle/`, `TC_DRIFT_DESIGN.md` §8). Two registered directions failed: trade-once does not approach the oracle as costs grow, and myopic holds slightly less than the oracle under λ only. Report both.
   - Check 5: the paired form is decided (2026-11-17); Albert is told. It runs once his `evaluate.py` exists.
   - `src/tc_drift/features.py` and `src/tc_drift/agent.py`: built (`feature/tc-drift-agent`, which includes the oracle branch), reviewed, all tests pass (`TC_DRIFT_DESIGN.md` §6, "Implemented").
   - **The tuning design is blocked on a decision.** The $1B pilot (§6) shows a constant α cannot learn the cost coefficients at the real noise level: after 500,000 blocks they are off by factors of 3 to 10. §9 now has a synthetic analysis of the cost block (`src/tc_drift/cost_learning_analysis.py`): decorrelating and decaying together reach the least-squares floor only with a b taken from the optimum (benchmark information); recursive least squares on the cost block reaches it model-free. Even the floor leaves a drift trade's cost uncertain by about two thirds after 500,000 exploring trades, so exploration length is the next limit. Pick the option, then pilot the D12 rule. *Albert, 2026-11-17:* `train()` and `config.py` are ready for the pilot; once you pick, set `N_BLOCKS`, `CHECKPOINT_EVERY` and `HYPERPARAMETERS` in `config.py` or tell Albert the values. D12 may be changed if the pilot fails it, before any final result is examined, labelled as a change (decided 2026-11-17).
   - **TC next steps, in order:**
     1. ~~Open a PR from `feature/tc-drift-oracle-runs` into `dev`~~ Merged (#36, 2026-11-18).
     2. Choose the tuning option. If recursive least squares: add it to the cost block in `agent.py` with a burn-in or forgetting factor, keep it model-free (it uses only the agent's own trades), and test it against `cost_learning_analysis.py`'s synthetic case.
     3. Pilot at $1B on Albert's `train.py`: check the learned c1 and c2 against the calibration offline, the exploration length, and whether D12 fires. Change D12 only before any final result is seen, and label the change.
     4. ~~Run check 5 with Albert~~ Albert ran it (2026-11-17, below); confirm the finite-horizon change. Help him set D10's m once the agent's gap SE is known.
     5. ~~Edit the prospectus for C12~~ Albert updated §4 and §5 at his own request (2026-11-17); please check you agree. Agree D7 with Albert.
     6. Full runs (4 levels × 10 seeds), Nov 20 to 26 on the current timeline.
2. **Choose the final runs** from `AUDIT_2026-11-06.md` §6. FA on the 10% grid (C1) is handed to Albert: it's registered on `run/fa-spline-grid10`, and he picks ε and runs it (`NEXT_STEPS_FOR_ALBERT.md`, 2026-11-10). A γ = 8 check (C2) would test the unfinished-directions explanation. Register each with an expected direction before it runs.
   - A run trained from the current `dev` will show "Training code changed: src/action_space/grid.py, src/train.py" against every earlier run. The flag is honest but no algorithm changed: the edits are LF writes and the increment guard, and `grid_sha256` still matches. Say so in each new run's caveats.
3. **Write the FA section** from `AUDIT_2026-11-06.md` §2.3 (the experiment, the failed expectations and what can be claimed, all re-checked) and `EXPERIMENTS.md` §8.7, including the measured shrinkage and the tuning disclosure (tuned by simulating the same environment, on separate seeds, at γ = 5 only; `AUDIT_2026-11-06.md` M12).
4. **`update()` return value for the tabular agent** (`REVIEW_HISTORY.md` §5.3). Low priority; speeds up 10% grid tabular runs only, so it matters if C1 includes a tabular 10% comparison.
5. **Your report sections** (prospectus §4): Theory/Benchmark, Q-learning (including the constant-α versus 1/N discussion), and the FA extension.

## To decide with Albert

**From Albert, 2026-11-17 (updated 2026-11-18 after your #36): train, evaluate and check 5 are built** (`feature/tc-drift-train`; numbers in `TC_DRIFT_DESIGN.md` §5 "Check 5 measured" and §7).
- **Check 3:** you decided it on the policy while I measured the same thing in the replay at $1B, so we have two forms that agree: M = 4,000 minus M = 2,000 is −0.003 in your independent model and −0.11 ± 0.15 bps/yr in my replay. Nothing to decide.
- **Check 5:** your paired form, with one change for you to OK: the model side uses the value over the same 90 blocks (`evaluate.model_value`) rather than V and `value_of`, so both sides discount and truncate the same way. Replay minus model is −0.03 ± 0.08 (zero), −0.38 ± 0.43 ($100M), 0.01 ± 0.70 ($1B), 0.31 ± 1.25 ($10B). All inside 0.5, but resolved only at zero cost; at m = 1,000 a pass at $1B bounds the error to about ±1.4. Lattice rounding alone costs at most 0.1. Please confirm the form, and whether a two-SE bound is acceptable at the cost levels.
- **Your "not verified" item is now checked:** the oracle's trading in a continuous replay at M = 2,000 matches your in-model numbers (hold fraction 0.15 / 0.28 / 0.32 / 0.45 against your 0.156 / 0.272 / 0.308 / 0.443; turnover within 0.1 point).
- **D10:** the replayed lead over always-rebalance is 1.30 ± 0.70 at $1B (your in-model 1.32) and 7.10 ± 1.26 at $10B, and not measurably above zero at $100M. I set m once the tuned agent's gap SE is known; 4,000 is likely.
- **Cross-test:** we both added one. Mine also checks V to 1e-12; either can go.
- **For tuning:** `train()` runs at 0.57 ms a block with the current `DriftAgent` (about 9.5 minutes per million blocks per seed; a recursive-least-squares cost block adds a 2×2 solve per update). `src/tc_drift/config.py` has the probe states, the seeds and `RESTART_EVERY = 90`; `N_BLOCKS`, the checkpoint interval and the hyperparameters wait on your §9 choice.

~~**TC evaluation size** (D10): waiting for Albert.~~ Approved 2026-11-13: 90 blocks and 1,000 starts, as you proposed. `env.py` is built on `feature/tc-drift-env` (see `NEXT_STEPS_FOR_ALBERT.md`).

**Final runs.** `AUDIT_2026-11-06.md` §6. A TC option first, then C1 to C3, P1 and P2 as time allows.

**`all-three`'s comparison set** (L7).

**Which post-hoc runs go in the report.** `grid10-robustness` is the action-space check and γ = 3 and 8 the economic check. `b63` and `n26m` cover "different training budgets, episode lengths". `eps-decay` stays short, as mostly a negative result. `all-three` is the combined check `grid10-robustness` is compared against. The three constant-α runs go in the FA extension and in the Part III learning-rate discussion.
