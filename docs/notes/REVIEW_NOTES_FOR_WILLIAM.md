# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-11-09, updated 2026-11-19
**Status:** open items only. Closed items are in `REVIEW_HISTORY.md` and `AUDIT_2026-11-06.md` §2.

---

## TL;DR / In plain terms

What is left is the transaction-cost rebuild and the report. The TC redesign is decided (`TC_DRIFT_DESIGN.md`): three ETFs with drift, realistic costs, an oracle benchmark and a spline agent. Most of it is built (`AUDIT_2026-11-06.md` §9.2); the agent's learning rule, a run driver and the runs remain. Nothing of the report is drafted yet, and it includes the FA expectations that failed.

**Key concepts to understand**

- **Shrinkage under a constant α.** With α constant and Q starting at 0, a portfolio updated N times holds about (1 − (1 − α)^N) of its true value. Portfolios tried only during exploration (about 1,100 times) end at about 0.3 of their value, measured at 0.29 to 0.30 in `tabular-const`'s logs. The inferred consequence (not tested by a counterfactual run) is that the greedy portfolio can then never be overtaken, which would explain why `tabular-const` "converged" in 28 of 30 seeds while finding the best portfolio in 2.
- **Unfinished FA directions.** Uniform exploration lasts about 1/β ≈ 1.41M episodes, not the 13M our §8.4 estimate assumed. At that count 36 of the 45 quadratic directions and 351 of the 369 spline directions barely move from θ = 0. This is the suspected cause of the γ = 8 miss (every FA seed holds 20% SPY where the best portfolio holds 40%); it is not tested yet.

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V.
- For the FA section: both FA models beat `tabular-const` at γ = 5 and 3 (paired −20.9 ± 9.8 and −99.2 ± 23.7 bps/yr for the quadratic) and rank the landscape better at every γ on Spearman, but neither beats `baseline-v1` reliably. `fa-spline` ranked the landscape better than the correctly specified quadratic, against our prediction. No "discovery" claim.

---

## Still open

The 2026-11-19 status review (`AUDIT_2026-11-06.md` §9) numbers the new TC items TC-1 to TC-13 and the decisions D-A to D-I. You decided D-A, D-B, D-D and D-H on 2026-11-19 (below), and Albert D-E (m = 4,000) on 2026-11-20; D-C was settled in practice when you ran the tuning pilot. D-F, D-G and D-I are open.

| Item | What | Waits on |
|---|---|---|
| M12, F-L10 | Report text: the tuning disclosure, caveats on weights where no seed converged | The report draft |
| F-L9 | Episodes column in `performance.csv` (partly fixed: `INDEX.md` gives days = episodes × B) | Low priority; or say it in the report |
| L7 | Whether `all-three`'s report should compare with its single-change runs | A decision with Albert |
| F-L13 | Tag `grid10-robustness` exists only on Albert's Mac | Closed: the tag is on `origin` at `e13a9a6` (checked 2026-11-20) |
| TC-1, tuning | Decided and done: recursive least squares on the cost block after a constant-α burn-in, θ at constant α_value, 4M blocks, ε 1% at halfway. Pilot 2026-11-19: **α_value = 3e-4** (paired difference 9.31, SE 6.32). The pilot's other values confirmed 2026-11-20: burn-in to ε = 0.5 (301,030 updates), α_cost = α_value, target refresh 1,000, l2 = 0 (`TC_DRIFT_DESIGN.md` §9) | Closed |
| TC-3, target error | In the 2026-11-15 pilot at α = 0.001 the target the agent traded to from the optimum is worth 10.4 bps/yr less than the best one (50 to 118 at α = 0.01), against a $1B timing value of 1.30 ± 0.70. A "favourite target" reference policy splits the agent's gap into favourite-target error and the rest | Built 2026-11-20 (Albert, PR #41): `evaluate.own_target_rebalance_policy`, and the split will be in every level's `summary.json` |
| TC-5, D12 | Changed 2026-11-19 (favourite target within one 2% step; tail-averaged θ) and the secondary restated 2026-11-20: 95% of probes with post-decision holdings within one 2% step throughout the window, labelled (`TC_DRIFT_DESIGN.md` §6). On the pilot at α_value = 3e-4 one run of three converges (seed 0, from block 3,284,100); expect a low convergence rate in the final runs and report it as Part IV asks. No further changes to the rule: the next one would be made with the pilot's results in full view | The run driver applies it (`tune.convergence` with probes; built 2026-11-20, PR #41) |
| TC-2, TC-6, TC-7 | Run driver (with BLAS pinned to one thread), θ in the training log (c1 and c2 already are), and m = 4,000 fixed before any agent result (D10, Albert's) | Built 2026-11-20 (Albert, PR #41); m = 4,000 decided. Runs not started |
| TC-9 | Albert's `feature/tc-drift-train` edits the prospectus and these notes | Closed: merged 2026-11-19 (PR #37) |
| TC exploration | Even at the least-squares floor, a drift trade's cost is uncertain by about two thirds after 500,000 exploring trades; the pilot's β explored about 50,000 blocks (§9) | Closed with D-B (2026-11-19): ε reaches 1% at halfway, about 434,000 exploring blocks |
| TC check 5 | Measured by Albert 2026-11-17: inside 0.5 bps/yr at every level, resolved only at zero cost. He asks whether the 90-block model side and a two-SE bound are acceptable (audit §9.4 D-I) | You |
| TC D7 | Pointer added to `AUDIT_2026-11-06.md` §6.2 on 2026-11-19; the one in `TRANSACTION_COSTS.md` is Albert's | Closed: Albert added the `TRANSACTION_COSTS.md` pointer 2026-11-20 (`c00f8c4`) |
| TC report | Two registered oracle directions not met (trade-once does not approach the oracle as costs grow; myopic holds less than the oracle under λ only). The oracle's in-sample V is about 5 bps/yr high at M = 2,000 | The TC section |
| TC-4, report | Not started; the prospectus planned a first draft by Nov 19. Its week 10 ("light schedule") clashes with the design's Nov 20 to 26 TC runs | Go/no-go for the agent decided: Nov 24. The report schedule still to agree with Albert (audit §9.4 D-H) |
| TC-13, disclosure | The target error and the true c1, c2 (from μ, Σ, κ, λ) inform the learning-rule choice; disclose it in design §14 risk 5 | You |
| D-B timing | 0.57 ms a block is Albert's machine. Yours: 4M blocks took 2.4 hours a run with six runs in parallel and BLAS pinned (2026-11-19, a quiet machine); 8.1 hours when the laptop was on low battery with Zoom and Chrome busy. The final runs are 40 of them | Re-timed 2026-11-20 on Albert's Mac, where the runs go: 0.26 ms a block alone, 0.39 with seven in parallel |
| TC Part IV counts | Done 2026-11-20: `tune.convergence` reports the first checkpoint from which both rules hold through the end, in blocks, episodes and daily observations; the run driver needs to call it | Built 2026-11-20: `run.py` writes `converged_from_*` to `per_seed.csv` (PR #41) |
| TC disclosure of the averaging change | The tail-average change was decided after seeing the pilot's oracle gaps and utility losses (design §6 says so). Disclose in the report with §14 risk 5 | The TC section |
| TC c1 | The pilot learns c2 well (0.94, 0.49, 1.01 of true at α_value = 3e-4) but c1 poorly (0.41, 3.19, 0), and c1 is the larger part of a small trade's cost (3.2e-7 against 1.5e-7 on a typical drift trade). Report it; it is the exploration limit §9 predicted | The TC section |

## Your next steps

1. **Build your Phase 2 TC files** (`TC_DRIFT_DESIGN.md` §5, §6, §9, §10):
   - `src/tc_drift/oracle.py`: merged. Checks 1 to 4 pass; the Nov 17 gate is met. Check 3 restated on the policy (decided 2026-11-17, post hoc) and scored in an independent model: mean difference at most 0.003 bps/yr at every setting (0.014 at one start), but uninformative at zero cost and $100M, where the oracle's own lead over always rebalancing is under the tolerance. M stays at 2,000. The V-based failure was a shared level shift in r̂, not maximization bias (`TC_DRIFT_DESIGN.md` §5).
   - **Oracle run on all six settings done** (`feature/tc-drift-oracle-runs`, `results/tc-drift/oracle/`, `TC_DRIFT_DESIGN.md` §8). Two registered directions failed: trade-once does not approach the oracle as costs grow, and myopic holds slightly less than the oracle under λ only. Report both.
   - Check 5: the paired form is decided (2026-11-17); Albert measured it on 2026-11-17 (below) and asks two questions (D-I).
   - `src/tc_drift/features.py` and `src/tc_drift/agent.py`: built and merged (PR #35), reviewed, all tests pass (`TC_DRIFT_DESIGN.md` §6, "Implemented").
   - **The tuning design is blocked on a decision.** The $1B pilot (§6) shows a constant α cannot learn the cost coefficients at the real noise level: after 500,000 blocks they are off by factors of 3 to 10. §9 now has a synthetic analysis of the cost block (`src/tc_drift/cost_learning_analysis.py`): decorrelating and decaying together reach the least-squares floor only with a b taken from the optimum (benchmark information); recursive least squares on the cost block reaches it model-free. Even the floor leaves a drift trade's cost uncertain by about two thirds after 500,000 exploring trades, so exploration length is the next limit. Pick the option, then pilot it. *Albert, 2026-11-17:* `train()` and `config.py` are ready for the pilot; once you pick, set `N_BLOCKS`, `CHECKPOINT_EVERY` and `HYPERPARAMETERS` in `config.py` or tell Albert the values. D12 may be changed if the pilot fails it, before any final result is examined, labelled as a change (decided 2026-11-17); the 2026-11-19 review recommends changing it before the tuning pilot instead (D-D).
   - **TC next steps, in order** (updated 2026-11-19, `AUDIT_2026-11-06.md` §9):
     1. ~~Open a PR from `feature/tc-drift-oracle-runs` into `dev`~~ Merged (#36, 2026-11-18).
     2. ~~Decide D-A, D-B, D-D, D-H~~ Decided 2026-11-19. Still to decide: D-C (who runs tuning, the α_value candidates), D-I (Albert's check 5 questions), the burn-in length; with Albert, D-F, D-G and the report schedule.
     3. ~~Implement the chosen rule in `agent.py`~~ Done 2026-11-19 on `feature/tc-drift-rls`, with `favourite_target()` and `with_parameters()`; tested against a batch ridge solve and `cost_learning_analysis.py`.
     4. ~~Review your files on Albert's `feature/tc-drift-train` and merge it~~ Merged 2026-11-19 (PR #37).
     5. ~~Tuning pilot at $1B~~ Run 2026-11-19 (`results/tc-drift/tuning/`): α_value = 3e-4. Diagnostics in design §9; disclose them (TC-13). Still to do: confirm the pilot's other values, decide the secondary rule, commit the branch and give Albert the hyperparameters.
     6. Register the agent's expected directions (D-F) with Albert, including the TC-3 split, then the 40 runs. Go/no-go Nov 24: if not tuned by then, report the oracle and a diagnosed null agent result.
2. **Choose the final runs** from `AUDIT_2026-11-06.md` §6. C1 (FA on the 10% grid) is done (`EXPERIMENTS.md` §8.8). C2, C3, P1 and P2 are not run; the 2026-11-19 review recommends none, except P2 if one of us has a free day (§9.4 D-H). Register any run with an expected direction before it starts.
   - A run trained from the current `dev` will show "Training code changed: src/action_space/grid.py, src/train.py" against every earlier run. The flag is honest but no algorithm changed: the edits are LF writes and the increment guard, and `grid_sha256` still matches. Say so in each new run's caveats.
3. **Write the FA section** from `AUDIT_2026-11-06.md` §2.3 (the experiment, the failed expectations and what can be claimed, all re-checked) and `EXPERIMENTS.md` §8.7, including the measured shrinkage and the tuning disclosure (tuned by simulating the same environment, on separate seeds, at γ = 5 only; `AUDIT_2026-11-06.md` M12).
4. **`update()` return value for the tabular agent** (`REVIEW_HISTORY.md` §5.3). Low priority; speeds up 10% grid tabular runs only, and C1 ran without a tabular 10% comparison, so probably not needed.
5. **Your report sections** (prospectus §4): Theory/Benchmark, Q-learning (including the constant-α versus 1/N discussion), and the FA extension.

## To decide with Albert

**From Albert, 2026-11-20: the final runs are ready** (`feature/tc-drift-run`).
- **Built.** `src/tc_drift/run.py` (TC-2): `python -m src.tc_drift.run all` trains 4 levels × 10 seeds, 7 at a time with BLAS pinned, then scores each level on m = 4,000 paths against the oracle, myopic, always rebalance, trade once then hold, and the own-target policy (TC-3, with the split checked as an identity on every path). It reports the tail-averaged policy as the headline and the last iterate alongside. All four D12 verdicts and the Part IV counts come from your `tune.convergence` and `tune.averaged_policy`. `train()` now logs θ and the favourite target at every checkpoint (TC-6). A test checks that one call gives exactly your `train_segmented` run.
- **Decided by Albert:** m = 4,000, fixed before any agent result (TC-7, D-E).
- **Checkpoint interval: 40,050, not the 40,000 you suggested.** It equals `tune.SEGMENT`, so the final runs have the layout design §6 registered and the pilot used: 11 tail-average snapshots and a 12-checkpoint D12 window from block 3,564,450. With 40,000 there would be 10 and 11. Found in the independent review.
- **Expected directions (D-F).** My preliminary E1 to E12 are in `TC_DRIFT_DESIGN.md` §7, written before the runs. Please add or amend them. Anything added after the runs start gets labelled post hoc.
- **Timing on my Mac:** 0.39 ms a block with 7 runs at once, so about 26 minutes a run and about 3 hours for all 40.
- **Your files:** I did not edit `tune.py`. Its `run_one` defaults to `m = config.M_EVAL`, which is now 4,000, while your pilot used 1,000. A rerun of `tune train` would therefore differ from the pilot unless it passes `m=config.M_EVAL_PILOTS`. Your call.
- **D7:** the pointer in `TRANSACTION_COSTS.md` is added.

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
