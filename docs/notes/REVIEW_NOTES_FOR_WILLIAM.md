# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-11-09, updated 2026-11-25
**Status:** open items only. Closed items are in `REVIEW_HISTORY.md`, `AUDIT_2026-11-06.md` §2 and the two "Closed" lists below. Albert's audit (`AUDIT_2026-11-20.md` §6) is the correction plan for what is left. Audit IDs: a bare ID (M8, F-L9, L7) is from `AUDIT_2026-11-06.md`; an ID prefixed A20 (A20-H1, A20-M2) is from `AUDIT_2026-11-20.md`.

---

## TL;DR / In plain terms

Everything is run, and the report and slides are written (final draft 2026-11-25). Two decisions are left with Albert (the E6 verdict, and making the final draft `REPORT.md`), then the merge into `main`, a fresh-clone check and submission by Nov 30.

**Key concepts to understand**

- **Shrinkage under a constant α.** With α constant and Q starting at 0, a portfolio updated N times holds about (1 − (1 − α)^N) of its true value. Portfolios tried only during exploration (about 1,100 times) end at about 0.3 of their value, measured at 0.29 to 0.30 in `tabular-const`'s logs. The inferred consequence (not tested by a counterfactual run) is that the greedy portfolio can then never be overtaken, which would explain why `tabular-const` "converged" in 28 of 30 seeds while finding the best portfolio in 2.
- **Unfinished FA directions.** Uniform exploration lasts about 1/β ≈ 1.41M episodes, not the 13M our §8.4 estimate assumed. At that count 36 of the 45 quadratic directions and 351 of the 369 spline directions have not finished learning (wording per `AUDIT_2026-11-06.md`). This is the suspected cause of the γ = 8 miss (every FA seed holds 20% SPY where the best portfolio holds 40%); it is not tested yet.
- **Where the TC agent heads versus when it trades.** Its gap to the oracle splits exactly into the cost of its favourite target and the value of its timing, minus the oracle's own lead over always-rebalancing (`TC_DRIFT_DESIGN.md` §7, TC-3).

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V.
- For the FA section: both FA models beat `tabular-const` at γ = 5 and 3 (paired −20.9 ± 9.8 and −99.2 ± 23.7 bps/yr for the quadratic) and rank the landscape better at every γ on Spearman, but neither beats `baseline-v1` reliably. `fa-spline` ranked the landscape better than the correctly specified quadratic, against our prediction. No "discovery" claim.
- For the TC section: agent − oracle is −13.1 / −13.1 / −10.0 / −2.8 bps/yr at zero / $100M / $1B / $10B, all 10 seeds below the oracle at every level. Where it heads (split: own target − always rebalance) is −13.1 / −12.8 / −9.0 / −1.1; its timing is worth +6.2 at $10B, where it beats always-rebalancing by 5.0. A separate diagnostic, the favourite target's utility loss (computed with μ and Σ), is 12.9 / 13.0 / 9.75 / 1.79. E1 to E12: 8 met, 4 partly met (`results/tc-drift/runs/README.md`).

---

## Still open (2026-11-25)

| Item | What | Who |
|---|---|---|
| E6 verdict | Table A3 marks E6 ("agent loses to myopic at every level") as met. At $10B the shortfall (−0.5, SE 0.54, ahead on 5 of 10 seeds) is not distinguishable from zero, so "partly met" may fit better. Changing it changes the eight-met count in report §6.4, the Table A3 caption, `results/tc-drift/runs/README.md` and `reports/tc-drift/` (`REPORT_final_CHANGES.md`) | Both |
| Promote the final draft | Copy `REPORT_final.md` over `REPORT.md` once accepted, so `sh docs/report/build.sh` and Appendix B's build command reproduce the submitted PDF. v1 stays in git history | Both |
| `dev` into `main` | `dev` is 8 commits ahead of `main` on 2026-11-25 (`main` last updated by PR #52, 2026-11-24). Merge after the last change, then a fresh-clone check (tests, data rebuild and `run_experiments --dry-run`; no retraining) | Both |
| Merged remote branches | Merged into `dev` and safe to delete on GitHub: `feature/docs-after-tc-runs`, `feature/report-draft`, `feature/report-final`, `feature/report-remove-drafting-notes`, `feature/report-skeleton-tc-update`, `feature/report-v2-aa`, `feature/stale-text-cleanup`, `feature/tc-drift-run`, `run/tc-drift-final` | Either |

## Your next steps

1. **With Albert:** decide E6 and whether to promote the final draft.
2. **Before submission:** merge `dev` into `main`, run the fresh-clone check, submit by Nov 30.

## Closed 2026-11-25

**TL;DR / In plain terms:** the report and slides are written, and they settle every open writing item. The code items are done (Albert, `58d23e5`) or no longer needed.

- **Report:** final draft `docs/report/REPORT_final.md` and `.pdf` (PRs #53 and #54), from Albert's v1 (2026-11-24) and v2, then your markup, shortened to 13 pages of main text and references. Changes in `REPORT_final_CHANGES.md`. Your sections (Theory/Benchmark, Q-learning, FA) are in it.
- **Presentation:** `docs/report/slides/`, 15 slides plus 3 backup, matched to the final draft, with a title slide and a pipeline diagram (slide 3). Notes in `SPEAKER_NOTES.md`.
- **A20-H2:** the derivation is in report §3. No separate `BENCHMARKS.md` was written.
- **A20-H3:** `src/benchmarks/closed_form_table.py` writes `results/benchmarks/closed_form_budget_only.csv` and checks it against a numerical solve (agreement 1.47e-6; Albert, `58d23e5`).
- **A20-H4:** no script committed. The report words the γ = 8 cause as "one hypothesis, which we have not tested" and reports neither the surrogate table nor the shrinkage slopes.
- **A20-M1:** disclosed in Appendix C, item 9 (the oracle summary and tuning records came from uncommitted code; the final agent runs did not). The oracle was not rerun.
- **A20-M2:** the six TC oracle tables are committed (`58d23e5`).
- **TC section:** written. It covers the split, the unexplained target improvement at $10B, the cost-coefficient results and the registered expectations (Table A3).
- **D-G (TC-8):** the first design gets one paragraph (report §6.4) and Figure A13, and no tables. So M7, M8, M9, M11, the TC halves of M1 and F-L12, and H2's `ignore_cost` column are closed as not reported (A20-M4).
- **Post-hoc coverage (A20-M8):** settled by report §6, which follows your 2026-10-28 paragraph under History.
- **M12:** the FA tuning disclosure is in report §6.3. **F-L9:** report Table 3 gives episodes and days (13M episodes, 260M days).
- **Report text, non-TC** (L9, L10, A20-M5, and F-L10): written into the report. A20-M5 is in §5 ("Gaps use the known μ and Σ") and L10 in §6.3. The `RUN_ISSUES_2026-10-22.md` findings 4, 5, 8 and 9 and F-L10 were not rechecked one by one against the final text.
- **L7** (`all-three`'s comparison set): left as it is. The report does not depend on it.

## Closed 2026-11-20 to 2026-11-22

**TL;DR / In plain terms:** the TC runs are done and checked, your three suggestions to Albert are settled, and two small items need no work.

- **TC agent runs** (Albert, 2026-11-20, PR #47): 40 runs from clean commit `1787f0d`, scored on m = 4,000 paths, with the HTML report (`src/tc_drift/report.py`, tested; `reports/tc-drift/report.html`). Albert recomputed the per-seed gaps from `values.npz` (within 1e-13). On 2026-11-22 I re-derived each level's agent − oracle mean and across-seed SD, the seed-averaged path SE, agent − always-rebalance, agent − myopic, the lowest seed against trade once then hold, the split and the oracle's lead with its SE from the per-path values; every one matches `results/tc-drift/runs/README.md` to the printed digit, and the split sums to agent − oracle on every path within 6e-14.
- **TC go/no-go (Nov 24):** met.
- **Your three `run.py` suggestions:** Albert took the first two (the seed `.npz` files and `values.npz` are committed; the seed-averaged gap's path SE is reported). The third, batched scoring, was not needed: scoring took about 13 minutes a level.
- **PR #43** (`feature/tc-drift-runs`) closed unmerged 2026-11-20. Its commits stay on GitHub under `refs/pull/43/head`.
- **Final non-TC runs:** none (2026-11-22). C2, C3, P1 and P2 are not run.
- **`update()` return value** (`REVIEW_HISTORY.md` §5.3): not needed, since no further 10% grid tabular run is planned.
- **Albert's audit edits to your files** (`AUDIT_2026-11-20.md` §3.1, §3.2): the test count (757 to 758), the PR #45 merge, PR #43 marked closed, and "barely move from θ = 0" reworded to "not finished learning" (his audit's L7, not ours). Seen 2026-11-22: the merge matches `origin/main`; PR #43's closed state comes from Albert's `gh` check and is not re-checked here (`gh` is not installed on this machine).
- **Local branches:** the seven whose remotes were deleted are pruned on this machine (2026-11-22). `origin/feature/tc-drift-run` and `origin/run/tc-drift-final` are fully merged into `dev` and can be deleted on GitHub.
- **Windows tests:** 760 of 761 passed on Windows (Python 3.10.2, numpy 2.2.6, matplotlib 3.10.9) on 2026-11-22; the one failure, `test_committed_report_matches_a_rebuild`, comes from the matplotlib version changing the SVG bytes, and `data.json` rebuilds identically (detail in `NEXT_STEPS_FOR_ALBERT.md`).

## History

**From Albert, 2026-11-20: the final runs are ready** (`feature/tc-drift-run`).
- **Built.** `src/tc_drift/run.py` (TC-2): `python -m src.tc_drift.run all` trains 4 levels × 10 seeds, 7 at a time with BLAS pinned, then scores each level on m = 4,000 paths against the oracle, myopic, always rebalance, trade once then hold, and the own-target policy (TC-3, with the split checked as an identity on every path). It reports the tail-averaged policy as the headline and the last iterate alongside. All four D12 verdicts and the Part IV counts come from your `tune.convergence` and `tune.averaged_policy`. `train()` now logs θ and the favourite target at every checkpoint (TC-6). A test checks that one call gives exactly your `train_segmented` run.
- **Decided by Albert:** m = 4,000, fixed before any agent result (TC-7, D-E).
- **Checkpoint interval: 40,050, not the 40,000 you suggested.** It equals `tune.SEGMENT`, so the final runs have the layout design §6 registered and the pilot used: 11 tail-average snapshots and a 12-checkpoint D12 window from block 3,564,450. With 40,000 there would be 10 and 11. Found in the independent review.
- **Expected directions (D-F).** My preliminary E1 to E12 are in `TC_DRIFT_DESIGN.md` §7, written before the runs. Please add or amend them. Anything added after the runs start gets labelled post hoc.
- **Timing on my Mac:** 0.39 ms a block with 7 runs at once, so about 26 minutes a run and about 3 hours for all 40.
- **Your files:** I did not edit `tune.py`. Its `run_one` defaults to `m = config.M_EVAL`, which is now 4,000, while your pilot used 1,000. A rerun of `tune train` would therefore differ from the pilot unless it passes `m=config.M_EVAL_PILOTS`. Your call.
- **D7:** the pointer in `TRANSACTION_COSTS.md` is added.

*William, 2026-11-20:* all answered. E1 to E12 stay as registered, unchanged (D-F). `tune.run_one` now defaults to `m = config.M_EVAL_PILOTS`, with a test. 40,050 is right. Check 5's form: yes to both (D-I, design §5).

**Which post-hoc runs go in the report.** `grid10-robustness` is the action-space check and γ = 3 and 8 the economic check. `b63` and `n26m` cover "different training budgets, episode lengths". `eps-decay` stays short, as mostly a negative result. `all-three` is the combined check `grid10-robustness` is compared against. The three constant-α runs go in the FA extension and in the Part III learning-rate discussion.
