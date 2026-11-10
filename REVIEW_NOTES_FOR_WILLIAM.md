# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-11-06
**Status:** the required work (Parts I to VI) is done and independently re-checked. The three constant-learning-rate runs (`tabular-const`, `fa-quadratic`, `fa-spline`) are run, merged into `dev` and reported. The transaction-cost sweep has run but its agent learns nothing; it needs a redesign. The whole-project audit is in `AUDIT_2026-11-06.md`. Closed items are in `REVIEW_HISTORY.md`.

---

## TL;DR / In plain terms

The audit found no correctness bug in the core code or in the FA code. It did find that the predictions we registered for the FA-era runs mostly failed (four of five checks, in whole or in part), and that two of our stated reasons were wrong; `EXPERIMENTS.md` §8.7 now says so plainly, and the report should too. The transaction-cost agent scores like a random portfolio even when trading is free, so the TC extension needs a design decision from both of us before any more TC runs. After that, pick the final runs and write.

**Key concepts to understand**

- **Shrinkage under a constant α.** With α constant and Q starting at 0, a portfolio updated N times holds about (1 − (1 − α)^N) of its true value. Portfolios tried only during exploration (about 1,100 times) end at about 0.3 of their value, measured at 0.29 to 0.30 in `tabular-const`'s logs. The inferred consequence (not tested by a counterfactual run) is that the greedy portfolio can then never be overtaken, which would explain why `tabular-const` "converged" in 28 of 30 seeds while finding the best portfolio in 2.
- **Unfinished FA directions.** Uniform exploration lasts about 1/β ≈ 1.41M episodes, not the 13M our §8.4 estimate assumed. At that count 36 of the 45 quadratic directions and 351 of the 369 spline directions barely move from θ = 0. This is the suspected cause of the γ = 8 miss (every FA seed holds 20% SPY where the best portfolio holds 40%); it is not tested yet.

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V.
- For the FA section: both FA models beat `tabular-const` at γ = 5 and 3 (paired −20.9 ± 9.8 and −99.2 ± 23.7 bps/yr for the quadratic) and fit the landscape better at every γ, but neither beats `baseline-v1` reliably. `fa-spline` fit the landscape better than the correctly specified quadratic on Spearman, against our prediction. No "discovery" claim.

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Benchmarks, grid, reward, tabular agent | `src/benchmarks/`, `src/action_space/`, `src/rewards.py`, `src/agent.py` | `REVIEW_HISTORY.md` §1; re-derived in the audit |
| Baseline and five tabular post-hoc runs | `results/`, `reports/` | `REVIEW_HISTORY.md` §8, `EXPERIMENTS.md` §7 |
| FA agent, features, training path, Q-value fit metrics | `src/fa_agent.py`, `src/features.py`, `src/train.py`, `src/evaluate.py` | `FA_INTEGRATION_NOTES.md`; audit found no bug |
| Step-size tuning | `src/tune_fa.py`, `results/tuning/` | `EXPERIMENTS.md` §8.1 |
| `tabular-const`, `fa-quadratic`, `fa-spline` run and reported | `results/<run>/`, `reports/<run>/` | `EXPERIMENTS.md` §8.7; report notes written 2026-11-06 |
| Whole-project audit | `AUDIT_2026-11-06.md` | Four independent reviewers |

425 tests on 2026-11-09: 424 pass, plus the known failure in item 1 below.

## From Albert, 2026-11-09: audit fixes on `dev`

**In plain terms:** the Albert-owned non-TC audit items (H3, M1 to M5, M10, L2, L4) are fixed in code and pushed to `dev`, and L8 is done for the runs whose logs are on the Mac; `AUDIT_2026-11-06.md` §3 marks each one. No reported number changed. Five things need you.

1. **Your test fails on `dev`, before and after these changes.** `src/tests/test_tune_fa.py` line 235, `test_a_clear_winner_is_decided_on_three_seeds`, asserts `shortfall_seed_se_bps == 0.0` and gets 5.0e-15 (floating-point rounding of a standard error over three equal values). Proposed fix: `== pytest.approx(0.0, abs=1e-12)`. Not changed, since the file is yours.
2. **L1 and L3 are yours and still open:** `generate_weight_grid` should raise when 1/increment is not an integer, and `requirements.txt` still says 244 tests and pins unused packages.
3. **M1 is fixed in `run_experiments.py` and `evaluate.py` only.** `tc_sweep_report.py` still needs the same `sys.stdout.reconfigure(encoding="utf-8")` line; it was left alone because TC work is on hold.
4. **After you pull, refresh your working tree once** so the new `.gitattributes` takes effect: `git rm --cached -r -q . ; git reset --hard` on a clean tree (commit or stash first). Text files then check out with LF, and `data/processed/` hashes the same on your machine as on the Mac. The code hashes line-ending-normalized bytes anyway (`src/hashing.py`), so nothing breaks if you skip this; and manifests you wrote on Windows before today are still accepted.
5. **Re-evaluate the runs whose logs only you have** (`b63`, `eps-decay`, `tabular-const`, `fa-quadratic`), so their summaries and `INDEX.md` gain the total gap, the post-hoc settled measure and provenance (audit L8). Same commands as before, for example `python -m src.run_experiments --run b63 --results-dir results/b63`; finished training is skipped and only the evaluation is rewritten. Then `python -m src.reporting.extract b63` and `python -m src.reporting.build --all`. Every earlier number should come back identical; on the Mac the four re-evaluated runs did.

## From Albert, 2026-11-09 (second round): flow-audit fixes

**In plain terms:** the flow audit (`AUDIT_2026-11-06.md` §8) found one break: the README's reproduce command crashed in a fresh clone because the training logs are not committed. That and the other Albert-owned items are fixed on `dev` (§8.3); no reported number changed. Three things need you.

1. **Re-extract your four runs after re-evaluating them** (item 5 above): `python -m src.reporting.extract <run>` for `eps-decay`, `b63`, `tabular-const` and `fa-quadratic`, then `python -m src.reporting.build --all`. Their `data.json` files are schema 2, so their reports show Q fit as n/a (F-M3). The new test `test_every_registered_report_builds_from_committed_data` compares each rebuilt report with the committed one, so commit the rebuilt reports with the data.
2. **Update the prospectus status** (F-M11). It still lists M10, M1, M2 and the Q-fit re-evaluation as open. Not changed, since the file is yours.
3. **Merging `dev` into `main`** (F-M1). `main`, GitHub's default branch, is about 50 commits behind, so a plain clone gets the old README. Proposed: one PR from `dev` to `main` once you have pulled and agree.

Also changed in shared code, all backward compatible: new manifests record scipy and pandas versions and hash `src/hashing.py` and `src/config.py`; evaluation provenance hashes `src/environment.py` and `src/action_space/grid.py`; a results folder whose manifest is committed but whose `.npz` logs are missing is now refused with a pointer to `--results-dir results/repro`.

## Your next steps

1. **Decide the TC redesign with Albert** (`AUDIT_2026-11-06.md` H1, H2, §6.2). The reviewer's suggestion splits the problem: learn each target's reward by sample average, as `baseline-v1` does, and plan over the known costs with the existing Bellman solver. Its one-seed test reached 0.8 bps/yr of the oracle at the 2 bp anchor. Decide whether that counts as model-free enough, whether to add drift, and which cost levels matter. Until then, do not run `discount.json` or `learning-rate.json`.
2. **Choose the final runs** from `AUDIT_2026-11-06.md` §6. The strongest FA candidate is FA on the 10% grid (C1), where tabular struggled most; a γ = 8 check (C2) would test the unfinished-directions explanation. Register each with an expected direction before it runs.
3. **Write the FA section** using `EXPERIMENTS.md` §8.7, including the failed expectations, the measured shrinkage and the tuning disclosure (tuned by simulating the same environment, on separate seeds, at γ = 5 only; `AUDIT_2026-11-06.md` M12).
4. **Merge run branches through PRs.** The hand-resolved `registry.json` conflict in the `fa-quadratic` merge left invalid JSON on `dev` (repaired on `feature/project-audit`). A test that loads the registry and builds every report would catch this (`AUDIT_2026-11-06.md` L6).
5. **`update()` return value for the tabular agent** (`REVIEW_HISTORY.md` §5.3). Low priority; speeds up 10% grid tabular runs only, so it matters if C1 includes a tabular 10% comparison.
6. **Your report sections** (prospectus §4): Theory/Benchmark, Q-learning (including the constant-α versus 1/N discussion), and the FA extension.

## To decide with Albert

**TC redesign** (step 1 above).

**Final runs.** `AUDIT_2026-11-06.md` §6: R0 (re-evaluations, no training) first, then a TC option, C1 to C3, P1 and P2 as time allows.

**Which post-hoc runs go in the report.** `grid10-robustness` is the action-space check and γ = 3 and 8 the economic check. `b63` and `n26m` cover "different training budgets, episode lengths". `eps-decay` stays short, as mostly a negative result. `all-three` is the combined check `grid10-robustness` is compared against. The three constant-α runs go in the FA extension and in the Part III learning-rate discussion.
