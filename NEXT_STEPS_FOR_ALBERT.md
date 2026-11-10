# Next Steps

**From:** William
**Date:** 2026-11-06, updated 2026-11-09
**Status:** the required work is done and independently re-checked. The FA runs are merged and reported. The whole-project audit (`AUDIT_2026-11-06.md`) found no correctness bug in the core, the FA code or the reports pipeline. It did find that the transaction-cost agent cannot learn as designed, plus a list of small fixes, most of them in your files.

---

## TL;DR / In plain terms

Three things need you. First, re-evaluate four runs from your training logs so their Q-value fit numbers exist in the repo; `fa-spline`'s notes quote `baseline-v1` numbers that nothing committed contains. Second, decide with me how to redesign the transaction-cost extension: in your sweep the agent scored the same as a random portfolio even with free trading, and the code is correct, so the problem is the design. Third, a batch of small code fixes that change no number.

**Key concepts to understand**

- **Why the TC agent cannot learn.** It keeps one value per (holding, target) pair: 1.66 million of them. Exploration supplies about 1.41M random actions, so 57% of pairs are ever tried, most of them once, against reward noise of about 3,900 bps/yr per block. With α = 1/N and discount 0.95, the estimated continuation value is also noisy and biased by how often each holding was visited. At zero cost the learned policies' median, 56.6 bps/yr, matches the average grid portfolio's 56.7.
- **Why costs barely matter here.** Targets never drift, so the best policy makes one move. The oracle beats simply ignoring costs by 0.3 bps/yr at the 2 bp anchor. A rebalancing problem needs drift, or larger costs.
- **Line-ending hashes.** Windows checks files out with CRLF line endings, your Mac with LF, so the same file hashes differently. Every "calibration changed" flag in the reports is this.

**Key takeaways**

- Nothing changes your Part I to V results.
- The TC report's text says things the results do not support (`TRANSACTION_COSTS.md`, Outcome). I added a status block and an Outcome section to `TRANSACTION_COSTS.md` but left your design text as it was.

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Data pipeline, simulator, benchmarks, grid, reward | `src/data/`, `src/environment.py`, `src/benchmarks/`, `src/action_space/`, `src/rewards.py` | Rebuilt and re-derived independently in the audit |
| Agent, runner, evaluation, batch runner, reports | `src/agent.py`, `src/train.py`, `src/evaluate.py`, `src/run_experiments.py`, `src/reporting/` | Every seed in every report matches its results to 1e-6 |
| Baseline and tabular post-hoc runs | `results/`, `reports/` | `REVIEW_HISTORY.md` §8, `EXPERIMENTS.md` §7 |
| FA runs `tabular-const`, `fa-quadratic`, `fa-spline` (your `fa-spline` run included) | `results/<run>/`, `reports/<run>/` | `EXPERIMENTS.md` §8.7 |
| TC implementation and quadratic-cost sweep | `src/transaction_costs.py`, `results/tc-quadratic-cost/` | `TRANSACTION_COSTS.md`, Outcome |

442 tests pass on 2026-11-09 (Windows, Python 3.10); the rounding failure in `test_tune_fa.py` is fixed.

## Changes the audit made to your files

All documentation or data, no code:
- `reports/registry.json`: repaired (the `fa-quadratic` merge left invalid JSON on `dev`); notes written for `tabular-const` and `fa-quadratic`; `fa-spline` got the outcome of its registered expectation, an explicit `compare_with` (`baseline-v1`, `tabular-const`, `fa-quadratic`) and a caveat that its `baseline-v1` Q-fit numbers are not committed. Every other run got an explicit `compare_with` equal to the set its report already used. All reports rebuilt; per-run data identical.
- `EXPERIMENTS.md` (status line, §6 shared-streams note, new §8.7 outcome), `EVALUATION.md` (constant-α notes, test count, γ justification done), `ENVIRONMENT.md` and `DATA_CLEANING.md` (stale "still open" items), `reports/README.md` (how variants are run now; the TC report), `TRANSACTION_COSTS.md` (status block and Outcome section).

## From William, 2026-11-09: your items for me are closed

**In plain terms:** everything you left in `REVIEW_NOTES_FOR_WILLIAM.md` outside transaction costs is done, on `feature/william-audit-fixes` (`AUDIT_2026-11-06.md` §2.2, `REVIEW_HISTORY.md` §9). While doing it I found one bug in the line-ending work and fixed it in your files, as below.

- **Closed:** the rounding test; L1 (grid increment check); L3 (`requirements.txt`); the working-tree refresh; `b63`, `eps-decay`, `tabular-const` and `fa-quadratic` re-evaluated at clean commit `3322e03`, re-extracted (schema 3) and every report rebuilt, so Q fit shows everywhere (L8, F-M3); the prospectus status (F-M11); and `tune_fa --summarize` is now byte-stable (F-L12, FA part). Every earlier number came back identical, checked field by field.
- **Registry:** the two caveats (`tabular-const` and `fa-quadratic`) that said the table "reads n/a until this run's data.json is re-extracted" now say it was re-extracted on 2026-11-09.
- **A bug in M2, fixed in your files.** `Path.write_text` without `newline=` writes CRLF on Windows. After a fresh checkout (LF, from `.gitattributes`), rebuilding a report on Windows produced CRLF bytes, so `test_every_registered_report_builds_from_committed_data` failed and `build --all` would rewrite every report with only line endings changed. Reproduced by converting one report to LF and running the test. Every non-TC `write_text` now passes `newline="\n"` (and `encoding="utf-8"` where it was missing): `src/reporting/build.py`, `src/reporting/extract.py`, `src/evaluate.py`, `src/run_experiments.py`, `src/data/clean.py`, `src/data/download_raw.py`, plus `src/train.py` and `src/tune_fa.py`. Your CSV writers already pinned LF. Two tests in `test_reporting.py`: a source check that every `write_text` outside the TC files sets `newline`, which fails on every machine including the Mac, and a check that a built report has no CRLF. The two TC writers (`tc_sweep_report.py`, `transaction_costs.py`) are exempt until the redesign.
- **Proposed, not done: merge `dev` into `main`** (F-M1), as one PR once `feature/william-audit-fixes` is merged into `dev`. Tell me if you agree.

## Your next steps

**Update 2026-11-09 (Albert):** steps 1, 3 and 4 are done on `dev`; `AUDIT_2026-11-06.md` §2.1 lists each fix and how it was checked. The second (flow) audit's Albert items are fixed too (§8.3), including the fresh-clone reproduce crash (F-H1). No reported number changed. Still Albert's: merging `dev` into `main` with William (F-M1), pushing or deleting the local `grid10-robustness` tag (F-L13), and caveating post-convergence weights in the report text (F-L10).

1. ~~**Re-evaluate `baseline-v1`, `n26m`, `all-three` and `grid10-robustness`**~~ Done 2026-11-09 (H3), plus `fa-spline`. Every earlier number came back identical. The `baseline-v1` Q fit matches the numbers in `fa-spline`'s notes (Spearman 0.821, 0.710, 0.920; top-10 recall 35%, 20%, 20% at γ = 5, 3, 8). `b63`, `eps-decay`, `tabular-const` and `fa-quadratic` need William, who holds their logs.
2. **TC redesign, with William** (`AUDIT_2026-11-06.md` H1, H2, §6.2). On hold. Do not run `discount.json` or `learning-rate.json` as designed. Once decided, also: commit the four single-variant configs (M9), rewrite the TC report text (M8), add the M1 line to `tc_sweep_report.py`, and add TC tests (M11).
3. ~~**Small code fixes**~~ Done 2026-11-09: M1 (except `tc_sweep_report.py`), M2, M3, M4, M5, plus L2.
4. ~~**`RUN_ISSUES` §4 steps 1 to 3** (M10)~~ Done 2026-11-09: total gap, post-hoc `settled` and `near_optimal_every_episode`, provenance in `summary.json`, definitions in `EVALUATION.md` §2.
5. **Your report sections** (prospectus §4): Data/Calibration, and convergence analysis and visualizations. Use the post-hoc `settled` measure next to the pre-registered rule, labelled post hoc. Who writes the TC section is proposed in the prospectus (you, since you built it) and is yours to accept or change.

## To decide together

**TC redesign** (step 2), and **the final runs** (`AUDIT_2026-11-06.md` §6).

**Which post-hoc runs go in the report.** My suggestion:
- `grid10-robustness` as the action-space check, and γ = 3 and 8 as the economic check;
- `b63` and `n26m` for training budgets and episode lengths;
- `eps-decay` short, as mostly a negative result;
- `all-three` as the combined check the 10% grid is compared against (and whether its report should compare with its single-change runs, L7);
- the three constant-α runs in the FA extension.
