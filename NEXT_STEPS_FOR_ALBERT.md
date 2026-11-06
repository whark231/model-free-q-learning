# Next Steps

**From:** William
**Date:** 2026-11-05
**Status:** the baseline and all five post-hoc runs are done and merged into `dev`. The function-approximation (FA) extension is built, its step size is tuned, and three constant-learning-rate runs are registered: `tabular-const`, `fa-quadratic` and `fa-spline`. They wait on your review. The professor agreed to a constant α for FA and a separately tuned constant α for the tabular comparison.

---

## TL;DR / In plain terms

The FA work changed several of your files, with my permission and Claude's documentation. Every edit is listed in `FA_INTEGRATION_NOTES.md` §1. Before the three new runs start, I need two things from you: a review of those edits, and your agreement to the expected results written in `EXPERIMENTS.md` §8. Those must be fixed before anything runs. The rest of your list affects the report tables, not the runs.

**Key concepts to understand**

- **Constant α.** `baseline-v1` uses α = 1/N, a running average whose noise shrinks with every visit. A constant α keeps a noise floor: about 49 bps/yr per Q-value for tabular at α = 3.2e-4. The new runs test what that costs, with and without shared features.
- **Why `tabular-const`.** It has the same α, seeds and settings as `fa-quadratic`, so the pair differs only in the features.
- **Q-value fit.** New evaluation metrics ask whether the agent learned the whole reward landscape, not just a good top pick: correlation of Q̂ with the true value, top-10 recall, and optimism of the pick (`EVALUATION.md` §2).

**Key takeaways**

- `baseline-v1` stays the headline. Everything FA and constant-α is labelled post hoc.
- The α values were chosen on separate tuning seeds (master seed 20261103), never the reported ones (`EXPERIMENTS.md` §8.1).

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Data pipeline, simulator, benchmarks, grid, reward | `src/data/`, `src/environment.py`, `src/benchmarks/`, `src/action_space/`, `src/rewards.py` | `DATA_CLEANING.md`, `ENVIRONMENT.md` |
| Agent, runner, evaluation, batch runner, reports | `src/agent.py`, `src/train.py`, `src/evaluate.py`, `src/run_experiments.py`, `src/reporting/` | `EVALUATION.md`, `EXPERIMENTS.md`, `reports/README.md` |
| Baseline and post-hoc runs, including `grid10-robustness` | `results/`, `reports/` | `REVIEW_HISTORY.md` §8, `EXPERIMENTS.md` §7.7 |
| FA agent, features, FA training path | `src/fa_agent.py`, `src/features.py`, `src/train.py` | `FA_INTEGRATION_NOTES.md` |
| Your fix for FA final-policy ties and missing settlement | `train.py`, `evaluate.py`, `src/reporting/` | Commit `1cbab3a` |
| Q-value fit metrics; `b63` and `eps-decay` re-evaluated | `src/evaluate.py` | `EVALUATION.md` §2 |
| Step-size tuning, and the three runs registered | `src/tune_fa.py`, `results/tuning/`, `src/config.py` | `EXPERIMENTS.md` §8 |

405 tests pass.

## Your next steps

1. **Review the FA PR** (`feature/function-approximation` → `dev`). Merged as PR #23 on 2026-11-05. `FA_INTEGRATION_NOTES.md` §1 lists every edit to your files:
   - `train.py`, `run_experiments.py` and `evaluate.py`;
   - `config.py`: the new run fields and the three registrations;
   - the reporting code: agent fields in the settings table, and α formatting.

   This blocks the runs.
2. **Agree to the expected directions** in `EXPERIMENTS.md` §8.3 to §8.5 (one line each in `config.py`). Change them now if you disagree; they can't change once a run starts. This blocks the runs.
3. **Re-evaluate `baseline-v1`, `n26m`, `all-three` and `grid10-robustness`** with the new metrics, from your run logs (`.npz`, not on my machine):
   ```
   python -m src.run_experiments --results-dir results
   python -m src.run_experiments --run n26m --results-dir results/n26m      # likewise all-three, grid10-robustness
   ```
   This skips training for finished folders and re-evaluates them with the committed settings. Every earlier number stays identical; only the new fields are added (checked on `b63` and `eps-decay`). The FA vs tabular comparison needs these.
4. **`RUN_ISSUES` §4 steps 1 to 6, still open.** The total-gap column and the post-hoc "settled" label are still missing from `per_seed.csv` and `INDEX.md`. I'll review the PR.
5. **Windows print crash, still open.** `run_experiments` stops on "γ" in a status print unless `$env:PYTHONIOENCODING = "utf-8"` is set. Fix: `sys.stdout.reconfigure(encoding="utf-8")` at the start of `main()`.
6. **The Q-value fit metrics are not in the HTML reports.** `src/reporting` reads none of them, so two of the new runs' expectations can only be checked in `evaluation/summary.json`. Adding them to `extract.py` and the template would put them in the comparison. Not blocking.
7. **A false "calibration changed" flag on Windows**, from before this work. Every report built on William's machine flags the calibration as changed against `baseline-v1`. The independent review found the processed CSVs hash to `baseline-v1`'s values once CRLF line endings become LF, so the hash should be taken over normalized line endings (or the files pinned to LF in `.gitattributes`).
8. **Your report sections** (prospectus §4): Data/Calibration, and convergence analysis and visualizations.

**`tabular-const` report notes (to do, William).** The run finished on 2026-11-06 (30 seeds, trained at tag `tabular-const`, commit `bc311a3`). It is registered in `reports/registry.json`, extracted and built (`reports/tabular-const/`), but its `notes` are empty. To do:
- Read the report and check it against the expectations in `EXPERIMENTS.md` §8.3: a larger mean learning gap than `baseline-v1` at every γ, and fewer seeds meeting the primary rule at every γ (baseline-v1: 9, 8 and 9).
- Fill in `tldr`, `takeaways` and `caveats`, then run `python -m src.reporting.build tabular-const`.
- Record the outcome under §8.3, as was done for §7.7.
- The Q-fit numbers are in each `evaluation/summary.json` under `q_fit`, not in the report.
- Put this in `caveats`: the report flags the calibration and `grid.py`, `environment.py` and `rewards.py` as changed against `baseline-v1`. That is false: the content is byte-identical once Windows line endings are normalized. `agent.py` and `train.py` really changed, but not in tabular behaviour (an `epsilon` property; the FA path and logging).

## To decide together

**Transaction costs.** First settle the two open prospectus decisions: a fixed cost or a sweep, and whether holdings are the exact prior weights or bucketed.

**Which post-hoc runs go in the report.** My suggestion:
- `grid10-robustness` as the action-space check, and γ = 3 and 8 as the economic check;
- `b63` and `n26m` for training budgets and episode lengths;
- `eps-decay` short, as mostly a negative result;
- `all-three` as the combined check the 10% grid is compared against;
- the three new runs in the FA extension.
