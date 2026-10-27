# Next Steps

**From:** William
**Date:** 2026-10-26
**Status:** the full-scale run is done and committed. Your fix plan in `RUN_ISSUES_2026-10-22.md` §4 is the next piece of work, and I agree with it.

---

## TL;DR / In plain terms

Thanks for running it. All four experiments trained 10 seeds each with no crash, and your reviewer re-derived every number. The results are usable, but three tables need fixing before anyone writes Parts IV to VI from them, and your plan already covers that. It only touches your files and needs no retraining.

Beyond the fixes, we have three things to decide together: whether to add follow-up experiments, the extension scope before Week 7 starts on Oct 30, and how the report gets rebuilt if we run again.

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Data pipeline, μ and Σ | `src/data/`, `data/processed/` | `DATA_CLEANING.md`, 20 tests |
| Simulator | `src/environment.py` | `ENVIRONMENT.md`, 69 tests |
| Benchmarks, grid, reward | `src/benchmarks/`, `src/action_space/`, `src/rewards.py` | 60 tests |
| Agent, runner, evaluation, batch runner | `src/agent.py`, `src/train.py`, `src/evaluate.py`, `src/config.py`, `src/run_experiments.py` | 98 tests, `EVALUATION.md`, `EXPERIMENTS.md` |
| **Full-scale run, 2026-10-22** | `results/`, `reports/full_run_2026-10-22/` | `RUN_ISSUES_2026-10-22.md`; every number re-derived independently |

247 tests pass.

---

## Your next steps

1. **`RUN_ISSUES` §4 steps 1 to 6.** Total-gap column, the post-hoc "settled from checkpoint" measure, checkpoint-resolution labels, provenance in `summary.json`, then the evaluation-only re-run and re-verification. Go ahead from my side. I'll review the PR (review swap).
2. **Step 7, the run log.** Keep it. I've added `!results/full_run_*.log` to `.gitignore` on `feature/post-run-fixes`; you add the log from your machine, since it only exists there.
3. **Review one change I made in your files.** `src/train.py` now logs `agent.epsilon` (a new property on my agent) instead of recomputing ε0·exp(−βt) itself, so the log always matches what the agent did. The stand-in agents in `test_train.py`, `test_evaluate.py` and `test_config.py` each gained an `epsilon`. Logs are bit-identical to before (checked on two seeds × 30,000 episodes), but the full run's manifests now record older `train.py` / `agent.py` hashes than the current code. Details in `REVIEW_NOTES_FOR_WILLIAM.md` §5.2.
4. **`EXPERIMENTS.md` line 3** still says "no full-scale run yet" (part of your step 3).
5. **Report sections that are yours** (prospectus §4): Data/Calibration, and convergence analysis and visualizations. Findings 4, 5, 8 and 9 become sentences in Parts IV to VI.

## To decide together

**Follow-up experiments.** `REVIEW_NOTES_FOR_WILLIAM.md` §7 has the numbers. Short version: the agent gets about 1,200 tries at each non-winning portfolio, and telling the top few apart takes about 600,000, so minimum-try floors, slower ε decay and simply longer budgets are all far too small to fix it. My proposal is two extra Part VI checks, written into `src/config.py` with their expected direction before running, with the existing run kept as the headline:

- **B = 63**, same 13M episodes: 3.15 times the simulated days for about 15% more compute. Expect a smaller learning gap, not reliable exact recovery.
- **10% grid**, later: the curse-of-dimensionality check. Expect the learning gap to grow.

Both need code in your files first: `EPISODE_LENGTH` and the grid list are global in `src/config.py` and locked in the results folder's `batch_settings`, so each needs a per-experiment setting and its own `--results-dir`. The 10% grid also needs the Q-table-copy fix (`REVIEW_NOTES_FOR_WILLIAM.md` §5.3), which is a change on both our sides.

**Extension scope.** Week 7 (transaction costs, paired) starts Oct 30 and Week 8 (function approximation, mine) Nov 6. The results argue for function approximation first, since sharing information between neighbouring portfolios is what the tabular agent lacks. Worth deciding whether both extensions still fit before the report draft in Week 9.

**Rebuilding the report.** `run_experiments.py` regenerates `results/INDEX.md` and every `evaluation/` folder on each run, but nothing regenerates `reports/full_run_2026-10-22/report.html`. That is a one-off: `build_data.py` hardcodes `results/` and the four experiment names, and it needs the `.npz` logs, which only exist on your machine. No run is scheduled either. If we do follow-up runs, `build_data.py` needs a results-folder and output-folder argument so each run gets its own dated report.
