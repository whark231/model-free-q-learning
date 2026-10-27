# Next Steps

**From:** William
**Date:** 2026-10-26
**Status:** the full-scale run is done and committed. Your fix plan in `RUN_ISSUES_2026-10-22.md` §4 is the next piece of work, and I agree with it.

---

## TL;DR / In plain terms

Thanks for running it. All four experiments trained 10 seeds each with no crash, and your reviewer re-derived every number. The results are usable, but three tables need fixing before anyone writes Parts IV to VI from them, and your plan already covers that. It only touches your files and needs no retraining.

Scope is now set. The robustness checks are the 10% grid (action space) and γ = 3 and 8 (already run). The extensions are transaction costs and function approximation. Two things to decide together: a higher-B episode-length check, and how the report gets rebuilt if we run again.

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
3. **Review one change I made in your files.** `src/train.py` now logs `agent.epsilon` (a new property on my agent) instead of recomputing ε0·exp(−βt) itself, so the log always matches what the agent did. The stand-in agents in `test_train.py`, `test_evaluate.py` and `test_config.py` each gained an `epsilon`. Logs are bit-identical to before (checked on two seeds × 30,000 episodes), but the full run's manifests now record older `train.py` / `agent.py` hashes than the current code. Details in `REVIEW_HISTORY.md` §5.2.
4. **`EXPERIMENTS.md` line 3** still says "no full-scale run yet" (part of your step 3).
5. **10% grid, the Part VI action-space check** (decided 2026-10-26). The assignment's action-space table lists 10% as the "finer robustness check if computationally feasible". The 25% grid stays as the Part V second resolution. Needs from you: the experiment in `src/config.py` (its purpose text currently calls the 25% grid the action-space robustness check), and your half of the Q-table-copy fix in `train.py` (`REVIEW_HISTORY.md` §5.3). I'll do the `update()` half. Write "learning gap grows" into `EXPERIMENTS.md` as the expected direction before it runs.
6. **Report sections that are yours** (prospectus §4): Data/Calibration, and convergence analysis and visualizations. Findings 4, 5, 8 and 9 become sentences in Parts IV to VI.

## To decide together

**Increase B as an extra episode-length check.** I'd like to add B = 63 and B = 252 runs (γ = 5, 20% grid, same seeds, 13M episodes), with B = 20 kept as the headline. Longer episodes make every reward less noisy and cost little: B = 63 measured +15% per episode for 3.15 times the simulated days. Simulator estimates over 100 seeds: B = 20 finds the exact best in 44% of seeds with a mean learning gap of 9.3 bps; B = 63 gives 44% and 7.6; B = 252 gives 68% and 3.7. The numbers and method are in `REVIEW_HISTORY.md` §7.2. That section also records why slower ε decay and a minimum-tries floor were tested and dropped. The assignment lists episode lengths as a robustness check, but we chose this after seeing results, so it goes into `src/config.py` and `EXPERIMENTS.md` with its expected direction first and is labelled post-hoc. Needs code in your files: B is global in `src/config.py` and locked in `batch_settings`, so each experiment needs its own B and `--results-dir`. The 10% grid would stay at B = 20 so it changes one thing from the baseline.

**Extension timing.** Transaction costs (paired) start Week 7 on Oct 30, and function approximation (mine) starts Week 8 on Nov 6.

**Rebuilding the report.** `run_experiments.py` regenerates `results/INDEX.md` and every `evaluation/` folder on each run, but nothing regenerates `reports/full_run_2026-10-22/report.html`. That is a one-off: `build_data.py` hardcodes `results/` and the four experiment names, and it needs the `.npz` logs, which only exist on your machine. No run is scheduled either. If we do follow-up runs, `build_data.py` needs a results-folder and output-folder argument so each run gets its own dated report.
