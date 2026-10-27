# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-10-26
**Status:** the Q-learning core is done and the full-scale run is committed. What's left on your side is the list below. Everything closed so far (review issues, decisions, rejected ideas) is in `REVIEW_HISTORY.md`.

---

## TL;DR / In plain terms

Your files are done and every review issue in them is closed. Next come four things: finish the review swap, make one speed fix the 10% grid needs, build the two extensions, and write your report sections. One proposal needs Albert's agreement before anything runs: longer episodes (a higher B).

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Benchmarks, grid, reward | `src/benchmarks/`, `src/action_space/`, `src/rewards.py` | 60 tests; review issues 1 to 5 and A to C closed (`REVIEW_HISTORY.md` §1) |
| Agent, including the `epsilon` property and `greedy_action` note | `src/agent.py` | 24 tests; `train.py` now logs `agent.epsilon`, bit-identical logs checked on 2 seeds × 30,000 episodes |
| γ justification | `DATA_CLEANING.md` §8.4 | Sources listed there |
| `.gitignore` for results sharing, keeping the run log | `.gitignore` | Checked with `git check-ignore` |
| Full-scale run, 2026-10-22 | `results/` | `RUN_ISSUES_2026-10-22.md`; every number re-derived |

247 tests pass.

## Your next steps

1. **Review swap.** Review `train.py`, still open from the split. Then review Albert's evaluation PR for `RUN_ISSUES` Findings 1 to 3 when he opens it. He reviews the `train.py` change from the `epsilon` property.
2. **Q-table copy fix, now needed** (`REVIEW_HISTORY.md` §5.3). The 10% grid is the Part VI action-space check, and copying a 43,758-entry Q-table every episode is its main cost (measured earlier: 91 µs per episode vs. 51.6 µs for the bare loop, so 3.3 hours for 10 seeds). Your half: `update()` returns the new Q(action). Albert's half: `train.py` keeps its own copy and updates one entry.
   *Update 2026-10-27 (Albert):* Albert's half is on branch `run/grid10-robustness`. It uses the returned value when there is one, falls back to the full copy when `update()` returns `None` (today's behaviour), and checks its copy against `agent.q_values` at every checkpoint. A test shows bit-identical logs from `TabularQAgent` and from a subclass whose `update()` returns the value. Your half, proposed and not applied (`src/agent.py` is yours): end `update()` with `return float(self._q[action])`, change its return annotation to `float`, and add a test that the returned value equals `q_values[action]`. Without it, post-hoc run `grid10-robustness` takes about 21 to 23 hours instead of 12 to 15 (`EXPERIMENTS.md` §7.3 on that branch).
3. **Extensions: transaction costs and function approximation.** Transaction costs are paired with Albert, Week 7 (starts Oct 30). Function approximation is yours, Week 8 (starts Nov 6). For the write-up: a Q-function quadratic in the weights can represent the mean-variance objective exactly, so say plainly that it learns from rewards only and never sees μ or Σ.
4. **Your report sections** (prospectus §4): Theory/Benchmark and Q-learning. `REVIEW_HISTORY.md` §7.1 explains why exact recovery fails, which is the core of the Q-learning results.

## To decide with Albert

**Increase B as an extra episode-length check.** The assignment lists "different training budgets, episode lengths, and random seeds" as a Part VI robustness check. Longer episodes make each reward less noisy for every portfolio at once, and they cost little: B = 63 was 43.8 µs per episode vs. 38.2 µs at B = 20 (+15%) for 3.15 times the simulated days. Simulator estimates on the baseline, 100 seeds each (`REVIEW_HISTORY.md` §7.2):

| B | Exactly best | Mean learning gap (bps/yr) |
|---|---|---|
| 20 (current) | 44% | 9.3 |
| 63 (a quarter) | 44% | 7.6 |
| 252 (a year) | 68% | 3.7 |

Proposal: run B = 63 and B = 252 at γ = 5 on the 20% grid with the same seeds and 13M episodes, and keep B = 20 as the headline. Because this comes after seeing results, write each run and its expected direction (smaller learning gap) into `src/config.py` and `EXPERIMENTS.md` before running, and label it post-hoc in the report. Note in the report that a 252-day batch stretches the assignment's "short batch of returns". Run the 10% grid at B = 20 so it changes one thing from the baseline.

Both runs need Albert's code first. B is global in `src/config.py` and locked in the results folder's `batch_settings`, so each experiment needs its own B and its own `--results-dir`.
