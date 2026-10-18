# Next Steps

**From:** William
**Date:** 2026-10-15
**Status:** the agent/train split worked. Both `agent.py` and `train.py` are done and tested. Two decisions are still mine (or ours) to make before a real experiment runs; everything else is deliberately parked.

---

## TL;DR / In plain terms

`src/agent.py` (mine) and `src/train.py` (yours) are both built, tested, and merged, against the interface we agreed on first. 201 tests pass across the project. The code can run a real experiment today (the CLI in `train.py`'s docstring works), except that a few arguments (γ, episode length, episode count, seed count, β) are required with no defaults, on purpose, because none of them are decided yet.

Two of those can't wait: γ, and the exact definition of convergence. The rest are implementation details with a real tuning tradeoff, and I'm deliberately not deciding them now. They get explored properly once there's a notebook to run that exploration in, not guessed at while we're still writing plumbing.

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Nine-ETF data pipeline | `src/data/` | `DATA_CLEANING.md`, 5,152-day sample, 20 tests |
| μ and Σ estimation | `data/processed/` | Part of the same pipeline |
| Ground-truth return simulator | `src/environment.py` | `ENVIRONMENT.md`, 69 tests |
| Theoretical mean-variance benchmark | `src/benchmarks/mean_variance.py` | 29 tests |
| Feasible action space (weight grid) | `src/action_space/grid.py` | 22 tests |
| Reward function | `src/rewards.py` | 9 tests |
| Tabular Q-learning agent | `src/agent.py` | 22 tests in `test_agent.py` |
| Training runner | `src/train.py` | 30 tests in `test_train.py` |

All bugs reported across both rounds of `REVIEW_NOTES_FOR_WILLIAM.md` are closed, each with a regression test. 201 tests pass (`python -m pytest src/tests -q`).

---

## Two decisions still open, not parked

These aren't hyperparameters, they're definitions, and the assignment is
explicit that both need settling before anyone looks at a result:

1. **γ**, mine to choose, on outside grounds (matching a known value, or whatever the three-asset version of this problem used), written down in `DATA_CLEANING.md` §8.4 before any results exist. Picking it by seeing which value makes a run look good is the researcher-degrees-of-freedom problem the assignment calls out by name.
2. **The convergence rule's definition**, ours jointly. The shape is basically agreed: agent-observable quantities only (greedy action unchanged for K episodes, or largest Q-change below a tolerance), never the gap to the benchmark, since that needs μ and Σ. Needs writing down as final.

## Everything else is intentionally parked

Episode length, episode count, seed count, β, learning-rate magnitude, and
the numeric convergence threshold K all have real defaults built into
`train.py`'s config already, as parameters, not hardcoded. They're parked
until the tuning/research phase, where they belong: that's when we can
actually see the tradeoff (compute time vs. how fast/reliably Q-values
settle) instead of guessing. Full list and reasoning is in
`REVIEW_NOTES_FOR_WILLIAM.md` §3.2. This isn't something you need to track;
it stops being an open item in these notes until we're actually at that
phase.

## What happens next

Nothing is blocked on you right now for the agent/train side. Once γ and the
convergence definition are settled, we can run a real experiment and move to
evaluation and convergence analysis (`src/evaluate.py`, still unstarted).
I'll let you know when that's ready.
