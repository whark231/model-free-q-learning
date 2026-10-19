# Next Steps

**From:** William
**Date:** 2026-10-19
**Status:** everything that was blocking a real experiment is resolved. The next step is running one.

---

## TL;DR / In plain terms

`agent.py`, `train.py`, `evaluate.py`, `config.py`, and `run_experiments.py` are all built and tested, 244 tests passing. γ, the episode length, and the convergence rule are decided, and the γ justification is written down (`DATA_CLEANING.md` §8.4). The results-sharing question is settled (your proposal, adopted). A Windows-only bug that would have crashed the real run (`run_experiments.py` writing `γ` into `INDEX.md` with no encoding) is fixed.

Nothing is blocking the full-scale run anymore.

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
| Tabular Q-learning agent | `src/agent.py` | 22 tests |
| Training runner | `src/train.py` | 30 tests |
| Evaluation against the benchmarks | `src/evaluate.py` | 19 tests, `EVALUATION.md` |
| Experiment settings and batch runner | `src/config.py`, `src/run_experiments.py` | 19 tests, `EXPERIMENTS.md` |

244 tests pass (`python -m pytest src/tests -q`). All bugs reported across both rounds of `REVIEW_NOTES_FOR_WILLIAM.md`, plus the Windows encoding bug found 2026-10-19, are closed with regression tests.

---

## Decisions that used to be open, now settled

Previously tracked here as "two decisions still open, not parked." Both were decided 2026-10-16 and the full record lives in `REVIEW_NOTES_FOR_WILLIAM.md` §6:

1. **γ** = 5 as the headline value, γ ∈ {3, 8} as the Part VI risk-aversion check. Justified on outside grounds in `DATA_CLEANING.md` §8.4 (finance-literature meta-analysis, household-survey median, the Mehra-Prescott convention), written 2026-10-19.
2. **Convergence rule**: greedy action unchanged for K = 10% of training episodes, read as "held through the end of training" primarily. In `src/config.py`.

Everything that was parked (episode length, episode count, seed count, β, learning-rate magnitude, the numeric K, the second robustness check) is now set as an adopted default, also in `src/config.py`, also recorded in `REVIEW_NOTES_FOR_WILLIAM.md` §6.1. None of it needs revisiting unless there's a specific, recorded reason.

One thing confirmed closed rather than fixed: the `update()` performance question (§5.3 in your notes) is deliberately deferred, not resolved, since no planned experiment uses the 10% grid where it would matter.

## What happens next

Nothing is blocked on either of us. The code, settings, and batch runner are ready for the real run (`python -m src.run_experiments`, about 3.5 hours). Once it's done, `results/INDEX.md` and each experiment's `evaluation/` folder are what we both look at next.
