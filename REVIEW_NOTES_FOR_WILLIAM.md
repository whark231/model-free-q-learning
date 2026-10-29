# Notes for William

**From:** Albert, kept up to date by William
**Date:** 2026-10-29
**Status:** the baseline and four post-hoc runs are done, reported and merged into `dev`. Left on your side: one small `agent.py` change, the two extensions, and your report sections. Everything closed is in `REVIEW_HISTORY.md` (post-hoc runs in §8).

---

## TL;DR / In plain terms

The tabular agent is finished and tested, and the runs answer the main question: it reliably reaches the right neighbourhood of portfolios, and more data per decision (longer episodes) is what helps it pick the exact best. What's left for you is building the two extensions, starting with transaction costs this week, and writing your report sections.

**Key concepts to understand**

- **Mean gap vs. exact recovery.** Every post-hoc run lowered the mean learning gap and the worst seed at every γ, but exact recovery of the best portfolio still depends on luck among the top few portfolios, because their rewards are too noisy to tell apart at these budgets.
- **Slower in days, not always in episodes.** Every variant takes more simulated days to converge because each episode or run holds more data. Counted in episodes, `b63` converges faster than the baseline.

**Key takeaways**

- `baseline-v1` stays the headline for Parts IV and V. The post-hoc runs are Part VI sensitivity checks, labelled post hoc.
- B = 63 was the cheapest improvement: γ = 5 mean gap 17.6 → 5.1 bps/yr with lower regret. That is a concrete point for the Q-learning section.

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Benchmarks, grid, reward | `src/benchmarks/`, `src/action_space/`, `src/rewards.py` | 60 tests; review issues closed (`REVIEW_HISTORY.md` §1) |
| Agent, with `epsilon` property | `src/agent.py` | 24 tests |
| γ justification | `DATA_CLEANING.md` §8.4 | Sources listed there |
| Baseline run, 2026-10-22 | `results/`, `reports/baseline-v1/` | `RUN_ISSUES_2026-10-22.md`; every number re-derived |
| Post-hoc runs `b63`, `n26m`, `eps-decay`, `all-three` | `results/<run_id>/`, `reports/<run_id>/` | `REVIEW_HISTORY.md` §8; each expected direction registered before running |
| Higher-B proposal | run `b63` | Closed; `REVIEW_HISTORY.md` §8 |

283 tests pass.

## Your next steps

1. **Transaction-cost extension, with Albert, Week 7 (Oct 30 to Nov 5).** It adds a state (current holdings) and a turnover penalty, which turns the bandit into a sequential problem. Settle the two open prospectus decisions first: a fixed cost or a sweep, and how finely to represent holdings.
2. **Function-approximation extension, yours, Week 8 (from Nov 6).** For the write-up: a Q-function quadratic in the weights can represent the mean-variance objective exactly, so say plainly that it learns from rewards only and never sees μ or Σ. Compare it with the tabular agent at the same grid and budget, as the prospectus says.
3. **`update()` return value** (`REVIEW_HISTORY.md` §5.3). Albert's half is in `train.py`. Yours: end `update()` with `return float(self._q[action])`, change the return annotation to `float`, and add a test that it equals `q_values[action]`. It saves about 6 µs of roughly 96 µs per episode on the 10% grid, so it's worth doing before `grid10-robustness` runs (about 21 hours).
4. **Review Albert's evaluation PR** for `RUN_ISSUES` Findings 1 to 3 when he opens it (total-gap column, post-hoc settled measure, checkpoint labels). Not opened yet.
5. **Your report sections** (prospectus §4): Theory/Benchmark and Q-learning. `REVIEW_HISTORY.md` §7.1 (why exact recovery fails) and §8 (what the post-hoc runs changed) are the core of the Q-learning results.

## To decide with Albert

**Which post-hoc runs go in the report.** The assignment needs one action-space check (the 10% grid, still to run) and one economic or statistical check (γ = 3 and 8, done). `b63` and `n26m` cover its listed "different training budgets, episode lengths". I'd keep `eps-decay` and `all-three` short: the first as mostly a negative result (no gain at γ = 5 and 8, higher regret), the second as a combined check, so the report doesn't read as a search over settings.
