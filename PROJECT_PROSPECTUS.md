# Project Prospectus: Model-Free Q-Learning and Optimal Portfolio Choice (Nine Assets)

**Due:** November 30, 2026
**Team:** William & Albert
**Status:** Draft — this is a rough plan and will be revised as we learn more. Treat every date here as a working target, not a hard commitment.

---

## 1. Research Question

Can a model-free reinforcement learning agent recover a near-optimal portfolio allocation when it observes only realized rewards — never the true expected returns or covariances used to generate them?

We test this in a controlled, synthetic nine-ETF environment: calibrate a mean-variance "ground truth" from real historical data, then train a Q-learning agent purely on simulated returns from that calibrated model, and measure how close its learned policy gets to the true optimum.

## 2. Assumptions About Team Skills (please correct if wrong)

This plan assumes:
- **William**: more comfortable with coding/data science, will lead on the RL implementation and technical math (optimization, Q-learning algorithm).
- **Albert**: newer to coding and quant finance, will lead on data construction, documentation, and results visualization early on — areas with a gentler learning curve — while pairing on the Q-learning core so both of us understand it deeply, not just one of us.

As skills develop, the split should shift — by the extension phase (weeks 7–8 or earlier), the goal is for both of us to be comfortable modifying the core algorithm.

## 3. What Needs to Be Built

Mapped to the assignment's required parts (Parts I–VI):

### Core (required for base grade)
| # | Component | Assignment Part | Status (2026-09-29) |
|---|---|---|---|
| 1 | ~~Nine-ETF data pipeline: download, align, clean, adjust for distributions~~ | Part I | ✅ Done — `src/data/`, `DATA_CLEANING.md`, 5,152-day sample |
| 2 | Ground-truth environment: estimate μ, Σ; build the synthetic Gaussian return generator | Part I | 🔶 Half done — μ, Σ estimated (`data/processed/`); the return generator itself (`src/environment.py`) is still empty |
| 3 | ~~Theoretical mean-variance benchmark: closed-form continuous optimum + best discrete-grid portfolio~~ | Part II | ✅ Done — `src/benchmarks/mean_variance.py`, 27 tests |
| 4 | ~~Feasible action space: portfolio-weight grid (20% increments, 1,287 actions baseline)~~ | Part III | ✅ Done — `src/action_space/grid.py`, 12 tests |
| 5 | ~~Reward function: episodic realized mean-variance reward~~ | Part III | ✅ Done — `src/rewards.py`, 5 tests |
| 6 | Stateless tabular Q-learning agent with epsilon-greedy exploration (extension will likely use functional approximation) | Part III | ⬜ Not started — `src/agent.py` is a placeholder |
| 7 | Training loop across multiple random seeds + convergence criterion | Part IV | ⬜ Not started — `src/train.py` is empty |
| 8 | Benchmark comparison: weight bar charts, L1/Euclidean distance, economic performance metrics | Part V | ⬜ Not started — `src/evaluate.py` is empty |
| 9 | Convergence analysis: learning curves, regret, comparison across ≥2 grid resolutions | Part V | ⬜ Not started |
| 10 | Robustness checks (at least 2: one action-space change, one economic/statistical assumption change) | Part VI | ⬜ Not started |

### Extension (our chosen bonus angle)
| # | Component | Notes | Status (2026-09-29) |
|---|---|---|---|
| 11 | Transaction-cost-augmented environment: introduces a state (current holdings), penalizes turnover | Turns the bandit problem into a true sequential MDP | ⬜ Not started |
| 12 | Function approximation Q-learning: replaces the table once state × action space is too large | Compare directly against the tabular baseline at the same grid resolution | ⬜ Not started |

### Deliverables
| # | Component | Notes | Status (2026-09-29) |
|---|---|---|---|
| 13 | Research report (~10–15 pages), following the assignment's required structure | Split into sections below | ⬜ Not started |
| 14 | Clean, reproducible Python code + README (run instructions, seeds, requirements) | | 🔶 Code and tests exist; `README.md` is still a one-line stub |
| 15 | Class presentation | | ⬜ Not started |

## 4. Task Ownership (draft — adjust as we go)

| Component | Primary | Support |
|---|---|---|
| Data pipeline & ground-truth calibration | Albert | William (review) |
| Theoretical mean-variance benchmark | William | Albert (learns derivation, writes it up) |
| Action space construction | William | Albert |
| Reward function | William | Albert (reviews math) |
| Tabular Q-learning core | William + Albert (paired) | — |
| Convergence analysis & visualizations | Albert | William |
| Robustness checks | Split: one each | Cross-review |
| Transaction-cost extension | William + Albert (paired) | — |
| Function approximation extension | William | Albert (learns concept, assists) |
| Report — Data/Calibration section | Albert | William |
| Report — Theory/Benchmark section | William (derivation) + Albert (writing) | — |
| Report — Q-learning/Extension sections | William | Albert reviews |
| Report — Intro/Conclusion/Robustness | Both | — |
| README + code cleanup | Albert | William |
| Presentation | Both | — |

## 5. Timeline

**Note:** Thanksgiving week (Nov 20–26) falls near the end of our runway — the schedule below front-loads the report draft to finish *before* that week, leaving it as a buffer/polish week rather than a working week.

| Week | Dates | Focus | Target Output |
|---|---|---|---|
| 1 | Sep 18–24 | Lock design decisions (grid resolution, reward formula, extension scope, transaction-cost approach); set up repo | Design doc committed to repo |
| 2 | Sep 25–Oct 1 | Data pipeline + ground-truth environment | Cleaned 9-ETF dataset, μ/Σ estimates, summary stats & correlation matrix |
| 3 | Oct 2–8 | Theoretical benchmark | Closed-form continuous optimum + best discrete-grid portfolio, verified |
| 4 | Oct 9–15 | Action space + reward implementation | Working 20% grid (1,287 actions), reward function coded & unit-tested |
| 5 | Oct 16–22 | Tabular Q-learning baseline | End-to-end training loop running on at least one seed |
| 6 | Oct 23–29 | Full baseline experiments | Multi-seed runs, convergence criterion applied, required comparisons (weights, distances, economic performance) |
| 7 | Oct 30–Nov 5 | Extension part 1: transaction costs | State-augmented environment running, compared against no-cost baseline |
| 8 | Nov 6–12 | Extension part 2: function approximation | FA agent trained, compared against tabular baseline at same settings |
| 9 | Nov 13–19 | Robustness checks + finalize results; start report draft | All figures/tables finalized; first full report draft |
| 10 | Nov 20–26 | **Thanksgiving buffer week** — light schedule | Report polish, code cleanup, README |
| Final | Nov 27–30 | Final review & submission | Submitted report + code + README |

## 6. Open Decisions Still to Finalize

- [ ] Confirm action library approach: fixed Cartesian grid (20%) vs. randomly sampled feasible-portfolio library
- [ ] Fix the transaction-cost penalty: single fixed parameter, or swept across values as one of our two required robustness checks
- [ ] Decide which second robustness check to run alongside the action-space one (risk-aversion sweep vs. covariance estimator choice vs. calibration window)
- [ ] Confirm random seed count for "multiple seeds" (suggest 5–10, balancing statistical confidence against compute time)
- [ ] Confirm state representation granularity for the transaction-cost extension (exact prior weights vs. coarsely bucketed)

## 7. Known Risks

- **Weeks 6–8 are the highest-risk stretch** — this is where the extension work (state-dependent environment + function approximation) lives, and it's new territory for both of us. If it slips, the fallback is to submit a strong required baseline (Parts I–VI) and a partially-complete extension, clearly labeled as such — a correctly scoped smaller extension beats a broken ambitious one per the rubric.
- **Thanksgiving week reduces available time** right before the deadline — hence the front-loaded schedule above.
- **Learning curve on RL concepts** — since this is new to both of us in places, pad weeks 4–5 mentally; if the core Q-learning loop takes longer than planned, borrow time from week 6 (multi-seed runs can be trimmed) rather than from the extension.

---

*This is a living document — update it as decisions get made and the timeline shifts.*
