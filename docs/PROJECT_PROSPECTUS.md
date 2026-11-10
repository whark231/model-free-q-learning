# Project Prospectus: Model-Free Q-Learning and Optimal Portfolio Choice (Nine Assets)

**Due:** November 30, 2026
**Team:** William & Albert
**Status:** living plan, updated 2026-11-09. The required work (Parts I to VI) is done. Both extensions are built; function approximation has results, transaction costs needs a redesign. Every non-TC code issue from the two audits is fixed (`AUDIT_2026-11-06.md` §2.1, §2.2, §8.3). Still open: H4, M12 and F-L10 (text for the final report), F-L9 (partly fixed), L7 (a decision on `all-three`'s comparison set) and F-L13 (a tag on Albert's Mac). `dev` was merged into `main` on 2026-11-09 (PR #28, F-M1). What remains is the TC decision, choosing the final runs, and the report.

---

## TL;DR / In plain terms

Everything the assignment requires has been built, run and independently checked. The function-approximation extension has results; the transaction-cost extension runs but its agent learns nothing yet, so it needs a design decision. The next three weeks are: decide the last runs, run them, write the 10 to 15 page report, and fix the small issues the audit found.

**Key concepts to understand**

- **Required work versus bonus.** Parts I to VI earn the base grade (five 20% buckets). Bonus credit needs an extension that is "correctly designed, benchmarked, and interpreted"; complexity alone earns nothing.
- **Post hoc.** Every run after `baseline-v1` was designed after its results were seen. Each was registered with an expected direction before it ran, and the report labels them post hoc.

**Key takeaways**

- `baseline-v1` is the headline result for Parts IV and V.
- The transaction-cost design decision (`AUDIT_2026-11-06.md` §6.2) blocks any further TC run and should be made first.

---

## 1. Research Question

Can a model-free reinforcement learning agent recover a near-optimal portfolio allocation when it observes only realized rewards, never the true expected returns or covariances used to generate them?

We test this in a controlled, synthetic nine-ETF environment: calibrate a mean-variance "ground truth" from real historical data, train a Q-learning agent purely on simulated returns from that calibrated model, and measure how close its learned policy gets to the true optimum.

## 2. Assumptions About Team Skills

- **William**: leads the RL implementation and the technical math (optimization, Q-learning, function approximation).
- **Albert**: led data construction, the simulator, evaluation, reporting and documentation, and built the transaction-cost extension.

Both of us now modify the core algorithm, as planned for the extension phase.

## 3. What Needs to Be Built

### Core (required for base grade)
| # | Component | Assignment Part | Status (2026-11-06) |
|---|---|---|---|
| 1 | Nine-ETF data pipeline: download, align, clean, adjust for distributions | Part I | ✅ Done. `src/data/`, `DATA_CLEANING.md`; 5,152 days, 2006-02-07 to 2026-07-31. Rebuilt independently in the audit |
| 2 | Ground-truth environment: μ, Σ and the synthetic Gaussian return generator | Part I | ✅ Done. `src/environment.py`, `ENVIRONMENT.md` |
| 3 | Theoretical benchmark: closed-form, long-only continuous optimum, best grid portfolio | Part II | ✅ Done. `src/benchmarks/mean_variance.py`; re-derived by exact enumeration in the audit |
| 4 | Feasible action space: 20% grid (1,287), plus 25% (495) and 10% (43,758) | Part III | ✅ Done. `src/action_space/grid.py` |
| 5 | Reward: episodic realized mean-variance reward | Part III | ✅ Done. `src/rewards.py`; E[reward] = U(w) checked by Monte Carlo |
| 6 | Stateless tabular Q-learning agent with ε-greedy exploration | Part III | ✅ Done. `src/agent.py` |
| 7 | Training across seeds and a convergence rule fixed in advance | Part IV | ✅ Done. `baseline-v1`, 2026-10-22: γ = 3, 5, 8 and the 25% grid, 10 seeds × 13M episodes (`results/INDEX.md`, `EXPERIMENTS.md`) |
| 8 | Benchmark comparison: weights, distances, economic performance | Part V | ✅ Done. Every experiment's `evaluation/` folder has the grouped weight bars, absolute differences, L1 and Euclidean distances, the performance table, both gaps and the per-seed distribution. The total gap against the continuous optimum and the post-hoc "settled" measure are in every summary, `per_seed.csv` and `INDEX.md` (`RUN_ISSUES` §4 steps 1 to 3, fixed 2026-11-09, `AUDIT_2026-11-06.md` M10) |
| 9 | Convergence analysis: learning curves, regret, two resolutions | Part V | ✅ Done. Learning curves, cumulative regret, observations to the 25 bps line; 20% vs 25% and 20% vs 10% grids |
| 10 | Robustness checks (one action-space, one economic or statistical) | Part VI | ✅ Done. Risk aversion γ = 3 and 8; the 10% grid (`grid10-robustness`, 2026-10-29); training budget, episode length and exploration (`b63`, `n26m`, `eps-decay`, `all-three`, 2026-10-28) |

Full suite: 442 tests passing on 2026-11-09 (`python -m pytest src/tests -q`).

### Extensions (bonus)
| # | Component | Notes | Status (2026-11-06) |
|---|---|---|---|
| 11 | Transaction costs: state = previous target, quadratic cost on the move | `TRANSACTION_COSTS.md`, `src/transaction_costs.py` | 🔶 Built; quadratic-cost sweep run (0, 1, 2, 5 bp; 10 seeds). The code is correct but the agent learns nothing: at zero cost it scores like a random portfolio, 468 bps/yr below the oracle. Discount and learning-rate comparisons registered, **not run**, and should not be run as designed. Needs a redesign decision (`AUDIT_2026-11-06.md` H1, H2, §6.2) |
| 12 | Function approximation: linear Q over quadratic or spline features | `FA_INTEGRATION_NOTES.md`, `EXPERIMENTS.md` §8 | ✅ Built, step size tuned on separate seeds, and run 2026-11-06: `tabular-const`, `fa-quadratic`, `fa-spline` (30 seeds each). Both FA models beat the constant-α tabular agent at γ = 5 and 3 and ranked the reward landscape better (Spearman); of the registered expectations, split into five checks, four failed in whole or part (`EXPERIMENTS.md` §8.7). Every run now carries Q-fit numbers and shows them in its report (2026-11-09) |

### Deliverables
| # | Component | Notes | Status (2026-11-06) |
|---|---|---|---|
| 13 | Research report (10 to 15 pages) in the assignment's structure | Sections split in §4 | ⬜ Not started. Inputs: `reports/<run_id>/report.html`, `results/<experiment>/evaluation/`, `EXPERIMENTS.md` §7 and §8.7, `REVIEW_HISTORY.md` §7 and §8 |
| 14 | Clean, reproducible code and README | | 🔶 Windows print crash and line-ending hashing fixed, and the README's reproduce command works in a fresh clone (2026-11-09; `AUDIT_2026-11-06.md` M1, M2, F-H1). Still open: the TC reproduction commands (M9) and the print fix in `tc_sweep_report.py`, both waiting on the TC redesign. `dev` merged into `main` 2026-11-09 (PR #28, F-M1) |
| 15 | Class presentation | | ⬜ Not started |

## 4. Task Ownership

| Component | Primary | Support |
|---|---|---|
| Data pipeline & ground-truth calibration | Albert | William (review) |
| Theoretical mean-variance benchmark | William | Albert (learns derivation, writes it up) |
| Action space construction | William | Albert |
| Reward function | William | Albert (reviews math) |
| Tabular Q-learning core | Agent (`agent.py`): William; training runner (`train.py`): Albert | Each reviews the other's PR |
| Convergence analysis & visualizations | Albert | William |
| Robustness checks | Split: one each | Cross-review |
| Transaction-cost extension | Albert built it; redesign paired (proposed 2026-11-06, to agree) | William |
| Function approximation extension | William | Albert (reviewed the edits to his files) |
| Report: Data/Calibration section | Albert | William |
| Report: Theory/Benchmark section | William (derivation) + Albert (writing) | |
| Report: Q-learning/Extension sections | William (FA); TC section proposed for Albert, to agree | Each reviews the other |
| Report: Intro/Conclusion/Robustness | Both | |
| README + code cleanup | Albert | William |
| Presentation | Both | |

## 5. Timeline (remaining)

Weeks 1 to 8 are done: data, environment, benchmark, grid and reward, tabular baseline, full baseline run and robustness runs (by 2026-10-29), FA and the TC sweep (2026-11-03 to 2026-11-06). The original order put TC in week 7 and FA in week 8; in practice both ran in week 8.

| Week | Dates | Focus | Target output |
|---|---|---|---|
| 8 (rest) | Nov 6 to 12 | Done by Nov 9: every run re-evaluated with Q fit, audit fixes M1 to M5, M10 and the flow-audit items. Still to do: decide the final runs and the TC redesign (`AUDIT_2026-11-06.md` §6, §7). `dev` merged into `main` Nov 9 (PR #28) | Final run list registered with expected directions |
| 9 | Nov 13 to 19 | Run the final runs; build their reports; first full report draft | All figures and tables final; draft complete |
| 10 | Nov 20 to 26 | Thanksgiving buffer: light schedule | Report polish, code cleanup, README |
| Final | Nov 27 to 30 | Final review and submission | Report, code and README submitted |

## 6. Open Decisions

- [x] Action library: fixed Cartesian grid (`src/action_space/grid.py`)
- [x] Robustness checks: the 10% grid (action space) and γ = 3 and 8 (risk aversion), decided 2026-10-26, both run
- [x] Extensions: both transaction costs and function approximation, confirmed 2026-10-26
- [x] Higher-B episode-length check: run as `b63` (2026-10-28), with `n26m`, `eps-decay` and `all-three`. B = 252 not run
- [x] FA learning rate: a constant α for FA and a separately tuned constant α for the tabular comparison, agreed with the professor 2026-11-05
- [x] TC cost penalty and state: swept at 0, 1, 2 and 5 bp per 20-point transfer, with the exact previous grid target as the state (`TRANSACTION_COSTS.md`)
- [ ] **TC redesign** before any further TC run: split learning and planning, add drift, a negative result, or a mix (`AUDIT_2026-11-06.md` §6.2)
- [ ] **Final runs** from the candidates in `AUDIT_2026-11-06.md` §6 (for example FA on the 10% grid, a γ = 8 FA check, three versus nine assets)
- [ ] Which post-hoc runs the report covers in full and which in a sentence (decide before writing Part VI)

The Q-learning core's decisions (γ, the convergence rule, episode length, episode count, seed count, β and the second Part VI robustness check) were settled on 2026-10-16 and live in `src/config.py`, with the reasoning in `EXPERIMENTS.md` and `REVIEW_HISTORY.md` §6. γ's written justification is in `DATA_CLEANING.md` §8.4 (2026-10-19).

## 7. Known Risks

- **Transaction costs may not reach a credible result in time.** The audit found the current design cannot learn. Fallback: report it as a carefully diagnosed negative result and put the bonus weight on function approximation. A correctly scoped smaller extension beats a broken ambitious one under the rubric.
- **Run time competes with writing.** A 30-seed run takes 2.5 to 7.5 hours per machine. Register and start the final runs early in week 9 so the draft is written while they train.
- **Thanksgiving week reduces available time** right before the deadline, so the draft should be finished by Nov 19.
- **Reproducibility.** The Windows print crash and the false "changed" flags are fixed (`AUDIT_2026-11-06.md` M1, M2). `dev` was merged into `main` on 2026-11-09 (PR #28, F-M1), so a plain clone gets the current code; merge `dev` into `main` again after each later change.

---

*This is a living document. Update it as decisions get made and the timeline shifts.*
