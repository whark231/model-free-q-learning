# Project Prospectus: Model-Free Q-Learning and Optimal Portfolio Choice (Nine Assets)

**Due:** November 30, 2026
**Team:** William & Albert
**Status:** living plan, updated 2026-11-20 (Albert; TC rows, timeline, decisions and risks). The required work (Parts I to VI) is done. Function approximation has results. The transaction-cost redesign (with drift, `TC_DRIFT_DESIGN.md`) is built and tuned: environment, oracle, agent, training, evaluation and the run driver (`src/tc_drift/run.py`, PRs #41 and #42), with the oracle run on every setting, all five oracle checks run (check 5's form signed off by William 2026-11-20, D-I), the hyperparameters fixed (William, 2026-11-20) and m = 4,000 (Albert, 2026-11-20). The 40 agent runs (4 cost levels × 10 seeds, projected at about 3 hours on Albert's Mac) are ready to launch. Every non-TC code issue from the two audits is fixed (`AUDIT_2026-11-06.md` §2.1, §2.2, §8.3). H4 closed 2026-11-10 (`AUDIT_2026-11-06.md` §2.3). Still open: M12 and F-L10 (text for the final report), F-L9 (partly fixed), L7 (a decision on `all-three`'s comparison set). F-L13 is closed: the `grid10-robustness` tag is on `origin` at `e13a9a6`. `dev` was merged into `main` on 2026-11-09 (PR #28, F-M1). What remains is the TC agent runs and their analysis, the decisions in §6, and the report.

---

## TL;DR / In plain terms

Everything the assignment requires has been built, run and independently checked. The function-approximation extension has results. The transaction-cost extension was redesigned (holdings now drift between quarterly decisions, against a numerical oracle) and its code is built, checked and tuned; its 40 agent runs are ready to start. The last ten days are: run the TC agent, write the 10 to 15 page report, and fix the small issues the audit found.

**Key concepts to understand**

- **Required work versus bonus.** Parts I to VI earn the base grade (five 20% buckets). Bonus credit needs an extension that is "correctly designed, benchmarked, and interpreted"; complexity alone earns nothing.
- **Post hoc.** Every run after `baseline-v1` was designed after its results were seen. Each was registered with an expected direction before it ran, and the report labels them post hoc.

**Key takeaways**

- `baseline-v1` is the headline result for Parts IV and V.
- The TC agent runs are the last computation. Their expected directions (E1 to E12, `TC_DRIFT_DESIGN.md` §7) were written down before they start; William kept them unchanged (2026-11-20). The oracle results can be reported regardless.
- The report has not been started and is now the main risk (§7).

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

Full suite: 757 tests passing on 2026-11-20 on Albert's Mac (`python -m pytest src/tests -q`).

### Extensions (bonus)
| # | Component | Notes | Status (2026-11-06; TC updated 2026-11-20) |
|---|---|---|---|
| 11 | Transaction costs: state = previous target, quadratic cost on the move | `TRANSACTION_COSTS.md`, `src/transaction_costs.py` | 🔶 Built; quadratic-cost sweep run (0, 1, 2, 5 bp; 10 seeds). The code is correct but the agent learns nothing: at zero cost it scores like a random portfolio, 468 bps/yr below the oracle. Discount and learning-rate comparisons registered, **not run**, and should not be run as designed. Kept as a labelled negative result. **Redesigned 2026-11-12 as TC with drift** (`TC_DRIFT_DESIGN.md`, `src/tc_drift/`): three ETFs, 63-day blocks, realistic costs at $100M, $1B and $10B. As of 2026-11-17: environment, oracle, agent, training and evaluation built and tested; the oracle is solved at all four sizes plus the κ-only and λ-only illustrations (William, PR #36); checks 1 to 4 pass (check 3 restated on the policy, post hoc), and check 5 is inside tolerance but resolved only at zero cost. The oracle beats always-rebalancing by 1.3 ± 0.7 bps/yr at $1B and 7.1 ± 1.3 at $10B. *2026-11-20:* tuned (William, PR #39: recursive least squares on the cost block, α_value = 3e-4, 4M blocks per seed, the policy scored on θ averaged over the last 10%; §6, §9). The run driver (`src/tc_drift/run.py`, Albert, PR #41) trains and scores the 40 runs; expected directions E1 to E12 registered before any run (§7). Runs not started |
| 12 | Function approximation: linear Q over quadratic or spline features | `FA_INTEGRATION_NOTES.md`, `EXPERIMENTS.md` §8 | ✅ Built, step size tuned on separate seeds, and run 2026-11-06: `tabular-const`, `fa-quadratic`, `fa-spline` (30 seeds each). Both FA models beat the constant-α tabular agent at γ = 5 and 3 and ranked the reward landscape better (Spearman); of the registered expectations, split into five checks, four failed in whole or part (`EXPERIMENTS.md` §8.7). Every run now carries Q-fit numbers and shows them in its report (2026-11-09) |

### Deliverables
| # | Component | Notes | Status (2026-11-20) |
|---|---|---|---|
| 13 | Research report (10 to 15 pages) in the assignment's structure | Sections split in §4 | ⬜ Not started (2026-11-20). Inputs: `reports/<run_id>/report.html`, `results/<experiment>/evaluation/`, `EXPERIMENTS.md` §7 and §8.7, `REVIEW_HISTORY.md` §7 and §8 |
| 14 | Clean, reproducible code and README | | 🔶 Windows print crash and line-ending hashing fixed, and the README's reproduce command works in a fresh clone (2026-11-09; `AUDIT_2026-11-06.md` M1, M2, F-H1). TC with drift has its README commands (2026-11-20, TC-10; C13 closed). Still open: the old TC design's single-variant configs (M9) and the print fix in `tc_sweep_report.py`, which depend on how the first design is reported (D-G). `dev` merged into `main` 2026-11-09 (PR #28, F-M1) |
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
| Transaction-cost extension | First design: Albert. Redesign with drift: William (oracle, agent, features, tuning design) and Albert (environment, training, evaluation, config, runs) | Each reviews the other |
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
| 8 (rest) | Nov 6 to 12 | Done: every run re-evaluated with Q fit, audit fixes M1 to M5, M10 and the flow-audit items; `dev` merged into `main` Nov 9 (PR #28); the TC redesign decided and its interface frozen (PR #33). Not done: the final run list | Final run list registered with expected directions |
| 9 | Nov 13 to 19 | Done: TC with drift built (PRs #34 to #37), the Nov 17 gate met, the oracle run on every setting, tuning decided and piloted (PR #39). Not done: the first report draft | All non-TC figures and tables final; draft complete |
| 10 | Nov 20 to 26 | TC agent runs (40 runs, projected at about 3 hours on Albert's Mac) and their analysis; TC go/no-go Nov 24 (William, D-H); the report draft. Proposed by Albert, to agree with William (D-H): draft sections by Nov 24, full draft by Nov 26 | All results final; full draft |
| Final | Nov 27 to 30 | Final review and submission | Report, code and README submitted |

## 6. Open Decisions

- [x] Action library: fixed Cartesian grid (`src/action_space/grid.py`)
- [x] Robustness checks: the 10% grid (action space) and γ = 3 and 8 (risk aversion), decided 2026-10-26, both run
- [x] Extensions: both transaction costs and function approximation, confirmed 2026-10-26
- [x] Higher-B episode-length check: run as `b63` (2026-10-28), with `n26m`, `eps-decay` and `all-three`. B = 252 not run
- [x] FA learning rate: a constant α for FA and a separately tuned constant α for the tabular comparison, agreed with the professor 2026-11-05
- [x] TC cost penalty and state: swept at 0, 1, 2 and 5 bp per 20-point transfer, with the exact previous grid target as the state (`TRANSACTION_COSTS.md`)
- [x] **TC redesign:** drift, a numerical oracle and a cost-learning agent, with the first design kept as a negative result (`TC_DRIFT_DESIGN.md`, decided 2026-11-12)
- [x] **TC agent tuning** (William, 2026-11-19 and 20): recursive least squares on the cost block after a constant-α burn-in, α_value = 3e-4, 4M blocks, the tail-averaged policy, D12 restated and frozen (`TC_DRIFT_DESIGN.md` §6, §9; post hoc)
- [x] **TC oracle check 3:** restated on the policy, M stays at 2,000 (William, 2026-11-17; post hoc)
- [x] **TC evaluation size** (Albert, 2026-11-20): m = 4,000 starts, fixed before any agent result (D-E)
- [x] **TC expected directions** (D-F): Albert's E1 to E12 registered 2026-11-20 (`TC_DRIFT_DESIGN.md` §7); William kept them unchanged the same day
- [ ] **How the first TC design is reported** (D-G, both): a paragraph, or with its tables (then M8 and M9 must be fixed first)
- [x] **Check 5's form** (D-I, William, 2026-11-20): yes to the 90-block model side and a two-SE bound at the cost levels, labelled post hoc (`TC_DRIFT_DESIGN.md` §5)
- [ ] **Report schedule** (D-H, both): the TC go/no-go is Nov 24 (William); sections and dates still to agree
- [ ] **Final runs** from the candidates in `AUDIT_2026-11-06.md` §6. FA on the 10% grid (C1) ran 2026-11-11; the rest are probably not run (audit D-H)
- [ ] Which post-hoc runs the report covers in full and which in a sentence (decide before writing Part VI)

The Q-learning core's decisions (γ, the convergence rule, episode length, episode count, seed count, β and the second Part VI robustness check) were settled on 2026-10-16 and live in `src/config.py`, with the reasoning in `EXPERIMENTS.md` and `REVIEW_HISTORY.md` §6. γ's written justification is in `DATA_CLEANING.md` §8.4 (2026-10-19).

## 7. Known Risks

- **Transaction costs may not reach a credible result in time.** The redesign's code, oracle and tuning are done, but in the tuning pilot the agent learned c2 better than c1, headed for a target 2 to 6 bps/yr worse than the best, and converged on one seed of three (`TC_DRIFT_DESIGN.md` §9, §14 risk 1). At $100M and $1B the oracle is within about 1 bp/yr of always-rebalancing, so the agent's result may not separate from the simple rule. Fallback: report the oracle's results and a diagnosed null agent result, and put the bonus weight on function approximation. A correctly scoped smaller extension beats a broken ambitious one under the rubric.
- **The report is late.** The plan had a first draft by Nov 19, and none exists on Nov 20. Write while the TC runs train (projected at about 3 hours on Albert's Mac from timings on 2026-11-20).
- **Thanksgiving week reduces available time** right before the deadline (Nov 26 to 30).
- **Reproducibility.** The Windows print crash and the false "changed" flags are fixed (`AUDIT_2026-11-06.md` M1, M2). `dev` was merged into `main` on 2026-11-09 (PR #28, F-M1), so a plain clone gets the current code; merge `dev` into `main` again after each later change.

---

*This is a living document. Update it as decisions get made and the timeline shifts.*
