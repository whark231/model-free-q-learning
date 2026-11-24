# Project Prospectus: Model-Free Q-Learning and Optimal Portfolio Choice (Nine Assets)

**Due:** November 30, 2026
**Team:** William & Albert
**Status:** living plan, updated 2026-11-22 (William). Parts I to VI are done, and so is every computation for both extensions: function approximation (2026-11-06 and 2026-11-11) and the 40 transaction-cost agent runs (Albert, 2026-11-20, PR #47; results in `results/tc-drift/runs/README.md`, HTML report `reports/tc-drift/report.html`). No further runs are planned (William, 2026-11-22). What remains is writing (the report, the presentation, the documentation gaps in Albert's audit) plus small code and repo tasks (A20-H3, A20-H4, A20-M2, the `dev` into `main` merge). Albert's audit (`AUDIT_2026-11-20.md` §6) found no blocker in code or results. Audit IDs: a bare ID (M8, F-L9, L7) is from `AUDIT_2026-11-06.md`; an ID prefixed A20 (A20-H1, A20-M2) is from `AUDIT_2026-11-20.md`. Still open from the older audit: D-G, M12 and F-L10 (report text), F-L9 (partly fixed), L7 (`all-three`'s comparison set). `dev` was last merged into `main` on 2026-11-20 (PR #45); PRs #46 and #47 came after.

---

## TL;DR / In plain terms

Everything the assignment requires has been built, run and independently checked. Both extensions are run; for the transaction-cost runs, the summary statistics re-derived from the committed per-path values by Albert (2026-11-20) and by William (2026-11-22); the per-path scoring itself was not re-run by a separate route. The transaction-cost agent loses to the oracle at every cost level, mostly because it heads for the wrong portfolio; at the largest cost level ($10B) its trade timing is worth +6.2 bps/yr against rebalancing to its own target, and it beats always-rebalancing by 5.0 bps/yr. The eight days left are for the 10 to 15 page report, the presentation, and the documentation gaps Albert's audit found.

**Key concepts to understand**

- **Required work versus bonus.** Parts I to VI earn the base grade (five 20% buckets). Bonus credit needs an extension that is "correctly designed, benchmarked, and interpreted"; complexity alone earns nothing.
- **Post hoc.** Every run after `baseline-v1` was designed after its results were seen. Each was registered with an expected direction before it ran, and the report labels them post hoc.
- **Where the TC agent heads versus when it trades.** Its gap to the oracle splits exactly into the cost of its favourite target (rebalancing to its own target against rebalancing to the best one) and the value of its timing (the agent against rebalancing to its own target), minus the oracle's own lead over always-rebalancing (`TC_DRIFT_DESIGN.md` §7, TC-3).

**Key takeaways**

- `baseline-v1` is the headline result for Parts IV and V.
- The TC agent runs are done. Of the 12 expectations registered before them (`TC_DRIFT_DESIGN.md` §7), 8 are met and 4 partly met (E1, E7, E9, E11; `results/tc-drift/runs/README.md`). The extension is reported with its agent results, labelled post hoc.
- The report has not been started and is the main risk (§7). Its outline, with every number tied to a committed file, is `docs/report/REPORT_SKELETON.md`.

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

Full suite: 758 tests passing on 2026-11-20 on Albert's Mac (`python -m pytest src/tests -q`); 761 tests now, 760 passing on Windows on 2026-11-22 (row 14).

### Extensions (bonus)
| # | Component | Notes | Status (2026-11-06; TC updated 2026-11-22) |
|---|---|---|---|
| 11 | Transaction costs: state = previous target, quadratic cost on the move | `TRANSACTION_COSTS.md`, `src/transaction_costs.py` | ✅ Redesign run and scored 2026-11-20 (end of this cell). First design: built; quadratic-cost sweep run (0, 1, 2, 5 bp; 10 seeds). The code is correct but the agent learns nothing: at zero cost it scores like a random portfolio, 468 bps/yr below the oracle. Discount and learning-rate comparisons registered, **not run**, and should not be run as designed. Kept as a labelled negative result. **Redesigned 2026-11-12 as TC with drift** (`TC_DRIFT_DESIGN.md`, `src/tc_drift/`): three ETFs, 63-day blocks, realistic costs at $100M, $1B and $10B. As of 2026-11-17: environment, oracle, agent, training and evaluation built and tested; the oracle is solved at all four sizes plus the κ-only and λ-only illustrations (William, PR #36); checks 1 to 4 pass (check 3 restated on the policy, post hoc), and check 5 is inside tolerance but resolved only at zero cost. The oracle beats always-rebalancing by 1.3 ± 0.7 bps/yr at $1B and 7.1 ± 1.3 at $10B. *2026-11-20:* tuned (William, PR #39: recursive least squares on the cost block, α_value = 3e-4, 4M blocks per seed, the policy scored on θ averaged over the last 10%; §6, §9). The run driver (`src/tc_drift/run.py`, Albert, PR #41) trains and scores the 40 runs; expected directions E1 to E12 registered before any run (§7). *2026-11-20 (Albert, PR #47):* the 40 runs trained and scored from clean commit `1787f0d`. Agent − oracle −13.1 / −13.1 / −10.0 / −2.8 bps/yr at zero / $100M / $1B / $10B (across-seed SD 4.6 / 5.5 / 4.3 / 1.7; all 10 seeds below the oracle at every level); agent − always-rebalance −13.1 / −12.7 / −8.2 / +5.0. Most of the gap is where it heads (split: own target − always rebalance −13.1 / −12.8 / −9.0 / −1.1); timing is worth +6.2 at $10B. The oracle's lead over always-rebalancing on these 4,000 paths is 1.78 ± 0.36 at $1B and 7.79 ± 0.64 at $10B (the 1.3 and 7.1 above are the replay check's 1,000 paths). E1 to E12: 8 met, 4 partly met. Results `results/tc-drift/runs/README.md`, report `reports/tc-drift/report.html`; the summary statistics re-derived from the committed per-path values by Albert (2026-11-20) and by William (2026-11-22); the per-path scoring itself was not re-run by a separate route |
| 12 | Function approximation: linear Q over quadratic or spline features | `FA_INTEGRATION_NOTES.md`, `EXPERIMENTS.md` §8 | ✅ Built, step size tuned on separate seeds, and run 2026-11-06: `tabular-const`, `fa-quadratic`, `fa-spline` (30 seeds each). Both FA models beat the constant-α tabular agent at γ = 5 and 3 and ranked the reward landscape better (Spearman); of the registered expectations, split into five checks, four failed in whole or part (`EXPERIMENTS.md` §8.7). Every run now carries Q-fit numbers and shows them in its report (2026-11-09) |

### Deliverables
| # | Component | Notes | Status (2026-11-22) |
|---|---|---|---|
| 13 | Research report (10 to 15 pages) in the assignment's structure | Sections split in §4 | ⬜ Not started (2026-11-22). Outline: `docs/report/REPORT_SKELETON.md` (Albert's audit, 2026-11-20), which maps each assignment item to a section with its numbers and sources; its TC results part is still to fill from `results/tc-drift/runs/README.md`. Inputs: `reports/<run_id>/report.html`, `results/<experiment>/evaluation/`, `EXPERIMENTS.md` §7 and §8.7, `REVIEW_HISTORY.md` §7 and §8 |
| 14 | Clean, reproducible code and README | | 🔶 Windows print crash and line-ending hashing fixed, and the README's reproduce command works in a fresh clone (2026-11-09; `AUDIT_2026-11-06.md` M1, M2, F-H1). TC with drift has its README commands (2026-11-20, TC-10; C13 closed). Still open: the old TC design's single-variant configs (M9) and the print fix in `tc_sweep_report.py`, which depend on how the first design is reported (D-G). `dev` merged into `main` 2026-11-09 (PR #28, F-M1) and 2026-11-20 (PR #45). *2026-11-22:* the TC runs' per-seed logs and per-path values are committed, so TC scoring can be redone without retraining; the oracle tables are still git-ignored (A20-M2); the budget-only closed-form table has no script (A20-H3); tests: 760 of 761 passed on Windows (Python 3.10.2, numpy 2.2.6, matplotlib 3.10.9) on 2026-11-22; the one failure, `test_committed_report_matches_a_rebuild`, comes from the matplotlib version changing the SVG bytes, and `data.json` rebuilds identically (detail in `NEXT_STEPS_FOR_ALBERT.md`) |
| 15 | Class presentation | | ⬜ Not started (2026-11-22) |

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
| 10 | Nov 20 to 26 | Done: the 40 TC agent runs, trained and scored Nov 20 (Albert, 2 h 14 min on his Mac; PR #47), so the Nov 24 go/no-go is met; the HTML TC report; Albert's repo audit and the report skeleton. To do: the TC interpretation and the report. Report schedule (D-H, proposed by Albert, agreed by William 2026-11-20): draft sections by Nov 24, full draft by Nov 26 | All results final (done); full draft |
| Final | Nov 27 to 30 | Final review, slides, `dev` into `main`, a fresh-clone check (tests, data rebuild and `run_experiments --dry-run`; no retraining), submission | Report, code and README submitted |

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
- [ ] **How the first TC design is reported** (D-G, both): a paragraph, or with its tables (then M8 and M9 must be fixed first). Albert's audit recommends one paragraph and the four-row table from `TRANSACTION_COSTS.md`, closing M7 to M11 as not reported (A20-M4)
- [x] **TC go/no-go** (Nov 24, William): met; the agent runs finished 2026-11-20, so the TC section reports agent results
- [ ] **Who writes the TC section** (prospectus §4: proposed for Albert, to agree)
- [ ] **Commit the six TC oracle tables** (about 1.05 MB; both; A20-M2). Recommended: yes, so a grader can score the TC runs without a rebuild of 20 minutes to 2.5 hours (8,452 s once, not a clean timing)
- [x] **Check 5's form** (D-I, William, 2026-11-20): yes to the 90-block model side and a two-SE bound at the cost levels, labelled post hoc (`TC_DRIFT_DESIGN.md` §5)
- [x] **Report schedule** (D-H, both): the TC go/no-go is Nov 24 (William); draft sections by Nov 24 and a full draft by Nov 26 (Albert's proposal, agreed 2026-11-20)
- [x] **Final runs** (William, 2026-11-22): none beyond those done. FA on the 10% grid (C1) ran 2026-11-11 and the TC agent 2026-11-20; C2, C3, P1 and P2 (`AUDIT_2026-11-06.md` §6) are not run. What is left is writing
- [ ] Which post-hoc runs the report covers in full and which in a sentence (decide before writing Part VI). Proposal in `REPORT_SKELETON.md` §6 and A20-M8

The Q-learning core's decisions (γ, the convergence rule, episode length, episode count, seed count, β and the second Part VI robustness check) were settled on 2026-10-16 and live in `src/config.py`, with the reasoning in `EXPERIMENTS.md` and `REVIEW_HISTORY.md` §6. *2026-11-22:* Albert's audit found the written reasons missing for episode length, episode count, seed count, ε at halfway and K (A20-H1, open). γ's written justification is in `DATA_CLEANING.md` §8.4 (2026-10-19).

## 7. Known Risks

- **The TC result is mixed and must be interpreted with care** (updated 2026-11-22; the time risk is gone). The agent is 10 to 13 bps/yr behind the oracle at zero, $100M and $1B and also loses to always-rebalancing there; only at $10B does it beat always-rebalancing (+5.0 bps/yr). The pilot understated the favourite target's utility loss (2 to 6 bps/yr in the pilot against 9.75 here, at $1B), c1 is poorly learned, and why the target error is so much smaller at $10B is not investigated. Benchmark information shaped the learning rule, the tail averaging and the convergence rule (design §14 risk 5). The report presents the oracle as the benchmark and the split as the explanation, and prints every disclosure listed in `REPORT_SKELETON.md` §6.5. The bonus is graded on whether the extension is "correctly designed, benchmarked, and interpreted".
- **Documentation gaps the rubric grades** (Albert's audit, 2026-11-20). Part III's "report and justify" list has values but no written reasons (A20-H1); there is no methods prose for Part II's derivation (A20-H2); the budget-only closed-form portfolio is in no committed table (A20-H3); three FA analyses have no committed script (A20-H4). Owners in `AUDIT_2026-11-20.md` §6.
- **The report is late.** The plan had a first draft by Nov 19, and none exists on Nov 22. Section drafts are due Nov 24 and the full draft Nov 26 (D-H).
- **Thanksgiving week reduces available time** right before the deadline (Nov 26 to 30).
- **Reproducibility.** The Windows print crash and the false "changed" flags are fixed (`AUDIT_2026-11-06.md` M1, M2). `dev` was merged into `main` on 2026-11-09 (PR #28, F-M1), so a plain clone gets the current code; merge `dev` into `main` again after each later change (last done 2026-11-20, PR #45; `dev` has gained PRs #46 and #47 since). Tests: 758 passed on Albert's Mac on 2026-11-20; 760 of 761 passed on Windows (Python 3.10.2, numpy 2.2.6, matplotlib 3.10.9) on 2026-11-22; the one failure, `test_committed_report_matches_a_rebuild`, comes from the matplotlib version changing the SVG bytes, and `data.json` rebuilds identically (detail in `NEXT_STEPS_FOR_ALBERT.md`).

---

*This is a living document. Update it as decisions get made and the timeline shifts.*
