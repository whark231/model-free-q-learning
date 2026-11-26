# Project Prospectus: Model-Free Q-Learning and Optimal Portfolio Choice (Nine Assets)

**Due:** November 30, 2026
**Team:** William & Albert
**Status:** closed 2026-11-26 (William). Every component in §3 is done, the report is `docs/report/REPORT.md` and `.pdf`, and the slides are in `docs/report/slides/`. Every open decision in §6 is closed; the last two are recorded there with their outcome. The one remaining step is William's pull request from `dev` into `main`, then submission. Audit IDs: a bare ID (M8, F-L9, L7) is from `AUDIT_2026-11-06.md`; an ID prefixed A20 (A20-H1, A20-M2) is from `AUDIT_2026-11-20.md`.

---

## TL;DR / In plain terms

Everything the assignment requires has been built, run and independently checked. Both extensions are run; for the transaction-cost runs, the summary statistics re-derived from the committed per-path values by Albert (2026-11-20) and by William (2026-11-22); the per-path scoring itself was not re-run by a separate route. The transaction-cost agent loses to the oracle at every cost level, mostly because it heads for the wrong portfolio; at the largest cost level ($10B) its trade timing is worth +6.2 bps/yr against rebalancing to its own target, and it beats always-rebalancing by 5.0 bps/yr. The report and the slides are written, and the project closed on 2026-11-26.

**Key concepts to understand**

- **Required work versus bonus.** Parts I to VI earn the base grade (five 20% buckets). Bonus credit needs an extension that is "correctly designed, benchmarked, and interpreted"; complexity alone earns nothing.
- **Post hoc.** Every run after `baseline-v1` was designed after its results were seen. Each was registered with an expected direction before it ran, and the report labels them post hoc.
- **Where the TC agent heads versus when it trades.** Its gap to the oracle splits exactly into the cost of its favourite target (rebalancing to its own target against rebalancing to the best one) and the value of its timing (the agent against rebalancing to its own target), minus the oracle's own lead over always-rebalancing (`TC_DRIFT_DESIGN.md` §7, TC-3).

**Key takeaways**

- `baseline-v1` is the headline result for Parts IV and V.
- The TC agent runs are done. Of the 12 expectations registered before them (`TC_DRIFT_DESIGN.md` §7), 8 are met and 4 partly met (E1, E7, E9, E11; `results/tc-drift/runs/README.md`). The extension is reported with its agent results, labelled post hoc.
- The report is `docs/report/REPORT.md` (the final draft, promoted 2026-11-26): 13 pages of main text and references, then Appendices A to C. The slides match it.

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
| # | Component | Notes | Status (2026-11-26) |
|---|---|---|---|
| 13 | Research report (10 to 15 pages) in the assignment's structure | Sections split in §4 | ✅ `docs/report/REPORT.md` and `.pdf`, 13 pages of main text and references, then Appendices A to C. Drafts in `docs/report/drafts/`: the skeleton, v1 (Albert, 2026-11-24, PR #51), v2 with Albert's markup (2026-11-25), and the final draft with William's markup, shortened (2026-11-25, PRs #53 and #54; changes in `CHANGES_v2_to_final.md`). Promoted to `REPORT.md` on 2026-11-26 |
| 14 | Clean, reproducible code and README | | ✅ Closed 2026-11-26. The README was rewritten for graders: results, a map from each assignment part to report section, code and docs, quick start, seeds, requirements and expected outputs, plus `results/README.md`. Fresh-clone check on Windows (Python 3.10.2, numpy 2.2.6, pandas 2.3.3), 2026-11-26: `src.data.clean` rebuilds `data/processed/` to floating-point rounding (largest relative difference in the covariance matrix 3.7e-15; the files differ in their last printed digits) and `run_experiments --dry-run` lists the four baseline experiments. Tests on Windows, 2026-11-26: 759 of 761 passed. One failure was a real bug, `src/benchmarks/closed_form_table.py` writing CRLF on Windows (`58d23e5`), fixed the same day (LF pinned; the rebuilt CSV is byte-identical to the committed one), after which that test passes; the other is the known `test_committed_report_matches_a_rebuild` matplotlib failure documented in `reports/README.md`. Earlier fixes: the Windows print crash, line-ending hashing and the README's reproduce command (2026-11-09; `AUDIT_2026-11-06.md` M1, M2, F-H1); the TC README commands (TC-10, A20-M3, A20-M7, `c58f50f`); A20-H3 (`src/benchmarks/closed_form_table.py`) and A20-M2 (the six TC oracle tables committed), both `58d23e5`. M9 and the `tc_sweep_report.py` print fix were closed as not needed (D-G). Not retrained in the check |
| 15 | Class presentation | | ✅ 2026-11-25: `docs/report/slides/` (`build_slides.py` builds `slides.html` and `slides.pdf`; the talk is in `SPEAKER_NOTES.md`). 15 slides plus 3 backup, matched to the final draft, with a title slide and a diagram of the data pipeline and training loop as slide 4. About 11 minutes at the times in the notes |

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

## 5. Timeline

Weeks 1 to 8 are done: data, environment, benchmark, grid and reward, tabular baseline, full baseline run and robustness runs (by 2026-10-29), FA and the TC sweep (2026-11-03 to 2026-11-06). The original order put TC in week 7 and FA in week 8; in practice both ran in week 8.

| Week | Dates | Focus | Target output |
|---|---|---|---|
| 8 (rest) | Nov 6 to 12 | Done: every run re-evaluated with Q fit, audit fixes M1 to M5, M10 and the flow-audit items; `dev` merged into `main` Nov 9 (PR #28); the TC redesign decided and its interface frozen (PR #33). Not done: the final run list | Final run list registered with expected directions |
| 9 | Nov 13 to 19 | Done: TC with drift built (PRs #34 to #37), the Nov 17 gate met, the oracle run on every setting, tuning decided and piloted (PR #39). Not done: the first report draft | All non-TC figures and tables final; draft complete |
| 10 | Nov 20 to 26 | Done: the 40 TC agent runs, trained and scored Nov 20 (Albert, 2 h 14 min on his Mac; PR #47), so the Nov 24 go/no-go is met; the HTML TC report; Albert's repo audit and the report skeleton; the report (v1 Nov 24, v2 and the final draft Nov 25) and the slides (Nov 24, matched to the final draft Nov 25) | All results final (done); full draft (done) |
| Final | Nov 26 | Done: final review, slides, the fresh-clone check, the report promoted to `REPORT.md`, the repo organized for graders. William opens the `dev` into `main` pull request and submits | Report, code and README submitted |

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
- [x] **How the first TC design is reported** (D-G): one paragraph in report §6.4 and Figure A13, no tables, so M7 to M11 are closed as not reported (A20-M4; 2026-11-25)
- [x] **TC go/no-go** (Nov 24, William): met; the agent runs finished 2026-11-20, so the TC section reports agent results
- [x] **Who writes the TC section:** written in Albert's v1 draft (2026-11-24), revised in v2 and the final draft
- [x] **Commit the six TC oracle tables** (A20-M2): committed 2026-11-25 (`58d23e5`, about 1 MB)
- [x] **Check 5's form** (D-I, William, 2026-11-20): yes to the 90-block model side and a two-SE bound at the cost levels, labelled post hoc (`TC_DRIFT_DESIGN.md` §5)
- [x] **Report schedule** (D-H, both): the TC go/no-go is Nov 24 (William); draft sections by Nov 24 and a full draft by Nov 26 (Albert's proposal, agreed 2026-11-20)
- [x] **Final runs** (William, 2026-11-22): none beyond those done. FA on the 10% grid (C1) ran 2026-11-11 and the TC agent 2026-11-20; C2, C3, P1 and P2 (`AUDIT_2026-11-06.md` §6) are not run. What is left is writing
- [x] Which post-hoc runs the report covers in full (A20-M8): settled by the report's §6, which follows `REPORT_SKELETON.md` §6
- [x] **E6 verdict** (closed 2026-11-26 by William at project close; registered as a joint decision, so Albert may reopen it): kept as met. E6 is scored on the across-seed mean, as E1 is, and the mean is below myopic at every level (−0.5 at $10B). Table A3 prints the SE (0.54) and the 5-of-10 seed count beside the verdict, and the conclusion says the agent is statistically indistinguishable from myopic at $10B, so the caveat is in the report. No count changed
- [x] **Promote the final draft to `REPORT.md`** (done 2026-11-26): `REPORT_final.md`, `.pdf` and `.tex` are now `docs/report/REPORT.*`, so `build.sh` and Appendix B describe the submitted file. v1, v2, the skeleton and both change logs are in `docs/report/drafts/`

The Q-learning core's decisions (γ, the convergence rule, episode length, episode count, seed count, β and the second Part VI robustness check) were settled on 2026-10-16 and live in `src/config.py`, with the reasoning in `EXPERIMENTS.md` and `REVIEW_HISTORY.md` §6. *2026-11-22:* Albert's audit found the written reasons missing for episode length, episode count, seed count, ε at halfway and K (A20-H1). *2026-11-25:* closed in the report, whose Table 3 gives each setting with its reason; `EXPERIMENTS.md` was not updated. γ's written justification is in `DATA_CLEANING.md` §8.4 (2026-10-19).

## 7. Known Risks

- **The TC result is mixed and must be interpreted with care** (updated 2026-11-22; the time risk is gone). The agent is 10 to 13 bps/yr behind the oracle at zero, $100M and $1B and also loses to always-rebalancing there; only at $10B does it beat always-rebalancing (+5.0 bps/yr). The pilot understated the favourite target's utility loss (2 to 6 bps/yr in the pilot against 9.75 here, at $1B), c1 is poorly learned, and why the target error is so much smaller at $10B is not investigated. Benchmark information shaped the learning rule, the tail averaging and the convergence rule (design §14 risk 5). The report presents the oracle as the benchmark and the split as the explanation, and prints every disclosure listed in `REPORT_SKELETON.md` §6.5. The bonus is graded on whether the extension is "correctly designed, benchmarked, and interpreted".
- ~~**Documentation gaps the rubric grades**~~ Closed 2026-11-25. A20-H1: the reasons are in report Table 3. A20-H2: the derivation is in report §3 (no separate `BENCHMARKS.md`). A20-H3: `src/benchmarks/closed_form_table.py`. A20-H4: the report words the γ = 8 cause as an untested hypothesis and reports neither the surrogate table nor the shrinkage slopes, so no script is needed.
- ~~**The report is late.**~~ Closed: v1 on Nov 24, the final draft on Nov 25.
- ~~**Thanksgiving week reduces available time**~~ Closed: the work finished on 2026-11-26.
- **Reproducibility** (closed 2026-11-26). The Windows print crash and the false "changed" flags are fixed (`AUDIT_2026-11-06.md` M1, M2). The fresh-clone check is in row 14. `dev` goes into `main` through William's final pull request; `main` was last updated 2026-11-24 (PR #52). Known failure: `test_committed_report_matches_a_rebuild` fails where matplotlib differs from the version that built the TC report, and `data.json` rebuilds identically (`reports/README.md`).

---

*Closed 2026-11-26. This file is kept as the record of the plan and how it changed.*
