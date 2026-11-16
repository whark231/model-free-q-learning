# Next Steps

**From:** William
**Date:** 2026-11-06, updated 2026-11-15
**Status:** the required work is done and independently re-checked. The FA runs are merged and reported. The whole-project audit (`AUDIT_2026-11-06.md`) found no correctness bug in the core, the FA code or the reports pipeline. It did find that the transaction-cost agent cannot learn as designed, plus a list of small fixes, most of them in your files.

---

## TL;DR / In plain terms

Three things need you. First, re-evaluate four runs from your training logs so their Q-value fit numbers exist in the repo; `fa-spline`'s notes quote `baseline-v1` numbers that nothing committed contains. Second, decide with me how to redesign the transaction-cost extension: in your sweep the agent scored the same as a random portfolio even with free trading, and the code is correct, so the problem is the design. Third, a batch of small code fixes that change no number.

**Key concepts to understand**

- **Why the TC agent cannot learn.** It keeps one value per (holding, target) pair: 1.66 million of them. Exploration supplies about 1.41M random actions, so 57% of pairs are ever tried, most of them once, against reward noise of about 3,900 bps/yr per block. With α = 1/N and discount 0.95, the estimated continuation value is also noisy and biased by how often each holding was visited. At zero cost the learned policies' median, 56.6 bps/yr, matches the average grid portfolio's 56.7.
- **Why costs barely matter here.** Targets never drift, so the best policy makes one move. The oracle beats simply ignoring costs by 0.3 bps/yr at the 2 bp anchor. A rebalancing problem needs drift, or larger costs.
- **Line-ending hashes.** Windows checks files out with CRLF line endings, your Mac with LF, so the same file hashes differently. Every "calibration changed" flag in the reports is this.

**Key takeaways**

- Nothing changes your Part I to V results.
- The TC report's text says things the results do not support (`TRANSACTION_COSTS.md`, Outcome). I added a status block and an Outcome section to `TRANSACTION_COSTS.md` but left your design text as it was.

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Data pipeline, simulator, benchmarks, grid, reward | `src/data/`, `src/environment.py`, `src/benchmarks/`, `src/action_space/`, `src/rewards.py` | Rebuilt and re-derived independently in the audit |
| Agent, runner, evaluation, batch runner, reports | `src/agent.py`, `src/train.py`, `src/evaluate.py`, `src/run_experiments.py`, `src/reporting/` | Every seed in every report matches its results to 1e-6 |
| Baseline and tabular post-hoc runs | `results/`, `reports/` | `REVIEW_HISTORY.md` §8, `EXPERIMENTS.md` §7 |
| FA runs `tabular-const`, `fa-quadratic`, `fa-spline` (your `fa-spline` run included) | `results/<run>/`, `reports/<run>/` | `EXPERIMENTS.md` §8.7 |
| TC implementation and quadratic-cost sweep | `src/transaction_costs.py`, `results/tc-quadratic-cost/` | `TRANSACTION_COSTS.md`, Outcome |

442 tests pass on 2026-11-09 (Windows, Python 3.10); the rounding failure in `test_tune_fa.py` is fixed.

## Changes the audit made to your files

All documentation or data, no code:
- `reports/registry.json`: repaired (the `fa-quadratic` merge left invalid JSON on `dev`); notes written for `tabular-const` and `fa-quadratic`; `fa-spline` got the outcome of its registered expectation, an explicit `compare_with` (`baseline-v1`, `tabular-const`, `fa-quadratic`) and a caveat that its `baseline-v1` Q-fit numbers are not committed. Every other run got an explicit `compare_with` equal to the set its report already used. All reports rebuilt; per-run data identical.
- `EXPERIMENTS.md` (status line, §6 shared-streams note, new §8.7 outcome), `EVALUATION.md` (constant-α notes, test count, γ justification done), `ENVIRONMENT.md` and `DATA_CLEANING.md` (stale "still open" items), `reports/README.md` (how variants are run now; the TC report), `TRANSACTION_COSTS.md` (status block and Outcome section).

## From William, 2026-11-09: your items for me are closed

**In plain terms:** everything you left in `REVIEW_NOTES_FOR_WILLIAM.md` outside transaction costs is done and merged into `dev` (`AUDIT_2026-11-06.md` §2.2, `REVIEW_HISTORY.md` §9). While doing it I found one bug in the line-ending work and fixed it in your files, as below.

- **Closed:** the rounding test; L1 (grid increment check); L3 (`requirements.txt`); the working-tree refresh; `b63`, `eps-decay`, `tabular-const` and `fa-quadratic` re-evaluated at clean commit `3322e03`, re-extracted (schema 3) and every report rebuilt, so Q fit shows everywhere (L8, F-M3); the prospectus status (F-M11); and `tune_fa --summarize` is now byte-stable (F-L12, FA part). Every earlier number came back identical, checked field by field.
- **Registry:** the two caveats (`tabular-const` and `fa-quadratic`) that said the table "reads n/a until this run's data.json is re-extracted" now say it was re-extracted on 2026-11-09.
- **A bug in M2, fixed in your files.** `Path.write_text` without `newline=` writes CRLF on Windows. After a fresh checkout (LF, from `.gitattributes`), rebuilding a report on Windows produced CRLF bytes, so `test_every_registered_report_builds_from_committed_data` failed and `build --all` would rewrite every report with only line endings changed. Reproduced by converting one report to LF and running the test. Every non-TC `write_text` now passes `newline="\n"` (and `encoding="utf-8"` where it was missing): `src/reporting/build.py`, `src/reporting/extract.py`, `src/evaluate.py`, `src/run_experiments.py`, `src/data/clean.py`, `src/data/download_raw.py`, plus `src/train.py` and `src/tune_fa.py`. Your CSV writers already pinned LF. Two tests in `test_reporting.py`: a source check that every `write_text` outside the TC files sets `newline`, which fails on every machine including the Mac, and a check that a built report has no CRLF. The two TC writers (`tc_sweep_report.py`, `transaction_costs.py`) are exempt until the redesign.
- **Merged:** `feature/william-audit-fixes` into `dev` (PR #27), then `dev` into `main` (PR #28, F-M1), both on 2026-11-09.
- **Docs moved into `docs/`** on 2026-11-09 (`docs/README.md` is the index). Contents unchanged; only the location tables in `README.md` and `CLAUDE.md` were updated. Pull before editing any doc.

## From William, 2026-11-10

H4 (William's item) is closed: the constant-α experiment (`tabular-const`, `fa-quadratic`, `fa-spline`) is written up in `AUDIT_2026-11-06.md` §2.3 with its settings, the five registered checks, the two wrong reasons and what the report can claim. Every number was re-checked against the committed summaries and matches `EXPERIMENTS.md` §8.7, so nothing in your files changes.

**Update 2026-11-11 (Albert):** replaced by two runs, `fa-spline-grid10-eps3` and `fa-spline-grid10-eps10`, both at the noise-matched α = 2.78e-5 and differing only in ε at halfway (`EXPERIMENTS.md` §8.8).

~~**Run `fa-spline-grid10` is yours (audit C1).** It's registered on branch `run/fa-spline-grid10` (`EXPERIMENTS.md` §8.8) but not trained. Before you train it, update the `RunVariant` in `src/config.py` and the expectations in §8.8, get an independent review, commit, and then run the §8.8 commands (about 30 h for the three γ in parallel).~~ Done 2026-11-11 (Albert): split into `fa-spline-grid10-eps3` and `fa-spline-grid10-eps10`, registered before training, trained and merged to `dev` (PRs #30 and #31; outcomes at the end of `EXPERIMENTS.md` §8.8). Each ε took about 8 h for the three γ in parallel on Albert's machine, not 30 h.

- **Keep:** the 10% grid, B = 63, 26M episodes, the seeds and all three γ, so the run pairs seed by seed with `grid10-robustness`.
- **α: William suggests the noise-matched 2.8e-5,** 3.9× `fa-spline`'s 7.11e-6. It keeps the noise level the tuning chose, rescaled for B = 63 and the 10% grid's features, and uses no μ or Σ. It isn't re-tuned, because re-tuning means changing `tune_fa.py` and about 2 days of compute.
- **Your call: ε at halfway, 3% or 10%.**
  - 3% keeps exploration identical to `grid10-robustness`.
  - 10% gives FA's slow directions more random portfolios to learn from (5.59M uniform episodes against 3.70M; corrected 2026-11-11 from 5.65M and 3.71M, which are 1/β and ignore the 26M-episode cap, to the (1 − e^(−βN))/β count used in `EXPERIMENTS.md` §8.8). The cost is higher regret, and the comparison then differs in exploration as well as learning rule and representation.
  - A mean-field check predicts gaps of about 33.6 / 0 / 1.3 bps/yr at γ = 5 / 3 / 8 with ε at 3%, and 4.4 / 0 / 1.3 with ε at 10%. It uses the true μ and Σ, so disclose it and don't choose by it.
- **Either way, rewrite §8.8's γ = 8 prediction.** It says the bias persists, which no longer holds at this α.

## From William, 2026-11-12: the transaction-cost redesign, Phase 2 starts now

**TL;DR / In plain terms:** the TC extension is redesigned so it can actually work. Three ETFs (SPY, IEF, GLD), holdings that drift between quarterly decisions, realistic trading costs, an oracle benchmark, and an agent that learns costs from its rewards. The design and the shared interface are on branch `feature/tc-redesign`. You build the environment, the training loop and the evaluation; I build the oracle and the agent, in parallel. One decision waits for you (D10, below).

**Key concepts to understand**
- **Block.** One decision period, 63 trading days (about a quarter). At the start the agent looks at what it holds and either holds or trades to one of 1,326 target portfolios on a 2% grid. Then the shares sit for 63 days, the weights drift with returns, and it is scored: the block's mean daily return minus γ/2 times the variance, minus the trading cost spread over the 63 days.
- **HOLD.** Not trading. Free, and the weights keep drifting.
- **Oracle.** The best possible policy, computed with the true μ and Σ. It is the benchmark, used for evaluation only; the agent never sees it.
- **Realistic costs.** A 1 bp spread plus market impact that grows with the square root of trade size over daily volume, at four portfolio sizes: zero (the control), $100M, $1B (headline) and $10B.

**Key takeaways**
- Read `docs/methods/TC_DRIFT_DESIGN.md`: §3 (the model), §7 (evaluation, yours), §8 (the cost levels), §10 (who owns which file and the frozen signatures), §11 (timeline and gate).
- The old design (`src/transaction_costs.py`, `results/tc-quadratic-cost/`, `TRANSACTION_COSTS.md`) stays untouched as a labelled negative result.

**Decided** (William, 2026-11-12; §13 of the design has the reasons):
- **Universe.** The three ETFs the nine-asset optimum holds, chosen to keep the oracle computable. The report discloses this as a benchmark-informed restriction.
- **Drift and timing.**
  - Drift on excess returns: the risk-free rate moves end weights by at most 7.1e-6, and `ENVIRONMENT.md` §4 now says so.
  - The discount is 0.95 per block, a 5-year investor.
- **Your interface.**
  - The environment is split into `draw_returns` and `block_outcome`, so the oracle can score every starting portfolio on the same paths.
  - Evaluation starts are drawn uniformly from the 2% grid.
- **The agent.**
  - One shared value model for trades and HOLD.
  - Exploration draws uniformly from all 1,327 actions.
  - Its learned cost coefficients are kept at zero or above.
- **Convergence.** The baseline's rule, applied on fixed probe holdings.

**Needs your approval: D10, the evaluation size.** 90 blocks per path (about 22 years; at 0.95 per block the rest of the future carries about 1% of the weight) and 1,000 starting holdings. Every policy runs on the same paths, so the gaps between policies are much less noisy than the values. The paths take 136 MB. Check the gap's standard error in your pilot and add paths if it is over a few bps/yr. Say yes, or propose other numbers.

**Your Phase 2 work** (signatures are frozen in `src/tc_drift/*.py` and checked by `src/tests/test_tc_drift_spec.py`; change them only by a PR we both approve):
1. **`src/tc_drift/env.py`**
   - `load_market()`: the SPY, IEF, GLD sub-block of μ and Σ.
   - `draw_returns(market, rng, m, block_length)`: daily excess returns, shape (m, 63, 3).
   - `block_outcome(w0, excess_returns, gamma, drift)`: the block's reward and end weights. With `drift=False` it must equal `episodic_mean_variance_reward` row by row; test that.
   - `simulate_block(...)`: the wrapper.
2. **`src/tc_drift/train.py`.** The loop:
   1. pick an action;
   2. play the block;
   3. charge `spec.trade_cost`;
   4. pass the agent `spec.net_reward` and the drifted holdings;
   5. restart from a random 2% target every `restart_every` blocks;
   6. log the greedy action on `probe_states` every `checkpoint_every` blocks.

   Until my agent lands, test it with stub agents (random, always HOLD, always rebalance) that implement `spec.TCAgent`.
3. **`src/tc_drift/evaluate.py`**
   - `evaluate_policy`, with the metrics in design §7: discounted net reward, oracle gap in bps/yr, hold fraction, trades, turnover per block, mean trade size, mean held weights.
   - The adapters `agent_policy` and `oracle_policy`.
   - The reference policies `always_rebalance_policy` and `never_trade_policy`. The myopic reference is my `oracle.solve(..., discount=0)`, scored through `oracle_policy`, so you need no code for it.
4. **Run config** (your choice: `src/tc_drift/config.py` or JSON under `experiments/tc-drift/`). Not in `src/config.py`, which is fingerprinted in every run's evaluation record. The cost levels come from `src/tc_drift/calibration.py` (`cost_settings()`).
5. **Random streams** (design §4): reported run i uses `SeedPlan(20261016, 10).run_streams(i)` for returns and agent, with starts from `agent_rng.spawn(1)[0]`. Evaluation paths come from the evaluation stream; tuning uses master seed 20261112.

**How we work in parallel:**
- Branch from `feature/tc-redesign` (or from `dev` once it merges), for example `feature/tc-drift-env`.
- Edit only your files; anything in mine goes through a PR.
- Whoever merges second writes the cross-test that checks your `block_outcome` against my reference loop on the same random numbers.

**Dates** (design §11):
- **Nov 13:** design and `spec.py` merged into `dev`.
- **Nov 13 to 16:** your environment, my oracle.
- **Nov 16 to 18:** your train/evaluate/config, my agent; then run the oracle on all four cost levels.
- **Nov 18 to 19:** smoke runs, timing pilot, tuning.
- **Nov 20 to 26:** full runs (4 levels × 10 seeds).
- **Gate Nov 17:** your environment plus my oracle's first two checks must pass, or we fall back to the no-drift design.

## From William, 2026-11-15: the oracle and the agent are built

**TL;DR / In plain terms:** my side of Phase 2 is written and tested. The oracle passes checks 1 and 2, so with your `env.py` the Nov 17 gate is met. The agent works, but a pilot on your environment shows it cannot yet learn the trading costs through the reward noise, so its tuning waits on a decision I owe you. You can build `train.py` and `evaluate.py` against both now.

**Key concepts to understand**
- **Numerical oracle.** For each of the 5,151 holdings on the 1% lattice it computes the best action by value iteration, using the true μ and Σ through fixed simulated paths. Its in-sample value is slightly too high (it maximizes over noisy estimates), so gaps must use its policy replayed on your evaluation paths, never its V.
- **Why the agent struggles.** One block's reward has noise of about 1e-3 per day; the trading cost it must learn is about 6e-6 per unit traded. Learning a number a thousand times smaller than the noise takes a very long average.

**Key takeaways**
- Both modules accept your `env.block_outcome` unchanged; I checked offline against your branch, and the oracle gives the same solution with it as with my test loop (reward estimates within 3.5e-19).
- Your timeline is unchanged for `train.py`, `evaluate.py` and the config. The agent runs (Nov 20 to 26) depend on the tuning decision.

**Where the code is.** `feature/tc-drift-oracle` (the oracle) and `feature/tc-drift-agent` (features and agent, built on top of the oracle branch). Details and numbers: `TC_DRIFT_DESIGN.md` §5 and §6, "Implemented".

**What you can call**
- **Oracle** (`src/tc_drift/oracle.py`):
  - `solve(market, settings, paths, block_outcome=env.block_outcome)` returns an `OracleSolution`. Paths must have 63 days; draw them with `np.random.default_rng(spec.ORACLE_PATH_SEED)`.
  - About 2 minutes per cost level at M = 2,000.
  - `solution.policy(holdings)` takes (3,) or (m, 3) and returns the optimal action(s), so `oracle_policy` can call it directly.
  - Myopic reference: `solve(..., discount=0.0)`, then score its `policy_table` with the ρ = 0.95 solution's `value_of`, or replay it through `oracle_policy`.
  - Fields: `V`, `G`, `policy_table`, `r_hat`, `cv_correction`, `next_index`, `U`, `cost`, `block_length`, `n_iterations`.
- **Agent** (`src/tc_drift/agent.py`):
  - `DriftAgent(rng, alpha_value=..., alpha_cost=..., beta=..., target_refresh_every=...)`, plus optional `l2`, `n_basis` and `degree`. Any other keyword raises TypeError.
  - The `spec.TCAgent` methods, and read-only diagnostics you may log: `cost_linear`, `cost_quadratic`, `theta`, `epsilon`, `n_updates`, `q_values(state)`, `holding_values(W)`.
  - States are (3,) holdings summing to one.
  - About 0.75 ms per block with your `block_outcome`.

**What `train.py` must do for the agent**
- Call `update(state, action, spec.net_reward(R, cost), next_state)` with the true drifted end-of-block holdings, even on a block after which you restart. A restart only changes where the next block starts. It is not a terminal state, so the agent still bootstraps from where this block ended.
- Use `greedy_action` for the probe-state log. It draws no random numbers and changes nothing.

**What is still open**
- **Cross-test.** Whoever merges second adds a test that `env.block_outcome` and my test loop (`loop_block_outcome` in `src/tests/test_tc_drift_oracle.py`) give the same oracle solution.
- **Check 5** (continuous replay) needs your `evaluate_policy`. Its tolerance is 0.5 bps/yr, but a direct replay of one policy has an SE of about 12 bps/yr (your pilot), so I proposed a paired form (design §5). Tell me if you see a better one.
- **Check 3 fails at M = 2,000** at $1B (3.1 bps/yr against 0.5). I decide between more paths and restating it on the policy (design §5).
- **Agent tuning** is blocked on my choice among the §9 options. Your pilot's risk note (an oracle lead over always-rebalance of about 1 bp/yr) still applies.

## Your next steps

**Update 2026-11-09 (Albert):** steps 1, 3 and 4 are done on `dev`; `AUDIT_2026-11-06.md` §2.1 lists each fix and how it was checked. The second (flow) audit's Albert items are fixed too (§8.3), including the fresh-clone reproduce crash (F-H1). No reported number changed. Still Albert's: pushing or deleting the local `grid10-robustness` tag (F-L13), and caveating post-convergence weights in the report text (F-L10).

1. ~~**Re-evaluate `baseline-v1`, `n26m`, `all-three` and `grid10-robustness`**~~ Done 2026-11-09 (H3), plus `fa-spline`. Every earlier number came back identical. The `baseline-v1` Q fit matches the numbers in `fa-spline`'s notes (Spearman 0.821, 0.710, 0.920; top-10 recall 35%, 20%, 20% at γ = 5, 3, 8). `b63`, `eps-decay`, `tabular-const` and `fa-quadratic` need William, who holds their logs.
2. **TC redesign, with William** (`AUDIT_2026-11-06.md` H1, H2, §6.2). On hold. Do not run `discount.json` or `learning-rate.json` as designed. Once decided, also: commit the four single-variant configs (M9), rewrite the TC report text (M8), add the M1 line to `tc_sweep_report.py`, and add TC tests (M11).
3. ~~**Small code fixes**~~ Done 2026-11-09: M1 (except `tc_sweep_report.py`), M2, M3, M4, M5, plus L2.
4. ~~**`RUN_ISSUES` §4 steps 1 to 3** (M10)~~ Done 2026-11-09: total gap, post-hoc `settled` and `near_optimal_every_episode`, provenance in `summary.json`, definitions in `EVALUATION.md` §2.
5. **Your report sections** (prospectus §4): Data/Calibration, and convergence analysis and visualizations. Use the post-hoc `settled` measure next to the pre-registered rule, labelled post hoc. Who writes the TC section is proposed in the prospectus (you, since you built it) and is yours to accept or change.

## To decide together

**D10, the TC evaluation size** (above). **The final runs** (`AUDIT_2026-11-06.md` §6).

**Which post-hoc runs go in the report.** My suggestion:
- `grid10-robustness` as the action-space check, and γ = 3 and 8 as the economic check;
- `b63` and `n26m` for training budgets and episode lengths;
- `eps-decay` short, as mostly a negative result;
- `all-three` as the combined check the 10% grid is compared against (and whether its report should compare with its single-change runs, L7);
- the three constant-α runs in the FA extension.
