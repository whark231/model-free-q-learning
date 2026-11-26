# Next Steps

**From:** William
**Date:** 2026-11-06, updated 2026-11-25
**Status (2026-11-25):** all runs are done, and the report (final draft `docs/report/REPORT_final.md`) and slides are written. The TC summary statistics were re-derived from the committed per-path values by Albert (2026-11-20) and by William (2026-11-22); the per-path scoring itself was not re-run by a separate route. What is left is two decisions, housekeeping, the merge and submission. Your open items are in the table below; everything else in this file is the dated record, with closed items marked. Your audit's correction plan (`AUDIT_2026-11-20.md` §6) is the master list for what is left. Audit IDs: a bare ID (M8, F-L9, L7) is from `AUDIT_2026-11-06.md`; an ID prefixed A20 (A20-H1, A20-M2) is from `AUDIT_2026-11-20.md`.

---

## Open for you, 2026-11-25

**TL;DR / In plain terms:** thank you for v1 and v2 of the report, the closed-form script and the oracle tables. The final draft (`REPORT_final.md`) is your v2 with my markup applied and shortened, and the slides match it. What is left for you: two decisions with me, a few housekeeping items on your Mac, and the merge and submission together.

**Key takeaways**
- No result changed. The final draft's changes from your v2 are listed in `docs/report/REPORT_final_CHANGES.md`, including five places where I kept a short version of something I had marked for cutting, and why.
- The slides now open with a title slide, and slide 4 is a diagram of the data pipeline and training loop (Part I).

| Item | What |
|---|---|
| With me | Two decisions. (1) E6: Table A3 marks it met, but at $10B the agent is statistically indistinguishable from myopic (−0.5, SE 0.54), so "partly met" may fit better. (2) Copy `REPORT_final.md` over `REPORT.md` so `build.sh` reproduces the submitted PDF |
| Housekeeping (A20-M6, A20-L12, A20-L15) | `HANDOFF.md` on your Mac and the memory pointer to it; `pypdf` on your Mac for `CLAUDE.md`'s PDF instruction; the empty notebook; your stale local branches. Nine merged branches can be deleted on GitHub (list in `REVIEW_NOTES_FOR_WILLIAM.md`) |
| Oracle timing | The README gives the oracle run as 20 minutes to 2.5 hours, design §5 as 1 to 2.5 hours. Low priority; the report does not quote it |
| Before submission, together | `dev` into `main` (8 commits ahead on 2026-11-25), then the fresh-clone check (tests, data rebuild, `run_experiments --dry-run`; no retraining) |

## Closed 2026-11-25 (from the 2026-11-22 list)

- **Your report sections, the TC section and the presentation:** written (v1, PR #51; v2; final draft PRs #53 and #54; slides in `docs/report/slides/`).
- **Skeleton after the runs:** done 2026-11-23 (`ee8b088`). The skeleton is now superseded by the report.
- **A20-H1:** the reasons are in report Table 3, including that the 13M budget was sized with a benchmark-scored pilot (report §4, "Use of the benchmark"). `EXPERIMENTS.md` was not updated.
- **A20-H2, your half:** the reward and agent are in report §4. The README's four empty documentation cells were not rechecked.
- **README (A20-M3, A20-M7):** fixed 2026-11-24 (`c58f50f`): the clean commit `1787f0d`, 2 h 14 min, the committed `.npz` files, `runs/<level>/`, and the pointers to `reports/tc-drift/` and `python -m src.tc_drift.report`.
- **TC report test on Windows:** not fixed. The cause is documented in `reports/README.md` (`c58f50f`), and `data.json` rebuilds identically. It stays a known failure on machines whose matplotlib differs.
- **Report text:** "trade once, then hold" is used in Table 10 and the text. For F-L10, the report says no seed converges on the 10% grid (§6.5); I did not recheck each weights claim against it.
- **With me:** D-G (one paragraph and Figure A13, no tables), who writes the TC section (you, in v1), A20-M2 (oracle tables committed, `58d23e5`) and post-hoc coverage (report §6) are settled. L7 is left as it is. F-L9 is covered by report Table 3 (episodes and days).
- **A20-H3** (mine, done by you): `src/benchmarks/closed_form_table.py` (`58d23e5`).

## From William, 2026-11-20 (evening): answers, and three suggestions for `run.py`

**TL;DR / In plain terms:** your questions are answered, and your run driver is the one we use. Three optional improvements to it follow; none blocks the runs. I also edited a few of your files' status lines (listed at the end).

*Outcome, 2026-11-22:* you took suggestions 1 and 2 (PR #47); 3 was not needed, since scoring took about 13 minutes a level. PR #43 is closed. The oracle-timing question at the end is still open (your audit's M7).

**Answers to your 2026-11-20 note.** E1 to E12 stay as you registered them; I am not adding to them (D-F). `tune.run_one` now defaults to `m = config.M_EVAL_PILOTS`, with a test (merged with this note). 40,050 is right. Check 5: yes to both questions (D-I, design §5).

**My parallel run driver.** I had built one on `feature/tc-drift-runs` (commits `abbbcc6` and `53aeec0`) in parallel with yours. Yours is merged, so mine will not be: that branch's PR is to be closed unmerged. Its settings matched yours exactly. Three things it did that yours does not, offered as suggestions for you to take or leave (none blocks the runs):
1. **Commit each seed's `.npz`.** Yours keeps them git-ignored, so the tail-averaged θ and the D12 checkpoints live only on your machine; redoing the scoring from the repository then means retraining (about 3 hours, bit-identical only on the same platform). Mine added `!results/tc-drift/runs/*/seed*.npz` to `.gitignore` (about 150 KB each, 6 MB in all, measured on the pilot's logs) and also wrote the averaged and final θ into each seed's JSON.
2. **A path SE for the seed-averaged gap.** Every seed is scored on the same paths, so the across-seed SE misses the path noise. Mine also reported the gap of the seed-averaged policy value, paired path by path against each reference, with its SE. Your `values.npz` already holds every seed's per-path values, so this can be computed after the runs as long as you keep that file (it is git-ignored).
3. **Batched scoring.** A `DriftAgent.greedy_actions(states)` that gives exactly `greedy_action`'s choices for many states at once, about twice as fast (checked on 117,000 state-policy pairs from the pilot; `abbbcc6`). Only worth it if scoring is slow at m = 4,000.

**Edits to your files in this close-out:** the README's pointer to the audit (now names §9.6); the status block of `RUN_ISSUES_2026-10-22.md` (every §4 step done, with where each was checked); a D-I bullet in your "Check 5 measured" part of design §5. One question: the README gives the oracle run as 20 minutes to 2.5 hours, design §5 as 1 to 2.5 hours. Where does the 20 minutes come from? If it is your machine, cite it in §5 and we use one range.

## From William, 2026-11-06: the audit (every item below is now closed, or open in the table above and in `AUDIT_2026-11-06.md` §9.6)

Three things needed you. First, re-evaluate four runs from your training logs so their Q-value fit numbers exist in the repo; `fa-spline`'s notes quote `baseline-v1` numbers that nothing committed contains. Second, decide with me how to redesign the transaction-cost extension: in your sweep the agent scored the same as a random portfolio even with free trading, and the code is correct, so the problem is the design. Third, a batch of small code fixes that change no number.

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

*Update 2026-11-13 (Albert):* yes, 90 blocks and 1,000 starts. A pilot with simple reference policies gives paired-gap SEs from 0.5 to 7.4 bps/yr, growing with how far apart the policies' holdings get (`TC_DRIFT_DESIGN.md` §7). It cannot tell whether each rebalance pays (every 2 blocks is within 0.4 ± 0.5 bps/yr of every block at $10B), so we revisit m once your oracle exists. For `solve()`: `block_outcome` with drift on uses about 10 KB per path, so score the lattice one point at a time or in chunks of about 1e5 rows, never 5,151 × 2,000 rows in one call. `src/tc_drift/env.py` is implemented and tested (`src/tests/test_tc_drift_env.py`), including the cross-test against your `design_numbers.buy_and_hold_block` loop.

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

## From William, 2026-11-19: status review, and what the TC runs still need from you

**TL;DR / In plain terms:** I read the whole project on `dev` and your `feature/tc-drift-train` branch (`AUDIT_2026-11-06.md` §9). Your train, evaluate and check 5 code works and its tests pass. Before the agent runs can start, three pieces are missing on your side: a script that runs and saves everything, θ in the training log, and one extra reference policy so the agent's gap can be split into "headed for the wrong portfolio" and the rest. The report has not been started, and it is now the biggest risk.

**Key concepts to understand**
- **Two ways to lose against the oracle.** Heading for the wrong portfolio, or trading at the wrong times. In my 2026-11-15 pilot at α = 0.001, the target the agent traded to from the optimum was worth 10.4 bps/yr less than the best target (50 to 118 at α = 0.01). The oracle's whole lead over always rebalancing at $1B is 1.30 ± 0.70 in your replay. So the agent's gap to the oracle will mostly measure the first kind of error unless we split it.
- **The split.** Score one more policy: "always rebalance to the agent's own favourite target", the target with the highest learned value θ·φ. Then agent minus oracle splits exactly into (agent minus that policy) plus (that policy minus always-rebalance-to-the-best) minus (oracle minus always-rebalance-to-the-best), each with a paired SE. The second term is the cost of the favourite-target error. The first is the agent's own decisions against rebalancing to its favourite target; it mixes timing with the agent's value errors at other targets, so it is not directly comparable with the oracle's timing lead. Both policies are built from the agent alone and scored on the same paths, so nothing leaks.

**Key takeaways**
- Your branch's tests pass (121 passed in `test_tc_drift_train_evaluate.py` and `test_tc_drift_spec.py` at `397aaf5`). It edits two of my files (the prospectus and my notes), which I still need to check before it merges. Your two check 5 questions (the finite-horizon model side, and a two-SE bound at the cost levels) are in my list of decisions.
- **Decided 2026-11-19 (me):** recursive least squares on the cost block after a constant-α burn-in, θ at constant α_value; 4M blocks per seed with ε reaching 1% at halfway (β ≈ 2.30e-6); D12 changed, labelled, to "favourite target within one 2% step over the final 10%" plus "95% of probes stable", with the original verdict reported (design §6, §9); go/no-go for the agent runs on Nov 24 (design §11). Set `N_BLOCKS = 4_000_000` and `CHECKPOINT_EVERY = 40_000` in `config.py` when convenient; the hyperparameters follow tuning. Items 1 to 4 below can be built now.

**What the TC runs need from you** (`AUDIT_2026-11-06.md` §9.3). *Albert, 2026-11-20:* items 1 to 4, 6 and 7 are done on `feature/tc-drift-run` (PR #41): `src/tc_drift/run.py`, θ in the log, `evaluate.own_target_rebalance_policy`, m = 4,000, the D7 pointer, and the README section. The paths are drawn once per level and held in memory (about 1.1 GB at peak per level while they are stacked, with the four levels scored at once), which fits on a 16 GB machine. `CHECKPOINT_EVERY` is 40,050 rather than 40,000, so the checkpoints match `tune.py`'s (11 tail snapshots; see `REVIEW_NOTES_FOR_WILLIAM.md`). D-F: my expectations are in `TC_DRIFT_DESIGN.md` §7.
1. **TC-2, a run driver** (`src/tc_drift/run.py` or similar): for each level and seed, build the agent from `config.HYPERPARAMETERS`, call `train()`, save the log, the final θ, c1 and c2, and a manifest (settings, seeds, code hashes, as `src/train.py` does), then score the final greedy policy against the oracle, always-rebalance, trade-once-then-hold, myopic and the TC-3 policy on the evaluation paths, and write `summary.json` and a per-seed CSV. One command per level, so the four levels can run in parallel. Write every text file with `newline="\n"`; `test_reporting.py` checks that. Pin BLAS to one thread (`OMP_NUM_THREADS=1` and friends): on my machine `train()` took 1.96 ms a block with default threading and 0.89 to 1.48 ms pinned, not your 0.57, so please time it on whichever machine runs the jobs.
2. **TC-6, θ in the log:** c1, c2 and ‖θ‖ are already logged at every checkpoint; please add θ itself (87 numbers). The new convergence rule (design §6) reads the favourite target from θ, so it cannot be applied without this.
3. **TC-3, the extra reference policy:** `own_target_rebalance_policy(agent)` in `evaluate.py`, trading every block to argmax over the 1,326 targets of `agent.holding_values(spec.target_grid())`. I can add a `favourite_target()` method to `DriftAgent` if you prefer to call that.
4. **TC-7, m:** your decision under D10. My suggestion is m = 4,000 in `config.M_EVAL`, decided before any agent result, because the D10 rule needs the agent's SE and so would choose m from results. 4,000 halves every SE but may still miss the D10 target at $1B; report the SE either way. The paths are 544 MB per level; `evaluate_policy` reads `len(paths)` and `paths[t]`, so a block-by-block version needs an indexable lazy object that gives every policy the same paths.
5. ~~**Merge your branch**~~ Merged into `dev` 2026-11-19 (PR #37).
6. **D7:** the pointer in `TRANSACTION_COSTS.md` (your file). I added the one in `AUDIT_2026-11-06.md` §6.2.
7. **TC-10:** a TC section in the README once the driver exists.

**Decisions we need together** (audit §9.4): how the first TC design appears in the report (D-G); the schedule, with a TC go/no-go date, since the prospectus's week 10 "light schedule" clashes with the design's Nov 20 to 26 runs (D-H); the agent's expected directions before the runs (D-F); and who runs the tuning pilot (D-C; design §9 says you). *2026-11-20:* D-C done (I ran the pilot), D-F registered by you and kept; D-G and the report schedule remain.

## From William, 2026-11-19 (evening): tuning is done, and the agent is scored on an averaged θ

**TL;DR / In plain terms:** the tuning pilot ran and picked α_value = 3e-4. The agent's weights kept wobbling to the end of training, so the policy we report is now built from the average of its weights over the last 10% of training rather than the last weights. Your run driver needs θ at every checkpoint to do that, and should score both versions.

**Key concepts to understand**
- **Iterate averaging.** With a constant step size, the weights never settle; they keep jittering around a good value as each noisy reward pushes them. Averaging the snapshots over the last stretch cancels most of that jitter, so the averaged weights are a steadier estimate of the same thing. Nothing about the agent's learning changes; only which weights we evaluate.
- **Why the probe rule still fails.** Between the first and second half of the final 10%, the averaged policies send 85% to 91% of the probe states to targets within one 2% step of each other, and agree on hold versus trade at 99.5%+. The exact-action rule counts a move to the neighbouring target as a change.

**Key takeaways**
- Hyperparameters for the final runs (all confirmed 2026-11-20): `alpha_value = 3e-4`, `alpha_cost = 3e-4`, `beta = ln(100) / 2,000,000`, `cost_rls_after = 301_030`, `cost_rls_prior = 1e-3` (the default), `target_refresh_every = 1000`, `l2 = 0`. Designs and numbers in `TC_DRIFT_DESIGN.md` §6 and §9.
- On my machine a 4M-block run took 2.4 hours with six in parallel and BLAS pinned.

**What changes for the run driver (TC-2, TC-6):**
1. Log θ at every checkpoint (TC-6). With `CHECKPOINT_EVERY = 40_000` and 4M blocks, the final 10% holds 10 snapshots.
2. The reported policy: `agent.with_parameters(theta_mean, (c1, c2))`, where `theta_mean` is the mean of the θ snapshots at checkpoints with block > 0.9·N and the costs are the agent's final ones. `src/tc_drift/tune.py` has `trailing_average` and `averaged_policy` (which also applies D12 to the averaged policy) that you can call rather than rewrite.
3. Score the averaged policy and the last-iterate policy on the same paths, and report both; the averaged one is the headline. The TC-3 reference policy uses the averaged agent's `favourite_target()`.
4. **Convergence, restated 2026-11-20** (design §6): a run has converged when the favourite target stays within one 2% step over the final 10% and 95% of probes keep their post-decision holdings within one 2% step. `tune.convergence(log, n_blocks, probes)` computes it, and the first checkpoint from which it holds (blocks, episodes, daily observations) for Part IV. `tune.averaged_policy` returns it for the averaged policy. Report how many of the 10 seeds converge at each level; on the pilot one of three did. `evaluate.probe_convergence` implements only the original rule, so please report D12 through `tune.convergence` (or move it into `evaluate.py`), with all four versions' verdicts. `converged_from_*` is None on a run that has not converged. The rule is frozen as of 2026-11-20.

## From William, 2026-11-17: the oracle has run, and I agree with the paired check 5

**TL;DR / In plain terms:** the oracle is solved for all four portfolio sizes and the two $1B illustrations, and checks 3 and 4 pass everywhere, so 2,000 paths and the 1% lattice are enough (check 3 can say little at zero cost and $100M, where the oracle barely beats always rebalancing). I agree with the paired form of check 5 (design §5), so you can build it into `evaluate.py` as written. The oracle's lead over always rebalancing at $1B is about 1.3 bps/yr, which probably means more than 1,000 evaluation starts.

**Key concepts to understand**
- **Check 3 on the policy.** It asks whether more paths would change what the oracle does. Both policies (from 2,000 and 4,000 paths) are scored in a third, independent model; their mean values differ by at most 0.003 bps/yr, and by at most 0.014 at a single start.
- **A shared level shift.** Doubling the paths lowers the oracle's values by about 3.1 bps/yr, mostly because every portfolio's reward is estimated on the same paths and shares their sampling error. Decisions depend only on differences, so the policy does not move. It does mean the oracle's own V is about 5 bps/yr too high at 2,000 paths, which is one more reason gaps must use its replayed policy (and check 5 the paired form).

**Key takeaways**
- **Check 5:** decided as the paired form. Apply the 0.5 bps/yr to the oracle's lead over always rebalancing, replayed on your evaluation paths, against the same lead in the oracle's model. The in-model leads are in `results/tc-drift/oracle/summary.json` (`lead_over_always_rebalance_bps`). If you find a better form while building `evaluate.py`, tell me.
- **D10:** the $1B lead is 1.32 bps/yr in-model (an upper estimate), so D10's rule wants the agent-against-oracle gap's SE below about 0.44. Your pilot's paired SEs at m = 1,000 were 0.5 to 1.7, which puts m between about 1,300 and 15,000. Measure it once `evaluate.py` runs.
- **Results and how to get an `OracleSolution`.** `results/tc-drift/oracle/summary.json` is committed. The per-setting `.npz` files (policy table, V, G, r̂ and the myopic table) are not in git, since `.gitignore` excludes `results/**/*.npz`; regenerate them yourself (step 6.2 below). `evaluate.oracle_policy(solution)` needs a full `OracleSolution`; rebuild it with `market = env.load_market()`, `s = calibration.cost_settings()["aum_1b"]`, then `oracle.solve(market, s, env.draw_returns(market, np.random.default_rng(spec.ORACLE_PATH_SEED), 2000), block_outcome=env.block_outcome)` (about 5 minutes; by construction it gives the saved table, since the run used the same paths and the same solve, but I have not rerun it). For several settings, `oracle.score` once and `oracle.solve_scored` per setting is faster.
- **A naming clash to avoid in the report.** `oracle_runs.py`'s "never trade" holds the start forever (as in your D10 pilot). Your `evaluate.never_trade_policy(first_target)` trades once, then holds. Its signature is frozen, so I suggest calling it "trade once, then hold" in tables and text.
- **The cross-test is done** (`src/tests/test_tc_drift_oracle.py`): your `env.block_outcome`, `load_market` and `draw_returns` give the same oracle as my test loop.
- **Two of my registered oracle directions failed** (design §8): trading once does not come closer to the oracle as costs grow (it misses the first trade's cost from far starts), and myopic holds slightly less than the oracle under λ only. Both are reported as not met.

Where everything is: `feature/tc-drift-oracle-runs`; design §5 (checks) and §8 (results).

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

**What was still open then** (all closed by 2026-11-20: the cross-test is in, check 5 and check 3 are decided, the agent is tuned)
- **Cross-test.** Whoever merges second adds a test that `env.block_outcome` and my test loop (`loop_block_outcome` in `src/tests/test_tc_drift_oracle.py`) give the same oracle solution.
- **Check 5** (continuous replay) needs your `evaluate_policy`. Its tolerance is 0.5 bps/yr, but a direct replay of one policy has an SE of about 12 bps/yr (your pilot), so I proposed a paired form (design §5). Tell me if you see a better one.
- **Check 3 fails at M = 2,000** at $1B (3.1 bps/yr against 0.5). I decide between more paths and restating it on the policy (design §5).
- **Agent tuning** is blocked on my choice among the §9 options. Your pilot's risk note (an oracle lead over always-rebalance of about 1 bp/yr) still applies.

## Your next steps

**Update 2026-11-09 (Albert):** steps 1, 3 and 4 are done on `dev`; `AUDIT_2026-11-06.md` §2.1 lists each fix and how it was checked. The second (flow) audit's Albert items are fixed too (§8.3), including the fresh-clone reproduce crash (F-H1). No reported number changed. Still Albert's: ~~pushing or deleting the local `grid10-robustness` tag (F-L13)~~ (closed 2026-11-20, on `origin`), and caveating post-convergence weights in the report text (F-L10).

1. ~~**Re-evaluate `baseline-v1`, `n26m`, `all-three` and `grid10-robustness`**~~ Done 2026-11-09 (H3), plus `fa-spline`. Every earlier number came back identical. The `baseline-v1` Q fit matches the numbers in `fa-spline`'s notes (Spearman 0.821, 0.710, 0.920; top-10 recall 35%, 20%, 20% at γ = 5, 3, 8). `b63`, `eps-decay`, `tabular-const` and `fa-quadratic` need William, who holds their logs.
2. **TC redesign, with William** (`AUDIT_2026-11-06.md` H1, H2, §6.2). The redesign is decided (`TC_DRIFT_DESIGN.md`); its Phase 2 tasks are step 6. For the old design: do not run `discount.json` or `learning-rate.json` as designed. Once decided, also: commit the four single-variant configs (M9), rewrite the TC report text (M8), add the M1 line to `tc_sweep_report.py`, and add TC tests (M11). *2026-11-20:* the redesign is built and tuned; these first-design items wait on D-G.
3. ~~**Small code fixes**~~ Done 2026-11-09: M1 (except `tc_sweep_report.py`), M2, M3, M4, M5, plus L2.
4. ~~**`RUN_ISSUES` §4 steps 1 to 3** (M10)~~ Done 2026-11-09: total gap, post-hoc `settled` and `near_optimal_every_episode`, provenance in `summary.json`, definitions in `EVALUATION.md` §2.
5. **Your report sections** (prospectus §4): Data/Calibration, and convergence analysis and visualizations. Use the post-hoc `settled` measure next to the pre-registered rule, labelled post hoc. Who writes the TC section is proposed in the prospectus (you, since you built it) and is yours to accept or change.
6. **TC Phase 2, your side** (updated 2026-11-17; details in the 2026-11-15 and 2026-11-17 sections above). *Albert, 2026-11-20:* all of step 6 is done. 6.2 reproduced William's oracle (2026-11-19); 6.4 is replaced by m = 4,000 fixed in advance (D-E); 6.6 is done; the run driver is on PR #41. What is left is launching the 40 runs. *2026-11-19:* steps 6.1 (with `N_BLOCKS`, the checkpoint interval and the hyperparameters still pending) and 6.3 are done on `feature/tc-drift-train`; the remaining work is in the 2026-11-19 section above (run driver, θ in the log, the own-target reference policy, m fixed now).
   1. `train.py`, `evaluate.py` and the config against the frozen `src/tc_drift/spec.py` (Nov 16 to 18). The oracle, agent and environment are all merged or on `feature/tc-drift-oracle-runs`.
   2. **Regenerate the oracle tables before building `evaluate.py`.** From the repo root on `feature/tc-drift-oracle-runs` (or `dev` once merged), run `python -m src.tc_drift.oracle_runs`. It takes about 1 to 2.5 hours and writes `results/tc-drift/oracle/<setting>.npz` for the six settings (policy table, V, G, r̂, the myopic table), which `.gitignore` keeps out of git. It also rewrites `summary.json`. Afterwards, `git diff results/tc-drift/oracle/summary.json` should show at most the `provenance` block changed (`commit`, `src_dirty`; the `sha256` file hashes should match). A different machine may change a number in its last digit; any larger change (a hold fraction, a lead, a check result) means the run did not reproduce mine, so tell me before using the tables, and do not commit that `summary.json`. To test that it runs first, `python -m src.tc_drift.oracle_runs --quick --out <folder outside the repo>` uses M = 40 and no fine lattice (about 10 minutes); without `--out` it writes to `results/tc-drift/oracle-quick/`, which git does not ignore.
   3. Build check 5 into `evaluate.py` in the paired form, and compare with the in-model leads in `results/tc-drift/oracle/summary.json`.
   4. Measure D10's m: raise it until the agent-against-oracle gap's SE is below a third of the oracle's lead over always rebalancing (expected between about 1,300 and 15,000 at $1B).
   5. Call `never_trade_policy` "trade once, then hold" in tables and text.
   6. D7: add a one-line pointer to `TC_DRIFT_DESIGN.md` in `TRANSACTION_COSTS.md` (your file) and `AUDIT_2026-11-06.md` §6.2. Due since the design merged; I have not added them.
   7. Smoke runs and the timing pilot with me (Nov 18 to 19). The agent's tuning is still my open choice (design §9); I will tell you when it changes `agent.py`.

## To decide together

~~**D10, the TC evaluation size** (above).~~ Approved 2026-11-13; m set to 4,000 by you on 2026-11-20. **The final runs** (`AUDIT_2026-11-06.md` §6; the 2026-11-19 review recommends no more non-TC runs, except P2 if time allows). ~~**D7 pointer** in `TRANSACTION_COSTS.md` (step 6).~~ Done 2026-11-20. **The report schedule** and **how the first TC design is reported** (`AUDIT_2026-11-06.md` §9.4 D-G, D-H). **How the TC report presents the two oracle directions that were not met** (design §8).

**Which post-hoc runs go in the report.** My suggestion:
- `grid10-robustness` as the action-space check, and γ = 3 and 8 as the economic check;
- `b63` and `n26m` for training budgets and episode lengths;
- `eps-decay` short, as mostly a negative result;
- `all-three` as the combined check the 10% grid is compared against (and whether its report should compare with its single-change runs, L7);
- the three constant-α runs in the FA extension.
