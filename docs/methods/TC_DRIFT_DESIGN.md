# Transaction Costs with Drift: Design

**Owners:** William (oracle, agent, features, tuning design) and Albert (environment, training, evaluation, config, runs) · **Status (2026-11-20):** Phase 2 built and tuned. Merged into `dev`: `spec.py`, `calibration.py`, `env.py`, `train.py`, `evaluate.py`, `config.py` and `oracle_replay.py` (Albert), `oracle.py`, `features.py`, `agent.py`, the oracle runs and the tuning pilot (`tune.py`, William). On `feature/tc-drift-run` (Albert, PR #41): the run driver `run.py`, θ in the training log, the own-target reference policy, the final settings in `config.py` and the expected directions (§7). The 40 agent runs have not started. **Post hoc** relative to `baseline-v1`; label every result that way.

---

## TL;DR / In plain terms

The first transaction-cost design failed for two reasons. The agent had 1.66 million (holding, target) values to learn and almost no data for each, and portfolios never drifted, so there was nothing to rebalance (`AUDIT_2026-11-06.md` H1, H2). That design stays in the repo, unchanged, as a labelled negative result.

The new design uses the assignment's original three ETFs (SPY, IEF, GLD). It lets holdings drift with returns during each 63-day block. At the start of each block the agent either holds or trades to a target on a 2% grid, and pays a cost with a linear part and a quadratic part. A numerical oracle, computed with the true μ and Σ on a fine lattice, is the benchmark. The agent is a spline function-approximation learner that never sees μ, Σ, the cost coefficients or the oracle. It learns the costs from the net rewards it receives.

**Key concepts to understand**

- **State.** The agent's holdings at the start of a block. They have drifted since the last trade, because the shares were fixed and prices moved. Holdings are observable at the decision date and they set the cost of any trade, which is the assignment's test for adding a state variable (Part III, "State").
- **HOLD.** Not trading. It costs nothing and leaves the weights to keep drifting.
- **Linear versus quadratic cost.** A linear cost charges the same per unit traded, so small trades are not worth it: the optimal policy leaves holdings alone inside a "no-trade region" and trades only when they wander out. A quadratic cost makes small trades nearly free and large trades expensive, so the optimal policy trades part of the way at a time.
- **Post-decision value G(x).** What it is worth to start a block holding x, before costs. Every action ends at some x (a target, or the current holdings for HOLD), so Q(s, a) = G(x) − cost(s, a)/B. The oracle and the agent both use this structure.
- **Control variate.** A way to estimate an average with less noise. Score the same simulated paths with a quantity whose average is known exactly, and subtract the noise the two share.

**Key takeaways**

- The oracle is affordable because three assets make the state a two-dimensional triangle. A 1% lattice has 5,151 states, and the 1,326 trade targets are all lattice points.
- One block's reward is very noisy: a standard deviation of 2,390 bps/yr at the optimum under buy-and-hold. The oracle therefore needs the control variate, which cuts that to 103 bps/yr on the same paths.
- The decisions are small next to that noise. Letting the optimum drift for one block costs 1.3 bps/yr, and for 20 blocks 26 bps/yr. The agent gets no control variate, so whether it can resolve these decisions is the main risk (§14).
- §13 lists 15 conflicts with the repo (C1 to C15) and 12 decisions (D1 to D12). All are decided; D10 (the evaluation horizon and path count) was approved by Albert on 2026-11-13.
- The costs are realistic: about 1 bp spreads plus square-root market impact from our own volume data, at portfolio sizes of $100M, $1B (headline) and $10B (§8).

---

## 1. Scope and status

- **What is new:** the package `src/tc_drift/` and `src/tests/test_tc_drift_spec.py`.
- **What is untouched, by instruction:**
  - `src/transaction_costs.py`, `src/tc_sweep_report.py` and `results/tc-quadratic-cost/`;
  - `TRANSACTION_COSTS.md` and `AUDIT_2026-11-06.md`. *Updated 2026-11-20:* the audit has since been edited at William's request (the D7 pointer in §6.2 and the 2026-11-19 status review in §9); `TRANSACTION_COSTS.md` has its pointer too (Albert, 2026-11-20).
- **Three shared files changed, each by one line or one table row:**
  - `src/tests/test_environment.py`: the two new agent modules are added to `AGENT_MODULES`, so the existing leakage scan covers them.
  - `CLAUDE.md` and `docs/README.md` list this document.
- **Phase 1 ends here.** After approval each person gets one implementation prompt.

## 2. Locked decisions, checked against the repo

| Decision | Value | Repo check |
|---|---|---|
| Universe | SPY, IEF, GLD; the μ, Σ sub-block at positions (0, 3, 6) of `TICKERS` | Positions verified. Chosen because the nine-asset optimum holds only these three, and three assets keep the oracle computable (C1, decided) |
| γ | 5 headline | Matches `GAMMA_BASELINE` |
| Actions | 1,326 targets on the 2% grid (`generate_weight_grid(3, 0.02)`) plus HOLD at index 1,326; 1,327 in all | Count verified (C(52, 2)); the grid builder accepts 0.02 |
| State | Continuous holdings; the oracle rounds to the 1% lattice (5,151 points) | Count verified (C(102, 2)); every 2% target is a lattice point (verified) |
| Block length B | 63 trading days | `MIN_EPISODE_LENGTH` is 2, so allowed |
| Discount ρ | 0.95 per block | About 0.81 per year at 4 blocks a year. The assignment asks to explain a positive discount (C15) |
| Agent step size | Small constant α, tuned on separate seeds | As for the 20% grid FA runs |
| Cost | κ‖a − s‖₁ + (λ/2)‖a − s‖₂², from the drifted pre-trade holdings; HOLD costs 0 | Implemented in `spec.trade_cost`, tested |
| Reward | Buy-and-hold within the block; realized mean − (γ/2) realized variance (ddof = 1) | Same estimator as `src/rewards.py` (ddof = 1) |
| No-drift mode | Weights reset daily, next state = post-trade weights; reward = baseline reward | Testable against the old grid oracle (§5, check 1) |
| Model-free agent | Never receives μ, Σ, κ, λ, γ or oracle output; may receive the target grid's weights | Enforced by the contract tests (§10) |

## 3. The model

**Timing within block k:**
1. The agent observes its holdings s_k, which lie on the simplex.
2. It picks an action a_k.
3. The post-trade weights are x_k = target(a_k) for a trade, or x_k = s_k for HOLD.
4. It pays cost c(s_k, a_k) = κ‖x_k − s_k‖₁ + (λ/2)‖x_k − s_k‖₂², a fraction of wealth, taken pro rata from every holding, so the post-trade weights are exactly the target. HOLD costs 0. ‖·‖₁ counts both legs: moving 20 points from SPY to IEF has ‖u‖₁ = 0.4 and ‖u‖₂² = 0.08.
5. Over B = 63 days the shares are fixed:
   - the day-t portfolio excess return is p_t = x_{k,t} · e_t, with x_{k,1} = x_k;
   - weights then move with gross returns, x_{k,t+1,i} ∝ x_{k,t,i}(1 + e_{t,i}), renormalized to sum to one (risk-free rate: C2);
   - the next state is s_{k+1} = x_{k,B+1}.
6. The agent receives the block reward R(x_k) = mean_t(p_t) − (γ/2)·var_t(p_t, ddof = 1), and the net reward r_k = R(x_k) − c(s_k, a_k)/B. Daily units throughout, as the assignment requires.

**Objective:** maximize E Σ_k ρ^k r_k.

**Bellman equation.** The future after a decision depends only on x, so define the post-decision value

G(x) = E[R(x) + ρ V(s')], where s' is the end-of-block holdings starting from x.

Then

Q(s, a) = G(target(a)) − c(s, a)/B for a trade, Q(s, HOLD) = G(s), and V(s) = max_a Q(s, a).

Trades and HOLD differ only in where G is evaluated and in the cost.

**No-drift mode.** The weights are x_k on every day of the block and s_{k+1} = x_k. With ddof = 1, E[R(x)] = U(x) = x'μ − (γ/2)x'Σx exactly. The reward equals `episodic_mean_variance_reward(x, block, γ)`, the baseline's reward.

## 4. Units and conventions

- **Weights:** positional in `spec.ASSETS` order. Costs are fractions of wealth (1 bp = 1e-4). Rewards are daily.
- **`spec.net_reward(R, c)`** is R − c/B.
- **`spec.annualized_bps(V)`** converts a discounted value to bps per year:
  - infinite horizon: (1 − ρ)·V·252·10⁴, the constant daily reward with the same value;
  - finite horizon of H blocks: divide by Σ_{t<H} ρ^t instead.
- **Gaps** between two policies are differences of these, on the same paths.
- **Reference trades:** a 20-point transfer (‖u‖₁ = 0.4, ‖u‖₂² = 0.08) and a 40-point transfer (0.8, 0.32), each given a total cost in bp. `spec.coefficients_from_reference_trades(c20, c40)` solves for κ and λ. Both are nonnegative only when 2·c20 ≤ c40 ≤ 4·c20:
  - c40 = 2·c20 means κ only;
  - c40 = 4·c20 means λ only.

**Ties.** On a lattice state that is itself a 2% target, HOLD and "trade to it" both cost 0 and lead to the same weights. HOLD wins ties, in the oracle and in the agent, and the metrics count a trade of zero size as a hold. This is the tie-breaking rule the assignment asks us to report (Part III).

**Random streams** (`spec.py`):
- Reported run i: `SeedPlan(MASTER_SEED, n_runs).run_streams(i)` gives the returns and agent generators. Training start states come from `agent_rng.spawn(1)[0]`, as the old TC code did.
- Evaluation paths: `SeedPlan(...).evaluation_rng()`.
- The oracle's fixed paths: `np.random.default_rng(spec.ORACLE_PATH_SEED)`, never a training or evaluation stream.
- Tuning: the same layout under `TUNING_MASTER_SEED`, never the reported seed.

## 5. Oracle (William)

**Discretization.** States are the 5,151 points of the 1% lattice. Each one is both a possible holding (for HOLD) and, if it is on the 2% grid, a target. So one simulation per lattice point gives everything:
- the expected block reward r̂(x);
- the lattice indices of the end-of-block holdings, rounded with `spec.nearest_lattice_index`, which is tested against brute force.

It is a sample-average approximation (M paths, rounding to the lattice, and a maximum taken over noisy estimates, which biases its in-sample V upward). So it is called a numerical oracle, and every reported gap uses its policy replayed on the evaluation paths, never its in-sample V.

**Fixed paths.** M blocks of returns are drawn once from `np.random.default_rng(spec.ORACLE_PATH_SEED)` and passed to `solve` as `excess_paths`, and every lattice point is scored on the same M paths (common random numbers). `solve` also takes the block-scoring function, so William can test with his own reference loop before Albert's `env.block_outcome` lands. The same generator seed is reused for every cost setting, so differences between settings are not sampling noise. M = 2,000, set by check 3 (2026-11-17, below).

**Control variate.** r̂(x) = U(x) + mean_m[R_bh(x; m) − R_const(x; m)]:
- R_bh is the buy-and-hold block reward on path m;
- R_const is the constant-weight reward on the same path, whose mean is exactly U(x).

At the γ = 5 optimum this takes the standard deviation from 2,390 to 103 bps/yr, about 540 times less variance. The correction there is −0.6 ± 0.7 bps/yr (20,000 paths, §12). The oracle reports the correction's size for every setting.

**Solve.** Value iteration on the §3 equations:
- G on the lattice is r̂ + ρ·mean_m V(next);
- V(s) = max(G(s), max_a [G(a) − c(s, a)/B]).

The cost matrix is 5,151 × 1,326, 55 MB in float64. The next indices are stored as int32, 5,151 × M, 41 MB at M = 2,000. Each iteration multiplies V by a sparse transition matrix built from them (321,395 nonzeros at M = 2,000, because one block of drift moves holdings only a few lattice points). `discount = 0` gives the myopic reference policy; `value_of(table, discount=0.95)` scores it.

**Implemented** (`src/tc_drift/oracle.py`, 2026-11-15; tests in `src/tests/test_tc_drift_oracle.py`):
- **Time.** At M = 2,000 on the 1% lattice, one setting solves in about 113 s on William's laptop: about 100 s to score the 5,151 lattice points (two `block_outcome` calls each) and about 10 s for 434 value-iteration sweeps. Albert's `env.block_outcome` and the tests' day-by-day loop give the same solution on the same paths.
- **Stopping rule.** The old grid solver's, max |V_new − V| ≤ 1e-12·(1 − ρ). One final sweep from the converged V produces G, V and the policy table, so V(s) = Q(s, policy(s)) exactly in terms of the returned G. G and the gather r̂ + ρ·mean V(next) then differ by at most the stopping tolerance.
- **Units.** `solve` rejects paths whose block length is not `spec.BLOCK_LENGTH`, so its cost per day (c/B) matches `spec.net_reward` in the environment and the evaluation.
- **Validation.** It rejects non-finite paths, returns at or below −100% (even with drift off, when the paths are otherwise unused), and a `block_outcome` that returns the wrong shapes or non-finite values.
- **Memory.** The cost matrix, the sweeps and the transition matrix are processed 1,024 rows at a time. On the 0.5% lattice (check 4) at M = 2,000 the solve peaks at 474 MB (tracemalloc) and takes about 6.4 minutes.
- **Extra fields** on `OracleSolution`, beyond those in §10: `U`, the cost matrix, `block_length` and `n_iterations`.
- **Checks 1 and 2 pass** (the gate). Check 1 runs at $1B on the 1% lattice and at $10B and a steep cost on the 2% lattice; values agree with the old solver to 1e-11, and policies agree on every state whose best two destinations differ by more than 1e-11. In check 2 with drift, the oracle's lead over always rebalancing to the best target is 0.008 bps/yr at the median lattice state (M = 2,000; at most 0.012).
- **Checks 3 and 4 pass at every setting** (2026-11-17; `python -m src.tc_drift.oracle_runs`, 52 minutes; merged `env.block_outcome`; numbers in `results/tc-drift/oracle/summary.json`). One scoring of the lattice serves every cost setting and the myopic reference, because the scoring does not depend on κ, λ or ρ (`oracle.score` and `oracle.solve_scored`; `solve` is the two in a row and its signature is unchanged).
  - **Check 3 is restated on the policy** (decided William 2026-11-17, after the V-based form failed, so post hoc). The M = 2,000 and M = 4,000 policy tables are both scored with `value_of` in an independent model: 4,000 paths from SeedSequence entropy (20261016, 4), which neither policy was fitted on. It passes if the mean difference over the 1,326 2% starts is under 0.5 bps/yr in size, so it can fail in either direction. The independent model is the reviewer's correction of my first version, which scored both in the M = 4,000 model; that model shares its first 2,000 paths with the M = 2,000 policy, so its bias has no fixed sign. That form is in `summary.json` too.
  - **Check 4** compares the mean V over the 2% starts on the 0.5% and 1% lattices, as decided.

| Setting | Oracle's lead over always rebalancing | Check 3: policy difference | Check 3 informative | Same action at 2% starts (M doubled) | V change with M doubled (old form) | Check 4: V change |
|---|---|---|---|---|---|---|
| Zero cost | 0.008 | −0.001 | No | 100% | −3.13 | −0.000 |
| $100M | 0.214 | −0.001 | No | 83.7% | −3.12 | +0.000 |
| $1B | 1.319 | −0.003 | Yes | 86.6% | −3.09 | +0.000 |
| $10B | 6.939 | −0.000 | Yes | 89.1% | −3.01 | −0.003 |
| κ only ($1B) | 1.542 | −0.002 | Yes | 77.3% | −3.07 | −0.012 |
| λ only ($1B) | 1.488 | −0.000 | Yes | 85.7% | −3.09 | +0.006 |

  All in bps/yr. The leads are in-sample in the oracle's own model, so biased upward (§8).
  - **Where check 3 says little.** At zero cost and $100M the oracle's lead over always rebalancing is itself under the 0.5 tolerance, so even always rebalancing would pass (independent review, 2026-11-17). There the check shows only that the path count costs less than that lead. Where it is informative, the largest difference is 0.2% of the lead. At every setting with costs, doubling M changes the action at 11% to 23% of the 2% starts, between actions worth almost the same: the largest difference at a single start is 0.014 bps/yr.
  - **Why the V-based form failed.** Doubling M lowers V by about 3.1 bps/yr at every start and every setting. Most of that is sampling error in r̂ shared by every lattice point; the preliminary note (2026-11-15) put it down to the maximization bias. At the best target alone, a fixed portfolio with no maximization involved, the control-variate correction falls by 3.13 bps/yr from the first 2,000 paths (+4.37, SE 2.39) to all 4,000 (+1.24), matching the 3.09 drop in V at $1B. Every lattice point is scored on the same paths, so the error is shared: it moves V at the 2% starts by nearly the same amount (the per-point correction itself ranges from 0 at the corners to 4.6 bps/yr), and decisions depend only on differences. The correction near the optimum is about −0.5 ± 0.7 bps/yr when −0.08 (independent 4,000 paths) and −0.62 ± 0.73 (20,000 paths at the continuous optimum, another seed, §12) are pooled, so on the M = 2,000 paths the oracle's in-sample V is about 4.9 ± 0.7 bps/yr too high, on top of the maximization bias. Gaps never use it, and check 5's paired form cancels it.
  - **M stays at 2,000.**
- **Cross-test done** (`src/tests/test_tc_drift_oracle.py`): `env.load_market` and `env.draw_returns` give the tests' market and paths, and `env.block_outcome` gives the same solution as the tests' loop (identical next states and policy, r̂ within 1e-15).
- **Cross-test, second copy** (Albert, `src/tests/test_tc_drift_train_evaluate.py`, written before William's landed): also checks V agrees to 1e-12. Either can be dropped later.
- **Check 5 measured** (Albert, 2026-11-17, `python -m src.tc_drift.oracle_replay --path-count`; results in `results/tc-drift/checks/oracle_replay_2026-11-17.json`). M = 2,000 oracle paths; replay on H = 90 blocks from 1,000 grid starts, tuning seed's evaluation stream. It uses the paired form decided below, with one change: the model side is the finite-horizon value over the same 90 blocks (`evaluate.model_value`), not V and `value_of`, so both sides discount and truncate the same way (ρ^90 is about 1% of the value).

  | bps/yr (SE) | zero | $100M | $1B | $10B |
  |---|---|---|---|---|
  | Oracle lead over always-rebalance, replay | −0.03 (0.08) | −0.17 (0.43) | 1.30 (0.70) | 7.10 (1.26) |
  | Same lead in the oracle's model | 0.01 | 0.21 | 1.29 | 6.79 |
  | **Check 5: replay minus model** | **−0.03 (0.08)** | **−0.38 (0.43)** | **0.01 (0.70)** | **0.31 (1.25)** |
  | Lattice rounding: rounded replay minus continuous | −0.04 (0.04) | −0.10 (0.06) | −0.09 (0.07) | −0.04 (0.13) |
  | Unpaired, as worded: replayed value minus model value | −22.4 (12.0) | −22.7 (12.0) | −22.3 (11.9) | −21.9 (11.8) |
  | Oracle hold fraction in the replay | 0.15 | 0.28 | 0.32 | 0.45 |
  | Turnover per block, oracle / always-rebalance | 2.9% / 3.0% | 2.2% / 3.0% | 2.1% / 3.0% | 1.5% / 3.0% |

  - Every check-5 mean is inside 0.5 bps/yr, but only the zero-cost one is resolved: at $1B and $10B the SE is above the tolerance, so a pass there rules out errors only to about ±1.4 and ±2.5 bps/yr (two SEs). The SE covers the evaluation paths only, not the oracle's own fixed-path noise.
  - Rounding holdings to the 1% lattice costs at most about 0.1 bps/yr, consistent with check 4.
  - The unpaired comparison is about −22 at every level because the levels share the evaluation paths; at 1.9 SEs it is the path noise §7 predicts, not a measured bias.
- **Check 3 in the replay** (Albert, $1B, a second form of William's restated check): the M = 4,000 policy minus the M = 2,000 policy, paired on the same evaluation paths, is −0.11 ± 0.15 bps/yr. It agrees with William's independent-model form above (−0.003).
- **The oracle's trading in a continuous replay at M = 2,000** (the replay above, 1,000 starts, against William's in-model numbers in §8): hold fraction 0.15 / 0.28 / 0.32 / 0.45 against 0.156 / 0.272 / 0.308 / 0.443, and turnover 2.9% / 2.2% / 2.1% / 1.5% against 2.95% / 2.26% / 2.13% / 1.57%. The in-model statistics hold up off the lattice.

**First look at the policy** (M = 1,000, Albert's `block_outcome`, γ = 5; a sanity check, not a result). The best target is (0.46, 0.14, 0.40) at every level. The share of lattice states within an L1 distance of 0.04 of it where the oracle holds rises with portfolio size: zero cost 0.16, $100M 0.47, $1B 0.63, $10B 1.00. At $10B, trades from within 0.10 of the best target stop about 0.046 short of it rather than going all the way. Both match the §8 expected directions. The zero-cost holds are the noise-driven ones §8 anticipates.

`OracleSolution` carries V, G, the policy table, r̂, the control-variate correction per lattice point and the next-state indices, so the checks and the report can read them directly.

**Checks the tests must run (Phase 2).** Two differ from the Phase 1 prompt's wording (C6).
1. **No drift equals the old grid oracle.** With drift off, from states on the 2% grid, compare with `src.transaction_costs.optimal_policy` (imported read-only), given U on the grid and the new cost matrix divided by B.
   - That function returns only a policy, so get its V by evaluating that policy, and compare values to a tolerance.
   - Compare policies by post-trade weights (HOLD equals "trade to the current target"), except where two actions are within the tolerance.
   - The old solver stops at 1e-12·(1 − ρ); use the same stopping rule.
   - r̂ = U holds exactly only if R_bh and R_const come from the identical routine when drift is off, or if `solve` uses U directly in that mode. The test must check which.
2. **Zero cost** (κ = λ = 0):
   - With drift off, V(s) = max(U(s), U*)/(1 − ρ) exactly, where U* is the best target's U. HOLD keeps an off-grid lattice start forever, and such a start can beat every target.
   - With drift on, V(s) ≥ r̂(a*)/(1 − ρ) for every s, because rebalancing to the best target every block is always available; record how close they are.
3. **Path count.** Doubling M changes the value by less than a tolerance fixed before running.
4. **Lattice size.** The 0.5% lattice (20,301 points) changes the value by less than a tolerance fixed before running.
5. **Continuous replay.** Replaying the oracle's policy on continuous, unrounded holdings, on fresh paths, gives a value close to V. Expect V to be above the replay, because of the maximization bias above and, at M = 2,000, the shared sampling error in r̂ (about 5 bps/yr; checks 3 and 4 above). The difference bounds the error the evaluation will see.

**Tolerances for checks 3 to 5 (decided William, 2026-11-15, before any of them ran): 0.5 bps/yr.** (Check 5 has since been measured; see "Check 5 measured" above.) Checks 3 and 4 compare the mean over the 1,326 2% starts (the evaluation's start distribution, D4) of the annualized V; the largest change at a single start is reported too. Check 5 cannot be held to 0.5 bps/yr as worded: one policy's replayed value has a standard error of about 12 bps/yr at the D10 size (§7 pilot), so a direct "replay against V" comparison cannot resolve it. Decided (William, 2026-11-17; Albert told in `NEXT_STEPS_FOR_ALBERT.md`): apply the 0.5 bps/yr to a paired quantity, the oracle's lead over always rebalancing, in the replay on the evaluation paths against the same lead in the oracle's own model (V − `value_of` of the always-rebalance table). The pilot's paired SEs (0.5 to 1.7 bps/yr) mean this may need more than 1,000 starts. Albert may still propose a better form once `evaluate.py` exists.

## 6. Agent (William)

**Value model** (one shared model, decided D5):
- trades: Q̂(s, a) = θ · φ(target(a)) − θ₁‖target(a) − s‖₁ − θ₂‖target(a) − s‖₂²;
- HOLD: Q̂(s, HOLD) = θ · φ(s).

By §3 both are the post-decision value G, evaluated at the weights the agent ends up holding, so one parameter vector θ serves both: every step, trade or hold, teaches the value of holding a portfolio. θ₁ should approach κ/B and θ₂ should approach λ/(2B), if θ·φ is unbiased.

**Features:**
- φ is the additive-plus-pairwise cubic B-spline map for three assets, in the new file `src/tc_drift/features.py` (C4), applied to a target's weights for a trade and to the current holdings for HOLD.
- On the 2% grid it has 87 columns of rank 45, and max ‖φ‖² = 6 everywhere on the simplex, because each asset's basis sums to one.
- The two distance features are bounded: ‖u‖₁ ≤ 2 and ‖u‖₂² ≤ 2.
- With three assets the third weight is one minus the other two, so a pairwise term is already a full function of the state. The only restriction the model imposes is smoothness. Say this in the report.
- **Stability.** For a trade the full feature vector (spline features plus the two distances) has max squared norm 6 + 4 + 4 = 14. With per-block step sizes the least-mean-squares condition is Σ_b α_b·max‖x_b‖² < 2, or α < 2/14 ≈ 0.14 with one α. This is the bound for regression on fixed targets; bootstrapped learning has no such guarantee, which is why the divergence check below exists.

**Learning:**
- **Target:** y = r + ρ·max_{a'} Q̂⁻(s', a'), where Q̂⁻ uses a frozen copy of the parameters refreshed every N updates.
- **Cached part:** the 1,326 values θ⁻·φ(target) are computed once per refresh; the max then only needs s' and the distances to each target.
- **Update:** semi-gradient, with a small constant α per parameter block (the shared value θ, and the two cost terms).
- **Cost coefficients stay at zero or above** (decided D11): after each update, θ₁ and θ₂ are set to max(0, ·). A negative estimate would mean the agent believes trading earns money, and it would trade constantly. This uses only the fact that costs cannot be negative, not their size; disclose it.
- **Exploration** (decided D6): ε-greedy with ε_t = e^{−βt}, as the assignment allows. An exploratory step draws uniformly from all 1,327 actions, HOLD included, so HOLD is tried on about 0.08% of exploratory steps. That is enough because of the shared model: HOLD's value θ·φ(s) is learned from every step, so HOLD needs exploring only for the transitions it leads to.

**Divergence risk (the "deadly triad").** Function approximation plus bootstrapping plus off-policy learning can diverge. The mitigations are the frozen target copy, a small α and bounded features. The diagnostic, after training and for evaluation only, compares the agent's implied G on the lattice with the oracle's and tracks ‖θ‖ over training.

**Ties and hyperparameters.** HOLD wins ties (§4). The constructor accepts only the keywords in `DriftAgent.HYPERPARAMETERS` and rejects anything else, so μ, Σ, κ, λ, γ or oracle values cannot be passed in by mistake.

**Convergence (D12, decided 2026-11-12).** The assignment says "you must define convergence before examining the final results" (Part IV). Proposed:
- **Primary:** the greedy action on a fixed set of probe states (every 2% target plus a fixed sample of off-grid lattice states, drawn once from the tuning seed) is unchanged at every checkpoint over the final 10% of training, the same K = 10% as the baseline's `greedy_stable_through_end`.
- **Secondary, as in the baseline's `near_optimal_for_good`:** from some checkpoint on, the greedy policy's replayed value stays within 25 bps/yr of the oracle's (evaluation only).
- **Descriptive:** the change in θ between checkpoints, and, at evaluation only, the replayed oracle gap over training.
- `train()` records the probe-state greedy actions every `checkpoint_every` blocks, so the rule can be applied without rerunning.
- **Changing the rule (William, 2026-11-17).** The review's 30,000-block run already failed it (below). If the tuning pilot shows the rule cannot be met, it may be changed, but only before any final run's results are examined, which is what Part IV requires. A change is labelled as one, here and in the report, with its date, the reason and the pilot evidence, and the original rule's verdict is reported next to the new one's.
- **Changed 2026-11-19 (William), before the tuning pilot and before any final run, labelled as a change.** The probe rule above would almost certainly fail: the 2026-11-15 review's 30,000-block run changed every probe in the last 20%, the first TC design's equivalent rule failed on every seed (`AUDIT_2026-11-06.md` M7), and the oracle itself changes its action at 11% to 23% of the 2% starts when its path count doubles, between actions worth nearly the same (§5). The new rule:
  - **Primary:** at every checkpoint over the final 10% of training, the favourite target (the target with the highest θ·φ, where the agent would trade with free trading) lies within L1 distance 0.04 (one 2% step) of the final checkpoint's favourite target.
  - **Secondary:** at least 95% of the probe states never change their greedy action over that window (the stricter reading: the same 95% throughout, not a different 95% at each checkpoint). It was meant to cover the hold-or-trade decision, but the probes barely test it (see the 2026-11-20 restatement below).
  - The original probe rule's verdict is reported next to both. All three measure stability, not accuracy (`AUDIT_2026-11-06.md` §2.3). They need θ at each checkpoint in the training log (`AUDIT_2026-11-06.md` TC-6).
- **Changed again 2026-11-19 (William), after the tuning pilot (§9) and before any final run, labelled as a change.** The pilot's last-iterate policies failed both rules above on all six runs: in the final 10% the favourite target moved by L1 0.08 to 0.32, and at most 0.3% of probes kept their action. ε was 2.7e-4 at the window's start and 1.0e-4 at the end (about 75 exploring blocks expected in the window's 435,550), so the movement is θ under a constant α_value fitting noisy rewards. **Disclosure:** this change was decided after seeing the last-iterate pilot's criterion values and its benchmark diagnostics (the oracle gaps and the favourite target's utility loss), as well as its D12 failure; disclose it with §14 risk 5. It is on the tuning seed only, no final agent run exists, it does not change the α_value choice (§9), and it did not raise every run's criterion (α_value = 3e-4 seed 0 lost 1.7 ± 0.4 bps/yr). The policy that is evaluated and reported is now greedy on the **tail average**: the mean of the θ snapshots at the checkpoints in the final 10% of training (11 snapshots at 4 million blocks and 40,050-block checkpoints), with the costs the agent ended with (RLS costs are already a running average). This is a thinned tail average of the iterates, in the spirit of Polyak–Ruppert averaging (which averages every iterate, classically under decaying steps). Under a constant step it reduces the iterates' noise around where they settle but not any bias of the constant step, and 11 snapshots 40,050 blocks apart can cut the variance by at most about a factor of 11. Part IV lists "a moving average of portfolio weights or rewards stabilizes" among acceptable criteria; this is the same idea applied to θ. The rules above are applied to the policy we would report had training stopped at each checkpoint of the final-10% window: the trailing 10% average of θ there, with the costs logged there. Consecutive trailing averages share most of their snapshots, so they move less than the raw iterates by construction; the window's first trailing average uses θ from 79% to 89% of training, so the primary rule's comparison of it with the final one is between two averages that share no snapshot. A split-half check compares the averages of the first and second halves (5 and 6 snapshots) of the final 10%, which share none, and is reported as a pass or fail. Two definitions of "final 10%" are in use: the D12 window starts at the last checkpoint at or before 90% (block 3,564,450, so 12 checkpoints), while the tail average uses the checkpoints after 90% (11 snapshots). The last-iterate policy's value and verdicts are reported alongside. Code: `tune.averaged_policy`, `DriftAgent.with_parameters`.
- **Secondary rule restated 2026-11-20 (William), after the tuning pilot and before any final run, labelled as a change.** The exact-action secondary failed on every pilot run, and the pilot showed why: at α_value = 3e-4, between the two halves of the final 10% the averaged policies' post-decision holdings are within one 2% step at 85% to 91% of probes, so most changes are moves to a neighbouring target among 1,326 targets with nearly equal values (§9). The secondary now reads: at least 95% of the probe states, the same ones throughout the window, have **post-decision holdings** (the target traded to, or the probe itself on HOLD) within L1 0.04 (one 2% step) of the final checkpoint's at every checkpoint of the window. The primary is unchanged. A run has **converged** when both hold. The 2026-11-19 exact-action secondary and the original rule are reported alongside. For Part IV ("how many observations and episodes are required to reach the criterion, how often convergence occurs across seeds"), each run also reports the first checkpoint from which both rules hold through the end of training, in blocks, episodes (90-block restarts) and daily observations (blocks × 63); a run whose rules hold only from inside the final 10% has not converged. The threshold of one step was chosen by William before this rule was applied to the pilot runs, but after the split-half distances in §9 had been seen; disclose it with the tail-average change. Code: `tune.convergence` with `probes`, applied to the six pilot runs with `tune recheck`; `evaluate.probe_convergence` implements the original rule only. The "converged from" figures are measured back from the final policy, so they are not a stopping rule; a run that has not converged reports none (its `rules_hold_from_block` says where the rules start to hold). **What the rule does not test:** almost every probe is a state where trading is right, so at the final averaged policy only 1 to 10 of the 1,526 probes choose HOLD, while the same agents hold in 12% to 55% of evaluation blocks. The secondary rule, which forgives 76 probes, therefore says almost nothing about the hold-or-trade decision, and it passes a probe that switches between HOLD and a trade to a neighbouring target. Each run reports, as a diagnostic, the probes that hold at any checkpoint of the window and the share of them whose hold-or-trade decision never changes (§9). **Frozen 2026-11-20:** no further changes to D12; the final runs report the verdicts of all four versions (2026-11-12, 2026-11-19 exact action, 2026-11-19 tail average, 2026-11-20). The rule's code is `tune.convergence` as committed on `feature/tc-drift-rls` on 2026-11-20.

**What to disclose.** The agent receives:
- the target grid's weights, which are action definitions;
- its own holdings;
- the net rewards.

The two distance features build in the cost family's shape (linear and quadratic in the trade) but not its size, which the agent learns. It never receives μ, Σ, κ, λ, γ or the oracle (§10).

**Implemented** (`src/tc_drift/agent.py` and `src/tc_drift/features.py`, 2026-11-15; tests in `src/tests/test_tc_drift_agent.py`):
- **Features.** 87 columns, rank 45 on the 2% grid, max ‖φ‖² = 6, as above. The pair products kept are those that can be nonzero on the simplex, decided from the knots, so the columns never depend on the input; `src/features.py` instead drops columns that are zero on the rows it is given, which works only because it is always called on the whole grid. The basis is evaluated as piecewise cubics fitted once from `src.features.bspline_basis` (scipy costs about 0.5 ms a call); it matches scipy to 1.6e-15, tested. Degree 0 is rejected.
- **Hyperparameters.** Required: `alpha_value`, `alpha_cost`, `beta`, `target_refresh_every`. Optional: `l2` (0), `n_basis` (5), `degree` (3). Anything else raises TypeError. The constructor rejects step sizes with α_value·(6 + l2) + α_cost·8 ≥ 2 (the stability bound above; it never binds at sensible values).
- **Ties.** A trade must beat HOLD by more than 1e-14 (daily units). Trading to the point already held differs from HOLD by at most 4e-18 at the real scale, and real decisions by about 1e-8. Greedy choices use no random numbers.
- **Interface for `train.py` and `evaluate.py`.** The `spec.TCAgent` methods, plus read-only diagnostics: `theta`, `cost_linear` (c1), `cost_quadratic` (c2), `epsilon`, `n_updates`, `q_values(state)` and `holding_values(W)` (the implied G, for the divergence check at evaluation only).
- **Speed.** About 0.4 to 0.7 ms a step for the agent alone, measured before a cache of each state's target distances that saves about 0.09 ms; about 0.75 ms a block including Albert's `block_outcome` in the pilot, so 1 million blocks take about 12 minutes per seed.
- **Checked against independent code.** A reference written directly from this section steps side by side with the agent (off-grid states, HOLD and trades, l2 > 0, a frozen copy). Nine deliberately broken versions of the agent all fail the tests. A noise-free toy world with known costs, solved exactly by the old grid solver, checks learning: across five seeds the agent recovers both cost coefficients to within 11% and loses 0.1% to 0.2% of what never trading loses.

**Pilot on the real environment** (2026-11-15, $1B, tuning seed 20261112, my own loop in place of `train.py`, Albert's unmerged `env.block_outcome`, 500,000 blocks, ε decaying to e⁻¹⁰, one α for both blocks). The true coefficients are c1 = κ/B = 6.2e-6 and c2 = λ/(2B) = 1.13e-4.

| α | Greedy target at the optimum, last 3 checkpoints | c1, last 3 checkpoints | c2, last 3 checkpoints |
|---|---|---|---|
| 0.01 | (0.28, 0.58, 0.14), (0.40, 0.42, 0.18), (0.34, 0.52, 0.14) | 7.0e-5, 1.3e-4, 1.3e-4 | 1.3e-5, 3.4e-5, 3.3e-5 |
| 0.001 | (0.38, 0.26, 0.36), (0.38, 0.24, 0.38), (0.38, 0.24, 0.38) | 1.8e-5, 3.7e-5, 4.1e-5 | 6.0e-6, 7.7e-6, 1.3e-5 |

The optimum is (0.46, 0.14, 0.40). At α = 0.001 the value model is close to it, but neither cost coefficient is resolved: c1 runs 3 to 7 times too high, and c2 10 times too low. The independent review found the same at α_value = 0.01 and α_cost = 2e-4 over 30,000 blocks: every probe state's greedy action still changed in the last 20%, so the D12 rule would not be met. Its two diagnoses: the two cost features are 0.97 correlated under uniform exploration (eigenvalues of E[dd′] 0.52 and 0.0039), and the reward noise (about 1e-3 a block) is a hundred times c2 and a thousand times c1. This is §14 risk 1, now measured; the tuning design (§9) has to address it before any agent run.

## 7. Evaluation and reference policies (Albert)

All policies are scored from the same start holdings on the same simulated paths, drawn once from the `SeedPlan` evaluation stream.

**Metrics**, per path and averaged:
- discounted net reward;
- oracle gap in bps/yr (`spec.annualized_bps`, finite horizon), against the oracle's policy replayed on the same paths;
- hold fraction (a zero-size trade counts as a hold), trades per path, turnover per block and mean trade size;
- mean held weights against the static optimum, which Part V asks for.

- **Start holdings** (decided D4): drawn uniformly from the 2% grid, matching the main project's grid and keeping lucky off-grid starts out of the comparison. Also report the paths that start far from the optimum, where costs bite most.
- **Horizon and paths** (D10, approved by Albert 2026-11-13): H = 90 blocks (about 22 years; at ρ = 0.95 per block it leaves about 1% of the weight beyond the horizon) and 1,000 starting holdings. Every policy runs on the same paths, so the gaps are much less noisy than the values. Check the gap's standard error against the size of the effect being measured (pilot below). Paths of shape (H, m, B, 3) in float64 take 136 MB at H = 90 and m = 1,000, so they fit in memory; generate them block by block if m grows.

**D10 pilot** (`python -m src.tc_drift.d10_pilot`, tuning seed, 2026-11-13). Simple policies (always rebalance to the best 2% target, rebalance to it every 2, 4 or 8 blocks, trade to it once then hold, never trade) from the same 1,000 random 2% starts on the same 90-block paths, drift on:

| Gap, bps/yr (SE) | zero | $100M | $1B | $10B |
|---|---|---|---|---|
| rebalance minus every 2 | 1.16 (0.53) | 1.05 (0.53) | 0.90 (0.53) | 0.42 (0.52) |
| rebalance minus every 4 | 3.12 (0.86) | 2.93 (0.86) | 2.65 (0.86) | 1.79 (0.84) |
| rebalance minus every 8 | 5.21 (1.28) | 4.95 (1.28) | 4.57 (1.27) | 3.38 (1.25) |
| rebalance minus once | 23.5 (1.7) | 23.0 (1.7) | 22.2 (1.7) | 19.8 (1.7) |
| rebalance minus never | 140.9 (7.4) | 139.2 (7.4) | 135.9 (7.3) | 125.3 (7.3) |

- The value of one policy has an SE of 12.0 bps/yr. A paired gap's SE grows with how far apart the two policies' holdings get: 0.5 for every 2 blocks, 1.7 for trading once, 7.4 for never trading. These bracket the agent-against-oracle gap's SE but do not predict it.
- In rebalance minus once the starts drop out exactly (both policies trade to the same target in block 1), so its SE is pure path noise. In the never-trade gap about 86% of the variance is still path noise (within-start variance 50,928 against 8,440 between starts), because the drifting never-trade portfolio is scored on different returns from the rebalanced one.
- SEs fall as 1/√m: m = 4,000 halves them (paths 544 MB, so generate them block by block).
- The weight beyond the horizon is ρ^90 = 0.0099, and the paths take 136.1 MB, as proposed.
- Always rebalancing beats never rebalancing after the first trade by about 20 bps/yr even at $10B. Rebalancing every 2 blocks is within 0.4 ± 0.5 bps/yr of every block at $10B, so the pilot cannot say whether each single rebalance pays; the oracle sweep (§8) will.
- Design risk: if the oracle beats always-rebalance by only about 1 bp/yr, the agent's gap to the oracle and its gap to always-rebalance will be hard to tell apart at m = 1,000. Revisit m once the oracle exists.
- **Measured 2026-11-17 (§5, check 5):** the oracle's lead over always-rebalance is −0.17 ± 0.43 at $100M, 1.30 ± 0.70 at $1B and 7.10 ± 1.26 at $10B. At $100M and $1B the risk is real: the oracle and always-rebalance are within about two SEs of each other at m = 1,000.

**Implemented** (`src/tc_drift/evaluate.py`, 2026-11-17; tests in `src/tests/test_tc_drift_train_evaluate.py`): `evaluate_policy` and the four adapters as frozen, plus `best_static_target`, `draw_evaluation_set` (the pilot's draw order), `paired_gap`, `model_value` (finite-horizon value of a lattice policy in the oracle's model), `lattice_table` and `probe_convergence` (the original D12 rule only; the operative D12 is `tune.convergence` with probes, §6). `evaluate_policy` is checked against a separate share-and-price loop for four policies, and `model_value` against `value_of` and hand-computed one- and two-step values. The policy gets a copy of the holdings. A trade of zero size counts as a hold; turnover is one-way.

`src/tc_drift/train.py` (same date) follows §6 and William's note: the agent is updated with the true drifted end holdings even on the block before a restart, the first block and every `restart_every`-th start from a draw from the start states, and probes are read with `greedy_action` after the checkpoint block's update. It is checked against a separate replay that redraws the returns one block at a time and recomputes every (s, a, r, s′) with the share loop and the cost formula. Returns are drawn 1,024 blocks at a time, which gives the same numbers as one at a time (tested). `src/tc_drift/config.py` holds H, m, the oracle's M, 10 seeds, restarts every 90 blocks (the evaluation horizon; Albert's proposal), the training starts (the 2% grid, D4), the probe states (the 1,326 targets plus 200 off-grid lattice points from seed (20261112, 1)) and the seed layout. *2026-11-20:* N_BLOCKS (4M), the checkpoint interval (40,050, so the checkpoints match `tune.py`'s), the hyperparameters (§9) and m = 4,000 are set; `config.pending()` is empty. `probe_convergence` checks probes only at checkpoints, so a flip that reverts between two checkpoints is not seen; the baseline's rule tracked every change. Disclose it.

**Reference policies:**
- the oracle;
- always rebalance to the best no-cost target;
- trade once to it, then always hold;
- myopic (the oracle solved with ρ = 0 when choosing, scored with ρ = 0.95).

A policy is a function of (holdings, block index); only "trade once, then hold" uses the index (C10). `evaluate.agent_policy(agent)` and `evaluate.oracle_policy(solution)` adapt the agent and the oracle to this form (Albert).

**Run driver** (`src/tc_drift/run.py`, Albert, 2026-11-20; tests in `src/tests/test_tc_drift_run.py`). `python -m src.tc_drift.run all --workers 7` trains every level and seed that has no finished record (seed by seed, $1B first within each seed), one process per run with BLAS pinned to one thread, then scores each level. Each run writes `results/tc-drift/runs/<level>/seed<i>.npz` (the log with θ at every checkpoint) and `seed<i>.json` (settings, seeds, code hashes, the npz hash, speed). Scoring rebuilds each agent from its log, checks it against the logged greedy actions, and scores the tail-averaged policy, the last iterate and the own-target policy against the oracle, myopic, always-rebalance and trade-once-then-hold on the same m = 4,000 paths. It applies all four D12 versions and the D10 rule, and writes `summary.json`, `per_seed.csv` and `values.npz` per level. It refuses to mix seeds trained with different settings or code, and checks the oracle tables against the committed `oracle/summary.json`. Timed on Albert's Mac (10 cores) at the final settings on short runs (2026-11-20, not committed): 0.26 ms a block alone, 0.31 (exploring) to 0.39 (greedy) with seven in parallel. Projected from that: about 26 minutes a 4M-block run and about 2.6 hours of training for the 40 in six waves, plus scoring (not timed at m = 4,000 for every reference; the agent alone took about 27 s).

### Expected directions for the agent runs (D-F; registered by Albert 2026-11-20, before any final run)

**In plain terms:** what we expect the 40 final runs (4 cost levels × 10 seeds) to show, written down before they start, so each result can be marked met or not met. The short version: the agent will lose to the oracle by a few bps/yr at every level, mostly because it heads for a slightly wrong portfolio, and it will lose to plain rebalancing at $100M and $1B. Its cost-aware timing should show up only at $10B. Fewer than half the seeds will converge.

**Basis and disclosure.** These are preliminary and Albert's. They lean on the $1B tuning pilot (§9: 3 seeds on the tuning seed, α_value = 3e-4), including its benchmark diagnostics (oracle gaps, the favourite target's utility loss, the true c1 and c2), so they are informed by benchmark information about the same agent on other seeds. William may add to or amend them before the runs start. Anything added after the runs start is labelled post hoc. Units: bps/yr, paired over m = 4,000 paths. "Agent" is the tail-averaged policy (§6). Each expectation names the number in `results/tc-drift/runs/<level>/summary.json` that tests it.

| # | Expectation | Tested by |
|---|---|---|
| E1 | The agent loses to the oracle at every level: the across-seed mean of agent − oracle is between −1 and −8 bps/yr at each level, negative on at least 8 of 10 seeds. Pilot at $1B: −3.9, −2.8, −3.2 | `agent_minus_oracle_mean`, and per seed in `per_seed.csv` |
| E2 | The favourite-target error is most of that gap at zero cost, $100M and $1B: own target − always rebalance is negative and larger in size than agent − own target, in the across-seed means | the two `split_*` means |
| E3 | The agent loses to always rebalancing to the best target at $100M and $1B (across-seed mean below 0; pilot at $1B: −2.6, −1.5, −1.9), and does relatively better as costs grow: agent − always rebalance at $10B is above its value at $1B. Its sign at $10B is not predicted | `agent_minus_always_rebalance_mean` |
| E4 | The agent's own decisions add value as costs grow: agent − own target rises from $1B to $10B and is positive at $10B. At zero cost and $100M it is within ±1 (pilot at $1B: +0.6, −0.5, +0.7) | `split_agent_minus_own_target_mean` |
| E5 | The agent beats "trade once, then hold" at every level and on every seed (that policy is about 25 bps/yr behind the oracle, §8) | `agent_minus_trade_once_then_hold` per seed |
| E6 | The agent loses to the myopic oracle at every level: it knows μ and Σ, and the agent's favourite-target error alone exceeds myopic's shortfall (at most 2.6, §8) | `agent_minus_myopic_mean` |
| E7 | The agent's hold fraction varies widely across seeds (pilot at $1B: 12%, 55%, 16%). Its across-seed mean is higher at $10B than at zero cost. The $100M and $1B ordering is not predicted. At zero cost it holds more than the oracle does in the same replay (`references.oracle.hold_fraction`; 0.15 in check 5), because the D11 clamp keeps learned costs at zero or above while the true costs are zero | `agent_hold_fraction` |
| E8 | Costs, a benchmark diagnostic: c2 is learned better than c1. At $1B and $10B, c2 is within a factor of 2 of the truth on at least 7 of 10 seeds. c1 is off by more than a factor of 2 on at least 3 of 10 seeds at every level with costs. Pilot at $1B: c2/true 0.94, 0.49, 1.01; c1/true 0.41, 3.19, 0.00 | `c1_over_true`, `c2_over_true` |
| E9 | Convergence: under the operative 2026-11-20 rule, fewer than 5 of 10 seeds converge at each level (pilot: 1 of 3). Under the original 2026-11-12 rule and the 2026-11-19 exact-action rule on the last iterate, none do | `d12_converged_count` |
| E10 | The tail average and the last iterate differ by less than 1 bps/yr in the across-seed mean at each level, sign not predicted (pilot: −1.7, 0.0, +0.7) | `agent_minus_last_iterate_mean` |
| E11 | D10 (read "does not apply" from `d10.applicable`): at $1B the mean agent − oracle SE is 0.5 to 0.8 (the pilot's 1.1 to 1.5 at m = 1,000, halved), so it misses the target of a third of the oracle's lead (about 0.43). At $10B (target about 2.3) the rule is met. At zero cost and $100M it does not apply | `d10` |
| E12 | Seeds differ: the across-seed SD of agent − oracle at $1B is at least 0.5 bps/yr, so the across-seed spread, not the paired SE alone, sets how precisely the agent's gap is known | `sd_across_seeds` |

**What would change the story.** If E1 fails toward zero (the agent within 1 bp/yr of the oracle on most seeds at $1B), the agent learned more than the pilot suggested. If E4 holds at $10B but E2 also holds there, the agent learned when to trade but not where to head. If E3 shows the agent beating always-rebalance at $1B, the cost learning matters more than the pilot showed.

## 8. Cost levels and the oracle sweep (decided D9, 2026-11-12)

**In plain terms:** the costs are what trading these three ETFs costs in practice, for portfolios of different sizes. The spread is about 1 bp for all three. Market impact grows with the square root of the trade's size relative to the ETF's daily volume, so a bigger portfolio pays more per dollar.

**Calibration** (`src/tc_drift/calibration.py`, tested in `src/tests/test_tc_drift_calibration.py`):
- **Spread.** About 1 bp quoted for IEF (0.01%, iShares' 30-day median) and GLD (about 0.01%, a sponsor-linked source); SPY's is lower, about 0.003% to 0.01% (ETF.com, State Street). We charge 0.5 bp per dollar traded for all three, which is conservative for SPY. Commissions are taken as zero.
- **Impact.** The practitioners' square-root rule: cost per dollar = Y·σ_daily·√(trade / daily dollar volume), with Y = 1 (an assumption; a published example, 5% of daily volume costing about 45 bp at 2% volatility, implies Y ≈ 1).
- **Liquidity from our own data.** Mean daily dollar volume (Close × Volume) and daily volatility from `data/raw/prices`, over the 12 months ending 2026-07-31, the last day of the calibration sample (the risk-free data's lag sets that end; `DATA_CLEANING.md`): SPY $49.4B and 0.81%, IEF $0.81B and 0.29%, GLD $5.20B and 1.78%. Impact uses these recent volatilities; the simulator's Σ is full-sample, so the two differ (GLD's full-sample daily volatility is lower).
- **Fit.** κ and λ minimize the relative error of the model's cost, 2κq + λq² for a transfer q between two ETFs, against the square-root cost at q = 0.5%, 1%, 2%, 3%, 5%, 10%, 20% and 40%, averaged over the three pairs. More of the points are small because drift trades are a few percent. The fit is within 27% of the square-root cost at every one of those sizes (tested at 30%): it overcharges the smallest and largest transfers by up to 27% and undercharges 3% to 10% transfers by 15% to 23%. A fit through only the 20- and 40-point reference trades would overcharge drift-sized trades by about 1.6 to 2.5 times.
- **The same cost for every pair.** `spec.trade_cost` has one κ and one λ, so the fit averages the three pairs. GLD is the costliest to trade (impact per dollar for a drift-sized trade at $1B: GLD 13.0 bp, IEF 5.4, SPY 1.9), and drift is mostly SPY against GLD, so the model undercharges the typical drift trade by roughly a fifth. That pushes the policy toward trading more than it would with pair-specific costs; disclose it.

**The levels**, γ = 5:

| Level | κ (bp per unit of ‖u‖₁) | λ | 20-point transfer (bp) | 40-point transfer (bp) | Undoing one block's drift (bp) |
|---|---|---|---|---|---|
| Zero (control) | 0 | 0 | 0 | 0 | 0 |
| $100M | 1.63 | 4.4e-3 | 2.40 | 8.32 | 0.115 |
| $1B (headline) | 3.90 | 1.43e-2 | 7.27 | 25.97 | 0.302 |
| $10B | 11.04 | 4.57e-2 | 22.69 | 81.93 | 0.892 |

One block's drift from the optimum is a transfer of about 2.6% (§12). Holding the drifted portfolio for that block instead costs about 0.33 bp of utility (1.33 bps/yr for a quarter), and the loss grows with every further block of drift. So $100M should rebalance almost every block, while $1B sits near break-even (0.30 against 0.33) and $10B should hold more often. That is a one-block estimate until the oracle runs; a rough two-block check (rebalancing every block against every second block) also comes out near even at $1B (reviewer, not re-checked). `src/tests/test_tc_drift_calibration.py` pins this table to the code.

**What changed from the Phase 1 prompt, and why.** The prompt proposed choosing the cost level by the oracle's hold fraction, and a sweep of κ only, λ only, both and both high. Calibrating to practice replaces both: the costs come from data, and the economic question becomes "at what portfolio size do trading costs change the policy?". The κ-only and λ-only versions of the $1B cost (`calibration.mechanism_settings()`: the same 20-point cost, with c40 = 2·c20 and 4·c20) are run on the oracle alone, as an illustration of the two mechanisms; they add no agent runs. William runs them with the oracle. The 20- and 40-point reference trades are kept as reporting units.

**Agent runs** (§11 timeline): the four levels, 10 seeds each, γ = 5. Run the oracle on all levels before any agent run (done 2026-11-17, below).

**Expected directions for the oracle, written before any run:**
- **Zero cost.**
  - The oracle's value is within simulation error of always rebalancing to the best target, or above it; it cannot be below, by construction.
  - The oracle holds only when drift has landed on a lattice point at least as good as the best target. These comparisons turn on differences of about 0.1 bps/yr (the continuous optimum beats the best 2% target by 0.10), so sampling noise in the oracle will drive much of the zero-cost hold fraction. No direction is registered for its size.
  - Trading once then holding is worse than both.
  - Myopic equals the oracle within simulation error, since free trading makes the next state's value almost the same from everywhere.
- **Across $100M, $1B and $10B.**
  - The oracle's hold fraction rises and its turnover per block falls as the portfolio grows.
  - Its lead over always-rebalance grows with size.
  - Trading once then holding comes closer to the oracle as size grows.
  - No direction for the size of a single trade. A continuous model would trade part of the way, but on a 2% grid the partial move may round to a full one, and the oracle may wait for drift to build up before trading.
- **Myopic, at every level with costs.** It weighs one block of benefit against the whole cost, so it trades less than the oracle (a higher hold fraction) and is worth less.
- **The mechanism illustration at $1B.** The κ-only version holds more often than the λ-only version: a 2-point transfer costs 0.1·c20 under κ only and 0.01·c20 under λ only, at the same 20-point cost c20.

**Oracle results** (2026-11-17, `python -m src.tc_drift.oracle_runs`, M = 2,000, 1% lattice, γ = 5; post hoc). These are in the oracle's own model: values from `value_of` on the paths it was solved on, and trading statistics computed exactly by pushing the distribution of holdings through the policy and the transition matrix, from the 1,326 2% starts over 90 blocks. The oracle's in-sample values and so its leads are biased upward; the reported gaps will be Albert's replays on fresh evaluation paths. On a smaller check (M = 300) the reviewer found in-model turnover 4% to 5% higher than a continuous replay, from lattice rounding; not re-checked at M = 2,000.

| Setting | Hold fraction | Turnover per block | Trades in 90 blocks | Mean trade | Lead over always rebalancing | over trade once, then hold | over never trading | over myopic | Myopic's hold fraction |
|---|---|---|---|---|---|---|---|---|---|
| Zero cost | 0.156 | 2.95% | 75.9 | 3.50% | 0.008 | 24.83 | 142.35 | 0.000 | 0.156 |
| $100M | 0.272 | 2.26% | 65.5 | 3.10% | 0.214 | 24.55 | 140.83 | 0.014 | 0.291 |
| $1B | 0.308 | 2.13% | 62.3 | 3.07% | 1.319 | 24.86 | 138.50 | 0.243 | 0.388 |
| $10B | 0.443 | 1.57% | 50.2 | 2.82% | 6.939 | 27.96 | 133.25 | 2.586 | 0.599 |
| κ only ($1B) | 0.577 | 1.57% | 38.1 | 3.70% | 1.542 | 22.67 | 137.69 | 1.691 | 0.735 |
| λ only ($1B) | 0.259 | 2.21% | 66.6 | 2.99% | 1.488 | 25.69 | 138.95 | 0.129 | 0.248 |

Leads in bps/yr. Turnover is one-way, ‖t − s‖₁/2. Always rebalancing holds in 5.4% of blocks (when drift lands on the target) and turns over 3.03% a block. "Never trading" holds the start forever; Albert's `evaluate.never_trade_policy` is trade once, then hold. The best target is (0.46, 0.14, 0.40) at every level.

**Against the expected directions above:**
- **Zero cost: all three hold.** The oracle is 0.008 bps/yr above always rebalancing; trading once then holding is 24.8 below; myopic equals the oracle (same policy at every 2% start, lead 0.000). The 15.6% zero-cost hold fraction is the noise-driven kind §8 anticipated; the reviewer found the zero-cost oracle's favourite target at M = 300 is (0.48, 0.12, 0.40), not the best target, which is the same noise.
- **Across $100M, $1B and $10B:** the hold fraction rises (0.27, 0.31, 0.44), turnover falls (2.26%, 2.13%, 1.57%) and the lead over always rebalancing grows (0.21, 1.32, 6.94). **Trading once then holding does not come closer to the oracle:** its gap grows, 24.55, 24.86, 27.96. Registered direction not met. A breakdown by start (run once, not part of the script, so post hoc) points to the starts: from the 19 starts within L1 0.1 of the best target the gap shrinks as registered (24.4, 23.9, 22.7), and from the 247 starts between 0.1 and 0.4 away it shrinks too (24.4, 24.0, 23.4), but from the 1,060 starts 0.4 or more away it grows (24.6, 25.1, 29.1), because trading once makes one full trade, which costs those starts 1.5, 4.7 and 14.8 bps/yr on average, while the oracle moves part of the way. The registered reasoning missed the first trade.
- **Myopic, at every level with costs: met at the three portfolio sizes with costs, not met under λ only.** It holds more often than the oracle (0.29, 0.39, 0.60 against 0.27, 0.31, 0.44) and is worth less (0.01, 0.24, 2.59 bps/yr). Under the κ-only illustration it also holds more (0.74 against 0.58). Under λ only it holds less (0.248 against 0.259), though it is still worth less (0.13).
- **Mechanisms: holds.** κ only holds in 58% of blocks, λ only in 26%, and κ only trades bigger (3.70% against 2.99% per trade).
- **For D10.** The $1B lead over always rebalancing is 1.32 bps/yr in-model (an upper estimate). D10's rule wants the agent-against-oracle gap's SE below a third of the oracle-minus-always-rebalance gap, so below about 0.44 bps/yr. The D10 pilot's paired SEs at m = 1,000 were 0.5 to 1.7, and SEs fall as 1/√m, so m between about 1,300 and 15,000, depending on where the agent's gap falls in that range; Albert measures it with `evaluate.py`.

## 9. Hyperparameter tuning (William designs, Albert runs)

- **Seeds:** tuning uses master seed 20261112 (`spec.TUNING_MASTER_SEED`), never 20261016 or 20261103.
- **What is tuned:**
  - α per parameter block;
  - l2;
  - the target-refresh interval;
  - the greedy-refresh interval, if one is used.
- **Setting:** the $1B headline level only. The same values are used for every setting, and the report says so.
- **Criterion:** the final greedy policy's realized discounted net reward on fresh evaluation paths from the tuning seed. Never the oracle gap. The tuning seed's evaluation stream is the one the D10 pilot and check 5 used; that is not leakage, since the agent never sees those paths.
- **Choosing rule, fixed before running:**
  1. The highest mean over 3 tuning seeds wins.
  2. If the runner-up is within one paired standard error, add 3 seeds and decide on 6. This is the rule `src/tune_fa.py` used.
- **Candidate grid:** William, Phase 2. Not fixed yet: the pilot (§6) shows that one constant α for both blocks cannot resolve the cost coefficients at the real noise level. Options to decide before tuning, none tried yet:
  - separate, smaller α_cost (the two blocks already have separate step sizes), with a much longer budget;
  - decaying step sizes, α_t ∝ 1/t from some point on, which average the noise away (the assignment allows them, and the tabular baseline used 1/N);
  - decorrelating the two cost features, for example ‖u‖₁ and ‖u‖₂² − b‖u‖₁ for a fixed b, which changes only how they are parameterized, not what the agent can represent;
  - a longer training budget, at about 12 minutes per million blocks per seed.

  Then pilot the D12 convergence rule on the tuning seed before fixing the grid.

  **Decided 2026-11-19 (William), after the cost-block analysis below and before any tuning run:**
  - **Rule:** recursive least squares on the cost block (c1, c2), started after a burn-in at constant α_cost; θ keeps a constant α_value. The fit starts at the switch from the costs the burn-in ended with, under a small prior (1e-3·I, decided William 2026-11-19): the burn-in's constant-step costs were 3 to 10 times off in the §6 pilot, so they get almost no weight, at the price of noisy costs for roughly the first 15,000 trades after the switch (an estimate from the synthetic regression, to be read off the pilot's checkpoints). The D11 clamp is applied when the costs are read, not inside the recursion (a review on 2026-11-19 found the clamp inside the recursion biases the costs upward). The burn-in length used in the pilot is listed below and is not yet confirmed. The choice was informed by benchmark information (the true c1 and c2, and the pilot's target error computed from μ and Σ); disclose it with §14 risk 5.
  - **Budget:** 4 million blocks per seed, with ε_t = e^{−βt} reaching 1% at the halfway point: β = ln 100 / 2,000,000 ≈ 2.30e-6, about 434,000 exploring blocks.
  - **Candidate grid:** α_value ∈ {3e-4, 1e-3}, 3 tuning seeds, chosen by the rule above.
  - **Chosen by Claude for the pilot, confirmed by William 2026-11-20 for the final runs:** the burn-in ends when ε first reaches 0.5 (cost_rls_after = round(ln 2 / β) = 301,030 updates); α_cost = α_value during the burn-in; target_refresh_every = 1,000; l2 = 0.

**Tuning pilot** (2026-11-19, `python -m src.tc_drift.tune`, $1B, tuning seed 20261112, seeds 0 to 2, 4 million blocks each, about 2.4 hours a run with six in parallel and BLAS pinned to one thread; outputs in `results/tc-drift/tuning/`, code hashes in each run's record; the runs used uncommitted code on `feature/tc-drift-rls`). A first set of six runs used an earlier version that clamped the costs inside the RLS recursion; it was discarded before its results were used and is not reported (its c1 came out 4 to 15 times too high, as the review predicted). The tail-averaged policy (§6, changed after this pilot) was applied to the six runs post hoc with `tune rescore`, which first checks that the rebuilt agent reproduces the logged greedy action on every probe state.

| α_value | Seed | Criterion, tail average (bps/yr) | Last iterate | Favourite target (tail average) | c1 / true | c2 / true |
|---|---|---|---|---|---|---|
| 3e-4 | 0 | 504.8 | 506.5 | (0.40, 0.20, 0.40) | 0.41 | 0.94 |
| 3e-4 | 1 | 505.9 | 505.8 | (0.44, 0.20, 0.36) | 3.19 | 0.49 |
| 3e-4 | 2 | 505.5 | 504.8 | (0.44, 0.22, 0.34) | 0.00 | 1.01 |
| 1e-3 | 0 | 503.8 | 506.9 | (0.42, 0.24, 0.34) | 2.24 | 0.97 |
| 1e-3 | 1 | 484.1 | 479.5 | (0.36, 0.36, 0.28) | 3.91 | 0.68 |
| 1e-3 | 2 | 500.4 | 497.7 | (0.44, 0.26, 0.30) | 3.05 | 0.99 |

- **Choice: α_value = 3e-4**, on 3 seeds: mean 505.4 against 496.1, paired difference 9.31 against a paired SE of 6.32. The last-iterate criterion picks the same (10.97 against 7.96).
- **Diagnostics, not used to choose** (bps/yr, paired over the 1,000 evaluation paths, ± one SE; the optimum target is (0.46, 0.14, 0.40)). For α_value = 3e-4, the agent minus the oracle is −3.9 ± 1.5, −2.8 ± 1.1 and −3.2 ± 1.3; the agent minus always rebalancing to the optimum target is −2.6 ± 1.4, −1.5 ± 1.3 and −1.9 ± 1.5, against the oracle's lead of 1.3 ± 0.7. The favourite target's utility loss is 5.6, 2.3 and 4.1. The agent beats rebalancing to its own favourite target by +0.6, −0.5 and +0.7. Tail averaging changed the criterion by −1.7 ± 0.4, +0.0 ± 0.5 and +0.7 ± 0.4.
- **Costs.** c2 is learned to within 6% on two seeds of three and at half its size on the third. c1 ranges from 0 to 3.2 times its true size, and the seed with c1 high has c2 low. On a typical one-block drift trade (‖u‖₁ = 0.052, ‖u‖₂² = 0.00135) the true c1 term is 3.2e-7 and the c2 term 1.5e-7, so c1, the worse-learned coefficient, is the larger part of the cost of small trades. c2 settles by 1 million blocks; c1 still moves after that (seed 0: 0.91, 0.31, 0.29 and 0.41 of its true size at 1, 2, 3 and 4 million blocks).
- **Convergence (§6), tail average, α_value = 3e-4.** Primary: passes on seeds 0 and 2 (the favourite target moves by at most L1 0.04 over the window) and fails on seed 1 (0.08). Split-half: passes on all three (the two halves' favourite targets are one 2% step apart). Secondary: fails on all three (3% to 25% of probes never change their action; the split-half policies agree on 24% to 36% of probes). Original: fails on all three. The last iterate fails all three rules on every run.
- **What the probe rule is measuring** (a check on the split-half policies, α_value = 3e-4): the hold-or-trade decision agrees on 99.5% to 99.8% of probes, which says little because only 0.1% to 0.7% of probes hold; the two halves' post-decision holdings are within one 2% step of each other at 85% to 91% of probes, and within two steps at over 99%. The exact-action rule treats a move to the next 2% target as a change, among 1,326 targets with nearly equal values. The secondary rule was restated on that distance on 2026-11-20 (§6).
- **Convergence under the restated rule (§6, 2026-11-20), tail average.** α_value = 3e-4: **seed 0 converges** (favourite within 0.04, 97.4% of probes within one step), from block 3,284,100 (36,490 episodes, 206.9 million daily observations); seed 1 fails both rules (favourite 0.08, 70.7% of probes); seed 2 passes the primary and fails the secondary (66.9%). So 1 of 3 tuning runs converges. α_value = 1e-3: none of 3 (seeds 0, 1, 2: 31.7%, 93.3% and 0.3% of probes; seed 1 passes the primary and misses the secondary). The last iterate converges on no run (at most 54.2% of probes). The window rule is stricter than the split-half check: it compares each of 12 checkpoints with the final one.
- **The hold-or-trade decision near the no-trade region is not stable** (diagnostic, not a rule). At α_value = 3e-4, 3, 16 and 4 probes hold at some checkpoint of the window (tail average), and the share of them whose hold-or-trade decision never changes is 0%, 38% and 0%. Seed 0, the run that converges, holds at 1 probe at the end. The hold fraction on evaluation paths (12%, 55%, 16%) also differs a lot across seeds. The report should say that convergence under D12 is about where the agent trades to, not when it holds.

**The cost block in isolation** (2026-11-17, `python -m src.tc_drift.cost_learning_analysis`, seed 20261117; synthetic, not a pilot of the agent). The cost part of the agent is a two-coefficient regression: the net reward falls by c1‖u‖₁ + c2‖u‖₂² for a trade u. The script runs only that regression with SGD, on trades drawn uniformly over the targets from holdings near the γ = 5 optimum, with noise of SD 1e-3 per block (the block reward's). It leaves out θ and the bootstrapped target, so the agent is expected to do worse.
- The two features are 0.97 correlated; E[dd′] has eigenvalues 0.67 and 0.0068, a ratio of 98. A step size small enough for the large direction moves the small one 98 times slower.
- **The floor.** No unbiased linear estimator beats least squares on the same trades (Gauss–Markov); a biased one, like the decaying row below or the D11 clamp, can have a smaller SD. After 500,000 exploring trades its SEs are 6.4e-6 for c1 (the size of c1, 6.2e-6) and 1.6e-5 for c2 (14% of 1.13e-4). The cost of a typical one-block drift trade (‖u‖₁ = 0.052, ‖u‖₂² = 0.00135; 4.7e-7 a day) has SE 3.1e-7, two thirds of itself. SEs fall as 1/√N: 1.6e-7 at 2 million trades, 7.0e-8 at 10 million. With ε_t = e^{−βt} the agent explores only about 1/β blocks in all (50,000 in the §6 pilot), so it is far from even 500,000.
- **Six step rules**, 200 replications of 500,000 updates (true values c1 6.18e-6, c2 1.13e-4, drift-trade cost 4.75e-7):

| Rule | c1 mean (SD) | c2 mean (SD) | Drift-trade cost mean (SD) |
|---|---|---|---|
| Constant α = 1e-3 | 5.7e-6 (2.2e-5) | 1.12e-4 (2.4e-5) | 4.5e-7 (1.2e-6) |
| Constant α = 1e-3, decorrelated | 4.9e-6 (8.5e-5) | 1.20e-4 (2.1e-4) | 4.2e-7 (4.1e-6) |
| Decaying, α_t = 1/(E₁₁(t + 100)) | 4.0e-5 (5.1e-6) | 2.6e-5 (1.3e-5) | 2.1e-6 (2.5e-7) |
| Decaying, decorrelated (b = 0.39, from holdings near the optimum) | 5.8e-6 (6.7e-6) | 1.14e-4 (1.7e-5) | 4.5e-7 (3.3e-7) |
| Decaying, decorrelated (b = 0.49, from holdings uniform on the lattice) | 1.8e-5 (1.6e-5) | 7.9e-5 (4.6e-5) | 1.0e-6 (7.6e-7) |
| Recursive least squares | 5.7e-6 (5.9e-6) | 1.15e-4 (1.5e-5) | 4.5e-7 (2.9e-7) |

- **Decorrelated** means the second feature is replaced by k(‖u‖₂² − b‖u‖₁), with b = E[d₁d₂]/E[d₁²] and k chosen so both features have the same second moment. The agent can represent exactly the same costs (c1 = w₁ − bkw₂, c2 = kw₂), so the D11 clamp has to be applied to c1 and c2, not to w.
- A constant α is unbiased but its noise scales with √α: the drift-trade cost's SD is 2.4 times the cost. Decorrelating alone makes the slow direction fast, and at a constant α that amplifies its noise too. Decaying alone averages the noise away but stops before the slow direction has converged, so it is biased (c1 6.5 times too high, c2 4 times too low, close to the §6 pilot's pattern). Both together reach the least-squares floor, but only with b = 0.39, which comes from holdings near the optimum: benchmark information the agent may not have.
- **Choosing b without the benchmark.** With b = 0.49, from the action geometry alone, the eigenvalue ratio falls from 98 to 4.7 rather than to 1, and the drift-trade cost comes out 2.2 times too high (SD 1.6 times the cost) after 500,000 trades. A fixed b only works if the holdings the agent visits are known in advance.
- **Recursive least squares needs no b.** It keeps a running sum of dd′ over the agent's own trades (started at 1e-3·I) and steps each update by that matrix's inverse. That is the decorrelation and the 1/t decay at once, learned from the agent's own data, so it stays model-free. It recovers both coefficients without bias and with SDs at the floor (c1 5.7e-6, SD 5.9e-6; c2 1.15e-4, SD 1.5e-5). Its cost is a 2×2 solve per update. Even so, the floor leaves the drift-trade cost uncertain by about two thirds of itself after 500,000 exploring trades, so the agent's exploration budget, not the step rule, is the binding constraint after this fix.
- **Not shown here:** in the agent, θ is learned at the same time and the targets are bootstrapped, so an early decay (or recursive least squares from the first update) would average over targets that are still wrong. A burn-in at constant α before the decay, or a forgetting factor in recursive least squares, addresses that, and needs the pilot.

## 10. Roles, files and frozen interfaces

**Frozen now, changed only by a PR both approve:** `src/tc_drift/spec.py`.
- Constants, as in §2.
- `TCSettings(kappa, lam, gamma)` and `Market(mu, sigma)`.
- `target_grid()`, `lattice()`, `nearest_lattice_index()`, `target_lattice_index()`.
- `post_trade_weights()`, `trade_cost()`, `coefficients_from_reference_trades()`.
- `net_reward()` and `annualized_bps()`.
- The `TCAgent` protocol.

**Stubs (signatures frozen by `test_signatures_are_frozen`):**

| File | Owner | Signatures |
|---|---|---|
| `src/tc_drift/env.py` | Albert | `load_market(processed_dir)`, `draw_returns(market, rng, m, block_length)`, `block_outcome(w0, excess_returns, gamma, drift)`, `simulate_block(w0, rng, B, drift, *, market, gamma)` |
| `src/tc_drift/train.py` | Albert | `train(agent, market, settings, *, n_blocks, returns_rng, start_rng, start_states, restart_every, checkpoint_every, probe_states, drift)` |
| `src/tc_drift/evaluate.py` | Albert | `evaluate_policy(policy, starts, excess_paths, settings, *, discount, drift)`, `agent_policy(agent)`, `oracle_policy(solution)`, `always_rebalance_policy(best_target)`, `never_trade_policy(first_target)` |
| `src/tc_drift/oracle.py` | William | `solve(market, settings, excess_paths, *, block_outcome, discount, drift, lattice_increment)` returning `OracleSolution` (V, G, policy table, r̂, correction, next indices) with `policy(state)` and `value_of(policy_table, discount)` |
| `src/tc_drift/agent.py` | William | `DriftAgent(rng, **hyperparameters)`, keywords limited to `DriftAgent.HYPERPARAMETERS`, with `select_action(state)`, `update(state, action, reward_net, next_state)`, `greedy_action(state)`. It reads the targets from `spec.target_grid()`, so a reordered copy cannot remap actions |
| `src/tc_drift/features.py` | William | `spline_features(W, n_basis, degree)` |

`src/tc_drift/calibration.py` (shared, written 2026-11-12) gives the four cost levels as `TCSettings`; the agent may not import it. A config module or JSON for the runs belongs to Albert (C5).

**Leakage, enforced by `src/tests/test_tc_drift_spec.py` and `test_environment.py`.** The check is an allowlist, so a new route fails by default. `agent.py` and `features.py` may import only:
- `numpy` and a few standard modules;
- the names in `spec.AGENT_SAFE_NAMES` (actions, grids, lattice; not the costs, the settings, the market or `GAMMA_HEADLINE`);
- `src.features` and `src.tc_drift.features`.

They may not use star imports, `importlib`, `__import__`, `open`, `eval`, `np.load` or similar file readers, `__dict__`, or `getattr` with a computed name. Their import graph may load no `src` module outside the allowlist. No agent method or hyperparameter may be named for μ, Σ, κ, λ, γ, the market or the oracle. The existing scan also forbids data paths and the benchmark modules. Each banned route is shown to be caught on an example snippet. These tests catch accidental leaks; a deliberate one would get past them, and code review covers that.

**Working without blocking:**
- Albert tests `train.py` with trivial stub agents (random, always hold, always rebalance) that implement `TCAgent`.
- William tests the oracle and the agent with a small loop simulator inside `src/tests/` until `env.py` lands.
- A cross-test checks that the two simulators agree on the same random numbers; whoever merges second owns it.
- Each works on their own branch and never edits the other's files without a PR.
- `spec.py` merges first.

## 11. Timeline and gate (from the Phase 1 prompt)

| Date | Work |
|---|---|
| Nov 13 | This document and `spec.py` merged, interface frozen |
| Nov 13 to 16 | Environment (Albert), oracle (William) |
| Nov 16 to 18 | Agent (William); train, evaluate and config (Albert); the oracle on all four cost levels plus the $1B κ-only and λ-only illustrations (William, by Nov 18) |
| Nov 18 to 19 | Integration smoke runs, timing pilot, then tuning on separate seeds |
| Nov 20 to 26 | Full runs; Thanksgiving buffer |
| Nov 26 to 30 | Analysis and the report |

*Status 2026-11-20:* on schedule to Nov 19, apart from tuning, which finished on Nov 20 (William). The run driver is ready (PR #41) and the 40 runs take about 3 hours, so the Nov 24 go/no-go can be met.

**Go/no-go for the agent runs, Nov 24 (William, 2026-11-19).** If the learning rule is implemented and tuned by Nov 24, run the agent at the four levels; otherwise report the oracle results and a diagnosed null agent result (prospectus §7).

**Gate, Nov 17.** The environment plus oracle checks 1 and 2 must pass. If not, fall back to the no-drift design with "learn each target's reward, then plan" (`AUDIT_2026-11-06.md` §6.2, T1), plus the negative result. The prospectus (week 9: "first full report draft") assumes a draft by Nov 19; this timeline moves TC results past that (C12).

## 12. Numbers verified for this design

Reproduced by `python -m src.tc_drift.design_numbers`, seed 20261113, 20,000 paths, γ = 5, B = 63.

| Quantity | Value |
|---|---|
| Nine-asset long-only optimum (`constrained_optimal_weights`) | γ = 3: SPY 0.5502, GLD 0.4498. γ = 5: SPY 0.4688, IEF 0.1363, GLD 0.3948. γ = 8: SPY 0.3562, IEF 0.3899, GLD 0.2539 |
| Three-asset optimum, γ = 5 | (0.4688, 0.1363, 0.3949), the same as the nine-asset one to within 1e-4 |
| Block reward SD at that optimum | constant weights 2,385 bps/yr; buy-and-hold 2,390; their difference on the same paths 103 |
| Expected utility lost by letting the optimum drift | 1.33 bps/yr after 1 block, 5.40 after 4, 25.9 after 20 |
| Continuous optimum minus the best 2% target | 0.10 bps/yr |
| Control-variate correction at that optimum | −0.62 ± 0.73 bps/yr (mean ± SE) |
| Drift over one block (one-way turnover back to the start) | median 2.26%, mean 2.62% |
| Effect of including the risk-free rate in drift | largest weight difference after a block: median 1.5e-6, max 7.1e-6 (mean daily rf 6.8e-5) |
| Three-asset spline map on the 2% grid | 87 columns, rank 45, max ‖φ‖² = 6.0 |
| Counts | 1,326 targets; 5,151 lattice points (1%); 20,301 (0.5%); every target on the 1% lattice |

**Not verified:**
- oracle run time on the 0.5% lattice (the 1% lattice is measured in §5);
- the agent's speed is machine-dependent: on Albert's Mac at the final settings (RLS cost block, BLAS pinned) 0.26 ms a block alone and 0.31 to 0.39 with seven in parallel (2026-11-20); on William's machine 2.4 hours a 4M-block run with six in parallel (about 2.2 ms a block; §9). The earlier 0.57 ms (2026-11-17, Albert's Mac) was measured on older code without BLAS pinned; the cause of the difference was not isolated.

## 13. Conflicts with the repo, and questions

**C1. How to describe the universe (assignment Part I and Replication Code).** The assignment calls SPY, IEF and GLD "the original three ETFs" (Part I) and requires code that "constructs the action space without using benchmark information".
- *Decided (William, 2026-11-12):* describe the universe as the three ETFs the nine-asset optimum holds, chosen because three assets keep the oracle computable.
- *Caveat to state in the report:* restricting the agent to the assets in the optimum's support uses benchmark information. Within the extension the agent never sees which portfolio is best, but it can no longer err by holding the six assets the optimum leaves out, so the extension's learning gap is smaller than it would be on nine assets. The assignment asks the report to discuss "discretization error and possible benchmark leakage" (Part VI), so this belongs there. That they are also the assignment's original three ETFs can be mentioned as well.

**C2. Total versus excess returns for drift.** `ENVIRONMENT.md` §4 says a drift layer "needs total returns (excess plus the risk-free rate)". `CLAUDE.md` §5 says the environment is defined over excess returns. Measured over one block, adding the sample-mean risk-free rate moves the end weights by at most 7.1e-6 (§12).
- *Decided (2026-11-12):* drift on excess returns. Both options are defensible: the assignment defines the environment as an excess-return process (Part I), and the omitted term moves end weights by at most 7.1e-6 per block. Excess returns are also the simpler one, with no new input. The report states the assumption and the bound. `ENVIRONMENT.md` §4's line now points here.

**C3. `simulate_block(w0, rng, B, drift)` as proposed cannot work.** It has no μ, Σ or γ, so it cannot compute the reward. It also draws new returns on every call, so the oracle could not score 5,151 starting points on the same fixed paths.
- *Decided (2026-11-12), implemented in the stubs:* `draw_returns(market, rng, m, B)` and `block_outcome(w0, excess_returns, gamma, drift)`, with `simulate_block(..., *, market, gamma)` kept as a wrapper. The oracle receives its fixed paths and the scoring function as arguments, so neither person waits for the other.
- One `block_outcome` then serves the environment, the oracle and the evaluation, which also makes the cross-test trivial. It is a new module, so nothing already built changes: `src/environment.py` and every baseline, FA and old TC run are untouched.

**C4. `src/features.py` accepts only nine assets.** `_check_weights` enforces nine columns, and the file is fingerprinted in every FA training manifest (`src/train.py` `FA_CODE_FILES`). Editing it would flag every FA run, including Albert's grid-10 runs, as "training code changed".
- *Decision taken:* a new file, `src/tc_drift/features.py`, reusing `bspline_basis` by import.

**C5. `src/config.py` cannot host this family.** `RunVariant` and `Experiment` describe stateless runs on the nine-asset grid. `config.py` is also fingerprinted in evaluation provenance (`src/evaluate.py` `EVAL_CODE_FILES`), so adding to it changes every run's evaluation code hash.
- *Recommendation:* a separate `src/tc_drift/config.py` (Albert), or JSON under `experiments/tc-drift/`, as the old TC design did.

**C6. Oracle check 2 as worded is not exact.** "κ = λ = 0 equals the best portfolio held forever" fails in two ways:
- HOLD keeps an off-grid lattice start, which can beat every target;
- with drift, "held forever" is ambiguous between buy-and-hold and rebalancing every block.

§5 restates check 2 as an exact equality without drift and a provable inequality with drift.

**C7. HOLD makes weights off the 2% grid reachable.** The Part II "best portfolio in the agent's discrete action set" is no longer a fixed set, because the reachable weights depend on the start and on drift. The extension's discrete benchmark is the oracle on the same action set. The static continuous optimum is still reported.
- *Open (D4):* draw evaluation starts from the 2% grid, from the 1% lattice, or uniformly on the triangle? Grid starts keep "off-grid luck" out of the comparison.

**C8. Two value models for one function.** θ_f·φ(target) and θ_h·ψ(s) both estimate G (§3), at different points. Separate parameters mean HOLD learns only from HOLD steps.
- *Decided (D5, 2026-11-12):* share them (one θ, one φ). This halves the parameters, lets every step teach both, and matches the exact structure. It still keeps the learned cost terms, so the agent is no less model-free.

**C9. HOLD is 1 of 1,327 actions.** Uniform exploration would try it on 0.08% of exploratory steps.
- *Decided (D6, 2026-11-12):* HOLD is one of the 1,327 actions an exploratory step draws uniformly. With the shared value model (D5) that is enough (§6).

**C10. "Trade once, then hold" needs the block index.** In the stubs a policy takes (holdings, block index); stationary policies ignore the index.

**C11. The myopic reference needs the oracle with ρ = 0.** In the stub, `solve` takes a `discount` argument.

**C12. The prospectus is out of date for TC** (William's file; I have not edited it). Rows that would change after approval:
- §4: "Transaction-cost extension | Albert built it; redesign paired" becomes this split;
- §5, week 9: the draft date against the Nov 20 to 26 TC runs.

**C13. Two documents still describe the old options as current.** `TRANSACTION_COSTS.md` and `AUDIT_2026-11-06.md` §6.2 were not touched, by instruction. *Updated 2026-11-20:* the audit's §6.2 now carries the pointer (2026-11-19); the one in `TRANSACTION_COSTS.md` was added by Albert on 2026-11-20. Closed.
- *Open (D7):* add a one-line pointer to this document in each after approval?

**C14. A one-line edit in Albert's test file.** `src/tests/test_environment.py` `AGENT_MODULES` gains `src.tc_drift.agent` and `src.tc_drift.features`.

**C15. The discount needs a justification in the report.** ρ = 0.95 per 63-day block is about 0.81 per year: a reward 13.5 blocks (3.4 years) ahead counts half. The assignment asks to "explain its role". The continuation now differs by action, which is the point of the extension.
- *Open (D8):* is 0.95 per block the intended value, or was 0.95 per year meant? That would be about 0.987 per block.

**Decisions needed before Phase 2:**
- **D1 (C1). Decided:** the three ETFs the nine-asset optimum holds, chosen for computational convenience; the benchmark-leakage caveat goes in the report.
- **D2 (C2). Decided:** excess returns.
- **D3 (C3). Decided:** split interface.
- **D4 (C7). Decided:** starts drawn uniformly from the 2% grid; far-from-optimum starts also reported.
- **D5 (C8). Decided:** one shared value model.
- **D6 (C9). Decided:** uniform over all 1,327 actions.
- **D7 (C13). Decided:** a one-line pointer to this document in each, added when this design merges into `dev`.
- **D8 (C15). Decided:** 0.95 per block, described in the report as a 5-year investor (half the weight at 3.4 years). 0.95 per year would need about 350-block evaluation paths and noisier learning.
- **D9 (§8). Decided:** realistic costs at four portfolio sizes (zero, $100M, $1B headline, $10B), from spreads and square-root impact on our own volume data, fitted across trade sizes.
- **D10 (§7). Decided (Albert, 2026-11-13):** H = 90 blocks and 1,000 starting holdings, as proposed. The §7 pilot gives paired-gap SEs from 0.5 to 7.4 bps/yr depending on how far apart the two policies' holdings get. Once the oracle exists, measure the oracle-minus-always-rebalance gap and the agent-against-oracle gap's SE, and raise m until that SE is below a third of the oracle-minus-always-rebalance gap (m = 4,000 halves every SE). *Measured 2026-11-17 (§5):* the lead is 1.30 ± 0.70 bps/yr at $1B and 7.10 ± 1.26 at $10B, so at $1B the rule asks for an agent-gap SE of about 0.43 or less. Albert decides m once the agent's gap SE is known; m = 4,000 is the likely answer.
- **D11 (§6). Decided:** θ₁ and θ₂ are kept at zero or above.
- **D12 (§6). Decided:** the rule mirrors the baseline's `greedy_stable_through_end`, applied to the probe states (§6). It may be revised after the tuning pilot, before any final results are examined, and labelled as a change (William, 2026-11-17). **Changed 2026-11-19 (William), before the tuning pilot:** favourite target within one 2% step over the final 10% (primary), 95% of probes stable (secondary), original verdict reported alongside (§6). **Changed again 2026-11-19 (William), after the tuning pilot and after seeing its benchmark diagnostics, before any final run:** the reported policy is greedy on the tail-averaged θ (mean of the checkpoint snapshots over the final 10%), and the rules are applied to its trailing averages; the last iterate's verdicts are reported alongside (§6, §9). **Secondary restated 2026-11-20 (William), labelled:** 95% of probes with post-decision holdings within one 2% step of the final checkpoint's throughout the window; a run has converged when the primary and this hold; Part IV's converged-from block is reported (§6). **Frozen 2026-11-20:** no further changes; the final runs report all four versions' verdicts.

C4, C10, C11 and C14 are already settled in the stubs; C5 and C12 need only the edits described.

## 14. Risks

1. **The agent may not resolve the decisions.** One block's reward has a standard deviation of 2,390 bps/yr, while one block of drift from the optimum costs 1.3 bps/yr. To resolve that loss at one standard error the agent would need about (2,390/1.3)², roughly 3 million, blocks spent on the comparison. The oracle sweep should pick costs where decisions matter, and the agent's expected directions are registered only after tuning. A null agent result against a working oracle is still a reportable finding. **Measured 2026-11-15 (§6 pilot):** after 500,000 blocks at constant α the cost coefficients are still off by factors of 3 to 10.
2. **Divergence (§6).** Checked against the oracle's G after training.
3. **The oracle is approximate.** Rounding and finite paths are bounded by checks 3 to 5, and its in-sample V is biased upward. Gaps use its replayed value, and a policy on continuous holdings can beat it slightly, so small negative gaps are possible.
4. **Compute and the gate.** Oracle time is unmeasured, and the 0.5% lattice check is four times the work. If checks 1 and 2 fail by Nov 17, fall back (§11).
5. **Disclosure.** The universe framing (C1), cost levels chosen with the oracle (§8), tuning on simulated data (as in `AUDIT_2026-11-06.md` M12), and post-hoc labels on everything. *Added 2026-11-20 (`AUDIT_2026-11-06.md` TC-13):* benchmark information shaped the agent's design, though never its inputs or the tuning criterion. The learning rule (§9) was chosen with the true c1 and c2 (from κ and λ) and the target error (from μ and Σ) in view; the tail-averaged policy (§6, 2026-11-19) was adopted after the tuning pilot's oracle gaps and utility losses had been seen; the restated secondary rule (§6, 2026-11-20) after its split-half distances had been seen. All of it is on the tuning seed, before any final run.
6. **Maximization bias** in the bootstrapped target, from noisy Q̂. Measured against the oracle.

## 15. Decisions not taken

- **Nine assets with drift.** Discussed 2026-11-11. The holdings would be an 8-dimensional continuous state, so an exact oracle is out of reach.
- **Pure quadratic cost without HOLD.** Drift would be economically immaterial: re-centering a 2% drift costs almost nothing under a quadratic cost.
- **A tabular agent over (lattice state, action).** About 6.8 million pairs, the same data-starvation that sank the first design (H1).
- **Giving the agent the cost formula (afterstate learning with known costs).** Accurate and simpler, but the agent would know κ and λ. Rejected so the agent stays model-free in costs too.

## 16. Review (2026-11-12)

An independent reviewer checked the math, `spec.py`, the tests and every number. It found no error in the Bellman equation, the post-decision split, the control variate, the annualization, the reference-trade algebra or the lattice rounding, and every §12 number reproduced. Applied from its findings:
- **Leakage.** γ was reachable through `spec`. The name-based scan missed routes such as `importlib`, `getattr` with a computed name, files read from `results/`, and `**hyperparameters`. Now an allowlist, with `spec.AGENT_SAFE_NAMES` and a fixed hyperparameter list (§10).
- **Convergence.** No rule was defined; D12 proposes one, and `train()` now records probe-state greedy actions.
- **Expected directions.** Three did not survive a 2% grid with 63-day blocks: κ-only trade size, the "both" trade-size ordering, and the λ-only partial first move. They are replaced or dropped (§8).
- **Oracle wording and interface.** "Exact" overclaimed. `solve` now takes the fixed paths and the scoring function, and returns the fields the checks need. Check 1 is spelled out, and check 5 states the direction of the bias.
- **Ties, random streams, adapters, the stability bound** (14, not 6, once the cost features are included), signal sizes, and several small validations in `spec.py`. Rounding is now vectorized: one million holdings in 0.3 s.

Not applied: none. The two items it left open, the convergence rule (D12) and the cost levels (D9), were decided the same day; §8 was rewritten for D9 after this review.
