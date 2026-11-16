# Transaction Costs with Drift: Design (Phase 1)

**Owners:** William (oracle, agent, features, tuning design) and Albert (environment, training, evaluation, config, runs) · **Branch:** `feature/tc-redesign` · **Status (2026-11-13):** Phase 1, decisions recorded (§13); Phase 2 has started. This document, the frozen interface `src/tc_drift/spec.py`, signature-only stubs and contract tests. `env.py` is implemented and tested on `feature/tc-drift-env`, and D10 is decided with a pilot (§7); the other modules are still stubs. One independent review is done and its findings are applied (§16). Waiting for William's and Albert's review. **Post hoc** relative to `baseline-v1`; label every result that way.

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
  - `TRANSACTION_COSTS.md` and `AUDIT_2026-11-06.md`.
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

**Fixed paths.** M blocks of returns are drawn once from `np.random.default_rng(spec.ORACLE_PATH_SEED)` and passed to `solve` as `excess_paths`, and every lattice point is scored on the same M paths (common random numbers). `solve` also takes the block-scoring function, so William can test with his own reference loop before Albert's `env.block_outcome` lands. The same generator seed is reused for every cost setting, so differences between settings are not sampling noise. M is set by check 3; it is not chosen yet.

**Control variate.** r̂(x) = U(x) + mean_m[R_bh(x; m) − R_const(x; m)]:
- R_bh is the buy-and-hold block reward on path m;
- R_const is the constant-weight reward on the same path, whose mean is exactly U(x).

At the γ = 5 optimum this takes the standard deviation from 2,390 to 103 bps/yr, about 540 times less variance. The correction there is −0.6 ± 0.7 bps/yr (20,000 paths, §12). The oracle reports the correction's size for every setting.

**Solve.** Value iteration on the §3 equations:
- G on the lattice is r̂ + ρ·mean_m V(next);
- V(s) = max(G(s), max_a [G(a) − c(s, a)/B]).

The cost matrix is 5,151 × 1,326, about 27 MB in float32. Each iteration gathers V over the stored next indices: 5,151 × M integers, about 41 MB at M = 2,000 if stored as int32 (`nearest_lattice_index` returns int64, twice that). Time is not measured yet. `discount = 0` gives the myopic reference policy; `value_of(table, discount=0.95)` scores it.

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
5. **Continuous replay.** Replaying the oracle's policy on continuous, unrounded holdings, on fresh paths, gives a value close to V. Expect V to be slightly above the replay, because of the maximization bias above. The difference bounds the error the evaluation will see.

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

**What to disclose.** The agent receives:
- the target grid's weights, which are action definitions;
- its own holdings;
- the net rewards.

The two distance features build in the cost family's shape (linear and quadratic in the trade) but not its size, which the agent learns. It never receives μ, Σ, κ, λ, γ or the oracle (§10).

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

**Reference policies:**
- the oracle;
- always rebalance to the best no-cost target;
- trade once to it, then always hold;
- myopic (the oracle solved with ρ = 0 when choosing, scored with ρ = 0.95).

A policy is a function of (holdings, block index); only "trade once, then hold" uses the index (C10). `evaluate.agent_policy(agent)` and `evaluate.oracle_policy(solution)` adapt the agent and the oracle to this form (Albert).

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

**Agent runs** (§11 timeline): the four levels, 10 seeds each, γ = 5. Run the oracle on all levels before any agent run.

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

## 9. Hyperparameter tuning (William designs, Albert runs)

- **Seeds:** tuning uses master seed 20261112 (`spec.TUNING_MASTER_SEED`), never 20261016 or 20261103.
- **What is tuned:**
  - α per parameter block;
  - l2;
  - the target-refresh interval;
  - the greedy-refresh interval, if one is used.
- **Setting:** the $1B headline level only. The same values are used for every setting, and the report says so.
- **Criterion:** the final greedy policy's realized discounted net reward on fresh evaluation paths from the tuning seed. Never the oracle gap.
- **Choosing rule, fixed before running:**
  1. The highest mean over 3 tuning seeds wins.
  2. If the runner-up is within one paired standard error, add 3 seeds and decide on 6. This is the rule `src/tune_fa.py` used.
- **Candidate grid:** William, Phase 2.

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
- oracle run time and memory;
- the path count M that check 3 needs;
- the agent's speed;
- how often the oracle holds at each cost level (§8 gives an estimate only).

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

**C13. Two documents still describe the old options as current.** `TRANSACTION_COSTS.md` and `AUDIT_2026-11-06.md` §6.2 were not touched, by instruction.
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
- **D10 (§7). Decided (Albert, 2026-11-13):** H = 90 blocks and 1,000 starting holdings, as proposed. The §7 pilot gives paired-gap SEs from 0.5 to 7.4 bps/yr depending on how far apart the two policies' holdings get. Once the oracle exists, measure the oracle-minus-always-rebalance gap and the agent-against-oracle gap's SE, and raise m until that SE is below a third of the oracle-minus-always-rebalance gap (m = 4,000 halves every SE).
- **D11 (§6). Decided:** θ₁ and θ₂ are kept at zero or above.
- **D12 (§6). Decided:** the rule mirrors the baseline's `greedy_stable_through_end`, applied to the probe states (§6).

C4, C10, C11 and C14 are already settled in the stubs; C5 and C12 need only the edits described.

## 14. Risks

1. **The agent may not resolve the decisions.** One block's reward has a standard deviation of 2,390 bps/yr, while one block of drift from the optimum costs 1.3 bps/yr. To resolve that loss at one standard error the agent would need about (2,390/1.3)², roughly 3 million, blocks spent on the comparison. The oracle sweep should pick costs where decisions matter, and the agent's expected directions are registered only after tuning. A null agent result against a working oracle is still a reportable finding.
2. **Divergence (§6).** Checked against the oracle's G after training.
3. **The oracle is approximate.** Rounding and finite paths are bounded by checks 3 to 5, and its in-sample V is biased upward. Gaps use its replayed value, and a policy on continuous holdings can beat it slightly, so small negative gaps are possible.
4. **Compute and the gate.** Oracle time is unmeasured, and the 0.5% lattice check is four times the work. If checks 1 and 2 fail by Nov 17, fall back (§11).
5. **Disclosure.** The universe framing (C1), cost levels chosen with the oracle (§8), tuning on simulated data (as in `AUDIT_2026-11-06.md` M12), and post-hoc labels on everything.
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
