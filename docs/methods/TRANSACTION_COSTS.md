# Transaction costs and rebalancing

**Owner:** Albert (design and runs), paired with William · **Code:** `src/transaction_costs.py`, `src/tc_sweep_report.py` · **Tests:** `src/tests/test_transaction_costs.py` (2 tests)
**Status (2026-11-06):** implemented. The quadratic-cost sweep ran at full scale (commit `da2b0fa`; `results/tc-quadratic-cost/`, `reports/tc-quadratic-cost/`). The discount and learning-rate comparisons are registered but **not run**, and should not be run as designed (see Outcome below and `AUDIT_2026-11-06.md` H1).

**Superseded (D7, 2026-11-12; pointer added 2026-11-20):** this first design is kept as a labelled negative result. The transaction-cost extension that is run and reported is the redesign with drift, in `TC_DRIFT_DESIGN.md` (`src/tc_drift/`).

**TL;DR / In plain terms:** the agent starts each 20-day block holding its previous target, picks a new one, and pays a quadratic cost for the move. The code is correct, but in the sweep the trained agent did no better than picking a random portfolio, even when trading was free. Most of its 1.66 million (holding, target) pairs were tried once or never, which is far too little data for rewards this noisy. The design needs to change before the extension can be reported as a success.

**Key concepts to understand**

- **State and action.** The state is the portfolio held at the start of the block, the action is the new target, and the next state equals the action. A tabular agent therefore keeps 1,287 × 1,287 ≈ 1.66 million values, against 1,287 for the stateless baseline.
- **Why the zero-cost level is a control.** With no cost, the best target is the same from every holding: the best grid portfolio. Any learner that works should find it, as `baseline-v1` does within 14 bps/yr.

**Key takeaways**

- As run, the sweep shows that this design fails. What costs do to a learner is still unmeasured. See the Outcome section.
- `AUDIT_2026-11-06.md` §6.2 lists redesign options. A decision is needed before any further TC run.

---

William, the first version adds a cost for moving between portfolio targets while keeping the baseline return simulator. The agent starts each 20-day block with its previous target, chooses a new target on the 20% grid, pays any trading cost once, and receives the block's realized daily mean–variance reward less the cost divided by 20. The target becomes its starting allocation for the next block. Keeping the same target costs zero. This is a target-weight model that abstracts from drift between trading dates.

We use Gârleanu and Pedersen's quadratic form with a scalar times the identity as the cost matrix. The model uses weight changes and a fixed return distribution. It has no return predictors. A 20 percentage point transfer between two ETFs is the reference trade. At the anchor cost of 2 bp, the quadratic coefficient is 0.005 and a 40 point transfer between the same ETFs costs 8 bp. Cost levels are illustrative.

| Branch | Comparison and rationale |
|---|---|
| `feature/transaction-costs` | Shared implementation, benchmark and this note. The state is the previous grid target. The next state equals the action. |
| `run/tc-quadratic-cost` | Sweep 0, 1, 2 and 5 bp per reference trade with discount 0.95 and learning rate 1/N(s,a). Zero cost checks the known optimal grid target. The 2 bp case anchors the other comparisons. |
| `run/tc-discount` | Compare 0, 0.90 and 0.95 at the 2 bp cost and 1/N(s,a). The zero-discount agent pays the same cost and optimizes the next block alone. Score all policies on the same 0.95 objective to measure the value of planning ahead. |
| `run/tc-learning-rate` | Optional comparison of 1/N(s,a) and constant 0.00032 at the anchor cost and discount. The constant matches tabular-const and fa-quadratic. 1/N now counts visits to each starting-target/action pair. The stateless spline step size does not transfer to this tabular model. |

All branches hold risk aversion at 5 and reuse the same return, action and start seeds across paired settings. The main discount of 0.95 per block gives rewards one trading year away about half the weight of immediate rewards. Today’s action affects the next switching cost, so continuation value can change action rankings. During training, restart from a uniformly sampled grid state every 100 blocks to improve coverage. These are sampling restarts for the continuing problem, so the update still uses the continuation value of the chosen target on the last block before a restart.

The assignment calls for a justified learning rate and discount, separate evaluation draws, multiple seeds, convergence criteria set in advance, and benchmarks with matched constraints. Keep the required baseline results separate. The extension benchmarks are a discounted grid oracle, a cost-aware myopic policy, and a policy that targets the no-cost grid optimum while paying the same costs. Only benchmark construction and evaluation use the calibrated moments directly. For the extension, quote the gap to the matched grid oracle separately from the original continuous-versus-grid discretization gap. The oracle solves a discounted problem over the same grid. Evaluation scores the first 100 blocks from paired starts with the same return paths and discount, including the first rebalance.

Report the oracle gap, realized and expected net mean–variance reward, trades, turnover and coverage. Track whether the full greedy policy stays unchanged at checkpoints through the final 10% of training. This checkpoint measure can miss switches between checkpoints and should be read alongside the economic gap. There are about 1.66 million state–action pairs on the 20% grid. Use the smoke run to check coverage and time before treating the default budget as sufficient.

The run configurations are under `experiments/transaction-costs/`. For example:

```bash
python -m src.transaction_costs experiments/transaction-costs/quadratic-cost.json --dry-run
python -m src.transaction_costs experiments/transaction-costs/quadratic-cost.json \
  --steps 200000 --seeds 1 --evaluation-paths 100 --out /tmp/tc-smoke.json
python -m src.transaction_costs experiments/transaction-costs/quadratic-cost.json \
  --out results/tc-quadratic-cost/summary.json
```

Run the other comparisons by substituting `discount.json` or `learning-rate.json` and a separate output path. Defaults are 10 seeds, 13 million training blocks per seed and 1,000 evaluation paths. The configured defaults are a starting budget. Record any revisions before running the reported comparisons.

## Outcome of the quadratic-cost sweep (written 2026-11-06, audit)

Full scale: 10 seeds × 13M blocks per cost level, γ = 5, discount 0.95, α = 1/N(s, a), 1,000 evaluation paths of 100 blocks. Values are expected net CE in bps/yr, discounted at 0.95 over the 100 blocks; learned values are medians over seeds.

| Cost per 20-point transfer | Grid oracle | Ignore costs | Myopic | Learned | Learned gap to oracle, median (range) |
|---|---|---|---|---|---|
| 0 bp | 524.8 | 524.8 | 524.8 | 56.6 | 468.2 (436.4 to 485.2) |
| 1 bp | 520.9 | 520.8 | 515.9 | −16.8 | 537.7 (482.4 to 589.8) |
| 2 bp (anchor) | 517.1 | 516.8 | 512.6 | −90.9 | 608.0 (572.5 to 658.7) |
| 5 bp | 507.6 | 504.9 | 479.2 | −296.8 | 804.3 (744.9 to 844.3) |

**What it shows.**
- **The learned policy is no better than a random target.** At zero cost the learned median, 56.6 bps/yr, matches the average over all 1,287 grid portfolios (56.7). Learned policies trade in 94.8 to 100 of the 100 blocks and turn over about 70 times the portfolio per path. The falling learned values at higher costs are those costs charged on random churn.
- **Why.** Exploration supplies about 1/β ≈ 1.41M random actions over 1.66M (holding, target) pairs, so about 57% of pairs are ever tried (observed 57.1%) and a typical tried pair is tried once. One block's reward has a standard deviation of about 3,900 bps/yr. In addition, with α = 1/N and discount 0.95 the estimated continuation value is noisy and carries a bias that depends on how often each row was visited. (A continuation value that differs by target is intended here; the estimate is what fails.) The code itself is correct: the independent reviewer ran the unmodified training loop on a 45-portfolio grid with exact rewards and it matched the oracle within 0.3 bps/yr (reviewer, not re-checked).
- **Costs barely matter in this environment.** Because targets never drift, the best policy makes one move. The dynamic oracle beats simply ignoring costs by 0.1 bps/yr at 1 bp, 0.3 at 2 bp and 2.6 at 5 bp, and the myopic benchmark does worse than ignoring costs. The report does not show the ignore-costs column.
- **Stability.** No seed meets the checkpoint-stability criterion, and none could: it fails if any of the 1,287 rows changes its best action in a 1,000-step chunk.

**What the report says that should change.** "Differences between the four cost levels isolate the configured cost" and "trading behavior may be concentrated near the starting allocation" do not hold for these policies. `src/tc_sweep_report.py` writes that text; rewrite it after the redesign (`AUDIT_2026-11-06.md` M8).

**Reproducing the committed results.** The committed results are four single-variant files (`no-cost.json`, `low-cost.json`, `anchor.json`, `high-cost.json`), which `python -m src.tc_sweep_report` requires. They were produced from single-variant configurations that are not in the repo; the command above with `quadratic-cost.json` writes one file with all four. Commit those configurations, or document the exact commands (`AUDIT_2026-11-06.md` M9).

**Next.** Options are in `AUDIT_2026-11-06.md` §6.2. The registered discount and learning-rate comparisons would inherit the same failure: on a small grid with about 19× more visits per pair, discount 0 left a 92 bps/yr gap and a constant α = 3.2e-4 a 264 bps/yr gap (reviewer's tests, not re-checked).
