# Transaction costs and rebalancing

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
