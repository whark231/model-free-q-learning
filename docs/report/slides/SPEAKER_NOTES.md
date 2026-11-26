# Speaker notes: Model-Free Q-Learning and Optimal Portfolio Choice with Nine Assets

**TL;DR / In plain terms:** fifteen slides plus three backup slides for questions. At the times given the talk runs about 11 minutes, so cut a few seconds a slide if the limit is 10. Each note is written to be spoken. Every number is in `docs/report/REPORT_final.md` and was checked against the committed result files.

**Key concepts to have ready**

- **Certainty-equivalent bps/yr.** Every gap is the sure annual return the investor gives up, in basis points.
- **Discretization gap and learning gap.** Best grid portfolio against the optimum, and learned portfolio against the best grid portfolio.
- **Pre-registered and post hoc.** Slides 4 to 9 and the 25% grid on slide 10 are pre-registered. Slides 11 to 14 and the 10% grid on slide 10 are post hoc, and each carries a tag. Two lines cross that boundary. The risk-aversion footnote on slide 11 (γ = 3 and 8) is pre-registered, and the Q-table diagnostic line on slide 9 is post hoc.

**Key takeaway to land:** the agent gets within about 20 bps/yr of the optimum out of 530, and reward noise about 370 times larger than the differences among the best portfolios stops it getting the rest.

Suggested split: Albert takes slides 1 to 9, William takes 10 to 15.

---

## 1. Title (10 s)

Introduce yourselves and the project, model-free Q-learning for portfolio choice across nine ETFs, three each in equities, bonds and real assets. Then move straight to the question.

## 2. Can an agent that only sees rewards find a near-optimal portfolio? (45 s)

Our question is the assignment's: can a model-free learner recover a near-optimal portfolio when it only observes realized rewards and never sees expected returns or covariances? We calibrate a simulated market on nine ETFs and let a Q-learning agent choose among 1,287 long-only portfolios. The chart is the answer in one picture. The continuous optimum is worth 530.5 basis points a year of certainty-equivalent return. Across our ten seeds the median gap to it is 19.8. An equal-weight portfolio ends 310.9 behind. So the agent closes almost all of the gap that a naive investor leaves open, and the rest of the talk explains the last 20 basis points.

## 3. From raw prices to a trained agent (45 s)

This is the whole pipeline on one slide. The top row runs once. We take daily adjusted closes for the nine ETFs from Yahoo and the daily risk-free rate from Kenneth French, turn them into excess returns, and align the dates, which leaves 5,152 days. From those we estimate the mean vector and the covariance matrix once and freeze them inside a Gaussian simulator. That simulator is the ground truth of the experiment. The bottom row is one training episode. The agent picks one of 1,287 portfolios, mostly at random early on and greedily later. It holds that portfolio for 20 simulated days, earns the realized mean minus half gamma times the realized variance, and updates its estimate for that portfolio. That repeats 13 million times per seed, and we run ten seeds. At the end we score each seed's favorite portfolio against the two benchmarks, exactly from the true moments, with 10,000 fresh episodes as a check. The agent only ever sees which portfolio it picked and the reward. It never sees the moments or the benchmarks.

## 4. The environment: 20 years of daily data, estimated once and frozen (45 s)

The data is the nine ETFs in the assignment: three equity, three fixed income, three real assets. We use Yahoo adjusted closes and simple excess returns over the French daily risk-free rate, from February 2006 to July 2026, which is 5,152 days. Dates are inner-joined and nothing is filled in. Everything is at one daily frequency. The correlation matrix already hints at the answer. SPY and gold have the two best Sharpe ratios, 0.55 and 0.50, and a correlation of 0.07. Treasuries are negatively correlated with equities. We estimate the mean vector and covariance once, freeze them, and draw Gaussian returns from them. The agent's code cannot import those moments, and tests enforce that.

## 5. Two benchmarks under the agent's own constraints (45 s)

The investor maximizes mean minus half of gamma times variance, with gamma equal to 5 as the baseline. We derived the budget-only closed form, but the agent is long-only, so the benchmark is solved numerically under the same long-only, fully invested constraints. That optimum holds only three ETFs: SPY, IEF and gold. As risk aversion goes from 3 to 8, weight moves from equities and gold into Treasuries. The second benchmark is the best portfolio on the agent's grid, found by checking every one. On the 20% grid it is 40, 20, 40, and it sits 5.7 basis points a year below the continuous optimum. That is the discretization gap. The grid is built from the asset count and the step alone.

## 6. The agent: one value per portfolio, updated with averaged rewards (50 s)

The agent is the stateless baseline. Each episode it picks one portfolio, holds it for 20 simulated days, and receives the realized mean minus half gamma times the realized variance. With the sample variance, the expected reward equals the theoretical objective exactly, so reward and benchmark agree. The update is the standard Q-learning update with discount zero, and the learning rate is one over the visit count, which makes each Q-value the average reward for that portfolio. Exploration is epsilon-greedy with exponential decay, reaching 1% at the halfway point. Each seed runs 13 million episodes, so every portfolio gets about 1,097 random tries. All of this, including the convergence rule, was fixed before any full-scale run.

## 7. What it learned: SPY and GLD on every seed, usually a bond fund (50 s)

Here is what the agent chose at gamma 5. On the left, the dark bars are the optimum and the best grid portfolio, and the green bars are seed 1, the median seed by gap. On the right, each row is one seed's final portfolio. We show seeds one by one and never average them, since an average is a portfolio no seed chose. Every seed holds SPY and gold. Eight of ten add a bond fund. No seed holds international equity, REITs or commodities. Three seeds land exactly on the best grid portfolio. Across seeds the median gap to the best grid portfolio is 14.1 basis points a year, and the worst seed is 47.3 behind. The agent found the right assets from rewards alone.

## 8. Convergence: 9 of 10 seeds meet the rule set in advance (45 s)

We defined convergence before looking at results. The primary rule is statistical: the greedy portfolio stops changing and stays unchanged to the end, for at least 10% of training. Nine of ten seeds meet it, at a median of 185 million simulated days. The economic rule asks whether the agent stays within 25 basis points of the best grid portfolio for good. Seven of ten do, by about 26 million days. The two differ. One seed is statistically converged while 47 basis points off, and seeds settle on their final portfolio early, at a median of 47 million days, then make brief excursions that delay the formal rule. The coarser 25% grid, in gold, converges on all ten seeds and sooner.

## 9. Why not exact: the top portfolios are 10 bps/yr apart and the noise is 3,900 (50 s)

Why only three seeds of ten? The left chart is on a log scale. The best and second-best grid portfolios differ by 10.5 basis points a year. One episode's reward has a standard deviation of about 3,900. To tell those two portfolios apart at two standard errors, you need about 1.1 million tries of each. On the right, in the seeds that missed, the best portfolio was tried between about 1,100 and 4,300 times. Once exploration fades, a good portfolio with an unlucky early average rarely gets another look. A post hoc look at the Q-table fits that story. It ranks the whole grid well, with a rank correlation of 0.82, and finds only about a third of the true top ten.

## 10. A finer grid trades discretization error for learning error (50 s)

This is our action-space robustness check. The left panel is pre-registered. The 25% grid has 495 portfolios. Its discretization gap is twice as large, its learning gap is much smaller, and its median total gap is 14.3 against 19.8 for the 20% grid. The right panel is post hoc. The 10% grid has 43,758 portfolios. We gave both grids longer episodes and twice the budget. At gamma 5 the finer grid offers 4.4 basis points less discretization error, and the agent captures none of it. The paired change is minus 0.6 with a standard error of 3.7. At gamma 3 and 8 it is worse. No seed of thirty converges on the 10% grid.

## 11. Longer episodes and more data cut the gap; slower exploration mostly does not (45 s)

These runs are post hoc, apart from risk aversion. For gamma 3 and 8, which were pre-registered, the median gaps are 10.0 and 5.7, and the learned portfolios follow the optimum toward bonds. Then we changed one setting at a time. Episodes of 63 days cut the mean gap at gamma 5 from 17.6 to 5.1, though they use three times the data per episode. Doubling the episodes cuts it to 8.3. Slower exploration decay barely moves it at gamma 5 and 8. With all three changes, five, eight and nine seeds of ten are exactly right. Most single differences are within two standard errors, so we read the direction and not the size.

## 12. Spline function approximation: better ranking, worse top pick on the large grid (50 s)

Our first extension replaces the table with a function of the weights, so that similar portfolios share information. It needs a constant step size, so the fair comparison is a table with a constant step. On the 20% grid, splines beat that table at gamma 5, 15.7 against 43.5, and at gamma 3, and lose at gamma 8. On the 10% grid, where we expected the biggest benefit, splines lose at every gamma, with 78.8 against the table's 11.7 at gamma 5. That comparison is confounded, because the table there uses the one-over-N rule and we ran no constant-step table on that grid. At gamma 5 the spline model ranks the whole grid far better, 0.98 against 0.54, and at the median finds none of the true top ten. We report it as a negative result for this configuration.

## 13. Transaction costs: a redesign after a first design that learned nothing (45 s)

Our second extension is transaction costs, and the first attempt failed. We made the state the previous portfolio on the nine-asset grid. That is 1.66 million state-action pairs, and even at zero cost the agent scored 56.6 against the oracle's 524.8. That is the same as the average over all 1,287 portfolios, 56.7, so it learned nothing. The redesign uses three ETFs, the ones in the optimum. Holdings drift with returns over 63-day blocks, which gives a reason to rebalance. Each block the agent holds or trades to a target, paying a spread cost and an impact cost sized for funds of 100 million, 1 billion and 10 billion dollars. A numerical oracle is the benchmark. It beats always rebalancing by only 1.8 basis points at 1 billion and 7.8 at 10 billion, so the gain available from timing is small.

## 14. The agent's gap to the oracle narrows as costs rise, from 13.1 bps/yr to 2.8 (50 s)

The agent trails the oracle at every cost level and on all forty seeds, but the gap narrows as costs rise: 13.1, 13.1, 10.0 and 2.8 basis points a year. The chart splits that gap. Orange is where the agent heads, meaning the target portfolio it prefers. Green is when it trades. At the three lower cost levels nearly the whole gap is orange. The agent's target holds too much in Treasuries, and it even loses to a simple always-rebalance rule, which is handed the right target. At 10 billion, timing is worth 6.2 basis points against the oracle's 7.8 lead over always rebalancing, the agent beats always rebalancing by 5.0, and it is statistically indistinguishable from the myopic policy, which knows the moments and the costs. Most of that narrowing comes from a better target at 10 billion, which we have not explained. We registered twelve expectations before these runs, leaning on a pilot of the same agent, and eight were met and four partly. Our gap prediction was too optimistic.

## 15. Conclusions (45 s)

To answer the question directly. What did the agent learn? The right assets and a near-optimal mix, from rewards alone. How close did it come? At gamma 5 the median gap across seeds is 14.1 basis points a year to the best attainable portfolio and 19.8 to the optimum, where equal weight loses 310.9. What stopped it? Reward noise about 370 times the difference between the top two portfolios. Finer grids and richer models moved the error around at our budget without removing it. The caveats are that this is a stationary Gaussian world with the moments taken as true and 260 million simulated days per seed. Only the four baseline experiments were pre-registered, and benchmark information entered some design choices but never the agent.

---

## Backup slides

**16. Gaps in all four pre-registered experiments.** Use if asked about the spread across seeds or about other risk aversions. Left: the median shortfall across seeds split into discretization and learning. Right: every seed's total gap; seeds on the line found the best grid portfolio. At gamma 3 the mean gap (30.6) is far above the median (10.0) because four seeds are more than 55 basis points behind.

**17. Where the transaction-cost agent heads, and how often it holds.** Use if asked why the agent loses at low cost. Every favorite target at zero cost, $100 million and $1 billion holds 18% to 32% in IEF against 14% in the best target. At $10 billion the agent holds in 17% of blocks against the oracle's 45%, with turnover of 2.11% against 1.55%. Always rebalancing almost never holds and turns over 2.98%, and we have no analysis of which part of the agent's trading produces its timing value. We have not investigated why the target error is smaller at $10 billion and should say so if asked.

**18. It learns the quadratic cost and not the linear one.** Use if asked what the agent learned about costs. The quadratic coefficient is within a factor of two on all ten seeds at $1 billion and $10 billion. The linear coefficient is zero or below 0.2 of the truth on all ten seeds at $10 billion. Our registered prediction for the gap was 1 to 8 basis points, and three of the four levels came in worse.

## Likely questions

- **Is 260 million days realistic?** No. It is about a million years of daily data. The experiment shows the problem is learnable in principle and expensive in data.
- **Did the agent ever see the benchmark?** No. We used the benchmark to size the episode budget and to design the transaction-cost extension, and both are disclosed in the report.
- **Why not the closed-form portfolio as the benchmark?** It is not long-only. The comparison uses identical constraints for theory and agent, as the assignment requires.
- **Why report exact-moment gaps and not simulated ones?** A 10,000-episode evaluation sample has a standard error near 39 basis points a year, larger than the gaps. Both are in the report.
- **How were the function-approximation step sizes chosen?** On a separate tuning seed, by training in full and scoring each candidate's final greedy portfolios on 200,000 simulated evaluation episodes. That is validation through the simulator, which a real investor with one history could not do. No closed-form benchmark was used.
- **Is the 10% grid spline result a fair fight?** Not fully. Splines use a constant step and the table there uses one over N. On the 20% grid the constant step alone costs the table 25.9, 76.6 and 30.2 basis points at gamma 5, 3 and 8, and we ran no constant-step table on the 10% grid.
- **Where did the transaction-cost predictions come from?** From a pilot of the same agent at the 1 billion level on a different master seed, including its gaps to the oracle. They were frozen before the final runs.
