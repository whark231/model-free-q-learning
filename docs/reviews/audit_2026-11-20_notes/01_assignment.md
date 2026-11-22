# 01 Assignment (working notes, audit 2026-11-20)

**TL;DR / In plain terms:** the assignment text was extracted from `assignment.pdf` with macOS PDFKit (pypdf is not installed) into `assignment_text.txt` in this folder. This note lists every requirement as a checkable item, so the audit and the report skeleton can tick each one.

Source: `assignment_text.txt` (233 lines). Quotes below are verbatim from it.

## Central question
"Can a model-free learning algorithm recover a near-optimal portfolio allocation when it observes realized rewards but does not know expected returns or covariances?"

A strong analysis separates:
- S1 continuous optimum vs best portfolio on the discrete grid
- S2 learning error vs approximation (discretization) error
- S3 calibration of the synthetic environment vs evaluation on fresh simulated returns
- S4 statistical convergence vs economically meaningful convergence

## Part I
- I-1 Nine ETFs: SPY EFA EEM IEF LQD TIP GLD VNQ DBC; Yahoo Adjusted Close, daily.
- I-2 "Record the download date and verify the adjustment convention used by your code."
- I-3 Risk-free: "obtain a daily risk-free rate from the Kenneth French Data Library or another clearly documented source" (if modelling excess returns).
- I-4 "common sample ... beginning no earlier than February 2006"
- I-5 "Use simple returns for the mean-variance analysis and explain any use of log returns elsewhere"
- I-6 "Align trading dates and state how missing observations are handled"
- I-7 "Treat distributions and splits consistently across ETFs"
- I-8 "Keep return means, covariance estimates, risk aversion, and simulated rewards at the same time frequency"
- I-9 "Report the calibration period, number of observations, annualized summary statistics, and the correlation matrix"
- I-10 Moments held fixed; r_{t+1} = μ + ε, ε ~ N(0, Σ), excess returns.
- I-11 "may not use the calibrated mean vector, covariance matrix, or theoretical optimal weights during training."

## Part II
- II-1 "clearly stated risk-aversion parameter"
- II-2 "Derive or verify the closed-form solution for the budget-only problem."
- II-3 "solve the same problem numerically under the feasible set used by the learning agent. The main comparison must use identical constraints for theory and learning."
- II-4 Long-only fully invested as primary spec.
- II-5 "For every specification, report two benchmarks: the continuous optimum and the best portfolio available in the agent's discrete action set."
- (Bonus: bounded shorting. Not attempted.)

## Part III
- III-1 State: stateless baseline.
- III-2 Action counts: 25% → 495; 20% → 1,287 (recommended baseline); 10% → 43,758; 5% → 3,108,105.
- III-3 "Do not insert the theoretical optimum into the action set." Grid built without benchmark info.
- III-4 Reward matches objective; recommended episodic realized mean-variance (mean − γ/2 × realized variance over a short batch).
- III-5 Standard Q-learning update, stateless form; ρ = 0 is the "clean bandit-like benchmark".
- III-6 "Epsilon-greedy exploration is required." ε_t = e^(−βt) permissible.
- III-7 Report and justify: learning-rate rule (and visits), initial ε and β, discount, episodes and returns per episode, Q init, seeds, tie-breaking.

## Part IV
- IV-1 Train on simulated returns; evaluate on new draws not used to update Q.
- IV-2 Multiple seeds.
- IV-3 "You must define convergence before examining the final results."
- IV-4 "Report how many observations and episodes are required to reach the criterion, how often convergence occurs across seeds, and whether the result changes with the size of the action set."

## Part V
- V-1 Average learned weights over a separate steady-state evaluation sample.
- V-2 Grouped bar chart (nine weights); absolute weight differences by asset; L1; Euclidean distance to w*.
- V-3 For learned, continuous optimum, best discrete, equal weight: mean return, volatility, objective value.
- V-4 Discretization gap; learning gap; distribution of gaps across seeds.
- V-5 Learning curve (moving-average reward or objective); cumulative regret vs best discrete; observations to a defined level of optimality; convergence under at least two resolutions.

## Part VI
- VI-1 "at least two robustness checks, including one that changes the action-space design and one that changes an economic or statistical assumption."
- VI-2 Discussion must address: estimation error in μ and Σ; conditioning and numerical stability of Σ; curse of dimensionality; discretization error and possible benchmark leakage; reward/objective consistency; limits of learning from a stationary simulated environment.

## Deliverables
- Report, about 10 to 15 pages, structure: Introduction and central research question / ETF data, sample construction, and calibration / Theoretical benchmark and feasible portfolio set / Q-learning design and implementation / Convergence and benchmark comparison / Robustness, research challenges, and limitations / Conclusion.
- "It should explain what the agent learned, how close it came to the best attainable action, and what prevented exact recovery."
- Code: downloads/imports and cleans; estimates and documents the environment; computes continuous and discrete benchmarks; constructs action space without benchmark info; trains across multiple seeds; "Reproduces every reported table, figure, and robustness check". README: run instructions, package requirements, seeds, expected output files.
- Evaluation: Data and environment 20; Benchmark and feasible set 20; Q-learning design 20; Convergence and comparison 20; Interpretation, robustness, insight 20.
- Bonus: bounded shorting, transaction costs and rebalancing, state-dependent returns, FA for the large action space, 3-asset vs 9-asset. "Complexity alone does not earn bonus credit. The extension must be correctly designed, benchmarked, and interpreted."
