# Next Steps: The Ground-Truth Environment

**From:** William
**Date:** 2026-09-29
**Status:** four of ten required core components are done and tested. This document scopes the next one.

---

## TL;DR / In plain terms

The data pipeline, the theoretical benchmark, the action space, and the reward function are all built and tested (74 tests passing across the project). What's missing is the piece the assignment calls the Ground Truth Environment: code that actually draws random simulated daily returns from the calibrated model. `src/environment.py` exists as a file but is empty.

That's the next piece, and it's yours to build, with me reviewing, matching how we split Part I. Once it's working and tested, I want to pair with you on the Q-learning agent itself (`src/agent.py`), which hasn't been started.

---

## What's done

| Component | Where | Evidence |
|---|---|---|
| Nine-ETF data pipeline | `src/data/` | `DATA_CLEANING.md`, 5,152-day sample, reviewed |
| μ and Σ estimation | `data/processed/covariance_daily.csv`, `summary_statistics.csv` | Part of the same pipeline |
| Theoretical mean-variance benchmark | `src/benchmarks/mean_variance.py` | 27 tests in `src/tests/test_mean_variance.py` |
| Feasible action space (weight grid) | `src/action_space/grid.py` | 12 tests in `src/tests/test_action_space.py` |
| Reward function | `src/rewards.py` | 5 tests in `src/tests/test_rewards.py`, including a Monte Carlo check that the reward is unbiased for the true mean-variance utility |

Everything above also has the bugs from your `REVIEW_NOTES_FOR_WILLIAM.md` fixed: the SLSQP tolerance issue, the hardcoded grid size, the float-drift in grid weights, and the tuple/int mix-up. All four are closed and covered by regression tests, including the two you wrote in that file (`test_daily_and_annual_solutions_agree` and `test_discretization_gap_is_non_negative_on_daily_moments`).

The `.vscode/settings.json` issue from that review is also closed. It's untracked now (`.gitignore` covers `.vscode/*` with no exception), so it stays personal to each machine and won't hand you a Windows-only path.

---

## What's next: the synthetic return generator

### In plain terms

The assignment splits "ground truth" into two steps. Step one is estimating μ and Σ from the real ETF data, which is done. Step two is treating those numbers as fixed and building a simulator that generates fake daily returns as if the world followed exactly that Gaussian model. The Q-learning agent will never see real market data. Every return it ever observes, during training and during evaluation, comes from this simulator.

### The specification

From the assignment, Part I:

> Let the nine-dimensional excess-return vector follow a stationary Gaussian process:
> `r_{t+1} = μ + ε_{t+1}`, with `ε_{t+1} ~ N(0, Σ)`

μ and Σ here are the calibrated estimates already sitting in `data/processed/`, held fixed. Each simulated day is an independent draw from `N(μ, Σ)`.

### The interface constraint that already exists

`src/rewards.py` is already built and tested, and it expects its input in a specific shape:

```python
episodic_mean_variance_reward(w: np.ndarray, returns_batch: np.ndarray, gamma: float) -> float
```

`returns_batch` is a `(B, 9)` array: B simulated days, one column per asset, in the ticker order fixed by `src/data/universe.py: TICKERS`. The reward function does not generate this array. Something upstream has to. That's the piece you're building: a function that takes μ, Σ, a batch size, and a seeded random generator, and returns a `(B, 9)` array of draws from `N(μ, Σ)`.

### Requirements to carry over from the rest of the project

- **No leakage.** The agent can see the returns the environment hands it and the reward computed from them. It can never see μ, Σ, or the theoretical optimum. This is a CLAUDE.md rule and an explicit assignment requirement.
- **Fixed, logged seeds.** Every draw needs a seeded `np.random.default_rng(...)`, per CLAUDE.md's reproducibility rules and the assignment's own requirement to report random seeds.
- **Stateless is fine for now.** The assignment states the required baseline is stateless, since returns are i.i.d. day to day. You don't need to build any notion of holdings or state transitions yet. That only shows up in the transaction-cost extension, which is much later.
- **Daily frequency, same as everything else.** μ and Σ in `data/processed/` are already daily. No rescaling needed.

### Suggested tests

Match the testing style already used in this repo (`test_rewards.py` is the closest example): draw a large number of batches, average them, and confirm the sample mean and sample covariance of the simulated returns converge to the calibrated μ and Σ. That's a statistical check, not just a shape check, and it's the kind of test that would have caught the SLSQP bug earlier if it had existed for that function.

---

## Key concepts to understand

- **Ground truth, in this project's specific sense.** It doesn't mean "the true statistics of financial markets." It means "the fixed parameters of our synthetic experiment." The assignment says this directly: μ and Σ "are not assumed to be the true moments of financial markets. They are the ground truth only within your controlled experiment."
- **Why the agent never sees μ or Σ.** The whole research question is whether a model-free learner can recover something close to the optimum using only realized outcomes. If the agent's code ever touched μ or Σ directly, the experiment would no longer be testing that.

## Key takeaways

- Four of ten core components are done. The environment is the next one, and it unblocks the Q-learning agent, which unblocks everything after it (training loop, convergence analysis, comparisons).
- Your file to build: `src/environment.py`. Your interface target: something that produces a `(B, 9)` array of `N(μ, Σ)` draws, since that's exactly what `episodic_mean_variance_reward` already expects.
- Not your problem right now: the value of γ (risk aversion) is still undecided. That's called out as my decision to make in `DATA_CLEANING.md` §8.4, and it doesn't block you, since the return generator itself doesn't use γ at all.

## What happens after this

Once the environment is built and tested, we pair on `src/agent.py` (the tabular Q-learning core, currently just a placeholder) together, as planned in the prospectus. No need to look at `src/agent.py` or `src/train.py` before then.
