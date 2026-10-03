# Part I: The Simulated Market

**Owner:** Albert · **Status:** built and tested on `feature/environment`, awaiting William's review
**Code:** `src/environment.py` · **Tests:** `src/tests/test_environment.py` (69 tests)
**Date:** 2026-10-02, numpy 2.3.5, pandas 2.3.3

---

## TL;DR / In plain terms

The agent never sees real market data. Every return it learns from comes from
this simulator, which draws fake daily excess returns for the nine ETFs from a
bell-curve (Gaussian) model built on the averages (μ) and co-movements (Σ) we
estimated in Part I. The agent is handed the simulator, which gives out
returns and nothing else, and tests fail if the agent's code imports the
loader, the benchmarks or the data files. Every random draw traces back to one
recorded seed, and the draws used to evaluate the agent are never used to
train it.

**Key concepts to understand**

- **Drawing correlated returns.** Start with nine independent standard-normal
  numbers z (mean 0, variance 1, uncorrelated). Compute μ + Lz, where L is the
  Cholesky factor of Σ, a lower-triangular matrix with L·Lᵀ = Σ. Multiplying by
  L mixes the independent numbers so that their covariance becomes exactly Σ,
  and adding μ shifts the average to μ.
- **Seeds and streams.** A seed makes random numbers repeatable. One master
  seed is split into separate streams (training returns, the agent's own
  randomness, evaluation returns), so changing one part of the experiment does
  not shift the random numbers another part sees.
- **Leakage.** If the agent's code could read μ or Σ, the experiment would stop
  testing model-free learning. The assignment reserves μ, Σ and the optimum
  for benchmark construction and evaluation.

**Key takeaways**

- The simulator's output matches numpy's reference Gaussian sampler exactly,
  and on 400,000 simulated days the sample means and covariances match μ and Σ
  within sampling error, with consecutive days uncorrelated as the stateless
  baseline assumes.
- The output feeds William's reward function directly. On the real
  calibration, the average reward over 20,000 simulated episodes matches the
  true mean-variance objective within sampling error.
- Next, the Q-learning agent takes a `ReturnSimulator`, and `train.py` owns the
  `SeedPlan`.

---

## 1. How to use

```python
from src.environment import ReturnSimulator, SeedPlan, load_calibration

mu, sigma = load_calibration()                  # daily, TICKERS order
plan = SeedPlan(master_seed=..., n_runs=10)     # record plan.as_dict() with results

returns_rng, agent_rng = plan.run_streams(run)  # one pair per training run
sim = ReturnSimulator(mu, sigma, returns_rng)   # hand only `sim` to the agent
batch = sim.sample_episode(20)                  # (20, 9) daily excess returns
blocks = sim.sample_episodes(10_000, 20)        # (10000, 20, 9): same numbers as
                                                # 10,000 sample_episode(20) calls

eval_sim = ReturnSimulator(mu, sigma, plan.evaluation_rng())  # identical for every policy
```

| Object | Used by | Purpose |
|---|---|---|
| `load_calibration()` | runner, benchmarks, tests | μ and Σ as arrays, ticker order checked |
| `simulate_returns(mu, sigma, n_days, rng)` | runner, tests | The interface in `NEXT_STEPS_FOR_ALBERT.md` |
| `ReturnSimulator(mu, sigma, rng)` | the agent, and evaluation | Hands out returns only; counts simulated days and episodes |
| `SeedPlan(master_seed, n_runs)` | `train.py`, `evaluate.py` | Every random stream, plus a JSON record for the results |

---

## 2. Design decisions

Albert approved these on 2026-10-02 before any code was written.

### 2.1 The process is exactly the assignment's

Each day is an independent draw of `r = μ + ε`, `ε ~ N(0, Σ)`, with μ and Σ
held at the Part I estimates (`summary_statistics.csv` column `mean_daily`, and
`covariance_daily.csv`, read with pandas' round-trip parser so the values are
exactly the ones `clean.py` wrote). All quantities are daily.

> **Assumption:** returns are stationary and i.i.d., as the assignment
> specifies for the baseline. The data itself has fat tails (LQD excess
> kurtosis 58.9, `DATA_CLEANING.md` §6), so the simulated world has thinner
> tails than the history it was calibrated on. That belongs in the report's
> limitations and in the Part VI fat-tail check.

### 2.2 Σ is factored once and reused

`ReturnSimulator` computes the Cholesky factor at construction and reuses it
for every episode. Its draws are identical to numpy's
`multivariate_normal(method="cholesky")` from the same generator state.

**Not taken:** calling `rng.multivariate_normal` per episode. It gives the same
distribution but re-factors Σ on every call. Measured per 20-day episode on
Albert's Mac (numpy 2.3.5, OpenBLAS 0.3.30), it takes about 40 µs with the
default SVD method and about 9 µs with `method="cholesky"`, against 3.8 µs
here. A block from `sample_episodes` costs 1.5 µs per episode and gives the
same numbers as drawing the episodes one at a time.

### 2.3 One master seed, separate streams

| Stream | `SeedSequence` spawn key | Used for |
|---|---|---|
| Training returns, run *r* | `(20261002, 0, r, 0)` | that run's `ReturnSimulator` |
| Agent randomness, run *r* | `(20261002, 0, r, 1)` | exploration and tie-breaking |
| Evaluation returns | `(20261002, 1)` | shared by every run and benchmark |

- Training and evaluation sit on different branches, so evaluation draws are
  never used to update Q-values (assignment Part IV).
- `evaluation_rng()` restarts the evaluation stream on every call, so every
  learned policy and every benchmark is scored on identical draws. These
  common random numbers cut the simulation noise in the differences between
  policies, which are the numbers Part V reports.
- The return draws do not depend on which portfolio the agent picks, and the
  agent's randomness has its own stream. The k-th training episode's returns
  are therefore the same whatever the agent does, and runs that differ only in
  grid resolution or learning rate face identical return paths.
- A run's streams do not depend on `n_runs`, so adding seeds later leaves the
  earlier runs unchanged.
- The leading tag 20261002 keeps the streams apart from anything seeded the
  plain numpy way. `default_rng(master_seed).spawn(n)` produces keys `(0,)`,
  `(1,)`, and so on, and without the tag `(1,)` would have been the evaluation
  stream.

No function has a default seed. Each requires a `numpy.random.Generator`
(CLAUDE.md §6), but it cannot check that the generator was seeded, which is why
generators should come from `SeedPlan`.

### 2.4 What the leakage guard enforces, and what it does not

Enforced by tests:

- The simulator's public surface is pinned to five names (two sampling methods,
  two counters and the asset count). Nothing public exposes μ or Σ.
- A source scan of `src/agent.py` fails on imports of `src.benchmarks`,
  `src.data.clean`, `load_calibration` or `simulate_returns`, and on any path
  into `data/` or any processed CSV name. Twelve test cases prove the scan
  catches each route, because `agent.py` is a placeholder today and the guard
  would otherwise pass vacuously.
- An import-graph check imports `src.agent` in a fresh interpreter and fails if
  the benchmarks or the cleaning code got loaded by any route, including a
  helper module that imports them on the agent's behalf.

Not enforced, and left to code review:

- **Values the training loop passes in by hand.** `train.py` legitimately holds
  μ and Σ for evaluation. The likeliest real leak is a value derived from them
  (an initial Q-table, a benchmark weight) handed to the agent's constructor.
  The agent's constructor should take only the action grid, its learning
  settings and `agent_rng`.
- **Deliberate bypasses** such as reading `sim._mu`, `vars(sim)`, `importlib`
  tricks, or `copy.deepcopy(sim)` to preview the next episode.

That leaves one design question for the agent session. The assignment lets the
agent see realized returns, but a 9-asset return batch also lets it score every
portfolio at once, or estimate μ and Σ itself. Both would change what is being tested.
The recommendation is that the training loop computes the reward and the agent
updates only the chosen portfolio's Q-value from it, and that the report says
so. If the agent is split across several modules, add them to `AGENT_MODULES`
in the test file.

### 2.5 Inputs are checked before anything runs

| Check | Why |
|---|---|
| CSV labels equal `TICKERS` on every axis | Weights are positional, so a reordered file would pair weights with the wrong assets (CLAUDE.md §5) |
| Σ symmetric, to 1e-10 relative | `np.linalg.cholesky` reads only the lower triangle and would accept an asymmetric matrix silently. The tolerance admits rounding-level asymmetry from shrinkage estimators |
| Σ positive definite, all values finite | Otherwise Σ is not a valid covariance |
| numpy arrays only | Converting a pandas object drops its labels, which hides a reorder |
| Episodes at least 2 days | The reward's ddof=1 variance is NaN for one day. `np.argmax` treats NaN as the maximum, so that action would become the permanent greedy choice |
| Seeds stored as plain `int` | A numpy integer would make `as_dict()` fail `json.dumps` after training has run |

The one-period quadratic reward that the assignment also permits would need
one-day episodes. If the team switches to it, lower `MIN_EPISODE_LENGTH` to 1
and put the length check into the reward function instead.

### 2.6 Gaussian only for now, with room for Part VI

- **Alternative calibration window or shrinkage covariance.** Pass a different
  μ and Σ to the same class.
- **Fat tails.** Override `ReturnSimulator._standard_shocks` with draws of zero
  mean and identity covariance. Multivariate Student-t shocks (one chi-square
  draw per day, shared across assets) rescaled by √((ν − 2)/ν) keep the
  covariance at Σ, so every benchmark is unchanged. Use ν > 4, which keeps the
  variance of the sample variance finite, so the reward's noise stays finite.
  Draw the chi-square variable from a second generator (a new `SeedPlan`
  stream). Drawing it from the same generator breaks the match between block
  and one-at-a-time sampling, and the Gaussian part of the shocks would no
  longer match the baseline run.

### 2.7 What the simulator deliberately does not do

- **It does not compute rewards.** The reward lives in `src/rewards.py` and
  needs γ, which is still undecided. The simulator never touches γ.
- **It holds no state.** The baseline is stateless. The transaction-cost
  extension needs holdings as state, which belongs in a layer on top of this
  class.

---

## 3. Verification

| Check | Result |
|---|---|
| Loader vs. moments re-derived from `excess_returns.csv` | Equal to 1e-14 relative |
| Draws vs. numpy's `multivariate_normal(method="cholesky")`, same seed | Identical (2,000 days; 100 episodes by both sampling methods) |
| Sample means, 400,000 simulated days | All 9 within 5 standard errors of μ |
| Sample covariances | All 45 distinct entries within 5 standard errors of Σ |
| Serial independence | All 81 lag-1 correlations within 5/√N of zero |
| Cholesky factor | Lower-triangular, L·Lᵀ = Σ to 1e-12 relative |
| Block vs. one-at-a-time sampling, same seed | Identical draws |
| Reward on 20,000 simulated 20-day episodes (equal weight, γ = 5 as test values) | Mean within 5 standard errors of w'μ − (γ/2)w'Σw |
| Mutation testing | 22 deliberate bugs injected into `environment.py`, all 22 caught; the unmutated control passes |

The statistical bounds alone cannot see small errors in μ. Daily Sharpe
ratios are low, so a 15% error in SPY's mean is only 3.3 standard errors even
on 400,000 days. The exact comparison with numpy's sampler closes that gap.
Four of the 22 mutations (μ 15% high, DBC's μ doubled, Σ 1% high, and
de-meaning every episode) passed the statistical tests and are caught only by
the exact comparison.

The other 18 mutations were:
- dropping μ
- the wrong Cholesky orientation
- Σ in place of its factor
- no symmetry check, or a tolerance a million times looser
- one-day episodes allowed
- evaluation reusing a training stream
- the agent sharing the return stream
- a missing day counter
- no ticker check
- annual in place of daily means
- not copying μ
- not checking the generator
- streams that depend on `n_runs`
- an evaluation stream that does not restart
- no stream tag
- numpy integers kept in `SeedPlan`
- pandas' default float parser

The seeds are fixed, so the suite gives the same result on every run. If a
seed or numpy's normal generator changed, a 5-SE bound would fail by chance
with probability about 6e-7 per statistic, and below 1e-4 across the suite.

---

## 4. Hand-offs and open items

- **Agent (William and Albert, paired).**
  - The agent takes a `ReturnSimulator` and an `agent_rng`, never μ or Σ (§2.4).
  - Episode length B is still to be chosen, and must be at least 2.
- **`train.py`.**
  - Builds the `SeedPlan`.
  - Records a run manifest. `as_dict()` alone cannot reproduce a run, because `clean.py` can rewrite `data/processed/`. The manifest should hold:
    - `plan.as_dict()`
    - the git commit
    - SHA-256 hashes of the two calibration CSVs
    - B and the episode count
    - the simulator class
  - Takes the Part IV observation count from the training loop (episodes used × B). With block sampling, `days_simulated` counts days drawn in advance, which can exceed the days used.
  - The master seed and the number of runs are still open (prospectus §6).
- **For William (flagged, not changed).**
  - `episodic_mean_variance_reward` still returns NaN for a one-day batch when called directly. The simulator refuses such episodes.
  - Training speed: per 20-day episode, the reward call takes 7.6 µs against 3.8 µs for the draw. The agent's choice in episode k depends on the reward in episode k−1, so rewards cannot be computed for a whole block at once. Each episode's mean vector m and covariance matrix S can be, though, and then the reward is w'm − (γ/2)w'Sw. That identity is exact (checked with `rewards.py`).
- **Part VI and bonus.**
  - Calibration windows need a helper that estimates μ and Σ from a date range of `excess_returns.csv`. Re-running `clean.py` would overwrite the baseline.
  - The three-asset comparison can reuse the nine-asset draws by taking columns SPY, IEF and GLD (positions 0, 3 and 6), which gives it exactly the same random numbers.
  - A transaction-cost layer that tracks weight drift needs total returns (excess plus the risk-free rate), not excess returns alone.
- **Prospectus.** Row 2 ("Ground-truth environment") can move to done after William's review.

---

## 5. Review history

An independent agent review on 2026-10-02 found no high-severity bug. It
confirmed that the sampling is bit-identical to numpy's Cholesky reference,
that the seeding claims hold, and that the first 14 mutation results
reproduced. It then found these, all since fixed:

| Finding | Fix |
|---|---|
| The statistical tests missed a 15% error in μ and per-episode de-meaning | Exact comparison with numpy's sampler (§3) |
| The leakage scan missed transitive imports and paths into `data/raw` | Import-graph check; scan widened to any path into `data/` |
| The fat-tail recipe would have broken block sampling and common random numbers | Recipe now draws the mixing variable from its own stream (§2.6) |
| `as_dict()` failed `json.dumps` when given numpy integers | Seeds stored as plain `int` |
| Plain `default_rng(seed).spawn(n)` could reproduce the evaluation stream | Spawn keys tagged (§2.3) |
| The loader parsed the CSVs with a lossy float parser (1e-12 relative) | Round-trip parser |
| The symmetry tolerance was not pinned by any test | Test added |
| Doc overclaims: "cannot look up μ", "removes simulation noise", "requires a seeded generator", the reward speed-up advice | Reworded (§2.3, §2.4, §4) |
