# Review Notes for William

**From:** Albert
**Date:** 2026-10-02
**Status:** the four issues from the 2026-09-24 review are all closed. The three new issues reported below (A, B, C) are now fixed too, by William on 2026-10-13, with tests. See the note under issue C: the recommended fix had a scale bug, found while testing it, now corrected.

---

## TL;DR / In plain terms

Your Sep 29 fixes work. I checked the long-only optimizer against an exact
solution on our real data, and every weight agrees within 0.000003 for every whole-number γ
from 1 to 15. The simulator you scoped in `NEXT_STEPS_FOR_ALBERT.md` is built and
in a PR for your review.

While building it I found three new problems in your code. None changes a
number we would report today. One of them (the reward returning NaN) can
silently wreck a training run, so it is worth fixing before we pair on the
agent.

Your next steps, in order:
1. Review the simulator PR.
2. Fix the three issues below.
3. Choose γ.
4. Settle the open decisions in prospectus §6 before the agent session.

---

## 1. What changed since the last notes

### Earlier issues: all closed

| # | Issue | Fixed in | Verified |
|---|---|---|---|
| 1 | SLSQP stopped early on daily moments | `51b3eb7` (yours) | Against an exact support-enumeration solve on the real calibration: max weight gap 2.1e-6 over integer γ 1 to 15; objective identical to about 1e-11 relative. Reproduces `DATA_CLEANING.md` §8.3 exactly |
| 2 | `best_discrete_weights` hardcoded the grid | `51b3eb7` (yours) | `increment` and `grid` arguments work; grid sizes 495 / 1,287 / 43,758 |
| 3 | Float drift in grid weights | `02f6a49` (Albert) | Regression test in `test_action_space.py` |
| 4 | `n = mu.shape` tuple | `51b3eb7` (yours) | |
| 5 | `.vscode/settings.json` Windows path | `51b3eb7` (yours) | Untracked now |

### New: the simulator (Albert, PR into `dev`)

`src/environment.py`, 69 tests in `src/tests/test_environment.py`, and the
write-up in `ENVIRONMENT.md`. In short:

- `load_calibration()` reads μ and Σ from `data/processed/`, checks ticker
  order, and parses the CSVs exactly (pandas' default parser was off by up to
  1e-12 relative).
- `simulate_returns(mu, sigma, n_days, rng)` is the interface you specified,
  returning `(n_days, 9)` draws from N(μ, Σ).
- `ReturnSimulator` is the object the agent gets. It factors Σ once (about 10
  times faster than `rng.multivariate_normal` per episode), keeps μ and the
  factor private, and refuses episodes shorter than 2 days (see issue A).
- `SeedPlan` turns one master seed into separate streams: training returns
  per run, the agent's own randomness per run, and one evaluation stream that
  every policy and benchmark is scored on.
- Leakage tests fail if `src/agent.py` imports the benchmarks, the loader or
  the data files, directly or through another module.

Verification: draws identical to numpy's `multivariate_normal(method="cholesky")`
for the same seed; 400,000 simulated days match μ and Σ within 5 standard
errors; the reward averaged over 20,000 simulated episodes matches
w'μ − (γ/2)w'Σw; 22 deliberately injected bugs all caught. Full suite: 143
passed.

---

## 2. New issues in your files

**Status as of 2026-10-13 (William): all three fixed and tested, not yet
committed.** Full suite: 149 passed (142 before these fixes plus 2 new
tests for issue C's replacement guard, 4 for issue A, 1 for issue B).

| # | Severity | File | Issue | Blocks |
|---|---|---|---|---|
| A | ~~Medium~~ **FIXED** | `src/rewards.py:19` | Returns NaN, with no error, for a one-day batch | Nothing today; any 1-day episode in training |
| B | ~~Low-medium~~ **FIXED** | `src/benchmarks/mean_variance.py:90` | `best_discrete_weights` returns a view into the caller's grid | Nothing today; sharing the agent's grid with the benchmark |
| C | ~~Low~~ **FIXED** | `src/benchmarks/mean_variance.py:69` | The `nit <= 1` guard gives false alarms and misses some failures | Nothing on our data |

### ~~Issue A: the reward returns NaN for a one-day episode~~ FIXED

`portfolio_returns.var(ddof=1)` divides by B − 1, so a batch of one day gives
NaN. numpy warns only once per code location, so inside a training loop it is
effectively silent. A 1-D `(9,)` input and an empty `(0, 9)` batch also give
NaN, and a transposed `(9, 9)` batch is accepted without complaint.

**Why it matters:** NaN does not just give one bad reward. Once a Q-value is
NaN it stays NaN, and `np.argmax` treats NaN as the largest value:

```python
>>> np.argmax(np.array([0.0, 0.0, np.nan, 0.0, 0.0]))
2
```

That action becomes the agent's permanent greedy choice, with no error. Part
VI lists "different episode lengths" as a robustness check, so someone will
eventually try a short episode.

The simulator already refuses episodes under 2 days, but the reward can be
called directly, and the guard belongs with the formula that needs it.

**Recommended fix** (`rewards.py`, top of the function):

```python
if returns_batch.ndim != 2 or returns_batch.shape[1] != w.shape[0]:
    raise ValueError(
        f"returns_batch must be (B, {w.shape[0]}), got {returns_batch.shape}"
    )
if returns_batch.shape[0] < 2:
    raise ValueError("episodic reward needs at least 2 days (ddof=1 variance)")
```

If we ever switch to the one-period quadratic reward the assignment also
allows, that reward needs B = 1 and gets its own function, and
`MIN_EPISODE_LENGTH` in `environment.py` drops to 1.

### ~~Issue B: `best_discrete_weights` returns a view, not a copy~~ FIXED

`grid[np.argmax(scores)]` with a single integer index returns a view of that
row. When a caller passes `grid=`, which is the point of the parameter (sharing
the agent's action grid with the benchmark), editing the returned weights
silently edits the grid:

```python
>>> w = best_discrete_weights(mu, sigma, 5.0, grid=G)
>>> np.shares_memory(w, G)
True
```

Any in-place step on `w` later (renormalizing, clipping) would corrupt one of
the agent's actions.

**Recommended fix** (`mean_variance.py:90`):

```python
return grid[np.argmax(mean_var_scores)].copy()
```

### ~~Issue C: the `nit <= 1` guard checks the wrong thing~~ FIXED, with one change from the recommendation below

The guard catches the old failure, but it uses iteration count as a proxy for
"did not converge," and that proxy fails in both directions:

- **False alarm.** If the equal-weight starting point is already the optimum
  (any symmetric problem, or a one-asset problem), SLSQP correctly stops after
  one iteration and the guard raises. Example: 3 assets, equal μ, Σ = I.
- **Missed failure.** Under the old settings at γ = 15, SLSQP stopped at a
  wrong interior point after 6 iterations, which the guard would not catch.

Neither happens on our data at any γ tested. It could bite in a robustness
check with a symmetric toy problem.

**Recommended fix:** check optimality directly. At the long-only optimum the
gradient μ − γΣw is equal across held assets and no higher on assets at zero
(the KKT conditions). Replace the `nit` guard with:

```python
grad = mu_scaled - gamma * sigma_scaled @ result.x
held = result.x > 1e-8
lam = grad[held].mean()
residual = max(np.abs(grad[held] - lam).max(), (grad[~held] - lam).max(initial=0.0))
if residual > 1e-5:
    raise RuntimeError(f"Optimizer result fails the optimality check (KKT residual {residual:.1e})")
```

Checked on the scaled (annualized) objective: residual 3.1e-7 or less for our
solutions at every integer γ from 1 to 15, exactly 0 at the symmetric optimum
that currently raises, and 0.061 for equal weights on our data at γ = 5. A
1e-5 threshold separates them by two orders of magnitude.

**One problem with the recommendation above, found while testing it.** The
threshold is absolute, but the residual scales linearly with whatever
magnitude of μ and Σ the caller passes in. Scaling a test problem's μ and Σ
by 1000x (nothing in the function stops a caller from doing this) raised the
residual from 1.8e-6 to 4.2e-4, past a 1e-5 absolute cutoff, on a solution
that was just as correct. That breaks the scale-invariance this function is
supposed to have (`SCALE = 252` already relies on it). Fixed by normalizing
the residual by the gradient's own magnitude before comparing it to a
threshold, so the check responds to how wrong the answer is rather than how
large the inputs happen to be:

```python
grad_scale = max(np.abs(grad).max(), 1e-12)
if residual / grad_scale > 1e-4:
    raise RuntimeError(...)
```

Checked across a 1,000,000x range of input scaling on the same test problem:
relative residual stays between 3.4e-8 and 1.7e-5 for correct solutions, and
0.42 for the wrong-answer case above. `test_does_not_raise_when_the_symmetric_optimum_is_the_equal_weight_start`
in `test_mean_variance.py` pins the false-alarm case this issue was originally
about.

### Minor: test count in your docs

`PROJECT_PROSPECTUS.md` row 4 and `NEXT_STEPS_FOR_ALBERT.md` say the action
space has 12 tests; pytest collects 22 (8 functions, some parametrized). Your
total of 74 was right.

---

## 3. Your next steps

1. **Review the simulator PR** (`feature/environment` → `dev`). Please run the
   suite on your machine: a few tests compare draws to 1e-15, which should
   hold across BLAS libraries but has only been run on Albert's Mac.
2. ~~**Fix issues A, B and C** above. A matters most before the agent exists.~~ Done 2026-10-13, 5 new regression tests, 149 passed.
3. **Choose γ** and write down the reason (`DATA_CLEANING.md` §8.4). It must
   be chosen on outside grounds before anyone looks at results: matching the
   original three-asset version of this problem if it used one, or a value in
   the range economists treat as plausible (Mehra and Prescott capped it at
   10). It doesn't block the simulator, but every results table needs it.
4. **Settle prospectus §6** before the agent session: seed count, which second
   robustness check, the action-library approach, and the two
   transaction-cost questions. Also choose the episode length B.
5. **Mark prospectus row 2 done** after merging the simulator.
6. **Agent session (paired).** One design question to settle first. The
   assignment lets the agent see realized returns, but a full 9-asset batch
   would let it score all 1,287 portfolios at once or estimate μ and Σ
   itself, which stops being standard Q-learning. Albert's recommendation:
   `train.py` computes the reward and the agent updates only the chosen
   portfolio's Q-value; the report says so.

## 4. Things to know for the agent work

- **The reward is noisy.** At γ = 5 on our data, one 20-day episode's reward
  has a standard deviation 7.5 times the best portfolio's true score, and
  the best and second-best menu portfolios differ by 2%. An idealized learner
  with a 1/N learning rate needs about 13 million episodes to get within
  about 2% of the best menu portfolio on a typical seed. With a constant
  learning rate the Q-values keep jittering by 17% to 53% of the objective.
  The learning-rate rule and training budget are the first agent decisions.
- **Convergence rule and leakage.** "Within X% of the best menu portfolio"
  needs μ and Σ, so it should be measured by evaluation code after the fact
  and never decide when training stops. The stopping rule should use only
  what the agent sees (for example, the greedy choice unchanged for K
  episodes).
- **Speed.** Per 20-day episode, the reward call costs 7.6 µs and the draw
  3.8 µs. Rewards can't be computed for a block at once (episode k's action
  depends on episode k−1's reward), but each episode's mean vector m and
  covariance S can, and then reward = w'm − (γ/2)w'Sw exactly.
- **Observation counts** for Part IV should come from the training loop
  (episodes used × B). `ReturnSimulator.days_simulated` counts days drawn,
  which includes block pre-draws.
