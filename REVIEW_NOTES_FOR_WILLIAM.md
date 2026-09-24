# Review Notes — `src/benchmarks/` and `src/action_space/`

**From:** Albert, found while wiring the Part I data into the Part II benchmarks
**Date:** 2026-09-24
**Status:** reported, not patched. These are your files, so nothing here has been changed.

---

## TL;DR / In plain terms

Your benchmark code is correct on the test data and wrong on our real data. The
optimizer's default stopping tolerance is too loose for daily returns, so it
quits early and returns a portfolio that is not the optimum, while reporting
success. At γ = 5 it gives up 59% of the achievable objective.

It is dangerous because nothing looks broken: no exception, `success` is `True`,
and the weights are valid. At low γ it returns the equal-weight portfolio, which
is one of our four Part V benchmarks, so the "continuous optimum" row and the
"equal-weight" row in our comparison table would come out identical and read as
a finding rather than a bug.

One of your existing tests will also start failing the moment real daily moments
are used. Details in issue 1.

Three smaller items follow. Issue 2 blocks a Part V requirement. Issue 3 has
already been fixed with Albert's go-ahead (commit `02f6a49`); everything else
here is untouched and still yours.

---

## Issue summary

| # | Severity | File | Issue | Blocks |
|---|---|---|---|---|
| 1 | **Critical** | `mean_variance.py:48` | SLSQP stops prematurely on daily-scale moments and returns a wrong portfolio while reporting success | Part II, Part V |
| 2 | **High** | `mean_variance.py:67` | `best_discrete_weights` hardcodes the 9-asset 20% grid; no way to vary resolution | Part V (≥2 grid resolutions), Part VI |
| 3 | ~~Medium~~ **FIXED** | `grid.py:24` | Grid weights carried floating-point error (`0.6000000000000001`) | — |
| 4 | **Low** | `mean_variance.py:16` | `n = mu.shape` assigns a tuple where an int is intended | Nothing today |

A fifth, unrelated item is at the bottom (`.vscode/settings.json`).

All evidence below is reproducible from `data/processed/excess_returns.csv` on
branch `feature/data_pipeline`.

---

# Issue 1 — CRITICAL: the long-only optimizer stops early on daily moments

### High level

`constrained_optimal_weights` uses SciPy's default convergence tolerance,
`ftol=1e-6`. Our mean-variance objective at **daily** frequency is around
`1.4e-4`, the same order of magnitude as the tolerance. SLSQP concludes it has
converged and stops, returning a point that is not the optimum.

It reports `status=0`, `success=True`, and `"Optimization terminated
successfully"`. The `if not result.success` guard on line 56 cannot catch this,
because from SciPy's point of view nothing failed.

`closed_Form_optimal_weights` is unaffected. It solves the budget-only problem
analytically with no optimizer, and its daily and annualized solutions agree to
4e-15.

### Why your tests don't catch it

Every fixture in `test_mean_variance.py` uses **annual-scale** expected returns
(`mu` in 0.05 to 0.09). At that scale the objective is around 7e-2, comfortably
above `ftol`, and SLSQP converges in 8 iterations. The bug appears only on real
daily moments, which is what Part II must use, since the assignment requires μ,
Σ, risk aversion and rewards to share one time frequency.

### Evidence, γ = 5, on the calibrated data

```
repo function returns : [0.111 0.111 0.111 0.111 0.111 0.111 0.111 0.111 0.111]
correct answer        : [0.469 0.    0.    0.136 0.    0.    0.395 0.    0.   ]
                         SPY                IEF               GLD

L1 distance between them:  1.3333   (maximum possible is 2.0)
objective: repo 8.71e-05  vs  correct 2.11e-04   ->  58.6% shortfall
annualized: repo    ret 5.62%  vol 11.71%
            correct ret 8.86%  vol 11.92%

SLSQP: nit=1  success=True  status=0  "Optimization terminated successfully"
```

Iteration counts across scales, same problem, γ = 3:

| Moments | `ftol` | Iterations | Objective |
|---|---|---|---|
| Annualized | `1e-6` (current) | 8 | 7.11e-2 |
| Daily | `1e-6` (current) | **1** | 1.42e-4 |
| Daily | `1e-16` | 113 | 2.82e-4 |

The daily solution at tight tolerance agrees with the annualized solution to
2e-8, so the maths is fine and only the tolerance is wrong.

### Three things worth knowing before you fix it

**(i) It does not always return equal weights.** That is only the behaviour at
low γ, where it stops on its first iteration at the starting guess. At γ = 15 it
stops at an arbitrary interior point:

```
shipped : SPY 0.093, EFA 0.044, IEF 0.204, LQD 0.175, TIP 0.192,
          GLD 0.171, VNQ 0.025, DBC 0.096
correct : SPY 0.264, IEF 0.532, TIP 0.062, GLD 0.142
max|difference| = 0.328
```

A regression test asserting `!= equal_weights(9)` would pass while the answer is
still wrong. "Stops prematurely" is the accurate description, not "returns the
equal-weight start."

**(ii) `test_mean_variance.py:217` will start failing.** That test asserts the
best discrete portfolio cannot beat the continuous optimum, which is correct
theory. On real daily moments the broken continuous optimum scores worse than
the grid, so the ordering inverts at every γ:

```
  gamma   continuous    best discrete   ordering holds?
      1   1.958847e-04   3.602096e-04   NO
      3   1.415106e-04   2.810814e-04   NO
      5   8.713658e-05   2.082434e-04   NO
      8   5.575514e-06   1.480594e-04   NO
     15   2.197185e-05   6.694274e-05   NO
```

This also makes the "discretization gap between the continuous and best discrete
benchmarks" that Part V explicitly requires come out **negative**.

**(iii) `maxiter` is load-bearing.** Tightening `ftol` alone converts a silent
wrong answer into a crash:

| Setting | Result |
|---|---|
| daily, `ftol=1e-6` (as shipped) | wrong answer, no failure reported |
| daily, `ftol=1e-16` only | γ = 1 and 3 raise "Iteration limit reached" (nit=100) |
| daily, `ftol=1e-16` + `maxiter=1000` | correct at every γ |

### Recommended fix

Two options. **The second is better** and I would pick it.

**Option A, tighten the tolerance.** `mean_variance.py:48`:

```python
result = minimize(
    neg_obj,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-16, "maxiter": 1000},
)
```

**Option B, solve on annualized moments internally.** `ftol` is an *absolute*
tolerance, so any specific value is a bet on the objective's scale. Since γ is
frequency-invariant (proved in `DATA_CLEANING.md` §8.1, verified to 4e-15), you
can scale the inputs up, solve, and return, with no magic number:

```python
SCALE = 252  # trading days; gamma is frequency-invariant so the argmax is unchanged

result = minimize(
    lambda w: -(w @ (mu * SCALE) - 0.5 * gamma * (w @ (sigma * SCALE) @ w)),
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 1000},
)
```

This is robust whether the caller passes daily, monthly or annual moments.

Either way, strengthen the guard, because `result.success` is not evidence of
convergence here:

```python
if not result.success:
    raise RuntimeError(f"Optimizer failed: {result.message}")

# SLSQP reports success after a single iteration when ftol is large relative to
# the objective, which at daily frequency means silently returning w0.
if result.nit <= 1:
    raise RuntimeError(
        f"Optimizer stopped after {result.nit} iteration(s); ftol is likely too "
        "large relative to the objective scale."
    )
```

### Recommended regression test

Must use **daily-scale** moments. That is the whole point:

```python
@pytest.fixture
def daily_scale_problem():
    """Realistic daily moments. The objective here is ~1e-4, which is where
    SLSQP's default ftol stops prematurely."""
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.05, 0.09, size=9) / 252
    sigma = np.diag(rng.uniform(0.01, 0.05, size=9)) / 252
    return mu, sigma, 5.0


def test_daily_and_annual_solutions_agree(daily_scale_problem):
    """gamma is frequency-invariant: mu and Sigma both scale linearly with the
    horizon, so annualizing multiplies the objective by 252 and leaves the
    argmax unchanged. Disagreement means one scale failed to converge.

    This is stronger than asserting the result differs from equal weights: at
    high gamma the bug returns an arbitrary interior point, not w0.
    """
    mu, sigma, gamma = daily_scale_problem
    w_daily = constrained_optimal_weights(mu, sigma, gamma)
    w_annual = constrained_optimal_weights(mu * 252, sigma * 252, gamma)
    np.testing.assert_allclose(w_daily, w_annual, atol=1e-6)


def test_discretization_gap_is_non_negative_on_daily_moments(daily_scale_problem):
    """Guards test_mean_variance.py:217 at the scale where it currently breaks."""
    mu, sigma, gamma = daily_scale_problem
    obj = lambda w: w @ mu - 0.5 * gamma * (w @ sigma @ w)
    assert obj(best_discrete_weights(mu, sigma, gamma)) <= obj(
        constrained_optimal_weights(mu, sigma, gamma)
    ) + 1e-12
```

---

# Issue 2 — HIGH: `best_discrete_weights` cannot vary the grid resolution

### High level

`best_discrete_weights` calls `generate_weight_grid()` with no arguments, so it
always builds the 9-asset 20% grid regardless of the `mu` it is handed. It
cannot be pointed at another resolution, and it raises a confusing matmul error
on a problem of a different size:

```python
>>> best_discrete_weights(np.array([0.10, 0.08, 0.06]), np.eye(3) * 0.04, 3.0)
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0
            (size 3 is different from 9)
```

### Why it matters

Two assignment requirements need this:

- **Part V**: "A comparison of convergence under at least two action-space
  resolutions." The best-discrete benchmark is needed at each resolution.
- **Part VI**: one of the two required robustness checks must change the
  action-space design, and coarser-versus-finer grids is the obvious candidate.

Both are blocked until this takes parameters. It is on the critical path for the
prospectus's Week 6 baseline experiments.

### Recommended fix

`mean_variance.py:66`:

```python
def best_discrete_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    gamma: float,
    increment: float = 0.2,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Best portfolio available in the agent's discrete action set.

    Pass `grid` to score a pre-built action set. This avoids rebuilding a large
    grid on every call, and lets the caller use a randomly sampled action
    library instead of a Cartesian grid.
    """
    if grid is None:
        grid = generate_weight_grid(mu.shape[0], increment=increment)

    if grid.shape[1] != mu.shape[0]:
        raise ValueError(f"grid has {grid.shape[1]} assets but mu has {mu.shape[0]}")

    scores = grid @ mu - 0.5 * gamma * ((grid @ sigma) * grid).sum(axis=1)
    return grid[np.argmax(scores)]
```

Deriving `n_assets` from `mu.shape[0]` also makes the three-asset comparison the
assignment lists for bonus credit work without further changes.

**Performance note.** The 10% grid is 43,758 × 9 and the 5% grid is 3.1M × 9.
Rebuilding inside the function on every call gets expensive once we sweep seeds
and resolutions, which is what the optional `grid` parameter is for.

---

# Issue 3 — FIXED in commit `02f6a49`

**This one is already done**, with Albert's go-ahead after it was flagged. It is
the only issue in this document that has been changed in the code. Issues 1, 2
and 4 are still open and still yours. The description below is kept so you can
review the change.

Sanity checks run before committing, against assignment Part III:

| Check | Result |
|---|---|
| Action counts 495 / 1,287 / 43,758 | unchanged |
| Weights nonnegative, rows sum to one | unchanged |
| Best discrete portfolio at γ = 1/3/5/8/15 | identical |
| Largest value change anywhere | 1.1e-16 |
| Full suite | 59 passed |
| New regression test on the old implementation | fails, verified by reverting in a sandbox |

### High level

`generate_weight_grid` builds integer unit counts and multiplies by the
increment. 0.2 has no exact binary representation, so the product drifts: the
value that should be `0.6` is stored as `0.6000000000000001`. Any code comparing
a grid weight to a literal, or using one as a dictionary key, silently misses.

```python
>>> sorted(set(generate_weight_grid(9, 0.2).flatten()))
[0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0]
>>> 0.6 in sorted(set(generate_weight_grid(9, 0.2).flatten()))
False
```

Only some multiples are affected, depending on which way the rounding lands:
0.2, 0.4, 0.8 and 1.0 are all exact; 0.6 is not.

### What is NOT affected (checked)

- All 1,287 rows of the 20% grid sum to exactly 1.0.
- All 1,287 rows are unique as tuples, so a dict keyed on `tuple(row)` does not
  collide today.
- Existing tests pass, because they use `np.allclose` and `pytest.approx`.

This is latent, not currently breaking anything.

### Where it will bite

The transaction-cost extension needs to map a current-holdings vector back to an
action index to compute turnover. Exact float comparison there works for
weights of 0.2, 0.4 and 0.8 and fails for 0.6.

### The fix that was applied

**Divide by the integer unit count instead of multiplying by the increment.**
Three fifths is exactly 0.6; three times 0.2 is not.

```python
# grid.py:24
weights = np.array(portfolios) / total_units
```

Verified before committing:

| Check | Result |
|---|---|
| 20% grid values | `0.6` exactly, no drift |
| Largest value change anywhere | 1.1e-16 |
| Best discrete portfolio at γ = 1/3/5/8/15 | identical at every γ |
| Full test suite | 55 passed |

One caveat, which does not matter but should not surprise you. The count of rows
summing to *bitwise* exactly 1.0 shifts slightly, in both directions:

| Grid | Current | After fix |
|---|---|---|
| 20% | 1287/1287 | 1287/1287 |
| 10% | 38,062/43,758 | 37,894/43,758 |
| 5% | 2,681,145/3,108,105 | 2,789,789/3,108,105 |

The current code sometimes gets an exact sum by accidental error cancellation,
and cleaning the individual values removes the accident. Nothing compares a row
sum to exactly 1.0, everything uses `allclose`, and the discrepancy is 1e-16
against a tolerance of 1e-8.

**Worth pairing with it:** return the integer unit counts as well. They are exact
by construction and are the natural key for a Q-table, which removes the
float-comparison question from the extension entirely:

```python
def generate_weight_grid(
    n_assets: int = 9, increment: float = 0.2, return_units: bool = False
) -> np.ndarray:
    ...
    units = np.array(portfolios)          # exact integers
    weights = units / total_units
    return (weights, units) if return_units else weights
```

---

# Issue 4 — LOW: `n = mu.shape` assigns a tuple

`mean_variance.py:16`, in `closed_Form_optimal_weights`:

```python
n = mu.shape          # tuple, e.g. (9,)
ones = np.ones(n)     # works: np.ones accepts a shape tuple
```

Right answer by accident. `n` reads as a count everywhere else in the file,
including `constrained_optimal_weights` three lines down, which correctly uses
`mu.shape[0]`. It would break if `n` were ever used arithmetically.

**Fix:** `n = mu.shape[0]`.

**While you are in this file:** `closed_Form_optimal_weights` has an inconsistent
capital `F`. Renaming to `closed_form_optimal_weights` touches 11 lines, all in
this file and `test_mean_variance.py` (the definition, the import on line 6,
eight call sites, two comment mentions). Verify with `grep -rn "closed_Form"
src/`. Mechanical, and easier now than once the report cites function names.

---

# Unrelated: `.vscode/settings.json` has a hardcoded Windows path

```json
"python.defaultInterpreterPath": "C:\\Python310\\python.exe"
```

Committed to the repo, so it applies to both of us, and wrong on Albert's Mac.
`.gitignore:26-27` ignores `.vscode/*` then explicitly un-ignores
`settings.json`, so this file is tracked on purpose. Two options:

- **Keep it shared:** drop the `python.defaultInterpreterPath` line, leave only
  portable settings.
- **Make it personal:** delete line 27 from `.gitignore` and run
  `git rm --cached .vscode/settings.json`.

Either is fine. A Windows-only absolute path shared across two machines is the
state that should not persist.

---

## Suggested order

1. **Issue 1.** Blocks Part II onward and is the one that could put a wrong
   number in the report. Note it will also make `test_mean_variance.py:217` go
   green again rather than red.
2. **Issue 2.** Blocks Part V and Part VI, needed before the Week 6 baseline.
3. ~~Issue 3.~~ Done, commit `02f6a49`. Review it in the PR.
4. **Issue 4** and `.vscode`, whenever convenient.

## Cross-check available

Albert's Part I data is on `feature/data_pipeline` and every number above is
reproducible from `data/processed/excess_returns.csv`. Happy to pair on issue 1.
The fix is a few lines; the regression test is the part worth agreeing on, since
it is what stops this reappearing at a different scale.
