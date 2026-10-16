# Notes for William

**From:** Albert
**Date:** 2026-10-15 (first written 2026-10-13)
**Status:** `agent.py` and `train.py` are both built, tested, and merged. Every
review issue across all rounds is closed. What's left is two decisions that
can't wait and a handful that can.

---

## TL;DR / In plain terms

The split worked: I wrote `train.py`, you wrote `agent.py`, and the interface
we agreed on first meant neither of us had to guess the other's method names.
201 tests pass across the project.

Two things are still open and genuinely need deciding before any real
experiment runs: γ, and the exact wording of the convergence rule. Everything
else that used to be in this file's decisions table (episode length, episode
count, seed count, β, learning-rate magnitude) is intentionally parked until
the tuning/research phase, where it belongs. See §3.

---

## 1. Review issues: all closed

| # | Issue | Fixed in |
|---|---|---|
| 1 | SLSQP stopped early on daily moments | `51b3eb7` |
| 2 | `best_discrete_weights` hardcoded the grid | `51b3eb7` |
| 3 | Float drift in grid weights | `02f6a49` |
| 4 | `n = mu.shape` tuple | `51b3eb7` |
| 5 | `.vscode/settings.json` Windows path | `51b3eb7` |
| A | Reward NaN for a one-day batch | `0e2201d` |
| B | `best_discrete_weights` returned a view | `0e2201d` |
| C | `nit <= 1` guard replaced with a relative KKT check | `0e2201d` |

Re-checked against real calibration (γ 0.25–30), 200 random 9-asset problems,
and deliberately bad solves. Nothing fires at realistic scales; everything
genuinely wrong gets caught. Full suite at the time: 149 passed.

---

## 2. The tabular Q-learning core: split by file, done

We split instead of building side by side: you own `src/agent.py` (the part
that picks portfolios and updates Q-values), I own `src/train.py` (the part
that runs episodes, manages seeds, saves results). Both are merged into `dev`.

The interface that made this work, exactly as agreed:

```python
agent = TabularQAgent(n_actions, *, alpha_rule, epsilon0, beta, q_init, discount, rng)
a = agent.select_action()          # epsilon-greedy, eps_t = epsilon0 * exp(-beta * t)
agent.update(a, reward)            # Q(a) <- Q(a) + alpha * [R + rho * max Q - Q(a)]
agent.greedy_action                # current best action index
agent.q_values, agent.visit_counts # read-only copies, for logging
```

`train.py` never sees returns, μ, Σ, or a benchmark, only an action index and
a reward. The leakage tests in `test_environment.py` guard `src/agent.py`.

**Rules that are still worth keeping**, now that both sides exist:
1. I touch `agent.py` only through its interface; you touch `train.py` only
   through its outputs (the `.npz` logs and `manifest.json`).
2. Once both are stable, run an integration check: fixed seed, a few thousand
   episodes on the 25% grid, rerun to confirm identical output.
3. Review swap: I review `agent.py` line by line, you review `train.py`.

**Prospectus:** §4's ownership row now reads "Agent (`agent.py`): William;
training runner (`train.py`): Albert; each reviews the other's PR."

---

## 3. Decisions

### 3.1 Still open: needed before any results are examined

These two aren't hyperparameters to tune. γ defines the research question
itself, and the assignment requires convergence to be defined before anyone
looks at how a run did, specifically so neither gets chosen because it makes
a result look good.

| Decision | Owner | Why it can't be deferred |
|---|---|---|
| γ | William | It's the investor's risk aversion, not an algorithm setting. Choosing it by seeing which value gives a flattering result is the exact researcher-degrees-of-freedom problem `DATA_CLEANING.md` §8.4 already warns about. Needs justification on outside grounds (matching a known value, or the three-asset version of this problem if it used one) before any results exist. |
| Convergence rule (the definition, not the threshold) | Both | Assignment: "define convergence before examining the final results." The shape of the rule is basically agreed already: agent-observable quantities only (greedy action unchanged for K episodes, or largest \|ΔQ\| below a tolerance), never the gap to the benchmark, since that needs μ and Σ. Needs writing down as final, not as a proposal, before any run is read as converged or not. |

### 3.2 Parked until the tuning/research phase

Everything below is an implementation detail with a real, explorable
tradeoff (compute time vs. statistical confidence, mostly). The code already
takes every one of these as a parameter rather than a hardcoded value, so
there's no urgency and no reason to guess now. Revisit when there's a
notebook to run the exploration in, and pick based on agent-observable
criteria (how fast/reliably Q-values settle), never by comparing to the
benchmark. That comparison is what makes a choice here a leakage risk
instead of ordinary tuning.

- Episode length B (proposed baseline: 20)
- Episode count / training budget
- Seed count (proposed: 10)
- β, the exploration-decay parameter
- Learning-rate magnitude, if not using sample averaging
- The numeric convergence threshold K (once 3.1's rule is written down)
- Which second Part VI robustness check runs alongside the grid-resolution one

### 3.3 Already settled, no longer open

- **What the agent sees**: rewards only, never a return batch. Settled, enforced by the leakage tests.
- **Learning-rate rule**: sample averaging, α = 1/N(a), is what `agent.py` and `train.py` both implement as the default.
- **Discount ρ**: defaults to 0, the bandit baseline. Still an explicit constructor argument, so a nonzero value stays a stated choice if one's ever needed.
- **Q initialization, tie-breaking**: `q_init` is a parameter; ties break uniformly at random. Built and tested (`test_agent.py`).

---

## 4. Numbers to keep in mind for the agent

- **Noise.** At γ = 5, B = 20, one episode's reward for the best menu
  portfolio has a standard deviation 7.5 times its true value. The best and
  second-best menu portfolios differ by 2%.
- **Budget.** With 1/N averaging and uniform exploration, about 13 million
  episodes (260 million simulated days) got within roughly 2% of the best
  menu portfolio on a typical seed, finding the exact best about a quarter
  of the time.
- **Lock-in risk.** In a 3-asset toy run with fast-decaying ε, the agent
  settled on the second-best portfolio because the best one got an unlucky
  early average and was rarely retried.
- **Speed**, measured with your agent filled in:

  | Grid | Actions | Runner | Bare loop | Per 13M-episode run |
  |---|---|---|---|---|
  | 20% | 1,287 | 23.6 µs/episode | 19.7 µs | ~5 minutes |
  | 10% | 43,758 | 91 µs/episode | 51.6 µs | ~20 minutes (3.3 hr for 10 seeds) |

  Most of the runner's overhead is copying the full Q-table after every
  update. See §5.3.

---

## 5. Four things found while building `train.py` against your interface

### 5.1 Learning-rate visit count: resolved, not a live risk

I'd worried `_alpha` might call `alpha_rule(0)` on the first visit, which
numpy turns into `inf` with only a warning. Checked your actual code: `_alpha`
reads `self._visits[action] + 1`, so the first call is always `alpha_rule(1)`.
Confirmed by `test_sample_average_first_update_overwrites_q_init_exactly`.
Nothing to do.

### 5.2 `greedy_action` and randomness: mostly already handled, two small adds left

Your tie-break uses a separately spawned `_tie_rng`, not the main `rng`, so
reading it for logging can't perturb explore/exploit decisions, confirmed by
`test_greedy_action_tiebreak_uses_a_separate_rng_stream`. `train.py` never
calls it anyway; it computes its own `np.argmax(q_values)` for logging.

Two cheap things still worth adding:
- a docstring note on `greedy_action` that reading it can consume a draw from `_tie_rng` (only when an actual tie exists, which is free to check for but costs nothing to document)
- a public `epsilon` property exposing `_epsilon()`, so `train.py` can stop recomputing `epsilon0 * exp(-beta * t)` independently. Harmless today since the two formulas agree, but would silently drift if the schedule ever changes shape.

### 5.3 Full Q-table copy every episode: real, open interface decision

`train.py` calls `agent.q_values` once per episode just to read one scalar
(`new_q[action]`). Fine at the 20% grid; it's the dominant cost at the 10%
grid (table above). Proposed fix: have `update()` return the new `Q(action)`
value, and `train.py` keeps its own local mirror it updates at one index
instead of re-copying the whole table every time. This changes the agreed
interface, so it needs your sign-off before either of us touches it. Not
urgent before the 20% grid baseline; worth doing before the 10% robustness
runs.

### 5.4 Log format differs from what §2 originally promised: informational

A per-episode log (action, reward, ε) would be hundreds of MB at 13M
episodes, so `train.py` saves windowed checkpoints instead (every
`log_every` episodes: mean reward, largest |ΔQ|, logged greedy action,
per-action choice counts, ε, a float32 Q snapshot) plus every change of the
logged greedy action. Enough for learning curves, exact regret, "greedy
unchanged for K episodes" at any K, and the final policy to evaluate. Not
enough for a per-episode reward plot: say so if the report needs one and
I'll add an optional full log for short runs.
