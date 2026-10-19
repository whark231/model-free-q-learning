# Notes for William

**From:** Albert
**Date:** 2026-10-16 (first written 2026-10-13)
**Status:** `agent.py`, `train.py` and `evaluate.py` are built, tested and on
`dev`. Every review issue across all rounds is closed, including a new one
found 2026-10-19 (§1, item D). γ, B and the convergence threshold are decided
and wired into `src/config.py`. The γ justification, the `requirements.txt`
pinning, and the results-sharing agreement, all previously open, are closed
as of 2026-10-19 (§6.3, §6.4). §5.3 (a per-action Q accessor for speed) is
not fixed, but is now a confirmed deferral rather than an open question.

---

## TL;DR / In plain terms

The split worked: I wrote `train.py`, you wrote `agent.py`, and the interface
we agreed on first meant neither of us had to guess the other's method names.
244 tests pass across the project.

Update 2026-10-16: γ, B and the convergence rule are now decided, and the
parked choices are set as adopted defaults (§6). §3 below is kept as the
record of how the decisions were framed.

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
| D | `run_experiments.py` wrote `INDEX.md` with no encoding; crashed on Windows (`UnicodeEncodeError` on γ under cp1252, never surfaced on Mac). Fixed on all six read/write calls in that file, not just the one that failed | William, 2026-10-19 |

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

### 3.1 Were open; decided 2026-10-16 (see §6)

These two aren't hyperparameters to tune. γ defines the research question
itself, and the assignment requires convergence to be defined before anyone
looks at how a run did, specifically so neither gets chosen because it makes
a result look good.

| Decision | Owner | Why it can't be deferred |
|---|---|---|
| γ | William | It's the investor's risk aversion, not an algorithm setting. Choosing it by seeing which value gives a flattering result is the exact researcher-degrees-of-freedom problem `DATA_CLEANING.md` §8.4 already warns about. Needs justification on outside grounds (matching a known value, or the three-asset version of this problem if it used one) before any results exist. |
| Convergence rule (the definition, not the threshold) | Both | Assignment: "define convergence before examining the final results." The shape of the rule is basically agreed already: agent-observable quantities only (greedy action unchanged for K episodes, or largest \|ΔQ\| below a tolerance), never the gap to the benchmark, since that needs μ and Σ. Needs writing down as final, not as a proposal, before any run is read as converged or not. |

### 3.2 Were parked; set as adopted defaults 2026-10-16 (see §6)

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

### 5.3 Full Q-table copy every episode: confirmed deferred, 2026-10-19

`train.py` calls `agent.q_values` once per episode just to read one scalar
(`new_q[action]`). Fine at the 20% grid; it's the dominant cost at the 10%
grid (table above). Proposed fix, still not applied: have `update()` return
the new `Q(action)` value, and `train.py` keeps its own local mirror it
updates at one index instead of re-copying the whole table every time.

Decided to defer rather than fix: none of the four planned experiments use
the 10% grid (`src/config.py` only has 20% and 25%), so this costs nothing
today. Fixing it would also need both files to change together (my `update()`
return value plus Albert's loop using it), so a one-sided fix would buy
nothing. Revisit only if a 10% grid run gets added to the plan.

### 5.4 Log format differs from what §2 originally promised: informational

A per-episode log (action, reward, ε) would be hundreds of MB at 13M
episodes, so `train.py` saves windowed checkpoints instead (every
`log_every` episodes: mean reward, largest |ΔQ|, logged greedy action,
per-action choice counts, ε, a float32 Q snapshot) plus every change of the
logged greedy action. Enough for learning curves, exact regret, "greedy
unchanged for K episodes" at any K, and the final policy to evaluate. Not
enough for a per-episode reward plot: say so if the report needs one and
I'll add an optional full log for short runs.


---

## 6. Decisions of 2026-10-16, and what I changed

### 6.1 The decisions

| Setting | Value | Where |
|---|---|---|
| γ | {3, 5, 8}; 5 is the baseline for headline tables, 3 and 8 are the Part VI risk-aversion check | `src/config.py` |
| Episode length B | 20 trading days | `src/config.py` |
| Convergence rule | Greedy action unchanged for K = 10% of training episodes (K = 1,300,000 at 13M episodes). Primary reading: the final greedy action held for at least K episodes through the end (`greedy_stable_through_end`); "first time any action held for K" is secondary | `src/config.py` |

That settles both items in §3.1. Most of §3.2 is now set as adopted defaults
(all one-line changes in `src/config.py`):

- 10 seeds, master seed 20261016;
- 13 million episodes per seed;
- β chosen so ε = 1% halfway through training (about 7.08e-7);
- \|ΔQ\| tolerance 1e-5 per window, held for good;
- near-optimal line 25 bps/yr of certainty-equivalent loss;
- 20% grid baseline, 25% grid as the second resolution;
- the second Part VI check is the γ sweep.

### 6.2 Edits to your files, flagged here as CLAUDE.md requires

`PROJECT_PROSPECTUS.md`:

- **Row 7** (training loop + convergence criterion) is now marked done, with
  the decisions listed.
- **Rows 8 and 9** said `evaluate.py` was empty. They now say the code is done
  and waiting on the full-scale run.

Nothing else in the prospectus changed. Revert or reword freely.

### 6.3 Done, 2026-10-19

`DATA_CLEANING.md` §8.4 now has the written outside-grounds justification:
γ = 5 within the 2–7 corrected finance-literature range (Elminejad, Havranek
and Irsova, 2022), below the ~7 household-survey median (Barsky et al.,
1997), under the Mehra-Prescott (1985) cap of 10; γ = 8 chosen to probe past
the finance range while staying under the other two. Sources are listed in
that section. The market-implied SPY-only candidate (≈2.9) was considered
and not used, since it's calibrated from the same dataset the benchmarks are
built on.

### 6.4 New code and docs

- `src/config.py`: every experiment setting, plus the list of four
  experiments (baseline, 25% grid, γ = 3, γ = 8).
- `src/run_experiments.py`: one command trains, evaluates, compares the two
  grids and writes `results/INDEX.md`. It resumes after interruption and
  refuses half-finished folders.
- `src/tests/test_config.py`: 12 tests pinning the decisions and exercising
  the runner.
- `EXPERIMENTS.md`: the runbook.
- `EVALUATION.md`: §7 updated.

One more item for awareness:

- **All four experiments share the master seed.** Seed i sees the same
  simulated returns in every experiment, and every experiment uses the same
  evaluation sample. That makes the comparisons paired; it is recorded in
  `EXPERIMENTS.md`.

A full run is about 3.5 hours. Two items closed, 2026-10-19:

- **`requirements.txt` is pinned**, to the versions tested on William's
  Python 3.10 (numpy capped at 2.2.6; 2.3+ requires Python 3.11+). Both
  combinations (3.10 and Albert's 3.13) are documented in `README.md` and
  `DATA_CLEANING.md` §7.5.
- **Results-sharing is adopted.** `.gitignore` now commits `INDEX.md`,
  `index.json`, each experiment's `manifest.json`/`settings.json`, and every
  `evaluation/` folder; only the `.npz` training logs stay local. Verified
  against the documented `results/` layout before trusting it.
